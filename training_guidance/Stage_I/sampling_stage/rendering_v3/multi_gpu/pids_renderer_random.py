"""
PIDS Stage 1 Renderer v3.5.0 (Random Augmentation)
==================================================

從頭設計的偏振立體渲染器，用於生成 PIDS 訓練數據。
Fork from v3.4.2，添加隨機化數據增強功能。

v3.5.0 更新:
-----------
- 新增隨機化數據增強 (randomize_for_augmentation)
- 光源 X 位置跟隨相機移動 (保持在相機上方)
- 可隨機化: LED_INTENSITY, CEILING_EMITTER_INTENSITY, CAMERA_X
- 深度相機、左右相機、光源一起移動，保持相對位置

v3.4.0 更新:
-----------
- 新增 right_parallel.exr 輸出（右相機 + 平行偏振）
- 用於 Criterion 1 (Geometric Consistency) 驗證
- 比較 left_parallel vs right_parallel 計算 vertical disparity
- 渲染步驟從 5 步增加到 6 步

v3.3.0 更新:
-----------
- 新增玻璃 mask 渲染功能
- 計算玻璃區域深度有效率 (PIDS Criterion 5: >90%)
- 輸出 {scene}_glass_mask.exr/png
- JSON 報告新增 glass_depth_validity 欄位

v3.2.0 更新:
-----------
- 修正相機光軸配置：從會聚光軸改為平行光軸
- 符合 PIDS 論文 Criterion 1 (Geometric Consistency) 要求
- 各相機使用獨立目標點，確保視線方向完全平行

座標系統 (重要！):
-----------------
Blender/OBJ 座標系 → Mitsuba 座標系 (經過 -90° X 軸旋轉)
    Blender X → Mitsuba X (左右)
    Blender Y → Mitsuba Z (深度/前後)
    Blender Z → Mitsuba Y (高度)

Chamber 配置 (Blender/OBJ 座標，單位 mm):
-----------------------------------------
    前牆:   Y = 350mm
    相機:   Y = 360mm (在 chamber 內！)
    玻璃區: Y = 528-695mm
    傢俱區: Y = 700-800mm
    後牆:   Y = 900mm
    寬度:   X = -300 ~ +300mm (共 600mm)
    高度:   Z = 0 ~ 300mm

相機配置 (平行光軸):
------------------
    感測器: Sony IMX296LQR-C (5.023 × 3.754 mm)
    焦距: 6mm
    基線: 65mm
    FOV: 45.4° (水平)
    左相機: X = -32.5mm, Y = 200mm, 0° 偏振片 (I∥)
    右相機: X = +32.5mm, Y = 200mm, 90° 偏振片 (I⊥)
    光軸方向: (0, 1, 0) - 兩相機完全平行，無會聚

渲染原理:
--------
1. 使用 Mitsuba 3 spectral_polarized variant
2. Stokes integrator 輸出 [S0, S1, S2, S3]
3. 偏振片透射: I(θ) = 0.5 * (S0 + S1*cos(2θ) + S2*sin(2θ))
4. I∥ (θ=0°) = 0.5 * (S0 + S1)
5. I⊥ (θ=90°) = 0.5 * (S0 - S1)

作者: PIDS Project
版本: 3.2.0
日期: 2025-12-22
"""

from __future__ import annotations

import numpy as np
import cv2
import os
import json
import argparse
import gc
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import time

# 延遲 import mitsuba，避免主進程初始化 CUDA
mi = None
dr = None

def lazy_import_mitsuba():
    """延遲載入 mitsuba，只在需要時才初始化"""
    global mi, dr
    if mi is None:
        import mitsuba as _mi
        import drjit as _dr
        mi = _mi
        dr = _dr


# ============================================================
# 配置
# ============================================================

class Config:
    """渲染配置 - 所有單位為毫米 (mm)，除非特別說明"""

    # 渲染設定
    WIDTH = 640
    HEIGHT = 480
    SPP = 16384           # 每像素樣本數 (16K)
    SPP_PER_BATCH = 1024  # 分批渲染，避免 GPU OOM (16 批次)
    MAX_DEPTH = 12        # 光線反彈次數

    # Chamber 尺寸 (Blender/OBJ 座標系，單位 mm)
    # 這些值來自 blender_furniture_randomizer_v17.py
    CHAMBER_WIDTH = 600.0           # X 方向: -300 ~ +300
    CHAMBER_HEIGHT = 300.0          # Z 方向: 0 ~ 300
    CHAMBER_Y_FRONT = 350.0         # 前牆 Y 位置
    CHAMBER_Y_BACK = 900.0          # 後牆 Y 位置

    # 立體相機位置 (在 chamber 內)
    CAMERA_X = -50.0                # 相機 X 偏移 (向左)
    CAMERA_Y = 400.0                # 相機 Y 位置 (往前移，深入 chamber)
    CAMERA_Z = 80.0                 # 相機高度 (較低)

    # 深度相機位置 (與立體相機同水平，光軸平行)
    DEPTH_CAMERA_OFFSET_Z = 0.0     # 深度相機與立體相機同高度

    # 立體相機
    BASELINE = 65.0                 # 基線距離 (mm)

    # 感測器: Sony IMX296LQR-C
    SENSOR_WIDTH = 5.023            # 感測器寬度 (mm) = 1456 × 3.45μm
    SENSOR_HEIGHT = 3.754           # 感測器高度 (mm) = 1088 × 3.45μm
    FOCAL_LENGTH = 6.0              # 鏡頭焦距 (mm)

    # FOV = 2 × arctan(sensor_width / (2 × focal_length)) ≈ 45.4°
    FOV = 45.4                      # 水平視場角 (度)

    # 目標區域 (玻璃物體範圍)
    GLASS_Y_MIN = 528.0
    GLASS_Y_MAX = 695.0

    # 光源配置
    LED_INTENSITY = 2000.0          # LED 強度
    LED_SIZE = (180.0, 100.0)       # LED 面光源尺寸 (寬, 高)
    LED_POSITION_Y = 280.0          # LED 高度 (chamber 頂部附近)
    LED_POSITION_Z = 420.0          # LED 深度 (相機前方)

    # 環境光
    AMBIENT_INTENSITY = 0.0 # 已停用，由天花板發光體取代

    # 天花板燈陣列（獨立光源，不依賴材質偵測）
    CEILING_LIGHTS_ENABLED = False # 已停用，改為將天花板直接設為發光體
    CEILING_EMITTER_INTENSITY = 100.0 # 將天花板作為發光體的強度（增強）

    # 四個燈的位置 (OBJ 座標)
    CEILING_LIGHT_POSITIONS = [
        (-150.0, 550.0, 295.0),  # 左前
        (150.0, 550.0, 295.0),   # 右前
        (-150.0, 750.0, 295.0),  # 左後
        (150.0, 750.0, 295.0),   # 右後
    ]

    # 材質
    GLASS_IOR = 1.5                 # 玻璃折射率
    GLASS_ROUGHNESS = 0.02          # 玻璃粗糙度

    # 輸出
    SAVE_PREVIEW = True
    SAVE_DOLP = True

    @classmethod
    def target_point(cls) -> Tuple[float, float, float]:
        """相機目標點 (玻璃區域中心) - 用於計算視線方向"""
        target_y = (cls.GLASS_Y_MIN + cls.GLASS_Y_MAX) / 2
        return (0.0, target_y, cls.CAMERA_Z)

    @classmethod
    def forward_direction(cls) -> Tuple[float, float, float]:
        """
        相機光軸方向（歸一化）

        從相機中心指向目標區域的方向向量。
        所有相機都應該使用相同的視線方向，實現平行光軸。
        """
        # 中心相機位置
        center_pos = np.array([0.0, cls.CAMERA_Y, cls.CAMERA_Z])
        target = np.array(cls.target_point())
        direction = target - center_pos
        direction = direction / np.linalg.norm(direction)
        return tuple(direction)

    @classmethod
    def left_camera_position(cls) -> Tuple[float, float, float]:
        """左相機位置"""
        return (cls.CAMERA_X - cls.BASELINE / 2, cls.CAMERA_Y, cls.CAMERA_Z)

    @classmethod
    def right_camera_position(cls) -> Tuple[float, float, float]:
        """右相機位置"""
        return (cls.CAMERA_X + cls.BASELINE / 2, cls.CAMERA_Y, cls.CAMERA_Z)

    @classmethod
    def depth_camera_position(cls) -> Tuple[float, float, float]:
        """深度相機位置 (在立體相機中央下方)"""
        return (cls.CAMERA_X, cls.CAMERA_Y, cls.CAMERA_Z + cls.DEPTH_CAMERA_OFFSET_Z)

    @classmethod
    def camera_target_for_position(cls, camera_pos: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        計算相機的個別目標點（確保平行光軸）

        每個相機的目標點是：相機位置 + 視線方向 * 固定距離
        這樣所有相機都有相同的視線方向，實現平行光軸。

        Args:
            camera_pos: 相機位置 (OBJ 座標)

        Returns:
            相機的目標點，確保光軸與其他相機平行
        """
        # 使用統一的視線方向
        direction = np.array(cls.forward_direction())

        # 計算到目標區域的距離
        center_pos = np.array([0.0, cls.CAMERA_Y, cls.CAMERA_Z])
        target = np.array(cls.target_point())
        distance = np.linalg.norm(target - center_pos)

        # 各相機的個別目標點 = 相機位置 + 方向 * 距離
        camera_pos_arr = np.array(camera_pos)
        camera_target = camera_pos_arr + direction * distance

        return tuple(camera_target)

    @classmethod
    def randomize_for_augmentation(cls, seed: int = None):
        """
        隨機化渲染參數（數據增強用）

        保持不變：
        - BASELINE (65mm) - 立體基線
        - FOV (45.4°) - 視場角
        - 感測器尺寸 - 硬體規格
        - 偏振片角度 (0°/90°) - 物理原理

        隨機化：
        - LED_INTENSITY: 偏振光強度 [1200, 3500]
        - CEILING_EMITTER_INTENSITY: 環境光強度 [50, 200]
        - CAMERA_X: 整體水平位置 [-120, 20]
          (左右相機、深度相機、光源一起移動)

        Args:
            seed: 隨機種子，用於可重現性
        """
        import random
        if seed is not None:
            random.seed(seed)

        # 偏振光強度 [2000, 4000] - 提高下限減少噪點
        cls.LED_INTENSITY = random.uniform(2000, 4000)

        # 環境光強度 [100, 250] - 提高下限減少噪點
        cls.CEILING_EMITTER_INTENSITY = random.uniform(100, 250)

        # 整體水平位置 [-120, 20]（所有相機和光源一起移動）
        cls.CAMERA_X = random.uniform(-120, 20)

        print(f"  [Augment] LED={cls.LED_INTENSITY:.0f}, "
              f"Ceiling={cls.CEILING_EMITTER_INTENSITY:.0f}, "
              f"CamX={cls.CAMERA_X:.1f}mm")


def mm_to_m(mm: float) -> float:
    """毫米轉米"""
    return mm / 1000.0


# ============================================================
# Mitsuba 設定
# ============================================================

def setup_mitsuba() -> str:
    """
    設定 Mitsuba variant

    必須使用 spectral_polarized variant！
    mono_polarized 在 luminance film 下會丟棄 S1/S2/S3
    """
    lazy_import_mitsuba()
    available = mi.variants()
    print(f"[Mitsuba] 可用 variants: {available}")

    # 優先順序：cuda > llvm > scalar
    preferred = [
        'cuda_ad_spectral_polarized',
        'cuda_spectral_polarized',
        'llvm_ad_spectral_polarized',
        'llvm_spectral_polarized',
        'scalar_spectral_polarized',
    ]

    for variant in preferred:
        if variant in available:
            try:
                # 初始化 DrJit CUDA backend
                if 'cuda' in variant:
                    dr.set_flag(dr.JitFlag.Debug, False)
                mi.set_variant(variant)
                print(f"[Mitsuba] 使用 variant: {variant}")
                return variant
            except ImportError as e:
                print(f"[Mitsuba] {variant} 初始化失敗: {e}")
                continue

    raise RuntimeError(
        f"找不到 spectral_polarized variant!\n"
        f"可用: {available}\n"
        f"請安裝包含 polarized 支援的 Mitsuba 3"
    )


# ============================================================
# 材質工廠
# ============================================================

class MaterialFactory:
    """材質創建工廠"""

    @staticmethod
    def glass(ior: float = Config.GLASS_IOR) -> Dict:
        """
        創建玻璃材質

        使用 dielectric 產生實心玻璃（有折射效果）
        """
        return {
            'type': 'dielectric',
            'int_ior': ior,
        }

    @staticmethod
    def diffuse(reflectance: float) -> Dict:
        """
        創建漫反射材質

        對於 spectral variant，使用單一反射率值
        """
        return {
            'type': 'diffuse',
            'reflectance': {
                'type': 'spectrum',
                'value': reflectance,
            },
        }

    @staticmethod
    def diffuse_rgb(color: Tuple[float, float, float]) -> Dict:
        """從 RGB 顏色創建漫反射材質"""
        # 計算亮度作為反射率
        luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
        return MaterialFactory.diffuse(luminance)


# ============================================================
# MTL 解析器
# ============================================================

class MTLParser:
    """OBJ MTL 材質檔案解析器"""

    # 非玻璃材質關鍵字（優先排除，避免誤判）
    NON_GLASS_KEYWORDS = [
        'background', 'wall', 'floor', 'ground', 'ceiling',
        'diffuse', 'opaque', 'solid', 'wood', 'metal', 'fabric',
        'concrete', 'brick', 'stone', 'plastic', 'rubber',
    ]

    # 玻璃材質關鍵字（保守判斷）
    GLASS_KEYWORDS = ['glass', 'transparent', 'acrylic']

    @classmethod
    def parse(cls, mtl_path: str) -> Dict[str, Dict]:
        """解析 MTL 檔案"""
        materials = {}
        current = None

        if not os.path.exists(mtl_path):
            print(f"  [MTL] 找不到: {mtl_path}")
            return materials

        with open(mtl_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if not parts:
                    continue

                cmd = parts[0].lower()

                if cmd == 'newmtl' and len(parts) > 1:
                    current = parts[1]
                    materials[current] = {
                        'is_glass': cls._is_glass_name(current),
                        'color': (0.5, 0.5, 0.5),
                    }
                elif current:
                    if cmd == 'kd' and len(parts) >= 4:
                        materials[current]['color'] = (
                            float(parts[1]),
                            float(parts[2]),
                            float(parts[3]),
                        )
                    elif cmd == 'd' and len(parts) >= 2:
                        if float(parts[1]) < 0.95:
                            materials[current]['is_glass'] = True
                    elif cmd == 'illum' and len(parts) >= 2:
                        if int(parts[1]) in [4, 6, 7, 9]:
                            materials[current]['is_glass'] = True

        return materials

    @classmethod
    def _is_glass_name(cls, name: str) -> bool:
        """根據名稱判斷是否為玻璃材質"""
        name_lower = name.lower()
        if any(kw in name_lower for kw in cls.NON_GLASS_KEYWORDS):
            return False
        return any(kw in name_lower for kw in cls.GLASS_KEYWORDS)


class OBJSplitter:
    """OBJ 檔案分離器 - 按材質分離玻璃、天花板和其他幾何"""

    # 天花板材質關鍵字
    CEILING_KEYWORDS = ['ceiling']

    def __init__(self, obj_path: str, materials: Dict[str, Dict]):
        self.obj_path = obj_path
        self.materials = materials
        self.vertices = []      # v
        self.normals = []       # vn
        self.texcoords = []     # vt
        self.glass_faces = []   # 玻璃材質的面
        self.ceiling_faces = [] # 天花板材質的面
        self.other_faces = []   # 其他材質的面

    @classmethod
    def _is_ceiling(cls, mat_name: str) -> bool:
        """判斷是否為天花板材質"""
        name_lower = mat_name.lower()
        return any(kw in name_lower for kw in cls.CEILING_KEYWORDS)

    def parse_and_split(self) -> Tuple[str, str, str]:
        """
        解析 OBJ 並分離為三個檔案

        Returns:
            (glass_obj_path, ceiling_obj_path, other_obj_path)
        """
        current_material = None
        current_type = 'other'  # 'glass', 'ceiling', 'other'

        with open(self.obj_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if not parts:
                    continue

                cmd = parts[0]

                if cmd == 'v' and len(parts) >= 4:
                    self.vertices.append(line)
                elif cmd == 'vn' and len(parts) >= 4:
                    self.normals.append(line)
                elif cmd == 'vt' and len(parts) >= 3:
                    self.texcoords.append(line)
                elif cmd == 'usemtl' and len(parts) >= 2:
                    current_material = parts[1]
                    mat_info = self.materials.get(current_material, {})

                    # 判斷材質類型（優先順序：玻璃 > 天花板 > 其他）
                    if mat_info.get('is_glass', False):
                        current_type = 'glass'
                        print(f"  [OBJ分離] 玻璃材質: {current_material}")
                    elif self._is_ceiling(current_material):
                        current_type = 'ceiling'
                        print(f"  [OBJ分離] 天花板材質: {current_material}")
                    else:
                        current_type = 'other'
                elif cmd == 'f':
                    if current_type == 'glass':
                        self.glass_faces.append(line)
                    elif current_type == 'ceiling':
                        self.ceiling_faces.append(line)
                    else:
                        self.other_faces.append(line)

        # 寫入分離的 OBJ 檔案到臨時目錄（避免污染場景目錄）
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix='pids_split_')
        base_name = os.path.splitext(os.path.basename(self.obj_path))[0]

        glass_path = os.path.join(temp_dir, f"{base_name}_glass.obj")
        ceiling_path = os.path.join(temp_dir, f"{base_name}_ceiling.obj")
        other_path = os.path.join(temp_dir, f"{base_name}_other.obj")

        self.temp_dir = temp_dir  # 保存臨時目錄路徑，供後續清理

        self._write_obj(glass_path, self.glass_faces)
        self._write_obj(ceiling_path, self.ceiling_faces)
        self._write_obj(other_path, self.other_faces)

        print(f"  [OBJ分離] 玻璃面數: {len(self.glass_faces)}")
        print(f"  [OBJ分離] 天花板面數: {len(self.ceiling_faces)}")
        print(f"  [OBJ分離] 其他面數: {len(self.other_faces)}")

        return glass_path, ceiling_path, other_path

    def _write_obj(self, path: str, faces: List[str]):
        """寫入 OBJ 檔案"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write("# Split OBJ file\n")
            # 寫入所有頂點（保持索引一致）
            for v in self.vertices:
                f.write(v + '\n')
            for vn in self.normals:
                f.write(vn + '\n')
            for vt in self.texcoords:
                f.write(vt + '\n')
            # 寫入面
            for face in faces:
                f.write(face + '\n')


# ============================================================
# 場景建構器
# ============================================================

class SceneBuilder:
    """Mitsuba 場景建構器"""

    def __init__(self, obj_path: str):
        # 使用絕對路徑，確保 Mitsuba 能找到檔案
        self.obj_path = str(Path(obj_path).resolve())
        self.mtl_path = self.obj_path.replace('.obj', '.mtl')

        # 驗證檔案存在
        if not os.path.exists(self.obj_path):
            raise FileNotFoundError(f"OBJ 檔案不存在: {self.obj_path}")

        print(f"[SceneBuilder] OBJ 路徑: {self.obj_path}")
        print(f"[SceneBuilder] MTL 路徑: {self.mtl_path}")

        self.materials = MTLParser.parse(self.mtl_path)

        # 分離玻璃、天花板和其他幾何
        splitter = OBJSplitter(self.obj_path, self.materials)
        self.glass_obj_path, self.ceiling_obj_path, self.other_obj_path = splitter.parse_and_split()
        self.has_glass = len(splitter.glass_faces) > 0
        self.has_ceiling = len(splitter.ceiling_faces) > 0
        self.temp_dir = splitter.temp_dir  # 保存臨時目錄路徑

    def cleanup(self):
        """清理臨時分離的 OBJ 檔案"""
        import shutil
        if hasattr(self, 'temp_dir') and self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"  [Cleanup] 已刪除臨時目錄: {self.temp_dir}")

    def build(self,
              camera_position: Tuple[float, float, float],
              camera_target: Tuple[float, float, float],
              spp: int,
              camera_polarizer_angle: Optional[float] = None) -> Dict:
        """
        建構完整場景

        座標轉換：OBJ (mm) → Mitsuba (m)，並旋轉 -90° X 軸

        Args:
            camera_position: 相機位置 (OBJ 座標)
            camera_target: 相機目標點 (OBJ 座標)
            spp: 每像素樣本數
            camera_polarizer_angle: 相機前偏振片角度 (度)，None 表示不加偏振片
        """
        scene = {
            'type': 'scene',
            'integrator': self._create_integrator(),
            'sensor': self._create_sensor(camera_position, camera_target, spp),
        }

        # 添加光源（新架構：非偏振光源 + 獨立偏振片）
        led_emitter, led_polarizer = self._create_led_light(camera_position)
        scene['led_emitter'] = led_emitter
        scene['led_polarizer'] = led_polarizer

        # 添加相機前偏振片（如果指定）
        if camera_polarizer_angle is not None:
            scene['camera_polarizer'] = self._create_camera_polarizer(
                camera_position, camera_target, camera_polarizer_angle
            )

        # 添加分離的 OBJ 網格
        for name, mesh_dict in self._create_meshes():
            scene[name] = mesh_dict

        return scene

    def _create_camera_polarizer(self,
                                  camera_position: Tuple[float, float, float],
                                  camera_target: Tuple[float, float, float],
                                  theta: float) -> Dict:
        """
        創建相機前方的偏振片

        Args:
            camera_position: 相機位置 (OBJ 座標)
            camera_target: 相機目標點 (OBJ 座標)
            theta: 偏振片角度 (度)，0°=水平偏振，90°=垂直偏振
        """
        # 轉換座標
        pos_m = self._transform_point(camera_position)
        tgt_m = self._transform_point(camera_target)

        # 計算相機朝向的方向向量
        direction = np.array(tgt_m) - np.array(pos_m)
        direction = direction / np.linalg.norm(direction)

        # 偏振片位置：在相機前方 10mm 處
        polarizer_offset = 0.010  # 10mm in meters
        polarizer_pos = np.array(pos_m) + direction * polarizer_offset

        # 偏振片尺寸（要足夠大以覆蓋整個視野）
        # 根據 FOV 和距離計算
        fov_rad = np.radians(Config.FOV)
        half_width = polarizer_offset * np.tan(fov_rad / 2) * 1.5  # 1.5x 安全邊距

        # 偏振片變換矩陣
        transform = mi.ScalarTransform4f.look_at(
            origin=polarizer_pos.tolist(),
            target=tgt_m,
            up=[0, 1, 0],
        ) @ mi.ScalarTransform4f.scale([half_width, half_width, 1])

        return {
            'type': 'rectangle',
            'to_world': transform,
            'bsdf': {
                'type': 'polarizer',
                'theta': theta,
            },
        }

    def _create_integrator(self) -> Dict:
        """創建 Stokes integrator"""
        return {
            'type': 'stokes',
            'integrator': {
                'type': 'path',
                'max_depth': Config.MAX_DEPTH,
            },
        }

    def _create_sensor(self,
                       position: Tuple[float, float, float],
                       target: Tuple[float, float, float],
                       spp: int) -> Dict:
        """創建相機"""
        # 座標轉換：OBJ (X,Y,Z) → Mitsuba (X,Z,Y) 且 mm → m
        # 因為 OBJ 匯出時 forward=-Y, up=Z，Mitsuba 需要 -90° X 軸旋轉
        pos_m = self._transform_point(position)
        tgt_m = self._transform_point(target)

        return {
            'type': 'perspective',
            'fov': Config.FOV,
            'fov_axis': 'x',
            'to_world': mi.ScalarTransform4f.look_at(
                origin=pos_m,
                target=tgt_m,
                up=[0, 1, 0],  # Mitsuba Y 軸朝上
            ),
            'film': {
                'type': 'hdrfilm',
                'width': Config.WIDTH,
                'height': Config.HEIGHT,
                'pixel_format': 'luminance',  # 灰階格式
                'component_format': 'float32',
                'rfilter': {'type': 'gaussian'},
            },
            'sampler': {
                'type': 'independent',
                'sample_count': spp,
            },
        }

    def _create_led_light(self, camera_pos: Tuple[float, float, float]) -> Tuple[Dict, Dict]:
        """
        創建偏振 LED 光源（新架構）

        使用獨立的非偏振光源 + 偏振片幾何體，確保光線正確經過偏振處理。

        Returns:
            Tuple of (emitter_dict, polarizer_dict)
        """
        # 光源位置：在相機上方偏前（跟隨相機 X 位置）
        light_pos = (
            Config.CAMERA_X,  # X: 跟隨相機中心位置
            Config.LED_POSITION_Z,  # Y: 相機前方 (OBJ 座標)
            Config.LED_POSITION_Y,  # Z: 頂部附近
        )

        # 光源朝向：指向玻璃區域中心
        target = Config.target_point()

        # 轉換座標
        pos_m = self._transform_point(light_pos)
        tgt_m = self._transform_point(target)

        # 計算變換矩陣
        size_x = mm_to_m(Config.LED_SIZE[0])
        size_y = mm_to_m(Config.LED_SIZE[1])

        # 計算光源朝向的方向向量（用於偏移偏振片）
        direction = np.array(tgt_m) - np.array(pos_m)
        direction = direction / np.linalg.norm(direction)

        # 偏振片位置：在光源前方 5mm 處
        polarizer_offset = 0.005  # 5mm in meters
        polarizer_pos = np.array(pos_m) + direction * polarizer_offset

        # 光源變換矩陣
        emitter_transform = mi.ScalarTransform4f.look_at(
            origin=pos_m,
            target=tgt_m,
            up=[0, 1, 0],
        ) @ mi.ScalarTransform4f.scale([size_x/2, size_y/2, 1])

        # 偏振片變換矩陣（稍微大一點以確保覆蓋所有光線）
        polarizer_transform = mi.ScalarTransform4f.look_at(
            origin=polarizer_pos.tolist(),
            target=tgt_m,
            up=[0, 1, 0],
        ) @ mi.ScalarTransform4f.scale([size_x/2 * 1.1, size_y/2 * 1.1, 1])

        # 非偏振光源
        emitter_dict = {
            'type': 'rectangle',
            'to_world': emitter_transform,
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'spectrum',
                    'value': Config.LED_INTENSITY,
                },
            },
        }

        # 獨立偏振片（無發光，只過濾穿過的光）
        polarizer_dict = {
            'type': 'rectangle',
            'to_world': polarizer_transform,
            'bsdf': {
                'type': 'polarizer',
                'theta': 0.0,  # 0° = 水平偏振
            },
        }

        return emitter_dict, polarizer_dict

    def _create_meshes(self) -> List[Tuple[str, Dict]]:
        """
        創建分離的 OBJ 網格

        使用 OBJSplitter 分離玻璃、天花板和其他幾何，
        分別載入並指定不同的 BSDF/emitter。

        Returns:
            List of (name, mesh_dict) tuples
        """
        meshes = []

        # 基本變換矩陣: mm → m 且 OBJ 座標 → Mitsuba 座標
        transform = mi.ScalarTransform4f.scale([0.001, 0.001, 0.001]) @ \
                    mi.ScalarTransform4f.rotate([1, 0, 0], -90)

        # 載入非玻璃幾何（使用漫反射）
        if os.path.exists(self.other_obj_path):
            print(f"[_create_meshes] 載入非玻璃: {self.other_obj_path}")
            meshes.append(('mesh_other', {
                'type': 'obj',
                'filename': self.other_obj_path,
                'face_normals': False,
                'to_world': transform,
                'bsdf': MaterialFactory.diffuse(0.5),
            }))

        # 載入天花板（普通漫反射，現在也作為發光體）
        if self.has_ceiling and os.path.exists(self.ceiling_obj_path):
            print(f"[_create_meshes] 載入天花板: {self.ceiling_obj_path}")
            meshes.append(('mesh_ceiling', {
                'type': 'obj',
                'filename': self.ceiling_obj_path,
                'face_normals': False,
                'to_world': transform,
                'bsdf': MaterialFactory.diffuse(0.85), # 天花板本身仍然有漫反射屬性
                'emitter': { # 添加發光體屬性
                    'type': 'area',
                    'radiance': {
                        'type': 'spectrum',
                        'value': Config.CEILING_EMITTER_INTENSITY,
                    },
                },
            }))

        # 載入玻璃幾何（使用 dielectric）
        if self.has_glass and os.path.exists(self.glass_obj_path):
            print(f"[_create_meshes] 載入玻璃: {self.glass_obj_path}")
            meshes.append(('mesh_glass', {
                'type': 'obj',
                'filename': self.glass_obj_path,
                'face_normals': False,
                'to_world': transform,
                'bsdf': MaterialFactory.glass(),
            }))

        return meshes

    def _transform_point(self, point: Tuple[float, float, float]) -> List[float]:
        """
        座標轉換：OBJ (mm) → Mitsuba (m)

        OBJ 座標: X(右), Y(深度/前), Z(上)
        經過 -90° X 軸旋轉後：
        Mitsuba: X(右), Y(上), Z(深度)
        """
        x, y, z = point
        # OBJ (X, Y, Z) → Mitsuba (X, Z, Y)，並 mm → m
        return [mm_to_m(x), mm_to_m(z), mm_to_m(y)]


# ============================================================
# Stokes 處理器
# ============================================================

class StokesProcessor:
    """Stokes Vector 處理器"""

    @staticmethod
    def extract_stokes(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        從渲染結果提取 Stokes 參數

        Returns:
            (S0, S1, S2) - 各為 2D array
        """
        if image.ndim != 3:
            raise ValueError(f"預期 3D 圖像，得到 {image.ndim}D")

        h, w = image.shape[:2]
        n_channels = image.shape[2]
        print(f"  [Stokes] 圖像形狀: {image.shape}, 通道數: {n_channels}")

        if n_channels >= 12:
            # Spectral polarized: 假設 n_spectral 波長 × 3 Stokes
            if n_channels == 15:
                n_spectral = 5
            elif n_channels == 16:
                n_spectral = 4
            elif n_channels == 13:
                n_spectral = 4
            else:
                n_spectral = n_channels // 3

            print(f"  [Stokes] 推測 {n_spectral} 波長")
            S0 = np.mean(image[:, :, 0:n_spectral], axis=2)
            S1 = np.mean(image[:, :, n_spectral:2*n_spectral], axis=2)
            S2_end = min(3*n_spectral, n_channels)
            S2 = np.mean(image[:, :, 2*n_spectral:S2_end], axis=2)

        elif n_channels == 4:
            S0 = image[:, :, 0]
            S1 = image[:, :, 1]
            S2 = image[:, :, 2]
        elif n_channels == 3:
            print("  [警告] RGB 輸出，無偏振信息")
            S0 = 0.2126 * image[:,:,0] + 0.7152 * image[:,:,1] + 0.0722 * image[:,:,2]
            S1 = np.zeros_like(S0)
            S2 = np.zeros_like(S0)
        else:
            raise ValueError(f"不支援的通道數: {n_channels}")

        # 輸出統計
        print(f"  [Stokes] S0 範圍: [{S0.min():.4f}, {S0.max():.4f}]")
        print(f"  [Stokes] S1 範圍: [{S1.min():.4f}, {S1.max():.4f}]")
        print(f"  [Stokes] S2 範圍: [{S2.min():.4f}, {S2.max():.4f}]")

        return S0.astype(np.float32), S1.astype(np.float32), S2.astype(np.float32)

    @staticmethod
    def compute_polarization_images(S0: np.ndarray,
                                     S1: np.ndarray,
                                     S2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        計算偏振片透射強度

        根據 Malus 定律：
        I(θ) = 0.5 * (S0 + S1*cos(2θ) + S2*sin(2θ))

        I_parallel (θ=0°) = 0.5 * (S0 + S1)
        I_cross (θ=90°) = 0.5 * (S0 - S1)
        """
        I_parallel = 0.5 * (S0 + S1)
        I_cross = 0.5 * (S0 - S1)

        # 確保非負
        I_parallel = np.maximum(I_parallel, 0)
        I_cross = np.maximum(I_cross, 0)

        # 輸出統計
        diff = np.abs(I_parallel - I_cross)
        print(f"  [偏振] I∥ 範圍: [{I_parallel.min():.4f}, {I_parallel.max():.4f}]")
        print(f"  [偏振] I⊥ 範圍: [{I_cross.min():.4f}, {I_cross.max():.4f}]")
        print(f"  [偏振] |I∥-I⊥| 平均: {diff.mean():.4f}, 最大: {diff.max():.4f}")

        return I_parallel.astype(np.float32), I_cross.astype(np.float32)

    @staticmethod
    def compute_dolp(S0: np.ndarray, S1: np.ndarray, S2: np.ndarray) -> np.ndarray:
        """計算偏振度 DoLP"""
        dolp = np.sqrt(S1**2 + S2**2) / (S0 + 1e-10)
        dolp = np.clip(dolp, 0, 1)
        return dolp.astype(np.float32)

    @staticmethod
    def compute_dolp_from_intensities(I_parallel: np.ndarray, I_cross: np.ndarray) -> np.ndarray:
        """從 I∥ 和 I⊥ 計算偏振度 DoLP"""
        I_sum = I_parallel + I_cross
        dolp = np.zeros_like(I_parallel)
        valid_mask = I_sum > 1e-6
        dolp[valid_mask] = np.abs(I_parallel[valid_mask] - I_cross[valid_mask]) / I_sum[valid_mask]
        dolp = np.clip(dolp, 0, 1)
        return dolp.astype(np.float32)


# ============================================================
# 場景報告生成器
# ============================================================

def generate_scene_report(scene_name: str,
                          I_parallel: np.ndarray,
                          I_cross: np.ndarray,
                          depth: np.ndarray,
                          glass_mask: np.ndarray = None) -> dict:
    """
    生成場景品質報告

    報告內容：
    1. DoLP 統計（全局、玻璃區域、背景區域）
    2. 強度比值 I∥/I⊥
    3. 噪點水平和 SNR
    4. 深度圖統計
    5. 強度平衡檢測
    6. 綜合品質評分
    7. 玻璃區域深度有效率 (Criterion 5)
    """
    from datetime import datetime

    report = {
        'scene_name': scene_name,
        'timestamp': datetime.now().isoformat(),
        'render_config': {
            'width': Config.WIDTH,
            'height': Config.HEIGHT,
            'spp': Config.SPP,
            'max_depth': Config.MAX_DEPTH,
        },
        'warnings': [],
    }

    # ============================================================
    # 1. 數值範圍
    # ============================================================
    report['value_range'] = {
        'I_parallel': {
            'min': float(np.min(I_parallel)),
            'max': float(np.max(I_parallel)),
            'mean': float(np.mean(I_parallel)),
        },
        'I_cross': {
            'min': float(np.min(I_cross)),
            'max': float(np.max(I_cross)),
            'mean': float(np.mean(I_cross)),
        },
    }

    # ============================================================
    # 2. 偏振分析（分區域）
    # ============================================================
    I_sum = I_parallel + I_cross
    valid_mask = I_sum > 1e-6

    diff = np.abs(I_parallel - I_cross)
    dolp = np.zeros_like(I_parallel)
    dolp[valid_mask] = diff[valid_mask] / I_sum[valid_mask]
    dolp = np.clip(dolp, 0, 1)

    # 分區域（DoLP > 0.1 為高偏振區，通常是玻璃）
    high_dolp_mask = (dolp > 0.1) & valid_mask  # 玻璃區域
    low_dolp_mask = (dolp <= 0.1) & valid_mask  # 背景區域

    # 強度比值
    ratio_mask = I_cross > 0.01
    intensity_ratio = np.zeros_like(I_parallel)
    if np.any(ratio_mask):
        intensity_ratio[ratio_mask] = I_parallel[ratio_mask] / I_cross[ratio_mask]

    # 玻璃區域統計
    if np.any(high_dolp_mask):
        glass_dolp = dolp[high_dolp_mask]
        glass_stats = {
            'pixel_count': int(np.sum(high_dolp_mask)),
            'pixel_ratio': float(np.sum(high_dolp_mask) / np.sum(valid_mask)),
            'dolp_mean': float(np.mean(glass_dolp)),
            'dolp_median': float(np.median(glass_dolp)),
            'dolp_max': float(np.max(glass_dolp)),
        }
    else:
        glass_stats = {
            'pixel_count': 0,
            'pixel_ratio': 0.0,
            'dolp_mean': 0.0,
            'dolp_median': 0.0,
            'dolp_max': 0.0,
        }

    # 背景區域統計
    if np.any(low_dolp_mask):
        bg_dolp = dolp[low_dolp_mask]
        bg_stats = {
            'pixel_count': int(np.sum(low_dolp_mask)),
            'pixel_ratio': float(np.sum(low_dolp_mask) / np.sum(valid_mask)),
            'dolp_mean': float(np.mean(bg_dolp)),
            'dolp_median': float(np.median(bg_dolp)),
        }
    else:
        bg_stats = {
            'pixel_count': 0,
            'pixel_ratio': 0.0,
            'dolp_mean': 0.0,
            'dolp_median': 0.0,
        }

    # 全局統計
    dolp_valid = dolp[valid_mask]

    report['polarization'] = {
        'global': {
            'dolp_mean': float(np.mean(dolp_valid)) if len(dolp_valid) > 0 else 0,
            'dolp_std': float(np.std(dolp_valid)) if len(dolp_valid) > 0 else 0,
            'dolp_p95': float(np.percentile(dolp_valid, 95)) if len(dolp_valid) > 0 else 0,
            'dolp_max': float(np.max(dolp_valid)) if len(dolp_valid) > 0 else 0,
        },
        'glass_region': glass_stats,
        'background_region': bg_stats,
        'intensity_ratio': {
            'mean': float(np.mean(intensity_ratio[ratio_mask])) if np.any(ratio_mask) else 0,
            'median': float(np.median(intensity_ratio[ratio_mask])) if np.any(ratio_mask) else 0,
            'max': float(np.max(intensity_ratio[ratio_mask])) if np.any(ratio_mask) else 0,
        },
        'diff': {
            'mean': float(np.mean(diff)),
            'max': float(np.max(diff)),
        },
    }

    # ============================================================
    # 3. 噪點分析
    # ============================================================
    def estimate_noise(image):
        diff_h = np.abs(image[:, 1:] - image[:, :-1])
        diff_v = np.abs(image[1:, :] - image[:-1, :])
        mad = (np.median(diff_h) + np.median(diff_v)) / 2
        return float(mad / (np.sqrt(2) * 0.6745))

    noise_parallel = estimate_noise(I_parallel)
    noise_cross = estimate_noise(I_cross)
    noise_std = (noise_parallel + noise_cross) / 2

    signal_diff = float(np.mean(diff[valid_mask])) if np.any(valid_mask) else 0
    noise_diff = np.sqrt(2) * noise_std
    snr_polarization = signal_diff / (noise_diff + 1e-10)

    report['noise'] = {
        'noise_std': noise_std,
        'noise_parallel': noise_parallel,
        'noise_cross': noise_cross,
        'snr_polarization': snr_polarization,
    }

    # ============================================================
    # 4. 深度圖分析
    # ============================================================
    if depth is not None:
        depth_valid = depth[depth > 0]
        report['depth'] = {
            'valid_ratio': float(np.sum(depth > 0) / depth.size),
            'min': float(np.min(depth_valid)) if len(depth_valid) > 0 else 0,
            'max': float(np.max(depth_valid)) if len(depth_valid) > 0 else 0,
            'mean': float(np.mean(depth_valid)) if len(depth_valid) > 0 else 0,
        }

    # ============================================================
    # 4.5 玻璃區域深度有效率 (PIDS Criterion 5)
    # ============================================================
    if glass_mask is not None and depth is not None:
        glass_pixels = glass_mask > 0.5
        glass_pixel_count = int(np.sum(glass_pixels))

        if glass_pixel_count > 0:
            glass_depth = depth[glass_pixels]
            valid_depth_in_glass = glass_depth > 0
            valid_count = int(np.sum(valid_depth_in_glass))
            validity_rate = valid_count / glass_pixel_count

            report['glass_depth_validity'] = {
                'glass_pixel_count': glass_pixel_count,
                'valid_depth_count': valid_count,
                'validity_rate': float(validity_rate),
                'pass': validity_rate >= 0.9,  # Criterion 5: > 90%
            }
        else:
            report['glass_depth_validity'] = {
                'glass_pixel_count': 0,
                'valid_depth_count': 0,
                'validity_rate': 0.0,
                'pass': False,
            }

    # ============================================================
    # 5. 強度平衡檢測（背景區域的 I∥/I⊥ 比值）
    # ============================================================
    if np.any(low_dolp_mask):
        bg_parallel = I_parallel[low_dolp_mask]
        bg_cross = I_cross[low_dolp_mask]

        bg_ratio_mask = bg_cross > 0.01
        if np.any(bg_ratio_mask):
            bg_intensity_ratio = bg_parallel[bg_ratio_mask] / bg_cross[bg_ratio_mask]
            bg_ratio_mean = float(np.mean(bg_intensity_ratio))
            bg_ratio_std = float(np.std(bg_intensity_ratio))
        else:
            bg_ratio_mean = 1.0
            bg_ratio_std = 0.0

        bg_mean_parallel = float(np.mean(bg_parallel))
        bg_mean_cross = float(np.mean(bg_cross))
    else:
        bg_ratio_mean = 1.0
        bg_ratio_std = 0.0
        bg_mean_parallel = 0.0
        bg_mean_cross = 0.0

    report['intensity_balance'] = {
        'background_ratio_mean': bg_ratio_mean,
        'background_ratio_std': bg_ratio_std,
        'background_mean_parallel': bg_mean_parallel,
        'background_mean_cross': bg_mean_cross,
        'is_balanced': 0.5 <= bg_ratio_mean <= 2.0,
    }

    # ============================================================
    # 6. 綜合品質評估
    # ============================================================
    glass_ratio = glass_stats['pixel_ratio']
    glass_dolp_mean = glass_stats['dolp_mean']
    bg_dolp_mean = bg_stats['dolp_mean']

    scores = []

    # 偏振分數
    if glass_ratio > 0.05 and glass_dolp_mean > 0.3:
        pol_score = 100
    elif glass_ratio > 0.01 and glass_dolp_mean > 0.1:
        pol_score = 70
    elif report['polarization']['global']['dolp_p95'] > 0.1:
        pol_score = 50
    else:
        pol_score = 30
    scores.append(pol_score)

    # 噪點分數
    if snr_polarization >= 10:
        noise_score = 100
    elif snr_polarization >= 2:
        noise_score = 80
    elif snr_polarization >= 1:
        noise_score = 60
    else:
        noise_score = 40
    scores.append(noise_score)

    # 區域對比分數
    if glass_dolp_mean > bg_dolp_mean * 3:
        contrast_score = 100
    elif glass_dolp_mean > bg_dolp_mean * 2:
        contrast_score = 80
    elif glass_dolp_mean > bg_dolp_mean:
        contrast_score = 60
    else:
        contrast_score = 40
    scores.append(contrast_score)

    # 強度平衡分數
    if report['intensity_balance']['is_balanced']:
        balance_score = 100
    elif 0.3 <= bg_ratio_mean <= 3.0:
        balance_score = 70
    elif 0.1 <= bg_ratio_mean <= 10.0:
        balance_score = 40
    else:
        balance_score = 20
    scores.append(balance_score)

    overall_score = int(np.mean(scores))

    if overall_score >= 80:
        level = 'excellent'
    elif overall_score >= 60:
        level = 'good'
    elif overall_score >= 40:
        level = 'acceptable'
    else:
        level = 'poor'

    report['quality'] = {
        'score': overall_score,
        'level': level,
        'polarization_score': pol_score,
        'noise_score': noise_score,
        'contrast_score': contrast_score,
        'balance_score': balance_score,
        'valid': overall_score >= 40,
    }

    # ============================================================
    # 7. 警告
    # ============================================================
    if glass_ratio > 0.8:
        report['warnings'].append('高偏振區域佔比過大（>80%），檢查光源/場景設置')
    if glass_ratio < 0.01:
        report['warnings'].append('高偏振區域過小（<1%），玻璃物體可能太小或偏振效果弱')
    if bg_dolp_mean > 0.2:
        report['warnings'].append('背景 DoLP 偏高，檢查漫反射材質設置')
    if snr_polarization < 1:
        report['warnings'].append(f'SNR 較低（{snr_polarization:.2f}），建議增加 SPP')
    if report['polarization']['intensity_ratio']['mean'] < 2:
        report['warnings'].append('I∥/I⊥ 比值較低，偏振效果可能不明顯')
    if 'glass_depth_validity' in report and not report['glass_depth_validity']['pass']:
        validity_rate = report['glass_depth_validity']['validity_rate'] * 100
        report['warnings'].append(f'玻璃區域深度有效率不足（{validity_rate:.1f}% < 90%），違反 Criterion 5')
    if not report['intensity_balance']['is_balanced']:
        report['warnings'].append(
            f'背景區域強度不平衡（I∥/I⊥={bg_ratio_mean:.2f}），'
            f'RAFT雙目對齊可能失敗'
        )

    return report


# ============================================================
# 渲染器
# ============================================================

class PIDSRenderer:
    """PIDS 偏振立體渲染器"""

    def __init__(self):
        self.variant = setup_mitsuba()

    def render_scene(self, obj_path: str, output_dir: str, scene_name: str = None):
        """
        渲染單一場景

        輸出:
            - {scene}_left_parallel.exr  (左相機 I∥ 灰階)
            - {scene}_right_parallel.exr (右相機 I∥ 灰階, 用於 vertical disparity 驗證)
            - {scene}_right_cross.exr    (右相機 I⊥ 灰階)
            - {scene}_depth.exr          (深度圖)
            - {scene}_disparity.exr      (視差圖)
            - {scene}_params.json        (參數)
            - *.png 預覽圖
        """
        if scene_name is None:
            scene_name = Path(obj_path).stem

        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"渲染場景: {scene_name}")
        print(f"{'='*60}")

        # 相機位置和目標（使用平行光軸配置）
        left_pos = Config.left_camera_position()
        right_pos = Config.right_camera_position()
        depth_pos = Config.depth_camera_position()

        # 各相機的個別目標點，確保平行光軸
        left_target = Config.camera_target_for_position(left_pos)
        right_target = Config.camera_target_for_position(right_pos)
        depth_target = Config.camera_target_for_position(depth_pos)

        print(f"[配置] 左相機位置: {left_pos}")
        print(f"[配置] 左相機目標: {left_target}")
        print(f"[配置] 右相機位置: {right_pos}")
        print(f"[配置] 右相機目標: {right_target}")
        print(f"[配置] 深度相機位置: {depth_pos}")
        print(f"[配置] 深度相機目標: {depth_target}")
        print(f"[配置] 視線方向: {Config.forward_direction()} (平行光軸)")

        # 建構場景
        builder = SceneBuilder(obj_path)

        # 渲染左相機 (I∥) - 使用 0° 偏振片
        print(f"\n[1/4] 渲染左相機 (I∥, θ=0°)...")
        left_image = self._render_camera(builder, left_pos, left_target, polarizer_angle=0.0)

        # 處理 Stokes → I∥
        S0_left, S1_left, S2_left = StokesProcessor.extract_stokes(left_image)
        I_parallel, _ = StokesProcessor.compute_polarization_images(S0_left, S1_left, S2_left)

        del left_image
        gc.collect()

        # 渲染右相機 (I∥) - 使用 0° 偏振片（用於 vertical disparity 驗證）
        print(f"\n[2/6] 渲染右相機 (I∥, θ=0°) - for vertical disparity...")
        right_parallel_image = self._render_camera(builder, right_pos, right_target, polarizer_angle=0.0)

        # 處理 Stokes → I∥
        S0_right_p, S1_right_p, S2_right_p = StokesProcessor.extract_stokes(right_parallel_image)
        I_right_parallel, _ = StokesProcessor.compute_polarization_images(S0_right_p, S1_right_p, S2_right_p)

        del right_parallel_image
        gc.collect()

        # 渲染右相機 (I⊥) - 使用 90° 偏振片
        print(f"\n[3/6] 渲染右相機 (I⊥, θ=90°)...")
        right_image = self._render_camera(builder, right_pos, right_target, polarizer_angle=90.0)

        # 處理 Stokes → I⊥
        S0_right, S1_right, S2_right = StokesProcessor.extract_stokes(right_image)
        _, I_cross = StokesProcessor.compute_polarization_images(S0_right, S1_right, S2_right)

        del right_image
        gc.collect()

        # 渲染深度圖（使用獨立深度相機，在立體相機中央下方）
        print(f"\n[4/6] 渲染深度圖 (深度相機)...")
        depth = self._render_depth(builder, depth_pos, depth_target)

        # 渲染玻璃 mask（使用深度相機視角，與深度圖對齊）
        print(f"\n[5/6] 渲染玻璃 mask...")
        glass_mask = self._render_glass_mask(builder, depth_pos, depth_target)

        # 計算視差
        print(f"\n[6/6] 計算視差...")
        disparity = self._compute_disparity(depth)

        # 保存結果
        print(f"\n[保存] 輸出檔案...")
        self._save_outputs(
            output_dir, scene_name,
            I_parallel, I_right_parallel, I_cross, depth, disparity,
            S0_left, S1_left, S2_left, glass_mask
        )

        # 清理臨時檔案
        builder.cleanup()

        print(f"\n{'='*60}")
        print(f"場景 {scene_name} 渲染完成!")
        print(f"{'='*60}\n")

    def _render_camera(self,
                       builder: SceneBuilder,
                       position: Tuple[float, float, float],
                       target: Tuple[float, float, float],
                       polarizer_angle: Optional[float] = None) -> np.ndarray:
        """
        渲染單一相機視角 (分批渲染)

        Args:
            builder: 場景建構器
            position: 相機位置
            target: 相機目標點
            polarizer_angle: 相機前偏振片角度 (度)，None 表示不加偏振片
        """
        scene_dict = builder.build(position, target, Config.SPP_PER_BATCH, polarizer_angle)
        scene = mi.load_dict(scene_dict)

        total_spp = Config.SPP
        batch_spp = Config.SPP_PER_BATCH
        n_batches = max(1, total_spp // batch_spp)

        print(f"  [渲染] SPP: {total_spp}, 批次: {n_batches} x {batch_spp}")

        accumulated = None

        for i in range(n_batches):
            image = mi.render(scene, spp=batch_spp)
            img_np = np.array(image)

            if accumulated is None:
                accumulated = img_np.astype(np.float64)
            else:
                accumulated += img_np.astype(np.float64)

            print(f"    批次 {i+1}/{n_batches} 完成")

            del image
            gc.collect()

        result = (accumulated / n_batches).astype(np.float32)
        print(f"  [渲染] 完成，形狀: {result.shape}")

        return result

    def _render_depth(self,
                      builder: SceneBuilder,
                      position: Tuple[float, float, float],
                      target: Tuple[float, float, float]) -> np.ndarray:
        """渲染深度圖"""
        scene_dict = builder.build(position, target, 256)

        # 修改 integrator 為 depth AOV
        scene_dict['integrator'] = {
            'type': 'aov',
            'aovs': 'dd.y:depth',
            'integrator': {
                'type': 'path',
                'max_depth': 2,
            },
        }

        scene = mi.load_dict(scene_dict)
        image = mi.render(scene, spp=256)
        img_np = np.array(image)

        # 提取深度通道
        if img_np.ndim == 3 and img_np.shape[2] >= 2:
            depth = img_np[:, :, 1]
        else:
            depth = img_np[:, :, 0] if img_np.ndim == 3 else img_np

        print(f"  [深度] 範圍: [{depth.min():.4f}, {depth.max():.4f}] m")

        return depth.astype(np.float32)

    def _compute_disparity(self, depth: np.ndarray) -> np.ndarray:
        """計算視差圖"""
        # 焦距 (pixels)
        fov_rad = np.radians(Config.FOV)
        focal_px = (Config.WIDTH / 2) / np.tan(fov_rad / 2)

        # 基線 (m)
        baseline_m = mm_to_m(Config.BASELINE)

        # 視差 = baseline * focal / depth
        disparity = np.zeros_like(depth)
        valid = depth > 0
        disparity[valid] = (baseline_m * focal_px) / depth[valid]

        print(f"  [視差] 焦距: {focal_px:.1f} px")
        print(f"  [視差] 範圍: [{disparity[valid].min():.1f}, {disparity[valid].max():.1f}] px")

        return disparity.astype(np.float32)

    def _render_glass_mask(self,
                           builder: SceneBuilder,
                           position: Tuple[float, float, float],
                           target: Tuple[float, float, float]) -> np.ndarray:
        """
        渲染玻璃區域 mask

        只渲染玻璃網格，使用純白材質，得到二值化 mask。
        用於計算 PIDS Criterion 5：玻璃區域深度有效率。

        Returns:
            glass_mask: 2D array, 玻璃區域為 1，其他為 0
        """
        if not builder.has_glass:
            print(f"  [Glass Mask] 場景無玻璃，返回空 mask")
            return np.zeros((Config.HEIGHT, Config.WIDTH), dtype=np.float32)

        # 座標轉換
        pos_m = builder._transform_point(position)
        tgt_m = builder._transform_point(target)

        # 基本變換矩陣
        transform = mi.ScalarTransform4f.scale([0.001, 0.001, 0.001]) @ \
                    mi.ScalarTransform4f.rotate([1, 0, 0], -90)

        # 只渲染玻璃 OBJ 的深度，有深度值 = 玻璃區域
        scene_dict = {
            'type': 'scene',
            'integrator': {
                'type': 'aov',
                'aovs': 'dd.y:depth',
                'integrator': {
                    'type': 'path',
                    'max_depth': 2,
                },
            },
            'sensor': {
                'type': 'perspective',
                'fov': Config.FOV,
                'fov_axis': 'x',
                'to_world': mi.ScalarTransform4f.look_at(
                    origin=pos_m,
                    target=tgt_m,
                    up=[0, 1, 0],
                ),
                'film': {
                    'type': 'hdrfilm',
                    'width': Config.WIDTH,
                    'height': Config.HEIGHT,
                    'pixel_format': 'luminance',
                    'component_format': 'float32',
                },
                'sampler': {
                    'type': 'independent',
                    'sample_count': 4,  # 深度只需極少 SPP
                },
            },
            # 只載入玻璃 OBJ
            'glass_mesh': {
                'type': 'obj',
                'filename': builder.glass_obj_path,
                'face_normals': False,
                'to_world': transform,
                'bsdf': MaterialFactory.diffuse(0.5),
            },
        }

        scene = mi.load_dict(scene_dict)
        image = mi.render(scene, spp=4)
        img_np = np.array(image)

        # 提取深度通道 (AOV 輸出格式: [radiance, depth])
        if img_np.ndim == 3 and img_np.shape[2] >= 2:
            depth_channel = img_np[:, :, 1]
        elif img_np.ndim == 3:
            depth_channel = img_np[:, :, 0]
        else:
            depth_channel = img_np

        # 二值化（有深度值的地方就是玻璃）
        glass_mask = (depth_channel > 0).astype(np.float32)

        pixel_count = int(np.sum(glass_mask))
        pixel_ratio = pixel_count / (Config.WIDTH * Config.HEIGHT)
        print(f"  [Glass Mask] 玻璃像素: {pixel_count} ({pixel_ratio*100:.1f}%)")

        return glass_mask

    def _save_outputs(self,
                      output_dir: str,
                      scene_name: str,
                      I_parallel: np.ndarray,
                      I_right_parallel: np.ndarray,
                      I_cross: np.ndarray,
                      depth: np.ndarray,
                      disparity: np.ndarray,
                      S0: np.ndarray,
                      S1: np.ndarray,
                      S2: np.ndarray,
                      glass_mask: np.ndarray = None):
        """保存所有輸出"""
        # EXR 檔案
        self._save_exr(I_parallel, f"{output_dir}/{scene_name}_left_parallel.exr")
        self._save_exr(I_right_parallel, f"{output_dir}/{scene_name}_right_parallel.exr")
        self._save_exr(I_cross, f"{output_dir}/{scene_name}_right_cross.exr")
        self._save_exr(depth, f"{output_dir}/{scene_name}_depth.exr")
        self._save_exr(disparity, f"{output_dir}/{scene_name}_disparity.exr")

        # 預覽 PNG
        if Config.SAVE_PREVIEW:
            # 使用統一範圍，方便比較
            vmax = max(I_parallel.max(), I_cross.max())
            self._save_png_fixed(I_parallel, f"{output_dir}/{scene_name}_left_parallel.png", 0, vmax)
            self._save_png_fixed(I_cross, f"{output_dir}/{scene_name}_right_cross.png", 0, vmax)

            # 偏振差異圖
            diff = np.abs(I_parallel - I_cross)
            self._save_png(diff, f"{output_dir}/{scene_name}_polarization_diff.png")

            # 深度圖 (colormap)
            self._save_depth_png(depth, f"{output_dir}/{scene_name}_depth.png")

        # DoLP（從 I∥ 和 I⊥ 計算）
        if Config.SAVE_DOLP:
            dolp = StokesProcessor.compute_dolp_from_intensities(I_parallel, I_cross)
            self._save_png(dolp, f"{output_dir}/{scene_name}_DoLP.png")

            # DoLP 彩色熱力圖
            dolp_uint8 = (dolp * 255).astype(np.uint8)
            dolp_colored = cv2.applyColorMap(dolp_uint8, cv2.COLORMAP_JET)
            cv2.imwrite(f"{output_dir}/{scene_name}_DoLP_color.png", dolp_colored)
            print(f"  [DoLP] 平均: {dolp.mean():.4f}, 最大: {dolp.max():.4f}")

        # 保存玻璃 mask
        if glass_mask is not None:
            self._save_exr(glass_mask, f"{output_dir}/{scene_name}_glass_mask.exr")
            self._save_png(glass_mask, f"{output_dir}/{scene_name}_glass_mask.png")

        # 生成品質報告
        print(f"\n[Report] 生成品質報告...")
        report = generate_scene_report(scene_name, I_parallel, I_cross, depth, glass_mask)

        report_path = f"{output_dir}/{scene_name}_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 印出關鍵指標
        print(f"    DoLP (全局): mean={report['polarization']['global']['dolp_mean']:.4f}, p95={report['polarization']['global']['dolp_p95']:.4f}")
        print(f"    DoLP (玻璃區域): {report['polarization']['glass_region']['dolp_mean']:.4f}")
        print(f"    DoLP (背景區域): {report['polarization']['background_region']['dolp_mean']:.4f}")
        print(f"    I∥/I⊥ 比值: {report['polarization']['intensity_ratio']['mean']:.2f}x")
        print(f"    背景平衡: {report['intensity_balance']['background_ratio_mean']:.2f}x " +
              ("✓ 平衡" if report['intensity_balance']['is_balanced'] else "⚠️ 不平衡"))
        print(f"    SNR: {report['noise']['snr_polarization']:.2f}")

        # 玻璃區域深度有效率 (Criterion 5)
        if 'glass_depth_validity' in report:
            gdv = report['glass_depth_validity']
            validity_pct = gdv['validity_rate'] * 100
            pass_str = "✓ 通過" if gdv['pass'] else "✗ 未通過"
            print(f"    玻璃深度有效率: {validity_pct:.1f}% {pass_str} (Criterion 5: >90%)")

        print(f"    品質: {report['quality']['level']} ({report['quality']['score']}分)")

        if report['warnings']:
            print(f"    ⚠️ 警告:")
            for w in report['warnings']:
                print(f"       - {w}")

        # 參數 JSON
        params = {
            'scene_name': scene_name,
            'config': {
                'width': Config.WIDTH,
                'height': Config.HEIGHT,
                'spp': Config.SPP,
                'baseline_mm': Config.BASELINE,
                'fov_deg': Config.FOV,
            },
            'camera': {
                'left_position': list(Config.left_camera_position()),
                'right_position': list(Config.right_camera_position()),
                'target': list(Config.target_point()),
            },
            'stats': {
                'I_parallel_range': [float(I_parallel.min()), float(I_parallel.max())],
                'I_cross_range': [float(I_cross.min()), float(I_cross.max())],
                'depth_range_m': [float(depth[depth > 0].min()) if (depth > 0).any() else 0,
                                  float(depth[depth > 0].max()) if (depth > 0).any() else 0],
            },
        }

        with open(f"{output_dir}/{scene_name}_params.json", 'w') as f:
            json.dump(params, f, indent=2)

    def _save_exr(self, image: np.ndarray, path: str):
        """保存 EXR"""
        if image.ndim == 3:
            image = image[:, :, 0]
        bitmap = mi.Bitmap(image.astype(np.float32))
        bitmap.write(path)
        print(f"    -> {path}")

    def _save_png(self, image: np.ndarray, path: str):
        """保存 PNG (自動範圍)"""
        if image.ndim == 3:
            image = image[:, :, 0]
        vmin, vmax = image.min(), image.max()
        if vmax > vmin:
            normalized = ((image - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(image, dtype=np.uint8)
        cv2.imwrite(path, normalized)

    def _save_png_fixed(self, image: np.ndarray, path: str, vmin: float, vmax: float):
        """保存 PNG (固定範圍)"""
        if image.ndim == 3:
            image = image[:, :, 0]
        clipped = np.clip(image, vmin, vmax)
        if vmax > vmin:
            normalized = ((clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(image, dtype=np.uint8)
        cv2.imwrite(path, normalized)

    def _save_depth_png(self, depth: np.ndarray, path: str):
        """保存深度圖 PNG (colormap)"""
        valid = depth > 0
        if not valid.any():
            cv2.imwrite(path, np.zeros((Config.HEIGHT, Config.WIDTH), dtype=np.uint8))
            return

        vmin, vmax = depth[valid].min(), depth[valid].max()
        normalized = np.zeros_like(depth)
        normalized[valid] = (depth[valid] - vmin) / (vmax - vmin + 1e-6)
        normalized = (normalized * 255).astype(np.uint8)

        # 反轉 (近=亮, 遠=暗)
        normalized = 255 - normalized
        normalized[~valid] = 0

        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        colored[~valid] = 0
        cv2.imwrite(path, colored)


# ============================================================
# 命令行介面
# ============================================================

def run_single_gpu_wrapper(args_tuple):
    """包裝函數，用於 multiprocessing 啟動前設置環境變量"""
    gpu_id, scenes, output_dir, spp, save_preview = args_tuple

    import os
    import sys

    # 必須在 import mitsuba 之前設置
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # 強制重新載入相關模組
    for mod_name in list(sys.modules.keys()):
        if 'mitsuba' in mod_name or 'drjit' in mod_name:
            del sys.modules[mod_name]

    # 現在 import mitsuba
    import mitsuba as mi
    import drjit as dr

    # 動態導入本模組的類和函數
    from pids_renderer_random import Config, PIDSRenderer

    Config.SPP = spp
    Config.SAVE_PREVIEW = save_preview

    print(f"[GPU {gpu_id}] 初始化完成，CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    renderer = PIDSRenderer()

    for i, scene_path in enumerate(scenes):
        print(f"[GPU {gpu_id}] 進度 {i+1}/{len(scenes)}: {scene_path.name}")
        Config.randomize_for_augmentation(seed=hash(scene_path.name) % 2**32)
        try:
            renderer.render_scene(str(scene_path), output_dir)
        except Exception as e:
            print(f"[GPU {gpu_id}] 錯誤 {scene_path}: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description='PIDS Stage 1 Renderer v3.5.1 (Multi-GPU Support)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 單一場景
  python pids_renderer_random.py --scene scene_0001.obj --output ./output

  # 批次渲染
  python pids_renderer_random.py --input_dir ./scenes --output ./output --max_scenes 100

  # 多 GPU 並行
  python pids_renderer_random.py --input_dir ./scenes --output ./output --num_gpus 4
        """
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--scene', type=str, help='單一 OBJ 場景')
    input_group.add_argument('--input_dir', type=str, help='OBJ 場景目錄')

    parser.add_argument('--output', type=str, required=True, help='輸出目錄')
    parser.add_argument('--spp', type=int, default=Config.SPP, help=f'SPP (預設: {Config.SPP})')
    parser.add_argument('--max_scenes', type=int, default=None, help='最大場景數')
    parser.add_argument('--skip', type=int, default=0, help='跳過前 N 個場景（用於多 GPU 並行）')
    parser.add_argument('--no_preview', action='store_true', help='不保存預覽 PNG')
    parser.add_argument('--num_gpus', type=int, default=1, help='使用的 GPU 數量（預設: 1）')

    args = parser.parse_args()

    # 更新配置
    Config.SPP = args.spp
    Config.SAVE_PREVIEW = not args.no_preview

    # 收集場景
    if args.scene:
        scenes = [args.scene]
    else:
        # 排除渲染過程中產生的分離 OBJ 文件 (_glass, _ceiling, _other)
        all_objs = sorted(Path(args.input_dir).glob('*.obj'))
        scenes = [f for f in all_objs
                  if not any(x in f.stem for x in ['_glass', '_ceiling', '_other'])]
        # 跳過前 N 個場景（用於多 GPU 並行）
        if args.skip > 0:
            scenes = scenes[args.skip:]
        if args.max_scenes:
            scenes = scenes[:args.max_scenes]

    print(f"[PIDS Renderer v3.5.1 - Multi-GPU Support]")
    print(f"  隨機化: LED強度, 環境光, 相機X位置")
    print(f"  GPU 數量: {args.num_gpus}")
    if args.skip > 0:
        print(f"  跳過前 {args.skip} 個場景")
    print(f"待渲染: {len(scenes)} 個場景")

    start_time = time.time()

    if args.num_gpus > 1:
        # 多 GPU 並行模式 - 使用 subprocess 確保環境變量正確
        import subprocess

        # 分配場景到各 GPU
        scenes_per_gpu = len(scenes) // args.num_gpus

        print(f"\n場景分配:")
        processes = []

        for gpu_id in range(args.num_gpus):
            start_idx = gpu_id * scenes_per_gpu
            if gpu_id == args.num_gpus - 1:
                num_scenes = len(scenes) - start_idx
            else:
                num_scenes = scenes_per_gpu

            print(f"  GPU {gpu_id}: {num_scenes} 個場景 (skip={args.skip + start_idx})")

            # 構建子進程命令
            cmd = [
                'python', __file__,
                '--input_dir', args.input_dir,
                '--output', args.output,
                '--spp', str(args.spp),
                '--skip', str(args.skip + start_idx),
                '--max_scenes', str(num_scenes),
            ]
            if args.no_preview:
                cmd.append('--no_preview')

            # 設置環境變量並啟動子進程
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

            log_file = open(f'gpu{gpu_id}.log', 'w')
            p = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
            processes.append((p, log_file))
            print(f"[啟動] GPU {gpu_id} 進程 PID: {p.pid}")

        # 等待所有進程完成
        print(f"\n等待所有 GPU 完成...")
        for p, log_file in processes:
            p.wait()
            log_file.close()

        print(f"所有 GPU 渲染完成！")

    else:
        # 單 GPU 模式
        renderer = PIDSRenderer()
        for i, scene_path in enumerate(scenes):
            print(f"\n[進度] {i+1}/{len(scenes)}")
            Config.randomize_for_augmentation(seed=hash(scene_path.name) % 2**32)
            try:
                renderer.render_scene(str(scene_path), args.output)
            except Exception as e:
                print(f"[錯誤] {scene_path}: {e}")
                import traceback
                traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"\n[完成] 總耗時: {elapsed/60:.1f} 分鐘")


if __name__ == '__main__':
    # 快速檢查是否為多 GPU 模式，如果是則直接啟動子進程，不初始化任何 CUDA
    import sys
    if '--num_gpus' in sys.argv:
        idx = sys.argv.index('--num_gpus')
        if idx + 1 < len(sys.argv):
            num_gpus = int(sys.argv[idx + 1])
            if num_gpus > 1:
                # 多 GPU 模式：只做調度，不 import mitsuba
                import subprocess

                # 解析必要參數
                parser = argparse.ArgumentParser()
                parser.add_argument('--input_dir', type=str, required=True)
                parser.add_argument('--output', type=str, required=True)
                parser.add_argument('--spp', type=int, default=16384)
                parser.add_argument('--max_scenes', type=int, default=None)
                parser.add_argument('--skip', type=int, default=0)
                parser.add_argument('--no_preview', action='store_true')
                parser.add_argument('--num_gpus', type=int, default=1)
                args = parser.parse_args()

                # 收集場景
                all_objs = sorted(Path(args.input_dir).glob('*.obj'))
                scenes = [f for f in all_objs
                          if not any(x in f.stem for x in ['_glass', '_ceiling', '_other'])]
                if args.skip > 0:
                    scenes = scenes[args.skip:]
                if args.max_scenes:
                    scenes = scenes[:args.max_scenes]

                print(f"[PIDS Renderer v3.5.2 - Multi-GPU Launcher]")
                print(f"  GPU 數量: {num_gpus}")
                print(f"  待渲染: {len(scenes)} 個場景")

                # 分配場景
                scenes_per_gpu = len(scenes) // num_gpus

                print(f"\n場景分配:")
                processes = []

                for gpu_id in range(num_gpus):
                    start_idx = gpu_id * scenes_per_gpu
                    if gpu_id == num_gpus - 1:
                        num_scenes_gpu = len(scenes) - start_idx
                    else:
                        num_scenes_gpu = scenes_per_gpu

                    print(f"  GPU {gpu_id}: {num_scenes_gpu} 個場景")

                    # 單 GPU 子進程命令
                    cmd = [
                        sys.executable, __file__,
                        '--input_dir', args.input_dir,
                        '--output', args.output,
                        '--spp', str(args.spp),
                        '--skip', str(args.skip + start_idx),
                        '--max_scenes', str(num_scenes_gpu),
                    ]
                    if args.no_preview:
                        cmd.append('--no_preview')

                    env = os.environ.copy()
                    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

                    log_file = open(f'gpu{gpu_id}.log', 'w')
                    p = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
                    processes.append((p, log_file))
                    print(f"[啟動] GPU {gpu_id} PID: {p.pid}")

                print(f"\n等待所有 GPU 完成...")
                print(f"監控: tail -f gpu0.log gpu1.log gpu2.log gpu3.log")

                for p, log_file in processes:
                    p.wait()
                    log_file.close()

                print(f"\n所有 GPU 渲染完成！")
                sys.exit(0)

    # 單 GPU 模式或子進程
    main()
