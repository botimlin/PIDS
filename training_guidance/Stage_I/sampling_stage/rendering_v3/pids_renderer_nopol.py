"""
PIDS Stage 1 Renderer v3.4.2-nopol (No Polarization)
=====================================================

無偏振版本渲染器，用於論文中的對比實驗。
與 pids_renderer.py (偏振版) 完全相同的場景設置，但移除偏振特性。

變更說明 (相對於偏振版):
------------------------
1. Mitsuba variant: spectral_polarized -> cuda_ad_rgb (或 llvm_ad_rgb)
2. Integrator: stokes -> path
3. 移除 LED 前偏振片
4. 移除相機前偏振片
5. 直接輸出灰度圖像，不經過 Stokes 處理

保持不變:
---------
- 相機位置 (left: X=-82.5mm, right: X=-17.5mm)
- 相機目標點、FOV、baseline
- LED 光源位置、大小、強度
- 天花板發光體設置
- 所有材質 (玻璃 dielectric, 漫反射等)
- 座標系統轉換
- 輸出格式 (EXR, PNG, JSON)

用途:
-----
生成與偏振版相同場景的無偏振立體對，用於：
- 論文中的消融實驗 (ablation study)
- 比較偏振 vs 無偏振對透明物體偵測的影響

作者: PIDS Project
版本: 3.4.2-nopol
日期: 2025-12-23
基於: pids_renderer.py v3.4.2
"""

from __future__ import annotations

import mitsuba as mi
import drjit as dr
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


# ============================================================
# 配置 (與偏振版完全相同)
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

    # 立體相機位置 (在 chamber 內) - 與偏振版完全相同
    CAMERA_X = -50.0                # 相機 X 偏移 (向左)
    CAMERA_Y = 400.0                # 相機 Y 位置 (往前移，深入 chamber)
    CAMERA_Z = 80.0                 # 相機高度 (較低)

    # 深度相機位置 (與立體相機同水平，光軸平行)
    DEPTH_CAMERA_OFFSET_Z = 0.0     # 深度相機與立體相機同高度

    # 立體相機
    BASELINE = 65.0                 # 基線距離 (mm) - 與偏振版相同

    # 感測器: Sony IMX296LQR-C - 與偏振版相同
    SENSOR_WIDTH = 5.023            # 感測器寬度 (mm) = 1456 × 3.45μm
    SENSOR_HEIGHT = 3.754           # 感測器高度 (mm) = 1088 × 3.45μm
    FOCAL_LENGTH = 6.0              # 鏡頭焦距 (mm)

    # FOV = 2 × arctan(sensor_width / (2 × focal_length)) ≈ 45.4°
    FOV = 45.4                      # 水平視場角 (度)

    # 目標區域 (玻璃物體範圍)
    GLASS_Y_MIN = 528.0
    GLASS_Y_MAX = 695.0

    # 光源配置 - 與偏振版完全相同
    LED_INTENSITY = 2000.0          # LED 強度
    LED_SIZE = (180.0, 100.0)       # LED 面光源尺寸 (寬, 高)
    LED_POSITION_Y = 280.0          # LED 高度 (chamber 頂部附近)
    LED_POSITION_Z = 420.0          # LED 深度 (相機前方)

    # 環境光
    AMBIENT_INTENSITY = 0.0  # 已停用，由天花板發光體取代

    # 天花板燈設置 - 與偏振版相同
    CEILING_LIGHTS_ENABLED = False
    CEILING_EMITTER_INTENSITY = 100.0

    # 四個燈的位置 (OBJ 座標) - 與偏振版相同
    CEILING_LIGHT_POSITIONS = [
        (-150.0, 550.0, 295.0),  # 左前
        (150.0, 550.0, 295.0),   # 右前
        (-150.0, 750.0, 295.0),  # 左後
        (150.0, 750.0, 295.0),   # 右後
    ]

    # 材質 - 與偏振版相同
    GLASS_IOR = 1.5                 # 玻璃折射率
    GLASS_ROUGHNESS = 0.02          # 玻璃粗糙度

    # 輸出
    SAVE_PREVIEW = True
    SAVE_DOLP = False  # 無偏振版本不輸出 DoLP

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


def mm_to_m(mm: float) -> float:
    """毫米轉米"""
    return mm / 1000.0


# ============================================================
# Mitsuba 設定 (無偏振版本)
# ============================================================

def setup_mitsuba() -> str:
    """
    設定 Mitsuba variant (無偏振版本)

    使用 RGB variant 而非 spectral_polarized
    """
    available = mi.variants()
    print(f"[Mitsuba-NoPol] 可用 variants: {available}")

    # 優先順序：cuda > llvm > scalar (RGB 版本)
    preferred = [
        'cuda_ad_rgb',
        'cuda_rgb',
        'llvm_ad_rgb',
        'llvm_rgb',
        'scalar_rgb',
    ]

    for variant in preferred:
        if variant in available:
            try:
                # 初始化 DrJit CUDA backend
                if 'cuda' in variant:
                    dr.set_flag(dr.JitFlag.Debug, False)
                mi.set_variant(variant)
                print(f"[Mitsuba-NoPol] 使用 variant: {variant}")
                return variant
            except ImportError as e:
                print(f"[Mitsuba-NoPol] {variant} 初始化失敗: {e}")
                continue

    raise RuntimeError(
        f"找不到 RGB variant!\n"
        f"可用: {available}\n"
        f"請安裝 Mitsuba 3"
    )


# ============================================================
# 材質工廠 (與偏振版相同)
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

        對於 RGB variant，使用單一反射率值
        """
        return {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                'value': [reflectance, reflectance, reflectance],
            },
        }

    @staticmethod
    def diffuse_rgb(color: Tuple[float, float, float]) -> Dict:
        """從 RGB 顏色創建漫反射材質"""
        return {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                'value': list(color),
            },
        }


# ============================================================
# MTL 解析器 (與偏振版相同)
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

        # 寫入分離的 OBJ 檔案
        base_dir = os.path.dirname(self.obj_path)
        base_name = os.path.splitext(os.path.basename(self.obj_path))[0]

        glass_path = os.path.join(base_dir, f"{base_name}_glass.obj")
        ceiling_path = os.path.join(base_dir, f"{base_name}_ceiling.obj")
        other_path = os.path.join(base_dir, f"{base_name}_other.obj")

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
# 場景建構器 (無偏振版本)
# ============================================================

class SceneBuilder:
    """Mitsuba 場景建構器 (無偏振版本)"""

    def __init__(self, obj_path: str):
        # 使用絕對路徑，確保 Mitsuba 能找到檔案
        self.obj_path = str(Path(obj_path).resolve())
        self.mtl_path = self.obj_path.replace('.obj', '.mtl')

        # 驗證檔案存在
        if not os.path.exists(self.obj_path):
            raise FileNotFoundError(f"OBJ 檔案不存在: {self.obj_path}")

        print(f"[SceneBuilder-NoPol] OBJ 路徑: {self.obj_path}")
        print(f"[SceneBuilder-NoPol] MTL 路徑: {self.mtl_path}")

        self.materials = MTLParser.parse(self.mtl_path)

        # 分離玻璃、天花板和其他幾何
        splitter = OBJSplitter(self.obj_path, self.materials)
        self.glass_obj_path, self.ceiling_obj_path, self.other_obj_path = splitter.parse_and_split()
        self.has_glass = len(splitter.glass_faces) > 0
        self.has_ceiling = len(splitter.ceiling_faces) > 0

    def build(self,
              camera_position: Tuple[float, float, float],
              camera_target: Tuple[float, float, float],
              spp: int) -> Dict:
        """
        建構完整場景 (無偏振版本)

        座標轉換：OBJ (mm) → Mitsuba (m)，並旋轉 -90° X 軸

        與偏振版的差異:
        - 使用 path integrator 而非 stokes
        - 不添加偏振片
        - LED 光源不加偏振片

        Args:
            camera_position: 相機位置 (OBJ 座標)
            camera_target: 相機目標點 (OBJ 座標)
            spp: 每像素樣本數
        """
        scene = {
            'type': 'scene',
            'integrator': self._create_integrator(),
            'sensor': self._create_sensor(camera_position, camera_target, spp),
        }

        # 添加光源 (無偏振)
        scene['led_emitter'] = self._create_led_light(camera_position)

        # 添加分離的 OBJ 網格
        for name, mesh_dict in self._create_meshes():
            scene[name] = mesh_dict

        return scene

    def _create_integrator(self) -> Dict:
        """創建 path integrator (無偏振)"""
        return {
            'type': 'path',
            'max_depth': Config.MAX_DEPTH,
        }

    def _create_sensor(self,
                       position: Tuple[float, float, float],
                       target: Tuple[float, float, float],
                       spp: int) -> Dict:
        """創建相機"""
        # 座標轉換：OBJ (X,Y,Z) → Mitsuba (X,Z,Y) 且 mm → m
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
                'pixel_format': 'rgb',  # RGB 格式 (無偏振)
                'component_format': 'float32',
                'rfilter': {'type': 'gaussian'},
            },
            'sampler': {
                'type': 'independent',
                'sample_count': spp,
            },
        }

    def _create_led_light(self, camera_pos: Tuple[float, float, float]) -> Dict:
        """
        創建 LED 光源 (無偏振版本)

        與偏振版相同的位置和強度，但不添加偏振片。
        """
        # 光源位置：在相機上方偏前 (與偏振版相同)
        light_pos = (
            0.0,  # X: 中央
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

        # 光源變換矩陣
        emitter_transform = mi.ScalarTransform4f.look_at(
            origin=pos_m,
            target=tgt_m,
            up=[0, 1, 0],
        ) @ mi.ScalarTransform4f.scale([size_x/2, size_y/2, 1])

        # 非偏振光源 (與偏振版強度相同)
        return {
            'type': 'rectangle',
            'to_world': emitter_transform,
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    'value': [Config.LED_INTENSITY, Config.LED_INTENSITY, Config.LED_INTENSITY],
                },
            },
        }

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
                'bsdf': MaterialFactory.diffuse(0.85),
                'emitter': {
                    'type': 'area',
                    'radiance': {
                        'type': 'rgb',
                        'value': [Config.CEILING_EMITTER_INTENSITY] * 3,
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
# 場景報告生成器 (無偏振版本)
# ============================================================

def generate_scene_report(scene_name: str,
                          left_image: np.ndarray,
                          right_image: np.ndarray,
                          depth: np.ndarray,
                          glass_mask: np.ndarray = None) -> dict:
    """
    生成場景品質報告 (無偏振版本)

    報告內容：
    1. 強度統計（左右圖像）
    2. 立體一致性（左右圖像差異）
    3. 深度圖統計
    4. 玻璃區域深度有效率 (Criterion 5)
    """
    from datetime import datetime

    report = {
        'scene_name': scene_name,
        'timestamp': datetime.now().isoformat(),
        'render_type': 'no_polarization',
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
        'left': {
            'min': float(np.min(left_image)),
            'max': float(np.max(left_image)),
            'mean': float(np.mean(left_image)),
        },
        'right': {
            'min': float(np.min(right_image)),
            'max': float(np.max(right_image)),
            'mean': float(np.mean(right_image)),
        },
    }

    # ============================================================
    # 2. 左右圖像一致性分析
    # ============================================================
    # 無偏振版本，左右圖像應該非常相似（除了視差）
    diff = np.abs(left_image - right_image)
    report['stereo_consistency'] = {
        'diff_mean': float(np.mean(diff)),
        'diff_max': float(np.max(diff)),
        'diff_std': float(np.std(diff)),
    }

    # ============================================================
    # 3. 深度圖分析
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
    # 4. 玻璃區域深度有效率 (PIDS Criterion 5)
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
    # 5. 品質評估
    # ============================================================
    # 無偏振版本主要檢查圖像品質
    score = 70  # 基礎分數

    # 檢查強度範圍
    if report['value_range']['left']['max'] > 0.1:
        score += 15
    if report['value_range']['right']['max'] > 0.1:
        score += 15

    # 檢查深度有效率
    if 'depth' in report and report['depth']['valid_ratio'] > 0.9:
        score += 10
    elif 'depth' in report and report['depth']['valid_ratio'] > 0.7:
        score += 5

    report['quality'] = {
        'score': min(100, score),
        'level': 'good' if score >= 70 else 'acceptable',
        'valid': True,
    }

    return report


# ============================================================
# 渲染器 (無偏振版本)
# ============================================================

class PIDSRendererNoPol:
    """PIDS 無偏振立體渲染器"""

    def __init__(self):
        self.variant = setup_mitsuba()

    def render_scene(self, obj_path: str, output_dir: str, scene_name: str = None):
        """
        渲染單一場景 (無偏振版本)

        輸出:
            - {scene}_left.exr  (左相機灰階)
            - {scene}_right.exr (右相機灰階)
            - {scene}_depth.exr (深度圖)
            - {scene}_disparity.exr (視差圖)
            - {scene}_params.json (參數)
            - *.png 預覽圖
        """
        if scene_name is None:
            scene_name = Path(obj_path).stem

        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"渲染場景 (無偏振): {scene_name}")
        print(f"{'='*60}")

        # 相機位置和目標（使用平行光軸配置）- 與偏振版相同
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

        # 渲染左相機 (無偏振)
        print(f"\n[1/4] 渲染左相機 (無偏振)...")
        left_image = self._render_camera(builder, left_pos, left_target)

        # 轉換為灰度
        left_gray = self._rgb_to_gray(left_image)

        del left_image
        gc.collect()

        # 渲染右相機 (無偏振)
        print(f"\n[2/4] 渲染右相機 (無偏振)...")
        right_image = self._render_camera(builder, right_pos, right_target)

        # 轉換為灰度
        right_gray = self._rgb_to_gray(right_image)

        del right_image
        gc.collect()

        # 渲染深度圖
        print(f"\n[3/4] 渲染深度圖 (深度相機)...")
        depth = self._render_depth(builder, depth_pos, depth_target)

        # 渲染玻璃 mask
        print(f"\n[4/4] 渲染玻璃 mask...")
        glass_mask = self._render_glass_mask(builder, depth_pos, depth_target)

        # 計算視差
        print(f"\n[計算] 視差...")
        disparity = self._compute_disparity(depth)

        # 保存結果
        print(f"\n[保存] 輸出檔案...")
        self._save_outputs(
            output_dir, scene_name,
            left_gray, right_gray, depth, disparity, glass_mask
        )

        print(f"\n{'='*60}")
        print(f"場景 {scene_name} 渲染完成 (無偏振)!")
        print(f"{'='*60}\n")

    def _rgb_to_gray(self, rgb_image: np.ndarray) -> np.ndarray:
        """RGB 轉灰度"""
        if rgb_image.ndim == 3 and rgb_image.shape[2] >= 3:
            # 使用標準亮度公式
            gray = 0.2126 * rgb_image[:, :, 0] + \
                   0.7152 * rgb_image[:, :, 1] + \
                   0.0722 * rgb_image[:, :, 2]
            return gray.astype(np.float32)
        elif rgb_image.ndim == 2:
            return rgb_image.astype(np.float32)
        else:
            return rgb_image[:, :, 0].astype(np.float32)

    def _render_camera(self,
                       builder: SceneBuilder,
                       position: Tuple[float, float, float],
                       target: Tuple[float, float, float]) -> np.ndarray:
        """
        渲染單一相機視角 (分批渲染)

        無偏振版本 - 不使用偏振片
        """
        scene_dict = builder.build(position, target, Config.SPP_PER_BATCH)
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

        與偏振版相同的實現方式。
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

        # 建構只有玻璃的簡化場景
        scene_dict = {
            'type': 'scene',
            'integrator': {
                'type': 'direct',
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
                    'sample_count': 16,
                },
            },
            'emitter': {
                'type': 'constant',
                'radiance': {'type': 'rgb', 'value': [1.0, 1.0, 1.0]},
            },
            'glass_mesh': {
                'type': 'obj',
                'filename': builder.glass_obj_path,
                'face_normals': False,
                'to_world': transform,
                'bsdf': {
                    'type': 'diffuse',
                    'reflectance': {'type': 'rgb', 'value': [1.0, 1.0, 1.0]},
                },
            },
        }

        scene = mi.load_dict(scene_dict)
        image = mi.render(scene, spp=16)
        img_np = np.array(image)

        # 提取亮度通道
        if img_np.ndim == 3:
            mask = img_np[:, :, 0]
        else:
            mask = img_np

        # 二值化
        glass_mask = (mask > 0.01).astype(np.float32)

        pixel_count = int(np.sum(glass_mask))
        pixel_ratio = pixel_count / (Config.WIDTH * Config.HEIGHT)
        print(f"  [Glass Mask] 玻璃像素: {pixel_count} ({pixel_ratio*100:.1f}%)")

        return glass_mask

    def _save_outputs(self,
                      output_dir: str,
                      scene_name: str,
                      left_gray: np.ndarray,
                      right_gray: np.ndarray,
                      depth: np.ndarray,
                      disparity: np.ndarray,
                      glass_mask: np.ndarray = None):
        """保存所有輸出"""
        # EXR 檔案 - 使用 _nopol 後綴以區分
        self._save_exr(left_gray, f"{output_dir}/{scene_name}_left_nopol.exr")
        self._save_exr(right_gray, f"{output_dir}/{scene_name}_right_nopol.exr")
        self._save_exr(depth, f"{output_dir}/{scene_name}_depth.exr")
        self._save_exr(disparity, f"{output_dir}/{scene_name}_disparity.exr")

        # 預覽 PNG
        if Config.SAVE_PREVIEW:
            # 使用統一範圍
            vmax = max(left_gray.max(), right_gray.max())
            self._save_png_fixed(left_gray, f"{output_dir}/{scene_name}_left_nopol.png", 0, vmax)
            self._save_png_fixed(right_gray, f"{output_dir}/{scene_name}_right_nopol.png", 0, vmax)

            # 左右差異圖 (用於檢驗立體一致性)
            diff = np.abs(left_gray - right_gray)
            self._save_png(diff, f"{output_dir}/{scene_name}_stereo_diff.png")

            # 深度圖 (colormap)
            self._save_depth_png(depth, f"{output_dir}/{scene_name}_depth.png")

        # 保存玻璃 mask
        if glass_mask is not None:
            self._save_exr(glass_mask, f"{output_dir}/{scene_name}_glass_mask.exr")
            self._save_png(glass_mask, f"{output_dir}/{scene_name}_glass_mask.png")

        # 生成品質報告
        print(f"\n[Report] 生成品質報告...")
        report = generate_scene_report(scene_name, left_gray, right_gray, depth, glass_mask)

        report_path = f"{output_dir}/{scene_name}_report_nopol.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 印出關鍵指標
        print(f"    左圖強度: mean={report['value_range']['left']['mean']:.4f}")
        print(f"    右圖強度: mean={report['value_range']['right']['mean']:.4f}")
        print(f"    立體差異: mean={report['stereo_consistency']['diff_mean']:.4f}")

        # 玻璃區域深度有效率 (Criterion 5)
        if 'glass_depth_validity' in report:
            gdv = report['glass_depth_validity']
            validity_pct = gdv['validity_rate'] * 100
            pass_str = "PASS" if gdv['pass'] else "FAIL"
            print(f"    玻璃深度有效率: {validity_pct:.1f}% {pass_str} (Criterion 5: >90%)")

        print(f"    品質: {report['quality']['level']} ({report['quality']['score']}分)")

        # 參數 JSON
        params = {
            'scene_name': scene_name,
            'render_type': 'no_polarization',
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
                'left_range': [float(left_gray.min()), float(left_gray.max())],
                'right_range': [float(right_gray.min()), float(right_gray.max())],
                'depth_range_m': [float(depth[depth > 0].min()) if (depth > 0).any() else 0,
                                  float(depth[depth > 0].max()) if (depth > 0).any() else 0],
            },
        }

        with open(f"{output_dir}/{scene_name}_params_nopol.json", 'w') as f:
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

def main():
    parser = argparse.ArgumentParser(
        description='PIDS Stage 1 Renderer v3.4.2-nopol (No Polarization)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 單一場景
  python pids_renderer_nopol.py --scene scene_0001.obj --output ./output_nopol

  # 批次渲染
  python pids_renderer_nopol.py --input_dir ./scenes --output ./output_nopol --max_scenes 100

  # 調整 SPP
  python pids_renderer_nopol.py --scene scene.obj --output ./output_nopol --spp 8192

注意: 此渲染器生成無偏振版本的立體對，用於論文中的對比實驗。
      相機位置、光源位置等設置與偏振版完全相同。
        """
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--scene', type=str, help='單一 OBJ 場景')
    input_group.add_argument('--input_dir', type=str, help='OBJ 場景目錄')

    parser.add_argument('--output', type=str, required=True, help='輸出目錄')
    parser.add_argument('--spp', type=int, default=Config.SPP, help=f'SPP (預設: {Config.SPP})')
    parser.add_argument('--max_scenes', type=int, default=None, help='最大場景數')
    parser.add_argument('--no_preview', action='store_true', help='不保存預覽 PNG')

    args = parser.parse_args()

    # 更新配置
    Config.SPP = args.spp
    Config.SAVE_PREVIEW = not args.no_preview

    # 收集場景
    if args.scene:
        scenes = [args.scene]
    else:
        scenes = sorted(Path(args.input_dir).glob('*.obj'))
        if args.max_scenes:
            scenes = scenes[:args.max_scenes]

    print(f"[PIDS Renderer v3.4.2-nopol - No Polarization]")
    print(f"找到 {len(scenes)} 個場景")

    # 渲染
    renderer = PIDSRendererNoPol()
    start_time = time.time()

    for i, scene_path in enumerate(scenes):
        print(f"\n[進度] {i+1}/{len(scenes)}")
        try:
            renderer.render_scene(str(scene_path), args.output)
        except Exception as e:
            print(f"[錯誤] {scene_path}: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"\n[完成] 總耗時: {elapsed/60:.1f} 分鐘")


if __name__ == '__main__':
    main()
