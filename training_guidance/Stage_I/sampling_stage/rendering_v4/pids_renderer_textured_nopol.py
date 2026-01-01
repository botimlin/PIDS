"""
PIDS Stage 1 Renderer v4.0.0-nopol (Textured, No Polarization)
================================================================

Fork from v4.0.0 (pids_renderer_textured.py) for ablation study.
Renders identical scenes WITHOUT polarization to demonstrate PIDS advantage.

v4.0.0-nopol 說明:
-----------------
- 使用 cuda_ad_rgb variant (非 spectral_polarized)
- 使用 path integrator (非 stokes)
- 無偏振片 (LED 和相機都不使用)
- 左右相機輸出相同的標準灰階影像
- 用於訓練基線模型 (baseline)，對比 PIDS 的優勢

座標系統 (與 v4.0.0 相同):
------------------------
Blender/OBJ 座標系 → Mitsuba 座標系 (經過 -90° X 軸旋轉)

作者: PIDS Project
版本: 4.0.0-nopol
日期: 2026-01-01

Copyright (c) 2025-2026 Po-Ting Lin
Released under the MIT License (see LICENSE file).
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
    SPP = 8192            # 每像素樣本數 (8K)
    SPP_PER_BATCH = 1024  # 分批渲染，避免 GPU OOM (8 批次)
    MAX_DEPTH = 12        # 光線反彈次數

    # Chamber 尺寸 (Blender/OBJ 座標系，單位 mm)
    CHAMBER_WIDTH = 600.0
    CHAMBER_HEIGHT = 300.0
    CHAMBER_Y_FRONT = 350.0
    CHAMBER_Y_BACK = 900.0

    # 立體相機位置 (在 chamber 內)
    CAMERA_X = -50.0
    CAMERA_Y = 400.0
    CAMERA_Z = 80.0

    # 深度相機位置
    DEPTH_CAMERA_OFFSET_Z = 0.0

    # 立體相機
    BASELINE = 65.0

    # 感測器: Sony IMX296LQR-C
    SENSOR_WIDTH = 5.023
    SENSOR_HEIGHT = 3.754
    FOCAL_LENGTH = 6.0
    FOV = 45.4

    # 目標區域 (玻璃物體範圍)
    GLASS_Y_MIN = 528.0
    GLASS_Y_MAX = 695.0

    # 光源配置 (無偏振)
    LED_INTENSITY = 2000.0
    LED_SIZE = (180.0, 100.0)
    LED_POSITION_Y = 280.0
    LED_POSITION_Z = 420.0

    # 天花板燈
    CEILING_EMITTER_INTENSITY = 100.0

    # 材質
    GLASS_IOR = 1.5
    GLASS_ROUGHNESS = 0.02

    # 輸出
    SAVE_PREVIEW = True

    @classmethod
    def target_point(cls) -> Tuple[float, float, float]:
        target_y = (cls.GLASS_Y_MIN + cls.GLASS_Y_MAX) / 2
        return (0.0, target_y, cls.CAMERA_Z)

    @classmethod
    def forward_direction(cls) -> Tuple[float, float, float]:
        center_pos = np.array([0.0, cls.CAMERA_Y, cls.CAMERA_Z])
        target = np.array(cls.target_point())
        direction = target - center_pos
        direction = direction / np.linalg.norm(direction)
        return tuple(direction)

    @classmethod
    def left_camera_position(cls) -> Tuple[float, float, float]:
        return (cls.CAMERA_X - cls.BASELINE / 2, cls.CAMERA_Y, cls.CAMERA_Z)

    @classmethod
    def right_camera_position(cls) -> Tuple[float, float, float]:
        return (cls.CAMERA_X + cls.BASELINE / 2, cls.CAMERA_Y, cls.CAMERA_Z)

    @classmethod
    def depth_camera_position(cls) -> Tuple[float, float, float]:
        return (cls.CAMERA_X, cls.CAMERA_Y, cls.CAMERA_Z + cls.DEPTH_CAMERA_OFFSET_Z)

    @classmethod
    def camera_target_for_position(cls, camera_pos: Tuple[float, float, float]) -> Tuple[float, float, float]:
        direction = np.array(cls.forward_direction())
        center_pos = np.array([0.0, cls.CAMERA_Y, cls.CAMERA_Z])
        target = np.array(cls.target_point())
        distance = np.linalg.norm(target - center_pos)
        camera_pos_arr = np.array(camera_pos)
        camera_target = camera_pos_arr + direction * distance
        return tuple(camera_target)

    @classmethod
    def randomize_for_augmentation(cls, seed: int = None):
        """
        隨機化渲染參數（數據增強用）
        與 v4.0.0 保持相同的隨機化範圍
        """
        import random
        if seed is not None:
            random.seed(seed)

        cls.LED_INTENSITY = random.uniform(2000, 4000)
        cls.CEILING_EMITTER_INTENSITY = random.uniform(100, 250)
        cls.CAMERA_X = random.uniform(-120, 20)

        print(f"  [Augment] LED={cls.LED_INTENSITY:.0f}, "
              f"Ceiling={cls.CEILING_EMITTER_INTENSITY:.0f}, "
              f"CamX={cls.CAMERA_X:.1f}mm")


def mm_to_m(mm: float) -> float:
    """毫米轉米"""
    return mm / 1000.0


# ============================================================
# Mitsuba 設定 (RGB variant, no polarization)
# ============================================================

def setup_mitsuba() -> str:
    """
    設定 Mitsuba variant

    使用 RGB variant (非 spectral_polarized)
    用於無偏振渲染
    """
    lazy_import_mitsuba()
    available = mi.variants()
    print(f"[Mitsuba] 可用 variants: {available}")

    # 優先順序：cuda > llvm > scalar (使用 RGB variant)
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
                if 'cuda' in variant:
                    dr.set_flag(dr.JitFlag.Debug, False)
                mi.set_variant(variant)
                print(f"[Mitsuba] 使用 variant: {variant} (無偏振)")
                return variant
            except ImportError as e:
                print(f"[Mitsuba] {variant} 初始化失敗: {e}")
                continue

    raise RuntimeError(
        f"找不到 RGB variant!\n"
        f"可用: {available}"
    )


# ============================================================
# 材質工廠
# ============================================================

class MaterialFactory:
    """材質創建工廠"""

    @staticmethod
    def glass(ior: float = Config.GLASS_IOR) -> Dict:
        return {
            'type': 'dielectric',
            'int_ior': ior,
        }

    @staticmethod
    def diffuse(reflectance: float) -> Dict:
        return {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                'value': [reflectance, reflectance, reflectance],
            },
        }

    @staticmethod
    def diffuse_rgb(color: Tuple[float, float, float]) -> Dict:
        return {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                'value': list(color),
            },
        }

    @staticmethod
    def diffuse_textured(texture_path: str) -> Dict:
        return {
            'type': 'diffuse',
            'reflectance': {
                'type': 'bitmap',
                'filename': texture_path,
                'filter_type': 'bilinear',
                'wrap_mode': 'repeat',
            },
        }


# ============================================================
# MTL 解析器
# ============================================================

class MTLParser:
    """OBJ MTL 材質檔案解析器"""

    NON_GLASS_KEYWORDS = [
        'background', 'wall', 'floor', 'ground', 'ceiling',
        'diffuse', 'opaque', 'solid', 'wood', 'metal', 'fabric',
        'concrete', 'brick', 'stone', 'plastic', 'rubber',
    ]

    GLASS_KEYWORDS = ['glass', 'transparent', 'acrylic']

    @classmethod
    def parse(cls, mtl_path: str) -> Dict[str, Dict]:
        materials = {}
        current = None
        mtl_dir = os.path.dirname(os.path.abspath(mtl_path))

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
                        'textures': {
                            'diffuse': None,
                        },
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
                    elif cmd == 'map_kd' and len(parts) >= 2:
                        tex_path_raw = ' '.join(parts[1:])
                        tex_abs_path = cls._resolve_texture_path(tex_path_raw, mtl_dir)
                        if tex_abs_path:
                            materials[current]['textures']['diffuse'] = tex_abs_path
                            print(f"  [MTL] 材質 '{current}' 紋理: {tex_abs_path}")

        return materials

    @classmethod
    def _resolve_texture_path(cls, tex_path: str, mtl_dir: str) -> Optional[str]:
        tex_path = tex_path.strip('"\'')
        tex_path = tex_path.replace('\\', '/')

        if os.path.isabs(tex_path):
            if os.path.exists(tex_path):
                return os.path.abspath(tex_path)
            else:
                return None

        abs_path = os.path.normpath(os.path.join(mtl_dir, tex_path))
        if os.path.exists(abs_path):
            return abs_path

        basename = os.path.basename(tex_path)
        same_dir_path = os.path.join(mtl_dir, basename)
        if os.path.exists(same_dir_path):
            return os.path.abspath(same_dir_path)

        common_dirs = ['textures', 'tex', 'maps', 'images']
        for subdir in common_dirs:
            subdir_path = os.path.join(mtl_dir, subdir, basename)
            if os.path.exists(subdir_path):
                return os.path.abspath(subdir_path)

        return None

    @classmethod
    def _is_glass_name(cls, name: str) -> bool:
        name_lower = name.lower()
        if any(kw in name_lower for kw in cls.NON_GLASS_KEYWORDS):
            return False
        return any(kw in name_lower for kw in cls.GLASS_KEYWORDS)


class OBJSplitter:
    """OBJ 檔案分離器"""

    CEILING_KEYWORDS = ['ceiling']

    def __init__(self, obj_path: str, materials: Dict[str, Dict]):
        self.obj_path = obj_path
        self.materials = materials
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.glass_faces = []
        self.ceiling_faces = []
        self.other_faces = []

    @classmethod
    def _is_ceiling(cls, mat_name: str) -> bool:
        name_lower = mat_name.lower()
        return any(kw in name_lower for kw in cls.CEILING_KEYWORDS)

    def parse_and_split(self) -> Tuple[str, str, str]:
        current_material = None
        current_type = 'other'

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

        import tempfile
        temp_dir = tempfile.mkdtemp(prefix='pids_split_nopol_')
        base_name = os.path.splitext(os.path.basename(self.obj_path))[0]

        glass_path = os.path.join(temp_dir, f"{base_name}_glass.obj")
        ceiling_path = os.path.join(temp_dir, f"{base_name}_ceiling.obj")
        other_path = os.path.join(temp_dir, f"{base_name}_other.obj")

        self.temp_dir = temp_dir

        self._write_obj(glass_path, self.glass_faces)
        self._write_obj(ceiling_path, self.ceiling_faces)
        self._write_obj(other_path, self.other_faces)

        print(f"  [OBJ分離] 玻璃面數: {len(self.glass_faces)}")
        print(f"  [OBJ分離] 天花板面數: {len(self.ceiling_faces)}")
        print(f"  [OBJ分離] 其他面數: {len(self.other_faces)}")

        return glass_path, ceiling_path, other_path

    def _write_obj(self, path: str, faces: List[str]):
        with open(path, 'w', encoding='utf-8') as f:
            f.write("# Split OBJ file (nopol)\n")
            for v in self.vertices:
                f.write(v + '\n')
            for vn in self.normals:
                f.write(vn + '\n')
            for vt in self.texcoords:
                f.write(vt + '\n')
            for face in faces:
                f.write(face + '\n')


# ============================================================
# 場景建構器 (無偏振)
# ============================================================

class SceneBuilder:
    """Mitsuba 場景建構器（無偏振版本）"""

    def __init__(self, obj_path: str):
        self.obj_path = str(Path(obj_path).resolve())
        self.mtl_path = self.obj_path.replace('.obj', '.mtl')

        if not os.path.exists(self.obj_path):
            raise FileNotFoundError(f"OBJ 檔案不存在: {self.obj_path}")

        print(f"[SceneBuilder-nopol] OBJ 路徑: {self.obj_path}")

        self.materials = MTLParser.parse(self.mtl_path)

        splitter = OBJSplitter(self.obj_path, self.materials)
        self.glass_obj_path, self.ceiling_obj_path, self.other_obj_path = splitter.parse_and_split()
        self.has_glass = len(splitter.glass_faces) > 0
        self.has_ceiling = len(splitter.ceiling_faces) > 0
        self.temp_dir = splitter.temp_dir

    def cleanup(self):
        import shutil
        if hasattr(self, 'temp_dir') and self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"  [Cleanup] 已刪除臨時目錄: {self.temp_dir}")

    def build(self,
              camera_position: Tuple[float, float, float],
              camera_target: Tuple[float, float, float],
              spp: int) -> Dict:
        """
        建構完整場景（無偏振）

        與 v4.0.0 的主要差異:
        - 使用 path integrator (非 stokes)
        - 無 LED 偏振片
        - 無相機偏振片
        """
        scene = {
            'type': 'scene',
            'integrator': self._create_integrator(),
            'sensor': self._create_sensor(camera_position, camera_target, spp),
        }

        # 添加光源（無偏振）
        scene['led_emitter'] = self._create_led_light(camera_position)

        # 添加分離的 OBJ 網格
        for name, mesh_dict in self._create_meshes():
            scene[name] = mesh_dict

        return scene

    def _create_integrator(self) -> Dict:
        """創建 path integrator (無 stokes wrapper)"""
        return {
            'type': 'path',
            'max_depth': Config.MAX_DEPTH,
        }

    def _create_sensor(self,
                       position: Tuple[float, float, float],
                       target: Tuple[float, float, float],
                       spp: int) -> Dict:
        """創建相機"""
        pos_m = self._transform_point(position)
        tgt_m = self._transform_point(target)

        return {
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
                'pixel_format': 'luminance',  # 灰階格式
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
        創建 LED 光源（無偏振）

        與 v4.0.0 的差異：只有發光體，無偏振片
        """
        light_pos = (
            Config.CAMERA_X,
            Config.LED_POSITION_Z,
            Config.LED_POSITION_Y,
        )

        target = Config.target_point()

        pos_m = self._transform_point(light_pos)
        tgt_m = self._transform_point(target)

        size_x = mm_to_m(Config.LED_SIZE[0])
        size_y = mm_to_m(Config.LED_SIZE[1])

        emitter_transform = mi.ScalarTransform4f.look_at(
            origin=pos_m,
            target=tgt_m,
            up=[0, 1, 0],
        ) @ mi.ScalarTransform4f.scale([size_x/2, size_y/2, 1])

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
        """創建分離的 OBJ 網格"""
        meshes = []

        transform = mi.ScalarTransform4f.scale([0.001, 0.001, 0.001]) @ \
                    mi.ScalarTransform4f.rotate([1, 0, 0], -90)

        other_bsdf = self._select_other_bsdf()

        if os.path.exists(self.other_obj_path):
            meshes.append(('mesh_other', {
                'type': 'obj',
                'filename': self.other_obj_path,
                'face_normals': False,
                'to_world': transform,
                'bsdf': other_bsdf,
            }))

        ceiling_bsdf = self._select_ceiling_bsdf()

        if self.has_ceiling and os.path.exists(self.ceiling_obj_path):
            meshes.append(('mesh_ceiling', {
                'type': 'obj',
                'filename': self.ceiling_obj_path,
                'face_normals': False,
                'to_world': transform,
                'bsdf': ceiling_bsdf,
                'emitter': {
                    'type': 'area',
                    'radiance': {
                        'type': 'rgb',
                        'value': [Config.CEILING_EMITTER_INTENSITY] * 3,
                    },
                },
            }))

        if self.has_glass and os.path.exists(self.glass_obj_path):
            meshes.append(('mesh_glass', {
                'type': 'obj',
                'filename': self.glass_obj_path,
                'face_normals': False,
                'to_world': transform,
                'bsdf': MaterialFactory.glass(),
            }))

        return meshes

    def _select_other_bsdf(self) -> Dict:
        for mat_name, mat_info in self.materials.items():
            if mat_info.get('is_glass', False):
                continue
            if OBJSplitter._is_ceiling(mat_name):
                continue
            tex_path = mat_info.get('textures', {}).get('diffuse')
            if tex_path and os.path.exists(tex_path):
                print(f"  [BSDF] 使用紋理材質: {mat_name}")
                return MaterialFactory.diffuse_textured(tex_path)

        return MaterialFactory.diffuse(0.5)

    def _select_ceiling_bsdf(self) -> Dict:
        for mat_name, mat_info in self.materials.items():
            if not OBJSplitter._is_ceiling(mat_name):
                continue
            tex_path = mat_info.get('textures', {}).get('diffuse')
            if tex_path and os.path.exists(tex_path):
                return MaterialFactory.diffuse_textured(tex_path)

        return MaterialFactory.diffuse(0.85)

    def _transform_point(self, point: Tuple[float, float, float]) -> List[float]:
        x, y, z = point
        return [mm_to_m(x), mm_to_m(z), mm_to_m(y)]


# ============================================================
# PIDS 無偏振渲染器
# ============================================================

class PIDSRendererNopol:
    """PIDS 無偏振立體渲染器（用於基線比較）"""

    def __init__(self):
        self.variant = setup_mitsuba()

    def render_scene(self, obj_path: str, output_dir: str, scene_name: str = None):
        """
        渲染單一場景（無偏振）

        輸出:
            - {scene}_left.exr  (左相機灰階)
            - {scene}_right.exr (右相機灰階)
            - {scene}_depth.exr
            - {scene}_disparity.exr
            - {scene}_glass_mask.exr
            - {scene}_params.json
        """
        if scene_name is None:
            scene_name = Path(obj_path).stem

        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"渲染場景 (nopol): {scene_name}")
        print(f"{'='*60}")

        left_pos = Config.left_camera_position()
        right_pos = Config.right_camera_position()
        depth_pos = Config.depth_camera_position()

        left_target = Config.camera_target_for_position(left_pos)
        right_target = Config.camera_target_for_position(right_pos)
        depth_target = Config.camera_target_for_position(depth_pos)

        print(f"[配置] 左相機位置: {left_pos}")
        print(f"[配置] 右相機位置: {right_pos}")
        print(f"[配置] 視線方向: {Config.forward_direction()} (平行光軸)")
        print(f"[配置] 無偏振模式")

        builder = SceneBuilder(obj_path)

        # 渲染左相機（無偏振）
        print(f"\n[1/4] 渲染左相機...")
        left_image = self._render_camera(builder, left_pos, left_target)

        # 渲染右相機（無偏振）
        print(f"\n[2/4] 渲染右相機...")
        right_image = self._render_camera(builder, right_pos, right_target)

        # 渲染深度圖
        print(f"\n[3/4] 渲染深度圖...")
        depth = self._render_depth(builder, depth_pos, depth_target)

        # 渲染玻璃 mask
        print(f"\n[4/4] 渲染玻璃 mask...")
        glass_mask = self._render_glass_mask(builder, depth_pos, depth_target)

        # 計算視差
        disparity = self._compute_disparity(depth)

        # 保存結果
        print(f"\n[保存] 輸出檔案...")
        self._save_outputs(
            output_dir, scene_name,
            left_image, right_image, depth, disparity, glass_mask
        )

        # 清理臨時檔案
        builder.cleanup()

        print(f"\n{'='*60}")
        print(f"場景 {scene_name} 渲染完成 (nopol)!")
        print(f"{'='*60}\n")

    def _render_camera(self,
                       builder: SceneBuilder,
                       position: Tuple[float, float, float],
                       target: Tuple[float, float, float]) -> np.ndarray:
        """渲染單一相機視角"""
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

        # 提取灰階（luminance film 輸出單通道）
        if result.ndim == 3:
            result = result[:, :, 0]

        print(f"  [渲染] 完成，形狀: {result.shape}")
        print(f"  [渲染] 範圍: [{result.min():.4f}, {result.max():.4f}]")

        return result

    def _render_depth(self,
                      builder: SceneBuilder,
                      position: Tuple[float, float, float],
                      target: Tuple[float, float, float]) -> np.ndarray:
        """渲染深度圖"""
        scene_dict = builder.build(position, target, 256)

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

        if img_np.ndim == 3 and img_np.shape[2] >= 2:
            depth = img_np[:, :, 1]
        else:
            depth = img_np[:, :, 0] if img_np.ndim == 3 else img_np

        print(f"  [深度] 範圍: [{depth.min():.4f}, {depth.max():.4f}] m")

        return depth.astype(np.float32)

    def _compute_disparity(self, depth: np.ndarray) -> np.ndarray:
        """計算視差圖"""
        fov_rad = np.radians(Config.FOV)
        focal_px = (Config.WIDTH / 2) / np.tan(fov_rad / 2)
        baseline_m = mm_to_m(Config.BASELINE)

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
        """渲染玻璃區域 mask"""
        if not builder.has_glass:
            print(f"  [Glass Mask] 場景無玻璃，返回空 mask")
            return np.zeros((Config.HEIGHT, Config.WIDTH), dtype=np.float32)

        pos_m = builder._transform_point(position)
        tgt_m = builder._transform_point(target)

        transform = mi.ScalarTransform4f.scale([0.001, 0.001, 0.001]) @ \
                    mi.ScalarTransform4f.rotate([1, 0, 0], -90)

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
                    'sample_count': 4,
                },
            },
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

        if img_np.ndim == 3 and img_np.shape[2] >= 2:
            depth_channel = img_np[:, :, 1]
        elif img_np.ndim == 3:
            depth_channel = img_np[:, :, 0]
        else:
            depth_channel = img_np

        glass_mask = (depth_channel > 0).astype(np.float32)

        pixel_count = int(np.sum(glass_mask))
        pixel_ratio = pixel_count / (Config.WIDTH * Config.HEIGHT)
        print(f"  [Glass Mask] 玻璃像素: {pixel_count} ({pixel_ratio*100:.1f}%)")

        return glass_mask

    def _save_outputs(self,
                      output_dir: str,
                      scene_name: str,
                      left_image: np.ndarray,
                      right_image: np.ndarray,
                      depth: np.ndarray,
                      disparity: np.ndarray,
                      glass_mask: np.ndarray = None):
        """保存所有輸出"""
        # EXR 檔案 (使用不同的命名以區分 nopol)
        self._save_exr(left_image, f"{output_dir}/{scene_name}_left.exr")
        self._save_exr(right_image, f"{output_dir}/{scene_name}_right.exr")
        self._save_exr(depth, f"{output_dir}/{scene_name}_depth.exr")
        self._save_exr(disparity, f"{output_dir}/{scene_name}_disparity.exr")

        # 預覽 PNG
        if Config.SAVE_PREVIEW:
            vmax = max(left_image.max(), right_image.max())
            self._save_png_fixed(left_image, f"{output_dir}/{scene_name}_left.png", 0, vmax)
            self._save_png_fixed(right_image, f"{output_dir}/{scene_name}_right.png", 0, vmax)

            # 深度圖
            self._save_depth_png(depth, f"{output_dir}/{scene_name}_depth.png")

        # 保存玻璃 mask
        if glass_mask is not None:
            self._save_exr(glass_mask, f"{output_dir}/{scene_name}_glass_mask.exr")
            self._save_png(glass_mask, f"{output_dir}/{scene_name}_glass_mask.png")

        # 參數 JSON
        params = {
            'scene_name': scene_name,
            'mode': 'nopol',
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
                'left_range': [float(left_image.min()), float(left_image.max())],
                'right_range': [float(right_image.min()), float(right_image.max())],
                'depth_range_m': [float(depth[depth > 0].min()) if (depth > 0).any() else 0,
                                  float(depth[depth > 0].max()) if (depth > 0).any() else 0],
            },
        }

        with open(f"{output_dir}/{scene_name}_params.json", 'w') as f:
            json.dump(params, f, indent=2)

    def _save_exr(self, image: np.ndarray, path: str):
        if image.ndim == 3:
            image = image[:, :, 0]
        bitmap = mi.Bitmap(image.astype(np.float32))
        bitmap.write(path)
        print(f"    -> {path}")

    def _save_png(self, image: np.ndarray, path: str):
        if image.ndim == 3:
            image = image[:, :, 0]
        vmin, vmax = image.min(), image.max()
        if vmax > vmin:
            normalized = ((image - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(image, dtype=np.uint8)
        cv2.imwrite(path, normalized)

    def _save_png_fixed(self, image: np.ndarray, path: str, vmin: float, vmax: float):
        if image.ndim == 3:
            image = image[:, :, 0]
        clipped = np.clip(image, vmin, vmax)
        if vmax > vmin:
            normalized = ((clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(image, dtype=np.uint8)
        cv2.imwrite(path, normalized)

    def _save_depth_png(self, depth: np.ndarray, path: str):
        valid = depth > 0
        if not valid.any():
            cv2.imwrite(path, np.zeros((Config.HEIGHT, Config.WIDTH), dtype=np.uint8))
            return

        vmin, vmax = depth[valid].min(), depth[valid].max()
        normalized = np.zeros_like(depth)
        normalized[valid] = (depth[valid] - vmin) / (vmax - vmin + 1e-6)
        normalized = (normalized * 255).astype(np.uint8)
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
        description='PIDS Stage 1 Renderer v4.0.0-nopol (No Polarization)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 單一場景
  python pids_renderer_textured_nopol.py --scene scene_0001.obj --output ./output_nopol

  # 批次渲染
  python pids_renderer_textured_nopol.py --input_dir ./scenes --output ./output_nopol --max_scenes 100
        """
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--scene', type=str, help='單一 OBJ 場景')
    input_group.add_argument('--input_dir', type=str, help='OBJ 場景目錄')

    parser.add_argument('--output', type=str, required=True, help='輸出目錄')
    parser.add_argument('--spp', type=int, default=Config.SPP, help=f'SPP (預設: {Config.SPP})')
    parser.add_argument('--max_scenes', type=int, default=None, help='最大場景數')
    parser.add_argument('--skip', type=int, default=0, help='跳過前 N 個場景')
    parser.add_argument('--no_preview', action='store_true', help='不保存預覽 PNG')

    args = parser.parse_args()

    Config.SPP = args.spp
    Config.SAVE_PREVIEW = not args.no_preview

    if args.scene:
        scenes = [Path(args.scene)]
    else:
        all_objs = sorted(Path(args.input_dir).glob('*.obj'))
        scenes = [f for f in all_objs
                  if not any(x in f.stem for x in ['_glass', '_ceiling', '_other'])]
        if args.skip > 0:
            scenes = scenes[args.skip:]
        if args.max_scenes:
            scenes = scenes[:args.max_scenes]

    print(f"[PIDS Renderer v4.0.0-nopol]")
    print(f"  模式: 無偏振 (baseline)")
    print(f"  待渲染: {len(scenes)} 個場景")

    start_time = time.time()

    renderer = PIDSRendererNopol()

    for i, scene_path in enumerate(scenes):
        print(f"\n進度 {i+1}/{len(scenes)}")
        Config.randomize_for_augmentation(seed=hash(scene_path.stem) % 2**32)
        try:
            renderer.render_scene(str(scene_path), args.output)
        except Exception as e:
            print(f"錯誤 {scene_path}: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"\n總耗時: {elapsed/60:.1f} 分鐘")
    print(f"平均每場景: {elapsed/len(scenes):.1f} 秒")


if __name__ == '__main__':
    main()
