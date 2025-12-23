"""
PIDS Mitsuba 3 進階渲染器 (Stage I)
====================================

此版本實現更精確的偏振物理模型:
1. 分離漫反射和鏡面反射分量
2. 根據 Fresnel 方程計算偏振反射
3. 支援 Mitsuba 3 偏振 variant (如果可用)

物理模型:
=========
根據論文公式:
- I∥(u) = α · (Id + Ib) + Is      (平行偏振)
- I⊥(u) = α · (Id + Ib)            (交叉偏振)

其中:
- α ≈ 0.5 (偏振片透射係數)
- Id: 漫反射分量
- Ib: 背景/環境光分量
- Is: 鏡面反射分量 (偏振)

實現策略:
=========
方法 A: 分離渲染 (Separate Rendering)
- 渲染 1: 完整場景 → I_full
- 渲染 2: 純漫反射場景 → I_diffuse
- 計算: Is = I_full - I_diffuse
- 組合: I∥ = α·I_diffuse + Is, I⊥ = α·I_diffuse

方法 B: 材質修改 (Material Modification)
- I∥: 使用標準 dielectric (保留 Fresnel 反射)
- I⊥: 使用 roughdielectric + 高粗糙度 (散射/抑制反射)

使用方式:
    python pids_renderer_advanced.py --scene_file scene.obj --output_dir ./output
    python pids_renderer_advanced.py --scene_file scene.obj --output_dir ./output --method separate
"""

from __future__ import annotations  # 延遲類型註解評估

import mitsuba as mi
import drjit as dr
import numpy as np
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

# 導入配置
try:
    from pids_config import get_stage1_config, CAMERA_CONFIG, CHAMBER_CONFIG, LIGHTING_CONFIG
except ImportError:
    print("[WARNING] pids_config.py 未找到，使用預設配置")
    
    def get_stage1_config():
        return {
            'render': {'width': 640, 'height': 480, 'spp': 256, 'max_depth': 8},
            'camera': {'baseline': 65.0, 'fov': 65.0, 'position_y': 150.0, 'position_z': 0.0},
            'chamber': {'dof_near': 528.0, 'dof_far': 695.0},
            'lighting': {'led': {'enabled': True, 'intensity': 5.0, 'color': [1,1,1], 'offset_y': 200.0, 'offset_z': -100.0, 'size_x': 300.0, 'size_y': 200.0}, 'ambient': {'enabled': False}},
            'material': {'polarization': {'alpha': 0.5, 'specular_suppression': 0.05}},
        }


# ============================================================
# Mitsuba Variant 設定
# ============================================================

def setup_variant(prefer_polarized: bool = False):
    """
    設定 Mitsuba variant
    
    Args:
        prefer_polarized: 是否優先使用偏振 variant
    """
    if prefer_polarized:
        # 偏振 variants
        preferred = [
            'cuda_ad_spectral_polarized',
            'llvm_ad_spectral_polarized',
            'scalar_spectral_polarized',
            'cuda_ad_rgb',
            'cuda_rgb',
            'llvm_ad_rgb',
            'scalar_rgb',
        ]
    else:
        # 標準 RGB variants (更快)
        preferred = [
            'cuda_ad_rgb',
            'cuda_rgb',
            'llvm_ad_rgb',
            'scalar_rgb',
        ]
    
    available = mi.variants()
    
    for v in preferred:
        if v in available:
            mi.set_variant(v)
            print(f"[INFO] Mitsuba variant: {v}")
            return v
    
    if available:
        mi.set_variant(available[0])
        print(f"[WARNING] 使用備選 variant: {available[0]}")
        return available[0]
    
    raise RuntimeError("無可用的 Mitsuba variant!")


# ============================================================
# 工具函數
# ============================================================

def mm_to_m(v): 
    """毫米轉公尺"""
    return v * 0.001


def parse_mtl(mtl_path: str) -> Dict:
    """解析 MTL 檔案"""
    materials = {}
    current = None
    
    if not os.path.exists(mtl_path):
        return materials
    
    with open(mtl_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            
            cmd = parts[0].lower()
            
            if cmd == 'newmtl' and len(parts) > 1:
                current = parts[1]
                materials[current] = {'kd': [0.5, 0.5, 0.5], 'is_glass': False}
            elif current:
                if cmd == 'kd' and len(parts) >= 4:
                    materials[current]['kd'] = [float(x) for x in parts[1:4]]
                elif cmd in ('d', 'tr') and len(parts) >= 2:
                    if float(parts[1]) < 0.9:
                        materials[current]['is_glass'] = True
    
    return materials


def is_glass_material(name: str) -> bool:
    """判斷是否為玻璃材質"""
    name_lower = name.lower()
    return any(k in name_lower for k in ['glass', 'clear', 'acrylic', 'transparent'])


# ============================================================
# 場景建構器
# ============================================================

class PIDSSceneBuilder:
    """PIDS 場景建構器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or get_stage1_config()
        self._mtl_cache = {}
    
    def _get_camera_transform(self, side: str = 'left') -> mi.ScalarTransform4f:
        """取得相機變換矩陣"""
        cfg = self.config
        cam = cfg['camera']
        chamber = cfg['chamber']
        
        baseline_m = mm_to_m(cam['baseline'])
        
        # 相機 X 位置
        if side == 'left':
            cam_x = -baseline_m / 2
        else:
            cam_x = baseline_m / 2
        
        # Y (高度), Z (深度)
        cam_y = mm_to_m(cam['position_y'])
        cam_z = mm_to_m(cam['position_z'])
        
        # 目標點
        target_z = mm_to_m((chamber['dof_near'] + chamber['dof_far']) / 2)
        
        return mi.ScalarTransform4f.look_at(
            origin=[cam_x, cam_y, cam_z],
            target=[0, cam_y, target_z],
            up=[0, 1, 0]
        )
    
    def _create_sensor(self, side: str = 'left', spp: int = None) -> Dict:
        """建立感測器 (相機) - 灰階輸出"""
        cfg = self.config
        render = cfg['render']
        cam = cfg['camera']
        
        return {
            'type': 'perspective',
            'fov': cam['fov'],
            'fov_axis': 'x',
            'to_world': self._get_camera_transform(side),
            'film': {
                'type': 'hdrfilm',
                'width': render['width'],
                'height': render['height'],
                'pixel_format': 'luminance',  # 灰階輸出
                'component_format': 'float32',
            },
            'sampler': {
                'type': 'independent',
                'sample_count': spp or render['spp'],
            },
        }
    
    def _create_light(self) -> Dict:
        """建立光源"""
        cfg = self.config
        led = cfg['lighting']['led']
        cam = cfg['camera']
        
        if not led['enabled']:
            return {}
        
        # 光源位置
        light_x = 0
        light_y = mm_to_m(cam['position_y'] + led['offset_y'])
        light_z = mm_to_m(cam['position_z'] + led['offset_z'])
        
        # 目標點
        chamber = cfg['chamber']
        target_z = mm_to_m((chamber['dof_near'] + chamber['dof_far']) / 2)
        
        return {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.look_at(
                origin=[light_x, light_y, light_z],
                target=[0, mm_to_m(cam['position_y']), target_z],
                up=[0, 1, 0]
            ) @ mi.ScalarTransform4f.scale([
                mm_to_m(led['size_x']) / 2,
                mm_to_m(led['size_y']) / 2,
                1
            ]),
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    'value': [c * led['intensity'] for c in led['color']],
                },
            },
        }
    
    def build_scene_parallel(self, obj_path: str, side: str = 'left') -> mi.Scene:
        """
        建立平行偏振場景 (I∥)
        - 使用標準 dielectric 材質 (保留鏡面反射)
        """
        return self._build_scene(obj_path, side, mode='parallel')
    
    def build_scene_cross(self, obj_path: str, side: str = 'right') -> mi.Scene:
        """
        建立交叉偏振場景 (I⊥)
        - 使用 roughdielectric 材質 (抑制鏡面反射)
        """
        return self._build_scene(obj_path, side, mode='cross')
    
    def build_scene_diffuse_only(self, obj_path: str, side: str = 'left') -> mi.Scene:
        """
        建立純漫反射場景 (用於分離渲染)
        - 所有材質轉為漫反射
        """
        return self._build_scene(obj_path, side, mode='diffuse_only')
    
    def _build_scene(self, obj_path: str, side: str, mode: str) -> mi.Scene:
        """內部場景建構方法"""
        cfg = self.config
        
        # 解析 MTL
        mtl_path = obj_path.replace('.obj', '.mtl')
        if mtl_path not in self._mtl_cache:
            self._mtl_cache[mtl_path] = parse_mtl(mtl_path)
        mtl_materials = self._mtl_cache[mtl_path]
        
        # 場景字典
        scene_dict = {
            'type': 'scene',
            'integrator': {
                'type': 'path',
                'max_depth': cfg['render']['max_depth'],
            },
        }
        
        # 建立材質
        for mat_name, mat_props in mtl_materials.items():
            bsdf_name = f'bsdf_{mat_name}'
            
            if mat_props['is_glass'] or is_glass_material(mat_name):
                # 玻璃材質
                if mode == 'parallel':
                    # 平行偏振: 標準 dielectric
                    scene_dict[bsdf_name] = {
                        'type': 'dielectric',
                        'int_ior': 1.5,
                        'ext_ior': 1.0,
                    }
                elif mode == 'cross':
                    # 交叉偏振: roughdielectric 抑制鏡面
                    scene_dict[bsdf_name] = {
                        'type': 'roughdielectric',
                        'int_ior': 1.5,
                        'ext_ior': 1.0,
                        'alpha': 0.35,  # 增加粗糙度抑制鏡面
                        'distribution': 'ggx',
                    }
                else:  # diffuse_only
                    # 純漫反射 (透明)
                    scene_dict[bsdf_name] = {
                        'type': 'diffuse',
                        'reflectance': {'type': 'rgb', 'value': [0.1, 0.1, 0.1]},
                    }
            else:
                # 漫反射材質
                scene_dict[bsdf_name] = {
                    'type': 'diffuse',
                    'reflectance': {'type': 'rgb', 'value': mat_props['kd']},
                }
        
        # 預設材質
        scene_dict['bsdf_default'] = {
            'type': 'diffuse',
            'reflectance': {'type': 'rgb', 'value': [0.5, 0.5, 0.5]},
        }
        
        # 載入網格
        scene_dict['mesh'] = {
            'type': 'obj',
            'filename': obj_path,
            'to_world': mi.ScalarTransform4f.scale([0.001, 0.001, 0.001]),
        }
        
        # 相機
        scene_dict['sensor'] = self._create_sensor(side)
        
        # 光源
        light = self._create_light()
        if light:
            scene_dict['light'] = light
        
        return mi.load_dict(scene_dict)


# ============================================================
# 渲染器
# ============================================================

class PIDSRenderer:
    """PIDS 渲染器"""
    
    def __init__(self, config: Dict = None, method: str = 'material'):
        """
        初始化渲染器
        
        Args:
            config: 配置字典
            method: 渲染方法
                - 'material': 材質修改法 (快速)
                - 'separate': 分離渲染法 (精確)
        """
        self.config = config or get_stage1_config()
        self.method = method
        self.builder = PIDSSceneBuilder(self.config)
    
    def render(self, obj_path: str, output_dir: str, scene_name: str = None):
        """
        渲染場景
        
        Returns:
            Dict 包含所有輸出檔案路徑
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if scene_name is None:
            scene_name = Path(obj_path).stem
        
        print(f"\n{'='*60}")
        print(f"PIDS 渲染: {scene_name}")
        print(f"方法: {self.method}")
        print(f"{'='*60}")
        
        cfg = self.config
        alpha = cfg['material']['polarization']['alpha']
        
        if self.method == 'separate':
            return self._render_separate(obj_path, output_dir, scene_name, alpha)
        else:
            return self._render_material(obj_path, output_dir, scene_name)
    
    def _render_material(self, obj_path: str, output_dir: str, scene_name: str) -> Dict:
        """材質修改渲染法 - 灰階輸出"""
        cfg = self.config
        spp = cfg['render']['spp']
        
        results = {}
        
        # 輔助函數: 確保灰階
        def ensure_grayscale(img_np):
            if img_np.ndim == 3:
                if img_np.shape[2] == 3:
                    # RGB to luminance
                    return 0.2126 * img_np[:,:,0] + 0.7152 * img_np[:,:,1] + 0.0722 * img_np[:,:,2]
                else:
                    return img_np[:,:,0]
            return img_np
        
        # 1. 渲染 I∥ (左相機，平行偏振) - 灰階
        print("\n[1/4] 渲染 I∥ (parallel, grayscale)...")
        t0 = time.time()
        
        scene_par = self.builder.build_scene_parallel(obj_path, 'left')
        img_par = mi.render(scene_par, spp=spp)
        img_par_np = ensure_grayscale(np.array(img_par))
        
        par_path = os.path.join(output_dir, f"{scene_name}_I_parallel.exr")
        mi.Bitmap(img_par_np.astype(np.float32)).write(par_path)
        results['I_parallel'] = par_path
        print(f"    耗時: {time.time()-t0:.2f}s")
        
        # 2. 渲染 I⊥ (右相機，交叉偏振) - 灰階
        print("\n[2/4] 渲染 I⊥ (cross, grayscale)...")
        t0 = time.time()
        
        scene_cross = self.builder.build_scene_cross(obj_path, 'right')
        img_cross = mi.render(scene_cross, spp=spp)
        img_cross_np = ensure_grayscale(np.array(img_cross))
        
        cross_path = os.path.join(output_dir, f"{scene_name}_I_cross.exr")
        mi.Bitmap(img_cross_np.astype(np.float32)).write(cross_path)
        results['I_cross'] = cross_path
        print(f"    耗時: {time.time()-t0:.2f}s")
        
        # 3. 渲染深度圖
        print("\n[3/4] 渲染深度圖...")
        depth, disparity = self._render_depth(obj_path, 'left')
        
        depth_path = os.path.join(output_dir, f"{scene_name}_depth.exr")
        mi.Bitmap(depth.astype(np.float32)).write(depth_path)
        results['depth'] = depth_path
        
        disp_path = os.path.join(output_dir, f"{scene_name}_disparity.exr")
        mi.Bitmap(disparity.astype(np.float32)).write(disp_path)
        results['disparity'] = disp_path
        
        # 4. 儲存預覽和參數
        print("\n[4/4] 儲存預覽 (灰階)...")
        self._save_previews(output_dir, scene_name, img_par_np, img_cross_np, depth)
        results['params'] = self._save_params(output_dir, scene_name, obj_path, depth, disparity)
        
        print(f"\n✓ 完成: {scene_name}")
        return results
    
    def _render_separate(self, obj_path: str, output_dir: str, scene_name: str, alpha: float) -> Dict:
        """
        分離渲染法 - 灰階輸出
        
        I∥ = α·I_diffuse + I_specular
        I⊥ = α·I_diffuse
        """
        cfg = self.config
        spp = cfg['render']['spp']
        
        results = {}
        
        # 輔助函數: 確保灰階
        def ensure_grayscale(img_np):
            if img_np.ndim == 3:
                if img_np.shape[2] >= 3:
                    # RGB to luminance
                    return 0.2126 * img_np[:,:,0] + 0.7152 * img_np[:,:,1] + 0.0722 * img_np[:,:,2]
                else:
                    return img_np[:,:,0]
            return img_np
        
        # 1. 渲染完整場景 (I_full = I_diffuse + I_specular)
        print("\n[1/5] 渲染完整場景 (grayscale)...")
        t0 = time.time()
        
        scene_full = self.builder.build_scene_parallel(obj_path, 'left')
        img_full = mi.render(scene_full, spp=spp)
        img_full_np = ensure_grayscale(np.array(img_full))
        print(f"    耗時: {time.time()-t0:.2f}s")
        
        # 2. 渲染純漫反射場景
        print("\n[2/5] 渲染漫反射場景 (grayscale)...")
        t0 = time.time()
        
        scene_diff = self.builder.build_scene_diffuse_only(obj_path, 'left')
        img_diff = mi.render(scene_diff, spp=spp)
        img_diff_np = ensure_grayscale(np.array(img_diff))
        print(f"    耗時: {time.time()-t0:.2f}s")
        
        # 3. 計算鏡面分量
        print("\n[3/5] 計算偏振影像 (grayscale)...")
        img_specular = np.maximum(img_full_np - img_diff_np, 0)
        
        # I∥ = α·I_diffuse + I_specular (左相機)
        img_parallel = alpha * img_diff_np + img_specular
        
        # I⊥ = α·I_diffuse (右相機 - 需要重新渲染以獲得正確視角)
        scene_diff_right = self.builder.build_scene_diffuse_only(obj_path, 'right')
        img_diff_right = mi.render(scene_diff_right, spp=spp)
        img_diff_right_np = ensure_grayscale(np.array(img_diff_right))
        img_cross = alpha * img_diff_right_np
        
        # 儲存 (灰階)
        par_path = os.path.join(output_dir, f"{scene_name}_I_parallel.exr")
        mi.Bitmap(img_parallel.astype(np.float32)).write(par_path)
        results['I_parallel'] = par_path
        
        cross_path = os.path.join(output_dir, f"{scene_name}_I_cross.exr")
        mi.Bitmap(img_cross.astype(np.float32)).write(cross_path)
        results['I_cross'] = cross_path
        
        # 4. 深度圖
        print("\n[4/5] 渲染深度圖...")
        depth, disparity = self._render_depth(obj_path, 'left')
        
        depth_path = os.path.join(output_dir, f"{scene_name}_depth.exr")
        mi.Bitmap(depth.astype(np.float32)).write(depth_path)
        results['depth'] = depth_path
        
        disp_path = os.path.join(output_dir, f"{scene_name}_disparity.exr")
        mi.Bitmap(disparity.astype(np.float32)).write(disp_path)
        results['disparity'] = disp_path
        
        # 5. 預覽
        print("\n[5/5] 儲存預覽 (灰階)...")
        self._save_previews(output_dir, scene_name, img_parallel, img_cross, depth)
        results['params'] = self._save_params(output_dir, scene_name, obj_path, depth, disparity)
        
        print(f"\n✓ 完成: {scene_name}")
        return results
    
    def _render_depth(self, obj_path: str, side: str) -> Tuple[np.ndarray, np.ndarray]:
        """渲染深度圖並計算視差"""
        cfg = self.config
        cam = cfg['camera']
        render = cfg['render']
        
        # 使用 AOV integrator 獲取深度
        scene_dict = {
            'type': 'scene',
            'integrator': {
                'type': 'aov',
                'aovs': 'dd:depth',
                'integrator': {'type': 'path', 'max_depth': 1},
            },
            'mesh': {
                'type': 'obj',
                'filename': obj_path,
                'to_world': mi.ScalarTransform4f.scale([0.001, 0.001, 0.001]),
                'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.5, 0.5, 0.5]}},
            },
            'sensor': self.builder._create_sensor(side, spp=16),
            'light': {'type': 'constant', 'radiance': {'type': 'rgb', 'value': [1, 1, 1]}},
        }
        
        scene = mi.load_dict(scene_dict)
        img = mi.render(scene, spp=16)
        img_np = np.array(img)
        
        # 提取深度
        if img_np.shape[2] > 3:
            depth = img_np[:, :, 3]
        else:
            depth = np.mean(img_np[:, :, :3], axis=2)
        
        # 計算視差
        fov_rad = np.radians(cam['fov'])
        focal_px = render['width'] / (2 * np.tan(fov_rad / 2))
        baseline_m = mm_to_m(cam['baseline'])
        
        depth_safe = np.maximum(depth, 1e-6)
        disparity = (focal_px * baseline_m) / depth_safe
        disparity[depth <= 0] = 0
        
        return depth, disparity
    
    def _save_previews(self, output_dir: str, scene_name: str,
                       img_par: np.ndarray, img_cross: np.ndarray, depth: np.ndarray):
        """儲存預覽圖 (灰階)"""
        try:
            from PIL import Image
            
            def tonemap_gray(img):
                """色調映射 (灰階)"""
                # 確保是 2D 灰階
                if img.ndim == 3:
                    img = img[:,:,0] if img.shape[2] == 1 else np.mean(img[:,:,:3], axis=2)
                img = np.clip(img, 0, None)
                img = img / (1 + img)
                img = np.power(img, 1/2.2)
                return (img * 255).astype(np.uint8)
            
            # I_parallel (灰階)
            preview_par = tonemap_gray(img_par)
            Image.fromarray(preview_par, mode='L').save(
                os.path.join(output_dir, f"{scene_name}_I_parallel.png"))
            
            # I_cross (灰階)
            preview_cross = tonemap_gray(img_cross)
            Image.fromarray(preview_cross, mode='L').save(
                os.path.join(output_dir, f"{scene_name}_I_cross.png"))
            
            # 偏振差異 (灰階)
            # 確保兩者都是 2D
            img_par_2d = img_par if img_par.ndim == 2 else img_par[:,:,0]
            img_cross_2d = img_cross if img_cross.ndim == 2 else img_cross[:,:,0]
            
            diff = np.abs(img_par_2d - img_cross_2d)
            diff_norm = diff / (np.max(diff) + 1e-6)
            Image.fromarray((diff_norm * 255).astype(np.uint8), mode='L').save(
                os.path.join(output_dir, f"{scene_name}_polarization_diff.png"))
            
            # 深度 (灰階)
            depth_valid = depth[depth > 0]
            if len(depth_valid) > 0:
                d_min, d_max = depth_valid.min(), depth_valid.max()
                depth_norm = (depth - d_min) / (d_max - d_min + 1e-6)
                depth_norm = np.clip(depth_norm, 0, 1)
                Image.fromarray((depth_norm * 255).astype(np.uint8), mode='L').save(
                    os.path.join(output_dir, f"{scene_name}_depth.png"))
                    
        except ImportError:
            print("[WARNING] PIL 未安裝，跳過預覽圖")
    
    def _save_params(self, output_dir: str, scene_name: str, obj_path: str,
                     depth: np.ndarray, disparity: np.ndarray) -> str:
        """儲存參數"""
        cfg = self.config
        cam = cfg['camera']
        render = cfg['render']
        
        fov_rad = np.radians(cam['fov'])
        focal_px = render['width'] / (2 * np.tan(fov_rad / 2))
        
        params = {
            'scene_name': scene_name,
            'obj_file': os.path.basename(obj_path),
            'method': self.method,
            'render': {
                'width': render['width'],
                'height': render['height'],
                'spp': render['spp'],
            },
            'camera': {
                'baseline_mm': cam['baseline'],
                'fov_deg': cam['fov'],
                'focal_length_px': float(focal_px),
            },
            'depth_stats': {
                'min_m': float(depth[depth > 0].min()) if np.any(depth > 0) else 0,
                'max_m': float(depth[depth > 0].max()) if np.any(depth > 0) else 0,
            },
            'disparity_stats': {
                'min_px': float(disparity[disparity > 0].min()) if np.any(disparity > 0) else 0,
                'max_px': float(disparity[disparity > 0].max()) if np.any(disparity > 0) else 0,
            },
        }
        
        path = os.path.join(output_dir, f"{scene_name}_params.json")
        with open(path, 'w') as f:
            json.dump(params, f, indent=2)
        
        return path


# ============================================================
# 批次處理
# ============================================================

def render_batch(input_dir: str, output_dir: str, method: str = 'material',
                 max_scenes: int = None):
    """批次渲染"""
    obj_files = list(Path(input_dir).glob('*.obj'))
    for subdir in Path(input_dir).iterdir():
        if subdir.is_dir():
            obj_files.extend(subdir.glob('*.obj'))
    
    obj_files = sorted(obj_files)
    
    if max_scenes:
        obj_files = obj_files[:max_scenes]
    
    print(f"\n找到 {len(obj_files)} 個場景")
    
    renderer = PIDSRenderer(method=method)
    results = []
    
    for i, obj_path in enumerate(obj_files):
        print(f"\n[{i+1}/{len(obj_files)}]")
        try:
            result = renderer.render(str(obj_path), output_dir)
            results.append(result)
        except Exception as e:
            print(f"[ERROR] {obj_path}: {e}")
    
    print(f"\n完成: {len(results)}/{len(obj_files)}")
    return results


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='PIDS Mitsuba 3 進階渲染器')
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--scene_file', type=str, help='單一 OBJ 場景')
    input_group.add_argument('--input_dir', type=str, help='場景目錄 (批次)')
    
    parser.add_argument('--output_dir', type=str, required=True, help='輸出目錄')
    parser.add_argument('--method', type=str, default='material',
                        choices=['material', 'separate'],
                        help='渲染方法: material(快速) 或 separate(精確)')
    parser.add_argument('--spp', type=int, default=256, help='每像素樣本數')
    parser.add_argument('--max_scenes', type=int, help='最大場景數 (批次)')
    
    args = parser.parse_args()
    
    # 設定 variant
    setup_variant()
    
    # 更新配置
    config = get_stage1_config()
    config['render']['spp'] = args.spp
    
    if args.scene_file:
        renderer = PIDSRenderer(config=config, method=args.method)
        renderer.render(args.scene_file, args.output_dir)
    else:
        render_batch(args.input_dir, args.output_dir, args.method, args.max_scenes)


if __name__ == '__main__':
    main()
