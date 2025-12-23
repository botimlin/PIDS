"""
PIDS Mitsuba 3 偏振渲染器 (物理正確版)
======================================

使用 Mitsuba 3 原生的 Stokes Vector 渲染，實現物理正確的偏振效果。
所有輸出均為**灰階**格式。

物理原理:
---------
1. 偏振 LED 發出線性偏振光（假設水平偏振）
2. 光線打到玻璃表面，根據 Fresnel 方程式：
   - p-偏振 (平行入射面): 在 Brewster 角 (~56°) 反射率 Rp = 0
   - s-偏振 (垂直入射面): 反射率 Rs ≈ 15%
3. 兩個相機前各有偏振片：
   - 左相機: 平行偏振片 → 看到 I∥
   - 右相機: 垂直偏振片 → 看到 I⊥

Mitsuba 3 偏振渲染:
------------------
- 使用 `polarized` variant (如 cuda_ad_rgb_polarized)
- 渲染輸出 Stokes Vector [S0, S1, S2, S3]
- S0 = 總強度
- S1 = I_H - I_V (水平 vs 垂直偏振差)
- 提取: I∥ = (S0 + S1) / 2, I⊥ = (S0 - S1) / 2
- 最終輸出為灰階圖像

輸出格式:
---------
- 所有圖像均為單通道灰階 EXR (float32)
- left_parallel.exr: I∥ 灰階
- right_cross.exr: I⊥ 灰階
- depth.exr: 深度圖 (米)
- disparity.exr: 視差圖 (像素)

作者: PIDS Project
版本: 2.0 (物理正確偏振版 - 灰階輸出)
"""

from __future__ import annotations

import mitsuba as mi
import drjit as dr
import numpy as np
import cv2
import os
import json
import argparse
import tempfile
import shutil
import re
import math
import gc
import copy
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time


# ============================================================
# 配置
# ============================================================

CONFIG = {
    'render': {
        'width': 640,
        'height': 480,
        'spp': 65536,           # 65536 = 256² (ldsampler 標準)
        'spp_per_batch': 4096,  # 正常分批
        'max_depth': 32,        # 解決玻璃光線陷阱
    },
    'camera': {
        'baseline': 65.0,       # 基線 (mm)
        'fov': 65.0,            # 視場角 (度)
        'position_y': 100.0,    # 相機高度 (mm)
        'position_z': 360.0,    # 相機深度 (mm)
    },
    'lighting': {
        'led': {
            # 恢復之前能工作的配置
            'intensity': 100.0,
            'size_x': 180.0,
            'size_y': 100.0,
            'position_x': 0.0,
            'position_y': 290.0,
            'position_z': 450.0,
            'target_y': 150.0,
        },
        'ambient': {
            'intensity': 0.02,
        },
    },
    'materials': {
        'glass': {
            'ior': 1.5,         # 玻璃折射率
        },
    },
    'scene': {
        'object_z_min': 528.0,  # 物體區域 Z 最小值 (mm)
        'object_z_max': 695.0,  # 物體區域 Z 最大值 (mm)
    },
    'output': {
        'save_preview': True,
        'save_stokes': False,   # 是否保存 Stokes 向量
    },
}


def mm_to_meters(mm: float) -> float:
    """毫米轉換為米"""
    return mm / 1000.0


def save_grayscale_exr(image: np.ndarray, path: str):
    """
    保存灰階 EXR 圖像
    
    Args:
        image: 灰階圖像 [H, W]，float32
        path: 輸出路徑
    """
    # 確保是 2D 灰階
    if image.ndim == 3:
        # 轉灰階
        if image.shape[2] == 3:
            image = 0.2126 * image[:,:,0] + 0.7152 * image[:,:,1] + 0.0722 * image[:,:,2]
        else:
            image = image[:,:,0]
    
    # 確保是 float32
    image = image.astype(np.float32)
    
    # 創建 Mitsuba Bitmap 並保存
    # Mitsuba 會根據形狀自動判斷為單通道
    bitmap = mi.Bitmap(image)
    bitmap.write(path)
    
    print(f"    -> 保存灰階 EXR: {path} (範圍: [{image.min():.4f}, {image.max():.4f}])")


def save_grayscale_png(image: np.ndarray, path: str):
    """
    保存灰階 PNG 圖像（用於預覽）
    
    Args:
        image: 灰階圖像 [H, W]，float32
        path: 輸出路徑
    """
    # 確保是 2D 灰階
    if image.ndim == 3:
        if image.shape[2] == 3:
            image = 0.2126 * image[:,:,0] + 0.7152 * image[:,:,1] + 0.0722 * image[:,:,2]
        else:
            image = image[:,:,0]
    
    # Tone mapping：簡單的線性映射到 0-255
    img_min, img_max = image.min(), image.max()
    if img_max > img_min:
        normalized = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(image, dtype=np.uint8)
    
    cv2.imwrite(path, normalized)
    print(f"    -> 保存 PNG 預覽: {path}")


# ============================================================
# 設定 Mitsuba Variant (偏振版)
# ============================================================

def setup_polarized_variant():
    """
    設定 Mitsuba 偏振 variant
    
    偏振渲染需要 'polarized' variant
    優先使用 mono_polarized（適合灰階輸出）
    """
    available = mi.variants()
    print(f"[INFO] 可用的 Mitsuba variants: {available}")
    
    # 優先使用 CUDA mono polarized（適合灰階輸出）
    polarized_variants = [
        'cuda_ad_mono_polarized',      # CUDA + 自動微分 + 單色 + 偏振 ✓
        'cuda_mono_polarized',          # CUDA + 單色 + 偏振
        'cuda_ad_spectral_polarized',   # CUDA + 自動微分 + 光譜 + 偏振
        'cuda_spectral_polarized',      # CUDA + 光譜 + 偏振
        'cuda_ad_rgb_polarized',        # CUDA + 自動微分 + RGB + 偏振
        'cuda_rgb_polarized',           # CUDA + RGB + 偏振
        'llvm_ad_mono_polarized',       # LLVM + 自動微分 + 單色 + 偏振
        'llvm_mono_polarized',          # LLVM + 單色 + 偏振
        'llvm_ad_spectral_polarized',   # LLVM + 自動微分 + 光譜 + 偏振
        'llvm_spectral_polarized',      # LLVM + 光譜 + 偏振
        'llvm_ad_rgb_polarized',        # LLVM + 自動微分 + RGB + 偏振
        'llvm_rgb_polarized',           # LLVM + RGB + 偏振
        'scalar_spectral_polarized',    # Scalar + 光譜 + 偏振
        'scalar_rgb_polarized',         # Scalar + RGB + 偏振
    ]
    
    for variant in polarized_variants:
        if variant in available:
            mi.set_variant(variant)
            print(f"[INFO] ✓ 使用偏振 variant: {variant}")
            return variant
    
    # 如果沒有偏振 variant，報錯
    raise RuntimeError(
        f"無法找到偏振 variant！\n"
        f"可用的 variants: {available}\n"
        f"需要包含 'polarized' 的 variant"
    )


# ============================================================
# 偏振材質創建
# ============================================================

def create_polarized_glass_bsdf(ior: float = 1.5) -> Dict:
    """
    創建偏振感知的玻璃材質
    
    使用 dielectric 材質，Mitsuba 會自動計算正確的 Fresnel 偏振反射
    """
    return {
        'type': 'dielectric',
        'int_ior': ior,
        'ext_ior': 1.0,
    }


def create_diffuse_bsdf(color: Tuple[float, float, float]) -> Dict:
    """
    創建漫反射材質
    
    對於 mono/spectral variant，使用亮度值
    """
    # 計算亮度
    luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
    
    # 檢查當前 variant
    variant = mi.variant()
    if 'mono' in variant or 'spectral' in variant:
        return {
            'type': 'diffuse',
            'reflectance': {
                'type': 'spectrum',
                'value': luminance,
            },
        }
    else:
        return {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                'value': list(color),
            },
        }


# ============================================================
# 偏振光源
# ============================================================

def create_polarized_emitter(position: Tuple[float, float, float],
                              target: Tuple[float, float, float],
                              size: Tuple[float, float],
                              intensity: float,
                              polarization_angle: float = 0.0) -> List[Dict]:
    """
    創建主動偏振光源（Active Polarized Light Source）
    
    PIDS 系統需要主動偏振：光源本身發出線偏振光，
    而不是被動偏振（普通光打到表面後自然產生的部分偏振）。
    
    實現方式：
    1. 發光面（emitter）發出非偏振光
    2. 在發光面前方放置一個線性偏振片（polarizer）
    3. 光線穿過偏振片後變成線偏振光
    
    Args:
        position: 光源位置 (meters)
        target: 目標點 (meters)
        size: 光源尺寸 (width, height) (meters)
        intensity: 光源強度（會因偏振片損失約50%）
        polarization_angle: 偏振角度 (度), 0 = 水平偏振
    
    Returns:
        列表，包含發光面和偏振片的場景字典
    """
    # 計算方向向量
    direction = np.array(target) - np.array(position)
    distance = np.linalg.norm(direction)
    direction = direction / distance
    
    # 發光面變換矩陣
    emitter_transform = mi.ScalarTransform4f.look_at(
        origin=position,
        target=target,
        up=[0, 1, 0]
    ) @ mi.ScalarTransform4f.scale([size[0]/2, size[1]/2, 1])
    
    # 偏振片位置（在發光面前方一小段距離）
    polarizer_offset = 0.001  # 1mm
    polarizer_pos = np.array(position) + direction * polarizer_offset
    
    polarizer_transform = mi.ScalarTransform4f.look_at(
        origin=tuple(polarizer_pos),
        target=target,
        up=[0, 1, 0]
    ) @ mi.ScalarTransform4f.scale([size[0]/2, size[1]/2, 1])
    
    # 發光面 - 強度加倍補償偏振片損失
    emitter_dict = {
        'type': 'rectangle',
        'to_world': emitter_transform,
        'bsdf': {
            'type': 'null',
        },
        'emitter': {
            'type': 'area',
            'radiance': {
                'type': 'spectrum',
                'value': intensity * 2.0,  # 補償偏振片 50% 損失
            },
        },
    }
    
    # 偏振片 - 使用 Mitsuba 3 的 polarizer BSDF
    # polarizer 會將非偏振光轉換為線偏振光
    polarizer_dict = {
        'type': 'rectangle',
        'to_world': polarizer_transform,
        'bsdf': {
            'type': 'polarizer',
            'theta': polarization_angle,  # 偏振方向
        },
    }
    
    return [emitter_dict, polarizer_dict]


def create_camera_polarizer(camera_position: Tuple[float, float, float],
                            camera_target: Tuple[float, float, float],
                            fov: float,
                            polarizer_angle: float = 0.0) -> Dict:
    """
    創建相機前的偏振片
    
    在真實 PIDS 系統中，每個相機前都有一個偏振片：
    - 左相機: 0° (水平) - 平行偏振
    - 右相機: 90° (垂直) - 交叉偏振
    
    注意：在 Mitsuba 3 偏振模式下，這個功能通過後處理 Stokes Vector 實現，
    而不是實際放置偏振片幾何體。
    """
    # 這個函數保留給未來擴展
    # 目前 Stokes Vector 後處理已經實現了相機偏振片的效果
    pass


# ============================================================
# 偏振相機
# ============================================================

def create_polarized_camera(position: Tuple[float, float, float],
                            target: Tuple[float, float, float],
                            fov: float,
                            width: int, height: int,
                            spp: int,
                            polarizer_angle: float = 0.0) -> Dict:
    """
    創建帶偏振片的相機
    
    Args:
        position: 相機位置 (meters)
        target: 目標點 (meters)
        fov: 視場角 (度)
        width, height: 解析度
        spp: 每像素樣本數
        polarizer_angle: 偏振片角度 (度), 0 = 水平, 90 = 垂直
    
    Note:
        在偏振模式下，Stokes Vector 會在渲染後提取
        使用 hdrfilm 來接收完整 Stokes 分量
    """
    # 獲取當前 variant 來決定 pixel_format
    variant = mi.variant()
    if 'mono' in variant:
        pixel_format = 'luminance'  # mono variant 用單通道
    elif 'spectral' in variant:
        pixel_format = 'rgb'  # spectral 轉換為 RGB
    else:
        pixel_format = 'rgb'
    
    return {
        'type': 'perspective',
        'fov': fov,
        'fov_axis': 'x',
        'to_world': mi.ScalarTransform4f.look_at(
            origin=position,
            target=target,
            up=[0, 1, 0]
        ),
        'film': {
            'type': 'hdrfilm',
            'width': width,
            'height': height,
            'pixel_format': pixel_format,
            'component_format': 'float32',
            'rfilter': {
                'type': 'gaussian',
            },
        },
        'sampler': {
            # 🔥 低差異序列採樣 - 數學上最均勻的撒點方式
            'type': 'ldsampler',
            'sample_count': spp,
        },
    }


# ============================================================
# 場景解析
# ============================================================

def parse_mtl_file(mtl_path: str) -> Dict[str, Dict]:
    """解析 MTL 材質檔案"""
    materials = {}
    current_mat = None
    
    if not os.path.exists(mtl_path):
        print(f"  [WARNING] MTL 檔案不存在: {mtl_path}")
        return materials
    
    with open(mtl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if not parts:
                continue
            
            cmd = parts[0].lower()
            
            if cmd == 'newmtl':
                current_mat = parts[1] if len(parts) > 1 else 'default'
                materials[current_mat] = {
                    'diffuse': [0.5, 0.5, 0.5],
                    'is_glass': False,
                }
                # 根據名稱判斷是否為玻璃
                if is_glass_material(current_mat):
                    materials[current_mat]['is_glass'] = True
            elif current_mat:
                if cmd == 'kd':
                    materials[current_mat]['diffuse'] = [float(x) for x in parts[1:4]]
                elif cmd == 'd':
                    alpha = float(parts[1])
                    if alpha < 0.95:
                        materials[current_mat]['is_glass'] = True
                elif cmd == 'illum':
                    illum = int(parts[1])
                    if illum in [4, 6, 7, 9]:
                        materials[current_mat]['is_glass'] = True
    
    return materials


def is_glass_material(mat_name: str) -> bool:
    """判斷材質是否為玻璃"""
    name_lower = mat_name.lower()
    glass_keywords = ['glass', 'clear', 'transparent', 'window', 'door', 'partition', 'panel', 'acrylic']
    return any(k in name_lower for k in glass_keywords)


# ============================================================
# 場景構建
# ============================================================

def build_polarized_scene(obj_path: str,
                          camera_position: Tuple[float, float, float],
                          camera_target: Tuple[float, float, float],
                          spp: int) -> Dict:
    """
    構建偏振渲染場景
    
    手動分離玻璃和非玻璃幾何，分別使用正確的 BSDF
    """
    render_cfg = CONFIG['render']
    cam_cfg = CONFIG['camera']
    light_cfg = CONFIG['lighting']
    
    # 解析 MTL
    mtl_path = obj_path.replace('.obj', '.mtl')
    mtl_materials = parse_mtl_file(mtl_path)
    
    # 分離 OBJ 文件
    glass_obj, non_glass_obj = split_obj_by_material(obj_path, mtl_materials)
    
    scene_dict = {
        'type': 'scene',
        'integrator': {
            # 🔥 使用 stokes integrator 包住 path，輸出完整 Stokes Vector [S0, S1, S2, S3]
            'type': 'stokes',
            'integrator': {
                'type': 'path',
                'max_depth': render_cfg['max_depth'],
            },
        },
    }
    
    # 環境光
    if light_cfg['ambient']['intensity'] > 0:
        scene_dict['ambient'] = {
            'type': 'constant',
            'radiance': {
                'type': 'spectrum',
                'value': light_cfg['ambient']['intensity'],
            },
        }
    
    # 偏振 LED 光源 - 設計使入射角接近 Brewster 角 (~56°)
    led = light_cfg['led']
    light_x = mm_to_meters(led.get('position_x', 0))
    light_y = mm_to_meters(led['position_y'])
    light_z = mm_to_meters(led['position_z'])
    light_pos = (light_x, light_y, light_z)
    
    # 光源朝向物體區域 - 調整目標點使入射角接近 Brewster 角
    scene_cfg = CONFIG['scene']
    target_z = (scene_cfg['object_z_min'] + scene_cfg['object_z_max']) / 2
    
    # 目標點使用配置的 target_y
    light_target_y = mm_to_meters(led.get('target_y', 100))
    light_target_z = mm_to_meters(target_z)
    light_target = (light_x, light_target_y, light_target_z)
    
    # 計算並打印入射角（假設玻璃是垂直的，法線沿 -Z）
    dy = light_y - light_target_y
    dz = light_target_z - light_z
    distance = math.sqrt(dy*dy + dz*dz)
    # 對於垂直玻璃，入射角 = 光線與水平面的夾角
    angle_from_horizontal = math.degrees(math.atan2(dy, dz))
    # 對於水平表面反射，入射角就是與垂直方向的夾角
    incident_angle = angle_from_horizontal
    
    print(f"    [LIGHT] 光源位置: ({light_x*1000:.1f}, {light_y*1000:.1f}, {light_z*1000:.1f}) mm")
    print(f"    [LIGHT] 目標位置: ({light_target[0]*1000:.1f}, {light_target_y*1000:.1f}, {light_target_z*1000:.1f}) mm")
    print(f"    [LIGHT] 入射角: {incident_angle:.1f}° (Brewster角 ≈ 56.3°)")
    print(f"    [LIGHT] 強度: {led['intensity']}")
    print(f"    [LIGHT] 類型: 主動偏振光源（帶偏振片）")
    
    # 創建偏振光源（返回列表：[發光面, 偏振片]）
    light_components = create_polarized_emitter(
        position=light_pos,
        target=light_target,
        size=(mm_to_meters(led['size_x']), mm_to_meters(led['size_y'])),
        intensity=led['intensity'],
        polarization_angle=90.0,  # S偏振（垂直）- 反射率更高
    )
    
    # 添加發光面和偏振片到場景
    scene_dict['light_emitter'] = light_components[0]  # 發光面
    scene_dict['light_polarizer'] = light_components[1]  # 偏振片
    
    # 變換矩陣：mm to m + 座標轉換
    transform = mi.ScalarTransform4f.scale([0.001, 0.001, 0.001]) @ \
                mi.ScalarTransform4f.rotate([1, 0, 0], -90)
    
    # 非玻璃幾何
    if non_glass_obj and os.path.exists(non_glass_obj):
        scene_dict['mesh_opaque'] = {
            'type': 'obj',
            'filename': non_glass_obj,
            'face_normals': False,
            'to_world': transform,
        }
        print(f"    [MESH] 非玻璃幾何: {non_glass_obj}")
    
    # 玻璃幾何 - 使用光滑玻璃（完美鏡面，最佳偏振效果）
    if glass_obj and os.path.exists(glass_obj):
        scene_dict['mesh_glass'] = {
            'type': 'obj',
            'filename': glass_obj,
            'face_normals': False,
            'to_world': transform,
            'bsdf': {
                # 🔥 光滑玻璃 - 確定性計算，無散射噪點，布魯斯特角效果最強
                'type': 'dielectric',
                'int_ior': CONFIG['materials']['glass']['ior'],
                'ext_ior': 1.0,
            },
        }
        print(f"    [MESH] 玻璃幾何: {glass_obj} (dielectric 光滑玻璃, IOR={CONFIG['materials']['glass']['ior']})")
    
    # 相機
    scene_dict['sensor'] = create_polarized_camera(
        position=camera_position,
        target=camera_target,
        fov=cam_cfg['fov'],
        width=render_cfg['width'],
        height=render_cfg['height'],
        spp=spp,
    )
    
    return scene_dict


def split_obj_by_material(obj_path: str, mtl_materials: Dict) -> Tuple[Optional[str], Optional[str]]:
    """
    將 OBJ 文件按材質分割為玻璃和非玻璃兩個檔案
    
    Returns:
        (glass_obj_path, non_glass_obj_path)
    """
    # 讀取 OBJ 文件
    with open(obj_path, 'r') as f:
        lines = f.readlines()
    
    # 找出玻璃材質名稱
    glass_mat_names = set()
    for mat_name, props in mtl_materials.items():
        if props.get('is_glass', False):
            glass_mat_names.add(mat_name)
    
    print(f"    [OBJ] 玻璃材質: {glass_mat_names}")
    
    # 分離頂點、法線、紋理座標（這些是共享的）
    vertices = []
    normals = []
    texcoords = []
    
    # 按材質分離面
    glass_faces = []
    non_glass_faces = []
    current_material = None
    is_current_glass = False
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('v '):
            vertices.append(line)
        elif line.startswith('vn '):
            normals.append(line)
        elif line.startswith('vt '):
            texcoords.append(line)
        elif line.startswith('usemtl '):
            current_material = line.split()[1] if len(line.split()) > 1 else None
            is_current_glass = current_material in glass_mat_names
        elif line.startswith('f '):
            if is_current_glass:
                glass_faces.append(line)
            else:
                non_glass_faces.append(line)
    
    print(f"    [OBJ] 頂點: {len(vertices)}, 玻璃面: {len(glass_faces)}, 非玻璃面: {len(non_glass_faces)}")
    
    # 創建臨時目錄
    temp_dir = tempfile.mkdtemp(prefix='pids_obj_')
    
    glass_obj = None
    non_glass_obj = None
    
    # 共同的頭部（頂點、法線、紋理座標）
    header = vertices + normals + texcoords
    
    # 寫入玻璃 OBJ
    if glass_faces:
        glass_obj = os.path.join(temp_dir, 'glass.obj')
        with open(glass_obj, 'w') as f:
            f.write('# Glass geometry\n')
            for line in header:
                f.write(line + '\n')
            for line in glass_faces:
                f.write(line + '\n')
    
    # 寫入非玻璃 OBJ
    if non_glass_faces:
        non_glass_obj = os.path.join(temp_dir, 'opaque.obj')
        with open(non_glass_obj, 'w') as f:
            f.write('# Opaque geometry\n')
            for line in header:
                f.write(line + '\n')
            for line in non_glass_faces:
                f.write(line + '\n')
    
    return glass_obj, non_glass_obj


def load_and_setup_scene(scene_dict: Dict) -> "mi.Scene":
    """
    載入場景
    """
    # 直接載入，因為我們已經在 build_polarized_scene 中正確設置了材質
    scene = mi.load_dict(scene_dict)
    return scene


# ============================================================
# 偏振渲染核心
# ============================================================

def render_stokes_vector(scene: "mi.Scene", spp: int, spp_per_batch: int = 1024) -> np.ndarray:
    """
    渲染 Stokes Vector
    
    在偏振模式下，Mitsuba 3 渲染輸出包含完整的 Stokes 參數。
    
    Args:
        scene: Mitsuba 場景
        spp: 總樣本數
        spp_per_batch: 每批樣本數
    
    Returns:
        Stokes Vector 圖像 [H, W, 4]，包含 [S0, S1, S2, S3]
    """
    render_cfg = CONFIG['render']
    width = render_cfg['width']
    height = render_cfg['height']
    
    # 分批渲染
    n_batches = max(1, spp // spp_per_batch)
    actual_spp_per_batch = spp // n_batches
    
    print(f"  [渲染] 開始偏振渲染: {spp} SPP, {n_batches} 批次")
    
    # 累積結果
    accumulated = None
    total_weight = 0
    
    for batch_idx in range(n_batches):
        # 直接渲染，使用 spp 參數
        image = mi.render(scene, spp=actual_spp_per_batch)
        
        # 轉換為 numpy
        img_np = np.array(image)
        
        if accumulated is None:
            accumulated = img_np.astype(np.float64) * actual_spp_per_batch
        else:
            accumulated += img_np.astype(np.float64) * actual_spp_per_batch
        
        total_weight += actual_spp_per_batch
        
        print(f"    批次 {batch_idx + 1}/{n_batches} 完成")
        
        # 清理 GPU 記憶體
        del image
        dr.flush_malloc_cache()
        gc.collect()
    
    # 計算加權平均
    result = (accumulated / total_weight).astype(np.float32)
    
    # 非負處理
    result = np.maximum(result, 0)
    
    print(f"  [渲染] 完成，輸出形狀: {result.shape}")
    print(f"  [渲染] 數值範圍: [{result.min():.4f}, {result.max():.4f}]")
    
    return result


def extract_polarization_images(stokes_image: np.ndarray,
                                 polarizer_angle_parallel: float = 0.0,
                                 polarizer_angle_cross: float = 90.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    從 Stokes Vector 提取偏振圖像（灰階）
    
    對於線性偏振片（角度 θ），透射強度為：
    I(θ) = 0.5 * (S0 + S1*cos(2θ) + S2*sin(2θ))
    
    Mitsuba 3 偏振渲染輸出格式：
    - mono_polarized: [H, W, 4] = [S0, S1, S2, S3]
    - spectral_polarized: [H, W, N*4] 每個波長有 4 個 Stokes 分量
    - rgb_polarized: [H, W, 12] = [R_S0..R_S3, G_S0..G_S3, B_S0..B_S3]
    
    Args:
        stokes_image: Stokes Vector 圖像
        polarizer_angle_parallel: 平行偏振片角度 (度)
        polarizer_angle_cross: 交叉偏振片角度 (度)
    
    Returns:
        (I_parallel, I_cross): 兩個灰階偏振圖像 [H, W]
    """
    print(f"  [DEBUG] 輸入圖像形狀: {stokes_image.shape}, dtype: {stokes_image.dtype}")
    
    # 檢查輸入格式
    if stokes_image.ndim == 2:
        # 已經是灰階（非偏振渲染）
        print("  [WARNING] 輸入為 2D 灰階，無偏振信息")
        return stokes_image.astype(np.float32), stokes_image.astype(np.float32)
    
    h, w = stokes_image.shape[:2]
    n_channels = stokes_image.shape[2] if stokes_image.ndim == 3 else 1
    
    print(f"  [DEBUG] 圖像尺寸: {w}x{h}, 通道數: {n_channels}")
    
    # 根據通道數判斷格式
    if n_channels == 4:
        # mono_polarized: 直接是 [S0, S1, S2, S3]
        S0 = stokes_image[:,:,0]
        S1 = stokes_image[:,:,1]
        S2 = stokes_image[:,:,2]
        S3 = stokes_image[:,:,3]
        print(f"  [INFO] mono_polarized 格式: [S0, S1, S2, S3]")
        
    elif n_channels == 1:
        # 單通道（非偏振）
        print("  [WARNING] 單通道輸入，無偏振信息")
        gray = stokes_image[:,:,0]
        return gray.astype(np.float32), gray.astype(np.float32)
        
    elif n_channels == 3:
        # RGB（非偏振）或 spectral 的前 3 個波長
        print("  [WARNING] 3 通道 RGB 輸入，轉換為灰階（無偏振信息）")
        S0 = 0.2126 * stokes_image[:,:,0] + 0.7152 * stokes_image[:,:,1] + 0.0722 * stokes_image[:,:,2]
        S1 = np.zeros_like(S0)
        S2 = np.zeros_like(S0)
        
    elif n_channels == 12:
        # rgb_polarized: [R_S0, R_S1, R_S2, R_S3, G_S0, ..., B_S0, ...]
        print(f"  [INFO] rgb_polarized 格式: 12 通道")
        # RGB 權重
        weights = [0.2126, 0.7152, 0.0722]
        S0 = np.zeros((h, w), dtype=np.float32)
        S1 = np.zeros((h, w), dtype=np.float32)
        S2 = np.zeros((h, w), dtype=np.float32)
        
        for c in range(3):
            S0 += weights[c] * stokes_image[:,:,c*4 + 0]
            S1 += weights[c] * stokes_image[:,:,c*4 + 1]
            S2 += weights[c] * stokes_image[:,:,c*4 + 2]
            
    elif n_channels >= 4:
        # 🔥 修復：處理所有 >= 4 通道的情況（包括 13 通道）
        # spectral_polarized: 多個波長，每個 4 通道
        n_wavelengths = n_channels // 4
        print(f"  [INFO] spectral_polarized 格式: {n_wavelengths} 波長 x 4 Stokes (總 {n_channels} 通道)")
        
        # 簡單平均所有波長
        S0 = np.zeros((h, w), dtype=np.float32)
        S1 = np.zeros((h, w), dtype=np.float32)
        S2 = np.zeros((h, w), dtype=np.float32)
        
        for wl in range(n_wavelengths):
            S0 += stokes_image[:,:,wl*4 + 0]
            S1 += stokes_image[:,:,wl*4 + 1]
            S2 += stokes_image[:,:,wl*4 + 2]
        
        S0 /= n_wavelengths
        S1 /= n_wavelengths
        S2 /= n_wavelengths
        
    else:
        print(f"  [ERROR] 未知的通道數: {n_channels}")
        gray = stokes_image[:,:,0] if stokes_image.ndim == 3 else stokes_image
        return gray.astype(np.float32), gray.astype(np.float32)
    
    # 打印 Stokes 統計
    print(f"  [INFO] S0 範圍: [{S0.min():.6f}, {S0.max():.6f}]")
    print(f"  [INFO] S1 範圍: [{S1.min():.6f}, {S1.max():.6f}]")
    print(f"  [INFO] S2 範圍: [{S2.min():.6f}, {S2.max():.6f}]")
    
    # 轉換角度為弧度
    theta_parallel = np.radians(polarizer_angle_parallel)
    theta_cross = np.radians(polarizer_angle_cross)
    
    # 計算透射強度（灰階）
    # I(θ) = 0.5 * (S0 + S1*cos(2θ) + S2*sin(2θ))
    I_parallel = 0.5 * (S0 + S1 * np.cos(2 * theta_parallel) + S2 * np.sin(2 * theta_parallel))
    I_cross = 0.5 * (S0 + S1 * np.cos(2 * theta_cross) + S2 * np.sin(2 * theta_cross))
    
    # 確保非負
    I_parallel = np.maximum(I_parallel, 0)
    I_cross = np.maximum(I_cross, 0)
    
    print(f"  [INFO] I_parallel (灰階) 範圍: [{I_parallel.min():.6f}, {I_parallel.max():.6f}]")
    print(f"  [INFO] I_cross (灰階) 範圍: [{I_cross.min():.6f}, {I_cross.max():.6f}]")
    
    # 計算偏振度 (Degree of Polarization)
    dop = np.sqrt(S1**2 + S2**2) / (S0 + 1e-10)
    dop_valid = dop[S0 > 0.01]  # 只看有效區域
    if len(dop_valid) > 0:
        print(f"  [INFO] 偏振度 (DoP) 範圍: [{dop_valid.min():.4f}, {dop_valid.max():.4f}], 均值: {dop_valid.mean():.4f}")
    
    # 計算偏振差異
    diff = np.abs(I_parallel - I_cross)
    diff_valid = diff[S0 > 0.01]
    if len(diff_valid) > 0:
        print(f"  [INFO] |I∥ - I⊥| 範圍: [{diff_valid.min():.6f}, {diff_valid.max():.6f}], 均值: {diff_valid.mean():.6f}")
    
    return I_parallel.astype(np.float32), I_cross.astype(np.float32)


def denoise_image(image: np.ndarray, use_optix: bool = True) -> np.ndarray:
    """
    🧠 對圖像進行降噪
    
    優先使用 OptiX AI 降噪器，如果不可用則使用 bilateral filter
    
    注意：只對最終的 I∥ 和 I⊥ 降噪，不要對 Stokes Vector 降噪（會破壞物理精度）
    """
    if image is None or image.size == 0:
        return image
    
    # 確保是 2D 灰階
    if image.ndim == 3:
        image = image[:,:,0] if image.shape[2] == 1 else np.mean(image, axis=2)
    
    # 嘗試使用 OptiX Denoiser
    if use_optix:
        try:
            # 轉換為 Mitsuba Bitmap 格式
            # OptiX denoiser 需要 RGB 輸入
            rgb_image = np.stack([image, image, image], axis=-1).astype(np.float32)
            bitmap = mi.Bitmap(rgb_image)
            
            # 應用 OptiX 降噪
            denoised_bitmap = bitmap.denoise()
            denoised = np.array(denoised_bitmap)
            
            # 轉回灰階
            result = 0.2126 * denoised[:,:,0] + 0.7152 * denoised[:,:,1] + 0.0722 * denoised[:,:,2]
            print(f"  [DENOISE] OptiX AI 降噪完成")
            return result.astype(np.float32)
            
        except Exception as e:
            print(f"  [DENOISE] OptiX 不可用 ({e})，使用 bilateral filter")
    
    # Fallback: 使用 OpenCV bilateral filter
    try:
        # 正規化到 0-1 範圍
        img_min, img_max = image.min(), image.max()
        if img_max > img_min:
            normalized = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(image, dtype=np.uint8)
        
        # Bilateral filter - 保留邊緣的降噪
        denoised = cv2.bilateralFilter(normalized, d=9, sigmaColor=75, sigmaSpace=75)
        
        # 轉回原始範圍
        result = denoised.astype(np.float32) / 255.0 * (img_max - img_min) + img_min
        print(f"  [DENOISE] Bilateral filter 降噪完成")
        return result
        
    except Exception as e:
        print(f"  [DENOISE] 降噪失敗: {e}")
        return image


# ============================================================
# 深度渲染
# ============================================================

def render_depth_map(obj_path: str, 
                     camera_position: Tuple[float, float, float],
                     camera_target: Tuple[float, float, float]) -> np.ndarray:
    """
    渲染深度圖
    
    使用單獨的場景配置渲染深度
    """
    render_cfg = CONFIG['render']
    cam_cfg = CONFIG['camera']
    
    # 解析 MTL
    mtl_path = obj_path.replace('.obj', '.mtl')
    mtl_materials = parse_mtl_file(mtl_path)
    
    # 分離 OBJ 文件
    glass_obj, non_glass_obj = split_obj_by_material(obj_path, mtl_materials)
    
    # 變換矩陣
    transform = mi.ScalarTransform4f.scale([0.001, 0.001, 0.001]) @ \
                mi.ScalarTransform4f.rotate([1, 0, 0], -90)
    
    # 構建深度渲染場景
    depth_scene_dict = {
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
            'fov': cam_cfg['fov'],
            'fov_axis': 'x',
            'to_world': mi.ScalarTransform4f.look_at(
                origin=camera_position,
                target=camera_target,
                up=[0, 1, 0]
            ),
            'film': {
                'type': 'hdrfilm',
                'width': render_cfg['width'],
                'height': render_cfg['height'],
                'pixel_format': 'rgb',
                'component_format': 'float32',
            },
            'sampler': {
                'type': 'ldsampler',
                'sample_count': 256,
            },
        },
    }
    
    # 添加非玻璃幾何
    if non_glass_obj and os.path.exists(non_glass_obj):
        depth_scene_dict['mesh_opaque'] = {
            'type': 'obj',
            'filename': non_glass_obj,
            'face_normals': False,
            'to_world': transform,
        }
    
    # 添加玻璃幾何
    if glass_obj and os.path.exists(glass_obj):
        # 🔥 深度圖專用：玻璃改為不透明 diffuse
        # 確保 Ground Truth 記錄在玻璃表面，而非穿透到背景
        # 這模擬 PIDS 論文中的 Proxy Object 方法
        depth_scene_dict['mesh_glass'] = {
            'type': 'obj',
            'filename': glass_obj,
            'face_normals': False,
            'to_world': transform,
            'bsdf': {
                'type': 'diffuse',  # 強制不透明，確保深度落在玻璃表面
                'reflectance': {
                    'type': 'spectrum',
                    'value': 0.5,
                },
            },
        }
    
    # 渲染
    scene = mi.load_dict(depth_scene_dict)
    image = mi.render(scene, spp=256)
    
    # 提取深度通道
    img_np = np.array(image)
    
    print(f"  [DEBUG] 深度圖輸出形狀: {img_np.shape}")
    
    # 深度通常在 aov 輸出的特定通道
    if img_np.ndim == 3:
        if img_np.shape[2] >= 2:
            depth = img_np[:,:,1]  # depth 通常在第二個通道
        else:
            depth = img_np[:,:,0]
    else:
        depth = img_np
    
    return depth.astype(np.float32)


# ============================================================
# 主渲染函數
# ============================================================

def render_scene(obj_path: str, output_dir: str, scene_name: str = None) -> Dict[str, str]:
    """
    渲染單一場景
    
    使用物理正確的偏振渲染
    
    Args:
        obj_path: OBJ 檔案路徑
        output_dir: 輸出目錄
        scene_name: 場景名稱（可選）
    
    Returns:
        輸出檔案路徑字典
    """
    if scene_name is None:
        scene_name = Path(obj_path).stem
    
    print(f"\n{'='*60}")
    print(f"渲染場景: {scene_name}")
    print(f"{'='*60}")
    
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    render_cfg = CONFIG['render']
    cam_cfg = CONFIG['camera']
    scene_cfg = CONFIG['scene']
    
    # 計算相機位置
    baseline_m = mm_to_meters(cam_cfg['baseline'])
    cam_y = mm_to_meters(cam_cfg['position_y'])
    cam_z = mm_to_meters(cam_cfg['position_z'])
    
    # 目標點（物體區域中心）
    target_z = mm_to_meters((scene_cfg['object_z_min'] + scene_cfg['object_z_max']) / 2)
    target_y = cam_y  # 平視
    
    # 左右相機位置
    left_cam_pos = (-baseline_m / 2, cam_y, cam_z)
    right_cam_pos = (baseline_m / 2, cam_y, cam_z)
    
    # 目標點 - 每個相機直視前方（光軸平行）
    left_target = (-baseline_m / 2, cam_y, target_z)
    right_target = (baseline_m / 2, cam_y, target_z)
    
    print(f"[INFO] 左相機位置: {left_cam_pos}")
    print(f"[INFO] 左相機目標: {left_target}")
    print(f"[INFO] 右相機位置: {right_cam_pos}")
    print(f"[INFO] 右相機目標: {right_target}")
    print(f"[INFO] 光軸: 平行（非匯聚）")
    
    outputs = {}
    
    # ============================================================
    # 1. 渲染左相機 (I∥) - 平行偏振
    # ============================================================
    print(f"\n[1/4] 渲染左相機 (I∥, 平行偏振)...")
    
    left_scene_dict = build_polarized_scene(
        obj_path=obj_path,
        camera_position=left_cam_pos,
        camera_target=left_target,  # 直視前方
        spp=render_cfg['spp'],
    )
    
    left_scene = load_and_setup_scene(left_scene_dict)
    left_stokes = render_stokes_vector(
        left_scene, 
        spp=render_cfg['spp'],
        spp_per_batch=render_cfg['spp_per_batch']
    )
    
    # 提取偏振分量（灰階）
    # 暫時恢復之前能工作的設定
    left_parallel, left_cross = extract_polarization_images(
        left_stokes,
        polarizer_angle_parallel=0.0,
        polarizer_angle_cross=90.0
    )
    
    # 🧠 降噪處理（可選，目前禁用以保留細節）
    # print(f"  [DENOISE] 對 I∥ 進行降噪...")
    # left_parallel = denoise_image(left_parallel)
    
    # 保存左相機 I∥（灰階 EXR + PNG）
    left_parallel_path = os.path.join(output_dir, f"{scene_name}_left_parallel.exr")
    save_grayscale_exr(left_parallel, left_parallel_path)
    left_parallel_png = os.path.join(output_dir, f"{scene_name}_left_parallel.png")
    save_grayscale_png(left_parallel, left_parallel_png)
    outputs['left_parallel'] = left_parallel_path
    outputs['left_parallel_png'] = left_parallel_png
    
    del left_scene, left_stokes
    gc.collect()
    
    # ============================================================
    # 2. 渲染右相機 (I⊥) - 交叉偏振
    # ============================================================
    print(f"\n[2/4] 渲染右相機 (I⊥, 交叉偏振)...")
    
    right_scene_dict = build_polarized_scene(
        obj_path=obj_path,
        camera_position=right_cam_pos,
        camera_target=right_target,  # 直視前方
        spp=render_cfg['spp'],
    )
    
    right_scene = load_and_setup_scene(right_scene_dict)
    right_stokes = render_stokes_vector(
        right_scene,
        spp=render_cfg['spp'],
        spp_per_batch=render_cfg['spp_per_batch']
    )
    
    # 提取偏振分量（灰階）
    right_parallel, right_cross = extract_polarization_images(
        right_stokes,
        polarizer_angle_parallel=0.0,
        polarizer_angle_cross=90.0
    )
    
    # 🧠 降噪處理（可選，目前禁用以保留細節）
    # print(f"  [DENOISE] 對 I⊥ 進行降噪...")
    # right_cross = denoise_image(right_cross)
    
    # 保存右相機 I⊥（灰階 EXR + PNG）
    right_cross_path = os.path.join(output_dir, f"{scene_name}_right_cross.exr")
    save_grayscale_exr(right_cross, right_cross_path)
    right_cross_png = os.path.join(output_dir, f"{scene_name}_right_cross.png")
    save_grayscale_png(right_cross, right_cross_png)
    outputs['right_cross'] = right_cross_path
    outputs['right_cross_png'] = right_cross_png
    
    del right_scene, right_stokes
    gc.collect()
    
    # ============================================================
    # 3. 渲染深度圖
    # ============================================================
    print(f"\n[3/4] 渲染深度圖...")
    
    try:
        depth = render_depth_map(obj_path, left_cam_pos, left_target)
        
        depth_path = os.path.join(output_dir, f"{scene_name}_depth.exr")
        save_grayscale_exr(depth, depth_path)
        outputs['depth'] = depth_path
    except Exception as e:
        print(f"    [WARNING] 深度渲染失敗: {e}")
        import traceback
        traceback.print_exc()
        depth = None
    
    # ============================================================
    # 4. 計算視差圖
    # ============================================================
    print(f"\n[4/4] 計算視差圖...")
    
    if depth is not None:
        # 計算焦距 (像素)
        fov_rad = np.radians(cam_cfg['fov'])
        focal_length_px = (render_cfg['width'] / 2) / np.tan(fov_rad / 2)
        
        # 計算視差: disparity = baseline * focal_length / depth
        disparity = np.zeros_like(depth)
        valid_mask = depth > 0
        disparity[valid_mask] = (baseline_m * focal_length_px) / depth[valid_mask]
        
        disparity_path = os.path.join(output_dir, f"{scene_name}_disparity.exr")
        save_grayscale_exr(disparity, disparity_path)
        outputs['disparity'] = disparity_path
    
    # ============================================================
    # 5. 保存預覽圖
    # ============================================================
    if CONFIG['output']['save_preview']:
        print(f"\n[Extra] 保存預覽圖...")
        
        from PIL import Image
        
        def tonemap(img, percentile_low=1, percentile_high=99):
            """色調映射"""
            img = np.clip(img, 0, None)
            valid = img[img > 0]
            if len(valid) > 0:
                low = np.percentile(valid, percentile_low)
                high = np.percentile(valid, percentile_high)
                if high > low:
                    img_norm = (img - low) / (high - low)
                    img_norm = np.clip(img_norm, 0, 1)
                else:
                    img_norm = img / (np.max(img) + 1e-6)
            else:
                img_norm = img / (np.max(img) + 1e-6)
            img_norm = np.power(img_norm, 1/2.2)
            return (img_norm * 255).astype(np.uint8)
        
        # I_parallel 預覽
        preview_parallel = tonemap(left_parallel)
        Image.fromarray(preview_parallel, mode='L').save(
            os.path.join(output_dir, f"{scene_name}_left_parallel.png")
        )
        
        # I_cross 預覽
        preview_cross = tonemap(right_cross)
        Image.fromarray(preview_cross, mode='L').save(
            os.path.join(output_dir, f"{scene_name}_right_cross.png")
        )
        
        # 偏振差異圖
        diff = np.abs(left_parallel - right_cross)
        diff_normalized = diff / (np.max(diff) + 1e-6)
        preview_diff = (diff_normalized * 255).astype(np.uint8)
        Image.fromarray(preview_diff, mode='L').save(
            os.path.join(output_dir, f"{scene_name}_polarization_diff.png")
        )
        
        # 深度預覽
        if depth is not None:
            import matplotlib.cm as cm
            depth_valid = depth[depth > 0]
            if len(depth_valid) > 0:
                depth_min, depth_max = depth_valid.min(), depth_valid.max()
                depth_norm = (depth_max - depth) / (depth_max - depth_min + 1e-6)
                depth_norm = np.clip(depth_norm, 0, 1)
                depth_norm[depth <= 0] = 0
                
                colormap = cm.get_cmap('turbo')
                depth_colored = colormap(depth_norm)[:, :, :3]
                depth_colored = (depth_colored * 255).astype(np.uint8)
                depth_colored[depth <= 0] = 0
                
                Image.fromarray(depth_colored).save(
                    os.path.join(output_dir, f"{scene_name}_depth.png")
                )
        
        print(f"    -> 預覽圖已保存")
    
    # ============================================================
    # 6. 保存參數
    # ============================================================
    params = {
        'scene_name': scene_name,
        'obj_file': os.path.basename(obj_path),
        'polarization': {
            'method': 'stokes_vector',
            'physically_correct': True,
            'left_polarizer_angle': 0.0,
            'right_polarizer_angle': 90.0,
        },
        'render_config': {
            'width': render_cfg['width'],
            'height': render_cfg['height'],
            'spp': render_cfg['spp'],
        },
        'camera_config': {
            'baseline_mm': cam_cfg['baseline'],
            'fov_deg': cam_cfg['fov'],
        },
    }
    
    if depth is not None:
        params['depth_stats'] = {
            'min_m': float(depth[depth > 0].min()) if np.any(depth > 0) else 0,
            'max_m': float(depth[depth > 0].max()) if np.any(depth > 0) else 0,
        }
    
    params_path = os.path.join(output_dir, f"{scene_name}_params.json")
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)
    outputs['params'] = params_path
    
    print(f"\n{'='*60}")
    print(f"場景 {scene_name} 渲染完成!")
    print(f"{'='*60}\n")
    
    return outputs


# ============================================================
# 命令行介面
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='PIDS Mitsuba 3 偏振渲染器 (物理正確版)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 單一場景
  python pids_polarized_renderer.py --scene_file scene.obj --output_dir ./output
  
  # 批次渲染
  python pids_polarized_renderer.py --input_dir ./scenes --output_dir ./output
  
  # 自訂 SPP
  python pids_polarized_renderer.py --scene_file scene.obj --output_dir ./output --spp 32768
        """
    )
    
    # 輸入選項
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--scene_file', type=str, help='單一 OBJ 場景檔案')
    input_group.add_argument('--input_dir', type=str, help='包含 OBJ 檔案的目錄')
    
    # 輸出選項
    parser.add_argument('--output_dir', type=str, required=True, help='輸出目錄')
    
    # 渲染選項
    parser.add_argument('--spp', type=int, default=CONFIG['render']['spp'],
                        help=f"每像素樣本數 (預設: {CONFIG['render']['spp']})")
    parser.add_argument('--spp_per_batch', type=int, default=CONFIG['render']['spp_per_batch'],
                        help=f"每批樣本數 (預設: {CONFIG['render']['spp_per_batch']})")
    parser.add_argument('--max_scenes', type=int, default=None,
                        help='最大渲染場景數')
    
    # 其他選項
    parser.add_argument('--no_preview', action='store_true', help='不保存預覽圖')
    
    args = parser.parse_args()
    
    # 更新配置
    CONFIG['render']['spp'] = args.spp
    CONFIG['render']['spp_per_batch'] = args.spp_per_batch
    CONFIG['output']['save_preview'] = not args.no_preview
    
    # 設定偏振 variant
    try:
        setup_polarized_variant()
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        print("\n[INFO] 偏振渲染需要 Mitsuba 3 的 polarized variant。")
        print("[INFO] 請確保 Mitsuba 3 編譯時包含 polarized 支援。")
        print("[INFO] 或者使用標準版渲染器 (pids_mitsuba_renderer.py)")
        return 1
    
    # 收集場景
    if args.scene_file:
        scenes = [args.scene_file]
    else:
        scenes = sorted(Path(args.input_dir).glob('*.obj'))
        if args.max_scenes:
            scenes = scenes[:args.max_scenes]
    
    print(f"\n[INFO] 找到 {len(scenes)} 個場景")
    
    # 渲染
    start_time = time.time()
    
    for i, scene_path in enumerate(scenes):
        scene_path = str(scene_path)
        print(f"\n[進度] {i+1}/{len(scenes)}")
        
        try:
            render_scene(scene_path, args.output_dir)
        except Exception as e:
            print(f"[ERROR] 渲染 {scene_path} 失敗: {e}")
            import traceback
            traceback.print_exc()
    
    elapsed = time.time() - start_time
    print(f"\n[完成] 總耗時: {elapsed/60:.1f} 分鐘")
    
    return 0


if __name__ == '__main__':
    exit(main())
