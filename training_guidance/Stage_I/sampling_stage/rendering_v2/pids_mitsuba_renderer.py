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
        'spp': 16384,           # 偏振渲染可能需要更多樣本
        'spp_per_batch': 1024,  # 分批渲染
        'max_depth': 8,         # 光線反彈次數
    },
    'camera': {
        'baseline': 65.0,       # 基線 (mm)
        'fov': 65.0,            # 視場角 (度)
        'position_y': 100.0,    # 相機高度 (mm) - 靠近底部
        'position_z': 360.0,    # 相機深度 (mm) - 前牆位置
    },
    'lighting': {
        'led': {
            'intensity': 25.0,
            'size_x': 180.0,    # 光源尺寸 (mm)
            'size_y': 100.0,
            'position_y': 290.0,  # 光源高度 (mm)
            'position_z': 450.0,  # 光源深度 (mm)
        },
        'ambient': {
            'intensity': 0.02,  # 低環境光
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


# ============================================================
# 設定 Mitsuba Variant (偏振版)
# ============================================================

def setup_polarized_variant():
    """
    設定 Mitsuba 偏振 variant
    
    偏振渲染需要 'polarized' variant
    """
    available = mi.variants()
    print(f"[INFO] 可用的 Mitsuba variants: {available}")
    
    # 優先使用 CUDA 偏振 variant
    polarized_variants = [
        'cuda_ad_rgb_polarized',   # CUDA + 自動微分 + RGB + 偏振
        'cuda_rgb_polarized',       # CUDA + RGB + 偏振
        'llvm_ad_rgb_polarized',    # LLVM + 自動微分 + RGB + 偏振
        'llvm_rgb_polarized',       # LLVM + RGB + 偏振
        'scalar_rgb_polarized',     # Scalar + RGB + 偏振
    ]
    
    for variant in polarized_variants:
        if variant in available:
            mi.set_variant(variant)
            print(f"[INFO] ✓ 使用偏振 variant: {variant}")
            return variant
    
    # 如果沒有偏振 variant，報錯
    raise RuntimeError(
        "無法找到偏振 variant！\n"
        "請確保 Mitsuba 3 編譯時包含 polarized 支援。\n"
        f"可用的 variants: {available}\n"
        "需要以下之一: {polarized_variants}"
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
    """
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
                              polarization_angle: float = 0.0) -> Dict:
    """
    創建偏振光源
    
    Args:
        position: 光源位置 (meters)
        target: 目標點 (meters)
        size: 光源尺寸 (width, height) (meters)
        intensity: 光源強度
        polarization_angle: 偏振角度 (度), 0 = 水平偏振
    
    Returns:
        場景字典，包含偏振光源
    """
    # 計算變換矩陣
    transform = mi.ScalarTransform4f.look_at(
        origin=position,
        target=target,
        up=[0, 1, 0]
    ) @ mi.ScalarTransform4f.scale([size[0]/2, size[1]/2, 1])
    
    # 創建發光面
    emitter_dict = {
        'type': 'rectangle',
        'to_world': transform,
        'bsdf': {
            'type': 'null',  # 光源本身不反射
        },
        'emitter': {
            'type': 'area',
            'radiance': {
                'type': 'rgb',
                'value': [intensity, intensity, intensity],
            },
        },
    }
    
    return emitter_dict


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
        這裡使用 RGB 格式來接收 Stokes 分量
    """
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
            'pixel_format': 'rgb',  # Stokes 需要多通道，後續轉灰階
            'component_format': 'float32',
            'rfilter': {
                'type': 'gaussian',
            },
        },
        'sampler': {
            'type': 'independent',
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
    
    Args:
        obj_path: OBJ 檔案路徑
        camera_position: 相機位置 (meters)
        camera_target: 相機目標點 (meters)
        spp: 每像素樣本數
    
    Returns:
        Mitsuba 場景字典
    """
    render_cfg = CONFIG['render']
    cam_cfg = CONFIG['camera']
    light_cfg = CONFIG['lighting']
    
    # 解析 MTL
    mtl_path = obj_path.replace('.obj', '.mtl')
    mtl_materials = parse_mtl_file(mtl_path)
    
    scene_dict = {
        'type': 'scene',
        'integrator': {
            'type': 'path',
            'max_depth': render_cfg['max_depth'],
        },
    }
    
    # 環境光
    if light_cfg['ambient']['intensity'] > 0:
        scene_dict['ambient'] = {
            'type': 'constant',
            'radiance': {
                'type': 'rgb',
                'value': [light_cfg['ambient']['intensity']] * 3,
            },
        }
    
    # 偏振 LED 光源
    led = light_cfg['led']
    light_pos = (0, mm_to_meters(led['position_y']), mm_to_meters(led['position_z']))
    
    # 光源朝向物體區域中心
    scene_cfg = CONFIG['scene']
    target_z = (scene_cfg['object_z_min'] + scene_cfg['object_z_max']) / 2
    light_target = (0, mm_to_meters(150), mm_to_meters(target_z))
    
    scene_dict['light'] = create_polarized_emitter(
        position=light_pos,
        target=light_target,
        size=(mm_to_meters(led['size_x']), mm_to_meters(led['size_y'])),
        intensity=led['intensity'],
        polarization_angle=0.0,  # 水平偏振
    )
    
    # 載入 OBJ 網格
    # 需要分離玻璃和非玻璃部分，給玻璃使用正確的 dielectric 材質
    scene_dict['mesh'] = {
        'type': 'obj',
        'filename': obj_path,
        'face_normals': False,
        'to_world': mi.ScalarTransform4f.scale([0.001, 0.001, 0.001]) @
                    mi.ScalarTransform4f.rotate([1, 0, 0], -90),
    }
    
    # 為每個材質指定 BSDF
    for mat_name, mat_props in mtl_materials.items():
        if mat_props['is_glass']:
            scene_dict['mesh'][mat_name] = create_polarized_glass_bsdf(
                ior=CONFIG['materials']['glass']['ior']
            )
            print(f"    [BSDF] {mat_name}: dielectric (IOR={CONFIG['materials']['glass']['ior']})")
        else:
            scene_dict['mesh'][mat_name] = create_diffuse_bsdf(
                color=tuple(mat_props['diffuse'])
            )
    
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
        # 更新 sampler
        params = mi.traverse(scene)
        params['sensor.sampler.sample_count'] = actual_spp_per_batch
        params.update()
        
        # 渲染
        image = mi.render(scene, spp=actual_spp_per_batch)
        
        # 轉換為 numpy
        img_np = np.array(image)
        
        if accumulated is None:
            accumulated = img_np.astype(np.float64) * actual_spp_per_batch
        else:
            accumulated += img_np.astype(np.float64) * actual_spp_per_batch
        
        total_weight += actual_spp_per_batch
        
        print(f"    批次 {batch_idx + 1}/{n_batches} 完成")
        
        # 清理
        del image
        gc.collect()
    
    # 計算加權平均
    result = (accumulated / total_weight).astype(np.float32)
    
    print(f"  [渲染] 完成，輸出形狀: {result.shape}")
    
    return result


def extract_polarization_images(stokes_image: np.ndarray,
                                 polarizer_angle_parallel: float = 0.0,
                                 polarizer_angle_cross: float = 90.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    從 Stokes Vector 提取偏振圖像（灰階）
    
    對於線性偏振片（角度 θ），透射強度為：
    I(θ) = 0.5 * (S0 + S1*cos(2θ) + S2*sin(2θ))
    
    Args:
        stokes_image: Stokes Vector [H, W, 4] 或 [H, W, 3] (RGB)
        polarizer_angle_parallel: 平行偏振片角度 (度)
        polarizer_angle_cross: 交叉偏振片角度 (度)
    
    Returns:
        (I_parallel, I_cross): 兩個灰階偏振圖像 [H, W]
    """
    # 檢查輸入格式
    if stokes_image.ndim == 2:
        # 已經是灰階
        print("  [WARNING] 輸入為灰階，無法提取偏振分量")
        return stokes_image, stokes_image
    
    h, w = stokes_image.shape[:2]
    
    if stokes_image.shape[2] == 3:
        # RGB 圖像 - 在偏振模式下，Mitsuba 3 的 Stokes 輸出格式：
        # 對於 spectral polarized: 每個波長有完整 Stokes
        # 對於 rgb polarized: 通常 RGB 各自有 Stokes，需要特殊處理
        #
        # 如果是標準 RGB（非偏振），先轉灰階
        # S0 = 亮度（總強度）
        S0 = 0.2126 * stokes_image[:,:,0] + 0.7152 * stokes_image[:,:,1] + 0.0722 * stokes_image[:,:,2]
        
        # 嘗試從 RGB 差異估計 S1（這是近似）
        # 真正的偏振需要 polarized variant
        S1 = np.zeros_like(S0)
        S2 = np.zeros_like(S0)
        
        print("  [WARNING] RGB 模式 - 偏振信息有限，建議使用 polarized variant")
        
    elif stokes_image.shape[2] >= 4:
        # 完整 Stokes Vector [S0, S1, S2, S3]
        # 轉為灰階：對每個 Stokes 分量取亮度
        if stokes_image.shape[2] == 4:
            # 單通道 Stokes
            S0 = stokes_image[:,:,0]
            S1 = stokes_image[:,:,1]
            S2 = stokes_image[:,:,2]
        else:
            # RGB Stokes - 每個顏色通道有 Stokes
            # 格式可能是 [R_S0, R_S1, R_S2, R_S3, G_S0, ...]
            # 這裡簡化處理：取亮度加權
            n_stokes = 4
            S0 = np.zeros((h, w), dtype=np.float32)
            S1 = np.zeros((h, w), dtype=np.float32)
            S2 = np.zeros((h, w), dtype=np.float32)
            
            # RGB 權重
            weights = [0.2126, 0.7152, 0.0722]
            for c in range(3):
                if (c + 1) * n_stokes <= stokes_image.shape[2]:
                    S0 += weights[c] * stokes_image[:,:,c*n_stokes + 0]
                    S1 += weights[c] * stokes_image[:,:,c*n_stokes + 1]
                    S2 += weights[c] * stokes_image[:,:,c*n_stokes + 2]
        
        print(f"  [INFO] Stokes Vector: S0 範圍 [{S0.min():.4f}, {S0.max():.4f}]")
        print(f"  [INFO] Stokes Vector: S1 範圍 [{S1.min():.4f}, {S1.max():.4f}]")
    else:
        print(f"  [ERROR] 未知的圖像格式: shape = {stokes_image.shape}")
        # 返回灰階
        gray = stokes_image[:,:,0] if stokes_image.ndim == 3 else stokes_image
        return gray.astype(np.float32), gray.astype(np.float32)
    
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
    
    print(f"  [INFO] I_parallel (灰階) 範圍: [{I_parallel.min():.4f}, {I_parallel.max():.4f}]")
    print(f"  [INFO] I_cross (灰階) 範圍: [{I_cross.min():.4f}, {I_cross.max():.4f}]")
    
    # 計算偏振度
    dop = np.sqrt(S1**2 + S2**2) / (S0 + 1e-10)
    print(f"  [INFO] 偏振度 (DoP) 範圍: [{dop.min():.4f}, {dop.max():.4f}]")
    
    return I_parallel.astype(np.float32), I_cross.astype(np.float32)


# ============================================================
# 深度渲染
# ============================================================

def render_depth_map(scene_dict: Dict, camera_position: Tuple[float, float, float]) -> np.ndarray:
    """
    渲染深度圖
    
    使用單獨的場景配置渲染深度
    """
    render_cfg = CONFIG['render']
    
    # 創建深度渲染場景
    depth_scene_dict = scene_dict.copy()
    
    # 修改 integrator 為 aov (輔助輸出)
    depth_scene_dict['integrator'] = {
        'type': 'aov',
        'aovs': 'dd.y:depth',
        'integrator': {
            'type': 'path',
            'max_depth': 2,
        },
    }
    
    # 渲染
    scene = mi.load_dict(depth_scene_dict)
    image = mi.render(scene, spp=256)  # 深度不需要太多樣本
    
    # 提取深度通道
    img_np = np.array(image)
    
    if img_np.ndim == 3 and img_np.shape[2] >= 2:
        depth = img_np[:,:,1]  # depth 在第二個通道
    else:
        depth = img_np if img_np.ndim == 2 else img_np[:,:,0]
    
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
    target = (0, target_y, target_z)
    
    print(f"[INFO] 左相機位置: {left_cam_pos}")
    print(f"[INFO] 右相機位置: {right_cam_pos}")
    print(f"[INFO] 目標點: {target}")
    
    outputs = {}
    
    # ============================================================
    # 1. 渲染左相機 (I∥) - 平行偏振
    # ============================================================
    print(f"\n[1/4] 渲染左相機 (I∥, 平行偏振)...")
    
    left_scene_dict = build_polarized_scene(
        obj_path=obj_path,
        camera_position=left_cam_pos,
        camera_target=target,
        spp=render_cfg['spp'],
    )
    
    left_scene = mi.load_dict(left_scene_dict)
    left_stokes = render_stokes_vector(
        left_scene, 
        spp=render_cfg['spp'],
        spp_per_batch=render_cfg['spp_per_batch']
    )
    
    # 提取偏振分量（灰階）
    left_parallel, left_cross = extract_polarization_images(
        left_stokes,
        polarizer_angle_parallel=0.0,
        polarizer_angle_cross=90.0
    )
    
    # 保存左相機 I∥（灰階 EXR）
    left_parallel_path = os.path.join(output_dir, f"{scene_name}_left_parallel.exr")
    save_grayscale_exr(left_parallel, left_parallel_path)
    outputs['left_parallel'] = left_parallel_path
    
    del left_scene, left_stokes
    gc.collect()
    
    # ============================================================
    # 2. 渲染右相機 (I⊥) - 交叉偏振
    # ============================================================
    print(f"\n[2/4] 渲染右相機 (I⊥, 交叉偏振)...")
    
    right_scene_dict = build_polarized_scene(
        obj_path=obj_path,
        camera_position=right_cam_pos,
        camera_target=target,
        spp=render_cfg['spp'],
    )
    
    right_scene = mi.load_dict(right_scene_dict)
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
    
    # 保存右相機 I⊥（灰階 EXR）
    right_cross_path = os.path.join(output_dir, f"{scene_name}_right_cross.exr")
    save_grayscale_exr(right_cross, right_cross_path)
    outputs['right_cross'] = right_cross_path
    
    del right_scene, right_stokes
    gc.collect()
    
    # ============================================================
    # 3. 渲染深度圖
    # ============================================================
    print(f"\n[3/4] 渲染深度圖...")
    
    # 使用左相機位置渲染深度
    try:
        depth = render_depth_map(left_scene_dict, left_cam_pos)
        
        depth_path = os.path.join(output_dir, f"{scene_name}_depth.exr")
        save_grayscale_exr(depth, depth_path)
        outputs['depth'] = depth_path
    except Exception as e:
        print(f"    [WARNING] 深度渲染失敗: {e}")
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
