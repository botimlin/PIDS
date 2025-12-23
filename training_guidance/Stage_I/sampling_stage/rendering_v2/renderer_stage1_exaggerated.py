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
        'spp': 16384,           # 🔥 高 SPP（RTX 5090 可輕鬆處理）
        'spp_per_batch': 2048,  # 🔥 增加 batch size
        'max_depth': 16,        # 🔥 支援多層玻璃重疊
    },
    'camera': {
        'baseline': 65.0,
        'fov': 65.0,
        'position_y': 100.0,
        'position_z': 360.0,
    },
    'lighting': {
        'lightbox': {
            'enabled': False,
        },
        'led': {
            'enabled': True,
            'intensity': 1500.0,  # 回滾
            'size_x': 180.0,
            'size_y': 100.0,
            'position_x': 0.0,
            'position_y': 290.0,
            'position_z': 450.0,
            'target_y': 150.0,
            'polarization_angle': 0.0,
        },
        'fill_light': {
            'enabled': True,
            'intensity': 8000.0,  # 回滾
        },
        'ambient': {
            'intensity': 0.0,
        },
    },
    'materials': {
        'glass': {
            'type': 'thindielectric',
            'ior': 1.5,  # 回滾到 1.5
            'roughness': 0.02,
        },
    },
    'scene': {
        'object_z_min': 528.0,
        'object_z_max': 695.0,
    },
    'output': {
        'save_preview': True,
        'save_stokes': False,
        'save_dolp_aolp': True,
        'auto_balance': True,  # 回滾：啟用自動平衡
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
    print(f"    -> 保存 PNG 預覽: {path} (範圍: [{img_min:.4f}, {img_max:.4f}])")


def save_grayscale_png_fixed_range(image: np.ndarray, path: str, vmin: float, vmax: float):
    """
    保存灰階 PNG 圖像（使用固定範圍，便於比較）
    """
    if image.ndim == 3:
        if image.shape[2] == 3:
            image = 0.2126 * image[:,:,0] + 0.7152 * image[:,:,1] + 0.0722 * image[:,:,2]
        else:
            image = image[:,:,0]
    
    # 使用固定範圍 normalize
    clipped = np.clip(image, vmin, vmax)
    if vmax > vmin:
        normalized = ((clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(image, dtype=np.uint8)
    
    cv2.imwrite(path, normalized)
    print(f"    -> 保存 PNG (固定範圍 [{vmin:.2f}, {vmax:.2f}]): {path}")


# ============================================================
# 設定 Mitsuba Variant (偏振版)
# ============================================================

def setup_polarized_variant():
    """
    設定 Mitsuba 偏振 variant
    
    偏振渲染需要 'polarized' variant
    🔥 必須使用 spectral_polarized（mono_polarized 會丟棄偏振信息！）
    """
    available = mi.variants()
    print(f"[INFO] 可用的 Mitsuba variants: {available}")
    
    # 🔥 優先使用 spectral_polarized（mono_polarized 在 luminance film 下會丟棄 S1/S2/S3！）
    polarized_variants = [
        'cuda_ad_spectral_polarized',   # 🔥 CUDA + 自動微分 + 光譜 + 偏振 ✓
        'cuda_spectral_polarized',      # CUDA + 光譜 + 偏振
        'llvm_ad_spectral_polarized',   # LLVM + 自動微分 + 光譜 + 偏振
        'llvm_spectral_polarized',      # LLVM + 光譜 + 偏振
        'scalar_spectral_polarized',    # Scalar + 光譜 + 偏振
        # mono_polarized 會在 luminance film 下丟棄偏振，不推薦
        # 'cuda_ad_mono_polarized',
        # 'cuda_mono_polarized',
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
    # 🔥 關鍵：不要設置 pixel_format，讓 Mitsuba 自動選擇
    # polarized variant 會自動輸出 Stokes vector
    # 設置 'luminance' 會丟失偏振信息！
    
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
            # 🔥 不設置 pixel_format，讓 Mitsuba 自動處理偏振輸出
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
                    print(f"      [MTL] {current_mat} → 玻璃 (名稱匹配)")
                else:
                    print(f"      [MTL] {current_mat} → 非玻璃")
            elif current_mat:
                if cmd == 'kd':
                    materials[current_mat]['diffuse'] = [float(x) for x in parts[1:4]]
                elif cmd == 'd':
                    alpha = float(parts[1])
                    if alpha < 0.95:
                        materials[current_mat]['is_glass'] = True
                        print(f"      [MTL] {current_mat} → 玻璃 (透明度 d={alpha})")
                elif cmd == 'illum':
                    illum = int(parts[1])
                    if illum in [4, 6, 7, 9]:
                        materials[current_mat]['is_glass'] = True
                        print(f"      [MTL] {current_mat} → 玻璃 (illum={illum})")
    
    return materials


def is_glass_material(mat_name: str) -> bool:
    """判斷材質是否為玻璃 - 嚴格版本"""
    name_lower = mat_name.lower()
    
    # 🔥 排除明確的非玻璃材質
    non_glass_keywords = ['background', 'wall', 'floor', 'ground', 'ceiling', 
                          'diffuse', 'opaque', 'solid', 'wood', 'metal', 'fabric']
    if any(k in name_lower for k in non_glass_keywords):
        return False
    
    # 🔥 只有明確包含 'glass' 才算玻璃
    # 移除了 'partition', 'panel', 'door', 'window' 等危險關鍵字
    glass_keywords = ['glass', 'transparent', 'clear_glass', 'acrylic']
    return any(k in name_lower for k in glass_keywords)


# ============================================================
# 場景構建
# ============================================================

def build_polarized_scene(obj_path: str,
                          camera_position: Tuple[float, float, float],
                          camera_target: Tuple[float, float, float],
                          spp: int,
                          camera_polarizer_angle: float = 0.0) -> Dict:
    """
    構建偏振渲染場景 - Stage 1 Exaggerated Version
    
    🔥 Stage 1 策略：
    1. 使用 roughdielectric 材質 - 透明 + 偏振反射
    2. 使用 stokes integrator 獲取完整 Stokes
    3. 在後處理時"放大"偏振差異
    4. max_depth 從 CONFIG 讀取（多層玻璃需要 16+）
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
            # 🔥 Stage 1: 使用 stokes + 限制 depth
            'type': 'stokes',
            'integrator': {
                'type': 'path',
                'max_depth': CONFIG['render']['max_depth'],  # 🔥 從 CONFIG 讀取
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
    
    # 偏振 LED 光源
    led = light_cfg['led']
    light_x = mm_to_meters(led.get('position_x', 0))
    light_y = mm_to_meters(led['position_y'])
    light_z = mm_to_meters(led['position_z'])
    light_pos = (light_x, light_y, light_z)
    
    scene_cfg = CONFIG['scene']
    target_z = (scene_cfg['object_z_min'] + scene_cfg['object_z_max']) / 2
    light_target_y = mm_to_meters(led.get('target_y', 100))
    light_target_z = mm_to_meters(target_z)
    light_target = (light_x, light_target_y, light_target_z)
    
    dy = light_y - light_target_y
    dz = light_target_z - light_z
    incident_angle = math.degrees(math.atan2(dy, dz))
    
    print(f"    [STAGE 1] 入射角: {incident_angle:.1f}° (Brewster角 ≈ 56°)")
    print(f"    [STAGE 1] 材質: roughdielectric（透明 + 偏振）")
    print(f"    [STAGE 1] max_depth: {CONFIG['render']['max_depth']}（多層玻璃需要較高值）")
    
    # 🔥 檢查是否啟用燈箱模式
    lightbox_cfg = light_cfg.get('lightbox', {})
    use_lightbox = lightbox_cfg.get('enabled', False)
    
    if use_lightbox:
        # ============================================================
        # 🔥 燈箱模式：多方向光源（論文檢測環境）
        # ============================================================
        print(f"    [LIGHTBOX] 啟用燈箱模式")
        lb_intensity = lightbox_cfg.get('intensity', 1000.0)
        
        # 計算場景中心
        scene_cfg = CONFIG['scene']
        target_z = (scene_cfg['object_z_min'] + scene_cfg['object_z_max']) / 2
        center = (0, mm_to_meters(100), mm_to_meters(target_z))
        
        # 定義燈箱光源位置（環繞物體）
        lightbox_lights = [
            # 頂部主光源
            {'name': 'top', 'pos': (0, 350, target_z), 'size': (200, 200)},
            # 左上角
            {'name': 'left_top', 'pos': (-250, 300, target_z - 100), 'size': (150, 150)},
            # 右上角
            {'name': 'right_top', 'pos': (250, 300, target_z - 100), 'size': (150, 150)},
            # 前方（靠近相機）
            {'name': 'front', 'pos': (0, 200, target_z - 200), 'size': (180, 100)},
            # 左側
            {'name': 'left', 'pos': (-300, 200, target_z), 'size': (100, 150)},
            # 右側
            {'name': 'right', 'pos': (300, 200, target_z), 'size': (100, 150)},
        ]
        
        for i, light in enumerate(lightbox_lights):
            pos = (mm_to_meters(light['pos'][0]), 
                   mm_to_meters(light['pos'][1]), 
                   mm_to_meters(light['pos'][2]))
            
            light_transform = mi.ScalarTransform4f.look_at(
                origin=pos,
                target=center,
                up=[0, 1, 0]
            ) @ mi.ScalarTransform4f.scale([mm_to_meters(light['size'][0])/2, 
                                            mm_to_meters(light['size'][1])/2, 1])
            
            scene_dict[f'lightbox_{light["name"]}'] = {
                'type': 'rectangle',
                'to_world': light_transform,
                'emitter': {
                    'type': 'area',
                    'radiance': {
                        'type': 'spectrum',
                        'value': lb_intensity,
                    },
                },
            }
            print(f"    [LIGHTBOX] 光源 {light['name']}: {pos}")
    else:
        # ============================================================
        # 🔥 被動偏振模式：只用非偏振環境光
        # 依靠玻璃的 Fresnel 反射產生偏振差異
        # ============================================================
        
        # 檢查是否要添加偏振光源
        if led.get('enabled', True) and led.get('intensity', 0) > 0:
            print(f"    [LED] 偏振光源強度: {led['intensity']}")
            
            light_transform = mi.ScalarTransform4f.look_at(
                origin=light_pos,
                target=light_target,
                up=[0, 1, 0]
            ) @ mi.ScalarTransform4f.scale([mm_to_meters(led['size_x'])/2, mm_to_meters(led['size_y'])/2, 1])

            scene_dict['light_emitter'] = {
                'type': 'rectangle',
                'to_world': light_transform,
                'emitter': {
                    'type': 'area',
                    'radiance': {
                        'type': 'spectrum',
                        'value': led['intensity'],
                    },
                },
            }
        else:
            print(f"    [LED] ❌ 偏振光源已關閉")
        
        # ============================================================
        # 🔥 恆定環境光（Constant Emitter）
        # ============================================================
        fill_cfg = light_cfg.get('fill_light', {})
        if fill_cfg.get('enabled', False):
            fill_intensity = fill_cfg.get('intensity', 8000.0)
            
            # constant emitter 的強度單位不同，需要縮放
            scene_dict['constant_emitter'] = {
                'type': 'constant',
                'radiance': {
                    'type': 'spectrum',
                    'value': fill_intensity / 1000.0,
                },
            }
            print(f"    [FILL LIGHT] 恆定環境光強度: {fill_intensity / 1000.0}")
    # 🔥 不再使用 light_polarizer - 會導致採樣問題和噪點
    
    # 變換矩陣
    transform = mi.ScalarTransform4f.scale([0.001, 0.001, 0.001]) @ \
                mi.ScalarTransform4f.rotate([1, 0, 0], -90)
    
    # 🔥 非玻璃幾何 - 必須指定純 diffuse BSDF 來正確 depolarize
    if non_glass_obj and os.path.exists(non_glass_obj):
        scene_dict['mesh_opaque'] = {
            'type': 'obj',
            'filename': non_glass_obj,
            'face_normals': False,
            'to_world': transform,
            # 🔥 強制使用純漫反射材質，忽略 OBJ 自帶的 MTL
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'spectrum',
                    'value': 0.5,
                },
            },
        }
        print(f"    [MESH] 非玻璃幾何: {non_glass_obj}")
        print(f"    [MESH] 🔥 材質: 純 diffuse (強制覆蓋 MTL)")
    
    # 🔥 Stage-1：玻璃材質（從配置讀取）
    mat_cfg = CONFIG['materials']['glass']
    glass_type = mat_cfg.get('type', 'dielectric')
    glass_ior = mat_cfg.get('ior', 1.5)
    
    if glass_obj and os.path.exists(glass_obj):
        if glass_type == 'thindielectric':
            # 🔥 薄玻璃 - 直接使用 thindielectric（它本身就處理雙面）
            scene_dict['mesh_glass'] = {
                'type': 'obj',
                'filename': glass_obj,
                'face_normals': True,
                'to_world': transform,
                'bsdf': {
                    'type': 'thindielectric',
                    'int_ior': glass_ior,
                    'ext_ior': 1.0,
                },
            }
            print(f"    [MESH] 玻璃幾何: {glass_obj}")
            print(f"    [MESH] 材質: thindielectric (IOR={glass_ior})")
        elif glass_type == 'dielectric':
            # 🔥 標準介電質玻璃 - 物理正確的 Fresnel 反射
            scene_dict['mesh_glass'] = {
                'type': 'obj',
                'filename': glass_obj,
                'face_normals': False,
                'to_world': transform,
                'bsdf': {
                    'type': 'dielectric',
                    'int_ior': glass_ior,
                    'ext_ior': 1.0,
                },
            }
            print(f"    [MESH] 玻璃幾何: {glass_obj}")
            print(f"    [MESH] 🔥 材質: dielectric (IOR={glass_ior}) - Fresnel 偏振反射")
        else:
            # roughdielectric：粗糙玻璃
            roughness = mat_cfg.get('roughness', 0.02)
            scene_dict['mesh_glass'] = {
                'type': 'obj',
                'filename': glass_obj,
                'face_normals': False,
                'to_world': transform,
                'bsdf': {
                    'type': 'roughdielectric',
                    'alpha': roughness,
                    'int_ior': glass_ior,
                    'ext_ior': 1.0,
                },
            }
            print(f"    [MESH] 玻璃幾何: {glass_obj}")
            print(f"    [MESH] 材質: roughdielectric (IOR={glass_ior}, roughness={roughness})")
    
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


def render_direct(scene: "mi.Scene", spp: int, spp_per_batch: int = 1024) -> np.ndarray:
    """
    🔥 Stage 1 專用：直接渲染（不使用 Stokes 提取）
    
    偏振選擇已經在相機前的 polarizer 完成，這裡只需要渲染並返回灰階圖像。
    
    Args:
        scene: Mitsuba 場景（已包含相機前 polarizer）
        spp: 總樣本數
        spp_per_batch: 每批樣本數
    
    Returns:
        灰階圖像 [H, W]
    """
    # 分批渲染
    n_batches = max(1, spp // spp_per_batch)
    actual_spp_per_batch = spp // n_batches
    
    print(f"  [渲染] Stage 1 直接渲染: {spp} SPP, {n_batches} 批次")
    
    accumulated = None
    total_weight = 0
    
    for batch_idx in range(n_batches):
        image = mi.render(scene, spp=actual_spp_per_batch)
        img_np = np.array(image)
        
        if accumulated is None:
            accumulated = img_np.astype(np.float64) * actual_spp_per_batch
        else:
            accumulated += img_np.astype(np.float64) * actual_spp_per_batch
        
        total_weight += actual_spp_per_batch
        
        print(f"    批次 {batch_idx + 1}/{n_batches} 完成")
        
        del image
        dr.flush_malloc_cache()
        gc.collect()
    
    result = (accumulated / total_weight).astype(np.float32)
    result = np.maximum(result, 0)
    
    # 轉為灰階
    if result.ndim == 3:
        if result.shape[2] == 1:
            result = result[:, :, 0]
        elif result.shape[2] == 3:
            result = 0.2126 * result[:,:,0] + 0.7152 * result[:,:,1] + 0.0722 * result[:,:,2]
        elif result.shape[2] == 4:
            # 可能是 Stokes，取 S0
            result = result[:, :, 0]
        else:
            result = np.mean(result, axis=2)
    
    print(f"  [渲染] 完成，輸出形狀: {result.shape}")
    print(f"  [渲染] 數值範圍: [{result.min():.4f}, {result.max():.4f}]")
    
    return result.astype(np.float32)
    
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
        
        print(f"  [DEBUG] S0 範圍: [{S0.min():.4f}, {S0.max():.4f}]")
        print(f"  [DEBUG] S1 範圍: [{S1.min():.4f}, {S1.max():.4f}]")
        print(f"  [DEBUG] S2 範圍: [{S2.min():.4f}, {S2.max():.4f}]")
        
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


def extract_polarization_images_stage1(stokes_image: np.ndarray,
                                        polarizer_angle_parallel: float = 90.0,
                                        polarizer_angle_cross: float = 0.0,
                                        amplify_factor: float = 10.0,
                                        exposure: float = 1.0,
                                        active_polarization: bool = True,
                                        source_polarization_angle: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    🔥 Stage 1 專用：支持主動偏振模式
    
    主動偏振模式 (active_polarization=True)：
    - 假設光源已經是偏振的（角度 = source_polarization_angle）
    - 直接計算通過不同相機偏振片的強度
    - 不需要放大，因為偏振信號本來就很強
    
    被動偏振模式 (active_polarization=False)：
    - 光源是非偏振的
    - 依靠 Fresnel 反射產生偏振
    - 需要放大差異
    """
    print(f"  [STAGE 1] 偏振模式: {'主動' if active_polarization else '被動'}")
    
    # 檢查輸入格式
    if stokes_image.ndim == 2:
        print("  [WARNING] 輸入為 2D 灰階，無偏振信息")
        return stokes_image.astype(np.float32), stokes_image.astype(np.float32)
    
    h, w = stokes_image.shape[:2]
    n_channels = stokes_image.shape[2] if stokes_image.ndim == 3 else 1
    
    # 提取 Stokes 分量
    print(f"  [DEBUG] 輸入通道數: {n_channels}")
    
    # 🔥 診斷：打印每個通道的範圍
    print(f"  [DEBUG] 各通道範圍診斷:")
    for ch in range(min(n_channels, 13)):
        ch_data = stokes_image[:,:,ch]
        print(f"    通道 {ch}: [{ch_data.min():.4f}, {ch_data.max():.4f}]")
    
    if n_channels == 4:
        # mono_polarized: 直接是 [S0, S1, S2, S3]
        S0 = stokes_image[:,:,0]
        S1 = stokes_image[:,:,1]
        S2 = stokes_image[:,:,2]
        print(f"  [DEBUG] 4 通道 mono_polarized 格式")
    elif n_channels == 1:
        print("  [WARNING] 單通道輸入，無偏振信息")
        gray = stokes_image[:,:,0]
        return gray.astype(np.float32), gray.astype(np.float32)
    elif n_channels == 3:
        print("  [WARNING] 3 通道 RGB 輸入，無偏振信息")
        S0 = 0.2126 * stokes_image[:,:,0] + 0.7152 * stokes_image[:,:,1] + 0.0722 * stokes_image[:,:,2]
        S1 = np.zeros_like(S0)
        S2 = np.zeros_like(S0)
    elif n_channels == 12:
        # rgb_polarized: [R_S0, R_S1, R_S2, R_S3, G_S0, ..., B_S0, ...]
        print(f"  [DEBUG] 12 通道 rgb_polarized 格式")
        weights = [0.2126, 0.7152, 0.0722]
        S0 = np.zeros((h, w), dtype=np.float32)
        S1 = np.zeros((h, w), dtype=np.float32)
        S2 = np.zeros((h, w), dtype=np.float32)
        for c in range(3):
            S0 += weights[c] * stokes_image[:,:,c*4 + 0]
            S1 += weights[c] * stokes_image[:,:,c*4 + 1]
            S2 += weights[c] * stokes_image[:,:,c*4 + 2]
    elif n_channels == 13:
        # 🔥 修正：13 通道 = 4 波長的 S0 + 4 波長的 S1 + 4 波長的 S2 + 1 額外
        # 或者：3 波長 × 4 Stokes + 1 額外
        # 根據觀察：通道 0-3 相同 (S0), 通道 4-7 不同 (可能是 S1)
        print(f"  [DEBUG] 13 通道 spectral_polarized 格式")
        print(f"  [DEBUG] 嘗試格式: [4×S0, 4×S1, 4×S2, 1×S3]")
        
        # S0 = 平均前 4 通道（或第一個）
        S0 = stokes_image[:,:,0].astype(np.float32)
        
        # S1 = 通道 4（或平均 4-7）
        if n_channels > 4:
            S1 = stokes_image[:,:,4].astype(np.float32)
        else:
            S1 = np.zeros_like(S0)
            
        # S2 = 通道 8（或平均 8-11）
        if n_channels > 8:
            S2 = stokes_image[:,:,8].astype(np.float32)
        else:
            S2 = np.zeros_like(S0)
        
        print(f"  [DEBUG] 重新解析後:")
        print(f"  [DEBUG] S0 (通道0) 範圍: [{S0.min():.4f}, {S0.max():.4f}]")
        print(f"  [DEBUG] S1 (通道4) 範圍: [{S1.min():.4f}, {S1.max():.4f}]")
        print(f"  [DEBUG] S2 (通道8) 範圍: [{S2.min():.4f}, {S2.max():.4f}]")
    elif n_channels >= 4:
        # 其他多通道：使用前 4 個作為 Stokes
        print(f"  [DEBUG] {n_channels} 通道，使用前 4 通道作為 Stokes [S0, S1, S2, S3]")
        S0 = stokes_image[:,:,0]
        S1 = stokes_image[:,:,1]
        S2 = stokes_image[:,:,2]
        
        print(f"  [DEBUG] S0 範圍: [{S0.min():.4f}, {S0.max():.4f}]")
        print(f"  [DEBUG] S1 範圍: [{S1.min():.4f}, {S1.max():.4f}]")
        print(f"  [DEBUG] S2 範圍: [{S2.min():.4f}, {S2.max():.4f}]")
    else:
        print(f"  [WARNING] 未知通道數 {n_channels}，無法提取偏振")
        gray = stokes_image[:,:,0] if stokes_image.ndim == 3 else stokes_image
        return gray.astype(np.float32), gray.astype(np.float32)
    
    # 計算偏振度 DoP（僅用於診斷）
    polarized_intensity = np.sqrt(S1**2 + S2**2)
    DoP = polarized_intensity / (S0 + 1e-10)
    DoP = np.clip(DoP, 0, 1)
    
    dop_valid = DoP[S0 > 0.01]
    if len(dop_valid) > 0:
        print(f"  [STAGE 1] DoP 範圍: [{dop_valid.min():.4f}, {dop_valid.max():.4f}], 均值: {dop_valid.mean():.4f}")
    
    # 根據偏振片角度計算透射強度
    theta_parallel = np.radians(polarizer_angle_parallel)
    theta_cross = np.radians(polarizer_angle_cross)
    
    if active_polarization:
        # ============================================================
        # 🔥 主動偏振模式：模擬偏振光源
        # ============================================================
        # 論文設置：光源通過 0° 偏振片（水平偏振）
        # 相機前有 ±45° 或 0°/90° 偏振片
        
        theta_source = np.radians(source_polarization_angle)
        
        # 主動偏振光通過場景後的 Malus 定律
        # I(θ) = I_0 * cos²(θ - θ_source)
        # 但場景會影響偏振狀態，所以我們結合 Fresnel 響應
        
        # 方法：使用場景的 DoP 來調制偏振對比度
        # 偏振區域（玻璃反射）會有高 DoP，非偏振區域低 DoP
        
        # 計算相機偏振片與光源偏振角度的夾角效應
        cos2_parallel = np.cos(theta_parallel - theta_source) ** 2
        cos2_cross = np.cos(theta_cross - theta_source) ** 2
        
        # 主動偏振：基礎強度 + DoP 調制的偏振對比度
        # 高 DoP 區域會有更強的偏振響應
        polarization_contrast = DoP  # DoP 代表該區域的偏振強度
        
        # I_parallel: 偏振片與光源一致時最亮
        # I_cross: 偏振片與光源垂直時最暗
        I_parallel = S0 * (0.5 + 0.5 * polarization_contrast * (2 * cos2_parallel - 1))
        I_cross = S0 * (0.5 + 0.5 * polarization_contrast * (2 * cos2_cross - 1))
        
        print(f"  [主動偏振] 光源角度: {source_polarization_angle}°")
        print(f"  [主動偏振] cos²(θ_parallel - θ_source) = {cos2_parallel:.4f}")
        print(f"  [主動偏振] cos²(θ_cross - θ_source) = {cos2_cross:.4f}")
        
        # 主動偏振不需要額外放大
        I_parallel_amp = I_parallel
        I_cross_amp = I_cross
        
    else:
        # ============================================================
        # 被動偏振模式：依靠 Fresnel 反射
        # ============================================================
        # 🔥 標準 Stokes 公式：I(θ) = 0.5 * (S0 + S1*cos(2θ) + S2*sin(2θ))
        I_parallel = 0.5 * (S0 + S1 * np.cos(2 * theta_parallel) + S2 * np.sin(2 * theta_parallel))
        I_cross = 0.5 * (S0 + S1 * np.cos(2 * theta_cross) + S2 * np.sin(2 * theta_cross))
        
        # 🔥 安全的放大方式：放大「差異」而非直接放大 S1
        I_mean = 0.5 * (I_parallel + I_cross)
        I_diff = I_parallel - I_cross
        
        # 放大差異，但限制最大變化量
        I_diff_amplified = I_diff * amplify_factor
        max_diff = I_mean * 0.95
        I_diff_clamped = np.clip(I_diff_amplified, -max_diff, max_diff)
        
        # 重建放大後的圖像
        I_parallel_amp = I_mean + 0.5 * I_diff_clamped
        I_cross_amp = I_mean - 0.5 * I_diff_clamped
        
        print(f"  [被動偏振] 放大因子: {amplify_factor}x")
    
    # 確保非負
    I_parallel_amp = np.maximum(I_parallel_amp, 0)
    I_cross_amp = np.maximum(I_cross_amp, 0)
    
    # 計算差異
    diff = np.abs(I_parallel_amp - I_cross_amp)
    diff_valid = diff[S0 > 0.01]
    if len(diff_valid) > 0:
        print(f"  [STAGE 1] |I∥ - I⊥| 均值: {diff_valid.mean():.4f}")
    
    # 🔥 曝光補償
    I_parallel_final = I_parallel_amp * exposure
    I_cross_final = I_cross_amp * exposure
    
    print(f"  [STAGE 1] I_parallel 範圍: [{I_parallel_final.min():.4f}, {I_parallel_final.max():.4f}] (曝光後)")
    print(f"  [STAGE 1] I_cross 範圍: [{I_cross_final.min():.4f}, {I_cross_final.max():.4f}] (曝光後)")
    
    return I_parallel_final.astype(np.float32), I_cross_final.astype(np.float32)


def compute_dolp_aolp(stokes_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    🔥 計算 DoLP (偏振度) 和 AoLP (偏振角)
    
    這是論文中用於可視化偏振特徵的標準輸出。
    
    DoLP (Degree of Linear Polarization):
        DoLP = sqrt(S1² + S2²) / S0
        範圍: [0, 1]
        物理意義: 光線的偏振程度
        - 0: 完全非偏振
        - 1: 完全偏振
        
    AoLP (Angle of Linear Polarization):
        AoLP = 0.5 * atan2(S2, S1)
        範圍: [-π/2, π/2] 或 [0, π]
        物理意義: 偏振方向（法線方向的投影）
        
    Returns:
        S0: 總強度圖
        DoLP: 偏振度圖 [0, 1]
        AoLP: 偏振角圖 [0, π] (弧度) 或 [0, 180] (度)
    """
    h, w = stokes_image.shape[:2]
    n_channels = stokes_image.shape[2] if stokes_image.ndim == 3 else 1
    
    # 提取 Stokes 分量（使用與 extract_polarization_images_stage1 相同的邏輯）
    if n_channels == 4:
        S0 = stokes_image[:,:,0].astype(np.float32)
        S1 = stokes_image[:,:,1].astype(np.float32)
        S2 = stokes_image[:,:,2].astype(np.float32)
    elif n_channels == 13:
        # spectral_polarized: 通道 0 = S0, 通道 4 = S1, 通道 8 = S2
        S0 = stokes_image[:,:,0].astype(np.float32)
        S1 = stokes_image[:,:,4].astype(np.float32)
        S2 = stokes_image[:,:,8].astype(np.float32)
    elif n_channels >= 4:
        # 嘗試前 4 通道
        S0 = stokes_image[:,:,0].astype(np.float32)
        S1 = stokes_image[:,:,4].astype(np.float32) if n_channels > 4 else stokes_image[:,:,1].astype(np.float32)
        S2 = stokes_image[:,:,8].astype(np.float32) if n_channels > 8 else stokes_image[:,:,2].astype(np.float32)
    else:
        # 無偏振信息
        S0 = stokes_image[:,:,0].astype(np.float32) if stokes_image.ndim == 3 else stokes_image.astype(np.float32)
        return S0, np.zeros_like(S0), np.zeros_like(S0)
    
    # 計算 DoLP
    polarized_intensity = np.sqrt(S1**2 + S2**2)
    DoLP = polarized_intensity / (S0 + 1e-10)
    DoLP = np.clip(DoLP, 0, 1)
    
    # 計算 AoLP（弧度）
    # atan2(S2, S1) 給出 [-π, π]，除以 2 得到 [-π/2, π/2]
    # 加上 π/2 並模 π 得到 [0, π]
    AoLP_rad = 0.5 * np.arctan2(S2, S1)
    AoLP_rad = (AoLP_rad + np.pi/2) % np.pi  # 轉換到 [0, π]
    
    # 轉換為度數以便可視化
    AoLP_deg = np.degrees(AoLP_rad)  # [0, 180]
    
    print(f"  [DoLP/AoLP] S0 範圍: [{S0.min():.4f}, {S0.max():.4f}]")
    print(f"  [DoLP/AoLP] DoLP 範圍: [{DoLP.min():.4f}, {DoLP.max():.4f}], 均值: {DoLP.mean():.4f}")
    print(f"  [DoLP/AoLP] AoLP 範圍: [{AoLP_deg.min():.1f}°, {AoLP_deg.max():.1f}°]")
    
    return S0, DoLP, AoLP_deg


def balance_background_intensity(I_parallel: np.ndarray, I_cross: np.ndarray,
                                  dolp_threshold: float = 0.15,
                                  verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    平衡背景區域的強度（使用比值乘法）
    """
    # 計算 DoLP
    I_sum = I_parallel + I_cross
    valid_mask = I_sum > 1e-6
    
    dolp = np.zeros_like(I_parallel)
    dolp[valid_mask] = np.abs(I_parallel - I_cross)[valid_mask] / I_sum[valid_mask]
    dolp = np.clip(dolp, 0, 1)
    
    # 背景 = 低偏振區域
    background_mask = (dolp < dolp_threshold) & valid_mask
    glass_mask = (dolp >= dolp_threshold) & valid_mask
    
    if verbose:
        bg_count = np.sum(background_mask)
        total = np.sum(valid_mask)
        print(f"    背景像素: {bg_count} ({bg_count/total*100:.1f}%)")
        print(f"    玻璃像素: {np.sum(glass_mask)} ({np.sum(glass_mask)/total*100:.1f}%)")
    
    if not np.any(background_mask):
        if verbose:
            print("    ⚠️ 沒有找到背景區域")
        return I_parallel, I_cross
    
    # 計算背景區域的比值
    bg_parallel = np.mean(I_parallel[background_mask])
    bg_cross = np.mean(I_cross[background_mask])
    bg_ratio = bg_parallel / (bg_cross + 1e-10)
    
    if verbose:
        print(f"    背景 I∥ 平均: {bg_parallel:.4f}")
        print(f"    背景 I⊥ 平均: {bg_cross:.4f}")
        print(f"    背景 I∥/I⊥ 比值: {bg_ratio:.3f}")
    
    # 判斷是否需要平衡
    if 0.8 <= bg_ratio <= 1.25:
        if verbose:
            print(f"    ✅ 背景已平衡，無需調整")
        return I_parallel, I_cross
    
    if bg_ratio > 1.0:
        # I∥ 比 I⊥ 亮，需要提升 I⊥
        if verbose:
            print(f"    🔧 平衡中：提升 I⊥ 強度...")
        
        # 對整張圖乘以 ratio，在玻璃區域用混合來保留原始對比
        balanced_cross = I_cross * bg_ratio
        
        # 玻璃區域：混合原始值和調整值
        blend_weight = np.clip((dolp - dolp_threshold) / 0.3, 0, 1)
        balanced_cross = (1 - blend_weight) * balanced_cross + blend_weight * I_cross
        
        balanced_parallel = I_parallel.copy()
    else:
        # I⊥ 比 I∥ 亮
        if verbose:
            print(f"    🔧 平衡中：提升 I∥ 強度...")
        
        inverse_ratio = 1.0 / bg_ratio
        balanced_parallel = I_parallel * inverse_ratio
        
        blend_weight = np.clip((dolp - dolp_threshold) / 0.3, 0, 1)
        balanced_parallel = (1 - blend_weight) * balanced_parallel + blend_weight * I_parallel
        
        balanced_cross = I_cross.copy()
    
    if verbose:
        new_bg_parallel = np.mean(balanced_parallel[background_mask])
        new_bg_cross = np.mean(balanced_cross[background_mask])
        new_ratio = new_bg_parallel / (new_bg_cross + 1e-10)
        print(f"    平衡後背景 I∥/I⊥: {new_ratio:.3f}")
    
    return balanced_parallel, balanced_cross


def save_dolp_aolp_visualization(S0: np.ndarray, DoLP: np.ndarray, AoLP: np.ndarray,
                                  output_dir: str, scene_name: str):
    """
    🎨 保存 DoLP 和 AoLP 的可視化圖像（使用 OpenCV，不需要 matplotlib）
    
    DoLP: 使用灰階或熱力圖
    AoLP: 使用 HSV 色輪（角度 → 色相）
    """
    # 1. 保存 DoLP（灰階）
    dolp_path = os.path.join(output_dir, f"{scene_name}_DoLP.png")
    dolp_uint8 = (DoLP * 255).astype(np.uint8)
    cv2.imwrite(dolp_path, dolp_uint8)
    print(f"    -> 保存 DoLP: {dolp_path}")
    
    # 2. 保存 DoLP（熱力圖 - 使用 OpenCV colormap）
    dolp_color_path = os.path.join(output_dir, f"{scene_name}_DoLP_color.png")
    dolp_colored = cv2.applyColorMap(dolp_uint8, cv2.COLORMAP_JET)
    cv2.imwrite(dolp_color_path, dolp_colored)
    print(f"    -> 保存 DoLP (彩色): {dolp_color_path}")
    
    # 3. 保存 AoLP（HSV 色輪）
    # 角度 [0, 180] → 色相 [0, 180]（OpenCV HSV 範圍）
    aolp_path = os.path.join(output_dir, f"{scene_name}_AoLP.png")
    
    # 使用 HSV 色輪：H = AoLP, S = DoLP, V = S0
    hsv = np.zeros((AoLP.shape[0], AoLP.shape[1], 3), dtype=np.uint8)
    hsv[:,:,0] = AoLP.astype(np.uint8)  # H: 角度 [0, 180]
    hsv[:,:,1] = (DoLP * 255).astype(np.uint8)  # S: 偏振度
    hsv[:,:,2] = np.clip(S0 / (S0.max() + 1e-10) * 255, 0, 255).astype(np.uint8)  # V: 強度
    
    aolp_colored = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(aolp_path, aolp_colored)
    print(f"    -> 保存 AoLP (HSV): {aolp_path}")
    
    # 4. 保存 AoLP（純色相，忽略亮度）
    aolp_pure_path = os.path.join(output_dir, f"{scene_name}_AoLP_pure.png")
    hsv_pure = np.zeros_like(hsv)
    hsv_pure[:,:,0] = AoLP.astype(np.uint8)  # H: 角度
    hsv_pure[:,:,1] = 255  # S: 飽和度最大
    hsv_pure[:,:,2] = 255  # V: 亮度最大
    aolp_pure_colored = cv2.cvtColor(hsv_pure, cv2.COLOR_HSV2BGR)
    cv2.imwrite(aolp_pure_path, aolp_pure_colored)
    print(f"    -> 保存 AoLP (純色相): {aolp_pure_path}")
    
    return {
        'dolp': dolp_path,
        'dolp_color': dolp_color_path,
        'aolp': aolp_path,
        'aolp_pure': aolp_pure_path,
    }


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
    注意：深度渲染使用非偏振 variant 以確保兼容性
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
    
    # 🔥 暫時切換到非偏振 variant 渲染深度
    current_variant = mi.variant()
    try:
        # 嘗試使用 scalar_rgb（最基本的 variant）
        depth_variant = 'scalar_rgb'
        mi.set_variant(depth_variant)
        print(f"  [INFO] 深度渲染使用 variant: {depth_variant}")
    except Exception as e:
        print(f"  [WARNING] 無法切換 variant: {e}")
        # 保持當前 variant
    
    # 構建深度渲染場景
    depth_scene_dict = {
        'type': 'scene',
        'integrator': {
            'type': 'path',
            'max_depth': 2,  # 深度只需要第一次碰撞
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
                'type': 'independent',
                'sample_count': 16,
            },
        },
        # 添加均勻環境光以便看到深度
        'emitter': {
            'type': 'constant',
            'radiance': {'type': 'spectrum', 'value': 1.0},
        },
    }
    
    # 添加非玻璃幾何 - 使用純白 diffuse
    if non_glass_obj and os.path.exists(non_glass_obj):
        depth_scene_dict['mesh_opaque'] = {
            'type': 'obj',
            'filename': non_glass_obj,
            'face_normals': False,
            'to_world': transform,
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'spectrum', 'value': 1.0},
            },
        }
    
    # 添加玻璃幾何 - 用不透明 diffuse 替代
    if glass_obj and os.path.exists(glass_obj):
        depth_scene_dict['mesh_glass'] = {
            'type': 'obj',
            'filename': glass_obj,
            'face_normals': False,
            'to_world': transform,
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'spectrum', 'value': 1.0},
            },
        }
    
    try:
        scene = mi.load_dict(depth_scene_dict)
        
        # 使用射線追蹤直接獲取深度
        depth = compute_depth_from_scene(scene, camera_position, camera_target, 
                                         render_cfg['width'], render_cfg['height'],
                                         cam_cfg['fov'])
        
        print(f"  [DEBUG] 深度範圍: [{depth[depth > 0].min():.4f}, {depth[depth > 0].max():.4f}] m")
        
    except Exception as e:
        print(f"  [ERROR] 深度渲染失敗: {e}")
        depth = np.zeros((render_cfg['height'], render_cfg['width']), dtype=np.float32)
    
    finally:
        # 🔥 切換回偏振 variant
        try:
            mi.set_variant(current_variant)
        except:
            pass
    
    return depth


def compute_depth_from_scene(scene, camera_position, camera_target, width, height, fov):
    """
    使用射線追蹤計算深度圖
    """
    import drjit as dr
    
    cam_pos = np.array(camera_position, dtype=np.float32)
    cam_target = np.array(camera_target, dtype=np.float32)
    
    # 相機方向
    cam_dir = cam_target - cam_pos
    cam_dir = cam_dir / np.linalg.norm(cam_dir)
    
    # 相機座標系
    up = np.array([0, 1, 0], dtype=np.float32)
    right = np.cross(cam_dir, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, cam_dir)
    
    # FOV
    aspect = width / height
    fov_rad = np.radians(fov)
    half_width = np.tan(fov_rad / 2)
    half_height = half_width / aspect
    
    # 生成像素座標
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    
    # 正規化到 [-1, 1]
    u = (2 * x_coords / width - 1) * half_width
    v = (1 - 2 * y_coords / height) * half_height
    
    # 射線方向
    ray_dirs = (cam_dir.reshape(1, 1, 3) + 
                u[:, :, np.newaxis] * right.reshape(1, 1, 3) + 
                v[:, :, np.newaxis] * up.reshape(1, 1, 3))
    ray_dirs = ray_dirs / np.linalg.norm(ray_dirs, axis=2, keepdims=True)
    
    # 展平
    ray_dirs_flat = ray_dirs.reshape(-1, 3).astype(np.float32)
    origins_flat = np.tile(cam_pos, (width * height, 1)).astype(np.float32)
    
    # 創建射線
    rays = mi.Ray3f(
        o=mi.Point3f(origins_flat[:, 0], origins_flat[:, 1], origins_flat[:, 2]),
        d=mi.Vector3f(ray_dirs_flat[:, 0], ray_dirs_flat[:, 1], ray_dirs_flat[:, 2])
    )
    
    # 射線追蹤
    si = scene.ray_intersect(rays)
    
    # 提取 t（射線參數 = 距離）
    t_values = np.array(si.t).reshape(height, width)
    valid_mask = np.array(si.is_valid()).reshape(height, width)
    
    # 深度 = 沿相機 z 軸的距離
    # 對於平行投影近似，直接使用 t * cos(angle) ≈ t（因為大部分射線接近中心）
    # 更精確：計算到相機平面的距離
    hit_points = origins_flat[:, np.newaxis] + ray_dirs_flat[:, np.newaxis, :].squeeze() * np.array(si.t)[:, np.newaxis]
    
    # 投影到相機 z 軸
    depths_flat = np.sum((hit_points - cam_pos) * cam_dir, axis=1)
    depth = depths_flat.reshape(height, width).astype(np.float32)
    
    # 無效區域設為 0
    depth[~valid_mask] = 0
    
    return depth


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
    # 1. 渲染左相機 (I∥)
    # ============================================================
    print(f"\n[1/4] 渲染左相機 (I∥)...")
    
    left_scene_dict = build_polarized_scene(
        obj_path=obj_path,
        camera_position=left_cam_pos,
        camera_target=left_target,
        spp=render_cfg['spp'],
    )
    
    left_scene = load_and_setup_scene(left_scene_dict)
    
    # 使用 Stokes 渲染
    left_stokes = render_stokes_vector(
        left_scene, 
        spp=render_cfg['spp'],
        spp_per_batch=render_cfg['spp_per_batch']
    )
    
    # 🔥 Stage 1: 被動偏振模式
    left_parallel, left_cross = extract_polarization_images_stage1(
        left_stokes,
        polarizer_angle_parallel=0.0,
        polarizer_angle_cross=90.0,
        amplify_factor=5.0,   # 回滾
        exposure=2.0,         # 回滾
        active_polarization=False,
        source_polarization_angle=0.0,
    )
    
    # 🔥 保存原始 Stokes 數據用於 DoLP/AoLP 計算
    left_stokes_saved = left_stokes.copy()
    
    # 暫存左相機圖像（稍後統一保存 PNG）
    del left_scene, left_stokes
    gc.collect()
    
    # ============================================================
    # 2. 渲染右相機 (I⊥)
    # ============================================================
    print(f"\n[2/4] 渲染右相機 (I⊥)...")
    
    right_scene_dict = build_polarized_scene(
        obj_path=obj_path,
        camera_position=right_cam_pos,
        camera_target=right_target,
        spp=render_cfg['spp'],
    )
    
    right_scene = load_and_setup_scene(right_scene_dict)
    
    # 使用 Stokes 渲染
    right_stokes = render_stokes_vector(
        right_scene,
        spp=render_cfg['spp'],
        spp_per_batch=render_cfg['spp_per_batch']
    )
    
    # 🔥 Stage 1: 被動偏振模式
    right_parallel, right_cross = extract_polarization_images_stage1(
        right_stokes,
        polarizer_angle_parallel=0.0,
        polarizer_angle_cross=90.0,
        amplify_factor=5.0,   # 回滾
        exposure=2.0,         # 回滾
        active_polarization=False,
        source_polarization_angle=0.0,
    )
    
    del right_scene, right_stokes
    gc.collect()
    
    # ============================================================
    # 🔥 新增：自動平衡背景強度
    # ============================================================
    if CONFIG['output'].get('auto_balance', True):
        print(f"\n[INFO] 檢測並平衡背景強度...")
        
        left_parallel, right_cross = balance_background_intensity(
            left_parallel, right_cross, 
            verbose=True
        )
    else:
        print(f"\n[INFO] 跳過背景強度平衡（已禁用）")
    
    # ============================================================
    # 🔥 保存 EXR 和 PNG（使用統一範圍）
    # ============================================================
    print(f"\n[INFO] 保存偏振圖像...")
    
    # 保存 EXR（原始數據）
    left_parallel_path = os.path.join(output_dir, f"{scene_name}_left_parallel.exr")
    save_grayscale_exr(left_parallel, left_parallel_path)
    outputs['left_parallel'] = left_parallel_path
    
    right_cross_path = os.path.join(output_dir, f"{scene_name}_right_cross.exr")
    save_grayscale_exr(right_cross, right_cross_path)
    outputs['right_cross'] = right_cross_path
    
    # 🔥 計算兩張圖的統一範圍
    vmin = min(left_parallel.min(), right_cross.min())
    vmax = max(left_parallel.max(), right_cross.max())
    print(f"    [INFO] 統一 normalization 範圍: [{vmin:.4f}, {vmax:.4f}]")
    
    # 保存 PNG（使用統一範圍，便於比較）
    left_parallel_png = os.path.join(output_dir, f"{scene_name}_left_parallel.png")
    save_grayscale_png_fixed_range(left_parallel, left_parallel_png, vmin, vmax)
    outputs['left_parallel_png'] = left_parallel_png
    
    right_cross_png = os.path.join(output_dir, f"{scene_name}_right_cross.png")
    save_grayscale_png_fixed_range(right_cross, right_cross_png, vmin, vmax)
    outputs['right_cross_png'] = right_cross_png
    
    # 🔥 另外保存自動範圍的 PNG（讓每張圖都能看清楚）
    left_auto_png = os.path.join(output_dir, f"{scene_name}_left_parallel_auto.png")
    save_grayscale_png(left_parallel, left_auto_png)
    
    right_auto_png = os.path.join(output_dir, f"{scene_name}_right_cross_auto.png")
    save_grayscale_png(right_cross, right_auto_png)
    
    # 🔥 保存差異圖（直接顯示偏振差異）
    diff = np.abs(left_parallel - right_cross)
    diff_path = os.path.join(output_dir, f"{scene_name}_polarization_diff.png")
    save_grayscale_png(diff, diff_path)
    outputs['polarization_diff'] = diff_path
    print(f"    [INFO] 偏振差異: min={diff.min():.4f}, max={diff.max():.4f}, mean={diff.mean():.4f}")
    
    # 🔥 Debug: 保存同視角的偏振比較（從左相機提取兩種偏振）
    # 這可以排除視角差異，純粹看偏振效果
    print(f"\n    [DEBUG] 保存同視角偏振比較...")
    same_view_diff = np.abs(left_parallel - left_cross)  # 同視角的 I∥ vs I⊥
    same_view_diff_path = os.path.join(output_dir, f"{scene_name}_same_view_polarization_diff.png")
    save_grayscale_png(same_view_diff, same_view_diff_path)
    print(f"    [DEBUG] 同視角偏振差異: min={same_view_diff.min():.4f}, max={same_view_diff.max():.4f}, mean={same_view_diff.mean():.4f}")
    
    # 保存 left_cross 作為參考
    left_cross_png = os.path.join(output_dir, f"{scene_name}_left_cross_debug.png")
    save_grayscale_png_fixed_range(left_cross, left_cross_png, vmin, vmax)
    print(f"    [DEBUG] 保存 left_cross 作為對比: {left_cross_png}")
    
    # ============================================================
    # 🔥 計算並保存 DoLP 和 AoLP（論文可視化）
    # ============================================================
    if CONFIG['output'].get('save_dolp_aolp', True):
        print(f"\n[2.5/4] 計算 DoLP 和 AoLP...")
        try:
            # 🔥 使用保存的原始 Stokes 數據計算
            S0, DoLP, AoLP = compute_dolp_aolp(left_stokes_saved)
            
            print(f"  [DoLP/AoLP] DoLP 範圍: [{DoLP.min():.4f}, {DoLP.max():.4f}], 均值: {DoLP.mean():.4f}")
            
            # 保存可視化（不需要 matplotlib，使用 OpenCV）
            # DoLP 灰階
            dolp_path = os.path.join(output_dir, f"{scene_name}_DoLP.png")
            dolp_uint8 = (DoLP * 255).astype(np.uint8)
            cv2.imwrite(dolp_path, dolp_uint8)
            print(f"    -> 保存 DoLP: {dolp_path}")
            
            # DoLP 彩色（使用 OpenCV 的 colormap）
            dolp_color_path = os.path.join(output_dir, f"{scene_name}_DoLP_color.png")
            dolp_colored = cv2.applyColorMap(dolp_uint8, cv2.COLORMAP_JET)
            cv2.imwrite(dolp_color_path, dolp_colored)
            print(f"    -> 保存 DoLP (彩色): {dolp_color_path}")
            
            # AoLP HSV
            aolp_path = os.path.join(output_dir, f"{scene_name}_AoLP.png")
            hsv = np.zeros((AoLP.shape[0], AoLP.shape[1], 3), dtype=np.uint8)
            hsv[:,:,0] = AoLP.astype(np.uint8)  # H: 角度 [0, 180]
            hsv[:,:,1] = (DoLP * 255).astype(np.uint8)  # S: 偏振度
            hsv[:,:,2] = np.clip(S0 / (S0.max() + 1e-10) * 255, 0, 255).astype(np.uint8)  # V: 強度
            aolp_colored = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imwrite(aolp_path, aolp_colored)
            print(f"    -> 保存 AoLP (HSV): {aolp_path}")
            
            outputs['dolp'] = dolp_path
            outputs['dolp_color'] = dolp_color_path
            outputs['aolp'] = aolp_path
            
            # 清理
            del left_stokes_saved
            gc.collect()
            
        except Exception as e:
            print(f"    [WARNING] DoLP/AoLP 計算失敗: {e}")
            import traceback
            traceback.print_exc()
    
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
        cv2.imwrite(
            os.path.join(output_dir, f"{scene_name}_left_parallel.png"),
            preview_parallel
        )
        
        # I_cross 預覽
        preview_cross = tonemap(right_cross)
        cv2.imwrite(
            os.path.join(output_dir, f"{scene_name}_right_cross.png"),
            preview_cross
        )
        
        # 偏振差異圖
        diff = np.abs(left_parallel - right_cross)
        diff_normalized = diff / (np.max(diff) + 1e-6)
        preview_diff = (diff_normalized * 255).astype(np.uint8)
        cv2.imwrite(
            os.path.join(output_dir, f"{scene_name}_polarization_diff.png"),
            preview_diff
        )
        
        # 深度預覽（使用 OpenCV colormap）
        if depth is not None:
            depth_valid = depth[depth > 0]
            if len(depth_valid) > 0:
                depth_min, depth_max = depth_valid.min(), depth_valid.max()
                depth_norm = (depth_max - depth) / (depth_max - depth_min + 1e-6)
                depth_norm = np.clip(depth_norm, 0, 1)
                depth_norm[depth <= 0] = 0
                
                depth_uint8 = (depth_norm * 255).astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
                depth_colored[depth <= 0] = 0
                
                cv2.imwrite(
                    os.path.join(output_dir, f"{scene_name}_depth.png"),
                    depth_colored
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
    
    # ============================================================
    # 7. 🔥 生成場景品質報告
    # ============================================================
    print(f"\n[Report] 生成品質報告...")
    
    report = generate_scene_report(
        scene_name=scene_name,
        I_parallel=left_parallel,
        I_cross=right_cross,
        depth=depth,
        render_cfg=render_cfg,
        cam_cfg=cam_cfg,
    )
    
    report_path = os.path.join(output_dir, f"{scene_name}_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    outputs['report'] = report_path
    
    # 打印摘要
    print(f"    DoLP (玻璃區域): {report['polarization']['glass_region']['dolp_mean']:.4f}")
    print(f"    DoLP (背景區域): {report['polarization']['background_region']['dolp_mean']:.4f}")
    print(f"    I∥/I⊥ 比值 (全局): {report['polarization']['intensity_ratio']['mean']:.1f}x")
    print(f"    I∥/I⊥ 比值 (背景): {report['intensity_balance']['background_ratio_mean']:.2f}x " + 
          ("✓ 平衡" if report['intensity_balance']['is_balanced'] else "⚠️ 不平衡"))
    print(f"    SNR: {report['noise']['snr_polarization']:.1f}x")
    print(f"    品質: {report['quality']['level']} ({report['quality']['score']}分)")
    
    if report['warnings']:
        print(f"    ⚠️ 警告:")
        for w in report['warnings']:
            print(f"       - {w}")
    
    print(f"\n{'='*60}")
    print(f"場景 {scene_name} 渲染完成!")
    print(f"{'='*60}\n")
    
    return outputs


def generate_scene_report(scene_name: str, I_parallel: np.ndarray, I_cross: np.ndarray,
                          depth: np.ndarray, render_cfg: dict, cam_cfg: dict) -> dict:
    """
    生成場景品質報告
    
    報告內容：
    1. 偏振信號強度（分區域）
    2. 噪點水平
    3. 深度圖有效性
    4. 綜合品質評分
    """
    from datetime import datetime
    
    report = {
        'scene_name': scene_name,
        'timestamp': datetime.now().isoformat(),
        'render_config': {
            'width': render_cfg['width'],
            'height': render_cfg['height'],
            'spp': render_cfg['spp'],
            'max_depth': render_cfg['max_depth'],
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
    
    # 分區域
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
        'snr_parallel': float(np.mean(I_parallel[I_parallel > 0])) / (noise_std + 1e-10) if np.any(I_parallel > 0) else 0,
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
    # 🔥 新增：強度平衡檢測（背景區域的 I∥/I⊥ 比值）
    # ============================================================
    # 背景區域應該 I∥ ≈ I⊥，這樣 RAFT 才能正常做雙目匹配
    if np.any(low_dolp_mask):
        bg_parallel = I_parallel[low_dolp_mask]
        bg_cross = I_cross[low_dolp_mask]
        
        # 背景區域的強度比值
        bg_ratio_mask = bg_cross > 0.01
        if np.any(bg_ratio_mask):
            bg_intensity_ratio = bg_parallel[bg_ratio_mask] / bg_cross[bg_ratio_mask]
            bg_ratio_mean = float(np.mean(bg_intensity_ratio))
            bg_ratio_std = float(np.std(bg_intensity_ratio))
        else:
            bg_ratio_mean = 1.0
            bg_ratio_std = 0.0
        
        # 背景區域的平均亮度
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
        'is_balanced': 0.5 <= bg_ratio_mean <= 2.0,  # 背景比值應接近 1
    }
    
    # ============================================================
    # 5. 綜合品質評估
    # ============================================================
    glass_ratio = glass_stats['pixel_ratio']
    glass_dolp_mean = glass_stats['dolp_mean']
    bg_dolp_mean = bg_stats['dolp_mean']
    
    # 計算分數
    scores = []
    
    # 偏振分數（玻璃區域）
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
    
    # 區域對比分數（玻璃 vs 背景）
    if glass_dolp_mean > bg_dolp_mean * 3:
        contrast_score = 100
    elif glass_dolp_mean > bg_dolp_mean * 2:
        contrast_score = 80
    elif glass_dolp_mean > bg_dolp_mean:
        contrast_score = 60
    else:
        contrast_score = 40
    scores.append(contrast_score)
    
    # 🔥 新增：強度平衡分數（背景區域 I∥/I⊥ 應接近 1）
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
    
    # 品質等級
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
        'balance_score': balance_score,  # 🔥 新增
        'valid': overall_score >= 40,
    }
    
    # ============================================================
    # 6. 警告
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
    
    # 🔥 新增：強度平衡警告
    if not report['intensity_balance']['is_balanced']:
        report['warnings'].append(
            f'背景區域強度不平衡（I∥/I⊥={bg_ratio_mean:.2f}），'
            f'RAFT雙目對齊可能失敗，建議啟用非偏振背景光'
        )
    
    return report


# ============================================================
# 命令行介面
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='PIDS Mitsuba 3 偏振渲染器 (物理正確版)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 基本渲染（預設已包含恆定環境光）
  python renderer_stage1_exaggerated.py --scene_file scene.obj --output_dir ./output
  
  # 批次渲染
  python renderer_stage1_exaggerated.py --input_dir ./scenes --output_dir ./output
  
  # 調整光源比例（偏振光:環境光 = 1:30 是好的起點）
  python renderer_stage1_exaggerated.py --scene_file scene.obj --output_dir ./output \\
      --polarized_intensity 500 --fill_intensity 15000
  
  # 關閉環境光（僅用於調試）
  python renderer_stage1_exaggerated.py --scene_file scene.obj --output_dir ./output --no_fill_light

光源說明:
  - 偏振光 (led): 只照射玻璃產生偏振效果，預設強度較低
  - 環境光 (fill_light): 恆定非偏振光，讓背景亮度一致，預設強度較高
  - 建議比例: fill_intensity ≈ 20-30x polarized_intensity
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
    parser.add_argument('--max_depth', type=int, default=CONFIG['render']['max_depth'],
                        help=f"光線最大反彈次數 (預設: {CONFIG['render']['max_depth']}，多層玻璃建議 16-32)")
    parser.add_argument('--max_scenes', type=int, default=None,
                        help='最大渲染場景數')
    
    # 🔥 材質選項
    parser.add_argument('--glass_type', type=str, default=CONFIG['materials']['glass']['type'],
                        choices=['dielectric', 'thindielectric', 'roughdielectric'],
                        help=f"玻璃材質類型 (預設: {CONFIG['materials']['glass']['type']})")
    parser.add_argument('--glass_ior', type=float, default=CONFIG['materials']['glass']['ior'],
                        help=f"玻璃折射率 (預設: {CONFIG['materials']['glass']['ior']})")
    
    # 🔥 光源選項
    parser.add_argument('--polarized_intensity', type=float, 
                        default=CONFIG['lighting']['led']['intensity'],
                        help=f"偏振光強度 (預設: {CONFIG['lighting']['led']['intensity']})")
    parser.add_argument('--fill_intensity', type=float,
                        default=CONFIG['lighting']['fill_light']['intensity'],
                        help=f"非偏振背景光強度 (預設: {CONFIG['lighting']['fill_light']['intensity']})")
    parser.add_argument('--no_fill_light', action='store_true',
                        help='關閉非偏振背景光')
    
    # 其他選項
    parser.add_argument('--no_preview', action='store_true', help='不保存預覽圖')
    parser.add_argument('--no_balance', action='store_true', 
                        help='禁用背景強度自動平衡（用於調試）')
    
    args = parser.parse_args()
    
    # 更新配置
    CONFIG['render']['spp'] = args.spp
    CONFIG['render']['spp_per_batch'] = args.spp_per_batch
    CONFIG['render']['max_depth'] = args.max_depth
    CONFIG['output']['save_preview'] = not args.no_preview
    
    # 🔥 更新材質配置
    CONFIG['materials']['glass']['type'] = args.glass_type
    CONFIG['materials']['glass']['ior'] = args.glass_ior
    
    # 🔥 更新光源配置
    CONFIG['lighting']['led']['intensity'] = args.polarized_intensity
    CONFIG['lighting']['fill_light']['intensity'] = args.fill_intensity
    CONFIG['lighting']['fill_light']['enabled'] = not args.no_fill_light
    
    print(f"\n[CONFIG] SPP: {args.spp}")
    print(f"[CONFIG] max_depth: {args.max_depth}")
    print(f"[CONFIG] 玻璃材質: {args.glass_type} (IOR={args.glass_ior})")
    print(f"[CONFIG] 偏振光強度: {args.polarized_intensity}")
    print(f"[CONFIG] 背景光強度: {args.fill_intensity} ({'啟用' if not args.no_fill_light else '關閉'})")
    print(f"[CONFIG] 背景強度平衡: {'啟用' if not args.no_balance else '禁用'}")
    
    # 🔥 更新平衡配置
    CONFIG['output']['auto_balance'] = not args.no_balance
    
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
