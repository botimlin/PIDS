"""
PIDS Mitsuba 3 偏振渲染器 (Blender 場景版)
==========================================

此腳本載入 Blender 匯出的 OBJ 場景，並執行偏振光追蹤渲染。

工作流程:
1. Blender 建立場景 → 匯出 .obj
2. 此腳本載入 .obj → 加入偏振相機和光源 → Mitsuba 渲染
3. 輸出 I∥, I⊥, 深度圖

使用方法:
    # 渲染單一場景
    python pids_renderer_blender.py --scene path/to/scene.obj --output ./output
    
    # 批次渲染
    python pids_renderer_blender.py --scene_dir path/to/scenes/ --output ./output
    
    # 測試模式（使用內建簡單場景）
    python pids_renderer_blender.py --test
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# 導入配置
from pids_config import (
    CAMERA_CONFIG, 
    LIGHT_CONFIG, 
    POLARIZATION_CONFIG,
    SCENE_CONFIG,
    RENDER_CONFIG,
    DATA_CONFIG,
    compute_disparity_range,
)

# ============================================================
# Mitsuba 初始化
# ============================================================

MI = None  # 全域 Mitsuba 模組

def init_mitsuba():
    """初始化 Mitsuba 3 並選擇適當的 variant"""
    global MI
    
    if MI is not None:
        return MI
        
    try:
        import mitsuba as mi
    except ImportError:
        print("錯誤: 請先安裝 Mitsuba 3")
        print("  pip install mitsuba")
        sys.exit(1)
    
    # 嘗試使用 CUDA 偏振 variant
    target_variants = [
        RENDER_CONFIG['mitsuba_variant'],
        RENDER_CONFIG['fallback_variant'],
    ]
    
    available = mi.variants()
    for variant in target_variants:
        if variant in available:
            mi.set_variant(variant)
            print(f"✓ 使用 Mitsuba variant: {variant}")
            MI = mi
            return mi
    
    print("錯誤: 找不到支援偏振的 Mitsuba variant")
    print(f"可用的 variants: {available}")
    sys.exit(1)


# ============================================================
# 材質定義
# ============================================================

def get_material_dict(material_name):
    """
    根據材質名稱返回 Mitsuba 材質定義
    
    材質名稱規則（從 Blender 匯出）:
    - Glass_* → 玻璃
    - Acrylic_* → 壓克力
    - Diffuse_* → 漫反射
    - 其他 → 預設漫反射
    """
    name_lower = material_name.lower()
    
    if 'glass' in name_lower:
        return {
            'type': 'dielectric',
            'int_ior': 1.5,
            'ext_ior': 1.0,
        }
    elif 'acrylic' in name_lower:
        return {
            'type': 'dielectric',
            'int_ior': 1.49,
            'ext_ior': 1.0,
        }
    elif 'mirror' in name_lower:
        return {
            'type': 'conductor',
            'material': 'Al',  # 鋁
        }
    else:
        # 預設漫反射，從材質名稱嘗試提取顏色
        color = extract_color_from_name(name_lower)
        return {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                'value': color,
            },
        }


def extract_color_from_name(name):
    """從材質名稱提取顏色（簡單實作）"""
    color_map = {
        'red': [0.7, 0.2, 0.2],
        'green': [0.2, 0.7, 0.2],
        'blue': [0.2, 0.2, 0.7],
        'white': [0.8, 0.8, 0.8],
        'black': [0.1, 0.1, 0.1],
        'gray': [0.5, 0.5, 0.5],
        'grey': [0.5, 0.5, 0.5],
        'wood': [0.6, 0.4, 0.2],
        'brown': [0.5, 0.3, 0.1],
    }
    
    for color_name, color_value in color_map.items():
        if color_name in name:
            return color_value
    
    # 預設灰色
    return [0.5, 0.5, 0.5]


# ============================================================
# 場景載入
# ============================================================

def load_obj_scene(mi, obj_path, samples=None):
    """
    載入 OBJ 場景並加入偏振相機系統
    
    Parameters:
    -----------
    mi : module
        Mitsuba 模組
    obj_path : str or Path
        OBJ 檔案路徑
    samples : int, optional
        採樣數，預設使用配置值
        
    Returns:
    --------
    scene_dict : dict
        Mitsuba 場景字典
    """
    obj_path = Path(obj_path)
    
    if not obj_path.exists():
        raise FileNotFoundError(f"找不到場景檔案: {obj_path}")
    
    print(f"載入場景: {obj_path}")
    
    # 相機參數
    resolution = CAMERA_CONFIG['output_resolution']
    fov_x = CAMERA_CONFIG['fov_horizontal_deg']
    half_baseline = CAMERA_CONFIG['baseline_mm'] / 2 / 1000  # 轉換為公尺
    
    # 光源參數
    light_width = LIGHT_CONFIG['panel_width_mm'] / 1000
    light_height = LIGHT_CONFIG['panel_height_mm'] / 1000
    
    # 工作距離中點
    look_distance = (CAMERA_CONFIG['min_distance_mm'] + CAMERA_CONFIG['max_distance_mm']) / 2 / 1000
    
    if samples is None:
        samples = RENDER_CONFIG['samples_training']
    
    # 建立場景字典
    scene_dict = {
        'type': 'scene',
        
        # 使用 Stokes integrator 進行偏振追蹤
        'integrator': {
            'type': 'stokes',
            'nested': {
                'type': 'path',
                'max_depth': RENDER_CONFIG['max_depth'],
            },
        },
        
        # 左相機 (I∥)
        'sensor_left': {
            'type': 'perspective',
            'fov': fov_x,
            'fov_axis': 'x',
            'to_world': mi.ScalarTransform4f.look_at(
                origin=[-half_baseline, 0, 0],
                target=[-half_baseline, 0, look_distance],
                up=[0, 1, 0]
            ),
            'film': {
                'type': 'hdrfilm',
                'width': resolution[0],
                'height': resolution[1],
                'pixel_format': 'rgb',
                'component_format': 'float32',
                'rfilter': {'type': 'gaussian'},
            },
            'sampler': {
                'type': 'independent',
                'sample_count': samples,
            },
        },
        
        # 右相機 (I⊥)
        'sensor_right': {
            'type': 'perspective',
            'fov': fov_x,
            'fov_axis': 'x',
            'to_world': mi.ScalarTransform4f.look_at(
                origin=[half_baseline, 0, 0],
                target=[half_baseline, 0, look_distance],
                up=[0, 1, 0]
            ),
            'film': {
                'type': 'hdrfilm',
                'width': resolution[0],
                'height': resolution[1],
                'pixel_format': 'rgb',
                'component_format': 'float32',
                'rfilter': {'type': 'gaussian'},
            },
            'sampler': {
                'type': 'independent',
                'sample_count': samples,
            },
        },
        
        # 偏振 LED 面板光源 (C30Bi)
        # 55° 入射角，接近 Brewster angle
        'light_shape': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.translate([0, 0.12, -0.08]) @
                        mi.ScalarTransform4f.rotate([1, 0, 0], -55) @
                        mi.ScalarTransform4f.scale([light_width/2, light_height/2, 1]),
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    'value': [50.0, 50.0, 50.0],
                },
            },
        },
        
        # 載入 OBJ 場景
        'imported_scene': {
            'type': 'obj',
            'filename': str(obj_path),
            # 注意：OBJ 的材質會被忽略，需要手動指定
            # 或使用下面的 shape group 方式
        },
    }
    
    return scene_dict


def load_obj_with_materials(mi, obj_path, mtl_mapping=None):
    """
    載入 OBJ 並手動指定材質
    
    這個函數解析 OBJ 檔案，為每個物體群組指定 Mitsuba 材質
    
    Parameters:
    -----------
    mi : module
        Mitsuba 模組
    obj_path : str or Path
        OBJ 檔案路徑
    mtl_mapping : dict, optional
        材質映射字典 {物體名稱: 材質類型}
    """
    obj_path = Path(obj_path)
    
    # 讀取 OBJ 檔案分析物體群組
    groups = parse_obj_groups(obj_path)
    print(f"找到 {len(groups)} 個物體群組")
    
    # 相機和光源設定（同上）
    resolution = CAMERA_CONFIG['output_resolution']
    fov_x = CAMERA_CONFIG['fov_horizontal_deg']
    half_baseline = CAMERA_CONFIG['baseline_mm'] / 2 / 1000
    look_distance = (CAMERA_CONFIG['min_distance_mm'] + CAMERA_CONFIG['max_distance_mm']) / 2 / 1000
    light_width = LIGHT_CONFIG['panel_width_mm'] / 1000
    light_height = LIGHT_CONFIG['panel_height_mm'] / 1000
    
    scene_dict = {
        'type': 'scene',
        
        # 使用 Stokes integrator 進行偏振追蹤
        'integrator': {
            'type': 'stokes',
            'nested': {
                'type': 'path',
                'max_depth': RENDER_CONFIG['max_depth'],
            },
        },
        
        # 相機（同前）
        'sensor_left': {
            'type': 'perspective',
            'fov': fov_x,
            'fov_axis': 'x',
            'to_world': mi.ScalarTransform4f.look_at(
                origin=[-half_baseline, 0, 0],
                target=[-half_baseline, 0, look_distance],
                up=[0, 1, 0]
            ),
            'film': {
                'type': 'hdrfilm',
                'width': resolution[0],
                'height': resolution[1],
                'pixel_format': 'rgb',
                'component_format': 'float32',
            },
            'sampler': {
                'type': 'independent',
                'sample_count': RENDER_CONFIG['samples_training'],
            },
        },
        
        'sensor_right': {
            'type': 'perspective',
            'fov': fov_x,
            'fov_axis': 'x',
            'to_world': mi.ScalarTransform4f.look_at(
                origin=[half_baseline, 0, 0],
                target=[half_baseline, 0, look_distance],
                up=[0, 1, 0]
            ),
            'film': {
                'type': 'hdrfilm',
                'width': resolution[0],
                'height': resolution[1],
                'pixel_format': 'rgb',
                'component_format': 'float32',
            },
            'sampler': {
                'type': 'independent',
                'sample_count': RENDER_CONFIG['samples_training'],
            },
        },
        
        'light_shape': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.translate([0, 0.08, 0.05]) @
                        mi.ScalarTransform4f.rotate([1, 0, 0], -30) @
                        mi.ScalarTransform4f.scale([light_width/2, light_height/2, 1]),
            'emitter': {
                'type': 'area',
                'radiance': {'type': 'rgb', 'value': [50.0, 50.0, 50.0]},
            },
        },
    }
    
    # 為每個物體群組加入對應材質
    for group_name in groups:
        # 決定材質
        if mtl_mapping and group_name in mtl_mapping:
            material = get_material_dict(mtl_mapping[group_name])
        else:
            # 根據命名自動決定
            material = get_material_dict(group_name)
        
        # 加入物體
        scene_dict[f'shape_{group_name}'] = {
            'type': 'obj',
            'filename': str(obj_path),
            'shape_group': group_name,
            'bsdf': material,
        }
    
    return scene_dict


def parse_obj_groups(obj_path):
    """解析 OBJ 檔案中的物體群組名稱"""
    groups = set()
    
    with open(obj_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('g ') or line.startswith('o '):
                group_name = line.split()[1] if len(line.split()) > 1 else 'default'
                groups.add(group_name)
    
    return list(groups)


# ============================================================
# 偏振影像處理
# ============================================================

def extract_polarization_images(stokes_image, debug=True):
    """
    最終修正版：
    1. 修正 Mitsuba Interleaved 資料讀取錯誤 (修復洋紅色噪點)
    2. 調整光譜轉 RGB 矩陣 (修復黃色色偏)
    3. 統一白平衡基準 (保留 I_para 與 I_cross 的物理相對強度)
    """
    stokes_image = np.array(stokes_image)
    
    # 預設值
    I_parallel = stokes_image
    I_cross = stokes_image
    
    if len(stokes_image.shape) == 3:
        h, w, num_channels = stokes_image.shape
        
        # -----------------------------------------------------------
        # 情況 A: 15 通道 (Mitsuba spectral_polarized 預設輸出)
        # -----------------------------------------------------------
        if num_channels == 15:
            # 1. 解碼 Interleaved 格式 (步進切片)
            # Layout: [W1_S0, W1_S1, W1_S2, W2_S0, W2_S1, W2_S2, ...]
            S0_spectral = stokes_image[..., 0::3]
            S1_spectral = stokes_image[..., 1::3]
            
            # 2. 計算光譜域的偏振分量
            # I_parallel = 0.5 * (S0 + S1)
            # I_cross    = 0.5 * (S0 - S1)
            I_parallel_spectral = np.maximum(0.5 * (S0_spectral + S1_spectral), 0)
            I_cross_spectral    = np.maximum(0.5 * (S0_spectral - S1_spectral), 0)
            
            # 3. 光譜轉 RGB 矩陣 (Blue Boost 版本，解決偏黃問題)
            # Row: R, G, B; Col: Wavelength 0~4
            spectral_to_rgb_matrix = np.array([
                # Ch0(UV/紫)  Ch1(藍)  Ch2(綠)  Ch3(紅)  Ch4(IR)
                [0.00,       0.05,    0.10,    0.80,    0.05],  # R (主要吃長波)
                [0.00,       0.20,    0.75,    0.05,    0.00],  # G (主要吃中波)
                [0.85,       0.60,    0.05,    0.00,    0.00],  # B (主要吃短波，加強權重!)
            ])
            
            # 4. 轉換為 RGB
            I_parallel = I_parallel_spectral @ spectral_to_rgb_matrix.T
            I_cross    = I_cross_spectral    @ spectral_to_rgb_matrix.T

        # -----------------------------------------------------------
        # 情況 B: 12 通道 (RGB 模式)
        # -----------------------------------------------------------
        elif num_channels == 12:
            S0 = stokes_image[..., 0:3]
            S1 = stokes_image[..., 3:6]
            I_parallel = np.maximum(0.5 * (S0 + S1), 0)
            I_cross    = np.maximum(0.5 * (S0 - S1), 0)
            
        else:
            # Fallback
            base_img = stokes_image[..., :3]
            I_parallel = base_img
            I_cross = base_img

    # -----------------------------------------------------------
    # 統一白平衡 (Joint White Balance) - 您的修正
    # -----------------------------------------------------------
    def normalize_and_white_balance_joint(img_main, img_secondary):
        """
        以 img_main (通常是 I_parallel) 為基準計算白點，
        並將相同係數套用到 img_secondary，以保留物理強度差異。
        """
        # 1. 找出主圖各通道的 99% 亮度位準 (作為白點)
        p_high = np.percentile(img_main, 99, axis=(0, 1))
        
        # 防止過暗或除以零
        p_high[p_high < 1e-5] = 1e-5
        
        # 2. 用「同一組係數」正規化兩張圖
        img_main_balanced = np.clip(img_main / p_high, 0, 1)
        img_secondary_balanced = np.clip(img_secondary / p_high, 0, 1)
        
        return img_main_balanced, img_secondary_balanced

    # 應用修正後的白平衡邏輯
    I_parallel, I_cross = normalize_and_white_balance_joint(I_parallel, I_cross)

    # -----------------------------------------------------------
    # Gamma 校正 (Linear -> sRGB)
    # -----------------------------------------------------------
    I_parallel = np.power(I_parallel, 1/2.2)
    I_cross    = np.power(I_cross, 1/2.2)

    return I_parallel, I_cross


# ============================================================
# 渲染函數
# ============================================================

def render_scene(mi, scene_dict, output_dir, scene_name, save_preview=True):
    """
    渲染場景並輸出結果
    
    Parameters:
    -----------
    mi : module
        Mitsuba 模組
    scene_dict : dict
        場景字典
    output_dir : Path
        輸出目錄
    scene_name : str
        場景名稱（用於檔案命名）
    save_preview : bool
        是否儲存 PNG 預覽
        
    Returns:
    --------
    dict : 輸出檔案路徑
    """
    import cv2
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 載入場景
    scene = mi.load_dict(scene_dict)
    
    # 渲染左相機
    print(f"  渲染左相機 (I∥)...")
    image_left = mi.render(scene, sensor=0)
    stokes_left = np.array(image_left)
    
    # 渲染右相機
    print(f"  渲染右相機 (I⊥)...")
    image_right = mi.render(scene, sensor=1)
    stokes_right = np.array(image_right)
    
    # 提取偏振分量
    I_parallel, _ = extract_polarization_images(stokes_left)
    _, I_cross = extract_polarization_images(stokes_right)
    
    # 計算深度和視差（使用 AOV）
    print(f"  計算深度圖...")
    # 這裡簡化處理，實際深度需要從場景幾何計算
    
    # 儲存結果 - 使用多種方式嘗試儲存 EXR
    def save_exr(filepath, image):
        """嘗試多種方式儲存 EXR"""
        filepath = str(filepath)
        image = image.astype(np.float32)
        
        # 方法 1: 嘗試使用 OpenCV
        try:
            import os
            os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
            cv2.imwrite(filepath, image)
            return True
        except Exception:
            pass
        
        # 方法 2: 使用 imageio
        try:
            import imageio.v3 as iio
            # imageio 需要 RGB 順序，OpenCV 是 BGR
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = image[:, :, ::-1]
            else:
                image_rgb = image
            iio.imwrite(filepath, image_rgb)
            return True
        except Exception:
            pass
        
        # 方法 3: 使用 OpenEXR (如果安裝了)
        try:
            import OpenEXR
            import Imath
            h, w = image.shape[:2]
            header = OpenEXR.Header(w, h)
            half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
            if len(image.shape) == 3:
                header['channels'] = {'R': half_chan, 'G': half_chan, 'B': half_chan}
                out = OpenEXR.OutputFile(filepath, header)
                out.writePixels({
                    'R': image[:, :, 2].tobytes(),
                    'G': image[:, :, 1].tobytes(),
                    'B': image[:, :, 0].tobytes()
                })
            else:
                header['channels'] = {'Y': half_chan}
                out = OpenEXR.OutputFile(filepath, header)
                out.writePixels({'Y': image.tobytes()})
            out.close()
            return True
        except Exception:
            pass
        
        # 方法 4: 退而求其次，儲存為 NPY
        npy_path = filepath.replace('.exr', '.npy')
        np.save(npy_path, image)
        print(f"    警告: 無法儲存 EXR，改存為 {npy_path}")
        return False
    
    save_exr(output_dir / f"{scene_name}_I_parallel.exr", I_parallel)
    save_exr(output_dir / f"{scene_name}_I_cross.exr", I_cross)
    
    if save_preview:
        def to_uint8(img):
            img = np.clip(img, 0, None)
            if img.max() > 0:
                img = img / np.percentile(img, 99)
            return (np.clip(img, 0, 1) * 255).astype(np.uint8)
        
        cv2.imwrite(str(output_dir / f"{scene_name}_I_parallel.png"), 
                    to_uint8(I_parallel))
        cv2.imwrite(str(output_dir / f"{scene_name}_I_cross.png"), 
                    to_uint8(I_cross))
        
        # 偏振差異圖
        diff = np.abs(I_parallel - I_cross)
        if len(diff.shape) == 3:
            diff = diff.mean(axis=2)
        diff_colored = cv2.applyColorMap(to_uint8(diff), cv2.COLORMAP_JET)
        cv2.imwrite(str(output_dir / f"{scene_name}_polarization_diff.png"), 
                    diff_colored)
    
    return {
        'I_parallel': output_dir / f"{scene_name}_I_parallel.exr",
        'I_cross': output_dir / f"{scene_name}_I_cross.exr",
    }


def render_obj_scene(obj_path, output_dir, save_preview=True):
    """
    渲染單一 OBJ 場景
    """
    mi = init_mitsuba()
    
    obj_path = Path(obj_path)
    scene_name = obj_path.stem
    
    print(f"\n處理場景: {scene_name}")
    
    # 載入場景
    scene_dict = load_obj_scene(mi, obj_path)
    
    # 渲染
    result = render_scene(mi, scene_dict, output_dir, scene_name, save_preview)
    
    return result


def batch_render(scene_dir, output_dir, save_preview=False):
    """
    批次渲染目錄中所有 OBJ 場景
    """
    from tqdm import tqdm
    
    mi = init_mitsuba()
    
    scene_dir = Path(scene_dir)
    output_dir = Path(output_dir)
    
    # 找出所有 OBJ 檔案
    obj_files = sorted(scene_dir.glob("*.obj"))
    print(f"找到 {len(obj_files)} 個場景檔案")
    
    results = []
    start_time = time.time()
    
    for obj_path in tqdm(obj_files, desc="渲染場景"):
        try:
            result = render_obj_scene(obj_path, output_dir, save_preview)
            results.append(result)
        except Exception as e:
            print(f"\n錯誤: {obj_path.name} - {e}")
            continue
    
    elapsed = time.time() - start_time
    print(f"\n完成！共渲染 {len(results)} 個場景")
    print(f"總耗時: {elapsed:.1f} 秒")
    if len(results) > 0:
        print(f"平均: {elapsed/len(results):.2f} 秒/場景")
    else:
        print("警告: 沒有成功渲染任何場景！")
    
    return results


# ============================================================
# 測試用內建場景
# ============================================================

def create_test_scene(mi):
    """建立測試用簡單場景（不需要 OBJ 檔案）"""
    
    resolution = CAMERA_CONFIG['output_resolution']
    fov_x = CAMERA_CONFIG['fov_horizontal_deg']
    half_baseline = CAMERA_CONFIG['baseline_mm'] / 2 / 1000
    
    # 使用配置的工作距離
    look_distance = 0.6  # 600mm，對焦點
    
    # 光源參數
    light_width = LIGHT_CONFIG['panel_width_mm'] / 1000
    light_height = LIGHT_CONFIG['panel_height_mm'] / 1000
    
    scene_dict = {
        'type': 'scene',
        
        # 使用 Stokes integrator 進行偏振追蹤
        'integrator': {
            'type': 'stokes',
            'nested': {
                'type': 'path',
                'max_depth': 12,
            },
        },
        
        # 左相機 - 朝向 +Z（場景方向）
        'sensor_left': {
            'type': 'perspective',
            'fov': fov_x,
            'fov_axis': 'x',
            'to_world': mi.ScalarTransform4f.look_at(
                origin=[-half_baseline, 0, 0],
                target=[-half_baseline, 0, look_distance],
                up=[0, 1, 0]
            ),
            'film': {
                'type': 'hdrfilm',
                'width': resolution[0],
                'height': resolution[1],
                'pixel_format': 'rgb',
                'component_format': 'float32',
            },
            'sampler': {
                'type': 'independent',
                'sample_count': 1024,  # RTX 3060Ti 高品質
            },
        },
        
        # 右相機
        'sensor_right': {
            'type': 'perspective',
            'fov': fov_x,
            'fov_axis': 'x',
            'to_world': mi.ScalarTransform4f.look_at(
                origin=[half_baseline, 0, 0],
                target=[half_baseline, 0, look_distance],
                up=[0, 1, 0]
            ),
            'film': {
                'type': 'hdrfilm',
                'width': resolution[0],
                'height': resolution[1],
                'pixel_format': 'rgb',
                'component_format': 'float32',
            },
            'sampler': {
                'type': 'independent',
                'sample_count': 1024,
            },
        },
        
        # 偏振 LED 光源 - 55° 入射角
        # 使用偏振發射器
        'light_shape': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.translate([0, 0.12, -0.08]) @
                        mi.ScalarTransform4f.rotate([1, 0, 0], -55) @
                        mi.ScalarTransform4f.scale([light_width/2, light_height/2, 1]),
            'bsdf': {
                'type': 'null',  # 透明，讓光通過
            },
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'spectrum',
                    'value': 150.0,  # 增加亮度
                },
            },
        },
        
        # 偏振片 (放在光源前方) - 0° 水平偏振
        'polarizer_source': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.translate([0, 0.11, -0.07]) @
                        mi.ScalarTransform4f.rotate([1, 0, 0], -55) @
                        mi.ScalarTransform4f.scale([light_width/2 + 0.005, light_height/2 + 0.005, 1]),
            'bsdf': {
                'type': 'polarizer',  # 線性偏振片
                'theta': 0.0,  # 0° = 水平偏振
            },
        },
        
        # 玻璃板 - 在工作距離範圍內 (528-695mm)
        # 傾斜角度接近 Brewster angle (~56° for glass)
        'glass_panel': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.translate([0.02, 0, 0.60]) @
                        mi.ScalarTransform4f.rotate([0, 1, 0], 35) @  # Y軸旋轉 35°
                        mi.ScalarTransform4f.scale([0.06, 0.10, 1]),  # 120x200mm 玻璃門
            'bsdf': {
                'type': 'dielectric',
                'int_ior': 1.5,
            },
        },
        
        # 第二塊玻璃 - 不同位置和角度
        'glass_panel_2': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.translate([-0.04, -0.02, 0.55]) @
                        mi.ScalarTransform4f.rotate([0, 1, 0], -30) @  # 反方向傾斜
                        mi.ScalarTransform4f.scale([0.04, 0.06, 1]),  # 80x120mm
            'bsdf': {
                'type': 'dielectric',
                'int_ior': 1.5,
            },
        },
        
        # 背景牆 - Z=750mm
        'background': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.translate([0, 0, 0.75]) @
                        mi.ScalarTransform4f.scale([0.4, 0.3, 1]),  # 800x600mm
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.7, 0.65, 0.6]},  # 米色牆
            },
        },
        
        # 地面 - 場景中心高度
        'ground': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.translate([0, -0.10, 0.6]) @
                        mi.ScalarTransform4f.rotate([1, 0, 0], -90) @
                        mi.ScalarTransform4f.scale([0.3, 0.3, 1]),  # 600x600mm
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.4, 0.35, 0.3]},  # 木地板色
            },
        },
        
        # 書架 (背景傢俱)
        'bookshelf': {
            'type': 'cube',
            'to_world': mi.ScalarTransform4f.translate([-0.08, 0, 0.72]) @
                        mi.ScalarTransform4f.scale([0.04, 0.09, 0.015]),  # 80x180x30mm
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.35, 0.25, 0.15]},  # 深木色
            },
        },
        
        # 桌子
        'table': {
            'type': 'cube',
            'to_world': mi.ScalarTransform4f.translate([0.06, -0.075, 0.65]) @
                        mi.ScalarTransform4f.scale([0.04, 0.0225, 0.025]),  # 80x45x50mm
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': [0.45, 0.35, 0.25]},  # 淺木色
            },
        },
    }
    
    return scene_dict


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='PIDS 渲染器 - Blender 場景版'
    )
    
    parser.add_argument(
        '--scene', '-s',
        type=str,
        help='單一 OBJ 場景檔案路徑'
    )
    
    parser.add_argument(
        '--scene_dir', '-d',
        type=str,
        help='OBJ 場景目錄（批次渲染）'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./rendered_output',
        help='輸出目錄'
    )
    
    parser.add_argument(
        '--preview',
        action='store_true',
        help='儲存 PNG 預覽圖'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='測試模式（使用內建場景）'
    )
    
    args = parser.parse_args()
    
    if args.test:
        # 測試模式
        print("=" * 60)
        print("PIDS 渲染器測試模式")
        print("=" * 60)
        
        mi = init_mitsuba()
        scene_dict = create_test_scene(mi)
        
        output_dir = Path('./test_output')
        render_scene(mi, scene_dict, output_dir, 'test_scene', save_preview=True)
        
        print(f"\n測試完成！輸出位於: {output_dir}")
        
    elif args.scene:
        # 單一場景
        render_obj_scene(args.scene, args.output, args.preview)
        
    elif args.scene_dir:
        # 批次渲染
        batch_render(args.scene_dir, args.output, args.preview)
        
    else:
        parser.print_help()
        print("\n請指定 --scene, --scene_dir, 或 --test")


if __name__ == '__main__':
    main()
