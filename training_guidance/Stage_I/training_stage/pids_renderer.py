"""
PIDS Mitsuba 3 偏振渲染器
========================

此腳本用於生成 Physics-Informed Deep Stereo 的訓練數據。
使用 Mitsuba 3 的偏振光追蹤功能，模擬非對稱偏振立體相機系統。

硬體配置:
- 相機: Sony IMX296LQR-C, 640×480 輸出
- 光源: Godox Litemons C30Bi
- 基線: 80mm
- 工作距離: 30-100mm

使用方法:
    python pids_renderer.py --num_samples 100 --output_dir ./data
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

def init_mitsuba():
    """初始化 Mitsuba 3 並選擇適當的 variant"""
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
            return mi
    
    # 找不到偏振 variant
    print("錯誤: 找不到支援偏振的 Mitsuba variant")
    print(f"可用的 variants: {available}")
    print("請確保安裝了完整版 Mitsuba 3")
    sys.exit(1)


# ============================================================
# 場景建構
# ============================================================

def create_camera_transform(position, target, up=[0, 1, 0]):
    """建立相機變換矩陣"""
    mi = init_mitsuba()
    return mi.ScalarTransform4f.look_at(
        origin=position,
        target=target,
        up=up
    )


def build_scene(mi, scene_params):
    """
    建構完整的 Mitsuba 場景
    
    Parameters:
    -----------
    mi : module
        Mitsuba 模組
    scene_params : dict
        場景參數，包含玻璃位置、旋轉、背景等
    
    Returns:
    --------
    scene : mi.Scene
        Mitsuba 場景物件
    """
    
    # 解析參數
    glass_pos = scene_params.get('glass_position', [0, 0, 0.07])
    glass_rot = scene_params.get('glass_rotation', [0, 0])  # [y_deg, x_deg]
    glass_size = scene_params.get('glass_size', [0.04, 0.04])
    bg_color = scene_params.get('background_color', [0.5, 0.5, 0.5])
    light_intensity = scene_params.get('light_intensity', 1.0)
    
    # 相機參數
    resolution = CAMERA_CONFIG['output_resolution']
    fov_x = CAMERA_CONFIG['fov_horizontal_deg']
    half_baseline = CAMERA_CONFIG['baseline_mm'] / 2 / 1000  # 轉換為公尺
    
    # 光源參數 (轉換為公尺)
    light_width = LIGHT_CONFIG['panel_width_mm'] / 1000
    light_height = LIGHT_CONFIG['panel_height_mm'] / 1000
    
    # 基礎光強度 (根據 C30Bi 的 8610 Lux @ 0.5m 估算)
    # 這個值需要根據實際渲染結果調整
    base_radiance = 50.0 * light_intensity
    
    # 計算相機目標點 (看向場景中心)
    look_distance = 0.6  # 600mm，工作距離中點 (528-695mm)
    
    scene_dict = {
        'type': 'scene',
        
        # ============================================
        # 積分器 (支援偏振的路徑追蹤)
        # ============================================
        'integrator': {
            'type': 'path',
            'max_depth': RENDER_CONFIG['max_depth'],
        },
        
        # ============================================
        # 左相機 (I∥ - 平行偏振)
        # ============================================
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
                'sample_count': scene_params.get('samples', RENDER_CONFIG['samples_training']),
            },
        },
        
        # ============================================
        # 右相機 (I⊥ - 正交偏振)
        # ============================================
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
                'sample_count': scene_params.get('samples', RENDER_CONFIG['samples_training']),
            },
        },
        
        # ============================================
        # 偏振 LED 面板光源 (C30Bi)
        # 位於相機後上方，55° 入射角 (接近 Brewster angle)
        # ============================================
        'light_panel': {
            'type': 'area',
            'radiance': {
                'type': 'rgb',
                'value': [base_radiance, base_radiance, base_radiance],
            },
            'shape': {
                'type': 'rectangle',
                # 光源位置：相機上方 120mm，後方 80mm，傾斜 55°
                'to_world': mi.ScalarTransform4f.translate([0, 0.12, -0.08]) @
                            mi.ScalarTransform4f.rotate([1, 0, 0], -55) @  # 55° 入射角
                            mi.ScalarTransform4f.scale([light_width/2, light_height/2, 1]),
            },
        },
    }
    
    # ============================================
    # 環境光 (Domain Randomization, 預設關閉)
    # ============================================
    ambient_config = DATA_CONFIG.get('ambient_light', {})
    if ambient_config.get('enabled', False):
        # 隨機環境光強度 (LED 強度的 0-10%)
        intensity_range = ambient_config.get('intensity_range', (0.0, 0.1))
        ambient_intensity = np.random.uniform(*intensity_range) * base_radiance
        
        # 隨機色溫
        color_temp_range = ambient_config.get('color_temp_range_k', (4000, 6500))
        color_temp = np.random.uniform(*color_temp_range)
        
        # 色溫轉 RGB (簡化近似)
        # 4000K: 暖色 (1.0, 0.85, 0.7)
        # 6500K: 日光 (1.0, 1.0, 1.0)
        temp_ratio = (color_temp - 4000) / 2500
        ambient_color = [
            ambient_intensity,
            ambient_intensity * (0.85 + 0.15 * temp_ratio),
            ambient_intensity * (0.7 + 0.3 * temp_ratio),
        ]
        
        # 加入環境光 (非偏振，均勻照明)
        scene_dict['ambient_light'] = {
            'type': 'constant',
            'radiance': {
                'type': 'rgb',
                'value': ambient_color,
            },
        }
    
    scene_dict.update({
        # ============================================
        # 透明玻璃板
        # ============================================
        'glass_panel': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.translate(glass_pos) @
                        mi.ScalarTransform4f.rotate([0, 1, 0], glass_rot[0]) @  # Y 軸旋轉
                        mi.ScalarTransform4f.rotate([1, 0, 0], glass_rot[1]) @  # X 軸旋轉
                        mi.ScalarTransform4f.scale([glass_size[0]/2, glass_size[1]/2, 1]),
            'bsdf': {
                'type': 'dielectric',
                'int_ior': SCENE_CONFIG['glass_ior'],
                'ext_ior': 1.0,
            },
        },
        
        # ============================================
        # 背景漫反射平面 (1:10 縮尺)
        # 工作距離 528-695mm，背景在 750mm
        # ============================================
        'background': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.translate([0, 0, 0.75]) @
                        mi.ScalarTransform4f.scale([0.2, 0.15, 1]),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'rgb',
                    'value': bg_color,
                },
            },
        },
        
        # ============================================
        # 地面 (250 × 200 mm 底座，1:10 縮尺)
        # ============================================
        'ground': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.translate([0, -0.12, 0.6]) @
                        mi.ScalarTransform4f.rotate([1, 0, 0], -90) @
                        mi.ScalarTransform4f.scale([0.125, 0.25, 1]),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'rgb',
                    'value': [0.35, 0.30, 0.25],
                },
            },
        },
        
        # ============================================
        # 左側牆壁
        # ============================================
        'wall_left': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.translate([-0.15, 0, 0.6]) @
                        mi.ScalarTransform4f.rotate([0, 1, 0], 90) @
                        mi.ScalarTransform4f.scale([0.25, 0.15, 1]),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'rgb',
                    'value': [0.6, 0.55, 0.5],
                },
            },
        },
        
        # ============================================
        # 右側牆壁
        # ============================================
        'wall_right': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.translate([0.15, 0, 0.6]) @
                        mi.ScalarTransform4f.rotate([0, 1, 0], -90) @
                        mi.ScalarTransform4f.scale([0.25, 0.15, 1]),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'rgb',
                    'value': [0.55, 0.6, 0.5],
                },
            },
        },
    }
    
    return mi.load_dict(scene_dict)


# ============================================================
# 偏振影像處理
# ============================================================

def extract_polarization_images(stokes_image):
    """
    從 Stokes vector 影像中提取 I∥ 和 I⊥
    
    根據 Malus's law:
    - I∥ = 0.5 * (S0 + S1)  # 0° 分析器
    - I⊥ = 0.5 * (S0 - S1)  # 90° 分析器
    
    Parameters:
    -----------
    stokes_image : np.ndarray
        Mitsuba 渲染的 Stokes vector 影像
        
    Returns:
    --------
    I_parallel : np.ndarray
        平行偏振影像 (I∥)
    I_cross : np.ndarray
        正交偏振影像 (I⊥)
    """
    
    # 檢查影像格式
    if len(stokes_image.shape) == 3:
        num_channels = stokes_image.shape[2]
        
        if num_channels >= 4:
            # Stokes vector: [S0, S1, S2, S3] 或 RGB 版本
            S0 = stokes_image[..., 0]
            S1 = stokes_image[..., 1]
            
            I_parallel = 0.5 * (S0 + S1)
            I_cross = 0.5 * (S0 - S1)
            
            # 確保非負
            I_parallel = np.maximum(I_parallel, 0)
            I_cross = np.maximum(I_cross, 0)
            
        elif num_channels == 3:
            # 標準 RGB (可能未啟用偏振模式)
            print("警告: 收到標準 RGB 輸出，可能未正確啟用偏振模式")
            I_parallel = stokes_image
            I_cross = stokes_image * 0.7  # 模擬衰減
            
        else:
            raise ValueError(f"未預期的通道數: {num_channels}")
    else:
        raise ValueError(f"未預期的影像維度: {stokes_image.shape}")
    
    return I_parallel, I_cross


def compute_polarization_difference(I_parallel, I_cross):
    """
    計算偏振差異圖
    
    這個差異主要出現在透明物體表面（鏡面反射區域）
    """
    diff = np.abs(I_parallel - I_cross)
    return diff


# ============================================================
# 深度圖渲染
# ============================================================

def render_depth_map(mi, scene, sensor_index=0):
    """
    渲染深度圖
    
    使用 Mitsuba 的 AOV integrator 獲取深度資訊
    """
    # 獲取場景參數
    params = mi.traverse(scene)
    
    # 建立深度 integrator
    depth_integrator = mi.load_dict({
        'type': 'aov',
        'aovs': 'dd.y:depth',
        'integrator': {
            'type': 'path',
            'max_depth': 2,
        }
    })
    
    # 渲染
    image = mi.render(scene, integrator=depth_integrator, sensor=sensor_index)
    depth = np.array(image)[..., 0]  # 取深度通道
    
    return depth


# ============================================================
# 主渲染函數
# ============================================================

def render_sample(mi, scene_params, output_dir, sample_id, save_intermediate=False):
    """
    渲染單一訓練樣本
    
    Parameters:
    -----------
    mi : module
        Mitsuba 模組
    scene_params : dict
        場景參數
    output_dir : Path
        輸出目錄
    sample_id : int
        樣本 ID
    save_intermediate : bool
        是否儲存中間結果
        
    Returns:
    --------
    dict : 包含所有輸出路徑的字典
    """
    import cv2
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 建構場景
    scene = build_scene(mi, scene_params)
    
    # 渲染左相機 (I∥)
    print(f"  渲染左相機 (I∥)...")
    image_left = mi.render(scene, sensor=0)
    stokes_left = np.array(image_left)
    
    # 渲染右相機 (I⊥)
    print(f"  渲染右相機 (I⊥)...")
    image_right = mi.render(scene, sensor=1)
    stokes_right = np.array(image_right)
    
    # 提取偏振分量
    I_parallel, _ = extract_polarization_images(stokes_left)
    _, I_cross = extract_polarization_images(stokes_right)
    
    # 渲染深度圖
    print(f"  渲染深度圖...")
    depth_left = render_depth_map(mi, scene, sensor_index=0)
    
    # 轉換深度為視差
    disp_info = compute_disparity_range()
    focal_px = disp_info['focal_length_pixels']
    baseline_mm = CAMERA_CONFIG['baseline_mm']
    
    # depth 單位是公尺，需要轉換
    depth_mm = depth_left * 1000
    disparity = np.where(depth_mm > 0, (baseline_mm * focal_px) / depth_mm, 0)
    
    # 儲存結果
    prefix = f"{sample_id:06d}"
    
    # EXR 格式 (HDR)
    cv2.imwrite(str(output_dir / f"{prefix}_I_parallel.exr"), 
                I_parallel.astype(np.float32))
    cv2.imwrite(str(output_dir / f"{prefix}_I_cross.exr"), 
                I_cross.astype(np.float32))
    cv2.imwrite(str(output_dir / f"{prefix}_disparity.exr"), 
                disparity.astype(np.float32))
    
    # 儲存場景參數
    with open(output_dir / f"{prefix}_params.json", 'w') as f:
        # 轉換 numpy 類型為 Python 原生類型
        params_serializable = {
            k: v.tolist() if isinstance(v, np.ndarray) else v 
            for k, v in scene_params.items()
        }
        json.dump(params_serializable, f, indent=2)
    
    # 可選：儲存預覽圖 (PNG)
    if save_intermediate:
        def to_uint8(img):
            img = np.clip(img, 0, None)
            if img.max() > 0:
                img = img / np.percentile(img, 99)
            return (np.clip(img, 0, 1) * 255).astype(np.uint8)
        
        cv2.imwrite(str(output_dir / f"{prefix}_I_parallel_preview.png"), 
                    to_uint8(I_parallel))
        cv2.imwrite(str(output_dir / f"{prefix}_I_cross_preview.png"), 
                    to_uint8(I_cross))
        
        # 視差偽彩色圖
        disp_normalized = (disparity / disparity.max() * 255).astype(np.uint8)
        disp_colored = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_JET)
        cv2.imwrite(str(output_dir / f"{prefix}_disparity_preview.png"), disp_colored)
    
    return {
        'I_parallel': output_dir / f"{prefix}_I_parallel.exr",
        'I_cross': output_dir / f"{prefix}_I_cross.exr",
        'disparity': output_dir / f"{prefix}_disparity.exr",
        'params': output_dir / f"{prefix}_params.json",
    }


# ============================================================
# 場景參數隨機化
# ============================================================

def randomize_scene_params(seed=None):
    """
    隨機生成場景參數
    
    Parameters:
    -----------
    seed : int, optional
        隨機種子
        
    Returns:
    --------
    dict : 場景參數字典
    """
    if seed is not None:
        np.random.seed(seed)
    
    cfg = DATA_CONFIG
    
    # 玻璃位置 (轉換為公尺)
    glass_position = [
        np.random.uniform(*cfg['glass_position_range']['x_mm']) / 1000,
        np.random.uniform(*cfg['glass_position_range']['y_mm']) / 1000,
        np.random.uniform(*cfg['glass_position_range']['z_mm']) / 1000,
    ]
    
    # 玻璃旋轉
    glass_rotation = [
        np.random.uniform(*cfg['glass_rotation_range']['y_deg']),
        np.random.uniform(*cfg['glass_rotation_range']['x_deg']),
    ]
    
    # 玻璃尺寸 (轉換為公尺)
    glass_size = [
        np.random.uniform(*cfg['glass_size_range']['width_mm']) / 1000,
        np.random.uniform(*cfg['glass_size_range']['height_mm']) / 1000,
    ]
    
    # 背景顏色
    bg_range = cfg['background_color_range']
    background_color = [
        np.random.uniform(bg_range[0], bg_range[1]),
        np.random.uniform(bg_range[0], bg_range[1]),
        np.random.uniform(bg_range[0], bg_range[1]),
    ]
    
    # 光源強度
    light_intensity = np.random.uniform(*cfg['light_intensity_range'])
    
    return {
        'glass_position': glass_position,
        'glass_rotation': glass_rotation,
        'glass_size': glass_size,
        'background_color': background_color,
        'light_intensity': light_intensity,
    }


# ============================================================
# 批次生成
# ============================================================

def generate_dataset(num_samples, output_dir, start_id=0, save_previews=False):
    """
    批次生成訓練數據
    
    Parameters:
    -----------
    num_samples : int
        樣本數量
    output_dir : str
        輸出目錄
    start_id : int
        起始 ID
    save_previews : bool
        是否儲存預覽圖
    """
    from tqdm import tqdm
    
    mi = init_mitsuba()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n開始生成 {num_samples} 個訓練樣本...")
    print(f"輸出目錄: {output_dir}")
    
    # 記錄開始時間
    start_time = time.time()
    
    for i in tqdm(range(num_samples), desc="生成訓練數據"):
        sample_id = start_id + i
        
        # 隨機化場景參數
        scene_params = randomize_scene_params(seed=sample_id)
        
        # 渲染
        try:
            render_sample(
                mi, 
                scene_params, 
                output_dir, 
                sample_id,
                save_intermediate=save_previews
            )
        except Exception as e:
            print(f"\n警告: 樣本 {sample_id} 渲染失敗: {e}")
            continue
    
    # 統計
    elapsed = time.time() - start_time
    avg_time = elapsed / num_samples
    
    print(f"\n生成完成!")
    print(f"總耗時: {elapsed:.1f} 秒")
    print(f"平均每樣本: {avg_time:.2f} 秒")
    
    # 儲存數據集元資料
    metadata = {
        'num_samples': num_samples,
        'start_id': start_id,
        'resolution': CAMERA_CONFIG['output_resolution'],
        'baseline_mm': CAMERA_CONFIG['baseline_mm'],
        'fov_horizontal_deg': CAMERA_CONFIG['fov_horizontal_deg'],
        'generation_time_sec': elapsed,
        'config': {
            'camera': CAMERA_CONFIG,
            'light': LIGHT_CONFIG,
            'polarization': POLARIZATION_CONFIG,
            'scene': SCENE_CONFIG,
        }
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"元資料已儲存至: {output_dir / 'metadata.json'}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='PIDS 訓練數據生成器 (Mitsuba 3 偏振渲染)'
    )
    
    parser.add_argument(
        '--num_samples', '-n',
        type=int,
        default=10,
        help='生成樣本數量 (預設: 10)'
    )
    
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='./training_data',
        help='輸出目錄 (預設: ./training_data)'
    )
    
    parser.add_argument(
        '--start_id',
        type=int,
        default=0,
        help='起始樣本 ID (預設: 0)'
    )
    
    parser.add_argument(
        '--save_previews',
        action='store_true',
        help='儲存 PNG 預覽圖'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='測試模式: 只渲染一張圖並顯示結果'
    )
    
    args = parser.parse_args()
    
    if args.test:
        # 測試模式
        print("=" * 60)
        print("PIDS 渲染器測試模式")
        print("=" * 60)
        
        mi = init_mitsuba()
        
        # 使用預設參數渲染一張
        scene_params = {
            'glass_position': [0, 0, 0.6],  # 600mm 工作距離
            'glass_rotation': [15, 5],
            'glass_size': [0.05, 0.08],  # 50×80mm 玻璃板
            'background_color': [0.5, 0.5, 0.5],
            'light_intensity': 1.0,
            'samples': RENDER_CONFIG['samples_preview'],
        }
        
        output_dir = Path('./test_output')
        render_sample(mi, scene_params, output_dir, 0, save_intermediate=True)
        
        print(f"\n測試完成! 輸出位於: {output_dir}")
        
    else:
        # 批次生成模式
        generate_dataset(
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            start_id=args.start_id,
            save_previews=args.save_previews
        )


if __name__ == '__main__':
    main()
