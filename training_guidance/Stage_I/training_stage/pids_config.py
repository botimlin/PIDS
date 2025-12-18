"""
PIDS (Physics-Informed Deep Stereo) 系統配置
=============================================

此檔案定義所有硬體參數和渲染設定。
更新日期: 2024
"""

# ============================================================
# 相機配置 (Sony IMX296LQR-C)
# ============================================================

CAMERA_CONFIG = {
    # 感測器原始規格
    'sensor_name': 'Sony IMX296LQR-C',
    'sensor_diagonal_mm': 6.3,
    'pixel_size_um': 3.45,
    'native_resolution': (1456, 1088),
    
    # 輸出解析度 (Resize 模式 - 使用全感測器)
    'output_resolution': (640, 480),
    
    # 感測器實際使用尺寸 (全感測器)
    'sensor_width_mm': 1456 * 0.00345,   # 5.0232 mm
    'sensor_height_mm': 1088 * 0.00345,  # 3.7536 mm
    
    # 鏡頭
    'focal_length_mm': 6.0,
    
    # 計算得出的 FOV (度)
    'fov_horizontal_deg': 45.4,
    'fov_vertical_deg': 34.7,
    
    # 立體配置
    'baseline_mm': 30.0,
    
    # 工作範圍 (F1.2 光圈景深限制)
    'min_distance_mm': 528.0,
    'max_distance_mm': 695.0,
    
    # 對焦距離
    'focus_distance_mm': 600.0,
}

# ============================================================
# 光源配置 (Godox Litemons C30Bi)
# ============================================================

LIGHT_CONFIG = {
    'model': 'Godox Litemons C30Bi',
    'power_w': 30,
    'lux_at_0_5m': 8610,
    'cct_range_k': (2800, 6500),
    'cri': 94,
    'tlci': 96,
    
    # 面板尺寸
    'panel_width_mm': 135,
    'panel_height_mm': 79,
    
    # 偏振片配置
    'source_polarizer_angle_deg': 0,  # 光源端偏振片角度
    
    # 預設色溫 (用於模擬)
    'default_cct_k': 5600,
    
    # 光源位置 (相對於相機原點)
    'position_offset_mm': {
        'x': 0,      # 水平置中
        'y': 120,    # 相機上方 120mm
        'z': -80,    # 相機後方 80mm
    },
    
    # 入射角度 (Brewster angle = 56.3°, 使用 55° 方便調整)
    'incidence_angle_deg': 55,
}

# ============================================================
# 偏振配置
# ============================================================

POLARIZATION_CONFIG = {
    # 左相機 (I∥): 平行偏振，保留鏡面反射
    'left_analyzer_angle_deg': 0,
    
    # 右相機 (I⊥): 正交偏振，抑制鏡面反射  
    'right_analyzer_angle_deg': 90,
    
    # 偏振片透射係數 (理想值約 0.5)
    'polarizer_transmission': 0.5,
}

# ============================================================
# 場景配置 (1:10 縮尺居家場景)
# ============================================================

SCENE_CONFIG = {
    'scale_ratio': 0.1,  # 1:10 縮尺
    
    # 玻璃材質
    'glass_ior': 1.5,  # 標準玻璃折射率
    'glass_thickness_mm': 0.5,  # 模型玻璃厚度 (真實 5mm)
    
    # 場景範圍 (mm) - 模型尺寸
    'scene_depth_range_mm': (528, 695),  # 工作距離 (F1.2 景深限制)
    'scene_width_mm': 250,   # 對應真實 2.5m
    'scene_height_mm': 200,  # 對應真實 2m
    'scene_depth_mm': 200,   # 場景深度 ~170mm + 餘量
    
    # 3D 列印底座尺寸
    'print_bed_size_mm': 256,
    
    # 對應真實世界 (僅供參考)
    'real_world_equiv': {
        'distance_m': (5.3, 6.9),   # 真實工作距離
        'fov_width_m': (4.4, 5.8),  # 真實視野寬度
        'baseline_mm': 300,         # 等效真實基線
    },
}

# ============================================================
# 渲染配置
# ============================================================

RENDER_CONFIG = {
    # Mitsuba variant
    'mitsuba_variant': 'cuda_ad_rgb_polarized',
    'fallback_variant': 'scalar_rgb_polarized',
    
    # 採樣數 (品質 vs 速度權衡)
    'samples_preview': 64,      # 預覽用
    'samples_training': 256,    # 訓練數據
    'samples_high_quality': 1024,  # 高品質驗證
    
    # 路徑追蹤深度
    'max_depth': 8,  # 足夠處理玻璃的多次反射/折射
    
    # 輸出格式
    'output_format': 'exr',  # HDR 格式保留完整動態範圍
    'output_dtype': 'float32',
}

# ============================================================
# 數據生成配置
# ============================================================

DATA_CONFIG = {
    # 輸出目錄
    'output_dir': './training_data',
    
    # 數據集大小 (可調整)
    'num_training_samples': 5000,
    'num_validation_samples': 500,
    'num_test_samples': 500,
    
    # 場景隨機化範圍 (1:10 縮尺，單位 mm)
    # 工作距離: 528-695mm
    'glass_position_range': {
        'x_mm': (-80, 80),     # 配合 250mm 場景寬度
        'y_mm': (-60, 60),     # 配合 200mm 場景高度
        'z_mm': (528, 695),    # 工作距離範圍 (F1.2 景深)
    },
    'glass_rotation_range': {
        'y_deg': (-30, 30),    # 水平旋轉
        'x_deg': (-15, 15),    # 俯仰
    },
    'glass_size_range': {
        'width_mm': (5, 60),   # 模型尺寸 (真實 50-600mm)
        'height_mm': (5, 80),  # 模型尺寸 (真實 50-800mm)
    },
    'light_intensity_range': (0.5, 1.5),  # 相對強度
    'background_color_range': (0.2, 0.8),  # RGB 各通道
    
    # ========================================
    # 環境光配置 (Domain Randomization)
    # ========================================
    # Phase 1: 關閉環境光，驗證系統可行性
    # Phase 2: 開啟環境光，提升泛化能力
    'ambient_light': {
        'enabled': False,  # 預設關閉
        'intensity_range': (0.0, 0.1),  # LED 強度的 0-10%
        'color_temp_range_k': (4000, 6500),  # 色溫範圍
    },
    
    # 1:10 縮尺參考 (模型 → 真實)
    # 訓練策略：優先使用大型平面透明物體
    'scale_reference': {
        # 大型透明物體（訓練用）
        'glass_door_model_mm': (90, 150),     # 真實 0.9 × 1.5 m
        'glass_window_model_mm': (100, 80),   # 真實 1.0 × 0.8 m
        'glass_partition_model_mm': (50, 40), # 真實 0.5 × 0.4 m
        'glass_table_model_mm': (80, 80),     # 真實 0.8 × 0.8 m
        'glass_cabinet_model_mm': (40, 60),   # 真實 0.4 × 0.6 m
        'glass_wall_model_mm': (120, 100),    # 真實 1.2 × 1.0 m
        
        # 背景家具
        'bookshelf_model_mm': 80,     # 真實 800mm
        'door_height_model_mm': 210,  # 真實 2100mm (2.1m)
        
        # 小型透明物體（真實部署測試用，模擬階段不需要）
        # 'glass_cup_model_mm': 7,    # 留待 1:1 真實測試
        # 'glass_bottle_model_mm': 8, # 留待 1:1 真實測試
    },
}

# ============================================================
# 視差計算
# ============================================================

def compute_disparity_range():
    """
    計算視差範圍
    
    disparity (pixels) = baseline * focal_length / depth
    
    注意：這裡的 focal_length 需要轉換為像素單位
    """
    # 原始感測器的 focal length in pixels
    focal_length_pixels_native = (
        CAMERA_CONFIG['focal_length_mm'] / 
        (CAMERA_CONFIG['pixel_size_um'] / 1000)
    )  # 6 / 0.00345 = 1739.13 pixels
    
    # Resize 後的等效 focal length
    scale_x = CAMERA_CONFIG['output_resolution'][0] / CAMERA_CONFIG['native_resolution'][0]
    focal_length_pixels = focal_length_pixels_native * scale_x  # 1739.13 * (640/1456) = 764.4 pixels
    
    baseline_mm = CAMERA_CONFIG['baseline_mm']
    
    # 最大視差 (最近距離)
    min_depth_mm = CAMERA_CONFIG['min_distance_mm']
    max_disparity = (baseline_mm * focal_length_pixels) / min_depth_mm
    
    # 最小視差 (最遠距離)
    max_depth_mm = CAMERA_CONFIG['max_distance_mm']
    min_disparity = (baseline_mm * focal_length_pixels) / max_depth_mm
    
    return {
        'focal_length_pixels': focal_length_pixels,
        'min_disparity': min_disparity,
        'max_disparity': max_disparity,
        'disparity_range': max_disparity - min_disparity,
    }


def depth_to_disparity(depth_mm):
    """將深度 (mm) 轉換為視差 (pixels)"""
    info = compute_disparity_range()
    return (CAMERA_CONFIG['baseline_mm'] * info['focal_length_pixels']) / depth_mm


def disparity_to_depth(disparity):
    """將視差 (pixels) 轉換為深度 (mm)"""
    info = compute_disparity_range()
    return (CAMERA_CONFIG['baseline_mm'] * info['focal_length_pixels']) / disparity


# ============================================================
# 打印配置摘要
# ============================================================

def print_config_summary():
    """打印配置摘要"""
    disp_info = compute_disparity_range()
    
    print("=" * 60)
    print("PIDS 系統配置摘要")
    print("=" * 60)
    
    print("\n【相機配置】")
    print(f"  感測器: {CAMERA_CONFIG['sensor_name']}")
    print(f"  原始解析度: {CAMERA_CONFIG['native_resolution'][0]} × {CAMERA_CONFIG['native_resolution'][1]}")
    print(f"  輸出解析度: {CAMERA_CONFIG['output_resolution'][0]} × {CAMERA_CONFIG['output_resolution'][1]}")
    print(f"  焦距: {CAMERA_CONFIG['focal_length_mm']} mm")
    print(f"  FOV: {CAMERA_CONFIG['fov_horizontal_deg']:.1f}° × {CAMERA_CONFIG['fov_vertical_deg']:.1f}°")
    print(f"  基線: {CAMERA_CONFIG['baseline_mm']} mm")
    
    print("\n【光源配置】")
    print(f"  型號: {LIGHT_CONFIG['model']}")
    print(f"  功率: {LIGHT_CONFIG['power_w']}W")
    print(f"  亮度: {LIGHT_CONFIG['lux_at_0_5m']} Lux @ 0.5m")
    print(f"  色溫範圍: {LIGHT_CONFIG['cct_range_k'][0]}K - {LIGHT_CONFIG['cct_range_k'][1]}K")
    print(f"  面板尺寸: {LIGHT_CONFIG['panel_width_mm']} × {LIGHT_CONFIG['panel_height_mm']} mm")
    
    print("\n【偏振配置】")
    print(f"  光源偏振: {POLARIZATION_CONFIG['left_analyzer_angle_deg']}°")
    print(f"  左相機 (I∥): {POLARIZATION_CONFIG['left_analyzer_angle_deg']}° 分析器")
    print(f"  右相機 (I⊥): {POLARIZATION_CONFIG['right_analyzer_angle_deg']}° 分析器")
    print(f"  光源入射角: {LIGHT_CONFIG['incidence_angle_deg']}° (Brewster ≈ 56.3°)")
    
    print("\n【視差範圍】")
    print(f"  等效焦距: {disp_info['focal_length_pixels']:.1f} pixels")
    print(f"  工作距離: {CAMERA_CONFIG['min_distance_mm']} - {CAMERA_CONFIG['max_distance_mm']} mm")
    print(f"  視差範圍: {disp_info['min_disparity']:.1f} - {disp_info['max_disparity']:.1f} pixels")
    print(f"  視差跨度: {disp_info['disparity_range']:.1f} pixels")
    
    print("\n【縮尺配置】")
    print(f"  縮尺比例: 1:{int(1/SCENE_CONFIG['scale_ratio'])}")
    print(f"  場景尺寸: {SCENE_CONFIG['scene_width_mm']} × {SCENE_CONFIG['scene_height_mm']} mm (模型)")
    print(f"  對應真實: {SCENE_CONFIG['scene_width_mm']/SCENE_CONFIG['scale_ratio']/1000:.1f} × {SCENE_CONFIG['scene_height_mm']/SCENE_CONFIG['scale_ratio']/1000:.1f} m")
    print(f"  等效工作距離: {CAMERA_CONFIG['min_distance_mm']/SCENE_CONFIG['scale_ratio']/1000:.1f} - {CAMERA_CONFIG['max_distance_mm']/SCENE_CONFIG['scale_ratio']/1000:.1f} m")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    print_config_summary()