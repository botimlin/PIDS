"""
PIDS (Physics-Informed Deep Stereo) 系統配置 (修正版)
=============================================

此檔案定義所有硬體參數和渲染設定。
修正項目: 加入 RENDER_CONFIG['spp'] 參數以支援新版渲染器。
"""

import numpy as np

# ============================================================
# 1. 相機配置 (Sony IMX296LQR-C)
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
    'focus_distance_mm': 600.0,
    'f_number': 1.2,
}

# ============================================================
# 2. 光源配置 (LED 面板)
# ============================================================

LIGHT_CONFIG = {
    # 光源類型
    'type': 'polarized_panel',
    
    # 光源位置 (相對於世界原點)
    # 放在左上方打光，製造偏振反射
    'position_mm': (-150, 0, 200), 
    'target_mm': (0, 600, 0),
    
    # 強度 (Watts/sr)
    'intensity_watts_sr': 50.0,
    
    # 物理偏振角度 (相對於垂直線)
    # 90度 = 水平偏振 (S-polarization 對於桌面)
    'polarization_angle_deg': 90.0,
    
    # 入射角 (接近 Brewster Angle 56度效果最好)
    'incidence_angle_deg': 55.0,
    
    # 色溫範圍 (用於隨機化)
    'cct_range_k': (4000, 6500),
    
    # 面板尺寸 (用於面積光模擬)
    'panel_width_mm': 300,
    'panel_height_mm': 300,
}

# ============================================================
# 3. 偏振相機配置 (Polarization)
# ============================================================

POLARIZATION_CONFIG = {
    # 左相機 (I_parallel): 分析器與光源平行 (看穿/最強反射)
    'left_analyzer_angle_deg': 90.0,
    
    # 右相機 (I_cross): 分析器與光源垂直 (過濾反射)
    'right_analyzer_angle_deg': 0.0,
    
    # 偏振片效率 (0.0~1.0)
    'extinction_ratio': 1000.0, # 1000:1
}

# ============================================================
# 4. 場景幾何配置 (Scene)
# ============================================================

SCENE_CONFIG = {
    # 單位比例 (Blender Unit -> Meters)
    # 1 Unit = 1mm = 0.001m
    'unit_scale': 0.001,
    
    # 地面與背景牆位置
    'ground_y_mm': 600.0,
    'wall_z_mm': 750.0,
}

# ============================================================
# 5. 渲染設定 (Render) - [修正重點]
# ============================================================

RENDER_CONFIG = {
    # [新增] 採樣數 (Samples Per Pixel)
    # 64: 快速預覽, 128: 標準, 256+: 高畫質
    'spp': 64,
    
    # 光線彈射次數
    'max_depth': 8,
    
    # 積分器類型
    'integrator': 'path',
}

# ============================================================
# 6. 數據生成配置 (Data)
# ============================================================

DATA_CONFIG = {
    'dataset_name': 'PIDS_Synthetic_v1',
    'train_split': 0.8,
    'val_split': 0.1,
    'test_split': 0.1,
}

# ============================================================
# 輔助函式
# ============================================================

def compute_disparity_range():
    """
    計算視差範圍 (Disparity Range)
    formula: d = (f * B) / Z
    """
    f_mm = CAMERA_CONFIG['focal_length_mm']
    B_mm = CAMERA_CONFIG['baseline_mm']
    
    # 感測器寬度 (mm)
    sensor_w = CAMERA_CONFIG['sensor_width_mm']
    # 輸出寬度 (pixels)
    img_w = CAMERA_CONFIG['output_resolution'][0]
    
    # 焦距轉換為像素單位
    # f_pix = f_mm * (img_w / sensor_w)
    f_pixel = f_mm * (img_w / sensor_w)
    
    Z_min = CAMERA_CONFIG['min_distance_mm']
    Z_max = CAMERA_CONFIG['max_distance_mm']
    
    max_disp = (f_pixel * B_mm) / Z_min
    min_disp = (f_pixel * B_mm) / Z_max
    
    return {
        'focal_length_pixels': f_pixel,
        'min_disparity': min_disp,
        'max_disparity': max_disp,
        'disparity_range': max_disp - min_disp
    }

# ============================================================
# 自我測試與資訊顯示
# ============================================================

if __name__ == "__main__":
    disp_info = compute_disparity_range()
    
    print("="*60)
    print("PIDS 系統配置資訊")
    print("="*60)
    
    print("\n【相機參數】")
    print(f"  型號: {CAMERA_CONFIG['sensor_name']}")
    print(f"  焦距: {CAMERA_CONFIG['focal_length_mm']} mm")
    print(f"  基線: {CAMERA_CONFIG['baseline_mm']} mm")
    print(f"  解析度: {CAMERA_CONFIG['output_resolution']}")
    print(f"  FOV (H/V): {CAMERA_CONFIG['fov_horizontal_deg']}° / {CAMERA_CONFIG['fov_vertical_deg']}°")
    
    print("\n【光源參數】")
    print(f"  強度: {LIGHT_CONFIG['intensity_watts_sr']} W/sr")
    print(f"  位置: {LIGHT_CONFIG['position_mm']} mm")
    print(f"  偏振角: {LIGHT_CONFIG['polarization_angle_deg']}°")
    
    print("\n【渲染參數】")
    print(f"  SPP (採樣數): {RENDER_CONFIG['spp']}")
    print(f"  Max Depth: {RENDER_CONFIG['max_depth']}")
    
    print("\n【視差計算】")
    print(f"  等效焦距: {disp_info['focal_length_pixels']:.1f} px")
    print(f"  工作距離: {CAMERA_CONFIG['min_distance_mm']} ~ {CAMERA_CONFIG['max_distance_mm']} mm")
    print(f"  視差範圍: {disp_info['min_disparity']:.1f} ~ {disp_info['max_disparity']:.1f} px")
    print("="*60)
