"""
PIDS Mitsuba 渲染配置檔案
=========================

此檔案包含所有可調整的渲染參數。
修改後導入 pids_mitsuba_renderer.py 使用。

Stage I 設定: 無環境光，只有偏振 LED 光源
"""

# ============================================================
# 渲染配置
# ============================================================

RENDER_CONFIG = {
    # 基本渲染設定
    'width': 640,                # 輸出寬度 (pixels)
    'height': 480,               # 輸出高度 (pixels)
    'spp': 256,                  # 每像素樣本數 (品質控制)
    'max_depth': 8,              # 光線最大反彈次數
    
    # 品質預設
    # 'draft': spp=64, 快速預覽
    # 'normal': spp=256, 一般訓練
    # 'high': spp=512, 高品質
    # 'ultra': spp=1024, 最高品質
}


# ============================================================
# 相機配置 (單位: mm)
# ============================================================

CAMERA_CONFIG = {
    # 立體相機參數
    'baseline': 65.0,            # 雙相機基線距離 (mm)
    'fov': 65.0,                 # 水平視場角 (度)
    'focus_distance': 600.0,     # 對焦距離 (mm)
    
    # 相機位置 (Mitsuba 座標系)
    # 注意: Blender Y → Mitsuba Z
    'position_y': 150.0,         # 相機高度 (mm) - chamber 高度的一半
    'position_z': 0.0,           # 相機深度位置 (mm) - 原點
    
    # 景深參數 (F1.2 光圈計算)
    'aperture': 1.2,             # 光圈值
    'dof_enabled': False,        # 是否啟用景深模擬 (訓練時通常關閉)
}


# ============================================================
# Chamber 尺寸 (來自 Blender 腳本)
# ============================================================

CHAMBER_CONFIG = {
    # 房間尺寸 (mm)
    'width': 600.0,              # X 方向 (左右)
    'height': 300.0,             # Y 方向 (上下, Mitsuba)
    
    # 深度範圍 (Blender Y → Mitsuba Z)
    'z_start': 350.0,            # 前牆位置 (mm)
    'z_end': 900.0,              # 後牆位置 (mm)
    
    # 景深範圍 (透明物體應在此範圍內)
    'dof_near': 528.0,           # 景深近端 (mm)
    'dof_far': 695.0,            # 景深遠端 (mm)
    
    # 傢俱區域
    'furniture_z_min': 700.0,    # 傢俱最近距離 (mm)
    'furniture_z_max': 800.0,    # 傢俱最遠距離 (mm)
}


# ============================================================
# 光源配置 (Stage I)
# ============================================================

LIGHTING_CONFIG = {
    # ========================================
    # 主光源: 偏振 LED
    # ========================================
    'led': {
        'enabled': True,
        'intensity': 5.0,            # 光源強度 (radiance)
        'color': [1.0, 1.0, 1.0],    # 色溫 (白光)
        
        # 光源位置 (相對於相機中心)
        # 入射角 ~55° 以接近玻璃的 Brewster angle
        # Brewster angle for glass (n=1.5): arctan(1.5) ≈ 56.3°
        'offset_y': 200.0,           # 上方偏移 (mm)
        'offset_z': -100.0,          # 後方偏移 (mm)
        
        # 面光源尺寸 (模擬實際 LED 面板)
        'size_x': 300.0,             # 寬度 (mm)
        'size_y': 200.0,             # 高度 (mm)
        
        # 偏振方向 (0° = 水平)
        'polarization_angle': 0.0,
    },
    
    # ========================================
    # 輔助光源 (可選)
    # ========================================
    'fill_light': {
        'enabled': False,            # Stage I 關閉
        'intensity': 1.0,
        'position_offset': [100, 50, -50],  # X, Y, Z offset (mm)
    },
    
    # ========================================
    # 環境光 (Stage I 關閉)
    # ========================================
    'ambient': {
        'enabled': False,            # Stage I: False, Stage II: True
        'intensity': 0.0,            # Stage I: 0, Stage II: 0.05-0.1
        'color_temperature_k': 5500, # 色溫 (K)
    },
}


# ============================================================
# 材質配置
# ============================================================

MATERIAL_CONFIG = {
    # 玻璃材質
    'glass': {
        'ior': 1.5,                  # 折射率 (標準玻璃)
        'absorption': [0.0, 0.0, 0.0],  # 吸收係數 (透明)
    },
    
    # 壓克力材質
    'acrylic': {
        'ior': 1.49,                 # 折射率 (PMMA)
        'absorption': [0.0, 0.0, 0.0],
    },
    
    # 偏振模擬參數
    'polarization': {
        # 偏振片透射係數
        # 理想偏振片: α = 0.5
        # 實際偏振片: α ≈ 0.42-0.48
        'alpha': 0.5,
        
        # 交叉偏振鏡面抑制率
        # 理想情況: 完全抑制
        # 實際情況: 殘餘 ~5%
        'specular_suppression': 0.05,
        
        # 用於模擬交叉偏振的粗糙度增加
        'cross_roughness_boost': 0.3,
    },
    
    # 漫反射材質預設顏色
    'diffuse_colors': {
        'wall': [0.75, 0.70, 0.65],      # 米色牆壁
        'ground': [0.45, 0.40, 0.35],    # 木地板
        'ceiling': [0.85, 0.83, 0.80],   # 淺色天花板
        'furniture': [0.40, 0.30, 0.20], # 傢俱
    },
}


# ============================================================
# 輸出配置
# ============================================================

OUTPUT_CONFIG = {
    # 檔案格式
    'image_format': 'exr',           # 主要輸出: EXR (32-bit HDR)
    'depth_format': 'exr',           # 深度圖: EXR (32-bit float)
    'preview_format': 'png',         # 預覽圖: PNG (8-bit)
    
    # 輸出選項
    'save_preview': True,            # 儲存預覽 PNG
    'save_params_json': True,        # 儲存參數 JSON
    'save_polarization_diff': True,  # 儲存偏振差異圖
    
    # 影像格式
    'grayscale': True,               # 灰階輸出 (單通道)
    'pixel_format': 'luminance',     # Mitsuba pixel format
    
    # 檔案命名
    'naming': {
        'parallel': 'I_parallel',    # 平行偏振影像後綴
        'cross': 'I_cross',          # 交叉偏振影像後綴
        'depth': 'depth',            # 深度圖後綴
        'disparity': 'disparity',    # 視差圖後綴
    },
}


# ============================================================
# 視差計算參數
# ============================================================

DISPARITY_CONFIG = {
    # 視差範圍 (根據景深計算)
    # disparity = (focal_length × baseline) / depth
    # 
    # 以 640×480, FOV=65°, baseline=65mm 計算:
    # focal_length ≈ 502 pixels
    # 
    # depth=528mm → disparity ≈ 62 px
    # depth=600mm → disparity ≈ 54 px
    # depth=695mm → disparity ≈ 47 px
    # depth=900mm → disparity ≈ 36 px
    
    'expected_range': {
        'min_px': 35,                # 最小視差 (背景)
        'max_px': 65,                # 最大視差 (前景)
    },
    
    # 無效值處理
    'invalid_value': 0.0,            # 無效區域的視差值
}


# ============================================================
# Stage 配置預設
# ============================================================

def get_stage1_config():
    """取得 Stage I 配置 (無環境光)"""
    config = {
        'render': RENDER_CONFIG.copy(),
        'camera': CAMERA_CONFIG.copy(),
        'chamber': CHAMBER_CONFIG.copy(),
        'lighting': LIGHTING_CONFIG.copy(),
        'material': MATERIAL_CONFIG.copy(),
        'output': OUTPUT_CONFIG.copy(),
        'disparity': DISPARITY_CONFIG.copy(),
    }
    
    # Stage I 特定設定
    config['lighting']['ambient']['enabled'] = False
    config['lighting']['ambient']['intensity'] = 0.0
    
    return config


def get_stage2_config():
    """取得 Stage II 配置 (有環境光)"""
    config = get_stage1_config()
    
    # Stage II 特定設定
    config['lighting']['ambient']['enabled'] = True
    config['lighting']['ambient']['intensity'] = 0.05  # LED 強度的 5%
    
    return config


# ============================================================
# 輔助函數
# ============================================================

def print_config_summary(config: dict):
    """印出配置摘要"""
    print("\n" + "="*60)
    print("PIDS 渲染配置摘要")
    print("="*60)
    
    print(f"\n[渲染設定]")
    print(f"  解析度: {config['render']['width']} × {config['render']['height']}")
    print(f"  SPP: {config['render']['spp']}")
    
    print(f"\n[相機設定]")
    print(f"  基線: {config['camera']['baseline']} mm")
    print(f"  FOV: {config['camera']['fov']}°")
    
    print(f"\n[Chamber]")
    print(f"  尺寸: {config['chamber']['width']} × {config['chamber']['height']} mm")
    print(f"  深度: {config['chamber']['z_start']} - {config['chamber']['z_end']} mm")
    print(f"  景深: {config['chamber']['dof_near']} - {config['chamber']['dof_far']} mm")
    
    print(f"\n[光源]")
    print(f"  LED: {'啟用' if config['lighting']['led']['enabled'] else '停用'}")
    print(f"  環境光: {'啟用' if config['lighting']['ambient']['enabled'] else '停用'}")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    # 測試配置
    config = get_stage1_config()
    print_config_summary(config)
