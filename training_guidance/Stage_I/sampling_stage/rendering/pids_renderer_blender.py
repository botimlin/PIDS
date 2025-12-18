"""
PIDS Mitsuba 3 偏振渲染器 (Blender 場景專用版)
================================================
核心邏輯：
1. 智慧材質分配: 
   - 偵測 *_glass.obj -> 賦予 Dielectric (BK7 玻璃)
   - 偵測 *_diffuse.obj -> 賦予 Roughplastic (漫反射物件)
2. 固定光學幾何 (符合 blender_guide.md):
   - 相機固定於 (0,0,0)，對焦於 (0,600,0)
   - 光源固定於 Brewster Angle (約 56.3度)
3. 物理一致性色彩:
   - 藍色增強矩陣 (修正黃色色偏)
   - 聯合白平衡 (Joint White Balance, 保留偏振強度差)

使用方法:
    python pids_renderer_blender.py --scene scenes/scene_00001
"""

import argparse
import os
import sys
import time
import numpy as np
from pathlib import Path

# 導入配置
try:
    from pids_config import (
        CAMERA_CONFIG, 
        LIGHT_CONFIG, 
        POLARIZATION_CONFIG
    )
except ImportError:
    print("錯誤: 找不到 pids_config.py，請確保配置檔存在。")
    sys.exit(1)

# ===========================================================
# 1. Mitsuba 初始化
# ===========================================================
MI = None

def init_mitsuba():
    global MI
    if MI is not None: return MI
    
    import mitsuba as mi
    # 設定變體：純量光譜偏振 (scalar_spectral_polarized)
    mi.set_variant('scalar_spectral_polarized')
    MI = mi
    return MI

# ===========================================================
# 2. 場景建構 (智慧材質分配)
# ===========================================================
def create_scene_dict(diffuse_obj_path, glass_obj_path):
    """
    建立符合 PIDS 規範的 Mitsuba 場景
    """
    
    # 1. 檢查檔案存在性
    objs_to_load = []
    if diffuse_obj_path and os.path.exists(diffuse_obj_path):
        objs_to_load.append(diffuse_obj_path)
    if glass_obj_path and os.path.exists(glass_obj_path):
        objs_to_load.append(glass_obj_path)
        
    if not objs_to_load:
        raise FileNotFoundError(f"找不到任何 OBJ 檔案。\n檢查路徑:\n  {diffuse_obj_path}\n  {glass_obj_path}")

    # 2. 定義幾何中心 (Reference Point) - 符合 README
    scene_center_y = 600.0
    cam_origin = [0, 0, 0]
    cam_target = [0, scene_center_y, 0]
    cam_up     = [0, 0, 1]

    # 3. 光源設定
    incidence_angle = LIGHT_CONFIG.get('incidence_angle_deg', 56.3)
    light_dist = 400.0 
    
    # 建立基礎場景字典
    scene_dict = {
        "type": "scene",
        
        # --- 整合器 (Path Tracer) ---
        "integrator": {
            "type": "stokes",
            "integrator": {
                "type": "path",
                "max_depth": 12, # 足夠的深度以穿透多層玻璃
            }
        },
        
        # --- 感測器 (Camera) ---
        "sensor": {
            "type": "perspective",
            "fov": CAMERA_CONFIG['fov_vertical_deg'], 
            "fov_axis": "y",
            "to_world": MI.ScalarTransform4f.look_at(
                origin=cam_origin,
                target=cam_target,
                up=cam_up
            ),
            "sampler": {
                "type": "independent",
                "sample_count": 256 # 預覽品質
            },
            "film": {
                "type": "hdrfilm",
                "width": CAMERA_CONFIG['output_resolution'][0],
                "height": CAMERA_CONFIG['output_resolution'][1],
                "pixel_format": "rgb",
                "component_format": "float32",
                "rfilter": { "type": "gaussian" }
            }
        },
        
        # --- 光源 (Light at Brewster Angle) ---
        "light_source": {
            "type": "rectangle",
            "to_world": MI.ScalarTransform4f.translate(
                [0, scene_center_y, 0] 
            ).rotate(
                [1, 0, 0], 90 - incidence_angle 
            ).translate(
                [0, 0, light_dist] 
            ).scale([150, 150, 1]), 
            "emitter": {
                "type": "area",
                "radiance": {
                    "type": "spectrum",
                    "value": 80.0 
                }
            }
        }
    }
    
    # 4. 智慧材質分配 (Smart Material Assignment)
    
    # (A) 不透明物件 (_diffuse.obj) -> Rough Plastic
    if diffuse_obj_path and os.path.exists(diffuse_obj_path):
        scene_dict["diffuse_obj"] = {
            "type": "obj",
            "filename": str(diffuse_obj_path),
            "bsdf": {
                "type": "twosided", # 雙面渲染防止破面
                "bsdf": {
                    "type": "roughplastic", 
                    "diffuse_reflectance": { "type": "rgb", "value": [0.4, 0.4, 0.4] }, # 中性灰
                    "distribution": "ggx",
                    "alpha": 0.25 # 微粗糙
                }
            }
        }
        
    # (B) 透明物件 (_glass.obj) -> Dielectric (Glass)
    if glass_obj_path and os.path.exists(glass_obj_path):
        scene_dict["glass_obj"] = {
            "type": "obj",
            "filename": str(glass_obj_path),
            "bsdf": {
                "type": "dielectric",
                "int_ior": "bk7",  # 光學玻璃折射率
                "ext_ior": "air"
            }
        }
        
    return scene_dict

# ===========================================================
# 3. 影像處理與存檔 (修正版色彩與聯合白平衡)
# ===========================================================
def extract_polarization_images(stokes_image):
    """
    從 Stokes 影像提取 I_parallel 和 I_cross
    """
    stokes_image = np.array(stokes_image)
    I_parallel = stokes_image
    I_cross = stokes_image
    
    if len(stokes_image.shape) == 3:
        h, w, num_channels = stokes_image.shape
        
        # --- Mitsuba Spectral Polarized (15 channels) ---
        if num_channels == 15:
            # 1. 解碼 Interleaved 格式 (步進切片 0::3)
            S0_spectral = stokes_image[..., 0::3]
            S1_spectral = stokes_image[..., 1::3]
            
            # 2. 計算光譜強度
            I_parallel_spectral = np.maximum(0.5 * (S0_spectral + S1_spectral), 0)
            I_cross_spectral    = np.maximum(0.5 * (S0_spectral - S1_spectral), 0)
            
            # 3. 光譜轉 RGB 矩陣 (Blue Boost 版本 - 修正偏黃)
            spectral_to_rgb_matrix = np.array([
                [0.00, 0.05, 0.10, 0.80, 0.05], # R
                [0.00, 0.20, 0.75, 0.05, 0.00], # G
                [0.85, 0.60, 0.05, 0.00, 0.00], # B (Boosted)
            ])
            
            I_parallel = I_parallel_spectral @ spectral_to_rgb_matrix.T
            I_cross    = I_cross_spectral    @ spectral_to_rgb_matrix.T

        # --- RGB Mode (Fallback) ---
        elif num_channels == 12:
            S0 = stokes_image[..., 0:3]
            S1 = stokes_image[..., 3:6]
            I_parallel = np.maximum(0.5 * (S0 + S1), 0)
            I_cross    = np.maximum(0.5 * (S0 - S1), 0)
        
        else:
            base_img = stokes_image[..., :3]
            I_parallel = base_img
            I_cross = base_img

    # --- 聯合白平衡 (Joint White Balance) ---
    def normalize_and_white_balance_joint(img_main, img_secondary):
        # 以主圖 (Parallel) 的 99% 亮度作為白點，確保不過曝
        p_high = np.percentile(img_main, 99, axis=(0, 1))
        p_high[p_high < 1e-5] = 1e-5 
        
        # 關鍵：使用相同的 p_high 係數縮放兩張圖
        img_main_balanced = np.clip(img_main / p_high, 0, 1)
        img_secondary_balanced = np.clip(img_secondary / p_high, 0, 1)
        return img_main_balanced, img_secondary_balanced

    I_parallel, I_cross = normalize_and_white_balance_joint(I_parallel, I_cross)

    # --- Gamma 校正 (Linear -> sRGB) ---
    I_parallel = np.power(I_parallel, 1/2.2)
    I_cross    = np.power(I_cross, 1/2.2)

    return I_parallel, I_cross

def save_images(output_dir, name_prefix, I_para, I_cross):
    import cv2
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    def to_uint8(img):
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)
    
    # RGB 轉 BGR (給 OpenCV 用)
    img_para_bgr = cv2.cvtColor(to_uint8(I_para), cv2.COLOR_RGB2BGR)
    img_cross_bgr = cv2.cvtColor(to_uint8(I_cross), cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(str(out_dir / f"{name_prefix}_I_parallel.png"), img_para_bgr)
    cv2.imwrite(str(out_dir / f"{name_prefix}_I_cross.png"), img_cross_bgr)
    print(f"  [Output] 已儲存至 {out_dir}")

# ===========================================================
# 4. 主程式 (路徑解析邏輯)
# ===========================================================
def main():
    parser = argparse.ArgumentParser(description='PIDS 渲染器 (Blender Guide 標準版)')
    parser.add_argument('--scene', '-s', type=str, required=True, help='場景名稱 (例如 scenes/scene_00001)')
    parser.add_argument('--output', '-o', type=str, default='./rendered_output', help='輸出目錄')
    args = parser.parse_args()
    
    mi = init_mitsuba()
    
    scene_path = Path(args.scene)
    
    # === 智慧路徑解析 ===
    # 無論使用者輸入 "scene_001", "scene_001.obj", 還是 "scene_001_glass.obj"
    # 我們都一律提取基礎名稱 "scene_001" 並尋找其對應的 _diffuse 和 _glass 檔
    
    base_name = scene_path.stem.replace('_diffuse', '').replace('_glass', '')
    parent_dir = scene_path.parent
    
    diffuse_path = parent_dir / f"{base_name}_diffuse.obj"
    glass_path = parent_dir / f"{base_name}_glass.obj"
    
    # 檢查檔案
    has_diffuse = diffuse_path.exists()
    has_glass = glass_path.exists()
    
    # Fallback: 如果完全找不到分層檔案，且輸入本身就是存在的檔案
    if not has_diffuse and not has_glass:
        if scene_path.exists():
            print(f"注意: 找不到 _diffuse/_glass 分層檔案，將單一檔案視為 Diffuse: {scene_path}")
            diffuse_path = scene_path
            glass_path = None
        else:
            print(f"錯誤: 找不到場景檔案。")
            print(f"  預期: {diffuse_path} (及 _glass.obj)")
            sys.exit(1)
            
    print(f"=== PIDS 渲染任務: {base_name} ===")
    if diffuse_path and diffuse_path.exists(): print(f"  Diffuse (Plastic): {diffuse_path.name}")
    if glass_path and glass_path.exists():     print(f"  Glass   (BK7):     {glass_path.name}")
    
    # 建立場景
    scene_dict = create_scene_dict(diffuse_path, glass_path)
    
    # 載入與渲染
    scene = mi.load_dict(scene_dict)
    print("  正在渲染 (Path Tracing)...")
    img = mi.render(scene)
    
    # 後處理與存檔
    print("  正在處理色彩與偏振...")
    I_para, I_cross = extract_polarization_images(img)
    save_images(args.output, base_name, I_para, I_cross)
    print("=== 完成 ===")

if __name__ == "__main__":
    main()
