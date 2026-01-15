import time
import os
import numpy as np
import cv2
from picamera2 import Picamera2

# ================= 配置區域 =================
# 這裡設定相機 ID，通常 Cam 0 和 Cam 1
CAM_ID_LEFT = 0
CAM_ID_RIGHT = 1

# 採集張數 (張數越多，隨機噪訊消除得越乾淨，建議 20-50 張)
SAMPLE_COUNT = 30

# 影像解析度 (必須與你之後跑推論的解析度一致!)
# IMX296 原始解析度較大，這裡範例設為 VGA 或 720p，請依你實際需求修改
RESOLUTION = (640, 480) 

# 輸出目錄 (對應 Docker 的掛載點)
OUTPUT_DIR = "./output/calibration_data"
# ===========================================

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def capture_average_frame(cam_idx, label, count):
    """
    開啟指定相機，連續拍攝 count 張照片，並計算平均值 (float32)
    """
    print(f"\n[Camera {cam_idx}] Initializing for {label} capture...")
    
    try:
        picam2 = Picamera2(camera_num=cam_idx)
        
        # 設定相機參數 (RGB888 格式)
        config = picam2.create_video_configuration(
            main={"size": RESOLUTION, "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
        
        # 讓相機熱機一下，讓 Auto White Balance / Gain 穩定 (如果是 Flat field)
        # 對於 Dark field，雖然是全黑，但讓 sensor 進入工作溫度也很重要
        time.sleep(2)
        
        frames = []
        print(f"[Camera {cam_idx}] Capturing {count} frames for {label}...")
        
        for i in range(count):
            # 擷取影像陣列
            frame = picam2.capture_array()
            # 轉為 float 為了高精度計算
            frames.append(frame.astype(np.float32))
            time.sleep(0.05) # 稍微間隔，避免同一瞬間的噪訊
            
        picam2.stop()
        picam2.close() # 務必關閉釋放資源，否則無法開另一顆鏡頭
        
        # 計算時間平均 (Temporal Averaging)
        avg_frame = np.mean(frames, axis=0)
        print(f"[Camera {cam_idx}] {label} capture complete. Mean value: {np.mean(avg_frame):.2f}")
        return avg_frame

    except Exception as e:
        print(f"Error capturing from Camera {cam_idx}: {e}")
        try:
            picam2.stop()
            picam2.close()
        except:
            pass
        return None

def main():
    ensure_dir(OUTPUT_DIR)
    print("=== PIDS Stereo Calibration Tool ===")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Resolution: {RESOLUTION}")
    print("====================================")

    # ---------------------------------------------------------
    # 第一階段：暗場校正 (Dark Field)
    # 目的：捕捉 Sensor Bias 和 Thermal Noise
    # ---------------------------------------------------------
    input("\n🔴 STEP 1: Dark Field Capture\n請【蓋上】兩個鏡頭的鏡頭蓋 (確保全黑)。\n按 Enter 開始...")

    print("\n--- Capturing LEFT Camera Dark Field ---")
    dark_L = capture_average_frame(CAM_ID_LEFT, "Dark-L", SAMPLE_COUNT)
    
    print("\n--- Capturing RIGHT Camera Dark Field ---")
    dark_R = capture_average_frame(CAM_ID_RIGHT, "Dark-R", SAMPLE_COUNT)

    if dark_L is None or dark_R is None:
        print("❌ Error: Failed to capture dark frames. Aborting.")
        return

    # 儲存原始暗場數據 (Master Dark)
    np.save(os.path.join(OUTPUT_DIR, "master_dark_L.npy"), dark_L)
    np.save(os.path.join(OUTPUT_DIR, "master_dark_R.npy"), dark_R)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "preview_dark_L.png"), dark_L.astype(np.uint8))
    print("✅ Dark Field data saved.")

    # ---------------------------------------------------------
    # 第二階段：平場校正 (Flat Field)
    # 目的：捕捉 Vignetting (暗角) 和 PRNU (像素響應不均)
    # ---------------------------------------------------------
    input("\n🟢 STEP 2: Flat Field Capture\n請拿掉鏡頭蓋，將相機對準【均勻白色光源/白紙】(確保不過曝)。\n按 Enter 開始...")

    print("\n--- Capturing LEFT Camera Flat Field ---")
    flat_L_raw = capture_average_frame(CAM_ID_LEFT, "Flat-L", SAMPLE_COUNT)
    
    print("\n--- Capturing RIGHT Camera Flat Field ---")
    flat_R_raw = capture_average_frame(CAM_ID_RIGHT, "Flat-R", SAMPLE_COUNT)

    if flat_L_raw is None or flat_R_raw is None:
        print("❌ Error: Failed to capture flat frames. Aborting.")
        return

    # ---------------------------------------------------------
    # 第三階段：計算與生成最終校正矩陣
    # 物理公式：Gain = (Flat_Avg - Dark_Avg) / Mean(Flat_Avg - Dark_Avg)
    # ---------------------------------------------------------
    print("\n🔵 STEP 3: Computing Radiometric Correction Matrices...")

    # 計算真正的光學響應 (扣除底噪)
    flat_L_clean = flat_L_raw - dark_L
    flat_R_clean = flat_R_raw - dark_R

    # 避免除以零或負值 (雖然理論上拍白紙不會發生，但工程上要防呆)
    flat_L_clean[flat_L_clean < 1] = 1
    flat_R_clean[flat_R_clean < 1] = 1

    # 歸一化 (Normalize) -> 產生 Flat Field Correction Gain
    # 我們需要的是 "校正因子"，所以是 Mean / Pixel_Value
    # 應用時：Corrected_Pixel = Raw_Pixel * Gain_Map
    gain_map_L = np.mean(flat_L_clean) / flat_L_clean
    gain_map_R = np.mean(flat_R_clean) / flat_R_clean

    # 儲存 Gain Map
    np.save(os.path.join(OUTPUT_DIR, "gain_map_L.npy"), gain_map_L)
    np.save(os.path.join(OUTPUT_DIR, "gain_map_R.npy"), gain_map_R)
    
    # 儲存預覽圖 (將 Gain Map 視覺化，這張圖可以放在論文裡！)
    # 為了視覺化，將 Gain Map 轉為 0-255 熱圖
    norm_vis_L = cv2.normalize(gain_map_L, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.applyColorMap(norm_vis_L, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "vis_gain_map_L.png"), norm_vis_L)

    print(f"\n✅ Calibration Complete! Files saved to {OUTPUT_DIR}")
    print("Files generated:")
    print("  - master_dark_L.npy, master_dark_R.npy (需在推理時扣除)")
    print("  - gain_map_L.npy, gain_map_R.npy     (需在推理時相乘)")

if __name__ == "__main__":
    main()