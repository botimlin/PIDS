import time
import cv2
import numpy as np
from flask import Flask, Response, redirect, url_for
from picamera2 import Picamera2

app = Flask(__name__)

# ================= 參數設定區 =================
CAMERA_INDEX = 0          # 第一顆鏡頭
FLIP_MODE = -1            # 翻轉設定
SOFTWARE_GAIN = 1.5       # 軟體增益
BRIGHTNESS_OFFSET = 10    # 亮度偏移
# ============================================

# 全域變數
picam2 = None
dark_frame = None       # 存全黑圖 (熱噪訊)
flat_frame = None       # 存全白圖 (灰塵 + 亮度不均)
is_dark_ready = False   # 是否已拍好暗場
is_flat_ready = False   # 是否已拍好平場

print(f"🔬 初始化雙重校正相機 (Camera {CAMERA_INDEX})...")

try:
    picam2 = Picamera2(camera_num=CAMERA_INDEX)
    config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2.0)
    print("✅ 相機啟動成功！")

except Exception as e:
    print(f"❌ 相機啟動失敗: {e}")
    exit()

def apply_full_calibration(raw_img):
    """
    究極校正公式：
    Corrected = (Raw - Dark) / (Flat - Dark) * Mean(Flat - Dark)
    """
    global dark_frame, flat_frame
    
    # 全部轉成 float32 防止運算溢位
    raw_f = raw_img.astype(np.float32)
    dark_f = dark_frame.astype(np.float32)
    flat_f = flat_frame.astype(np.float32)
    
    # 1. 分子：原始圖 - 暗場 (扣除熱噪訊)
    # 使用 max(0) 確保不會變成負數
    numerator = np.maximum(raw_f - dark_f, 0)
    
    # 2. 分母：平場 - 暗場 (算出純淨的光路髒汙圖)
    # 加 1.0 防止除以零
    denominator = np.maximum(flat_f - dark_f, 1.0)
    
    # 3. 計算平均亮度 (作為歸一化係數)
    mean_val = np.mean(denominator)
    
    # 4. 執行除法與修正
    corrected = (numerator / denominator) * mean_val
    
    # 5. 轉回 0-255 整數
    return np.clip(corrected, 0, 255).astype(np.uint8)

def generate_frames():
    global is_dark_ready, is_flat_ready
    
    while True:
        try:
            frame = picam2.capture_array()
            if FLIP_MODE is not None: frame = cv2.flip(frame, FLIP_MODE)
            frame = cv2.convertScaleAbs(frame, alpha=SOFTWARE_GAIN, beta=BRIGHTNESS_OFFSET)
            
            output_frame = frame.copy()
            status_text = []
            color = (0, 0, 255) # 預設紅色 (未校正)

            # --- 判斷校正狀態 ---
            if is_dark_ready and is_flat_ready:
                # 兩個都準備好了 -> 執行究極校正
                try:
                    output_frame = apply_full_calibration(frame)
                    status_text.append("MODE: SUPER CLEAN (Dark+Flat)")
                    color = (0, 255, 0) # 綠色 (完美)
                except Exception as e:
                    print(e)
            else:
                # 顯示目前的進度
                if not is_dark_ready:
                    status_text.append("[1] Need Dark Frame (Cover Lens)")
                else:
                    status_text.append("[OK] Dark Frame Ready")
                    
                if not is_flat_ready:
                    status_text.append("[2] Need Flat Frame (Aim White)")
                else:
                    status_text.append("[OK] Flat Frame Ready")

            # 把文字印在畫面上
            for i, line in enumerate(status_text):
                cv2.putText(output_frame, line, (10, 30 + i*30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # 轉 JPEG
            ret, buffer = cv2.imencode('.jpg', output_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.01)

        except Exception as e:
            print(e)
            time.sleep(0.1)

@app.route('/')
def index():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- 校正路由 ---

@app.route('/calibrate_dark')
def calibrate_dark():
    global dark_frame, is_dark_ready
    print("⬛ 正在拍攝暗場 (Dark Frame)...")
    
    # 抓 10 張全黑圖取平均，減少隨機誤差
    frames = []
    for _ in range(10):
        raw = picam2.capture_array()
        if FLIP_MODE is not None: raw = cv2.flip(raw, FLIP_MODE)
        raw = cv2.convertScaleAbs(raw, alpha=SOFTWARE_GAIN, beta=BRIGHTNESS_OFFSET)
        frames.append(raw.astype(np.float32))
        time.sleep(0.05)
        
    dark_frame = np.mean(frames, axis=0)
    is_dark_ready = True
    print("✅ 暗場已紀錄。")
    return redirect(url_for('index'))

@app.route('/calibrate_flat')
def calibrate_flat():
    global flat_frame, is_flat_ready
    print("⬜ 正在拍攝平場 (Flat Frame)...")
    
    # 抓 10 張全白圖取平均
    frames = []
    for _ in range(10):
        raw = picam2.capture_array()
        if FLIP_MODE is not None: raw = cv2.flip(raw, FLIP_MODE)
        raw = cv2.convertScaleAbs(raw, alpha=SOFTWARE_GAIN, beta=BRIGHTNESS_OFFSET)
        frames.append(raw.astype(np.float32))
        time.sleep(0.05)
        
    flat_frame = np.mean(frames, axis=0)
    is_flat_ready = True
    print("✅ 平場已紀錄。")
    return redirect(url_for('index'))

@app.route('/reset')
def reset():
    global is_dark_ready, is_flat_ready
    is_dark_ready = False
    is_flat_ready = False
    return redirect(url_for('index'))

if __name__ == '__main__':
    print("==================================================")
    print("🚀 雙重校正相機啟動 (Port 5004)")
    print("步驟 1: 蓋上鏡頭蓋 -> 瀏覽器輸入 /calibrate_dark")
    print("步驟 2: 對準白紙   -> 瀏覽器輸入 /calibrate_flat")
    print("==================================================")
    app.run(host='0.0.0.0', port=5004, threaded=True, debug=False)
