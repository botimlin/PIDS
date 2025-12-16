import time
import os
import cv2
import numpy as np
from flask import Flask, Response, redirect, url_for
from picamera2 import Picamera2

app = Flask(__name__)

# ================= 參數設定區 =================
# 1. 鏡頭選擇 (您現在用的是第一顆)
CAMERA_INDEX = 0

# 2. 翻轉設定 (如果畫面上下顛倒請改這個)
FLIP_MODE = -1 

# 3. 亮度設定 (保持適中，不要過曝)
SOFTWARE_GAIN = 1.5
BRIGHTNESS_OFFSET = 10

# ============================================

# 全域變數
picam2 = None
background_frame = None  # 這是我們的「除塵參考圖」
is_calibrated = False    # 是否已經校正過的旗標

print(f"🧹 初始化軟體除塵相機 (Camera {CAMERA_INDEX})...")

try:
    picam2 = Picamera2(camera_num=CAMERA_INDEX)
    config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    
    # 等待相機暖機
    time.sleep(2.0)
    print("✅ 相機啟動成功！請打開瀏覽器。")

except Exception as e:
    print(f"❌ 相機啟動失敗: {e}")
    exit()

def perform_flat_field_correction(img):
    """
    核心演算法：平場校正
    公式：修正圖 = (原圖 / 背景髒圖) * 背景平均亮度
    """
    global background_frame
    
    # 防呆：如果沒有背景圖，就回傳原圖
    if background_frame is None:
        return img

    # 1. 轉成浮點數運算 (Float32) 以避免精度流失
    # 加 1.0 是為了防止除以 0 的錯誤
    img_float = img.astype(np.float32) + 1.0
    bg_float = background_frame.astype(np.float32) + 1.0
    
    # 2. 計算背景的平均亮度 (Mean Intensity)
    mean_bg = np.mean(bg_float)
    
    # 3. 套用公式
    corrected = (img_float / bg_float) * mean_bg
    
    # 4. 數值限制在 0-255 之間，並轉回整數 (uint8)
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    
    return corrected

def generate_frames():
    global background_frame, is_calibrated
    
    while True:
        try:
            # 抓取原始影像
            frame = picam2.capture_array()

            # 翻轉
            if FLIP_MODE is not None:
                frame = cv2.flip(frame, FLIP_MODE)
            
            # 亮度調整
            frame = cv2.convertScaleAbs(frame, alpha=SOFTWARE_GAIN, beta=BRIGHTNESS_OFFSET)
            
            # --- 處理流程 ---
            output_frame = frame.copy()
            
            if is_calibrated and background_frame is not None:
                # 如果已經校正過，執行除塵運算
                try:
                    output_frame = perform_flat_field_correction(frame)
                    
                    # 畫上綠色標記，表示現在是乾淨模式
                    cv2.putText(output_frame, "MODE: CLEAN (Corrected)", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                except:
                    pass
            else:
                # 還沒校正，顯示紅色提示
                cv2.putText(output_frame, "MODE: RAW (Dusty)", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(output_frame, "Please aim at white paper & calibrate", (10, 450), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # 編碼成 JPEG
            ret, buffer = cv2.imencode('.jpg', output_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.01)
            
        except Exception as e:
            print(f"串流錯誤: {e}")
            time.sleep(0.1)

@app.route('/')
def index():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 新增這個路由：只要瀏覽器輸入 http://IP:5003/calibrate 就會觸發
@app.route('/calibrate')
def calibrate():
    global background_frame, is_calibrated
    print("🔄 正在執行校正 (捕捉背景)...")
    
    # 為了準確，我們連續抓 5 張圖取平均
    temp_frames = []
    for _ in range(5):
        raw = picam2.capture_array()
        if FLIP_MODE is not None:
            raw = cv2.flip(raw, FLIP_MODE)
        # 記得要用跟顯示時一樣的亮度來校正
        raw = cv2.convertScaleAbs(raw, alpha=SOFTWARE_GAIN, beta=BRIGHTNESS_OFFSET)
        temp_frames.append(raw.astype(np.float32))
        time.sleep(0.05)
    
    # 計算平均影像
    background_frame = np.mean(temp_frames, axis=0)
    is_calibrated = True
    
    print("✅ 校正完成！背景已記憶。")
    return redirect(url_for('index'))

# 重設按鈕 (如果校正錯了可以用這個重來)
@app.route('/reset')
def reset():
    global is_calibrated, background_frame
    is_calibrated = False
    background_frame = None
    print("R 重設校正。")
    return redirect(url_for('index'))

if __name__ == '__main__':
    # 使用 Port 5003
    print("------------------------------------------------")
    print("1. 請開啟畫面: http://<樹莓派IP>:5003")
    print("2. 準備一張白紙，讓畫面填滿白色 (要看到灰塵黑點)")
    print("3. 在瀏覽器網址列輸入: http://<樹莓派IP>:5003/calibrate")
    print("   (或者在另一個終端機輸入 curl http://localhost:5003/calibrate)")
    print("------------------------------------------------")
    app.run(host='0.0.0.0', port=5003, threaded=True, debug=False)
