import time
import cv2
import numpy as np
from flask import Flask, Response
from picamera2 import Picamera2

app = Flask(__name__)

# ================= 設定區 =================
# 1. 鏡頭選擇 (0 或 1)
CAMERA_INDEX = 1  

# 2. 翻轉設定 (-1, 0, 1, or None)
FLIP_MODE = -1

# 3. 亮度增益 (為了對焦清楚，我們先把畫面弄亮一點)
SOFTWARE_GAIN = 3.0 
BRIGHTNESS_OFFSET = 30
# ========================================

print(f"🔬 啟動對焦輔助模式 (Camera {CAMERA_INDEX})...")

try:
    picam2 = Picamera2(camera_num=CAMERA_INDEX)
    config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2.0)
    print("✅ 相機啟動！請看瀏覽器畫面進行對焦")

except Exception as e:
    print(f"❌ 錯誤: {e}")
    exit()

def calculate_sharpness(img):
    # 使用拉普拉斯變異數來計算清晰度
    # 影像越清晰，邊緣越銳利，這個數值就越高
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score

def generate_frames():
    while True:
        try:
            frame = picam2.capture_array()

            if FLIP_MODE is not None:
                frame = cv2.flip(frame, FLIP_MODE)

            # --- 增亮處理 ---
            frame = cv2.convertScaleAbs(frame, alpha=SOFTWARE_GAIN, beta=BRIGHTNESS_OFFSET)
            
            # --- 步驟 1: 計算全圖的清晰度分數 ---
            score = calculate_sharpness(frame)
            
            # --- 步驟 2: 製作「中央放大鏡」 ---
            # 裁切畫面正中間 200x200 的區域
            h, w, _ = frame.shape
            center_x, center_y = w // 2, h // 2
            crop_size = 200
            x1 = max(0, center_x - crop_size // 2)
            y1 = max(0, center_y - crop_size // 2)
            x2 = min(w, center_x + crop_size // 2)
            y2 = min(h, center_y + crop_size // 2)
            
            # 裁切出來
            zoomed_frame = frame[y1:y2, x1:x2]
            
            # 把它放大回 640x480 (讓你容易看細節)
            display_frame = cv2.resize(zoomed_frame, (640, 480), interpolation=cv2.INTER_NEAREST)

            # --- 步驟 3: 把分數寫在畫面上 ---
            # 顏色邏輯：分數越高顏色越綠，低分則是紅
            color = (0, 0, 255) # 紅
            if score > 100: color = (0, 255, 255) # 黃
            if score > 300: color = (0, 255, 0)   # 綠
            
            cv2.putText(display_frame, f"Focus Score: {int(score)}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            
            cv2.putText(display_frame, "MODE: ZOOM 3x", (50, 450), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # 轉成 JPEG
            ret, buffer = cv2.imencode('.jpg', display_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.01)
            
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(0.1)

@app.route('/')
def index():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # 使用 Port 5002 避免衝突
    print("請開啟: http://IP:5002")
    app.run(host='0.0.0.0', port=5002, threaded=True, debug=False)
