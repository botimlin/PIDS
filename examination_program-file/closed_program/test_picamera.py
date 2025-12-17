import time
import cv2
import numpy as np
from flask import Flask, Response
from picamera2 import Picamera2

# ================= 參數調整區 =================

# 1. 軟體暴力增亮 (關鍵！)
# 1.0 = 原圖亮度
# 2.0 = 兩倍亮
# 3.0 = 三倍亮 (如果不夠亮，請把這個數字改大)
SOFTWARE_GAIN = 3.0

# 2. 亮度偏移 (讓黑色變灰色，提升暗部細節)
BRIGHTNESS_OFFSET = 40

# 3. 翻轉設定
# -1 = 旋轉 180 度
FLIP_MODE = -1 

# ============================================

app = Flask(__name__)

print("📷 初始化相機中 (安全模式)...")
try:
    picam2 = Picamera2()
    
    # ❌ 移除所有 controls 設定，避免型別錯誤導致崩潰
    # 我們改用下面的 OpenCV 數學運算來增亮
    config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )

    picam2.configure(config)
    picam2.start()
    
    # 暖機 2 秒
    time.sleep(2.0)
    print(f"✅ 相機啟動成功！(使用軟體增亮: {SOFTWARE_GAIN}倍)")

except Exception as e:
    print(f"❌ 相機啟動失敗: {e}")
    exit()

def generate_frames():
    while True:
        try:
            # 1. 抓取影像
            frame = picam2.capture_array()

            # 2. 【軟體翻轉】
            if FLIP_MODE is not None:
                frame = cv2.flip(frame, FLIP_MODE)
            
            # 3. 【軟體暴力增亮】 (絕對有效)
            # 這行會強制把每個像素的亮度乘以 SOFTWARE_GAIN
            frame = cv2.convertScaleAbs(frame, alpha=SOFTWARE_GAIN, beta=BRIGHTNESS_OFFSET)

            # 4. 【顏色修正】
            # 如果顏色正常請保留 # 
            # 如果變成藍色人，請把下面這行的 # 拿掉
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 5. 壓成 JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
