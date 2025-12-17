import time
import cv2
import numpy as np
from flask import Flask, Response
from picamera2 import Picamera2

# ================= 參數調整區 =================

# 1. 選擇鏡頭
# 0 = 第一顆鏡頭 (您現在要用的)
CAMERA_INDEX = 0

# 2. 軟體亮度控制 (關鍵修改)
# 之前的暴力版是 3.0 (太亮)
# 1.0 = 原汁原味 (不做任何增亮)
# 0.8 = 變暗 (如果現場光很強，可以用小於 1 的數字)
# 1.5 = 微亮 (目前設定)
SOFTWARE_GAIN = 1.5

# 3. 亮度偏移 (讓黑色更黑)
# 之前是 40，現在改小一點，畫面會有對比度
BRIGHTNESS_OFFSET = 10

# 4. 翻轉設定 (第一顆鏡頭通常需要翻轉)
FLIP_MODE = -1 

# ============================================

app = Flask(__name__)

print(f"📷 啟動第一顆相機 (Index: {CAMERA_INDEX}) - 正常曝光模式...")

try:
    picam2 = Picamera2(camera_num=CAMERA_INDEX)
    
    # 建立基礎設定
    config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )

    picam2.configure(config)
    picam2.start()
    
    time.sleep(2.0)
    print(f"✅ 相機啟動成功！目前增益: {SOFTWARE_GAIN} 倍")

except Exception as e:
    print(f"❌ 相機啟動失敗: {e}")
    exit()

def generate_frames():
    while True:
        try:
            frame = picam2.capture_array()

            # 1. 翻轉
            if FLIP_MODE is not None:
                frame = cv2.flip(frame, FLIP_MODE)
            
            # 2. 亮度調整 (數學運算)
            # 如果 SOFTWARE_GAIN 是 1.0，這行幾乎沒作用 (原圖)
            # 如果是 1.5，亮度增加 50%
            frame = cv2.convertScaleAbs(frame, alpha=SOFTWARE_GAIN, beta=BRIGHTNESS_OFFSET)

            # 3. 壓成 JPEG
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
    # 換回 Port 5000
    print("請開啟瀏覽器： http://<樹莓派IP>:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
