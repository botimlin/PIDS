import time
import cv2
import numpy as np
from flask import Flask, Response
from picamera2 import Picamera2

# ================= 參數調整區 =================

# 1. 選擇鏡頭 (關鍵修改)
# 0 = 第一顆鏡頭 (原本那顆)
# 1 = 第二顆鏡頭 (現在要測的)
CAMERA_INDEX = 1 

# 2. 軟體暴力增亮
SOFTWARE_GAIN = 3.0
BRIGHTNESS_OFFSET = 40

# 3. 翻轉設定
# 注意：第二顆鏡頭安裝方向可能不同，如果發現顛倒，請改這裡
# -1 = 旋轉 180 度
#  0 = 垂直翻轉
#  1 = 水平翻轉
# None = 不翻轉 (建議先設為 None 看看原始方向)
FLIP_MODE = None 

# ============================================

app = Flask(__name__)

print(f"📷 初始化第二顆相機 (Index: {CAMERA_INDEX})...")
try:
    # 【這裡改了】指定要開啟哪一顆鏡頭
    picam2 = Picamera2(camera_num=CAMERA_INDEX)
    
    # 使用純軟體增亮，不做硬體控制，避免崩潰
    config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )

    picam2.configure(config)
    picam2.start()
    
    time.sleep(2.0)
    print(f"✅ 第二顆相機啟動成功！")

except Exception as e:
    print(f"❌ 相機啟動失敗: {e}")
    print("提示：如果失敗，可能是系統只抓到一顆，或者編號不是 1")
    exit()

def generate_frames():
    while True:
        try:
            frame = picam2.capture_array()

            # 1. 【翻轉】
            if FLIP_MODE is not None:
                frame = cv2.flip(frame, FLIP_MODE)
            
            # 2. 【增亮】
            frame = cv2.convertScaleAbs(frame, alpha=SOFTWARE_GAIN, beta=BRIGHTNESS_OFFSET)

            # 3. 【顏色修正】
            # 如果顏色變藍色人，請解開下面這行註解
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

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
    # 為了避免跟上一個程式衝突，我們換個 Port：5001
    print("請開啟瀏覽器： http://<樹莓派IP>:5001")
    app.run(host='0.0.0.0', port=5001, threaded=True, debug=False)
