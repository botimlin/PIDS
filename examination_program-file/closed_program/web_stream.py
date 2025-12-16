import time
import threading
import cv2
import numpy as np
from flask import Flask, Response

# 嘗試匯入 Picamera2，如果報錯代表沒裝
try:
    from picamera2 import Picamera2, Picamera2Config
except ImportError:
    print("❌ 請先安裝 Picamera2: pip install picamera2")
    exit()

app = Flask(__name__)

class PidsCamera:
    def __init__(self):
        # 初始化兩個 Picamera2 實例
        print("📷 初始化相機中...")
        self.cam_left = Picamera2(device_interface=0)  # 對應 CAM 0
        self.cam_right = Picamera2(device_interface=1) # 對應 CAM 1 (Pi 5 自動對應第二個接口)

        # 設定配置 (預覽模式)
        config_l = self.cam_left.create_configuration(main={"size": (640, 480), "format": "RGB888"})
        config_r = self.cam_right.create_configuration(main={"size": (640, 480), "format": "RGB888"})
        
        self.cam_left.configure(config_l)
        self.cam_right.configure(config_r)

        self.cam_left.start()
        self.cam_right.start()
        print("✅ 雙鏡頭啟動成功！")

    def get_frames(self):
        # 抓取最新的影像 (非同步請求)
        # 注意：這只是簡單預覽，未來的同步需要更高級的寫法
        frame_l = self.cam_left.capture_array()
        frame_r = self.cam_right.capture_array()
        
        # Picamera2 抓出來的是 RGB，OpenCV 習慣 BGR，轉一下顏色
        frame_l = cv2.cvtColor(frame_l, cv2.COLOR_RGB2BGR)
        frame_r = cv2.cvtColor(frame_r, cv2.COLOR_RGB2BGR)
        
        return frame_l, frame_r

    def stop(self):
        self.cam_left.stop()
        self.cam_right.stop()

# 全域相機物件
camera_system = None

def generate_frames():
    global camera_system
    while True:
        if camera_system:
            frame_l, frame_r = camera_system.get_frames()
            
            # 左右合併
            combined = np.hstack((frame_l, frame_r))
            
            ret, buffer = cv2.imencode('.jpg', combined)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.1)

@app.route('/')
def index():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # 確保在 Context 內安裝
    # pip install picamera2 
    try:
        camera_system = PidsCamera()
        print("🚀 伺服器啟動: http://0.0.0.0:5000")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        if camera_system:
            camera_system.stop()
