import time
import threading
import cv2
import numpy as np
from flask import Flask, Response
from picamera2 import Picamera2

app = Flask(__name__)

global_frame = None
lock = threading.Lock()

# 🔴 改動 1: 先降低解析度來測試流暢度 (之後可以慢慢加回來)
# 320x240 對於預覽已經足夠清楚
RESOLUTION = (320, 240)

def camera_thread():
    global global_frame
    
    print("Initializing Cameras...")
    
    # 初始化 Cam 0
    cam0 = Picamera2(camera_num=0)
    config0 = cam0.create_video_configuration(main={"size": RESOLUTION, "format": "BGR888"})
    cam0.configure(config0)
    cam0.start()

    # 初始化 Cam 1
    cam1 = Picamera2(camera_num=1)
    config1 = cam1.create_video_configuration(main={"size": RESOLUTION, "format": "BGR888"})
    cam1.configure(config1)
    cam1.start()
    
    print("Cameras Started! Streaming...")
    
    # 預先分配記憶體，避免迴圈內不斷 malloc
    # 這是加速的小技巧
    dummy_frame = np.zeros((RESOLUTION[1], RESOLUTION[0], 3), dtype=np.uint8)
    
    while True:
        try:
            # 這裡還是會稍微卡一點，因為 capture_array 會等待下一幀
            # 但因為解析度變小，複製記憶體的時間會大幅縮短
            img0 = cam0.capture_array()
            img1 = cam1.capture_array()
            
            if img0 is None: img0 = dummy_frame
            if img1 is None: img1 = dummy_frame

            # 左右拼接
            combined = np.hstack((img0, img1))
            
            with lock:
                global_frame = combined
            
            # 🔴 改動 2: 移除 sleep，讓它全力跑
            # time.sleep(0.01) 
            
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

def generate_frames():
    while True:
        with lock:
            if global_frame is None:
                time.sleep(0.01)
                continue
            # 複製一份出來處理，避免鎖太久
            output_frame = global_frame.copy()
            
        # 🔴 改動 3: 大幅降低 JPEG 品質以換取速度
        # 預設是 95 (非常慢)，改成 50 (快很多且畫質可接受)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
        
        (flag, encodedImage) = cv2.imencode(".jpg", output_frame, encode_param)
        
        if not flag:
            continue
        
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encodedImage) + b'\r\n')

@app.route("/")
def index():
    return "<h1>PIDS Stereo Stream (Fast Mode)</h1><img src='/video_feed' width='100%'>"

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    t = threading.Thread(target=camera_thread)
    t.daemon = True
    t.start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)