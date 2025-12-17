import time
import threading
import cv2
import numpy as np
import json
from flask import Flask, Response
from picamera2 import Picamera2

app = Flask(__name__)

# ================= 初始參數 =================
# 曝光時間 (微秒, us)。例如 20000 us = 20ms = 1/50秒
current_exposure = 20000 
# 類比增益 (1.0 = 原始, 數字越大噪訊越多)
current_gain = 1.0

# 目標解析度 (預覽用低解析度即可)
RESOLUTION = (320, 240)

# 全域變數
global_frame = None
lock = threading.Lock()
running = True
# ===========================================

def apply_manual_controls(picam, exposure_us, gain):
    """ 強制寫入相機硬體參數 """
    controls = {
        "ExposureTime": exposure_us,
        "AnalogueGain": gain,
        "AwbMode": 0, # 關閉自動白平衡 (0=Off) 避免顏色亂飄
        "ColourGains": (1.5, 1.5) # 手動固定白平衡 (R, B gain)
    }
    picam.set_controls(controls)

def camera_thread():
    global global_frame, current_exposure, current_gain
    
    print("Initializing Cameras...")
    cam0 = Picamera2(camera_num=0)
    config0 = cam0.create_video_configuration(main={"size": RESOLUTION, "format": "BGR888"})
    cam0.configure(config0)
    cam0.start()

    cam1 = Picamera2(camera_num=1)
    config1 = cam1.create_video_configuration(main={"size": RESOLUTION, "format": "BGR888"})
    cam1.configure(config1)
    cam1.start()
    
    print("Cameras Ready. Loop started.")
    
    while running:
        try:
            # 1. 應用當前的參數 (持續鎖定)
            apply_manual_controls(cam0, current_exposure, current_gain)
            apply_manual_controls(cam1, current_exposure, current_gain)

            # 2. 抓圖
            img0 = cam0.capture_array()
            img1 = cam1.capture_array()

            # 3. 計算亮度 (用於回饋)
            mean0 = np.mean(img0)
            mean1 = np.mean(img1)

            # 4. 繪製資訊在畫面上
            # 左鏡頭資訊
            cv2.putText(img0, f"L Bright: {mean0:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # 右鏡頭資訊
            cv2.putText(img1, f"R Bright: {mean1:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 拼接
            combined = np.hstack((img0, img1))
            
            # 顯示當前參數
            info_text = f"Exposure: {current_exposure} us | Gain: {current_gain:.1f}x"
            cv2.putText(combined, info_text, (20, combined.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            with lock:
                global_frame = combined
            
            # 稍微休息讓硬體反應
            time.sleep(0.05)
            
        except Exception as e:
            print(f"Cam Error: {e}")
            time.sleep(1)

    # 結束時釋放
    cam0.stop()
    cam1.stop()

def generate_frames():
    while True:
        with lock:
            if global_frame is None:
                time.sleep(0.05)
                continue
            output_frame = global_frame.copy()
            
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
        (flag, encodedImage) = cv2.imencode(".jpg", output_frame, encode_param)
        if not flag: continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route("/")
def index():
    return "<h1>Exposure Tuning Tool</h1><p>Check terminal to adjust controls.</p><img src='/video_feed' width='100%'>"

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def input_loop():
    """ 接收終端機輸入來調整參數 """
    global current_exposure, current_gain, running
    
    print("\n" + "="*40)
    print(" 🎛️  CONTROL PANEL (Keyboard Input) ")
    print("="*40)
    print(" [u] Exposure UP   (+1000us)")
    print(" [d] Exposure DOWN (-1000us)")
    print(" [U] Exposure UP++ (+5000us)")
    print(" [D] Exposure DOWN--(-5000us)")
    print(" [g] Gain UP       (+0.1)")
    print(" [f] Gain DOWN     (-0.1)")
    print(" [s] SAVE & QUIT   (Save params)")
    print("="*40)

    while running:
        cmd = input("Command > ").strip()
        
        if cmd == 'u':
            current_exposure += 1000
        elif cmd == 'd':
            current_exposure = max(100, current_exposure - 1000)
        elif cmd == 'U':
            current_exposure += 5000
        elif cmd == 'D':
            current_exposure = max(100, current_exposure - 5000)
        elif cmd == 'g':
            current_gain += 0.1
        elif cmd == 'f':
            current_gain = max(1.0, current_gain - 0.1)
        elif cmd == 's':
            print("Saving parameters to 'exposure_params.json'...")
            data = {"exposure": current_exposure, "gain": current_gain}
            with open("exposure_params.json", "w") as f:
                json.dump(data, f)
            print("✅ Saved! Exiting...")
            running = False
            break
        
        print(f"Set -> Exp: {current_exposure} | Gain: {current_gain:.1f}")

if __name__ == "__main__":
    # 啟動相機
    t_cam = threading.Thread(target=camera_thread)
    t_cam.daemon = True
    t_cam.start()

    # 啟動 Web Server (背景跑)
    t_web = threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False))
    t_web.daemon = True
    t_web.start()

    # 主執行緒跑輸入控制
    time.sleep(2) # 等一下 web server
    input_loop()