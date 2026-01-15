import time
import threading
import cv2
import numpy as np
from flask import Flask, Response
from picamera2 import Picamera2

app = Flask(__name__)

global_frame = None
lock = threading.Lock()

# 🔴 Change 1: Lower resolution for testing smoothness (can increase later)
# 320x240 is sufficient for preview
RESOLUTION = (320, 240)

def camera_thread():
    global global_frame
    
    print("Initializing Cameras...")
    
    # Initialize Cam 0
    cam0 = Picamera2(camera_num=0)
    config0 = cam0.create_video_configuration(main={"size": RESOLUTION, "format": "BGR888"})
    cam0.configure(config0)
    cam0.start()

    # Initialize Cam 1
    cam1 = Picamera2(camera_num=1)
    config1 = cam1.create_video_configuration(main={"size": RESOLUTION, "format": "BGR888"})
    cam1.configure(config1)
    cam1.start()
    
    print("Cameras Started! Streaming...")
    
    # Pre-allocate memory to avoid repeated malloc in the loop
    # This is a small optimization
    dummy_frame = np.zeros((RESOLUTION[1], RESOLUTION[0], 3), dtype=np.uint8)
    
    while True:
        try:
            # This may still stall a bit because capture_array waits for the next frame
            # but with lower resolution the memory copy time is much shorter
            img0 = cam0.capture_array()
            img1 = cam1.capture_array()
            
            if img0 is None: img0 = dummy_frame
            if img1 is None: img1 = dummy_frame

            # stitch left and right
            combined = np.hstack((img0, img1))
            
            with lock:
                global_frame = combined
            
            # 🔴 Change 2: Remove sleep to maximize throughput
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
            # copy frame for processing to avoid holding the lock too long
            output_frame = global_frame.copy()
            
        # 🔴 Change 3: Reduce JPEG quality significantly for speed
        # default is 95 (very slow); change to 50 (much faster and acceptable quality)
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