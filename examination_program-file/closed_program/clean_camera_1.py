import time
import cv2
import numpy as np
from flask import Flask, Response, redirect, url_for
from picamera2 import Picamera2

app = Flask(__name__)

# ================= 參數設定區 =================
# 1. 鏡頭選擇
# 改成 1 (第二顆鏡頭)
CAMERA_INDEX = 1

# 2. 翻轉設定 
# 注意：第二顆鏡頭的安裝方向可能不同，如果畫面顛倒，請改成 0 或 1
FLIP_MODE = -1 

# 3. 亮度設定
SOFTWARE_GAIN = 1.5
BRIGHTNESS_OFFSET = 10
# ============================================

# 全域變數
picam2 = None
flat_frame = None       # 這是我們的「平場校正圖」(白紙)
is_calibrated = False   # 是否已校正

print(f"🧹 啟動第二顆相機 (Index: {CAMERA_INDEX}) - 平場校正模式...")

try:
    picam2 = Picamera2(camera_num=CAMERA_INDEX)
    config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    
    time.sleep(2.0)
    print("✅ 相機 1 啟動成功！")

except Exception as e:
    print(f"❌ 相機啟動失敗: {e}")
    # 如果啟動失敗，通常是因為 Index 錯誤，或是相機排線沒接好
    exit()

def apply_flat_field(img):
    """
    平場校正公式：
    修正圖 = (原圖 / 平場圖) * 平場平均亮度
    """
    global flat_frame
    
    if flat_frame is None:
        return img

    # 轉成 float32 運算，並加 1 防止除以 0
    img_f = img.astype(np.float32) + 1.0
    flat_f = flat_frame.astype(np.float32) + 1.0
    
    # 計算平場的平均亮度
    mean_flat = np.mean(flat_f)
    
    # 套用公式
    corrected = (img_f / flat_f) * mean_flat
    
    # 限制數值並轉回整數
    return np.clip(corrected, 0, 255).astype(np.uint8)

def generate_frames():
    global flat_frame, is_calibrated
    
    while True:
        try:
            # 抓圖
            frame = picam2.capture_array()

            # 翻轉
            if FLIP_MODE is not None:
                frame = cv2.flip(frame, FLIP_MODE)
            
            # 亮度調整
            frame = cv2.convertScaleAbs(frame, alpha=SOFTWARE_GAIN, beta=BRIGHTNESS_OFFSET)
            
            output_frame = frame.copy()
            
            # --- 處理流程 ---
            if is_calibrated and flat_frame is not None:
                try:
                    # 執行校正
                    output_frame = apply_flat_field(frame)
                    
                    # 顯示綠色狀態
                    cv2.putText(output_frame, f"CAM {CAMERA_INDEX}: CLEAN (Flat Corrected)", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                except:
                    pass
            else:
                # 顯示紅色狀態
                cv2.putText(output_frame, f"CAM {CAMERA_INDEX}: RAW (Needs Calibration)", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(output_frame, "Please aim at white paper & calibrate", (10, 450), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # 轉 JPEG
            ret, buffer = cv2.imencode('.jpg', output_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.01)
            
        except Exception as e:
            print(f"串流錯誤: {e}")
            time.sleep(0.1)

@app.route('/')
def index():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 校正按鈕路由
@app.route('/calibrate')
def calibrate():
    global flat_frame, is_calibrated
    print(f"🔄 CAM {CAMERA_INDEX} 正在執行平場校正...")
    
    # 連續抓 5 張圖取平均
    frames = []
    for _ in range(5):
        raw = picam2.capture_array()
        if FLIP_MODE is not None: raw = cv2.flip(raw, FLIP_MODE)
        raw = cv2.convertScaleAbs(raw, alpha=SOFTWARE_GAIN, beta=BRIGHTNESS_OFFSET)
        frames.append(raw.astype(np.float32))
        time.sleep(0.05)
    
    flat_frame = np.mean(frames, axis=0)
    is_calibrated = True
    
    print("✅ 校正完成！")
    return redirect(url_for('index'))

# 重設按鈕
@app.route('/reset')
def reset():
    global is_calibrated, flat_frame
    is_calibrated = False
    flat_frame = None
    print("R 重設校正狀態。")
    return redirect(url_for('index'))

if __name__ == '__main__':
    # 為了不跟第一顆相機衝突，這裡改用 Port 5002
    print("==================================================")
    print(f"🚀 第二顆相機啟動 (Port 5002)")
    print(f"1. 畫面預覽: http://<樹莓派IP>:5002")
    print(f"2. 執行校正: http://<樹莓派IP>:5002/calibrate")
    print("==================================================")
    app.run(host='0.0.0.0', port=5002, threaded=True, debug=False)
