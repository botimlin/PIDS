#!/usr/bin/env python3
"""
雙相機測試腳本
同時捕獲左右相機的 YUV420 Y 通道（灰階）

執行: python test_dual_camera.py
瀏覽: http://<pi-ip>:5000
"""
from flask import Flask, Response, render_template_string
from picamera2 import Picamera2
import cv2
import numpy as np
import time
import threading

app = Flask(__name__)

# 全域變數
cam_left = None   # Camera 0 (I∥ - 左相機, 0° 偏振)
cam_right = None  # Camera 1 (I⊥ - 右相機, 90° 偏振)
frame_left = None
frame_right = None
lock = threading.Lock()

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>PIDS Dual Camera Test</title>
    <style>
        body { font-family: monospace; padding: 20px; background: #1a1a1a; color: #eee; }
        h1 { color: #4CAF50; }
        .container { display: flex; gap: 20px; flex-wrap: wrap; }
        .camera-box { background: #2d2d2d; padding: 15px; border-radius: 5px; }
        .camera-box h3 { margin-top: 0; }
        img { border: 2px solid #444; }
        .info { background: #333; padding: 10px; margin: 10px 0; border-radius: 3px; }
        a { color: #2196F3; margin-right: 10px; }
        .stats { font-size: 12px; color: #aaa; }
    </style>
    <script>
        setInterval(() => {
            fetch('/stats').then(r => r.json()).then(data => {
                document.getElementById('stats-left').innerText =
                    `Mean: ${data.left.mean.toFixed(1)}, Std: ${data.left.std.toFixed(1)}`;
                document.getElementById('stats-right').innerText =
                    `Mean: ${data.right.mean.toFixed(1)}, Std: ${data.right.std.toFixed(1)}`;
                document.getElementById('stats-diff').innerText =
                    `|Left - Right| Mean: ${data.diff.toFixed(2)}`;
            });
        }, 500);
    </script>
</head>
<body>
    <h1>PIDS Dual Camera Test</h1>

    <div class="info">
        <strong>Configuration:</strong><br>
        - Left (Camera 0): I∥ - 0° Polarizer<br>
        - Right (Camera 1): I⊥ - 90° Polarizer<br>
        - Output: YUV420 Y-channel (Grayscale)
    </div>

    <div class="container">
        <div class="camera-box">
            <h3>Left Camera (I∥)</h3>
            <img src="/stream/left" width="320" height="240">
            <p class="stats" id="stats-left">Loading...</p>
        </div>
        <div class="camera-box">
            <h3>Right Camera (I⊥)</h3>
            <img src="/stream/right" width="320" height="240">
            <p class="stats" id="stats-right">Loading...</p>
        </div>
        <div class="camera-box">
            <h3>Side by Side</h3>
            <img src="/stream/both" width="640" height="240">
            <p class="stats" id="stats-diff">Loading...</p>
        </div>
    </div>

    <div class="info">
        <h3>Actions:</h3>
        <a href="/capture">Capture & Save</a>
        <a href="/diff">Show Difference |I∥ - I⊥|</a>
    </div>

    <div class="info">
        <h3>Polarization Test:</h3>
        <p>Point at glass surface. On glass: I∥ >> I⊥ (left brighter than right)</p>
        <p>On diffuse background: I∥ ≈ I⊥ (similar brightness)</p>
    </div>
</body>
</html>
"""

# 統計數據
stats = {
    'left': {'mean': 0, 'std': 0},
    'right': {'mean': 0, 'std': 0},
    'diff': 0
}

def init_cameras():
    """初始化雙相機"""
    global cam_left, cam_right

    print("Initializing cameras...")

    # Camera 0 = Left (I∥)
    cam_left = Picamera2(0)
    config_left = cam_left.create_video_configuration(
        main={"size": (640, 480), "format": "YUV420"}
    )
    cam_left.configure(config_left)

    # Camera 1 = Right (I⊥)
    cam_right = Picamera2(1)
    config_right = cam_right.create_video_configuration(
        main={"size": (640, 480), "format": "YUV420"}
    )
    cam_right.configure(config_right)

    # 啟動相機
    cam_left.start()
    cam_right.start()

    print("Both cameras started!")
    time.sleep(1)

def extract_y_channel(frame):
    """從 YUV420 提取 Y 通道"""
    h, w = 480, 640
    y_plane = frame[:h, :w]
    if len(y_plane.shape) == 3:
        y_plane = y_plane[:, :, 0]
    return y_plane

def capture_thread():
    """背景執行緒：持續捕獲兩個相機"""
    global frame_left, frame_right, stats

    while True:
        try:
            # 捕獲
            raw_left = cam_left.capture_array()
            raw_right = cam_right.capture_array()

            # 提取 Y 通道
            y_left = extract_y_channel(raw_left)
            y_right = extract_y_channel(raw_right)

            # 更新統計
            with lock:
                stats['left']['mean'] = float(np.mean(y_left))
                stats['left']['std'] = float(np.std(y_left))
                stats['right']['mean'] = float(np.mean(y_right))
                stats['right']['std'] = float(np.std(y_right))
                stats['diff'] = float(np.mean(np.abs(y_left.astype(float) - y_right.astype(float))))

                frame_left = y_left.copy()
                frame_right = y_right.copy()

            time.sleep(0.03)  # ~30 fps

        except Exception as e:
            print(f"Capture error: {e}")
            time.sleep(0.1)

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/stats')
def get_stats():
    with lock:
        return stats

@app.route('/stream/<camera>')
def stream(camera):
    def gen():
        while True:
            with lock:
                if camera == 'left' and frame_left is not None:
                    frame = cv2.cvtColor(frame_left, cv2.COLOR_GRAY2BGR)
                    cv2.putText(frame, "Left (I//)", (10, 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                elif camera == 'right' and frame_right is not None:
                    frame = cv2.cvtColor(frame_right, cv2.COLOR_GRAY2BGR)
                    cv2.putText(frame, "Right (I_|_)", (10, 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                elif camera == 'both' and frame_left is not None and frame_right is not None:
                    left_bgr = cv2.cvtColor(frame_left, cv2.COLOR_GRAY2BGR)
                    right_bgr = cv2.cvtColor(frame_right, cv2.COLOR_GRAY2BGR)

                    # 縮小
                    left_small = cv2.resize(left_bgr, (320, 240))
                    right_small = cv2.resize(right_bgr, (320, 240))

                    # 加標籤
                    cv2.putText(left_small, "Left (I//)", (10, 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(right_small, "Right (I_|_)", (10, 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    frame = np.hstack([left_small, right_small])
                else:
                    # 黑畫面
                    frame = np.zeros((240, 320, 3), dtype=np.uint8)
                    cv2.putText(frame, "Loading...", (100, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.03)

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture')
def capture():
    """捕獲並保存"""
    with lock:
        if frame_left is not None and frame_right is not None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")

            cv2.imwrite(f'/tmp/left_{timestamp}.png', frame_left)
            cv2.imwrite(f'/tmp/right_{timestamp}.png', frame_right)

            # 計算差異
            diff = np.abs(frame_left.astype(float) - frame_right.astype(float))
            diff_norm = (diff / diff.max() * 255).astype(np.uint8) if diff.max() > 0 else diff.astype(np.uint8)
            cv2.imwrite(f'/tmp/diff_{timestamp}.png', diff_norm)

            return f"""
            <h2>Captured!</h2>
            <p>Files saved to /tmp/:</p>
            <ul>
                <li>left_{timestamp}.png</li>
                <li>right_{timestamp}.png</li>
                <li>diff_{timestamp}.png</li>
            </ul>
            <p>Left mean: {stats['left']['mean']:.1f}, Right mean: {stats['right']['mean']:.1f}</p>
            <p>|Left - Right| mean: {stats['diff']:.2f}</p>
            <p><a href="/">Back</a></p>
            <p>SCP: <code>scp tim@raspberrypi.local:/tmp/*_{timestamp}.png .</code></p>
            """

    return "No frames available. <a href='/'>Back</a>"

@app.route('/diff')
def show_diff():
    """顯示差異圖"""
    def gen():
        while True:
            with lock:
                if frame_left is not None and frame_right is not None:
                    # 計算差異
                    diff = np.abs(frame_left.astype(float) - frame_right.astype(float))

                    # 正規化到 0-255
                    if diff.max() > 0:
                        diff_norm = (diff / diff.max() * 255).astype(np.uint8)
                    else:
                        diff_norm = diff.astype(np.uint8)

                    # 用熱力圖顯示
                    diff_color = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)

                    cv2.putText(diff_color, f"|I// - I_|_| max={diff.max():.0f}", (10, 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    frame = diff_color
                else:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)

            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.03)

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("="*60)
    print("PIDS Dual Camera Test")
    print("="*60)
    print("Left Camera (0): I∥ - 0° Polarizer")
    print("Right Camera (1): I⊥ - 90° Polarizer")
    print("="*60)

    # 初始化相機
    init_cameras()

    # 啟動捕獲執行緒
    capture_th = threading.Thread(target=capture_thread, daemon=True)
    capture_th.start()

    print("Open http://<pi-ip>:5000")
    print("="*60)

    app.run(host='0.0.0.0', port=5000, threaded=True)
