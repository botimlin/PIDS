#!/usr/bin/env python3
"""
測試不同灰階提取方法的偏振對比效果
比較: YUV420 Y通道 vs Bayer RAW G通道

執行: python test_grayscale_methods.py
瀏覽: http://<pi-ip>:5000
"""
from flask import Flask, Response, render_template_string
from picamera2 import Picamera2
import cv2
import numpy as np
import time

app = Flask(__name__)

# 全域變數
cam = None
current_mode = "Y_channel"  # Y_channel, G_channel, RGB, side_by_side

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Grayscale Method Test</title>
    <style>
        body { font-family: monospace; padding: 20px; background: #1a1a1a; color: #eee; }
        h1 { color: #4CAF50; }
        a { color: #2196F3; margin-right: 15px; padding: 5px 10px;
            background: #333; border-radius: 3px; text-decoration: none; }
        a:hover { background: #444; }
        a.active { background: #4CAF50; color: #000; }
        .info { background: #2d2d2d; padding: 15px; margin: 10px 0; border-radius: 5px; }
        img { border: 2px solid #444; margin: 10px 0; }
        pre { background: #333; padding: 10px; font-size: 12px; }
        .controls { margin: 20px 0; }
    </style>
    <script>
        // 每秒更新統計
        setInterval(() => {
            fetch('/stats').then(r => r.text()).then(t => {
                document.getElementById('stats').innerHTML = t;
            });
        }, 1000);
    </script>
</head>
<body>
    <h1>Grayscale Extraction Test</h1>

    <div class="info">
        <strong>Current Mode:</strong> {{ mode }}<br>
        <strong>Purpose:</strong> 比較不同灰階方法對偏振訊號的保留程度
    </div>

    <div class="controls">
        <a href="/mode/Y_channel" {% if mode == 'Y_channel' %}class="active"{% endif %}>YUV420 Y通道</a>
        <a href="/mode/G_channel" {% if mode == 'G_channel' %}class="active"{% endif %}>Bayer G通道</a>
        <a href="/mode/RGB" {% if mode == 'RGB' %}class="active"{% endif %}>RGB (參考)</a>
        <a href="/mode/side_by_side" {% if mode == 'side_by_side' %}class="active"{% endif %}>並排比較</a>
    </div>

    <img src="/stream" width="640">

    <div class="info">
        <h3>即時統計:</h3>
        <pre id="stats">Loading...</pre>
    </div>

    <div class="info">
        <h3>說明:</h3>
        <ul>
            <li><strong>Y通道:</strong> YUV420 的亮度通道 (0.299R + 0.587G + 0.114B)</li>
            <li><strong>G通道:</strong> 從 Bayer RAW 提取綠色像素 (50% 採樣)</li>
            <li><strong>偏振測試:</strong> 對準玻璃表面，觀察哪個方法的對比更明顯</li>
        </ul>
    </div>

    <h3>單張捕獲 (高品質分析):</h3>
    <a href="/capture">捕獲並分析</a>
</body>
</html>
"""

# 統計數據
stats = {
    'mean': 0,
    'std': 0,
    'min': 0,
    'max': 0,
    'contrast': 0
}

def init_camera_yuv():
    """初始化 YUV420 模式"""
    global cam
    if cam:
        cam.stop()
        cam.close()

    cam = Picamera2(0)
    config = cam.create_video_configuration(
        main={"size": (640, 480), "format": "YUV420"}
    )
    cam.configure(config)
    cam.start()
    time.sleep(0.5)

def init_camera_raw():
    """初始化 RAW 模式"""
    global cam
    if cam:
        cam.stop()
        cam.close()

    cam = Picamera2(0)
    # 同時捕獲 main (用於預覽) 和 raw
    config = cam.create_video_configuration(
        main={"size": (640, 480), "format": "BGR888"},
        raw={"size": (1456, 1088)}
    )
    cam.configure(config)
    cam.start()
    time.sleep(0.5)

def init_camera_rgb():
    """初始化 RGB 模式"""
    global cam
    if cam:
        cam.stop()
        cam.close()

    cam = Picamera2(0)
    config = cam.create_video_configuration(
        main={"size": (640, 480), "format": "BGR888"}
    )
    cam.configure(config)
    cam.start()
    time.sleep(0.5)

def extract_y_channel(frame):
    """從 YUV420 提取 Y 通道"""
    # YUV420 格式: Y plane 在前，完整解析度
    h, w = 480, 640
    y_plane = frame[:h, :w]
    if len(y_plane.shape) == 3:
        y_plane = y_plane[:, :, 0]
    return y_plane

def extract_g_channel_from_bayer(raw):
    """
    從 Bayer RAW (RGGB) 提取 G 通道
    RGGB pattern:
    R G R G ...
    G B G B ...
    """
    # raw 是 10-bit packed，需要處理
    # 取 G 像素 (位置 [0,1], [1,0] 在每個 2x2 block)
    h, w = raw.shape[:2]

    # 取兩個 G 通道的平均
    g1 = raw[0::2, 1::2]  # 第一行的 G (位置 0,1)
    g2 = raw[1::2, 0::2]  # 第二行的 G (位置 1,0)

    # 平均兩個 G
    g_avg = ((g1.astype(np.float32) + g2.astype(np.float32)) / 2).astype(np.uint8)

    # 縮放到目標解析度
    g_resized = cv2.resize(g_avg, (640, 480), interpolation=cv2.INTER_LINEAR)

    return g_resized

def update_stats(gray_frame):
    """更新統計數據"""
    global stats
    stats['mean'] = float(np.mean(gray_frame))
    stats['std'] = float(np.std(gray_frame))
    stats['min'] = int(np.min(gray_frame))
    stats['max'] = int(np.max(gray_frame))
    stats['contrast'] = stats['max'] - stats['min']

@app.route('/')
def index():
    return render_template_string(HTML, mode=current_mode)

@app.route('/mode/<mode>')
def set_mode(mode):
    global current_mode
    current_mode = mode

    if mode == 'Y_channel':
        init_camera_yuv()
    elif mode == 'G_channel':
        init_camera_raw()
    elif mode == 'RGB':
        init_camera_rgb()
    elif mode == 'side_by_side':
        init_camera_raw()  # 用 raw 模式可以同時取得兩者

    return render_template_string(HTML, mode=current_mode)

@app.route('/stats')
def get_stats():
    return f"""Mean: {stats['mean']:.1f}
Std:  {stats['std']:.1f}
Min:  {stats['min']}
Max:  {stats['max']}
Contrast: {stats['contrast']}"""

@app.route('/capture')
def capture_analysis():
    """捕獲並詳細分析"""
    global cam

    results = []

    # 1. Y 通道分析
    init_camera_yuv()
    time.sleep(0.3)
    frame_yuv = cam.capture_array()
    y_channel = extract_y_channel(frame_yuv)

    results.append({
        'name': 'Y Channel (YUV420)',
        'mean': np.mean(y_channel),
        'std': np.std(y_channel),
        'min': np.min(y_channel),
        'max': np.max(y_channel),
        'shape': y_channel.shape
    })

    # 保存
    cv2.imwrite('/tmp/y_channel.png', y_channel)

    # 2. G 通道分析
    init_camera_raw()
    time.sleep(0.3)
    arrays = cam.capture_arrays(["main", "raw"])
    raw = arrays[1]  # tuple: (main, raw)

    # 處理 raw (可能是 16-bit)
    if raw.dtype == np.uint16:
        raw_8bit = (raw >> 2).astype(np.uint8)  # 10-bit to 8-bit
    else:
        raw_8bit = raw

    g_channel = extract_g_channel_from_bayer(raw_8bit)

    results.append({
        'name': 'G Channel (Bayer RAW)',
        'mean': np.mean(g_channel),
        'std': np.std(g_channel),
        'min': np.min(g_channel),
        'max': np.max(g_channel),
        'shape': g_channel.shape,
        'raw_shape': raw.shape,
        'raw_dtype': str(raw.dtype)
    })

    # 保存
    cv2.imwrite('/tmp/g_channel.png', g_channel)

    # 3. RGB 灰階
    init_camera_rgb()
    time.sleep(0.3)
    frame_rgb = cam.capture_array()
    gray_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)

    results.append({
        'name': 'RGB to Gray (cv2)',
        'mean': np.mean(gray_rgb),
        'std': np.std(gray_rgb),
        'min': np.min(gray_rgb),
        'max': np.max(gray_rgb),
        'shape': gray_rgb.shape
    })

    cv2.imwrite('/tmp/rgb_gray.png', gray_rgb)

    # 生成報告
    html = "<h2>Capture Analysis</h2>"
    html += "<p>Images saved to /tmp/</p>"

    for r in results:
        html += f"""
        <div style="background:#333; padding:10px; margin:10px 0;">
        <h3>{r['name']}</h3>
        <pre>
Shape: {r['shape']}
Mean:  {r['mean']:.2f}
Std:   {r['std']:.2f}
Min:   {r['min']}
Max:   {r['max']}
Contrast: {r['max'] - r['min']}
        </pre>
        </div>
        """
        if 'raw_shape' in r:
            html += f"<pre>RAW shape: {r['raw_shape']}, dtype: {r['raw_dtype']}</pre>"

    # 顯示並排圖
    html += """
    <h3>Saved Files:</h3>
    <ul>
        <li>/tmp/y_channel.png</li>
        <li>/tmp/g_channel.png</li>
        <li>/tmp/rgb_gray.png</li>
    </ul>
    <p>Use SCP to download: <code>scp pi@raspberrypi.local:/tmp/*.png .</code></p>
    """

    html += '<p><a href="/">Back</a></p>'

    # 恢復當前模式
    if current_mode == 'Y_channel':
        init_camera_yuv()
    elif current_mode == 'G_channel':
        init_camera_raw()
    else:
        init_camera_rgb()

    return html

def gen_frames():
    """生成視訊串流"""
    global current_mode

    while True:
        try:
            if current_mode == 'Y_channel':
                frame = cam.capture_array()
                gray = extract_y_channel(frame)
                update_stats(gray)
                display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                label = "Y Channel (YUV420)"

            elif current_mode == 'G_channel':
                arrays = cam.capture_arrays(["main", "raw"])
                raw = arrays[1]  # tuple: (main, raw)

                if raw.dtype == np.uint16:
                    raw_8bit = (raw >> 2).astype(np.uint8)
                else:
                    raw_8bit = raw

                gray = extract_g_channel_from_bayer(raw_8bit)
                update_stats(gray)
                display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                label = "G Channel (Bayer)"

            elif current_mode == 'RGB':
                frame = cam.capture_array()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                update_stats(gray)
                display = frame.copy()
                label = "RGB"

            elif current_mode == 'side_by_side':
                arrays = cam.capture_arrays(["main", "raw"])
                main_frame = arrays[0]  # tuple: (main, raw)
                raw = arrays[1]

                # Y from main (approximate)
                y_approx = cv2.cvtColor(main_frame, cv2.COLOR_BGR2GRAY)

                # G from raw
                if raw.dtype == np.uint16:
                    raw_8bit = (raw >> 2).astype(np.uint8)
                else:
                    raw_8bit = raw
                g_channel = extract_g_channel_from_bayer(raw_8bit)

                # 並排
                y_bgr = cv2.cvtColor(y_approx, cv2.COLOR_GRAY2BGR)
                g_bgr = cv2.cvtColor(g_channel, cv2.COLOR_GRAY2BGR)

                # 縮小各一半
                y_small = cv2.resize(y_bgr, (320, 240))
                g_small = cv2.resize(g_bgr, (320, 240))

                # 加標籤
                cv2.putText(y_small, "Y Channel", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(g_small, "G Channel", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                display = np.hstack([y_small, g_small])
                display = cv2.resize(display, (640, 240))

                # 加黑邊讓高度一致
                display = cv2.copyMakeBorder(display, 120, 120, 0, 0,
                                            cv2.BORDER_CONSTANT, value=(0,0,0))
                label = "Side by Side"
                update_stats(y_approx)

            else:
                frame = cam.capture_array()
                display = frame
                label = "Unknown"

            # 加上模式標籤
            cv2.putText(display, label, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 加上統計
            cv2.putText(display, f"Mean:{stats['mean']:.0f} Std:{stats['std']:.0f}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            _, buffer = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        except Exception as e:
            print(f"Stream error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.5)

@app.route('/stream')
def stream():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("="*60)
    print("Grayscale Method Test")
    print("="*60)
    print("Modes:")
    print("  - Y Channel: YUV420 luminance")
    print("  - G Channel: Bayer RAW green pixels")
    print("  - RGB: Full color (reference)")
    print("  - Side by Side: Compare Y vs G")
    print("="*60)
    print("Open http://<pi-ip>:5000")
    print("="*60)

    # 默認用 Y 通道
    init_camera_yuv()

    app.run(host='0.0.0.0', port=5000, threaded=True)
