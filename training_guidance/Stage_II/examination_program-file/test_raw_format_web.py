#!/usr/bin/env python3
"""
Web 預覽 - 測試 IMX296 不同格式輸出
包含 raw 模式測試

執行: python test_raw_format_web.py
瀏覽: http://<pi-ip>:5000
"""
from flask import Flask, Response, render_template_string
from picamera2 import Picamera2
import cv2
import numpy as np

app = Flask(__name__)

# 全域變數
cam = None
current_format = "BGR888"
sensor_modes_str = ""

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>IMX296 Format Test</title>
    <style>
        body { font-family: monospace; padding: 20px; background: #1a1a1a; color: #eee; }
        h1 { color: #4CAF50; }
        a { color: #2196F3; margin-right: 10px; }
        pre { background: #333; padding: 10px; overflow-x: auto; }
        img { border: 2px solid #444; margin: 10px 0; }
        .info { background: #2d2d2d; padding: 15px; margin: 10px 0; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>IMX296 Format Test</h1>

    <div class="info">
        <strong>Current format:</strong> {{ format }}<br>
        <strong>Stream:</strong> <a href="/stream">Live Stream</a>
    </div>

    <img src="/stream" width="640">

    <h3>Switch Format:</h3>
    <a href="/set/BGR888">BGR888</a>
    <a href="/set/RGB888">RGB888</a>
    <a href="/set/YUV420">YUV420</a>
    <a href="/set/XBGR8888">XBGR8888</a>

    <h3>RAW Test:</h3>
    <a href="/raw">Capture RAW (check terminal)</a>
    <a href="/raw_gray">RAW as Grayscale Stream</a>

    <h3>Sensor Modes:</h3>
    <pre>{{ modes }}</pre>
</body>
</html>
"""

def get_sensor_modes():
    """取得 sensor modes 資訊"""
    global sensor_modes_str
    if sensor_modes_str:
        return sensor_modes_str

    try:
        temp_cam = Picamera2(0)
        lines = []
        for i, mode in enumerate(temp_cam.sensor_modes):
            lines.append(f"Mode {i}:")
            for key, value in mode.items():
                lines.append(f"  {key}: {value}")
            lines.append("")
        temp_cam.close()
        sensor_modes_str = "\n".join(lines)
    except Exception as e:
        sensor_modes_str = f"Error: {e}"

    return sensor_modes_str

def init_camera(fmt="BGR888"):
    """初始化相機"""
    global cam, current_format
    if cam:
        try:
            cam.stop()
            cam.close()
        except:
            pass

    cam = Picamera2(0)
    config = cam.create_video_configuration(main={"size": (640, 480), "format": fmt})
    cam.configure(config)
    cam.start()
    current_format = fmt
    print(f"Camera initialized with format: {fmt}")

@app.route('/')
def index():
    modes = get_sensor_modes()
    return render_template_string(HTML, format=current_format, modes=modes)

@app.route('/set/<fmt>')
def set_format(fmt):
    """切換格式"""
    try:
        init_camera(fmt)
        return f"""
        <p>Switched to <strong>{fmt}</strong></p>
        <p><a href="/">Back to main</a></p>
        """
    except Exception as e:
        return f"""
        <p>Error switching to {fmt}: {e}</p>
        <p><a href="/">Back to main</a></p>
        """

@app.route('/raw')
def test_raw():
    """測試 raw 捕獲並輸出資訊"""
    global cam

    # 暫停當前相機
    if cam:
        cam.stop()
        cam.close()

    result_lines = []

    try:
        cam = Picamera2(0)

        # 嘗試 raw 配置
        config = cam.create_still_configuration(raw={})
        result_lines.append(f"RAW Config: {config}")

        cam.configure(config)
        cam.start()

        # 捕獲
        arrays = cam.capture_arrays(["raw", "main"])
        raw_array = arrays[0]
        main_array = arrays[1]

        result_lines.append("")
        result_lines.append(f"RAW shape: {raw_array.shape}")
        result_lines.append(f"RAW dtype: {raw_array.dtype}")
        result_lines.append(f"RAW min: {raw_array.min()}, max: {raw_array.max()}")
        result_lines.append(f"RAW mean: {raw_array.mean():.2f}")
        result_lines.append("")
        result_lines.append(f"MAIN shape: {main_array.shape}")
        result_lines.append(f"MAIN dtype: {main_array.dtype}")

        # 保存 raw 到檔案
        np.save('/tmp/test_raw.npy', raw_array)
        result_lines.append("")
        result_lines.append("RAW saved to /tmp/test_raw.npy")

        cam.stop()
        cam.close()

    except Exception as e:
        result_lines.append(f"Error: {e}")
        import traceback
        result_lines.append(traceback.format_exc())

    # 恢復正常模式
    init_camera()

    result = "\n".join(result_lines)
    print(result)  # 也輸出到終端

    return f"""
    <h2>RAW Capture Result</h2>
    <pre>{result}</pre>
    <p><a href="/">Back to main</a></p>
    """

@app.route('/raw_gray')
def raw_gray_mode():
    """嘗試用 raw 作為灰階串流"""
    global cam, current_format

    if cam:
        cam.stop()
        cam.close()

    try:
        cam = Picamera2(0)

        # 嘗試直接用較低位元的格式
        # 或者用 raw stream
        config = cam.create_video_configuration(
            main={"size": (640, 480), "format": "YUV420"}
        )
        cam.configure(config)
        cam.start()
        current_format = "YUV420 (Y channel = grayscale)"

        return f"""
        <p>Switched to YUV420 mode (Y channel is grayscale)</p>
        <p><a href="/">Back to main</a></p>
        """
    except Exception as e:
        init_camera()
        return f"""
        <p>Error: {e}</p>
        <p><a href="/">Back to main</a></p>
        """

def gen_frames():
    """生成視訊串流"""
    while True:
        try:
            frame = cam.capture_array()

            # 處理不同格式
            if len(frame.shape) == 2:
                # 灰階
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                # XBGR8888
                frame = frame[:, :, :3]
            elif current_format == "YUV420":
                # YUV420 - 只取 Y channel 作為灰階
                h, w = frame.shape[:2]
                y_channel = frame[:h//2*2, :w//2*2, 0]  # Y channel
                frame = cv2.cvtColor(y_channel, cv2.COLOR_GRAY2BGR)
            elif current_format == "RGB888":
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 加上格式標籤
            cv2.putText(frame, f"Format: {current_format}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        except Exception as e:
            print(f"Stream error: {e}")
            break

@app.route('/stream')
def stream():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("="*60)
    print("IMX296 Format Test Server")
    print("="*60)
    print("Open http://<pi-ip>:5000 in browser")
    print("="*60)

    # 先取得 sensor modes
    get_sensor_modes()

    # 初始化相機
    init_camera()

    # 啟動伺服器
    app.run(host='0.0.0.0', port=5000, threaded=True)
