import time
import threading
import cv2
import numpy as np
import os
from flask import Flask, Response
from picamera2 import Picamera2

# ================= Global Settings =================
RESOLUTION = (640, 480)  # Resolution used for calibration and inference
OUTPUT_DIR = "./output/calibration_data"

# Initial camera parameters
current_exposure = 20000  # 20ms
current_gain = 1.0
is_tuning_mode = True     # whether in tuning mode

# Image buffer (for Web Stream)
global_frame = None
lock = threading.Lock()

# Capture buffer (for calibration computation)
capture_queue = []        # used to temporarily store images to average
frames_to_capture = 0     # countdown: how many frames left to capture
capture_label = ""        # label of current capture (for display)

app = Flask(__name__)
# ===========================================

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def apply_controls(picam):
    """Force-write global Exposure/Gain variables to the camera"""
    try:
        controls = {
            "ExposureTime": current_exposure,
            "AnalogueGain": current_gain,
            "AwbMode": 0,           # disable auto white balance
            "ColourGains": (1.5, 1.5) # fixed white balance (avoid color drift affecting calibration)
        }
        picam.set_controls(controls)
    except Exception as e:
        print(f"[Warning] Control set failed: {e}")

def camera_thread_func():
    """Camera background thread: handles preview + capture"""
    global global_frame, frames_to_capture, capture_queue
    
    print("[System] Initializing Dual Cameras...")
    cam0 = Picamera2(camera_num=0)
    config0 = cam0.create_video_configuration(main={"size": RESOLUTION, "format": "BGR888"})
    cam0.configure(config0)
    cam0.start()

    cam1 = Picamera2(camera_num=1)
    config1 = cam1.create_video_configuration(main={"size": RESOLUTION, "format": "BGR888"})
    cam1.configure(config1)
    cam1.start()
    print("[System] Cameras Running.")

    while True:
        try:
            # 1. Apply current parameters every frame (ensure lock)
            apply_controls(cam0)
            apply_controls(cam1)

            # 2. Capture images
            img0 = cam0.capture_array()
            img1 = cam1.capture_array()

            # 3. Handle capture logic (if main program requests photos)
            if frames_to_capture > 0:
                # convert to float32 and store
                # Note: stored image is RGB or BGR depending on needs, OpenCV defaults to BGR
                # For computation convenience, we either combine or store separately
                capture_queue.append((img0.astype(np.float32), img1.astype(np.float32)))
                frames_to_capture -= 1
                status_text = f"CAPTURING {capture_label}: {len(capture_queue)}"
                color_status = (0, 0, 255) # red indicates recording
            else:
                status_text = "PREVIEW MODE"
                color_status = (0, 255, 255) # yellow indicates preview

            # 4. Prepare Web preview frame
            # compute mean brightness to show to user
            mean0 = np.mean(img0)
            mean1 = np.mean(img1)

            # overlay info
            cv2.putText(img0, f"L: {mean0:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img1, f"R: {mean1:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            combined = np.hstack((img0, img1))
            
            # draw bottom parameter info
            info = f"Exp: {current_exposure} | Gain: {current_gain:.1f} | {status_text}"
            cv2.putText(combined, info, (20, combined.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_status, 2)

            with lock:
                global_frame = combined
            
            time.sleep(0.02)

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

# ================= Web Stream =================
def generate_frames():
    while True:
        with lock:
            if global_frame is None:
                time.sleep(0.05)
                continue
            output = global_frame.copy()
        
        # compress to JPEG for transmission
        (flag, encoded) = cv2.imencode(".jpg", output, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        if not flag: continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded) + b'\r\n')

@app.route("/")
def index():
    return "<h1>Manual Calibration Tool</h1><img src='/video_feed' width='100%'>"

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ================= Main Control Logic =================
def capture_batch(label, count=30):
    """Trigger background thread to take photos and wait until done"""
    global frames_to_capture, capture_queue, capture_label
    
    # clear buffer
    capture_queue = []
    capture_label = label
    frames_to_capture = count
    
    print(f"\n[Action] Starting capture for {label}...")
    
    # wait until capture finishes
    while frames_to_capture > 0:
        time.sleep(0.1)
        print(f"  Capturing... remaining: {frames_to_capture}", end='\r')
    
    print(f"\n[Action] Capture {label} finished. Processing...")
    
    # compute averages
    # capture_queue contains [(L1, R1), (L2, R2), ...]
    # split them to compute averages
    list_L = [pair[0] for pair in capture_queue]
    list_R = [pair[1] for pair in capture_queue]
    
    avg_L = np.mean(list_L, axis=0)
    avg_R = np.mean(list_R, axis=0)
    
    return avg_L, avg_R

def main_logic():
    global current_exposure, current_gain
    
    ensure_dir(OUTPUT_DIR)
    
    print("\n" + "="*50)
    print(" 📸 PIDS Manual Calibration System")
    print(" Please open browser: http://<raspberry_pi_ip>:5000")
    print("="*50)

    # --- Step 1: Parameter Tuning (for Flat Field) ---
    print("\n[Step 1: Parameter Tuning]")
    print("Please place a white paper or point at a white light source.")
    print("Use keys below to adjust, target brightness: 130 ~ 160 (green text)")
    print("  u/d : Exposure +/- 1000")
    print("  U/D : Exposure +/- 5000")
    print("  g/f : Gain +/- 0.1")
    print("  Enter : Lock parameters and continue")
    
    while True:
        cmd = input("Command > ").strip()
        if cmd == '':
            print(f"🔒 Parameters locked! Exp={current_exposure}, Gain={current_gain}")
            break
        elif cmd == 'u': current_exposure += 1000
        elif cmd == 'd': current_exposure = max(100, current_exposure - 1000)
        elif cmd == 'U': current_exposure += 5000
        elif cmd == 'D': current_exposure = max(100, current_exposure - 5000)
        elif cmd == 'g': current_gain += 0.1
        elif cmd == 'f': current_gain = max(1.0, current_gain - 0.1)
        
        print(f"Set -> Exp: {current_exposure} | Gain: {current_gain:.1f}")

    # --- Step 2: Flat/Dark Field Capture ---
    # In practice it doesn't matter which is captured first as long as parameters remain unchanged.
    # Often we capture Dark first, but since we were just adjusting using white paper we briefly considered Flat first.
    # To be rigorous we will follow the standard flow:
    
    print("\n[Step 2: Dark Field Capture]")
    print("❗❗ Please COVER the lens cap to ensure full darkness ❗❗")
    input("When ready, press Enter to start capturing 30 dark frames...")
    
    dark_L, dark_R = capture_batch("DARK", 30)
    
    # save dark frames
    np.save(os.path.join(OUTPUT_DIR, "master_dark_L.npy"), dark_L)
    np.save(os.path.join(OUTPUT_DIR, "master_dark_R.npy"), dark_R)
    print("✅ Dark Field saved.")

    # --- Step 3: Flat Field Capture ---
    print("\n[Step 3: Flat Field Capture]")
    print("💡 Remove the lens cap and put back the white paper/light source.")
    print("Ensure lighting is the same as during tuning!")
    input("When ready, press Enter to start capturing 30 flat frames...")
    
    flat_L_raw, flat_R_raw = capture_batch("FLAT", 30)
    
    # --- Step 4: Compute Gain Map ---
    print("\n[Step 4: Compute and Generate Calibration Files]")
    
    # numerical computation
    flat_L_clean = flat_L_raw - dark_L
    flat_R_clean = flat_R_raw - dark_R
    
    # safety (avoid division by zero)
    flat_L_clean[flat_L_clean < 1] = 1
    flat_R_clean[flat_R_clean < 1] = 1
    
    # compute gain
    gain_L = np.mean(flat_L_clean) / flat_L_clean
    gain_R = np.mean(flat_R_clean) / flat_R_clean
    
    # save gain maps
    np.save(os.path.join(OUTPUT_DIR, "gain_map_L.npy"), gain_L)
    np.save(os.path.join(OUTPUT_DIR, "gain_map_R.npy"), gain_R)
    
    # generate preview images (visual check)
    vis_L = cv2.normalize(gain_L, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vis_R = cv2.normalize(gain_R, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    cv2.imwrite(os.path.join(OUTPUT_DIR, "vis_gain_L.png"), vis_L)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "vis_gain_R.png"), vis_R)
    
    print(f"🎉 Calibration complete! All files saved to: {OUTPUT_DIR}")
    print("The program will exit in 10 seconds...")
    time.sleep(10)
    os._exit(0)

if __name__ == "__main__":
    # 1. Start camera background thread
    t_cam = threading.Thread(target=camera_thread_func)
    t_cam.daemon = True
    t_cam.start()

    # 2. Start web server background thread
    t_web = threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False))
    t_web.daemon = True
    t_web.start()
    
    # 3. Start main console logic
    # wait a bit for cameras to start
    time.sleep(3)
    main_logic()