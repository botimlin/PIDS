import threading
import cv2
import numpy as np
import time
from picamera2 import Picamera2

# ================= Settings =================
# Resolution setting (both cameras should match)
# Recommend using a lower resolution first for smoothness; increase for inference
RESOLUTION = (640, 480) 
FRAME_RATE = 30
# =========================================

class CameraStream:
    """
    Read camera frames in a dedicated thread to avoid blocking the main loop
    """
    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.running = False
        self.frame = None
        self.lock = threading.Lock()
        
        print(f"[Cam {camera_id}] Initializing...")
        self.picam2 = Picamera2(camera_num=camera_id)
        
        # Set camera parameters
        config = self.picam2.create_video_configuration(
            main={"size": RESOLUTION, "format": "BGR888", "fps": FRAME_RATE}
        )
        self.picam2.configure(config)
        self.picam2.start()
        print(f"[Cam {camera_id}] Started!")

    def start(self):
        if self.running:
            return self
        self.running = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while self.running:
            try:
                # grab latest frame
                img = self.picam2.capture_array()
                
                # thread-safe lock (GIL helps, but locking is good practice)
                with self.lock:
                    self.frame = img
            except Exception as e:
                print(f"[Cam {self.camera_id}] Error: {e}")
                self.running = False

    def read(self):
        with self.lock:
            return self.frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.picam2.stop()
        self.picam2.close()
        print(f"[Cam {self.camera_id}] Stopped.")

def main():
    print("=== PIDS Stereo Viewer ===")
    
    # 1. Initialize two cameras
    cam_left = CameraStream(0)
    cam_right = CameraStream(1)
    
    # 2. Start capture threads
    cam_left.start()
    cam_right.start()
    
    # Wait for cameras to warm up
    time.sleep(2)
    
    print("Starting Main Loop. Press 'q' to exit.")
    
    try:
        while True:
            # 3. Fetch latest frames
            frame_L = cam_left.read()
            frame_R = cam_right.read()
            
            # check both frames are available
            if frame_L is not None and frame_R is not None:
                
                # --- (calibration code can be inserted here in the future) ---
                # frame_L = apply_calibration(frame_L, 'L')
                # frame_R = apply_calibration(frame_R, 'R')
                # -------------------------------------------------------------
                
                # 4. Stitch frames (Horizontal Stack)
                # Left is Cam 0, Right is Cam 1
                combined_view = np.hstack((frame_L, frame_R))
                
                # 5. Display
                cv2.imshow('PIDS Stereo View (Left / Right)', combined_view)
            
            # press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
        
    finally:
        # 6. Graceful exit, release resources
        print("\nStopping cameras...")
        cam_left.stop()
        cam_right.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()