import threading
import cv2
import numpy as np
import time
from picamera2 import Picamera2

# ================= 設定區 =================
# 解析度設定 (兩顆要一樣)
# 建議先用低一點的解析度確保流暢，推論時再調高
RESOLUTION = (640, 480) 
FRAME_RATE = 30
# =========================================

class CameraStream:
    """
    使用獨立執行緒讀取相機，避免卡住主迴圈
    """
    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.running = False
        self.frame = None
        self.lock = threading.Lock()
        
        print(f"[Cam {camera_id}] Initializing...")
        self.picam2 = Picamera2(camera_num=camera_id)
        
        # 設定相機參數
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
                # 抓取最新影像
                img = self.picam2.capture_array()
                
                # 線程安全鎖 (雖然 Python GIL 會幫忙，但加鎖是好習慣)
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
    
    # 1. 初始化兩顆鏡頭
    cam_left = CameraStream(0)
    cam_right = CameraStream(1)
    
    # 2. 啟動擷取執行緒
    cam_left.start()
    cam_right.start()
    
    # 等待相機暖機
    time.sleep(2)
    
    print("Starting Main Loop. Press 'q' to exit.")
    
    try:
        while True:
            # 3. 獲取最新畫面
            frame_L = cam_left.read()
            frame_R = cam_right.read()
            
            # 檢查是否都讀到了
            if frame_L is not None and frame_R is not None:
                
                # --- (未來這裡可以插入校正代碼) ---
                # frame_L = apply_calibration(frame_L, 'L')
                # frame_R = apply_calibration(frame_R, 'R')
                # --------------------------------
                
                # 4. 拼接畫面 (Horizontal Stack)
                # 左邊是 Cam 0，右邊是 Cam 1
                combined_view = np.hstack((frame_L, frame_R))
                
                # 5. 顯示
                cv2.imshow('PIDS Stereo View (Left / Right)', combined_view)
            
            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
        
    finally:
        # 6. 優雅退出，釋放資源
        print("\nStopping cameras...")
        cam_left.stop()
        cam_right.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()