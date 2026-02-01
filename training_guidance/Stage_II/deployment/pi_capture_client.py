#!/usr/bin/env python3
"""
Raspberry Pi: 相機採集 + PIDS 推理客戶端

執行: python pi_capture_client.py --server <IP> --port 5555
"""
import argparse
import time
import numpy as np
import cv2
from pids_client import PIDSClient

try:
    from picamera2 import Picamera2
    HAS_PICAMERA = True
except ImportError:
    HAS_PICAMERA = False
    print("Warning: picamera2 not available, using dummy images")


class PIDSCaptureClient:
    def __init__(self, server_ip: str, port: int = 5555):
        # 初始化相機
        if HAS_PICAMERA:
            self._init_cameras()

        # 初始化客戶端
        self.client = PIDSClient(server_ip=server_ip, port=port)

        # 統計
        self.frame_count = 0
        self.t_start = time.time()

    def _init_cameras(self):
        """初始化雙相機"""
        print("Initializing cameras...")

        # Camera 0 = Left (I∥)
        self.cam_left = Picamera2(0)
        config_left = self.cam_left.create_video_configuration(
            main={"size": (640, 480), "format": "YUV420"}
        )
        self.cam_left.configure(config_left)

        # Camera 1 = Right (I⊥)
        self.cam_right = Picamera2(1)
        config_right = self.cam_right.create_video_configuration(
            main={"size": (640, 480), "format": "YUV420"}
        )
        self.cam_right.configure(config_right)

        self.cam_left.start()
        self.cam_right.start()

        time.sleep(1)
        print("Cameras ready!")

    def _extract_y_channel(self, frame: np.ndarray) -> np.ndarray:
        """從 YUV420 提取 Y 通道"""
        h, w = 480, 640
        y_plane = frame[:h, :w]
        if len(y_plane.shape) == 3:
            y_plane = y_plane[:, :, 0]
        return y_plane

    def capture(self) -> tuple:
        """採集左右圖像"""
        if HAS_PICAMERA:
            left_raw = self.cam_left.capture_array()
            right_raw = self.cam_right.capture_array()
            left = self._extract_y_channel(left_raw)
            right = self._extract_y_channel(right_raw)
        else:
            # Dummy images for testing
            left = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
            right = np.random.randint(0, 255, (480, 640), dtype=np.uint8)

        return left, right

    def run_once(self) -> tuple:
        """採集一次並推理"""
        # 採集
        t0 = time.time()
        left, right = self.capture()
        t_capture = time.time() - t0

        # 推理
        disparity, t_infer = self.client.infer(left, right)

        self.frame_count += 1

        return left, right, disparity, t_capture, t_infer

    def run_continuous(self, save_dir: str = None):
        """連續採集和推理"""
        print("\nStarting continuous capture...")
        print("Press Ctrl+C to stop\n")

        try:
            while True:
                left, right, disparity, t_cap, t_infer = self.run_once()

                # 計算 FPS
                elapsed = time.time() - self.t_start
                fps = self.frame_count / elapsed

                print(f"[{self.frame_count}] "
                      f"Capture: {t_cap*1000:.1f}ms, "
                      f"Infer: {t_infer*1000:.1f}ms, "
                      f"FPS: {fps:.1f}, "
                      f"Disp: [{disparity.min():.1f}, {disparity.max():.1f}]")

                # 保存 (可選)
                if save_dir and self.frame_count % 30 == 0:
                    cv2.imwrite(f'{save_dir}/left_{self.frame_count:04d}.png', left)
                    cv2.imwrite(f'{save_dir}/right_{self.frame_count:04d}.png', right)
                    disp_vis = (disparity / disparity.max() * 255).astype(np.uint8)
                    cv2.imwrite(f'{save_dir}/disp_{self.frame_count:04d}.png', disp_vis)

        except KeyboardInterrupt:
            print("\nStopping...")

        finally:
            self.close()

    def close(self):
        """清理資源"""
        self.client.close()
        if HAS_PICAMERA:
            self.cam_left.stop()
            self.cam_right.stop()
            self.cam_left.close()
            self.cam_right.close()


def main():
    parser = argparse.ArgumentParser(description='PIDS Capture Client')
    parser.add_argument('--server', type=str, required=True,
                        help='Server IP address')
    parser.add_argument('--port', type=int, default=5555,
                        help='Server port')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save images (optional)')

    args = parser.parse_args()

    client = PIDSCaptureClient(
        server_ip=args.server,
        port=args.port,
    )

    client.run_continuous(save_dir=args.save_dir)


if __name__ == '__main__':
    main()
