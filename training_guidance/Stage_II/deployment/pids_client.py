#!/usr/bin/env python3
"""
PIDS Client (ZeroMQ)
用於 Raspberry Pi 連接遠端推理伺服器

使用: python pids_client.py --server <IP> --port 5555
"""
import argparse
import zmq
import numpy as np
import cv2
import time


class PIDSClient:
    def __init__(self, server_ip: str = "127.0.0.1", port: int = 5555,
                 timeout: int = 10000):
        """
        初始化 PIDS 客戶端

        Args:
            server_ip: 伺服器 IP 地址
            port: ZeroMQ 端口
            timeout: 超時時間 (ms)
        """
        self.server_ip = server_ip
        self.port = port

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout)
        self.socket.connect(f"tcp://{server_ip}:{port}")

        print(f"Connected to tcp://{server_ip}:{port}")

    def infer(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """
        發送圖像到伺服器進行推理

        Args:
            left: (H, W) uint8 灰階圖像 (I∥)
            right: (H, W) uint8 灰階圖像 (I⊥)

        Returns:
            disparity: (H, W) float32 視差圖
        """
        # 編碼為 PNG
        _, left_enc = cv2.imencode('.png', left,
                                    [cv2.IMWRITE_PNG_COMPRESSION, 3])
        _, right_enc = cv2.imencode('.png', right,
                                     [cv2.IMWRITE_PNG_COMPRESSION, 3])

        left_bytes = left_enc.tobytes()
        right_bytes = right_enc.tobytes()

        # 打包訊息: [left_size (4 bytes)][left_data][right_data]
        message = len(left_bytes).to_bytes(4, 'little') + left_bytes + right_bytes

        # 發送並接收
        t0 = time.time()
        self.socket.send(message)
        result = self.socket.recv()
        t1 = time.time()

        if result == b"ERROR":
            raise RuntimeError("Server returned error")

        # 解碼結果
        h, w = left.shape[:2]
        disparity = np.frombuffer(result, dtype=np.float32).reshape(h, w)

        return disparity, (t1 - t0)

    def close(self):
        """關閉連接"""
        self.socket.close()
        self.context.term()


def test_with_dummy():
    """用假圖像測試連接"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5555)
    args = parser.parse_args()

    client = PIDSClient(server_ip=args.server, port=args.port)

    # 生成測試圖像
    left = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    right = np.random.randint(0, 255, (480, 640), dtype=np.uint8)

    print("Sending test images...")

    for i in range(10):
        disparity, latency = client.infer(left, right)
        print(f"[{i+1}] Latency: {latency*1000:.1f}ms, "
              f"Disp range: [{disparity.min():.1f}, {disparity.max():.1f}]")

    client.close()
    print("Test complete!")


if __name__ == '__main__':
    test_with_dummy()
