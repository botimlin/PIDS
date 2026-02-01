#!/usr/bin/env python3
"""
PIDS Real-time Inference Server (ZeroMQ)
用於 Vast.ai / 任何 GPU 伺服器

啟動: python realtime_server.py --port 5555 --iters 12 --fp16
"""
import argparse
import time
import zmq
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import sys
import os

# 加入 core 路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class PIDSRealtimeServer:
    def __init__(self, checkpoint_path: str, port: int = 5555,
                 iters: int = 12, use_fp16: bool = True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.iters = iters
        self.use_fp16 = use_fp16

        print(f"Device: {self.device}")
        print(f"Iterations: {iters}")
        print(f"FP16: {use_fp16}")

        # 載入模型
        self.model = self._load_model(checkpoint_path)

        # ZeroMQ 設置
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")

        # 預熱
        self._warmup()

        print(f"\nServer ready at tcp://*:{port}")
        print("Waiting for connections...")

    def _load_model(self, checkpoint_path: str):
        """載入 PIDS 模型"""
        print(f"Loading model from {checkpoint_path}...")

        try:
            # 嘗試載入 PIDS 模型
            from pids_model import PIDSStereoPolVolume

            model = PIDSStereoPolVolume(
                hidden_dims=[128, 128, 128],
                n_downsample=2,
                pol_dim=32,
            )

            if os.path.exists(checkpoint_path):
                state_dict = torch.load(checkpoint_path, map_location=self.device)
                model.load_state_dict(state_dict, strict=False)
                print("Checkpoint loaded!")
            else:
                print(f"Warning: Checkpoint not found at {checkpoint_path}")
                print("Using random weights for testing")

        except ImportError:
            print("Warning: pids_model not found, using RAFT-Stereo")
            # Fallback to RAFT-Stereo
            from core.raft_stereo import RAFTStereo

            class Args:
                hidden_dims = [128, 128, 128]
                corr_implementation = "reg"
                shared_backbone = False
                corr_levels = 4
                corr_radius = 4
                n_downsample = 2
                context_norm = "batch"
                slow_fast_gru = False
                n_gru_layers = 3
                mixed_precision = False

            model = RAFTStereo(Args())

            if os.path.exists(checkpoint_path):
                state_dict = torch.load(checkpoint_path, map_location=self.device)
                model.load_state_dict(state_dict, strict=False)
                print("RAFT-Stereo checkpoint loaded!")

        model = model.to(self.device)
        model.eval()

        if self.use_fp16 and self.device.type == 'cuda':
            model = model.half()
            print("Model converted to FP16")

        return model

    def _warmup(self):
        """預熱 GPU，避免第一次推理慢"""
        print("\nWarming up GPU...")

        dummy_left = torch.randn(1, 1, 480, 640).to(self.device)
        dummy_right = torch.randn(1, 1, 480, 640).to(self.device)

        if self.use_fp16 and self.device.type == 'cuda':
            dummy_left = dummy_left.half()
            dummy_right = dummy_right.half()

        with torch.no_grad():
            for i in range(5):
                _ = self._infer_tensor(dummy_left, dummy_right)
                print(f"  Warmup {i+1}/5")

        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        print("Warmup complete!")

    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        """預處理: numpy (H,W) -> tensor (1,1,H,W)"""
        tensor = torch.from_numpy(img).float() / 255.0
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(self.device)

        if self.use_fp16 and self.device.type == 'cuda':
            tensor = tensor.half()

        return tensor

    def _infer_tensor(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """執行推理"""
        with torch.no_grad():
            # RAFT-Stereo / PIDS 推理
            _, disp = self.model(left, right, iters=self.iters, test_mode=True)
        return disp

    def infer(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """完整推理流程"""
        left_tensor = self._preprocess(left)
        right_tensor = self._preprocess(right)

        disp = self._infer_tensor(left_tensor, right_tensor)

        # 轉回 numpy
        disp_np = disp.squeeze().cpu().float().numpy()
        return disp_np.astype(np.float32)

    def run(self):
        """主循環"""
        frame_count = 0
        t_start = time.time()

        while True:
            try:
                # 接收訊息
                message = self.socket.recv()

                # 解碼圖像
                # 格式: [4 bytes: left_size][left_png][right_png]
                left_size = int.from_bytes(message[:4], 'little')
                left_data = message[4:4+left_size]
                right_data = message[4+left_size:]

                left = cv2.imdecode(
                    np.frombuffer(left_data, np.uint8),
                    cv2.IMREAD_GRAYSCALE
                )
                right = cv2.imdecode(
                    np.frombuffer(right_data, np.uint8),
                    cv2.IMREAD_GRAYSCALE
                )

                # 推理
                t0 = time.time()
                disparity = self.infer(left, right)
                t1 = time.time()

                # 發送結果
                self.socket.send(disparity.tobytes())

                # 統計
                frame_count += 1
                if frame_count % 10 == 0:
                    elapsed = time.time() - t_start
                    fps = frame_count / elapsed
                    print(f"[{frame_count}] FPS: {fps:.1f}, "
                          f"Inference: {(t1-t0)*1000:.1f}ms, "
                          f"Disp range: [{disparity.min():.1f}, {disparity.max():.1f}]")

            except KeyboardInterrupt:
                print("\nShutting down...")
                break
            except Exception as e:
                print(f"Error: {e}")
                # 發送錯誤響應
                self.socket.send(b"ERROR")

        self.socket.close()
        self.context.term()


def main():
    parser = argparse.ArgumentParser(description='PIDS Real-time Server')
    parser.add_argument('--checkpoint', type=str,
                        default='./checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--port', type=int, default=5555,
                        help='ZeroMQ port')
    parser.add_argument('--iters', type=int, default=12,
                        help='Number of GRU iterations')
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='Use FP16 inference')
    parser.add_argument('--no-fp16', dest='fp16', action='store_false',
                        help='Disable FP16')

    args = parser.parse_args()

    server = PIDSRealtimeServer(
        checkpoint_path=args.checkpoint,
        port=args.port,
        iters=args.iters,
        use_fp16=args.fp16,
    )

    server.run()


if __name__ == '__main__':
    main()
