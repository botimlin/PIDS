"""
PIDS Inference Script
用於測試訓練好的模型

Usage:
    python inference.py --checkpoint ./checkpoints/checkpoint_best.pth \
                       --left image_left.exr \
                       --right image_right.exr \
                       --output disparity.png

Copyright (c) 2025-2026 Po-Ting Lin
Released under the MIT License (see LICENSE file).
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import cv2

from pids_model import build_model, build_model_dual_stream, PIDSStereoDualStream
from pids_dataset import EXRReader


def load_model(checkpoint_path: str, device: torch.device):
    """載入模型（支援 Dual-Stream）"""
    print(f"Loading model from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 獲取模型配置
    args = checkpoint.get('args', {})
    is_dual_stream = args.get('dual_stream', False)

    if is_dual_stream:
        # Dual-Stream 模型
        model_cfg = {
            'hidden_dim': args.get('hidden_dim', 128),
            'context_dim': args.get('context_dim', 128),
            'feature_dim': args.get('feature_dim', 128),
            'corr_levels': args.get('corr_levels', 4),
            'corr_radius': args.get('corr_radius', 4),
            'iters': args.get('iters', 12),
            'pol_dim': args.get('pol_dim', 64),
            'pol_threshold': args.get('pol_threshold', 0.05),
            'pol_sharpness': args.get('pol_sharpness', 20.0),
        }
        model = build_model_dual_stream(model_cfg)
        print(f"  Architecture: Dual-Stream (pol_dim={model_cfg['pol_dim']})")
    else:
        # 標準模型
        model_cfg = {
            'hidden_dim': args.get('hidden_dim', 128),
            'context_dim': args.get('context_dim', 128),
            'feature_dim': args.get('feature_dim', 128),
            'corr_levels': args.get('corr_levels', 4),
            'corr_radius': args.get('corr_radius', 4),
            'iters': args.get('iters', 12),
        }
        model = build_model(model_cfg)
        print(f"  Architecture: Standard RAFT-Stereo")

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    best_metric = checkpoint.get('best_glass_epe', checkpoint.get('best_epe', 'N/A'))
    print(f"  Best Glass EPE: {best_metric}")

    return model, is_dual_stream


def load_image(path: str) -> np.ndarray:
    """載入圖像"""
    path = Path(path)

    if path.suffix.lower() == '.exr':
        img = EXRReader.read_rgb(str(path))
    else:
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

    return img


def normalize_image(img: np.ndarray) -> np.ndarray:
    """正規化圖像"""
    img = np.clip(img, 0, None)
    p99 = np.percentile(img, 99)
    if p99 > 0:
        img = img / p99
    img = np.clip(img, 0, 1)
    return img


def pad_to_multiple(img: torch.Tensor, multiple: int = 32) -> tuple:
    """填充圖像到指定倍數"""
    _, _, h, w = img.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    if pad_h > 0 or pad_w > 0:
        img = F.pad(img, [0, pad_w, 0, pad_h], mode='replicate')

    return img, (h, w)


@torch.no_grad()
def inference(
    model,
    left_path: str,
    right_path: str,
    device: torch.device,
    iters: int = 20,
    is_dual_stream: bool = False,
    pol_update_iters: list = None,
    use_two_pass: bool = False,
) -> np.ndarray:
    """
    運行推理（支援兩階段偏振對齊）

    Args:
        model: 模型
        left_path: 左圖像路徑
        right_path: 右圖像路徑
        device: 計算設備
        iters: 迭代次數
        is_dual_stream: 是否為 Dual-Stream 模型
        pol_update_iters: 偏振特徵更新時機（例如 [6] 表示第6次迭代後更新）
        use_two_pass: 是否使用完整兩階段推論（更精確但較慢）

    Returns:
        視差圖 (H, W)
    """
    # 載入圖像
    left = load_image(left_path)
    right = load_image(right_path)

    # 正規化
    left = normalize_image(left)
    right = normalize_image(right)

    # 轉換為 tensor
    left_tensor = torch.from_numpy(left).permute(2, 0, 1).float().unsqueeze(0)
    right_tensor = torch.from_numpy(right).permute(2, 0, 1).float().unsqueeze(0)

    # 填充
    left_tensor, (h, w) = pad_to_multiple(left_tensor)
    right_tensor, _ = pad_to_multiple(right_tensor)

    # 移到 GPU
    left_tensor = left_tensor.to(device)
    right_tensor = right_tensor.to(device)

    # 推理
    if is_dual_stream and hasattr(model, 'forward_inference'):
        if use_two_pass:
            # 完整兩階段推論（更精確）
            print("  Using two-pass inference...")
            flow = model.forward_two_pass(
                left_tensor, right_tensor,
                iters_pass1=iters // 2,
                iters_pass2=iters // 2,
            )
        else:
            # 高效兩階段推論（推薦）
            if pol_update_iters is None:
                pol_update_iters = [iters // 2]
            print(f"  Using forward_inference with pol_update at {pol_update_iters}...")
            flow = model.forward_inference(
                left_tensor, right_tensor,
                iters=iters,
                pol_update_iters=pol_update_iters,
            )
        # flow shape: (B, 2, H, W)，取 x-component 作為 disparity
        disp = flow[:, :1, :, :]
    else:
        # 標準推論
        flow = model(left_tensor, right_tensor, iters=iters, test_mode=True)
        # 處理不同的輸出格式
        if flow.shape[1] == 2:
            disp = flow[:, :1, :, :]
        else:
            disp = flow

    # 裁切回原始大小
    disp = disp[:, :, :h, :w]

    # 轉換為 numpy
    disp = disp.squeeze().cpu().numpy()

    return disp


def save_disparity(disp: np.ndarray, output_path: str, colormap: bool = True):
    """保存視差圖"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存原始視差 (numpy)
    np.save(output_path.with_suffix('.npy'), disp)

    # 視覺化
    disp_vis = disp.copy()
    disp_vis = np.clip(disp_vis, 0, 192)  # 裁切到合理範圍

    if colormap:
        # 使用 colormap
        disp_norm = (disp_vis / 192 * 255).astype(np.uint8)
        disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_MAGMA)
        cv2.imwrite(str(output_path.with_suffix('.png')), disp_color)
    else:
        # 灰度圖
        disp_norm = (disp_vis / 192 * 255).astype(np.uint8)
        cv2.imwrite(str(output_path.with_suffix('.png')), disp_norm)

    print(f"Saved disparity to: {output_path}")
    print(f"  Range: [{disp.min():.2f}, {disp.max():.2f}]")


def main():
    parser = argparse.ArgumentParser(description='PIDS Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--left', type=str, required=True,
                        help='Path to left image (I∥)')
    parser.add_argument('--right', type=str, required=True,
                        help='Path to right image (I⊥)')
    parser.add_argument('--output', type=str, default='output_disparity.png',
                        help='Output path for disparity')
    parser.add_argument('--iters', type=int, default=24,
                        help='Number of iterations')
    parser.add_argument('--no_colormap', action='store_true',
                        help='Save grayscale instead of colormap')
    # 兩階段推論參數
    parser.add_argument('--two_pass', action='store_true',
                        help='Use full two-pass inference (more accurate, slower)')
    parser.add_argument('--pol_update_at', type=int, nargs='+', default=None,
                        help='Iterations to update polarization features (e.g., --pol_update_at 6 12)')

    args = parser.parse_args()

    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 載入模型
    model, is_dual_stream = load_model(args.checkpoint, device)

    # 運行推理
    print(f"\nRunning inference...")
    print(f"  Left: {args.left}")
    print(f"  Right: {args.right}")
    print(f"  Iterations: {args.iters}")
    if is_dual_stream:
        print(f"  Two-pass mode: {args.two_pass}")
        print(f"  Pol update at: {args.pol_update_at or [args.iters // 2]}")

    disp = inference(
        model, args.left, args.right, device,
        iters=args.iters,
        is_dual_stream=is_dual_stream,
        pol_update_iters=args.pol_update_at,
        use_two_pass=args.two_pass,
    )

    # 保存結果
    save_disparity(disp, args.output, colormap=not args.no_colormap)


if __name__ == '__main__':
    main()
