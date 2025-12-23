"""
PIDS Inference Script
用於測試訓練好的模型

Usage:
    python inference.py --checkpoint ./checkpoints/checkpoint_best.pth \
                       --left image_left.exr \
                       --right image_right.exr \
                       --output disparity.png
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import cv2

from pids_model import build_model
from pids_dataset import EXRReader


def load_model(checkpoint_path: str, device: torch.device):
    """載入模型"""
    print(f"Loading model from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 獲取模型配置
    args = checkpoint.get('args', {})
    model_cfg = {
        'hidden_dim': args.get('hidden_dim', 128),
        'context_dim': args.get('context_dim', 128),
        'feature_dim': args.get('feature_dim', 128),
        'corr_levels': args.get('corr_levels', 4),
        'corr_radius': args.get('corr_radius', 4),
        'iters': args.get('iters', 12),
    }

    model = build_model(model_cfg)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded (Best EPE: {checkpoint.get('best_epe', 'N/A')})")

    return model


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
) -> np.ndarray:
    """
    運行推理

    Args:
        model: 模型
        left_path: 左圖像路徑
        right_path: 右圖像路徑
        device: 計算設備
        iters: 迭代次數

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
    disp = model(left_tensor, right_tensor, iters=iters, test_mode=True)

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
    parser.add_argument('--iters', type=int, default=20,
                        help='Number of iterations')
    parser.add_argument('--no_colormap', action='store_true',
                        help='Save grayscale instead of colormap')

    args = parser.parse_args()

    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 載入模型
    model = load_model(args.checkpoint, device)

    # 運行推理
    print(f"\nRunning inference...")
    print(f"  Left: {args.left}")
    print(f"  Right: {args.right}")

    disp = inference(model, args.left, args.right, device, iters=args.iters)

    # 保存結果
    save_disparity(disp, args.output, colormap=not args.no_colormap)


if __name__ == '__main__':
    main()
