"""
PIDS Batch Evaluation Script
批量評估訓練好的模型，計算 EPE、D1、Glass EPE 等指標

Usage:
    python evaluate.py --checkpoint ./checkpoints/checkpoint_best.pth \
                       --data_dir ./dataset_V3 \
                       --output_dir ./evaluation_results

Copyright (c) 2025-2026 Po-Ting Lin
Released under the MIT License (see LICENSE file).
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import cv2
import json
from datetime import datetime

from pids_model import build_model
from pids_dataset import PIDSDataset, EXRReader


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

    # 顯示 checkpoint 資訊
    print(f"  Step: {checkpoint.get('step', 'N/A')}")
    print(f"  Best EPE: {checkpoint.get('best_epe', 'N/A')}")
    print(f"  Best Val Loss: {checkpoint.get('best_val_loss', 'N/A')}")

    return model, checkpoint


def pad_to_multiple(img: torch.Tensor, multiple: int = 32) -> tuple:
    """填充圖像到指定倍數"""
    _, _, h, w = img.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    if pad_h > 0 or pad_w > 0:
        img = F.pad(img, [0, pad_w, 0, pad_h], mode='replicate')

    return img, (h, w)


def compute_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray = None) -> dict:
    """
    計算評估指標

    Args:
        pred: 預測視差 (H, W)
        gt: Ground truth 視差 (H, W)
        mask: 有效像素 mask (H, W), 1=valid, 0=invalid

    Returns:
        dict: 包含各種指標
    """
    if mask is None:
        mask = np.ones_like(gt, dtype=bool)
    else:
        mask = mask.astype(bool)

    # 確保 GT 有效
    valid = mask & (gt > 0) & np.isfinite(gt) & np.isfinite(pred)

    if valid.sum() == 0:
        return {
            'epe': float('nan'),
            'bad_1px': float('nan'),
            'bad_2px': float('nan'),
            'bad_3px': float('nan'),
            'bad_5px': float('nan'),
            'bad_10px': float('nan'),
            'bad_20px': float('nan'),
            'valid_ratio': 0.0,
        }

    # End-Point Error
    error = np.abs(pred[valid] - gt[valid])
    epe = error.mean()

    # Bad pixel ratios at different thresholds
    bad_1px = (error > 1).mean() * 100
    bad_2px = (error > 2).mean() * 100
    bad_3px = (error > 3).mean() * 100   # 標準 D1
    bad_5px = (error > 5).mean() * 100
    bad_10px = (error > 10).mean() * 100
    bad_20px = (error > 20).mean() * 100

    # 有效像素比例
    valid_ratio = valid.sum() / mask.sum() * 100

    return {
        'epe': float(epe),
        'bad_1px': float(bad_1px),
        'bad_2px': float(bad_2px),
        'bad_3px': float(bad_3px),
        'bad_5px': float(bad_5px),
        'bad_10px': float(bad_10px),
        'bad_20px': float(bad_20px),
        'valid_ratio': float(valid_ratio),
    }


def compute_glass_metrics(pred: np.ndarray, gt: np.ndarray, glass_mask: np.ndarray) -> dict:
    """
    計算玻璃區域的指標

    Args:
        pred: 預測視差 (H, W)
        gt: Ground truth 視差 (H, W)
        glass_mask: 玻璃區域 mask (H, W), 1=glass, 0=non-glass

    Returns:
        dict: 玻璃區域指標
    """
    glass_mask = glass_mask.astype(bool)

    if glass_mask.sum() == 0:
        return {
            'glass_epe': float('nan'),
            'glass_bad_1px': float('nan'),
            'glass_bad_3px': float('nan'),
            'glass_bad_5px': float('nan'),
            'glass_bad_10px': float('nan'),
            'glass_coverage': 0.0,
        }

    # 玻璃區域有效像素
    valid = glass_mask & (gt > 0) & np.isfinite(gt) & np.isfinite(pred)

    if valid.sum() == 0:
        return {
            'glass_epe': float('nan'),
            'glass_bad_1px': float('nan'),
            'glass_bad_3px': float('nan'),
            'glass_bad_5px': float('nan'),
            'glass_bad_10px': float('nan'),
            'glass_coverage': 0.0,
        }

    error = np.abs(pred[valid] - gt[valid])
    glass_epe = error.mean()
    glass_bad_1px = (error > 1).mean() * 100
    glass_bad_3px = (error > 3).mean() * 100
    glass_bad_5px = (error > 5).mean() * 100
    glass_bad_10px = (error > 10).mean() * 100
    glass_coverage = glass_mask.sum() / glass_mask.size * 100

    return {
        'glass_epe': float(glass_epe),
        'glass_bad_1px': float(glass_bad_1px),
        'glass_bad_3px': float(glass_bad_3px),
        'glass_bad_5px': float(glass_bad_5px),
        'glass_bad_10px': float(glass_bad_10px),
        'glass_coverage': float(glass_coverage),
    }


def compute_background_metrics(pred: np.ndarray, gt: np.ndarray, glass_mask: np.ndarray) -> dict:
    """
    計算背景區域 (非玻璃) 的指標
    """
    bg_mask = ~glass_mask.astype(bool)

    if bg_mask.sum() == 0:
        return {
            'bg_epe': float('nan'),
            'bg_bad_1px': float('nan'),
            'bg_bad_3px': float('nan'),
            'bg_bad_5px': float('nan'),
            'bg_bad_10px': float('nan'),
        }

    valid = bg_mask & (gt > 0) & np.isfinite(gt) & np.isfinite(pred)

    if valid.sum() == 0:
        return {
            'bg_epe': float('nan'),
            'bg_bad_1px': float('nan'),
            'bg_bad_3px': float('nan'),
            'bg_bad_5px': float('nan'),
            'bg_bad_10px': float('nan'),
        }

    error = np.abs(pred[valid] - gt[valid])

    return {
        'bg_epe': float(error.mean()),
        'bg_bad_1px': float((error > 1).mean() * 100),
        'bg_bad_3px': float((error > 3).mean() * 100),
        'bg_bad_5px': float((error > 5).mean() * 100),
        'bg_bad_10px': float((error > 10).mean() * 100),
    }


@torch.no_grad()
def evaluate_sample(
    model,
    left: torch.Tensor,
    right: torch.Tensor,
    device: torch.device,
    iters: int = 20,
) -> np.ndarray:
    """運行單個樣本的推理"""
    # 填充
    left_padded, (h, w) = pad_to_multiple(left)
    right_padded, _ = pad_to_multiple(right)

    # 移到 GPU
    left_padded = left_padded.to(device)
    right_padded = right_padded.to(device)

    # 推理
    disp = model(left_padded, right_padded, iters=iters, test_mode=True)

    # 裁切回原始大小
    disp = disp[:, :, :h, :w]

    return disp.squeeze().cpu().numpy()


def save_visualization(
    pred: np.ndarray,
    gt: np.ndarray,
    glass_mask: np.ndarray,
    output_path: Path,
    max_disp: float = 576.0,
):
    """保存視覺化結果"""
    # 正規化到 0-255
    pred_vis = np.clip(pred / max_disp * 255, 0, 255).astype(np.uint8)
    gt_vis = np.clip(gt / max_disp * 255, 0, 255).astype(np.uint8)

    # 應用 colormap
    pred_color = cv2.applyColorMap(pred_vis, cv2.COLORMAP_MAGMA)
    gt_color = cv2.applyColorMap(gt_vis, cv2.COLORMAP_MAGMA)

    # 錯誤圖
    error = np.abs(pred - gt)
    error_vis = np.clip(error / 10 * 255, 0, 255).astype(np.uint8)  # 10px = 最大紅
    error_color = cv2.applyColorMap(error_vis, cv2.COLORMAP_JET)

    # 玻璃區域標記
    glass_overlay = pred_color.copy()
    if glass_mask is not None and glass_mask.sum() > 0:
        glass_overlay[glass_mask > 0] = [0, 255, 0]  # 綠色標記玻璃

    # 組合圖像 (2x2)
    top = np.hstack([pred_color, gt_color])
    bottom = np.hstack([error_color, glass_overlay])
    combined = np.vstack([top, bottom])

    cv2.imwrite(str(output_path), combined)


def main():
    parser = argparse.ArgumentParser(description='PIDS Batch Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Output directory for results')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'all'],
                        help='Which split to evaluate')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--iters', type=int, default=20,
                        help='Number of inference iterations')
    parser.add_argument('--save_vis', action='store_true',
                        help='Save visualization for each sample')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to evaluate')

    args = parser.parse_args()

    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 創建輸出目錄
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_vis:
        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)

    # 載入模型
    model, checkpoint_info = load_model(args.checkpoint, device)

    # 載入數據集
    print(f"\nLoading dataset from: {args.data_dir}")

    # 獲取所有樣本
    dataset = PIDSDataset(
        data_dir=args.data_dir,
        split='val' if args.split == 'val' else 'train',
        val_split=args.val_split,
        augment=False,  # 評估時不做增強
    )

    if args.split == 'all':
        # 合併 train 和 val
        train_dataset = PIDSDataset(
            data_dir=args.data_dir,
            split='train',
            val_split=args.val_split,
            augment=False,
        )
        # 簡單合併 indices
        all_indices = list(range(len(train_dataset))) + [i + len(train_dataset) for i in range(len(dataset))]
        print(f"Evaluating ALL data: {len(train_dataset)} train + {len(dataset)} val = {len(all_indices)} samples")
    else:
        print(f"Evaluating {args.split} split: {len(dataset)} samples")

    # 限制樣本數
    num_samples = len(dataset)
    if args.max_samples is not None:
        num_samples = min(num_samples, args.max_samples)
        print(f"Limited to {num_samples} samples")

    # 評估
    print(f"\nRunning evaluation with {args.iters} iterations...")

    all_metrics = []

    for idx in tqdm(range(num_samples), desc="Evaluating"):
        sample = dataset[idx]

        # 準備輸入
        left = sample['left'].unsqueeze(0)  # (1, 3, H, W)
        right = sample['right'].unsqueeze(0)
        gt_disp = sample['disparity'].numpy()  # (H, W)
        valid_mask = sample['valid_mask'].numpy()
        glass_mask = sample.get('glass_mask', torch.zeros_like(sample['valid_mask'])).numpy()

        # 推理
        pred_disp = evaluate_sample(model, left, right, device, iters=args.iters)

        # 計算指標
        metrics = compute_metrics(pred_disp, gt_disp, valid_mask)
        glass_metrics = compute_glass_metrics(pred_disp, gt_disp, glass_mask)
        bg_metrics = compute_background_metrics(pred_disp, gt_disp, glass_mask)

        # 合併
        sample_metrics = {
            'index': idx,
            **metrics,
            **glass_metrics,
            **bg_metrics,
        }
        all_metrics.append(sample_metrics)

        # 保存視覺化
        if args.save_vis:
            vis_path = vis_dir / f'sample_{idx:04d}.png'
            save_visualization(pred_disp, gt_disp, glass_mask, vis_path)

    # 計算統計
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    # 過濾有效結果
    valid_metrics = [m for m in all_metrics if not np.isnan(m['epe'])]

    if len(valid_metrics) == 0:
        print("No valid samples!")
        return

    # 計算平均值
    summary = {
        'num_samples': len(valid_metrics),
        'checkpoint': args.checkpoint,
        'data_dir': args.data_dir,
        'split': args.split,
        'iters': args.iters,
        'timestamp': datetime.now().isoformat(),
    }

    metric_keys = [
        # Overall metrics
        'epe', 'bad_1px', 'bad_2px', 'bad_3px', 'bad_5px', 'bad_10px', 'bad_20px', 'valid_ratio',
        # Glass metrics
        'glass_epe', 'glass_bad_1px', 'glass_bad_3px', 'glass_bad_5px', 'glass_bad_10px', 'glass_coverage',
        # Background metrics
        'bg_epe', 'bg_bad_1px', 'bg_bad_3px', 'bg_bad_5px', 'bg_bad_10px',
    ]

    print(f"\nTotal samples: {len(valid_metrics)}")
    print("-" * 40)

    for key in metric_keys:
        values = [m[key] for m in valid_metrics if not np.isnan(m.get(key, float('nan')))]
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            summary[f'{key}_mean'] = float(mean_val)
            summary[f'{key}_std'] = float(std_val)

            # 格式化輸出
            if 'ratio' in key or 'd1' in key or 'd3' in key:
                print(f"{key:15s}: {mean_val:7.2f}% ± {std_val:.2f}%")
            else:
                print(f"{key:15s}: {mean_val:7.3f} ± {std_val:.3f}")

    print("-" * 40)

    # 關鍵指標總結
    print("\n" + "=" * 60)
    print("📊 KEY METRICS SUMMARY")
    print("=" * 60)

    print("\n[Overall]")
    print(f"  EPE:      {summary.get('epe_mean', float('nan')):.3f} px")
    print(f"  Bad@1px:  {summary.get('bad_1px_mean', float('nan')):.2f}%")
    print(f"  Bad@3px:  {summary.get('bad_3px_mean', float('nan')):.2f}%  (D1 standard)")
    print(f"  Bad@5px:  {summary.get('bad_5px_mean', float('nan')):.2f}%")
    print(f"  Bad@10px: {summary.get('bad_10px_mean', float('nan')):.2f}%")

    print("\n[Glass Region] ⭐")
    print(f"  EPE:      {summary.get('glass_epe_mean', float('nan')):.3f} px")
    print(f"  Bad@1px:  {summary.get('glass_bad_1px_mean', float('nan')):.2f}%")
    print(f"  Bad@3px:  {summary.get('glass_bad_3px_mean', float('nan')):.2f}%")
    print(f"  Bad@5px:  {summary.get('glass_bad_5px_mean', float('nan')):.2f}%")
    print(f"  Bad@10px: {summary.get('glass_bad_10px_mean', float('nan')):.2f}%")
    print(f"  Coverage: {summary.get('glass_coverage_mean', float('nan')):.2f}%")

    print("\n[Background]")
    print(f"  EPE:      {summary.get('bg_epe_mean', float('nan')):.3f} px")
    print(f"  Bad@3px:  {summary.get('bg_bad_3px_mean', float('nan')):.2f}%")
    print(f"  Bad@5px:  {summary.get('bg_bad_5px_mean', float('nan')):.2f}%")

    # 保存結果
    results_file = output_dir / 'evaluation_summary.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Summary saved to: {results_file}")

    # 保存詳細結果
    details_file = output_dir / 'evaluation_details.json'
    with open(details_file, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"✅ Details saved to: {details_file}")

    # 保存為 CSV (方便 Excel 分析)
    csv_file = output_dir / 'evaluation_details.csv'
    with open(csv_file, 'w', encoding='utf-8') as f:
        # Header
        f.write(','.join(['index'] + metric_keys) + '\n')
        # Data
        for m in all_metrics:
            values = [str(m['index'])] + [f"{m.get(k, 'nan'):.4f}" for k in metric_keys]
            f.write(','.join(values) + '\n')
    print(f"✅ CSV saved to: {csv_file}")


if __name__ == '__main__':
    main()
