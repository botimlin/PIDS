"""
Val Set Outlier Analysis Script
分析驗證集中的 Glass EPE 分布，找出離群值

Usage:
    python analyze_val_outliers.py \
        --checkpoint ./checkpoints/exp39_v2e/checkpoint_best.pth \
        --data_dir ./data \
        --output_dir ./outlier_analysis

Copyright (c) 2025-2026 Po-Ting Lin
Released under the MIT License (see LICENSE file).
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm

# 添加 RAFT-Stereo core 路徑
sys.path.append('core')

from pids_dataset import PIDSSyntheticDataset
from pids_model import PIDSStereoPolVolumeV2E


def compute_glass_epe(pred: torch.Tensor, gt: torch.Tensor,
                      valid_mask: torch.Tensor, glass_mask: torch.Tensor) -> float:
    """計算單個樣本的 Glass EPE"""
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if gt.dim() == 4:
        gt = gt.squeeze(1)
    if valid_mask.dim() == 4:
        valid_mask = valid_mask.squeeze(1)
    if glass_mask.dim() == 4:
        glass_mask = glass_mask.squeeze(1)

    glass_valid = valid_mask * glass_mask
    error = torch.abs(pred - gt)
    valid_error = error[glass_valid > 0.5]

    if valid_error.numel() == 0:
        return float('nan')
    return valid_error.mean().item()


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """載入 V2-E 模型"""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    saved_args = checkpoint.get('args', {})

    model = PIDSStereoPolVolumeV2E(
        hidden_dim=saved_args.get('hidden_dim', 128),
        context_dim=saved_args.get('context_dim', 128),
        feature_dim=saved_args.get('feature_dim', 128),
        corr_levels=saved_args.get('corr_levels', 4),
        corr_radius=saved_args.get('corr_radius', 4),
        pol_weight_hidden=saved_args.get('pol_weight_hidden', 16),
    )

    # 載入權重
    state_dict = checkpoint['model_state_dict']
    # 處理 DataParallel 前綴
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    print(f"Model loaded successfully")
    return model


def analyze_outliers(args):
    """執行 outlier 分析"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 載入模型
    model = load_model(args.checkpoint, device)

    # 載入驗證集（使用整個資料夾，不做內部分割）
    print(f"\nLoading validation dataset from: {args.data_dir}")
    val_dataset = PIDSSyntheticDataset(
        data_dir=args.data_dir,
        split='test',  # 使用 test split 會用 100% 數據
        val_split=1.0,  # 確保用全部數據
        augment=False,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,  # 單個樣本處理以記錄 per-sample 結果
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Validation samples: {len(val_dataset)}")

    # 收集 per-sample 結果
    results = []

    print("\nRunning inference...")
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
            left = batch['left'].to(device)
            right = batch['right'].to(device)
            disp_gt = batch['disparity'].to(device)
            valid_mask = batch['valid_mask'].to(device)
            glass_mask = batch['glass_mask'].to(device)
            scene_name = batch['scene_name'][0] if 'scene_name' in batch else f"sample_{idx:04d}"

            # 取得偏振資訊
            pol_left = batch.get('pol_left')
            pol_right = batch.get('pol_right')
            if pol_left is not None:
                pol_left = pol_left.to(device)
            if pol_right is not None:
                pol_right = pol_right.to(device)

            # 推論
            result = model(left, right, iters=args.iters)

            # 處理不同返回格式
            if isinstance(result, list):
                flow_pred = result[-1]
            else:
                flow_pred = result

            # RAFT-Stereo 輸出負 disparity，取負號得到正 disparity
            pred_disp = -flow_pred[:, :1]

            # 計算 Glass EPE
            glass_epe = compute_glass_epe(pred_disp, disp_gt, valid_mask, glass_mask)

            # 計算 overall EPE
            if valid_mask.dim() == 4:
                valid_mask_squeezed = valid_mask.squeeze(1)
            else:
                valid_mask_squeezed = valid_mask
            if pred_disp.dim() == 4:
                pred_squeezed = pred_disp.squeeze(1)
            else:
                pred_squeezed = pred_disp
            if disp_gt.dim() == 4:
                gt_squeezed = disp_gt.squeeze(1)
            else:
                gt_squeezed = disp_gt

            error = torch.abs(pred_squeezed - gt_squeezed)
            valid_error = error[valid_mask_squeezed > 0.5]
            overall_epe = valid_error.mean().item() if valid_error.numel() > 0 else float('nan')

            # 計算 glass 區域大小
            glass_pixels = (glass_mask > 0.5).sum().item()
            valid_glass_pixels = ((glass_mask > 0.5) & (valid_mask > 0.5)).sum().item()

            results.append({
                'scene': scene_name,
                'glass_epe': glass_epe,
                'overall_epe': overall_epe,
                'glass_pixels': glass_pixels,
                'valid_glass_pixels': valid_glass_pixels,
            })

    # 轉換為 DataFrame
    df = pd.DataFrame(results)

    # 過濾掉 NaN (沒有玻璃區域的樣本)
    df_valid = df.dropna(subset=['glass_epe'])

    print(f"\n{'='*60}")
    print("OUTLIER ANALYSIS RESULTS")
    print(f"{'='*60}")

    # 基本統計
    glass_epes = df_valid['glass_epe'].values

    mean_epe = np.mean(glass_epes)
    median_epe = np.median(glass_epes)
    std_epe = np.std(glass_epes)
    min_epe = np.min(glass_epes)
    max_epe = np.max(glass_epes)

    q25 = np.percentile(glass_epes, 25)
    q75 = np.percentile(glass_epes, 75)
    iqr = q75 - q25

    print(f"\n[Glass EPE Statistics]")
    print(f"  Total samples:  {len(df_valid)}")
    print(f"  Mean:           {mean_epe:.3f} px")
    print(f"  Median:         {median_epe:.3f} px")
    print(f"  Std:            {std_epe:.3f} px")
    print(f"  Min:            {min_epe:.3f} px")
    print(f"  Max:            {max_epe:.3f} px")
    print(f"\n[Percentiles]")
    print(f"  Q25 (25%):      {q25:.3f} px")
    print(f"  Q50 (Median):   {median_epe:.3f} px")
    print(f"  Q75 (75%):      {q75:.3f} px")
    print(f"  IQR:            {iqr:.3f} px")

    # 計算 Mean vs Median 差距
    gap = mean_epe - median_epe
    gap_pct = (gap / median_epe * 100) if median_epe > 0 else 0
    print(f"\n[Mean vs Median Gap]")
    print(f"  Gap:            {gap:+.3f} px ({gap_pct:+.1f}%)")
    if gap > 0.5:
        print(f"  >>> Mean >> Median: 有 outlier 拉高平均!")
    else:
        print(f"  >>> Mean ≈ Median: 分布較均勻")

    # 識別 outliers (IQR method: > Q75 + 1.68*IQR)
    iqr_multiplier = 1.68
    upper_fence = q75 + iqr_multiplier * iqr
    lower_fence = q25 - iqr_multiplier * iqr

    df_valid['is_outlier'] = df_valid['glass_epe'] > upper_fence
    outliers = df_valid[df_valid['is_outlier']].sort_values('glass_epe', ascending=False)

    print(f"\n[Outlier Detection (IQR Method)]")
    print(f"  Upper fence:    {upper_fence:.3f} px (Q75 + {iqr_multiplier}*IQR)")
    print(f"  Outliers:       {len(outliers)} / {len(df_valid)} ({100*len(outliers)/len(df_valid):.1f}%)")

    if len(outliers) > 0:
        print(f"\n[Top 10 Outliers]")
        for _, row in outliers.head(10).iterrows():
            print(f"  {row['scene']}: Glass EPE = {row['glass_epe']:.3f} px")

    # 沒有 outlier 時的 EPE
    df_no_outliers = df_valid[~df_valid['is_outlier']]
    if len(df_no_outliers) > 0:
        mean_no_outliers = df_no_outliers['glass_epe'].mean()
        median_no_outliers = df_no_outliers['glass_epe'].median()
        print(f"\n[Without Outliers]")
        print(f"  Samples:        {len(df_no_outliers)}")
        print(f"  Mean:           {mean_no_outliers:.3f} px")
        print(f"  Median:         {median_no_outliers:.3f} px")
        print(f"  Improvement:    {mean_epe - mean_no_outliers:.3f} px ({100*(mean_epe - mean_no_outliers)/mean_epe:.1f}%)")

    # Best/Worst scenes
    print(f"\n[Best 5 Scenes]")
    for _, row in df_valid.nsmallest(5, 'glass_epe').iterrows():
        print(f"  {row['scene']}: Glass EPE = {row['glass_epe']:.3f} px")

    print(f"\n[Worst 5 Scenes]")
    for _, row in df_valid.nlargest(5, 'glass_epe').iterrows():
        print(f"  {row['scene']}: Glass EPE = {row['glass_epe']:.3f} px")

    # 儲存結果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = output_dir / 'per_sample_results.csv'
    df_valid.to_csv(csv_path, index=False)
    print(f"\n[Saved] Per-sample results: {csv_path}")

    # Outlier list
    if len(outliers) > 0:
        outlier_path = output_dir / 'outliers.csv'
        outliers.to_csv(outlier_path, index=False)
        print(f"[Saved] Outlier list: {outlier_path}")

    # Summary report
    summary_path = output_dir / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write("Val Set Outlier Analysis Summary\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Data dir: {args.data_dir}\n")
        f.write(f"\n{'='*50}\n\n")
        f.write(f"Total samples: {len(df_valid)}\n")
        f.write(f"Mean Glass EPE: {mean_epe:.3f} px\n")
        f.write(f"Median Glass EPE: {median_epe:.3f} px\n")
        f.write(f"Std: {std_epe:.3f} px\n")
        f.write(f"Min: {min_epe:.3f} px\n")
        f.write(f"Max: {max_epe:.3f} px\n")
        f.write(f"\nQ25: {q25:.3f} px\n")
        f.write(f"Q75: {q75:.3f} px\n")
        f.write(f"IQR: {iqr:.3f} px\n")
        f.write(f"Upper fence: {upper_fence:.3f} px\n")
        f.write(f"\nOutliers: {len(outliers)} ({100*len(outliers)/len(df_valid):.1f}%)\n")
        f.write(f"Mean-Median gap: {gap:+.3f} px ({gap_pct:+.1f}%)\n")
        if len(df_no_outliers) > 0:
            f.write(f"\nWithout outliers:\n")
            f.write(f"  Mean: {mean_no_outliers:.3f} px\n")
            f.write(f"  Median: {median_no_outliers:.3f} px\n")
    print(f"[Saved] Summary: {summary_path}")

    # 嘗試畫圖 (如果有 matplotlib)
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        ax1 = axes[0]
        ax1.hist(glass_epes, bins=50, edgecolor='black', alpha=0.7)
        ax1.axvline(mean_epe, color='red', linestyle='--', label=f'Mean: {mean_epe:.2f}')
        ax1.axvline(median_epe, color='green', linestyle='--', label=f'Median: {median_epe:.2f}')
        ax1.axvline(upper_fence, color='orange', linestyle=':', label=f'Upper fence: {upper_fence:.2f}')
        ax1.set_xlabel('Glass EPE (px)')
        ax1.set_ylabel('Count')
        ax1.set_title('Glass EPE Distribution')
        ax1.legend()

        # Boxplot
        ax2 = axes[1]
        bp = ax2.boxplot(glass_epes, vert=True)
        ax2.set_ylabel('Glass EPE (px)')
        ax2.set_title('Glass EPE Boxplot')

        # 標記 outliers
        if len(outliers) > 0:
            ax2.scatter([1] * len(outliers), outliers['glass_epe'].values,
                       c='red', marker='x', s=50, label='Outliers')
            ax2.legend()

        plt.tight_layout()

        plot_path = output_dir / 'distribution.png'
        plt.savefig(plot_path, dpi=150)
        print(f"[Saved] Distribution plot: {plot_path}")
        plt.close()

    except ImportError:
        print("[Warning] matplotlib not available, skipping plots")

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}")

    return df_valid


def main():
    parser = argparse.ArgumentParser(description='Val Set Outlier Analysis')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to V2-E checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset (with val split)')
    parser.add_argument('--output_dir', type=str, default='./outlier_analysis',
                        help='Output directory for results')
    parser.add_argument('--iters', type=int, default=32,
                        help='Number of GRU iterations')

    args = parser.parse_args()

    analyze_outliers(args)


if __name__ == '__main__':
    main()
