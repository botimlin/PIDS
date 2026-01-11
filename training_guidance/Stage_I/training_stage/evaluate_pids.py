"""
PIDS Evaluation Script
對測試集進行完整評估，輸出詳細統計數據

Usage:
    python evaluate_pids.py \
        --checkpoint ./checkpoints_pol_v5_exp21/checkpoint_best.pth \
        --data_dir ./PIDS_dataset_pol_V5 \
        --output_dir ./eval_results

Copyright (c) 2025-2026 Po-Ting Lin
Released under the MIT License (see LICENSE file).
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm

# 添加 RAFT-Stereo core 路徑
sys.path.append('core')

from pids_dataset import PIDSSyntheticDataset
from pids_model import PIDSStereoDualStream


class MetricsCalculator:
    """計算深度估計指標"""

    @staticmethod
    def compute_epe(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> float:
        """計算 End-Point Error"""
        error = torch.abs(pred - gt)
        valid_error = error[valid > 0.5]
        if valid_error.numel() == 0:
            return 0.0
        return valid_error.mean().item()

    @staticmethod
    def compute_d_error(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor,
                        threshold: float) -> float:
        """計算 D-threshold 錯誤率 (%)"""
        error = torch.abs(pred - gt)
        valid_mask = valid > 0.5
        if valid_mask.sum() == 0:
            return 0.0
        bad_pixels = (error > threshold) & valid_mask
        return 100.0 * bad_pixels.sum().item() / valid_mask.sum().item()

    @staticmethod
    def compute_all_metrics(pred: torch.Tensor, gt: torch.Tensor,
                           valid_mask: torch.Tensor, glass_mask: torch.Tensor) -> Dict[str, float]:
        """計算所有指標"""
        # 確保維度正確
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        if gt.dim() == 4:
            gt = gt.squeeze(1)
        if valid_mask.dim() == 4:
            valid_mask = valid_mask.squeeze(1)
        if glass_mask.dim() == 4:
            glass_mask = glass_mask.squeeze(1)

        # Overall metrics
        epe = MetricsCalculator.compute_epe(pred, gt, valid_mask)
        d1 = MetricsCalculator.compute_d_error(pred, gt, valid_mask, 1.0)
        d3 = MetricsCalculator.compute_d_error(pred, gt, valid_mask, 3.0)
        d5 = MetricsCalculator.compute_d_error(pred, gt, valid_mask, 5.0)
        d10 = MetricsCalculator.compute_d_error(pred, gt, valid_mask, 10.0)

        # Glass-specific metrics
        glass_valid = valid_mask * glass_mask
        glass_epe = MetricsCalculator.compute_epe(pred, gt, glass_valid)
        glass_d1 = MetricsCalculator.compute_d_error(pred, gt, glass_valid, 1.0)
        glass_d3 = MetricsCalculator.compute_d_error(pred, gt, glass_valid, 3.0)
        glass_d5 = MetricsCalculator.compute_d_error(pred, gt, glass_valid, 5.0)
        glass_d10 = MetricsCalculator.compute_d_error(pred, gt, glass_valid, 10.0)

        # Background-specific metrics (非玻璃區域)
        bg_valid = valid_mask * (1.0 - glass_mask)
        bg_epe = MetricsCalculator.compute_epe(pred, gt, bg_valid)
        bg_d1 = MetricsCalculator.compute_d_error(pred, gt, bg_valid, 1.0)

        return {
            # Overall
            'epe': epe,
            'd1': d1,
            'd3': d3,
            'd5': d5,
            'd10': d10,
            # Glass
            'glass_epe': glass_epe,
            'glass_d1': glass_d1,
            'glass_d3': glass_d3,
            'glass_d5': glass_d5,
            'glass_d10': glass_d10,
            # Background
            'bg_epe': bg_epe,
            'bg_d1': bg_d1,
        }


class Evaluator:
    """PIDS 模型評估器"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # 載入模型
        self.model = self._load_model()

        # 載入測試數據
        self.test_loader = self._load_data()

        # 輸出目錄
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_model(self):
        """載入訓練好的模型"""
        print(f"Loading checkpoint: {self.args.checkpoint}")
        checkpoint = torch.load(self.args.checkpoint, map_location=self.device)

        # 從 checkpoint 讀取配置
        saved_args = checkpoint.get('args', {})

        # 建立模型
        model = PIDSStereoDualStream(
            hidden_dim=saved_args.get('hidden_dim', 128),
            context_dim=saved_args.get('context_dim', 128),
            feature_dim=saved_args.get('feature_dim', 128),
            corr_levels=saved_args.get('corr_levels', 4),
            corr_radius=saved_args.get('corr_radius', 4),
            pol_dim=saved_args.get('pol_dim', 64),
            pol_threshold=saved_args.get('pol_threshold', 0.05),
            pol_sharpness=saved_args.get('pol_sharpness', 20.0),
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()

        # 顯示 checkpoint 資訊
        print(f"  Step: {checkpoint.get('global_step', 'N/A')}")
        print(f"  Best Glass EPE: {checkpoint.get('best_glass_epe', 'N/A'):.4f}")
        print(f"  Best Composite: {checkpoint.get('best_composite_score', 'N/A'):.4f}")

        return model

    def _load_data(self):
        """載入測試數據"""
        print(f"Loading test data from: {self.args.data_dir}")

        # 創建測試數據集
        test_dataset = PIDSSyntheticDataset(
            data_dir=self.args.data_dir,
            split='test',
            augment=False,  # 測試不做 augmentation
        )

        print(f"  Test samples: {len(test_dataset)}")

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,  # 逐個評估
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

        return test_loader

    @torch.no_grad()
    def evaluate(self) -> Tuple[pd.DataFrame, Dict]:
        """執行評估"""
        print("\n" + "=" * 60)
        print("Starting Evaluation")
        print("=" * 60)

        all_results = []

        for idx, batch in enumerate(tqdm(self.test_loader, desc="Evaluating")):
            # 準備數據
            left = batch['left'].to(self.device)
            right = batch['right'].to(self.device)
            disp_gt = batch['disparity'].to(self.device)
            valid_mask = batch['valid_mask'].to(self.device)
            glass_mask = batch.get('glass_mask_strict', batch['glass_mask']).to(self.device)

            # 推理
            flow_preds = self.model(left, right, iters=self.args.iters, disparity_gt=disp_gt)
            disp_pred = -flow_preds[-1][:, :1]  # 取最後一個迭代的結果

            # 計算指標
            metrics = MetricsCalculator.compute_all_metrics(
                disp_pred, disp_gt, valid_mask, glass_mask
            )

            # 記錄場景名稱
            scene_name = batch.get('scene_name', [f'scene_{idx:04d}'])[0]
            metrics['scene'] = scene_name

            all_results.append(metrics)

        # 轉換為 DataFrame
        df = pd.DataFrame(all_results)

        # 計算統計量
        stats = self._compute_statistics(df)

        return df, stats

    def _compute_statistics(self, df: pd.DataFrame) -> Dict:
        """計算統計量"""
        metrics_cols = [col for col in df.columns if col != 'scene']

        stats = {}
        for col in metrics_cols:
            values = df[col].values
            stats[col] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'q25': float(np.percentile(values, 25)),
                'q75': float(np.percentile(values, 75)),
            }

        return stats

    def save_results(self, df: pd.DataFrame, stats: Dict):
        """保存結果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存每個場景的結果
        csv_path = self.output_dir / f'per_scene_results_{timestamp}.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nPer-scene results saved to: {csv_path}")

        # 保存統計結果
        stats_path = self.output_dir / f'statistics_{timestamp}.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to: {stats_path}")

        # 生成報告
        report_path = self.output_dir / f'evaluation_report_{timestamp}.txt'
        self._generate_report(df, stats, report_path)
        print(f"Report saved to: {report_path}")

    def _generate_report(self, df: pd.DataFrame, stats: Dict, path: Path):
        """生成評估報告"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("PIDS Evaluation Report\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Checkpoint: {self.args.checkpoint}\n")
            f.write(f"Test Dataset: {self.args.data_dir}\n")
            f.write(f"Number of Scenes: {len(df)}\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 主要指標統計
            f.write("-" * 70 + "\n")
            f.write("OVERALL METRICS\n")
            f.write("-" * 70 + "\n\n")

            main_metrics = ['epe', 'd1', 'd3', 'd5', 'd10']
            f.write(f"{'Metric':<12} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Median':>10}\n")
            f.write("-" * 62 + "\n")
            for m in main_metrics:
                s = stats[m]
                unit = '%' if m.startswith('d') else 'px'
                f.write(f"{m:<12} {s['mean']:>9.3f}{unit} {s['std']:>9.3f} {s['min']:>9.3f} {s['max']:>9.3f} {s['median']:>9.3f}\n")

            # 玻璃區域指標
            f.write("\n" + "-" * 70 + "\n")
            f.write("GLASS REGION METRICS\n")
            f.write("-" * 70 + "\n\n")

            glass_metrics = ['glass_epe', 'glass_d1', 'glass_d3', 'glass_d5', 'glass_d10']
            f.write(f"{'Metric':<12} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Median':>10}\n")
            f.write("-" * 62 + "\n")
            for m in glass_metrics:
                s = stats[m]
                unit = '%' if 'd' in m and m != 'glass_epe' else 'px'
                f.write(f"{m:<12} {s['mean']:>9.3f}{unit} {s['std']:>9.3f} {s['min']:>9.3f} {s['max']:>9.3f} {s['median']:>9.3f}\n")

            # 背景區域指標
            f.write("\n" + "-" * 70 + "\n")
            f.write("BACKGROUND REGION METRICS\n")
            f.write("-" * 70 + "\n\n")

            bg_metrics = ['bg_epe', 'bg_d1']
            f.write(f"{'Metric':<12} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Median':>10}\n")
            f.write("-" * 62 + "\n")
            for m in bg_metrics:
                s = stats[m]
                unit = '%' if 'd' in m else 'px'
                f.write(f"{m:<12} {s['mean']:>9.3f}{unit} {s['std']:>9.3f} {s['min']:>9.3f} {s['max']:>9.3f} {s['median']:>9.3f}\n")

            # 最佳/最差場景
            f.write("\n" + "-" * 70 + "\n")
            f.write("BEST / WORST SCENES (by Glass EPE)\n")
            f.write("-" * 70 + "\n\n")

            df_sorted = df.sort_values('glass_epe')
            f.write("Top 5 Best:\n")
            for _, row in df_sorted.head(5).iterrows():
                f.write(f"  {row['scene']}: Glass EPE = {row['glass_epe']:.3f} px, EPE = {row['epe']:.3f} px\n")

            f.write("\nTop 5 Worst:\n")
            for _, row in df_sorted.tail(5).iterrows():
                f.write(f"  {row['scene']}: Glass EPE = {row['glass_epe']:.3f} px, EPE = {row['epe']:.3f} px\n")

            f.write("\n" + "=" * 70 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 70 + "\n")

    def print_summary(self, stats: Dict):
        """打印摘要"""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)

        print("\n[Overall Metrics]")
        print(f"  EPE:  {stats['epe']['mean']:.3f} ± {stats['epe']['std']:.3f} px")
        print(f"  D1:   {stats['d1']['mean']:.2f} ± {stats['d1']['std']:.2f} %")
        print(f"  D3:   {stats['d3']['mean']:.2f} ± {stats['d3']['std']:.2f} %")

        print("\n[Glass Region Metrics]")
        print(f"  Glass EPE:  {stats['glass_epe']['mean']:.3f} ± {stats['glass_epe']['std']:.3f} px")
        print(f"  Glass D1:   {stats['glass_d1']['mean']:.2f} ± {stats['glass_d1']['std']:.2f} %")
        print(f"  Glass D3:   {stats['glass_d3']['mean']:.2f} ± {stats['glass_d3']['std']:.2f} %")

        print("\n[Background Region Metrics]")
        print(f"  BG EPE:  {stats['bg_epe']['mean']:.3f} ± {stats['bg_epe']['std']:.3f} px")
        print(f"  BG D1:   {stats['bg_d1']['mean']:.2f} ± {stats['bg_d1']['std']:.2f} %")

        print("\n[Glass EPE Distribution]")
        print(f"  Min:    {stats['glass_epe']['min']:.3f} px")
        print(f"  Q25:    {stats['glass_epe']['q25']:.3f} px")
        print(f"  Median: {stats['glass_epe']['median']:.3f} px")
        print(f"  Q75:    {stats['glass_epe']['q75']:.3f} px")
        print(f"  Max:    {stats['glass_epe']['max']:.3f} px")

        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='PIDS Evaluation')

    # 必要參數
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to test dataset')

    # 可選參數
    parser.add_argument('--output_dir', type=str, default='./eval_results',
                        help='Output directory for results')
    parser.add_argument('--iters', type=int, default=32,
                        help='Number of iterations for inference')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    args = parser.parse_args()

    # 執行評估
    evaluator = Evaluator(args)
    df, stats = evaluator.evaluate()

    # 打印摘要
    evaluator.print_summary(stats)

    # 保存結果
    evaluator.save_results(df, stats)

    print("\nEvaluation completed!")


if __name__ == '__main__':
    main()
