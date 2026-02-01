"""
PIDS Evaluation Script
對測試集進行完整評估，輸出詳細統計數據

支援三種架構:
  - Baseline: 標準 RAFT-Stereo (--baseline)
  - Dual-Stream: 偏振編碼器架構 (--dual_stream)
  - Pol Volume: 偏振體積架構，無 Oracle/Real Gap (--pol_volume)

支援兩種推論模式 (僅 Dual-Stream):
  - Real Mode (預設): 使用 forward_inference，不依賴 GT disparity
  - Oracle Mode: 使用 GT disparity 對齊偏振特徵 (理論上限)

Usage:
    # Baseline 評估
    python evaluate_pids.py \
        --checkpoint ./checkpoints_baseline/checkpoint_best.pth \
        --data_dir ./PIDS_dataset_pol_V6 \
        --baseline

    # Dual-Stream Real Mode (實戰能力)
    python evaluate_pids.py \
        --checkpoint ./checkpoints_dual_stream/checkpoint_best.pth \
        --data_dir ./PIDS_dataset_pol_V6 \
        --dual_stream

    # Dual-Stream Oracle Mode (理論上限)
    python evaluate_pids.py \
        --checkpoint ./checkpoints_dual_stream/checkpoint_best.pth \
        --data_dir ./PIDS_dataset_pol_V6 \
        --dual_stream --oracle

    # Pol Volume 評估 (Oracle = Real，無 Gap)
    python evaluate_pids.py \
        --checkpoint ./checkpoints_pol_volume/checkpoint_best.pth \
        --data_dir ./PIDS_dataset_pol_V6 \
        --pol_volume

Copyright (c) 2025-2026 Po-Ting Lin
Released under the MIT License (see LICENSE file).
"""

import os
import sys
import argparse
import json
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
from pids_model import (
    PIDSStereoDualStream,
    PIDSStereoPolVolume,
    PIDSStereoPolVolumeV2,  # V2: Polarization Attention + Gated Fusion
    PIDSStereoPolVolumeV2A,  # V2-A: Pol-Conditioned Corr Residual
    PIDSStereoPolVolumeV2B,  # V2-B: Scheduled Residual
    PIDSStereoPolVolumeV2C,  # V2-C: Gradient Gating
    PIDSStereoPolVolumeV2D,  # V2-D: Disparity-Aware Pol Modulation
    PIDSStereoPolVolumeV2E,  # V2-E: Pre-Corr Pol Weighting (最後嘗試)
    PIDSStereoPolVolumeV3,   # V3: Pol-in-Feature (已證明失敗)
    PIDSStereoPolVolumeV3B,  # V3-B: Additive Pol Fusion (已證明失敗)
    PIDSStereoLearnablePol,
)

# 嘗試載入官方 RAFT-Stereo (baseline)
try:
    from raft_stereo import RAFTStereo
    RAFT_AVAILABLE = True
except ImportError:
    RAFT_AVAILABLE = False


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

        # 決定架構類型
        if self.args.baseline:
            arch_name = "Baseline (RAFT-Stereo)"
            if not RAFT_AVAILABLE:
                raise ImportError("RAFTStereo not available. Check RAFT-Stereo installation.")
            # RAFTStereo 使用 args 對象而非單獨參數
            class RAFTArgs:
                def __init__(self):
                    self.hidden_dims = [128, 128, 128]
                    self.corr_implementation = "reg"
                    self.shared_backbone = False
                    self.corr_levels = saved_args.get('corr_levels', 4)
                    self.corr_radius = saved_args.get('corr_radius', 4)
                    self.n_downsample = 2
                    self.context_norm = "batch"
                    self.slow_fast_gru = False
                    self.n_gru_layers = 3
                    self.mixed_precision = False
            model = RAFTStereo(RAFTArgs())

        elif self.args.learnable_pol:
            arch_name = "Learnable Polarization Volume"
            model = PIDSStereoLearnablePol(
                hidden_dim=saved_args.get('hidden_dim', 128),
                context_dim=saved_args.get('context_dim', 128),
                feature_dim=saved_args.get('feature_dim', 128),
                corr_levels=saved_args.get('corr_levels', 4),
                corr_radius=saved_args.get('corr_radius', 4),
                pol_dim=saved_args.get('pol_dim', 32),
                pol_levels=saved_args.get('pol_levels', self.args.pol_levels),
                pol_radius=saved_args.get('pol_radius', self.args.pol_radius),
            )

        elif self.args.pol_volume:
            arch_name = "Polarization Volume"
            model = PIDSStereoPolVolume(
                hidden_dim=saved_args.get('hidden_dim', 128),
                context_dim=saved_args.get('context_dim', 128),
                feature_dim=saved_args.get('feature_dim', 128),
                corr_levels=saved_args.get('corr_levels', 4),
                corr_radius=saved_args.get('corr_radius', 4),
                pol_levels=saved_args.get('pol_levels', self.args.pol_levels),
                pol_radius=saved_args.get('pol_radius', self.args.pol_radius),
            )

        elif self.args.pol_volume_v2:
            arch_name = "Polarization Volume V2 (Attention + Gated Fusion)"
            model = PIDSStereoPolVolumeV2(
                hidden_dim=saved_args.get('hidden_dim', 128),
                context_dim=saved_args.get('context_dim', 128),
                feature_dim=saved_args.get('feature_dim', 128),
                corr_levels=saved_args.get('corr_levels', 4),
                corr_radius=saved_args.get('corr_radius', 4),
                pol_levels=saved_args.get('pol_levels', self.args.pol_levels),
                pol_radius=saved_args.get('pol_radius', self.args.pol_radius),
                fused_dim=saved_args.get('fused_dim', 128),
            )

        elif self.args.pol_volume_v2a:
            arch_name = "Polarization Volume V2-A (Corr Residual)"
            model = PIDSStereoPolVolumeV2A(
                hidden_dim=saved_args.get('hidden_dim', 128),
                context_dim=saved_args.get('context_dim', 128),
                feature_dim=saved_args.get('feature_dim', 128),
                corr_levels=saved_args.get('corr_levels', 4),
                corr_radius=saved_args.get('corr_radius', 4),
                pol_levels=saved_args.get('pol_levels', self.args.pol_levels),
                pol_radius=saved_args.get('pol_radius', self.args.pol_radius),
                residual_hidden_dim=saved_args.get('residual_hidden_dim', 64),
                residual_init_scale=saved_args.get('residual_init_scale', 0.1),
            )

        elif self.args.pol_volume_v2b:
            arch_name = "Polarization Volume V2-B (Scheduled Residual)"
            model = PIDSStereoPolVolumeV2B(
                hidden_dim=saved_args.get('hidden_dim', 128),
                context_dim=saved_args.get('context_dim', 128),
                feature_dim=saved_args.get('feature_dim', 128),
                corr_levels=saved_args.get('corr_levels', 4),
                corr_radius=saved_args.get('corr_radius', 4),
                pol_levels=saved_args.get('pol_levels', self.args.pol_levels),
                pol_radius=saved_args.get('pol_radius', self.args.pol_radius),
                residual_hidden_dim=saved_args.get('residual_hidden_dim', 64),
                residual_init_scale=saved_args.get('residual_init_scale', 0.1),
            )

        elif self.args.pol_volume_v2c:
            arch_name = "Polarization Volume V2-C (Gradient Gating)"
            model = PIDSStereoPolVolumeV2C(
                hidden_dim=saved_args.get('hidden_dim', 128),
                context_dim=saved_args.get('context_dim', 128),
                feature_dim=saved_args.get('feature_dim', 128),
                corr_levels=saved_args.get('corr_levels', 4),
                corr_radius=saved_args.get('corr_radius', 4),
                pol_levels=saved_args.get('pol_levels', self.args.pol_levels),
                pol_radius=saved_args.get('pol_radius', self.args.pol_radius),
                residual_hidden_dim=saved_args.get('residual_hidden_dim', 64),
                residual_init_scale=saved_args.get('residual_init_scale', 0.1),
                gating_hidden_dim=saved_args.get('gating_hidden_dim', 32),
            )

        elif self.args.pol_volume_v2d:
            arch_name = "Polarization Volume V2-D (Disparity-Aware Pol Modulation)"
            model = PIDSStereoPolVolumeV2D(
                hidden_dim=saved_args.get('hidden_dim', 128),
                context_dim=saved_args.get('context_dim', 128),
                feature_dim=saved_args.get('feature_dim', 128),
                corr_levels=saved_args.get('corr_levels', 4),
                corr_radius=saved_args.get('corr_radius', 4),
                pol_gate_hidden=saved_args.get('pol_gate_hidden', 8),
                pol_alpha=saved_args.get('pol_alpha', 0.2),
            )

        elif self.args.pol_volume_v2e:
            arch_name = "Polarization Volume V2-E (Pre-Corr Pol Weighting)"
            model = PIDSStereoPolVolumeV2E(
                hidden_dim=saved_args.get('hidden_dim', 128),
                context_dim=saved_args.get('context_dim', 128),
                feature_dim=saved_args.get('feature_dim', 128),
                corr_levels=saved_args.get('corr_levels', 4),
                corr_radius=saved_args.get('corr_radius', 4),
                pol_weight_hidden=saved_args.get('pol_weight_hidden', 8),
            )

        elif self.args.pol_volume_v3:
            arch_name = "Polarization Volume V3 (Pol-in-Feature)"
            model = PIDSStereoPolVolumeV3(
                hidden_dim=saved_args.get('hidden_dim', 128),
                context_dim=saved_args.get('context_dim', 128),
                feature_dim=saved_args.get('feature_dim', 128),
                corr_levels=saved_args.get('corr_levels', 4),
                corr_radius=saved_args.get('corr_radius', 4),
            )

        elif self.args.pol_volume_v3b:
            arch_name = "Polarization Volume V3-B (Additive Pol Fusion)"
            model = PIDSStereoPolVolumeV3B(
                hidden_dim=saved_args.get('hidden_dim', 128),
                context_dim=saved_args.get('context_dim', 128),
                feature_dim=saved_args.get('feature_dim', 128),
                corr_levels=saved_args.get('corr_levels', 4),
                corr_radius=saved_args.get('corr_radius', 4),
                pol_scale=saved_args.get('pol_scale', 0.1),
            )

        else:  # dual_stream (default)
            arch_name = "Dual-Stream"
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

        print(f"  Architecture: {arch_name}")
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()

        # 顯示 checkpoint 資訊
        step = checkpoint.get('global_step', 'N/A')
        best_epe = checkpoint.get('best_glass_epe', checkpoint.get('best_epe', 'N/A'))
        best_composite = checkpoint.get('best_composite_score', 'N/A')
        print(f"  Step: {step}")
        if isinstance(best_epe, (int, float)):
            print(f"  Best Glass EPE: {best_epe:.4f}")
        if isinstance(best_composite, (int, float)):
            print(f"  Best Composite: {best_composite:.4f}")

        return model

    def _load_data(self):
        """載入測試數據"""
        print(f"Loading test data from: {self.args.data_dir}")

        # 創建測試數據集
        # 使用 val_split=1.0 來使用目錄內全部數據
        # （假設用戶已經手動分好 train/test 目錄）
        test_dataset = PIDSSyntheticDataset(
            data_dir=self.args.data_dir,
            split='test',
            val_split=1.0,  # 100% 用於測試 = 全部數據
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

        # 決定架構和推論模式
        if self.args.baseline:
            print("Architecture: Baseline (RAFT-Stereo)")
            print("  -> Standard stereo matching without polarization")
            self.arch_mode = 'baseline'

        elif self.args.learnable_pol:
            print("Architecture: Learnable Polarization Volume")
            print("  -> Learnable PolHead extracts polarization features")
            print("  -> No Oracle/Real gap (prediction independent of GT)")
            self.arch_mode = 'learnable_pol'

        elif self.args.pol_volume:
            print("Architecture: Polarization Volume")
            print("  -> Pre-computed pol_diff at all disparities")
            print("  -> No Oracle/Real gap (they are identical)")
            self.arch_mode = 'pol_volume'

        elif self.args.pol_volume_v2 or self.args.pol_volume_v2a or self.args.pol_volume_v2b or \
             self.args.pol_volume_v2c or self.args.pol_volume_v2d or self.args.pol_volume_v2e:
            # V2 系列都是 Pol Volume 變體，沒有 Oracle/Real gap
            variant = "V2" if self.args.pol_volume_v2 else \
                      "V2-A" if self.args.pol_volume_v2a else \
                      "V2-B" if self.args.pol_volume_v2b else \
                      "V2-C" if self.args.pol_volume_v2c else \
                      "V2-D" if self.args.pol_volume_v2d else "V2-E"
            print(f"Architecture: Polarization Volume {variant}")
            print("  -> Pre-computed pol_diff at all disparities")
            print("  -> No Oracle/Real gap (they are identical)")
            self.arch_mode = 'pol_volume'  # 使用相同的推論邏輯

        else:  # dual_stream
            print("Architecture: Dual-Stream")
            if self.args.oracle:
                print("Inference Mode: ORACLE (using GT disparity for polarization alignment)")
                print("  -> This shows theoretical upper bound, NOT real deployment performance")
                self.arch_mode = 'dual_stream_oracle'
            elif self.args.two_pass:
                print("Inference Mode: REAL (two-pass, no GT)")
                print("  -> Pass 1: Get rough disparity without pol alignment")
                print("  -> Pass 2: Use rough disparity for pol alignment, refine")
                self.arch_mode = 'dual_stream_two_pass'
            else:
                print("Inference Mode: REAL (single-pass forward_inference, no GT)")
                print("  -> This shows actual deployment performance")
                if self.args.pol_update_iters is None:
                    self.pol_update_iters = [self.args.iters // 2]
                else:
                    self.pol_update_iters = self.args.pol_update_iters
                print(f"  -> Polarization update at iterations: {self.pol_update_iters}")
                self.arch_mode = 'dual_stream_real'

        print("=" * 60 + "\n")

        # AMP 設定 (Stability Test 用)
        _use_amp = (os.environ.get("STABILITY_TEST_AMP") == "1")
        if _use_amp:
            print("[Stability Test] Using AMP (torch.amp.autocast) for inference")

        all_results = []

        for idx, batch in enumerate(tqdm(self.test_loader, desc="Evaluating")):
            # 準備數據
            left = batch['left'].to(self.device)
            right = batch['right'].to(self.device)
            disp_gt = batch['disparity'].to(self.device)
            valid_mask = batch['valid_mask'].to(self.device)
            glass_mask = batch.get('glass_mask_strict', batch['glass_mask']).to(self.device)

            # 推理 (根據架構和模式選擇)
            # AMP context manager for Stability Test
            from contextlib import nullcontext
            _amp_ctx = torch.amp.autocast('cuda') if _use_amp else nullcontext()

            with _amp_ctx:
                if self.arch_mode == 'baseline':
                    # ========== Baseline (RAFT-Stereo) ==========
                    flow_preds = self.model(left, right, iters=self.args.iters)
                    disp_pred = -flow_preds[-1][:, :1]

                elif self.arch_mode == 'learnable_pol':
                    # ========== Learnable Polarization Volume ==========
                    # 學習偏振特徵，不需要 GT disparity
                    flow_preds = self.model(left, right, iters=self.args.iters)
                    disp_pred = -flow_preds[-1][:, :1]

                elif self.arch_mode == 'pol_volume':
                    # ========== Polarization Volume (包括 V2 系列) ==========
                    # 不需要 GT disparity，預先計算所有 disparity 的 pol_diff
                    result = self.model(
                        left, right,
                        iters=self.args.iters,
                    )
                    # V2-E 可能返回 list 或單一 tensor
                    if isinstance(result, list):
                        flow_pred = result[-1]
                    else:
                        flow_pred = result
                    # RAFT-Stereo 輸出負 disparity，需要取負號
                    disp_pred = -flow_pred[:, :1]

                elif self.arch_mode == 'dual_stream_oracle':
                    # ========== Dual-Stream: Oracle Mode ==========
                    flow_preds = self.model(left, right, iters=self.args.iters, disparity_gt=disp_gt)
                    disp_pred = -flow_preds[-1][:, :1]

                elif self.arch_mode == 'dual_stream_two_pass':
                    # ========== Dual-Stream: Two-Pass ==========
                    if hasattr(self.model, 'forward_two_pass'):
                        flow_pred = self.model.forward_two_pass(
                            left, right,
                            iters_pass1=self.args.iters // 2,
                            iters_pass2=self.args.iters // 2
                        )
                        disp_pred = -flow_pred[:, :1]
                    else:
                        flow_preds = self.model(left, right, iters=self.args.iters)
                        disp_pred = -flow_preds[-1][:, :1]

                else:  # dual_stream_real
                    # ========== Dual-Stream: Real Single-Pass ==========
                    if hasattr(self.model, 'forward_inference'):
                        flow_pred = self.model.forward_inference(
                            left, right,
                            iters=self.args.iters,
                            pol_update_iters=self.pol_update_iters
                        )
                        disp_pred = -flow_pred[:, :1]
                    else:
                        flow_preds = self.model(left, right, iters=self.args.iters)
                        disp_pred = -flow_preds[-1][:, :1]

            # 確保 FP32 用於指標計算
            disp_pred = disp_pred.float()

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
        """計算統計量（含 Trimmed Mean, P90, P95, Outlier Rate）"""
        metrics_cols = [col for col in df.columns if col != 'scene']

        stats = {}
        for col in metrics_cols:
            values = df[col].values
            sorted_values = np.sort(values)
            n = len(values)

            # Trimmed Mean: 移除 top 10%
            trim_idx = int(n * 0.9)
            trimmed_values = sorted_values[:trim_idx] if trim_idx > 0 else sorted_values

            # Outlier rate: > 20px (僅對 EPE 類指標有意義)
            outlier_threshold = 20.0
            outlier_count = np.sum(values > outlier_threshold)
            outlier_rate = outlier_count / n if n > 0 else 0.0

            stats[col] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'q25': float(np.percentile(values, 25)),
                'q75': float(np.percentile(values, 75)),
                'p90': float(np.percentile(values, 90)),
                'p95': float(np.percentile(values, 95)),
                'trimmed_mean': float(np.mean(trimmed_values)),  # top 10% removed
                'outlier_rate': float(outlier_rate),  # % > 20px
                'outlier_count': int(outlier_count),
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
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            # 架構和推論模式
            if self.args.baseline:
                f.write(f"Architecture: Baseline (RAFT-Stereo)\n")
            elif self.args.pol_volume:
                f.write(f"Architecture: Polarization Volume\n")
                f.write(f"  (No Oracle/Real gap - prediction independent of GT)\n")
            else:
                f.write(f"Architecture: Dual-Stream\n")
                if self.args.oracle:
                    f.write(f"Inference Mode: ORACLE (uses GT disparity for pol alignment)\n")
                elif self.args.two_pass:
                    f.write(f"Inference Mode: REAL (two-pass)\n")
                else:
                    f.write(f"Inference Mode: REAL (single-pass)\n")
                    if hasattr(self, 'pol_update_iters'):
                        f.write(f"  pol_update_iters: {self.pol_update_iters}\n")
            f.write("\n")

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

            # Glass EPE Robust Statistics (for paper & checkpoint selection)
            f.write("\n" + "-" * 70 + "\n")
            f.write("GLASS EPE - ROBUST STATISTICS (for paper & checkpoint selection)\n")
            f.write("-" * 70 + "\n\n")
            ge = stats['glass_epe']
            f.write(f"  Mean:          {ge['mean']:.3f} px\n")
            f.write(f"  Median:        {ge['median']:.3f} px\n")
            f.write(f"  Trimmed Mean:  {ge['trimmed_mean']:.3f} px  (top 10% removed)\n")
            f.write(f"  P90:           {ge['p90']:.3f} px\n")
            f.write(f"  P95:           {ge['p95']:.3f} px\n")
            f.write(f"  Outlier Rate:  {ge['outlier_rate']*100:.1f}%  ({ge['outlier_count']} scenes > 20px)\n")

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

        # 顯示架構和推論模式
        if self.args.baseline:
            print(f"\nArchitecture: Baseline (RAFT-Stereo)")
        elif self.args.learnable_pol:
            print(f"\nArchitecture: Learnable Polarization Volume")
            print(f"  (Learnable PolHead + Glass-aware Loss)")
        elif self.args.pol_volume or self.args.pol_volume_v2 or self.args.pol_volume_v2a or \
             self.args.pol_volume_v2b or self.args.pol_volume_v2c or self.args.pol_volume_v2d or \
             self.args.pol_volume_v2e:
            variant = "V1" if self.args.pol_volume else \
                      "V2" if self.args.pol_volume_v2 else \
                      "V2-A" if self.args.pol_volume_v2a else \
                      "V2-B" if self.args.pol_volume_v2b else \
                      "V2-C" if self.args.pol_volume_v2c else \
                      "V2-D" if self.args.pol_volume_v2d else "V2-E"
            print(f"\nArchitecture: Polarization Volume {variant}")
            print(f"  (No Oracle/Real gap - prediction independent of GT)")
        elif self.args.dual_stream:
            if self.args.oracle:
                mode = "ORACLE (uses GT disparity)"
            elif self.args.two_pass:
                mode = "REAL (two-pass)"
            else:
                mode = "REAL (single-pass)"
            print(f"\nArchitecture: Dual-Stream")
            print(f"Inference Mode: {mode}")
            if hasattr(self, 'pol_update_iters') and not self.args.oracle and not self.args.two_pass:
                print(f"  (pol_update_iters: {self.pol_update_iters})")

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

        print("\n[Glass EPE - Robust Statistics]")
        print(f"  Mean:         {stats['glass_epe']['mean']:.3f} px")
        print(f"  Median:       {stats['glass_epe']['median']:.3f} px")
        print(f"  Trimmed Mean: {stats['glass_epe']['trimmed_mean']:.3f} px  (top 10% removed)")
        print(f"  P90:          {stats['glass_epe']['p90']:.3f} px")
        print(f"  P95:          {stats['glass_epe']['p95']:.3f} px")
        print(f"  Outlier Rate: {stats['glass_epe']['outlier_rate']*100:.1f}%  ({stats['glass_epe']['outlier_count']} scenes > 20px)")

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

    # 架構選擇 (互斥)
    arch_group = parser.add_mutually_exclusive_group()
    arch_group.add_argument('--baseline', action='store_true',
                            help='Use Baseline RAFT-Stereo architecture (no polarization)')
    arch_group.add_argument('--dual_stream', action='store_true',
                            help='Use Dual-Stream architecture (with pol encoder)')
    arch_group.add_argument('--pol_volume', action='store_true',
                            help='Use Polarization Volume architecture (no Oracle/Real gap)')
    arch_group.add_argument('--pol_volume_v2', action='store_true',
                            help='Use Polarization Volume V2 (Attention + Gated Fusion)')
    arch_group.add_argument('--pol_volume_v2a', action='store_true',
                            help='Use Polarization Volume V2-A (Corr Residual)')
    arch_group.add_argument('--pol_volume_v2b', action='store_true',
                            help='Use Polarization Volume V2-B (Scheduled Residual)')
    arch_group.add_argument('--pol_volume_v2c', action='store_true',
                            help='Use Polarization Volume V2-C (Gradient Gating)')
    arch_group.add_argument('--pol_volume_v2d', action='store_true',
                            help='Use Polarization Volume V2-D (Disparity-Aware Pol Modulation)')
    arch_group.add_argument('--pol_volume_v2e', action='store_true',
                            help='Use Polarization Volume V2-E (Pre-Corr Pol Weighting, final attempt)')
    arch_group.add_argument('--pol_volume_v3', action='store_true',
                            help='Use Polarization Volume V3 (Pol-in-Feature, DEPRECATED)')
    arch_group.add_argument('--pol_volume_v3b', action='store_true',
                            help='Use Polarization Volume V3-B (Additive Pol Fusion, DEPRECATED)')
    arch_group.add_argument('--learnable_pol', action='store_true',
                            help='Use Learnable Polarization Volume architecture (with PolHead)')

    # 推論模式 (僅 Dual-Stream 適用)
    parser.add_argument('--oracle', action='store_true',
                        help='[Dual-Stream only] Use Oracle mode (with GT disparity for pol alignment). '
                             'Default is Real mode (no GT).')
    parser.add_argument('--two_pass', action='store_true',
                        help='[Dual-Stream only] Use two-pass inference in Real mode. '
                             'Pass 1: get rough disparity, Pass 2: use it for pol alignment.')
    parser.add_argument('--pol_update_iters', type=int, nargs='+', default=None,
                        help='[Dual-Stream only] Iterations at which to update polarization features. '
                             'Default: [iters//2] (update once at middle).')

    # Pol Volume 專用參數
    parser.add_argument('--pol_levels', type=int, default=4,
                        help='[Pol Volume only] Number of polarization volume pyramid levels')
    parser.add_argument('--pol_radius', type=int, default=4,
                        help='[Pol Volume only] Lookup radius for polarization volume')

    args = parser.parse_args()

    # === 跨平台一致性: 預設關閉 TF32 (Exp #42 結論) ===
    # TF32 在 Hopper/Blackwell 架構上會導致 RAFT-Stereo 結果偏差 ~50%
    # FP32 (TF32=OFF) 在三代架構 (Ada/Hopper/Blackwell) 上完全一致
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # === Stability Test: 精度設定 (透過環境變數覆蓋，僅用於測試) ===
    import os as _os
    _st_tf32 = _os.environ.get("STABILITY_TEST_TF32")
    _st_bench = _os.environ.get("STABILITY_TEST_BENCHMARK")
    _st_amp = _os.environ.get("STABILITY_TEST_AMP")

    if _st_tf32 is not None:
        _tf32_on = (_st_tf32 == "1")
        torch.backends.cuda.matmul.allow_tf32 = _tf32_on
        torch.backends.cudnn.allow_tf32 = _tf32_on
        print(f"[Stability Test] TF32 = {_tf32_on}")

    if _st_bench is not None:
        _bench_on = (_st_bench == "1")
        torch.backends.cudnn.benchmark = _bench_on
        torch.backends.cudnn.deterministic = not _bench_on
        print(f"[Stability Test] cuDNN benchmark = {_bench_on}, deterministic = {not _bench_on}")

    if _st_amp is not None:
        _amp_on = (_st_amp == "1")
        if _amp_on and hasattr(args, 'mixed_precision'):
            args.mixed_precision = True
        print(f"[Stability Test] AMP = {_amp_on}")

    # 預設架構: Dual-Stream
    if not args.baseline and not args.dual_stream and not args.pol_volume and not args.pol_volume_v2 and not args.pol_volume_v2a and not args.pol_volume_v2b and not args.pol_volume_v2c and not args.pol_volume_v2d and not args.pol_volume_v2e and not args.pol_volume_v3 and not args.pol_volume_v3b and not args.learnable_pol:
        args.dual_stream = True
        print("No architecture specified, defaulting to --dual_stream")

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
