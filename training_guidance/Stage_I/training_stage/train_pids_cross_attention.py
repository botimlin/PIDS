"""
PIDS Cross-Attention Training Script (Fork from train_pids.py)
===============================================================

Fork 版本，不修改原始 train_pids.py。
新增 Cross-Attention Fusion 訓練支援。

基於 Exp #16 的訓練流程，加入:
1. Cross-Attention 模型支援
2. Alpha Cap Warmup 機制
3. Alpha 值監控與 logging

使用方式:
    python train_pids_cross_attention.py \
        --data_dir ./dataset_pol \
        --output_dir ./checkpoints_cross_attention \
        --pol_dim 128 \
        --alpha_cap_start 0.05 \
        --alpha_cap_warmup 5000

Copyright (c) 2025-2026 Po-Ting Lin
Released under the MIT License (see LICENSE file).
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# 從原始訓練腳本導入基礎組件
from train_pids import (
    Trainer,
    AverageMeter,
)
from pids_dataset import create_data_loaders
from pids_model import PIDSStereoLoss

# 導入 Cross-Attention 模型
from pids_model_cross_attention import (
    PIDSStereoDualStreamCrossAttention,
    build_model_cross_attention,
)


class CrossAttentionTrainer(Trainer):
    """
    Cross-Attention 訓練器

    繼承自 PIDSTrainer，覆寫模型建立和訓練邏輯。
    """

    def __init__(self, args):
        # 強制啟用 dual_stream (Cross-Attention 需要)
        args.dual_stream = True
        super().__init__(args)

    def _build_model(self) -> nn.Module:
        """建立 Cross-Attention 模型"""

        print("Building Cross-Attention Dual-Stream model...")
        model = PIDSStereoDualStreamCrossAttention(
            pol_dim=self.args.pol_dim,
            pol_threshold=self.args.pol_threshold,
            pol_sharpness=self.args.pol_sharpness,
            hidden_dim=self.args.hidden_dim,
            context_dim=self.args.context_dim,
            feature_dim=self.args.feature_dim,
            corr_levels=self.args.corr_levels,
            corr_radius=self.args.corr_radius,
            iters=self.args.iters,
            mixed_precision=self.args.mixed_precision or self.args.bf16,
            # Cross-Attention 專用參數
            num_heads=self.args.num_heads,
            pool_size=self.args.pool_size,
            enable_stereo_to_pol=self.args.enable_stereo_to_pol,
        )

        # 載入預訓練權重 (部分匹配)
        if self.args.pretrained:
            self._load_pretrained_weights(model, self.args.pretrained)
            print("  Cross-Attention Fusion initialized randomly (new layers)")

        # 凍結 Feature Encoder
        if self.args.freeze_fnet:
            frozen_count = 0
            for name, param in model.named_parameters():
                if 'fnet' in name:
                    param.requires_grad = False
                    frozen_count += 1
            print(f"Frozen {frozen_count} parameters in feature encoder (fnet)")

        # 多 GPU 支援
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)

        # 移動模型到 GPU
        model = model.to(self.device)

        # 參數統計
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        return model

    def _compute_alpha_cap(self) -> float:
        """
        計算當前 step 的 alpha cap

        Warmup 策略:
        - step < warmup_steps: alpha_cap 從 start 線性增長到 1.0
        - step >= warmup_steps: alpha_cap = 1.0

        Ablation 模式:
        - nopol_ablation=True: alpha_cap 永遠為 0，pol 分支不貢獻
        """
        # Ablation: 強制 alpha=0，等效無偏振
        if getattr(self.args, 'nopol_ablation', False):
            return 0.0

        if self.global_step >= self.args.alpha_cap_warmup:
            return 1.0

        progress = self.global_step / self.args.alpha_cap_warmup
        return self.args.alpha_cap_start + progress * (1.0 - self.args.alpha_cap_start)

    def _train_epoch(self):
        """訓練一個 epoch (覆寫以加入 alpha cap)"""
        self.model.train()

        meters = {
            'loss': AverageMeter(),
            'epe': AverageMeter(),
            'd1': AverageMeter(),
            'd5': AverageMeter(),
            'd10': AverageMeter(),
            'glass_epe': AverageMeter(),
            'alpha_ps': AverageMeter(),  # 新增: 監控 alpha 值
        }

        start_time = time.time()
        accum_steps = self.args.accumulation_steps

        for batch_idx, batch in enumerate(self.train_loader):
            # 檢查是否達到最大步數
            if self.global_step >= self.args.num_steps:
                break

            # 移動數據到 GPU
            left = batch['left'].to(self.device)
            right = batch['right'].to(self.device)
            disp_gt = batch['disparity'].to(self.device)
            valid_mask = batch['valid_mask'].to(self.device)
            glass_mask = batch['glass_mask'].to(self.device)

            # 計算並設置 alpha cap
            alpha_cap = self._compute_alpha_cap()
            model_ref = self.model.module if hasattr(self.model, 'module') else self.model
            model_ref.set_alpha_cap(alpha_cap)

            # 混合精度訓練
            if self.scaler:
                # FP16 模式
                with autocast('cuda', dtype=self.amp_dtype):
                    flow_preds = self.model(left, right, iters=self.args.iters, disparity_gt=disp_gt)
                    disp_preds = [-f[:, :1] for f in flow_preds]

                    pol_diff = None
                    if hasattr(model_ref, 'get_pol_diff'):
                        pol_diff = model_ref.get_pol_diff()
                    loss, metrics = self.criterion(disp_preds, disp_gt, valid_mask, glass_mask, pol_diff)
                    loss = loss / accum_steps

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                    if self.scheduler:
                        self.scheduler.step()

                    self.global_step += 1

            elif self.amp_dtype is not None:
                # BF16 模式
                with autocast('cuda', dtype=self.amp_dtype):
                    flow_preds = self.model(left, right, iters=self.args.iters, disparity_gt=disp_gt)
                    disp_preds = [-f[:, :1] for f in flow_preds]

                    pol_diff = None
                    if hasattr(model_ref, 'get_pol_diff'):
                        pol_diff = model_ref.get_pol_diff()
                    loss, metrics = self.criterion(disp_preds, disp_gt, valid_mask, glass_mask, pol_diff)
                    loss = loss / accum_steps

                loss.backward()

                if (batch_idx + 1) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    if self.scheduler:
                        self.scheduler.step()

                    self.global_step += 1

            else:
                # FP32 模式
                flow_preds = self.model(left, right, iters=self.args.iters, disparity_gt=disp_gt)
                disp_preds = [-f[:, :1] for f in flow_preds]

                pol_diff = None
                if hasattr(model_ref, 'get_pol_diff'):
                    pol_diff = model_ref.get_pol_diff()
                loss, metrics = self.criterion(disp_preds, disp_gt, valid_mask, glass_mask, pol_diff)
                loss = loss / accum_steps

                loss.backward()

                if (batch_idx + 1) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    if self.scheduler:
                        self.scheduler.step()

                    self.global_step += 1

            # 更新 meters
            batch_size = left.size(0)
            meters['loss'].update(metrics['loss'], batch_size)
            meters['epe'].update(metrics['epe'], batch_size)
            meters['d1'].update(metrics['d1'], batch_size)
            meters['d5'].update(metrics.get('d5', 0), batch_size)
            meters['d10'].update(metrics.get('d10', 0), batch_size)
            meters['glass_epe'].update(metrics['glass_epe'], batch_size)

            # 記錄 alpha 值
            alpha_values = model_ref.get_alpha_values()
            if 'alpha_ps' in alpha_values and alpha_values['alpha_ps'] is not None:
                meters['alpha_ps'].update(alpha_values['alpha_ps'], 1)

            # Logging
            if self.global_step > 0 and self.global_step % 100 == 0:
                elapsed = time.time() - start_time
                print(f"  Step {self.global_step}/{self.args.num_steps} | "
                      f"Loss: {meters['loss'].avg:.4f} | "
                      f"Glass EPE: {meters['glass_epe'].avg:.2f} | "
                      f"D1: {meters['d1'].avg:.2f}% | "
                      f"Alpha_ps: {meters['alpha_ps'].avg:.4f} | "
                      f"Alpha_cap: {alpha_cap:.4f} | "
                      f"Time: {elapsed:.1f}s")

                # TensorBoard logging
                if self.writer:
                    self.writer.add_scalar('train/loss', meters['loss'].avg, self.global_step)
                    self.writer.add_scalar('train/epe', meters['epe'].avg, self.global_step)
                    self.writer.add_scalar('train/glass_epe', meters['glass_epe'].avg, self.global_step)
                    self.writer.add_scalar('train/d1', meters['d1'].avg, self.global_step)
                    self.writer.add_scalar('train/alpha_ps', meters['alpha_ps'].avg, self.global_step)
                    self.writer.add_scalar('train/alpha_cap', alpha_cap, self.global_step)

                # Reset meters
                for m in meters.values():
                    m.reset()
                start_time = time.time()

            # Validation
            if self.global_step > 0 and self.global_step % self.args.val_freq == 0:
                val_metrics = self._validate()
                self._log_validation(val_metrics)
                self.model.train()


def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(
        description='PIDS Cross-Attention Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 數據參數
    parser.add_argument('--data_dir', type=str, required=True,
                        help='數據目錄')
    parser.add_argument('--output_dir', type=str, default='./checkpoints_cross_attention',
                        help='輸出目錄')

    # 模型參數
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--context_dim', type=int, default=128)
    parser.add_argument('--feature_dim', type=int, default=128)
    parser.add_argument('--corr_levels', type=int, default=4)
    parser.add_argument('--corr_radius', type=int, default=4)
    parser.add_argument('--iters', type=int, default=12)

    # 偏振參數
    parser.add_argument('--pol_dim', type=int, default=64)
    parser.add_argument('--pol_threshold', type=float, default=0.05)
    parser.add_argument('--pol_sharpness', type=float, default=20.0)
    parser.add_argument('--pol_weight', type=float, default=2.0)
    parser.add_argument('--pol_lr_mult', type=float, default=5.0)

    # Cross-Attention 專用參數
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Cross-Attention heads 數量')
    parser.add_argument('--pool_size', type=int, default=8,
                        help='Pooled Attention 的池化大小')
    parser.add_argument('--enable_stereo_to_pol', action='store_true',
                        help='啟用 Stereo->Pol 方向的 attention')
    parser.add_argument('--alpha_cap_start', type=float, default=0.05,
                        help='Alpha cap warmup 起始值')
    parser.add_argument('--alpha_cap_warmup', type=int, default=5000,
                        help='Alpha cap warmup 步數')
    parser.add_argument('--nopol_ablation', action='store_true',
                        help='Ablation: 強制 alpha=0，等效無偏振（同架構但 pol 分支不貢獻）')

    # 訓練參數
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--num_steps', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--weight_decay', type=float, default=0.00001)
    parser.add_argument('--adam_eps', type=float, default=1e-6)
    parser.add_argument('--clip_grad', type=float, default=1.0)
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adamw', 'adam'])
    parser.add_argument('--scheduler', type=str, default='onecycle',
                        choices=['cosine', 'onecycle', 'none'])

    # Loss 參數
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--glass_weight', type=float, default=5.0)
    parser.add_argument('--d1_weight', type=float, default=0.1)
    parser.add_argument('--max_disp', type=float, default=576.0)

    # 精度參數
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--bf16', action='store_true')

    # 其他
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--freeze_fnet', action='store_true')
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--val_freq', type=int, default=500)
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    # 為了相容性，加入 dual_stream (會被強制設為 True)
    parser.add_argument('--dual_stream', action='store_true', default=True)

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("PIDS Cross-Attention Training")
    if args.nopol_ablation:
        print("*** ABLATION MODE: nopol (alpha=0, pol branch disabled) ***")
    print("=" * 60)
    print(f"Data dir: {args.data_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Pol dim: {args.pol_dim}")
    print(f"Num heads: {args.num_heads}")
    print(f"Pool size: {args.pool_size}")
    print(f"Alpha cap start: {args.alpha_cap_start}")
    print(f"Alpha cap warmup: {args.alpha_cap_warmup}")
    print(f"Nopol ablation: {args.nopol_ablation}")
    print("=" * 60)

    trainer = CrossAttentionTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
