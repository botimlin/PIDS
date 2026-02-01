#!/usr/bin/env python3
"""
Baseline RAFT-Stereo Training Script (Clean Version)

這是一個乾淨的 RAFT-Stereo 訓練腳本，用於 baseline 對比實驗。
不包含任何 PIDS 偏振相關的修改。

Usage:
    python train_baseline.py \
        --data_dir ./PIDS_dataset_pol_V6 \
        --output_dir ./checkpoints_baseline \
        --pretrained ./models/raftstereo-sceneflow.pth \
        --batch_size 8 \
        --num_steps 60000

Author: Po-Ting Lin
Date: 2026-01-16
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Optional
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

# 添加 RAFT-Stereo 路徑 (假設在 /workspace/RAFT-Stereo)
sys.path.insert(0, '/workspace/RAFT-Stereo')
sys.path.insert(0, '/workspace/RAFT-Stereo/core')

# 官方 RAFT-Stereo
from raft_stereo import RAFTStereo

# 我們的 dataset (只用 stereo 部分，不用 pol_diff)
from pids_dataset import PIDSSyntheticDataset


class AverageMeter:
    """計算並存儲平均值和當前值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def sequence_loss(flow_preds, flow_gt, valid_mask, gamma=0.9, max_flow=576):
    """
    官方 RAFT-Stereo 的 sequence loss

    Args:
        flow_preds: List of flow predictions [(B, 2, H, W), ...]
        flow_gt: Ground truth flow (B, 2, H, W) - 只用 x-component
        valid_mask: Valid pixels mask (B, 1, H, W)
        gamma: Loss weight decay
        max_flow: Maximum valid flow value
    """
    n_preds = len(flow_preds)

    # 確保 flow_gt 是 (B, 2, H, W) 格式
    if flow_gt.dim() == 3:
        flow_gt = flow_gt.unsqueeze(1)
    if flow_gt.shape[1] == 1:
        flow_gt = torch.cat([flow_gt, torch.zeros_like(flow_gt)], dim=1)

    # Valid mask
    valid = (valid_mask >= 0.5) & (flow_gt[:, :1].abs() < max_flow)

    flow_loss = 0.0
    for i, flow_pred in enumerate(flow_preds):
        weight = gamma ** (n_preds - i - 1)
        # L1 loss on x-component (disparity)
        diff = (flow_pred[:, :1] - flow_gt[:, :1]).abs()
        flow_loss += weight * (valid * diff).mean()

    # 計算 metrics (用最後一個預測)
    flow_pred = flow_preds[-1]
    epe = torch.abs(flow_pred[:, :1] - flow_gt[:, :1])
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        'd1': (epe > 1).float().mean().item() * 100,
        'd3': (epe > 3).float().mean().item() * 100,
        'd5': (epe > 5).float().mean().item() * 100,
    }

    return flow_loss, metrics


def compute_glass_epe(flow_pred, flow_gt, glass_mask):
    """計算 Glass 區域的 EPE"""
    if glass_mask is None or glass_mask.sum() == 0:
        return 0.0

    # 確保維度匹配
    if flow_pred.dim() == 4 and flow_pred.shape[1] == 2:
        disp_pred = flow_pred[:, :1]
    else:
        disp_pred = flow_pred

    if flow_gt.dim() == 3:
        flow_gt = flow_gt.unsqueeze(1)

    epe = torch.abs(disp_pred - flow_gt[:, :1])
    glass_epe = (epe * glass_mask).sum() / (glass_mask.sum() + 1e-6)

    return glass_epe.item()


class BaselineTrainer:
    """乾淨的 RAFT-Stereo Baseline 訓練器"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 建立輸出目錄
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        self.writer = SummaryWriter(self.output_dir / 'tensorboard')

        # 建立模型
        self.model = self._build_model()
        self.model = self.model.to(self.device)

        # 多 GPU
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)

        # 建立 dataset
        self.train_loader, self.val_loader = self._build_dataloaders()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            eps=1e-8
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=args.lr,
            total_steps=args.num_steps + 100,
            pct_start=0.01,
            cycle_momentum=False,
            anneal_strategy='linear'
        )

        # AMP
        self.scaler = GradScaler() if args.mixed_precision else None

        # 訓練狀態
        self.global_step = 0
        self.best_epe = float('inf')
        self.best_glass_epe = float('inf')

        print(f"Baseline Trainer initialized on {self.device}")
        print(f"Output directory: {self.output_dir}")

    def _build_model(self):
        """建立官方 RAFT-Stereo 模型"""
        print("Building original RAFT-Stereo model...")

        # 官方 RAFT-Stereo 需要的 args 格式
        class ModelArgs:
            pass

        model_args = ModelArgs()
        model_args.hidden_dims = [128, 128, 128]
        model_args.context_dims = [128, 128, 128]
        model_args.corr_levels = self.args.corr_levels
        model_args.corr_radius = self.args.corr_radius
        model_args.n_downsample = 2
        model_args.slow_fast_gru = False
        model_args.n_gru_layers = 3
        model_args.mixed_precision = self.args.mixed_precision
        model_args.shared_backbone = False
        model_args.context_norm = 'batch'
        model_args.corr_implementation = 'reg'

        model = RAFTStereo(model_args)

        # 載入預訓練權重
        if self.args.pretrained and os.path.exists(self.args.pretrained):
            print(f"Loading pretrained weights from {self.args.pretrained}")
            checkpoint = torch.load(self.args.pretrained, map_location='cpu')

            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # 處理 module. prefix
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace('module.', '')
                new_state_dict[name] = v

            model.load_state_dict(new_state_dict, strict=False)
            print("  Pretrained weights loaded successfully")

        return model

    def _build_dataloaders(self):
        """建立 DataLoader"""
        print(f"Loading dataset from {self.args.data_dir}")

        train_dataset = PIDSSyntheticDataset(
            data_dir=self.args.data_dir,
            split='train',
            augment=True,
        )

        val_dataset = PIDSSyntheticDataset(
            data_dir=self.args.data_dir,
            split='val',
            augment=False,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")

        return train_loader, val_loader

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """驗證"""
        self.model.eval()

        meters = {
            'epe': AverageMeter(),
            'd1': AverageMeter(),
            'glass_epe': AverageMeter(),
        }

        for batch in self.val_loader:
            left = batch['left'].to(self.device)
            right = batch['right'].to(self.device)
            disp_gt = batch['disparity'].to(self.device)
            valid_mask = batch['valid_mask'].to(self.device)
            glass_mask = batch.get('glass_mask', None)
            if glass_mask is not None:
                glass_mask = glass_mask.to(self.device)

            # Forward (原版 RAFT-Stereo)
            flow_preds = self.model(left, right, iters=self.args.iters + 4)

            # 轉換格式
            disp_preds = [-f[:, :1] for f in flow_preds]
            flow_gt = -disp_gt
            if flow_gt.dim() == 3:
                flow_gt = flow_gt.unsqueeze(1)
            flow_gt = torch.cat([flow_gt, torch.zeros_like(flow_gt)], dim=1)

            # Loss and metrics
            _, metrics = sequence_loss(flow_preds, flow_gt, valid_mask, gamma=self.args.gamma, max_flow=self.args.max_disp)

            # Glass EPE
            glass_epe = compute_glass_epe(flow_preds[-1], flow_gt, glass_mask)

            b = left.shape[0]
            meters['epe'].update(metrics['epe'], b)
            meters['d1'].update(metrics['d1'], b)
            meters['glass_epe'].update(glass_epe, b)

        return {k: m.avg for k, m in meters.items()}

    def train(self):
        """主訓練循環"""
        print("=" * 60)
        print("Baseline RAFT-Stereo Training")
        print(f"Total steps: {self.args.num_steps}")
        print(f"Batch size: {self.args.batch_size}")
        print(f"Learning rate: {self.args.lr}")
        print("=" * 60)

        epoch = 0
        while self.global_step < self.args.num_steps:
            self.model.train()
            epoch += 1
            print(f"\nEpoch {epoch}")

            meters = {
                'loss': AverageMeter(),
                'epe': AverageMeter(),
                'glass_epe': AverageMeter(),
            }

            start_time = time.time()

            for batch_idx, batch in enumerate(self.train_loader):
                if self.global_step >= self.args.num_steps:
                    break

                left = batch['left'].to(self.device)
                right = batch['right'].to(self.device)
                disp_gt = batch['disparity'].to(self.device)
                valid_mask = batch['valid_mask'].to(self.device)
                glass_mask = batch.get('glass_mask', None)
                if glass_mask is not None:
                    glass_mask = glass_mask.to(self.device)

                self.optimizer.zero_grad()

                # Forward
                if self.scaler:
                    with autocast():
                        flow_preds = self.model(left, right, iters=self.args.iters)

                        # 轉換格式 (RAFT-Stereo 輸出 flow，我們要 disparity)
                        flow_gt = -disp_gt
                        if flow_gt.dim() == 3:
                            flow_gt = flow_gt.unsqueeze(1)
                        flow_gt = torch.cat([flow_gt, torch.zeros_like(flow_gt)], dim=1)

                        loss, metrics = sequence_loss(
                            flow_preds, flow_gt, valid_mask,
                            gamma=self.args.gamma,
                            max_flow=self.args.max_disp
                        )

                        # Glass weight
                        if glass_mask is not None and self.args.glass_weight > 1.0:
                            glass_loss = compute_glass_epe(flow_preds[-1], flow_gt, glass_mask)
                            loss = loss + (self.args.glass_weight - 1.0) * glass_loss

                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    flow_preds = self.model(left, right, iters=self.args.iters)

                    flow_gt = -disp_gt
                    if flow_gt.dim() == 3:
                        flow_gt = flow_gt.unsqueeze(1)
                    flow_gt = torch.cat([flow_gt, torch.zeros_like(flow_gt)], dim=1)

                    loss, metrics = sequence_loss(
                        flow_preds, flow_gt, valid_mask,
                        gamma=self.args.gamma,
                        max_flow=self.args.max_disp
                    )

                    if glass_mask is not None and self.args.glass_weight > 1.0:
                        glass_loss = compute_glass_epe(flow_preds[-1], flow_gt, glass_mask)
                        loss = loss + (self.args.glass_weight - 1.0) * glass_loss

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                    self.optimizer.step()

                self.scheduler.step()
                self.global_step += 1

                # Update meters
                b = left.shape[0]
                meters['loss'].update(loss.item(), b)
                meters['epe'].update(metrics['epe'], b)
                glass_epe = compute_glass_epe(flow_preds[-1], flow_gt, glass_mask) if glass_mask is not None else 0
                meters['glass_epe'].update(glass_epe, b)

                # TensorBoard
                if self.global_step % self.args.log_freq == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                    self.writer.add_scalar('train/epe', metrics['epe'], self.global_step)
                    self.writer.add_scalar('train/glass_epe', glass_epe, self.global_step)
                    self.writer.add_scalar('train/lr', lr, self.global_step)

                # Print
                if self.global_step % self.args.print_freq == 0:
                    elapsed = time.time() - start_time
                    lr = self.optimizer.param_groups[0]['lr']
                    print(f"  Step {self.global_step:6d} | "
                          f"Loss: {meters['loss'].avg:.4f} | "
                          f"EPE: {meters['epe'].avg:.3f} | "
                          f"Glass EPE: {meters['glass_epe'].avg:.3f} | "
                          f"LR: {lr:.6f} | "
                          f"Time: {elapsed:.1f}s")

                # Validation
                if self.global_step % self.args.val_freq == 0 and self.global_step > 0:
                    val_metrics = self.validate()

                    print(f"\n  [Val @ Step {self.global_step}]")
                    print(f"    EPE: {val_metrics['epe']:.3f}")
                    print(f"    D1: {val_metrics['d1']:.2f}%")
                    print(f"    Glass EPE: {val_metrics['glass_epe']:.3f}")

                    # TensorBoard
                    self.writer.add_scalar('val/epe', val_metrics['epe'], self.global_step)
                    self.writer.add_scalar('val/d1', val_metrics['d1'], self.global_step)
                    self.writer.add_scalar('val/glass_epe', val_metrics['glass_epe'], self.global_step)

                    # Save best
                    if val_metrics['glass_epe'] < self.best_glass_epe:
                        self.best_glass_epe = val_metrics['glass_epe']
                        self.best_epe = val_metrics['epe']
                        self._save_checkpoint('best')
                        print(f"    ★ New best! Glass EPE: {self.best_glass_epe:.3f}")

                    self.model.train()

                # Save checkpoint
                if self.global_step % self.args.save_freq == 0:
                    self._save_checkpoint(f'step_{self.global_step}')

        # Final save
        self._save_checkpoint('final')
        print(f"\nTraining complete!")
        print(f"Best Glass EPE: {self.best_glass_epe:.3f}")
        print(f"Best EPE: {self.best_epe:.3f}")

    def _save_checkpoint(self, name: str):
        """儲存 checkpoint"""
        model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()

        checkpoint = {
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_epe': self.best_epe,
            'best_glass_epe': self.best_glass_epe,
            'args': vars(self.args),
        }

        path = self.output_dir / f'baseline_{name}.pth'
        torch.save(checkpoint, path)
        print(f"  Checkpoint saved: {path}")


def main():
    parser = argparse.ArgumentParser(description='Baseline RAFT-Stereo Training')

    # Data
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to training dataset')
    parser.add_argument('--output_dir', type=str, default='./checkpoints_baseline',
                        help='Output directory for checkpoints')

    # Model
    parser.add_argument('--corr_levels', type=int, default=4)
    parser.add_argument('--corr_radius', type=int, default=4)
    parser.add_argument('--iters', type=int, default=22)
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained RAFT-Stereo weights')

    # Training
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_steps', type=int, default=60000)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--clip_grad', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--glass_weight', type=float, default=5.0,
                        help='Extra weight for glass region loss')
    parser.add_argument('--max_disp', type=float, default=576.0,
                        help='Maximum valid disparity')
    parser.add_argument('--mixed_precision', action='store_true')

    # Logging
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--val_freq', type=int, default=500)
    parser.add_argument('--save_freq', type=int, default=5000)

    args = parser.parse_args()

    trainer = BaselineTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
