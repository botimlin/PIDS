"""
PIDS Training Script
Stage I: 合成數據預訓練

Usage:
    python train_pids.py --data_dir ./output/output --output_dir ./checkpoints

Copyright (c) 2025-2026 Po-Ting Lin
Released under the MIT License (see LICENSE file).
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
import numpy as np

from pids_dataset import PIDSSyntheticDataset, create_data_loaders

# 使用官方 RAFT-Stereo 模型
sys.path.append('core')
from raft_stereo import RAFTStereo

# 保留自定義 Loss
from pids_model import PIDSStereoLoss


class AverageMeter:
    """追蹤平均值和當前值"""

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


class Trainer:
    """PIDS 訓練器"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # 創建輸出目錄
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 設置 TensorBoard
        self.writer = SummaryWriter(self.output_dir / 'logs')

        # 建立模型
        self.model = self._build_model()

        # 建立損失函數
        self.criterion = PIDSStereoLoss(
            gamma=args.gamma,
            max_disp=args.max_disp,
            glass_weight=args.glass_weight,
        )

        # 建立優化器
        self.optimizer = self._build_optimizer()

        # 建立學習率調度器
        self.scheduler = self._build_scheduler()

        # 混合精度訓練
        # BFloat16: 數值範圍同 FP32 (不會 Loss 爆炸)，顯存同 FP16 (不會 OOM)
        # H100 對 BF16 有特殊優化，速度最快
        if args.bf16:
            self.amp_dtype = torch.bfloat16
            self.scaler = None  # BF16 不需要 GradScaler
        elif args.mixed_precision:
            self.amp_dtype = torch.float16
            self.scaler = GradScaler('cuda')
        else:
            self.amp_dtype = None
            self.scaler = None

        # 載入數據
        self.train_loader, self.val_loader = self._build_data_loaders()

        # 訓練狀態
        self.start_epoch = 0
        self.global_step = 0
        self.best_glass_epe = float('inf')  # 使用 Glass EPE 作為最佳模型標準
        self.best_composite_score = float('inf')  # 複合式指標 (Glass EPE + D1_weight * D1_error)

        # 載入檢查點（如果有）
        if args.resume:
            self._load_checkpoint(args.resume)

        # 保存配置
        self._save_config()

    def _build_model(self) -> nn.Module:
        """建立模型 - 使用官方 RAFT-Stereo"""
        # 創建 args 對象給 RAFTStereo (官方格式)
        class ModelArgs:
            pass

        model_args = ModelArgs()
        # 官方用 hidden_dims (list)，不是 hidden_dim
        model_args.hidden_dims = [self.args.hidden_dim] * 3  # [128, 128, 128]
        model_args.context_dims = [self.args.context_dim] * 3
        model_args.corr_levels = self.args.corr_levels
        model_args.corr_radius = self.args.corr_radius
        model_args.n_downsample = 2  # 官方預設
        model_args.slow_fast_gru = False  # 官方預設
        model_args.n_gru_layers = 3  # 官方預設
        model_args.mixed_precision = self.args.mixed_precision or self.args.bf16
        model_args.shared_backbone = False  # 官方預設
        model_args.context_norm = 'batch'  # 官方預設
        model_args.corr_implementation = 'reg'  # 官方預設 (reg, alt, reg_cuda, alt_cuda)

        model = RAFTStereo(model_args)

        # 載入預訓練權重 (論文: θ_init ← θ_pre from Scene Flow)
        if self.args.pretrained:
            self._load_pretrained_weights(model, self.args.pretrained)

        # 凍結 Feature Encoder (減少過擬合，只微調 context + GRU)
        if self.args.freeze_fnet:
            frozen_count = 0
            for name, param in model.named_parameters():
                if 'fnet' in name:
                    param.requires_grad = False
                    frozen_count += 1
            print(f"Frozen {frozen_count} parameters in feature encoder (fnet)")

        # 多 GPU
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)

        model = model.to(self.device)

        # 參數統計
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        return model

    def _load_pretrained_weights(self, model: nn.Module, pretrained_path: str):
        """
        載入預訓練權重

        支援:
        1. 官方 RAFT-Stereo 權重 (Scene Flow)
        2. 我們自己的 checkpoint

        論文: "We initialize the network parameters θ_init with weights θ_pre
              pre-trained on a large-scale opaque dataset (e.g., Scene Flow)"
        """
        print(f"Loading pretrained weights from: {pretrained_path}")

        checkpoint = torch.load(pretrained_path, map_location='cpu')

        # 判斷 checkpoint 格式
        if 'model_state_dict' in checkpoint:
            # 我們自己的格式
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            # 常見格式
            state_dict = checkpoint['state_dict']
        else:
            # 直接是 state_dict
            state_dict = checkpoint

        # 處理 DataParallel 前綴
        new_state_dict = {}
        for k, v in state_dict.items():
            # 移除 'module.' 前綴 (如果有)
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v

        # 嘗試載入，允許部分匹配
        model_dict = model.state_dict()
        matched_keys = []
        unmatched_keys = []

        for k, v in new_state_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                model_dict[k] = v
                matched_keys.append(k)
            else:
                unmatched_keys.append(k)

        model.load_state_dict(model_dict)

        print(f"  Matched: {len(matched_keys)}/{len(new_state_dict)} layers")
        if unmatched_keys:
            print(f"  Unmatched layers: {unmatched_keys[:5]}..." if len(unmatched_keys) > 5 else f"  Unmatched: {unmatched_keys}")

    def _build_optimizer(self) -> optim.Optimizer:
        """建立優化器

        Note: 根據 PIDS 論文，使用 eps=1e-6 而非預設的 1e-8
        這是為了處理 specular highlights 造成的極端梯度，確保數值穩定性
        """
        if self.args.optimizer == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
                eps=self.args.adam_eps,  # 論文建議 1e-6
            )
        else:
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
                eps=self.args.adam_eps,
            )

        return optimizer

    def _build_scheduler(self):
        """建立學習率調度器"""
        if self.args.scheduler == 'onecycle':
            scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.args.lr,
                total_steps=self.args.num_steps,
                pct_start=0.05,
                cycle_momentum=False,
                anneal_strategy='linear',
            )
        elif self.args.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.args.num_steps,
                eta_min=self.args.lr * 0.01,
            )
        else:
            scheduler = None

        return scheduler

    def _build_data_loaders(self):
        """建立數據載入器 (使用全圖，不 crop)"""
        train_loader, val_loader = create_data_loaders(
            data_dir=self.args.data_dir,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            val_split=self.args.val_split,
        )

        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")

        return train_loader, val_loader

    def _save_config(self):
        """保存訓練配置"""
        config = vars(self.args)
        config['device'] = str(self.device)
        config['timestamp'] = datetime.now().isoformat()

        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存檢查點"""
        model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()

        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_glass_epe': self.best_glass_epe,
            'best_composite_score': self.best_composite_score,
            'args': vars(self.args),
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # 保存最新檢查點
        torch.save(checkpoint, self.output_dir / 'checkpoint_latest.pth')

        # 定期保存
        if (epoch + 1) % self.args.save_freq == 0:
            torch.save(checkpoint, self.output_dir / f'checkpoint_epoch_{epoch+1:04d}.pth')

        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, self.output_dir / 'checkpoint_best.pth')
            print(f"  -> Saved best model (Glass EPE: {self.best_glass_epe:.3f})")

    def _load_checkpoint(self, path: str):
        """載入檢查點"""
        print(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)

        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        # 兼容舊 checkpoint (best_epe) 和新 checkpoint (best_glass_epe)
        self.best_glass_epe = checkpoint.get('best_glass_epe', checkpoint.get('best_epe', float('inf')))
        self.best_composite_score = checkpoint.get('best_composite_score', float('inf'))

        print(f"Resumed from epoch {self.start_epoch}, step {self.global_step}")
        print(f"Best Glass EPE: {self.best_glass_epe:.3f}, Best Composite: {self.best_composite_score:.3f}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """訓練一個 epoch"""
        self.model.train()

        meters = {
            'loss': AverageMeter(),
            'epe': AverageMeter(),
            'd1': AverageMeter(),
            'd5': AverageMeter(),
            'd10': AverageMeter(),
            'glass_epe': AverageMeter(),
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

            # 混合精度訓練 (FP16 / BF16 / FP32)
            if self.scaler:
                # FP16 模式 (需要 GradScaler)
                with autocast('cuda', dtype=self.amp_dtype):
                    flow_preds = self.model(left, right, iters=self.args.iters)
                    disp_preds = [-f for f in flow_preds]
                    loss, metrics = self.criterion(disp_preds, disp_gt, valid_mask, glass_mask)
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
                # BF16 模式 (不需要 GradScaler，數值範圍同 FP32)
                with autocast('cuda', dtype=self.amp_dtype):
                    flow_preds = self.model(left, right, iters=self.args.iters)
                    disp_preds = [-f for f in flow_preds]
                    loss, metrics = self.criterion(disp_preds, disp_gt, valid_mask, glass_mask)
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
                # FP32 模式 (全精度)
                flow_preds = self.model(left, right, iters=self.args.iters)
                disp_preds = [-f for f in flow_preds]
                loss, metrics = self.criterion(disp_preds, disp_gt, valid_mask, glass_mask)
                loss = loss / accum_steps

                loss.backward()

                if (batch_idx + 1) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    if self.scheduler:
                        self.scheduler.step()

                    self.global_step += 1

            # 更新統計
            batch_size = left.size(0)
            meters['loss'].update(metrics['loss'], batch_size)
            meters['epe'].update(metrics['epe'], batch_size)
            meters['d1'].update(metrics['d1'], batch_size)
            meters['d5'].update(metrics.get('d5', 0), batch_size)
            meters['d10'].update(metrics.get('d10', 0), batch_size)
            meters['glass_epe'].update(metrics['glass_epe'], batch_size)

            # TensorBoard (每個 global step)
            if (batch_idx + 1) % accum_steps == 0 and self.global_step % self.args.log_freq == 0:
                lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('train/loss', metrics['loss'], self.global_step)
                self.writer.add_scalar('train/epe', metrics['epe'], self.global_step)
                self.writer.add_scalar('train/d1', metrics['d1'], self.global_step)
                self.writer.add_scalar('train/d5', metrics.get('d5', 0), self.global_step)
                self.writer.add_scalar('train/d10', metrics.get('d10', 0), self.global_step)
                self.writer.add_scalar('train/glass_epe', metrics['glass_epe'], self.global_step)
                self.writer.add_scalar('train/lr', lr, self.global_step)

            # 打印進度
            if (batch_idx + 1) % accum_steps == 0 and self.global_step % self.args.print_freq == 0:
                elapsed = time.time() - start_time
                lr = self.optimizer.param_groups[0]['lr']
                eff_batch = self.args.batch_size * accum_steps
                print(f"  Step {self.global_step:6d} | "
                      f"Loss: {meters['loss'].avg:.4f} | "
                      f"EPE: {meters['epe'].avg:.3f} | "
                      f"D1: {meters['d1'].avg:.2f}% | "
                      f"Glass EPE: {meters['glass_epe'].avg:.3f} | "
                      f"LR: {lr:.6f} | "
                      f"EffBatch: {eff_batch} | "
                      f"Time: {elapsed:.1f}s")

            # 按 step 驗證 (更密集以抓 best checkpoint)
            if (batch_idx + 1) % accum_steps == 0 and self.global_step % self.args.val_freq == 0 and self.global_step > 0:
                val_metrics = self.validate()
                self._log_validation(val_metrics)
                self.model.train()  # 切回訓練模式

        return {k: v.avg for k, v in meters.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """驗證"""
        self.model.eval()

        meters = {
            'loss': AverageMeter(),
            'epe': AverageMeter(),
            'd1': AverageMeter(),
            'd3': AverageMeter(),
            'd5': AverageMeter(),
            'd10': AverageMeter(),
            'glass_epe': AverageMeter(),
        }

        for batch in self.val_loader:
            left = batch['left'].to(self.device)
            right = batch['right'].to(self.device)
            disp_gt = batch['disparity'].to(self.device)
            valid_mask = batch['valid_mask'].to(self.device)
            glass_mask = batch['glass_mask'].to(self.device)

            # 使用更多迭代進行驗證
            flow_preds = self.model(left, right, iters=self.args.iters + 4)
            disp_preds = [-f for f in flow_preds]
            loss, metrics = self.criterion(disp_preds, disp_gt, valid_mask, glass_mask)

            batch_size = left.size(0)
            meters['loss'].update(metrics['loss'], batch_size)
            meters['epe'].update(metrics['epe'], batch_size)
            meters['d1'].update(metrics['d1'], batch_size)
            meters['d3'].update(metrics['d3'], batch_size)
            meters['d5'].update(metrics.get('d5', 0), batch_size)
            meters['d10'].update(metrics.get('d10', 0), batch_size)
            meters['glass_epe'].update(metrics['glass_epe'], batch_size)

        return {k: v.avg for k, v in meters.items()}

    def _log_validation(self, val_metrics: Dict[str, float]):
        """記錄驗證結果並保存 checkpoint"""
        # 計算複合式指標: Glass EPE + d1_weight * D1_error_rate
        # D1 是錯誤率(越低越好)，所以直接加權
        # 這樣可以同時考慮 Glass EPE (精度) 和 D1 (閾值誤差率)
        d1_error = val_metrics['d1']  # 已經是百分比形式
        composite_score = val_metrics['glass_epe'] + self.args.d1_weight * d1_error

        print(f"\n  [Val @ Step {self.global_step}] "
              f"Loss: {val_metrics['loss']:.4f} | "
              f"EPE: {val_metrics['epe']:.3f} | "
              f"D1: {val_metrics['d1']:.2f}% | "
              f"D3: {val_metrics['d3']:.2f}% | "
              f"D5: {val_metrics.get('d5', 0):.2f}% | "
              f"D10: {val_metrics.get('d10', 0):.2f}% | "
              f"Glass EPE: {val_metrics['glass_epe']:.3f} | "
              f"Composite: {composite_score:.3f}\n")

        # TensorBoard
        self.writer.add_scalar('val/loss', val_metrics['loss'], self.global_step)
        self.writer.add_scalar('val/epe', val_metrics['epe'], self.global_step)
        self.writer.add_scalar('val/d1', val_metrics['d1'], self.global_step)
        self.writer.add_scalar('val/d3', val_metrics['d3'], self.global_step)
        self.writer.add_scalar('val/d5', val_metrics.get('d5', 0), self.global_step)
        self.writer.add_scalar('val/d10', val_metrics.get('d10', 0), self.global_step)
        self.writer.add_scalar('val/glass_epe', val_metrics['glass_epe'], self.global_step)
        self.writer.add_scalar('val/composite_score', composite_score, self.global_step)

        # 檢查是否為最佳 (使用複合式指標)
        is_best = composite_score < self.best_composite_score
        if is_best:
            old_best = self.best_composite_score
            self.best_composite_score = composite_score
            # 同時更新 best_glass_epe (用於兼容舊 checkpoint)
            if val_metrics['glass_epe'] < self.best_glass_epe:
                self.best_glass_epe = val_metrics['glass_epe']
            print(f"  *** New best Composite Score: {self.best_composite_score:.3f} "
                  f"(Glass EPE: {val_metrics['glass_epe']:.3f}, D1: {d1_error:.2f}%) ***\n")

        # 保存 checkpoint
        self._save_checkpoint_by_step(is_best)

    def _save_checkpoint_by_step(self, is_best: bool = False):
        """按 step 保存 checkpoint"""
        model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()

        checkpoint = {
            'global_step': self.global_step,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_glass_epe': self.best_glass_epe,
            'best_composite_score': self.best_composite_score,
            'args': vars(self.args),
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # 保存最新 checkpoint
        torch.save(checkpoint, self.output_dir / 'checkpoint_latest.pth')

        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, self.output_dir / 'checkpoint_best.pth')

    def train(self):
        """主訓練循環"""
        print("=" * 60)
        print("PIDS Training - Stage I (Synthetic Pre-training)")
        print("=" * 60)
        eff_batch = self.args.batch_size * self.args.accumulation_steps
        print(f"Effective batch size: {eff_batch} (batch={self.args.batch_size} x accum={self.args.accumulation_steps})")
        print(f"Validation every {self.args.val_freq} steps")
        print("=" * 60)

        epoch = self.start_epoch
        while self.global_step < self.args.num_steps:
            print(f"\nEpoch {epoch + 1}")
            print("-" * 40)

            # 訓練 (驗證已在 train_epoch 內按 step 進行)
            train_metrics = self.train_epoch(epoch)

            # 每個 epoch 結束時也保存一次
            self._save_checkpoint(epoch, is_best=False)

            epoch += 1

        print("\n" + "=" * 60)
        print(f"Training completed! Best Glass EPE: {self.best_glass_epe:.3f}")
        print(f"Checkpoints saved to: {self.output_dir}")
        print("=" * 60)

        self.writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description='PIDS Training')

    # 數據
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to synthetic data directory')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Output directory for checkpoints')

    # 模型
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--context_dim', type=int, default=128)
    parser.add_argument('--feature_dim', type=int, default=128)
    parser.add_argument('--corr_levels', type=int, default=4)
    parser.add_argument('--corr_radius', type=int, default=4)
    parser.add_argument('--iters', type=int, default=12,
                        help='Number of iterations for disparity refinement')

    # 訓練
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps (effective batch = batch_size * accumulation_steps)')
    parser.add_argument('--num_steps', type=int, default=100000,
                        help='Total number of training steps')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00001)
    parser.add_argument('--adam_eps', type=float, default=1e-6,
                        help='Adam epsilon (論文建議 1e-6 處理 specular highlights)')
    parser.add_argument('--clip_grad', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.9,
                        help='Loss weight decay factor')
    parser.add_argument('--glass_weight', type=float, default=3.0,
                        help='Extra weight for glass region loss (實驗 #11 最佳值)')
    parser.add_argument('--max_disp', type=float, default=576.0)
    parser.add_argument('--d1_weight', type=float, default=0.1,
                        help='Weight for D1 in composite score (composite = glass_epe + d1_weight * d1_error)')

    # 優化器
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw'])
    parser.add_argument('--scheduler', type=str, default='onecycle',
                        choices=['onecycle', 'cosine', 'none'])

    # 數據載入 (不使用 crop，保持全圖以維持全局幾何上下文)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio (預設 0.2 = 80/20 分割)')

    # 其他
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training (FP16)')
    parser.add_argument('--bf16', action='store_true',
                        help='Use BFloat16 mixed precision (推薦 H100，數值範圍同 FP32 但顯存同 FP16)')
    parser.add_argument('--freeze_fnet', action='store_true',
                        help='Freeze feature encoder (fnet), only train context encoder + GRU')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained weights (e.g., RAFT-Stereo Scene Flow)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--val_freq', type=int, default=500,
                        help='Validate every N steps (更密集以抓 best checkpoint)')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Save checkpoint every N epochs')

    return parser.parse_args()


def main():
    args = parse_args()

    # 設置隨機種子
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.benchmark = True

    # 開始訓練
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
