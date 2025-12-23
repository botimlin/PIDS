"""
PIDS Training Script
Stage I: 合成數據預訓練

Usage:
    python train_pids.py --data_dir ./output/output --output_dir ./checkpoints
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
from torch.cuda.amp import GradScaler, autocast
import numpy as np

from pids_dataset import PIDSSyntheticDataset, create_data_loaders
from pids_model import PIDSStereo, PIDSStereoLoss, build_model


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
        self.scaler = GradScaler() if args.mixed_precision else None

        # 載入數據
        self.train_loader, self.val_loader = self._build_data_loaders()

        # 訓練狀態
        self.start_epoch = 0
        self.global_step = 0
        self.best_epe = float('inf')

        # 載入檢查點（如果有）
        if args.resume:
            self._load_checkpoint(args.resume)

        # 保存配置
        self._save_config()

    def _build_model(self) -> nn.Module:
        """建立模型"""
        model_cfg = {
            'hidden_dim': self.args.hidden_dim,
            'context_dim': self.args.context_dim,
            'feature_dim': self.args.feature_dim,
            'corr_levels': self.args.corr_levels,
            'corr_radius': self.args.corr_radius,
            'iters': self.args.iters,
        }

        model = build_model(model_cfg)

        # 載入預訓練權重 (論文: θ_init ← θ_pre from Scene Flow)
        if self.args.pretrained:
            self._load_pretrained_weights(model, self.args.pretrained)

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
        """建立數據載入器"""
        train_loader, val_loader = create_data_loaders(
            data_dir=self.args.data_dir,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            crop_size=(self.args.crop_height, self.args.crop_width),
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
            'best_epe': self.best_epe,
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
            print(f"  -> Saved best model (EPE: {self.best_epe:.3f})")

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
        self.best_epe = checkpoint.get('best_epe', float('inf'))

        print(f"Resumed from epoch {self.start_epoch}, step {self.global_step}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """訓練一個 epoch"""
        self.model.train()

        meters = {
            'loss': AverageMeter(),
            'epe': AverageMeter(),
            'd1': AverageMeter(),
            'glass_epe': AverageMeter(),
        }

        start_time = time.time()

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

            self.optimizer.zero_grad()

            # 混合精度訓練
            if self.scaler:
                with autocast():
                    disp_preds = self.model(left, right, iters=self.args.iters)
                    loss, metrics = self.criterion(disp_preds, disp_gt, valid_mask, glass_mask)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                disp_preds = self.model(left, right, iters=self.args.iters)
                loss, metrics = self.criterion(disp_preds, disp_gt, valid_mask, glass_mask)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                self.optimizer.step()

            # 更新學習率
            if self.scheduler:
                self.scheduler.step()

            # 更新統計
            batch_size = left.size(0)
            meters['loss'].update(metrics['loss'], batch_size)
            meters['epe'].update(metrics['epe'], batch_size)
            meters['d1'].update(metrics['d1'], batch_size)
            meters['glass_epe'].update(metrics['glass_epe'], batch_size)

            # TensorBoard
            if self.global_step % self.args.log_freq == 0:
                lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('train/loss', metrics['loss'], self.global_step)
                self.writer.add_scalar('train/epe', metrics['epe'], self.global_step)
                self.writer.add_scalar('train/d1', metrics['d1'], self.global_step)
                self.writer.add_scalar('train/glass_epe', metrics['glass_epe'], self.global_step)
                self.writer.add_scalar('train/lr', lr, self.global_step)

            # 打印進度
            if self.global_step % self.args.print_freq == 0:
                elapsed = time.time() - start_time
                lr = self.optimizer.param_groups[0]['lr']
                print(f"  Step {self.global_step:6d} | "
                      f"Loss: {meters['loss'].avg:.4f} | "
                      f"EPE: {meters['epe'].avg:.3f} | "
                      f"D1: {meters['d1'].avg:.2f}% | "
                      f"Glass EPE: {meters['glass_epe'].avg:.3f} | "
                      f"LR: {lr:.6f} | "
                      f"Time: {elapsed:.1f}s")

            self.global_step += 1

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
            'glass_epe': AverageMeter(),
        }

        for batch in self.val_loader:
            left = batch['left'].to(self.device)
            right = batch['right'].to(self.device)
            disp_gt = batch['disparity'].to(self.device)
            valid_mask = batch['valid_mask'].to(self.device)
            glass_mask = batch['glass_mask'].to(self.device)

            # 使用更多迭代進行驗證
            disp_preds = self.model(left, right, iters=self.args.iters + 4)
            loss, metrics = self.criterion(disp_preds, disp_gt, valid_mask, glass_mask)

            batch_size = left.size(0)
            meters['loss'].update(metrics['loss'], batch_size)
            meters['epe'].update(metrics['epe'], batch_size)
            meters['d1'].update(metrics['d1'], batch_size)
            meters['d3'].update(metrics['d3'], batch_size)
            meters['glass_epe'].update(metrics['glass_epe'], batch_size)

        return {k: v.avg for k, v in meters.items()}

    def train(self):
        """主訓練循環"""
        print("=" * 60)
        print("PIDS Training - Stage I (Synthetic Pre-training)")
        print("=" * 60)

        epoch = self.start_epoch
        while self.global_step < self.args.num_steps:
            print(f"\nEpoch {epoch + 1}")
            print("-" * 40)

            # 訓練
            train_metrics = self.train_epoch(epoch)

            # 驗證
            if (epoch + 1) % self.args.val_freq == 0:
                val_metrics = self.validate()

                print(f"\n  Validation | "
                      f"Loss: {val_metrics['loss']:.4f} | "
                      f"EPE: {val_metrics['epe']:.3f} | "
                      f"D1: {val_metrics['d1']:.2f}% | "
                      f"D3: {val_metrics['d3']:.2f}% | "
                      f"Glass EPE: {val_metrics['glass_epe']:.3f}")

                # TensorBoard
                self.writer.add_scalar('val/loss', val_metrics['loss'], self.global_step)
                self.writer.add_scalar('val/epe', val_metrics['epe'], self.global_step)
                self.writer.add_scalar('val/d1', val_metrics['d1'], self.global_step)
                self.writer.add_scalar('val/d3', val_metrics['d3'], self.global_step)
                self.writer.add_scalar('val/glass_epe', val_metrics['glass_epe'], self.global_step)

                # 檢查是否為最佳
                is_best = val_metrics['epe'] < self.best_epe
                if is_best:
                    self.best_epe = val_metrics['epe']

                # 保存檢查點
                self._save_checkpoint(epoch, is_best)
            else:
                self._save_checkpoint(epoch)

            epoch += 1

        print("\n" + "=" * 60)
        print(f"Training completed! Best EPE: {self.best_epe:.3f}")
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
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_steps', type=int, default=100000,
                        help='Total number of training steps')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00001)
    parser.add_argument('--adam_eps', type=float, default=1e-6,
                        help='Adam epsilon (論文建議 1e-6 處理 specular highlights)')
    parser.add_argument('--clip_grad', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.9,
                        help='Loss weight decay factor')
    parser.add_argument('--glass_weight', type=float, default=1.0,
                        help='Extra weight for glass region loss (1.0=no extra weight, 論文未使用此功能)')
    parser.add_argument('--max_disp', type=float, default=192.0)

    # 優化器
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw'])
    parser.add_argument('--scheduler', type=str, default='onecycle',
                        choices=['onecycle', 'cosine', 'none'])

    # 數據增強
    parser.add_argument('--crop_height', type=int, default=320)
    parser.add_argument('--crop_width', type=int, default=480)
    parser.add_argument('--num_workers', type=int, default=4)

    # 其他
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained weights (e.g., RAFT-Stereo Scene Flow)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--val_freq', type=int, default=1,
                        help='Validate every N epochs')
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
