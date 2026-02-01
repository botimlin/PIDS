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
from curriculum_sampler import (
    CurriculumSampler,
    CurriculumConfig,
    PolModuleFreezer,
    BackboneFreezer,  # Exp #29: Freeze Backbone First
    GradualUnfreezeLRScheduler,  # Exp #30: Gradual Unfreeze
    MixedBatchCollator,
    V2PolModuleFreezer,  # Exp #31: Freeze V2 modules on nopol data
)

# 使用官方 RAFT-Stereo 模型
sys.path.append('core')
from raft_stereo import RAFTStereo

# 保留自定義 Loss 和 Dual-Stream 架構
from pids_model import (
    PIDSStereoLoss,
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
    build_model_dual_stream,
)


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


class DirectionalImpulseDescent:
    """
    Directional Impulse Descent (DID) - 狀態驅動的局部脈衝控制器

    設計哲學：
    - OneCycle = 全局退火曲線 (reference trajectory)
    - DID = 狀態驅動的局部脈衝 (event-driven impulse disturbance)

    狀態機：
    NORMAL → (trigger) → IMPULSE → (T_impulse) → COOLDOWN → (T_cool) → PROTECTED → (P cycles) → NORMAL

    觸發條件（同時滿足）：
    1. Train slope plateau: EMA slope < ε (percentile threshold)
    2. Val best stale: moving-best 沒更新超過 M 個 cycles（主要條件）
    3. Val oscillating: val EPE 在 ±δ 內震盪（輔助證據，非主要門檻）
    4. Direction consistent: slope EMA 方向穩定
    5. Not in late phase: progress < late_lock
    6. Under trigger limit: trigger_count < max_triggers
    7. Not in protection: not in_protection_period

    改進點：
    - B: 漸進式 impulse (第一次 ×2，之後 ×3)
    - C: 參數群分離 (pol module 全擾動，backbone 只 ×1.2)
    - D: oscillation 變輔助證據，moving-best 才是主要條件
    - E: EPE-adaptive impulse (EPE 低時脈衝自動縮小，避免過度擾動)
    """

    # 狀態常數
    NORMAL = 0
    IMPULSE = 1
    COOLDOWN = 2
    PROTECTED = 3

    STATE_NAMES = {0: 'NORMAL', 1: 'IMPULSE', 2: 'COOLDOWN', 3: 'PROTECTED'}

    def __init__(
        self,
        # Plateau detection
        epsilon_percentile: float = 0.10,
        train_ema_window: int = 300,
        slope_history_size: int = 2000,
        val_M: int = 5,
        delta: float = 0.5,
        # Direction check
        ema_alpha: float = 0.3,
        min_slope_ratio: float = 0.1,
        # DID impulse
        gamma_up: float = 3.0,
        gamma_up_first: float = 2.0,  # 改進 B: 第一次 impulse 更保守
        gamma_backbone: float = 1.2,  # 改進 C: backbone 只輕微擾動
        t_impulse: int = 200,
        gamma_down: float = 0.1,
        t_cool: int = 100,
        cooldown_mode: str = 'relative_base',  # 'relative_base' or 'relative_impulse'
        # Protection
        max_triggers: int = 3,
        protection_cycles: int = 3,
        late_lock: float = 0.75,
        val_freq: int = 500,
        # 改進 D: moving-best staleness
        best_stale_cycles: int = 4,  # moving-best 沒更新多少 cycles 才觸發
        # 改進 E: EPE-adaptive impulse
        epe_ref: float = 8.0,  # EPE 參考值，EPE >= epe_ref 時用完整 γ
        epe_smooth_alpha: float = 0.4,  # EPE EMA 平滑係數 (保險絲 1)
        delta_gamma_max: float = 0.3,  # 每次 impulse γ 最大變化量 (保險絲 2)
    ):
        # Plateau detection params
        self.epsilon_percentile = epsilon_percentile
        self.train_ema_window = train_ema_window
        self.slope_history_size = slope_history_size
        self.val_M = val_M
        self.delta = delta

        # Direction check params
        self.ema_alpha = ema_alpha
        self.min_slope_ratio = min_slope_ratio

        # DID impulse params
        self.gamma_up = gamma_up
        self.gamma_up_first = gamma_up_first  # 改進 B
        self.gamma_backbone = gamma_backbone  # 改進 C
        self.t_impulse = t_impulse
        self.gamma_down = gamma_down
        self.t_cool = t_cool
        self.cooldown_mode = cooldown_mode

        # Protection params
        self.max_triggers = max_triggers
        self.protection_cycles = protection_cycles
        self.late_lock = late_lock
        self.val_freq = val_freq

        # 改進 D: moving-best staleness
        self.best_stale_cycles = best_stale_cycles

        # 改進 E: EPE-adaptive impulse
        self.epe_ref = epe_ref
        self.epe_smooth_alpha = epe_smooth_alpha
        self.delta_gamma_max = delta_gamma_max

        # ========== 狀態 ==========
        self.state = self.NORMAL
        self.trigger_count = 0
        self.last_trigger_step = -999999

        # 狀態轉換計時
        self.impulse_start_step = 0
        self.cooldown_start_step = 0
        self.protection_start_cycle = 0
        self.lr_at_impulse_start = 0.0  # 用於 relative_impulse mode

        # ========== 訊號追蹤 ==========
        # Train slope tracking
        self.train_losses = []  # 原始 loss 歷史
        self.train_slope_history = []  # 相對斜率歷史 (s_t = |ΔL| / L)
        self.train_slope_ema = 0.0

        # Val tracking
        self.val_history = []  # val Glass EPE 歷史
        self.val_slope_ema = 0.0
        self.val_slope_history = []  # val slope 歷史 (用於 direction check)

        # Spike filtering (from previous design)
        self.val_baseline = None
        self.outlier_pct = 0.10

        # 改進 D: moving-best tracking
        self.val_best = float('inf')
        self.cycles_since_best = 0

        # 改進 E: EPE-adaptive state (保險絲 1: smoothed EPE)
        self.epe_smooth = None  # EMA of val Glass EPE, 初始化為 None
        self.last_gamma_eff = 1.0  # 上一次 impulse 的有效 γ (保險絲 2: 變化率限制)

        # 驗證週期計數
        self.val_cycle_count = 0

    def update_train(self, loss: float, step: int):
        """每個 training step 更新 train 訊號"""
        self.train_losses.append(loss)

        # 計算相對斜率 s_t = |ΔL| / L
        if len(self.train_losses) >= 2:
            delta_l = abs(self.train_losses[-1] - self.train_losses[-2])
            l_t = max(self.train_losses[-1], 1e-8)  # 避免除以零
            s_t = delta_l / l_t
            self.train_slope_history.append(s_t)

            # 維持歷史大小
            if len(self.train_slope_history) > self.slope_history_size:
                self.train_slope_history.pop(0)

            # 更新 EMA
            if self.train_slope_ema == 0:
                self.train_slope_ema = s_t
            else:
                self.train_slope_ema = (
                    self.ema_alpha * s_t +
                    (1 - self.ema_alpha) * self.train_slope_ema
                )

        # 維持 train_losses 大小
        if len(self.train_losses) > self.train_ema_window:
            self.train_losses.pop(0)

    def update_val(self, val_glass_epe: float, step: int, total_steps: int) -> dict:
        """
        每次 validation 後更新 val 訊號並檢查是否觸發 DID

        Returns:
            dict with keys: triggered, state, message, multiplier, multiplier_backbone
        """
        self.val_cycle_count += 1
        result = {
            'triggered': False,
            'state': self.STATE_NAMES[self.state],
            'message': '',
            'multiplier': 1.0,
            'multiplier_backbone': 1.0,  # 改進 C: backbone 的乘數
        }

        # ========== Spike filtering ==========
        if self.val_baseline is None:
            self.val_baseline = val_glass_epe

        lower = self.val_baseline * (1 - self.outlier_pct)
        upper = self.val_baseline * (1 + self.outlier_pct)

        is_spike = val_glass_epe < lower or val_glass_epe > upper
        if not is_spike:
            # 有效值，納入計算
            self.val_history.append(val_glass_epe)

            # 更新 baseline (如果更好)
            if val_glass_epe < self.val_baseline:
                self.val_baseline = val_glass_epe

            # 改進 D: 更新 moving-best tracking
            if val_glass_epe < self.val_best:
                self.val_best = val_glass_epe
                self.cycles_since_best = 0
            else:
                self.cycles_since_best += 1

            # 改進 E: 更新 smoothed EPE (保險絲 1)
            if self.epe_smooth is None:
                self.epe_smooth = val_glass_epe
            else:
                self.epe_smooth = (
                    self.epe_smooth_alpha * val_glass_epe +
                    (1 - self.epe_smooth_alpha) * self.epe_smooth
                )

            # 計算 val slope 並更新 EMA
            if len(self.val_history) >= 2:
                delta_val = self.val_history[-1] - self.val_history[-2]
                self.val_slope_history.append(delta_val)

                if self.val_slope_ema == 0:
                    self.val_slope_ema = delta_val
                else:
                    self.val_slope_ema = (
                        self.ema_alpha * delta_val +
                        (1 - self.ema_alpha) * self.val_slope_ema
                    )

            # 維持歷史大小
            if len(self.val_history) > self.val_M + 2:
                self.val_history.pop(0)
            if len(self.val_slope_history) > self.val_M + 2:
                self.val_slope_history.pop(0)

        # ========== 狀態機更新 ==========
        if self.state == self.IMPULSE:
            # 檢查是否該轉到 COOLDOWN
            if step - self.impulse_start_step >= self.t_impulse:
                self.state = self.COOLDOWN
                self.cooldown_start_step = step
                result['message'] = f"IMPULSE → COOLDOWN @ step {step}"

        elif self.state == self.COOLDOWN:
            # 檢查是否該轉到 PROTECTED
            if step - self.cooldown_start_step >= self.t_cool:
                self.state = self.PROTECTED
                self.protection_start_cycle = self.val_cycle_count
                result['message'] = f"COOLDOWN → PROTECTED @ step {step}"

        elif self.state == self.PROTECTED:
            # 檢查是否該回到 NORMAL
            cycles_in_protection = self.val_cycle_count - self.protection_start_cycle
            if cycles_in_protection >= self.protection_cycles:
                self.state = self.NORMAL
                result['message'] = f"PROTECTED → NORMAL @ cycle {self.val_cycle_count}"

        elif self.state == self.NORMAL:
            # 檢查是否該觸發 DID
            if self._should_trigger(step, total_steps):
                self.state = self.IMPULSE
                self.impulse_start_step = step
                self.trigger_count += 1
                self.last_trigger_step = step
                result['triggered'] = True
                result['message'] = f"*** DID TRIGGERED #{self.trigger_count} @ step {step} ***"

        result['state'] = self.STATE_NAMES[self.state]
        return result

    def _should_trigger(self, step: int, total_steps: int) -> bool:
        """
        檢查是否滿足所有觸發條件

        改進 D: 重新設計觸發邏輯
        - 主要條件: moving-best 沒更新 + train slope plateau
        - 輔助證據: val oscillating (有則更確定，沒有也可以)
        """
        progress = step / total_steps

        # 1. 未進入後段
        if progress >= self.late_lock:
            return False

        # 2. 未達觸發上限
        if self.trigger_count >= self.max_triggers:
            return False

        # 3. 不在保護期 (by state machine, already handled)

        # 4. Train slope plateau (必要條件)
        if not self._train_plateau():
            return False

        # 5. Moving-best stale (必要條件) - 改進 D 的核心
        if not self._val_best_stale():
            return False

        # 6. Direction consistent (必要條件)
        if not self._direction_consistent():
            return False

        # 7. Val oscillating (輔助證據，不是必要條件)
        # 如果 val 也在震盪，更確定是 plateau；但沒有也可以觸發
        # oscillating = self._val_oscillating()
        # (這行保留但不作為必要條件)

        return True

    def _val_best_stale(self) -> bool:
        """改進 D: 檢查 moving-best 是否長期未更新"""
        return self.cycles_since_best >= self.best_stale_cycles

    def _train_plateau(self) -> bool:
        """檢查 train slope 是否 plateau"""
        if len(self.train_slope_history) < 100:
            return False

        # 計算 ε = percentile(s_history, epsilon_percentile)
        sorted_slopes = sorted(self.train_slope_history)
        idx = int(len(sorted_slopes) * self.epsilon_percentile)
        epsilon = sorted_slopes[idx]

        # 檢查最近的 slope 是否低於 ε
        recent_window = self.train_slope_history[-50:]  # 最近 50 個
        median_recent = sorted(recent_window)[len(recent_window) // 2]

        return median_recent < epsilon

    def _val_oscillating(self) -> bool:
        """檢查 val 是否在震盪"""
        if len(self.val_history) < self.val_M:
            return False

        recent = self.val_history[-self.val_M:]
        val_range = max(recent) - min(recent)

        return val_range < self.delta

    def _direction_consistent(self) -> bool:
        """檢查方向一致性 (用 EMA)"""
        if len(self.val_slope_history) < 3:
            return False

        # min_slope = 0.1 × median(|Δval|)
        abs_slopes = [abs(s) for s in self.val_slope_history]
        median_abs = sorted(abs_slopes)[len(abs_slopes) // 2]
        min_slope = self.min_slope_ratio * median_abs

        # 方向一致: |EMA| > min_slope 且 sign(EMA) == sign(last)
        last_slope = self.val_slope_history[-1]

        direction_consistent = (
            abs(self.val_slope_ema) > min_slope and
            (self.val_slope_ema * last_slope > 0)  # 同號
        )

        return direction_consistent

    def _compute_epe_adaptive_gamma(self, gamma_raw: float) -> float:
        """
        改進 E: 根據 smoothed EPE 縮放 γ，並套用變化率限制

        公式: γ_scaled = 1.0 + (γ_raw - 1.0) × clamp(epe_smooth / epe_ref, 0, 1)
        保險絲 1: 用 epe_smooth (EMA) 而非即時值
        保險絲 2: |γ_eff - γ_prev| ≤ delta_gamma_max
        """
        # 如果 epe_smooth 尚未初始化，用完整 γ
        if self.epe_smooth is None or self.epe_ref <= 0:
            return gamma_raw

        # EPE-adaptive scaling
        epe_ratio = min(self.epe_smooth / self.epe_ref, 1.0)
        gamma_scaled = 1.0 + (gamma_raw - 1.0) * epe_ratio

        # 保險絲 2: 變化率限制
        gamma_clamped = max(
            self.last_gamma_eff - self.delta_gamma_max,
            min(gamma_scaled, self.last_gamma_eff + self.delta_gamma_max)
        )

        # 更新上一次的有效 γ
        self.last_gamma_eff = gamma_clamped
        return gamma_clamped

    def get_lr_multiplier(self, lr_base: float, step: int, is_backbone: bool = False) -> float:
        """
        計算當前 LR 乘數

        改進 B: 漸進式 impulse (第一次 ×2，之後 ×3)
        改進 C: 參數群分離 (pol module 全擾動，backbone 只 ×1.2)
        改進 E: EPE-adaptive impulse (EPE 低時 γ 自動縮小)

        Args:
            lr_base: OneCycle 當前的 LR (reference)
            step: 當前 step
            is_backbone: True 表示這是 backbone (fnet/cnet) 的 LR

        Returns:
            multiplier to apply to lr_base
        """
        if self.state == self.IMPULSE:
            # 記錄 impulse 開始時的 lr_base (用於 relative_impulse mode)
            if step == self.impulse_start_step:
                self.lr_at_impulse_start = lr_base

            # 改進 C: backbone 只輕微擾動 (也受 EPE-adaptive 影響)
            if is_backbone:
                return self._compute_epe_adaptive_gamma(self.gamma_backbone)

            # 改進 B: 漸進式 impulse
            if self.trigger_count == 1:
                gamma_raw = self.gamma_up_first  # 第一次用 ×2
            else:
                gamma_raw = self.gamma_up  # 之後用 ×3

            # 改進 E: EPE-adaptive + 保險絲
            return self._compute_epe_adaptive_gamma(gamma_raw)

        elif self.state == self.COOLDOWN:
            # 改進 C: backbone cooldown 也更溫和
            if is_backbone:
                return 1.0  # backbone 在 cooldown 時直接回到 reference

            if self.cooldown_mode == 'relative_impulse':
                # lr_effective = lr_at_impulse * gamma_up * gamma_down
                gamma_used = self.gamma_up_first if self.trigger_count == 1 else self.gamma_up
                lr_impulse = self.lr_at_impulse_start * gamma_used
                lr_cool = lr_impulse * self.gamma_down
                return lr_cool / lr_base if lr_base > 0 else self.gamma_down
            else:  # relative_base (default)
                return self.gamma_down

        else:  # NORMAL, PROTECTED
            return 1.0

    def get_status(self) -> dict:
        """獲取當前狀態用於 logging"""
        epsilon = 0.0
        if len(self.train_slope_history) >= 100:
            sorted_slopes = sorted(self.train_slope_history)
            idx = int(len(sorted_slopes) * self.epsilon_percentile)
            epsilon = sorted_slopes[idx]

        return {
            'state': self.state,
            'state_name': self.STATE_NAMES[self.state],
            'trigger_count': self.trigger_count,
            'val_cycle_count': self.val_cycle_count,
            'train_slope_ema': self.train_slope_ema,
            'val_slope_ema': self.val_slope_ema,
            'epsilon': epsilon,
            'val_baseline': self.val_baseline,
            'val_best': self.val_best,
            'cycles_since_best': self.cycles_since_best,
            'val_history_len': len(self.val_history),
            'epe_smooth': self.epe_smooth if self.epe_smooth is not None else 0.0,
            'last_gamma_eff': self.last_gamma_eff,
            'multiplier': self.get_lr_multiplier(1.0, 0),  # 用 1.0 來獲取純乘數
        }


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

        # 建立損失函數 (強化版: 含 Polarization-aware 權重)
        self.criterion = PIDSStereoLoss(
            gamma=args.gamma,
            max_disp=args.max_disp,
            glass_weight=args.glass_weight,
            pol_weight=getattr(args, 'pol_weight', 2.0),  # Dual-Stream 專用
            strict_glass_weight=getattr(args, 'strict_glass_weight', 1.0),  # 嚴格 mask 額外權重
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

        # Curriculum Learning 組件 (Exp #28, #29, #31)
        self.curriculum_sampler = None
        self.pol_freezer = None
        self.backbone_freezer = None  # Exp #29: Freeze Backbone First
        self.v2_pol_freezer = None  # Exp #31: V2 模組 (PolarizationAttention, GatedFusion) 凍結器
        if getattr(args, 'curriculum', False):
            self._setup_curriculum()

        # 訓練狀態
        self.start_epoch = 0
        self.global_step = 0
        self.best_glass_epe = float('inf')  # 使用 Glass EPE 作為最佳模型標準
        self.best_composite_score = float('inf')  # 複合式指標 (Glass EPE + D1_weight * D1_error)
        # Robust statistics for checkpoint selection
        self.best_glass_epe_median = float('inf')  # Primary metric (更穩定)
        self.best_glass_epe_p90 = float('inf')  # Safety gate (確保尾端不會太差)

        # Directional Impulse Descent (DID) - 狀態驅動的局部脈衝控制器
        self.did = None
        if getattr(args, 'did', False):
            self.did = DirectionalImpulseDescent(
                # Plateau detection
                epsilon_percentile=getattr(args, 'did_epsilon_pct', 0.10),
                train_ema_window=getattr(args, 'did_train_window', 300),
                val_M=getattr(args, 'did_val_m', 5),
                delta=getattr(args, 'did_delta', 0.5),
                best_stale_cycles=getattr(args, 'did_best_stale', 4),  # 改進 D
                # DID impulse
                gamma_up=getattr(args, 'did_gamma_up', 3.0),
                gamma_up_first=getattr(args, 'did_gamma_up_first', 2.0),  # 改進 B
                gamma_backbone=getattr(args, 'did_gamma_backbone', 1.2),  # 改進 C
                gamma_down=getattr(args, 'did_gamma_down', 0.1),
                t_impulse=getattr(args, 'did_t_impulse', 200),
                t_cool=getattr(args, 'did_t_cool', 100),
                cooldown_mode=getattr(args, 'did_cooldown_mode', 'relative_base'),
                # Protection
                max_triggers=getattr(args, 'did_max_triggers', 3),
                protection_cycles=getattr(args, 'did_protection', 3),
                late_lock=getattr(args, 'did_late_lock', 0.75),
                val_freq=args.val_freq,
                # 改進 E: EPE-adaptive impulse
                epe_ref=getattr(args, 'did_epe_ref', 8.0),
                epe_smooth_alpha=getattr(args, 'did_epe_smooth_alpha', 0.4),
                delta_gamma_max=getattr(args, 'did_delta_gamma_max', 0.3),
            )
            print(f"DID enabled: γ_up={args.did_gamma_up} (first: {args.did_gamma_up_first}), "
                  f"γ_backbone={args.did_gamma_backbone}, γ_down={args.did_gamma_down}, "
                  f"T_impulse={args.did_t_impulse}, T_cool={args.did_t_cool}, "
                  f"max_triggers={args.did_max_triggers}, best_stale={args.did_best_stale}, "
                  f"epe_ref={args.did_epe_ref}, Δγ_max={args.did_delta_gamma_max}")

        # 載入檢查點（如果有）
        if args.resume:
            self._load_checkpoint(args.resume)

        # 保存配置
        self._save_config()

    def _build_model(self) -> nn.Module:
        """建立模型 - 支持標準 RAFT-Stereo、Dual-Stream 或 Polarization Volume 架構"""

        if getattr(self.args, 'learnable_pol', False):
            # ========== NEW: Learnable Polarization Volume 架構 (Exp #25) ==========
            # 核心優勢: 學習偏振特徵，而不是固定公式
            print("Building Learnable Polarization Volume model...")
            model = PIDSStereoLearnablePol(
                hidden_dim=self.args.hidden_dim,
                context_dim=self.args.context_dim,
                feature_dim=self.args.feature_dim,
                corr_levels=self.args.corr_levels,
                corr_radius=self.args.corr_radius,
                pol_dim=getattr(self.args, 'pol_dim', 32),
                pol_levels=getattr(self.args, 'pol_levels', 4),
                pol_radius=getattr(self.args, 'pol_radius', 4),
                iters=self.args.iters,
                mixed_precision=self.args.mixed_precision or self.args.bf16,
            )

            # 載入預訓練權重 (部分匹配到 fnet/cnet)
            if self.args.pretrained:
                self._load_pretrained_weights(model, self.args.pretrained)
                print("  PolHead initialized randomly (new learnable layer, ~18K params)")

            # 凍結 Feature Encoder
            if self.args.freeze_fnet:
                frozen_count = 0
                for name, param in model.named_parameters():
                    if 'fnet' in name:
                        param.requires_grad = False
                        frozen_count += 1
                print(f"Frozen {frozen_count} parameters in feature encoder (fnet)")

        elif getattr(self.args, 'pol_volume', False):
            # ========== Polarization Volume 架構 ==========
            # 核心優勢: 不需要 disparity 來計算 pol_diff，Oracle = Real
            print("Building Polarization Volume model (No Oracle/Real Gap!)...")
            model = PIDSStereoPolVolume(
                hidden_dim=self.args.hidden_dim,
                context_dim=self.args.context_dim,
                feature_dim=self.args.feature_dim,
                corr_levels=self.args.corr_levels,
                corr_radius=self.args.corr_radius,
                pol_levels=getattr(self.args, 'pol_levels', 4),
                pol_radius=getattr(self.args, 'pol_radius', 4),
                iters=self.args.iters,
                mixed_precision=self.args.mixed_precision or self.args.bf16,
            )

            # 載入預訓練權重 (部分匹配到 fnet/cnet)
            if self.args.pretrained:
                self._load_pretrained_weights(model, self.args.pretrained)
                print("  UpdateBlockWithPol initialized randomly (new layer)")

            # 凍結 Feature Encoder
            if self.args.freeze_fnet:
                frozen_count = 0
                for name, param in model.named_parameters():
                    if 'fnet' in name:
                        param.requires_grad = False
                        frozen_count += 1
                print(f"Frozen {frozen_count} parameters in feature encoder (fnet)")

        elif getattr(self.args, 'pol_volume_v2', False):
            # ========== Polarization Volume V2 架構 ==========
            # 核心改進: Polarization Attention + Gated Fusion
            print("Building Polarization Volume V2 model (Attention + Gated Fusion)...")
            model = PIDSStereoPolVolumeV2(
                hidden_dim=self.args.hidden_dim,
                context_dim=self.args.context_dim,
                feature_dim=self.args.feature_dim,
                corr_levels=self.args.corr_levels,
                corr_radius=self.args.corr_radius,
                pol_levels=getattr(self.args, 'pol_levels', 4),
                pol_radius=getattr(self.args, 'pol_radius', 4),
                fused_dim=getattr(self.args, 'fused_dim', 128),
                iters=self.args.iters,
                mixed_precision=self.args.mixed_precision or self.args.bf16,
            )

            # 載入預訓練權重 (部分匹配到 fnet/cnet)
            if self.args.pretrained:
                self._load_pretrained_weights(model, self.args.pretrained)
                print("  UpdateBlockV2 initialized randomly (new layers)")
                print("  PolarizationAttention initialized randomly")
                print("  GatedFusion initialized randomly")

            # 凍結 Feature Encoder
            if self.args.freeze_fnet:
                frozen_count = 0
                for name, param in model.named_parameters():
                    if 'fnet' in name:
                        param.requires_grad = False
                        frozen_count += 1
                print(f"Frozen {frozen_count} parameters in feature encoder (fnet)")

        elif getattr(self.args, 'pol_volume_v2a', False):
            # ========== Polarization Volume V2-A 架構 ==========
            # 核心改進: Pol-Conditioned Corr Residual (Disparity-aware modulation)
            # corr_enhanced = corr + α * f(pol_corr)
            print("Building Polarization Volume V2-A model (Corr Residual)...")
            model = PIDSStereoPolVolumeV2A(
                hidden_dim=self.args.hidden_dim,
                context_dim=self.args.context_dim,
                feature_dim=self.args.feature_dim,
                corr_levels=self.args.corr_levels,
                corr_radius=self.args.corr_radius,
                pol_levels=getattr(self.args, 'pol_levels', 4),
                pol_radius=getattr(self.args, 'pol_radius', 4),
                residual_hidden_dim=getattr(self.args, 'residual_hidden_dim', 64),
                residual_init_scale=getattr(self.args, 'residual_init_scale', 0.1),
                iters=self.args.iters,
                mixed_precision=self.args.mixed_precision or self.args.bf16,
            )

            # 載入預訓練權重 (部分匹配到 fnet/cnet/update_block)
            if self.args.pretrained:
                self._load_pretrained_weights(model, self.args.pretrained)
                print("  PolCorrResidual initialized randomly (new layer)")
                print(f"  Initial residual scale: {model.get_residual_scale():.3f}")

            # 凍結 Feature Encoder
            if self.args.freeze_fnet:
                frozen_count = 0
                for name, param in model.named_parameters():
                    if 'fnet' in name:
                        param.requires_grad = False
                        frozen_count += 1
                print(f"Frozen {frozen_count} parameters in feature encoder (fnet)")

        elif getattr(self.args, 'pol_volume_v2b', False):
            # ========== Polarization Volume V2-B 架構 ==========
            # 核心改進: Iteration-Scheduled Residual
            # corr_enhanced = corr + α(iter) * f(pol_corr), where α = iter/(iters-1)
            print("Building Polarization Volume V2-B model (Scheduled Residual)...")
            model = PIDSStereoPolVolumeV2B(
                hidden_dim=self.args.hidden_dim,
                context_dim=self.args.context_dim,
                feature_dim=self.args.feature_dim,
                corr_levels=self.args.corr_levels,
                corr_radius=self.args.corr_radius,
                pol_levels=getattr(self.args, 'pol_levels', 4),
                pol_radius=getattr(self.args, 'pol_radius', 4),
                residual_hidden_dim=getattr(self.args, 'residual_hidden_dim', 64),
                residual_init_scale=getattr(self.args, 'residual_init_scale', 0.1),
                iters=self.args.iters,
                mixed_precision=self.args.mixed_precision or self.args.bf16,
            )

            # 載入預訓練權重 (部分匹配到 fnet/cnet/update_block)
            if self.args.pretrained:
                self._load_pretrained_weights(model, self.args.pretrained)
                print("  PolCorrResidual initialized randomly (new layer)")
                print(f"  Initial residual scale: {model.get_residual_scale():.3f}")

            # 凍結 Feature Encoder
            if self.args.freeze_fnet:
                frozen_count = 0
                for name, param in model.named_parameters():
                    if 'fnet' in name:
                        param.requires_grad = False
                        frozen_count += 1
                print(f"Frozen {frozen_count} parameters in feature encoder (fnet)")

        elif getattr(self.args, 'pol_volume_v2c', False):
            # ========== Polarization Volume V2-C 架構 ==========
            # 核心改進: Gradient Gating
            # gate = GatingNet(pol_corr, disp_grad.detach())
            # corr_enhanced = corr + α * gate * f(pol_corr)
            print("Building Polarization Volume V2-C model (Gradient Gating)...")
            model = PIDSStereoPolVolumeV2C(
                hidden_dim=self.args.hidden_dim,
                context_dim=self.args.context_dim,
                feature_dim=self.args.feature_dim,
                corr_levels=self.args.corr_levels,
                corr_radius=self.args.corr_radius,
                pol_levels=getattr(self.args, 'pol_levels', 4),
                pol_radius=getattr(self.args, 'pol_radius', 4),
                residual_hidden_dim=getattr(self.args, 'residual_hidden_dim', 64),
                residual_init_scale=getattr(self.args, 'residual_init_scale', 0.1),
                gating_hidden_dim=getattr(self.args, 'gating_hidden_dim', 32),
                iters=self.args.iters,
                mixed_precision=self.args.mixed_precision or self.args.bf16,
            )

            # 載入預訓練權重 (部分匹配到 fnet/cnet/update_block)
            if self.args.pretrained:
                self._load_pretrained_weights(model, self.args.pretrained)
                print("  PolCorrResidual + GradientGating initialized randomly (new layers)")
                print(f"  Initial residual scale: {model.get_residual_scale():.3f}")

            # 凍結 Feature Encoder
            if self.args.freeze_fnet:
                frozen_count = 0
                for name, param in model.named_parameters():
                    if 'fnet' in name:
                        param.requires_grad = False
                        frozen_count += 1
                print(f"Frozen {frozen_count} parameters in feature encoder (fnet)")

        elif getattr(self.args, 'pol_volume_v2d', False):
            # ========== Polarization Volume V2-D 架構 (推薦) ==========
            # 核心改進: Disparity-Aware Pol Modulation
            # pol_volume[H,W,D] → per-disparity gate[H,W,D]
            # corr_mod = corr * (1 + α * (gate - 0.5) * 2)
            #
            # 關鍵創新：pol 參與 disparity 判別，而非只是 spatial importance
            # "這個 disparity 是真實匹配還是反射假匹配？"
            print("Building Polarization Volume V2-D model (Disparity-Aware Pol Modulation)...")
            model = PIDSStereoPolVolumeV2D(
                hidden_dim=self.args.hidden_dim,
                context_dim=self.args.context_dim,
                feature_dim=self.args.feature_dim,
                corr_levels=self.args.corr_levels,
                corr_radius=self.args.corr_radius,
                iters=self.args.iters,
                mixed_precision=self.args.mixed_precision or self.args.bf16,
                pol_gate_hidden=getattr(self.args, 'pol_gate_hidden', 8),
                pol_alpha=getattr(self.args, 'pol_alpha', 0.2),
            )
            print(f"  pol_gate_hidden: {getattr(self.args, 'pol_gate_hidden', 8)}")
            print(f"  pol_alpha (initial): {getattr(self.args, 'pol_alpha', 0.2)}")

            # 載入預訓練權重（fnet/cnet/update_block 完全匹配）
            if self.args.pretrained:
                self._load_pretrained_weights(model, self.args.pretrained)
                print("  PolGate3D initialized randomly (new module)")

            # 凍結 Feature Encoder
            if self.args.freeze_fnet:
                frozen_count = 0
                for name, param in model.named_parameters():
                    if 'fnet' in name:
                        param.requires_grad = False
                        frozen_count += 1
                print(f"Frozen {frozen_count} parameters in feature encoder (fnet)")

        elif getattr(self.args, 'pol_volume_v2e', False):
            # ========== Polarization Volume V2-E 架構（最後嘗試）==========
            # 核心改進: Pre-Corr Pol Weighting
            # pol 在 correlation volume 構建時就介入（源頭介入）
            # corr_weighted = corr * pol_weight（在構建時一次性應用）
            print("Building Polarization Volume V2-E model (Pre-Corr Pol Weighting)...")
            model = PIDSStereoPolVolumeV2E(
                hidden_dim=self.args.hidden_dim,
                context_dim=self.args.context_dim,
                feature_dim=self.args.feature_dim,
                corr_levels=self.args.corr_levels,
                corr_radius=self.args.corr_radius,
                iters=self.args.iters,
                mixed_precision=self.args.mixed_precision or self.args.bf16,
                pol_weight_hidden=getattr(self.args, 'pol_weight_hidden', 8),
            )
            print(f"  pol_weight_hidden: {getattr(self.args, 'pol_weight_hidden', 8)}")

            # 載入預訓練權重（fnet/cnet/update_block 完全匹配）
            if self.args.pretrained:
                self._load_pretrained_weights(model, self.args.pretrained)
                print("  PolWeightNet initialized randomly (new module)")

            # 凍結 Feature Encoder
            if self.args.freeze_fnet:
                frozen_count = 0
                for name, param in model.named_parameters():
                    if 'fnet' in name:
                        param.requires_grad = False
                        frozen_count += 1
                print(f"Frozen {frozen_count} parameters in feature encoder (fnet)")

        elif getattr(self.args, 'pol_volume_v3', False):
            # ========== Polarization Volume V3 架構 (已證明失敗) ==========
            # 核心改進: Pol-in-Feature (Early Fusion)
            # fmap1 = fnet(concat(left, pol_diff))
            # fmap2 = fnet(concat(right, pol_diff))
            print("Building Polarization Volume V3 model (Pol-in-Feature)...")
            print("  WARNING: V3 has been proven to fail. Consider using V2-E instead.")
            model = PIDSStereoPolVolumeV3(
                hidden_dim=self.args.hidden_dim,
                context_dim=self.args.context_dim,
                feature_dim=self.args.feature_dim,
                corr_levels=self.args.corr_levels,
                corr_radius=self.args.corr_radius,
                iters=self.args.iters,
                mixed_precision=self.args.mixed_precision or self.args.bf16,
            )

            # 載入預訓練權重 (部分匹配，fnet.conv1 維度不匹配會跳過)
            if self.args.pretrained:
                self._load_pretrained_weights(model, self.args.pretrained)
                print("  fnet.conv1 initialized randomly (input_dim 3->6)")

            # 凍結 Feature Encoder（V3 不建議凍結，因為 fnet 需要學習新輸入）
            if self.args.freeze_fnet:
                print("  WARNING: freeze_fnet not recommended for V3 (fnet needs to learn 6-channel input)")
                frozen_count = 0
                for name, param in model.named_parameters():
                    if 'fnet' in name:
                        param.requires_grad = False
                        frozen_count += 1
                print(f"Frozen {frozen_count} parameters in feature encoder (fnet)")

        elif getattr(self.args, 'pol_volume_v3b', False):
            # ========== Polarization Volume V3-B 架構 ==========
            # 修復 V3: Additive Pol Fusion（保留 pretrained weights）
            print("Building Polarization Volume V3-B model (Additive Pol Fusion)...")
            model = PIDSStereoPolVolumeV3B(
                hidden_dim=self.args.hidden_dim,
                context_dim=self.args.context_dim,
                feature_dim=self.args.feature_dim,
                corr_levels=self.args.corr_levels,
                corr_radius=self.args.corr_radius,
                iters=self.args.iters,
                mixed_precision=self.args.mixed_precision or self.args.bf16,
                pol_scale=getattr(self.args, 'pol_scale', 0.1),
            )
            print(f"  pol_scale: {getattr(self.args, 'pol_scale', 0.1)}")

            # 載入預訓練權重（V3-B 保持 3ch input，pretrained weights 完全匹配）
            if self.args.pretrained:
                self._load_pretrained_weights(model, self.args.pretrained)
                print("  Pretrained weights loaded (fnet.conv1 3ch preserved)")
                print("  pol_conv1/pol_norm1 initialized randomly (side branch)")

            # 凍結 Feature Encoder（V3-B 可以凍結主幹，只訓練 pol branch）
            if self.args.freeze_fnet:
                frozen_count = 0
                pol_count = 0
                for name, param in model.named_parameters():
                    if 'fnet' in name:
                        if 'pol_' in name:
                            # 不凍結 pol side branch
                            pol_count += 1
                        else:
                            param.requires_grad = False
                            frozen_count += 1
                print(f"Frozen {frozen_count} parameters in fnet main branch")
                print(f"Kept {pol_count} parameters trainable in fnet pol branch")

        elif self.args.dual_stream:
            # ========== Dual-Stream 偏振架構 ==========
            print("Building Dual-Stream model with Polarization Encoder...")
            model = PIDSStereoDualStream(
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
                disparity_noise_std=self.args.disparity_noise_std,
            )

            # 載入預訓練權重 (部分匹配到 fnet/cnet/update_block)
            if self.args.pretrained:
                self._load_pretrained_weights(model, self.args.pretrained)
                print("  Polarization Encoder & Fusion initialized randomly (new layers)")

            # 凍結 Feature Encoder
            if self.args.freeze_fnet:
                frozen_count = 0
                for name, param in model.named_parameters():
                    if 'fnet' in name:
                        param.requires_grad = False
                        frozen_count += 1
                print(f"Frozen {frozen_count} parameters in feature encoder (fnet)")

        else:
            # ========== 標準 RAFT-Stereo ==========
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

        Dual-Stream 模式: 新層 (pol_encoder, fusion) 用更高學習率
        Gradual Unfreeze 模式 (Exp #30): 分離 backbone 和 head 參數組
        """
        # 處理 DataParallel
        model = self.model.module if hasattr(self.model, 'module') else self.model

        # Exp #30: Gradual Unfreeze - 需要分離 backbone 和 head
        if getattr(self.args, 'curriculum_strategy', 'default') == 'gradual_unfreeze':
            fnet_params = []
            cnet_params = []
            update_params = []
            pol_params = []

            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if 'fnet' in name:
                    fnet_params.append(param)
                elif 'cnet' in name:
                    cnet_params.append(param)
                elif any(kw in name for kw in ['pol', 'Pol', 'POL']):
                    pol_params.append(param)
                else:
                    # update_block 和其他
                    update_params.append(param)

            # 4 個 param groups: [fnet, cnet, update_block, pol]
            # GradualUnfreezeLRScheduler 會動態調整每個 group 的 LR
            param_groups = [
                {'params': fnet_params, 'lr': self.args.lr, 'name': 'fnet'},       # group 0
                {'params': cnet_params, 'lr': self.args.lr, 'name': 'cnet'},       # group 1
                {'params': update_params, 'lr': self.args.lr, 'name': 'update'},   # group 2
                {'params': pol_params, 'lr': self.args.lr, 'name': 'pol'},         # group 3
            ]

            print(f"Optimizer param groups (Gradual Unfreeze):")
            print(f"  [0] fnet: {len(fnet_params)} tensors")
            print(f"  [1] cnet: {len(cnet_params)} tensors")
            print(f"  [2] update_block: {len(update_params)} tensors")
            print(f"  [3] pol: {len(pol_params)} tensors")

            if self.args.optimizer == 'adamw':
                optimizer = optim.AdamW(
                    param_groups,
                    weight_decay=self.args.weight_decay,
                    eps=self.args.adam_eps,
                )
            else:
                optimizer = optim.Adam(
                    param_groups,
                    weight_decay=self.args.weight_decay,
                    eps=self.args.adam_eps,
                )

        elif self.args.dual_stream:
            # Dual-Stream: 不同參數組用不同學習率
            # 新層 (pol_encoder, fusion) 學習率 = base_lr * pol_lr_mult
            base_params = []
            new_params = []

            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if 'pol_encoder' in name or 'fusion' in name:
                    new_params.append(param)
                else:
                    base_params.append(param)

            param_groups = [
                {'params': base_params, 'lr': self.args.lr},
                {'params': new_params, 'lr': self.args.lr * self.args.pol_lr_mult},
            ]

            print(f"Optimizer param groups:")
            print(f"  Base params (RAFT-Stereo): {len(base_params)} tensors, lr={self.args.lr}")
            print(f"  New params (Pol Encoder): {len(new_params)} tensors, lr={self.args.lr * self.args.pol_lr_mult}")

            if self.args.optimizer == 'adamw':
                optimizer = optim.AdamW(
                    param_groups,
                    weight_decay=self.args.weight_decay,
                    eps=self.args.adam_eps,
                )
            else:
                optimizer = optim.Adam(
                    param_groups,
                    weight_decay=self.args.weight_decay,
                    eps=self.args.adam_eps,
                )
        else:
            # 標準模式: 所有參數同一學習率
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
        # Exp #30: Gradual Unfreeze 使用自定義 5 階段 LR Scheduler
        if getattr(self.args, 'curriculum_strategy', 'default') == 'gradual_unfreeze':
            scheduler = GradualUnfreezeLRScheduler(
                optimizer=self.optimizer,
                base_lr=self.args.lr,
                total_steps=self.args.num_steps,
                backbone_param_groups=[0, 1],  # fnet, cnet
                head_param_groups=[2, 3],       # update_block, pol
            )
            print(f"Using GradualUnfreezeLRScheduler (5-phase)")
            return scheduler

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

    def _setup_curriculum(self):
        """設置 Curriculum Learning 組件 (Exp #28: Mixed Pol/Nopol, Exp #29: Freeze Backbone, Exp #30: Gradual Unfreeze)"""
        strategy = getattr(self.args, 'curriculum_strategy', 'default')

        print("\n" + "=" * 60)
        if strategy == 'gradual_unfreeze':
            print("Setting up Curriculum Learning (Exp #30: Gradual Unfreeze)")
        elif strategy == 'freeze_backbone':
            print("Setting up Curriculum Learning (Exp #29: Freeze Backbone First)")
        else:
            print("Setting up Curriculum Learning (Exp #28: Mixed Pol/Nopol)")
        print("=" * 60)

        # 創建 CurriculumConfig 根據策略
        config = CurriculumConfig(strategy=strategy)

        # 創建 CurriculumSampler
        self.curriculum_sampler = CurriculumSampler(
            dataset=self.train_loader.dataset,
            total_steps=self.args.num_steps,
            batch_size=self.args.batch_size,
            config=config,
            seed=getattr(self.args, 'curriculum_seed', 42),
        )

        # 創建 PolModuleFreezer (用於 default 策略)
        self.pol_freezer = PolModuleFreezer(
            model=self.model,
            pol_keywords=['pol', 'Pol', 'POL'],
        )

        # 創建 BackboneFreezer (用於 freeze_backbone 策略)
        if strategy == 'freeze_backbone':
            self.backbone_freezer = BackboneFreezer(model=self.model)
            # 顯示參數統計
            stats = self.backbone_freezer.get_trainable_param_count()
            print(f"Backbone params: {stats['backbone_total']} total")
            print(f"Pol params: {stats['pol_total']} total")

        # 創建 V2PolModuleFreezer (用於 pol_volume_v2 架構)
        # 核心理念: V2 模組 (PolarizationAttention, GatedFusion) 只應從 pol 數據學習
        # 在 nopol 批次時凍結這些模組，避免混合數據集稀釋偏振特徵學習
        if getattr(self.args, 'pol_volume_v2', False):
            self.v2_pol_freezer = V2PolModuleFreezer(model=self.model)
            print(f"V2PolModuleFreezer: Freeze PolarizationAttention + GatedFusion on nopol batches")

        # 統計 pol/nopol 場景數量
        dataset = self.train_loader.dataset
        if hasattr(dataset, 'scene_naming'):
            n_pol = sum(1 for v in dataset.scene_naming.values() if v == 'pol')
            n_nopol = sum(1 for v in dataset.scene_naming.values() if v == 'nopol')
            print(f"Dataset composition: {n_pol} pol + {n_nopol} nopol")

        print("=" * 60 + "\n")

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
        # 新增: Robust statistics tracking
        self.best_glass_epe_median = checkpoint.get('best_glass_epe_median', float('inf'))
        self.best_glass_epe_p90 = checkpoint.get('best_glass_epe_p90', float('inf'))

        print(f"Resumed from epoch {self.start_epoch}, step {self.global_step}")
        print(f"Best Glass EPE: {self.best_glass_epe:.3f}, Best Composite: {self.best_composite_score:.3f}")
        if self.best_glass_epe_median < float('inf'):
            print(f"Best Median: {self.best_glass_epe_median:.3f}, Best P90: {self.best_glass_epe_p90:.3f}")

    def _get_noise_ratio(self, step: int) -> float:
        """
        Curriculum Learning: 計算當前 step 的 noise_ratio

        訓練初期 (step < warmup): noise_ratio = 0 (完全用 GT disparity)
        訓練後期 (step >= warmup): noise_ratio = max_noise_ratio

        線性增長: step 0 → warmup_steps 時，noise_ratio 從 0 → max_noise_ratio
        """
        if not self.args.dual_stream:
            return 0.0

        warmup = self.args.noise_warmup_steps
        max_ratio = self.args.max_noise_ratio

        if step >= warmup:
            return max_ratio
        else:
            # 線性增長
            return max_ratio * (step / warmup)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """訓練一個 epoch"""
        self.model.train()

        meters = {
            'loss': AverageMeter(),
            'epe': AverageMeter(),
            'd1': AverageMeter(),
            'd5': AverageMeter(),
            # Curriculum Learning: 分開追蹤 pol/nopol metrics
            'pol_epe': AverageMeter(),
            'nopol_epe': AverageMeter(),
            'pol_glass_epe': AverageMeter(),
            'nopol_glass_epe': AverageMeter(),
            'd10': AverageMeter(),
            'glass_epe': AverageMeter(),
            'glass_epe_strict': AverageMeter(),
        }

        start_time = time.time()
        accum_steps = self.args.accumulation_steps

        for batch_idx, batch in enumerate(self.train_loader):
            # 檢查是否達到最大步數
            if self.global_step >= self.args.num_steps:
                break

            # ============================================================
            # Curriculum Learning: 更新 sampler 和處理凍結策略
            # ============================================================
            batch_data_types = batch.get('data_type', ['pol'] * self.args.batch_size)

            if self.curriculum_sampler is not None:
                self.curriculum_sampler.set_step(self.global_step)
                phase = self.curriculum_sampler.current_phase

                # ========== Exp #29: Freeze Backbone First 策略 ==========
                if self.backbone_freezer is not None:
                    if phase.freeze_backbone:
                        self.backbone_freezer.freeze()
                    else:
                        self.backbone_freezer.unfreeze()

                # ========== Exp #28: Freeze Pol on Nopol 策略 ==========
                if phase.freeze_pol_on_nopol and self.pol_freezer is not None:
                    # 檢查 batch 中是否有 nopol
                    n_nopol = sum(1 for dt in batch_data_types if dt == 'nopol')
                    if n_nopol > 0:
                        self.pol_freezer.freeze()
                    else:
                        self.pol_freezer.unfreeze()
                elif self.pol_freezer is not None:
                    self.pol_freezer.unfreeze()

                # ========== Exp #31: V2 Freeze Pol Modules on Nopol 策略 ==========
                # V2 架構特有: PolarizationAttention 和 GatedFusion 只從 pol 數據學習
                if self.v2_pol_freezer is not None:
                    n_nopol = sum(1 for dt in batch_data_types if dt == 'nopol')
                    if n_nopol > 0:
                        self.v2_pol_freezer.freeze()
                    else:
                        self.v2_pol_freezer.unfreeze()

            # 移動數據到 GPU
            left = batch['left'].to(self.device)
            right = batch['right'].to(self.device)
            disp_gt = batch['disparity'].to(self.device)
            valid_mask = batch['valid_mask'].to(self.device)
            glass_mask = batch['glass_mask'].to(self.device)
            glass_mask_strict = batch.get('glass_mask_strict', glass_mask).to(self.device)

            # Curriculum Learning: 計算當前 noise_ratio
            noise_ratio = self._get_noise_ratio(self.global_step)

            # 混合精度訓練 (FP16 / BF16 / FP32)
            if self.scaler:
                # FP16 模式 (需要 GradScaler)
                with autocast('cuda', dtype=self.amp_dtype):
                    # Dual-Stream: 傳遞 GT disparity 用於對齊 pol_diff 計算
                    if self.args.dual_stream:
                        flow_preds = self.model(left, right, iters=self.args.iters, disparity_gt=disp_gt, noise_ratio=noise_ratio)
                    else:
                        flow_preds = self.model(left, right, iters=self.args.iters)
                    # flow shape: (B, 2, H, W)，只取第一個 channel 作為 disparity
                    disp_preds = [-f[:, :1] for f in flow_preds]
                    # Dual-Stream: 傳遞 pol_diff 給 Polarization-aware Loss
                    pol_diff = None
                    if self.args.dual_stream and hasattr(self.model, 'get_pol_diff'):
                        model_ref = self.model.module if hasattr(self.model, 'module') else self.model
                        pol_diff = model_ref.get_pol_diff()
                    loss, metrics = self.criterion(disp_preds, disp_gt, valid_mask, glass_mask, pol_diff, glass_mask_strict)
                    loss = loss / accum_steps

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                    self.global_step += 1

                    if self.scheduler:
                        # GradualUnfreezeLRScheduler 需要 step 參數
                        if isinstance(self.scheduler, GradualUnfreezeLRScheduler):
                            self.scheduler.step(self.global_step)
                        else:
                            self.scheduler.step()

            elif self.amp_dtype is not None:
                # BF16 模式 (不需要 GradScaler，數值範圍同 FP32)
                with autocast('cuda', dtype=self.amp_dtype):
                    # Dual-Stream: 傳遞 GT disparity 用於對齊 pol_diff 計算
                    if self.args.dual_stream:
                        flow_preds = self.model(left, right, iters=self.args.iters, disparity_gt=disp_gt, noise_ratio=noise_ratio)
                    else:
                        flow_preds = self.model(left, right, iters=self.args.iters)
                    # flow shape: (B, 2, H, W)，只取第一個 channel 作為 disparity
                    disp_preds = [-f[:, :1] for f in flow_preds]
                    # Dual-Stream: 傳遞 pol_diff 給 Polarization-aware Loss
                    pol_diff = None
                    if self.args.dual_stream and hasattr(self.model, 'get_pol_diff'):
                        model_ref = self.model.module if hasattr(self.model, 'module') else self.model
                        pol_diff = model_ref.get_pol_diff()
                    loss, metrics = self.criterion(disp_preds, disp_gt, valid_mask, glass_mask, pol_diff, glass_mask_strict)
                    loss = loss / accum_steps

                loss.backward()

                if (batch_idx + 1) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1

                    if self.scheduler:
                        # GradualUnfreezeLRScheduler 需要 step 參數
                        if isinstance(self.scheduler, GradualUnfreezeLRScheduler):
                            self.scheduler.step(self.global_step)
                        else:
                            self.scheduler.step()

            else:
                # FP32 模式 (全精度)
                # Dual-Stream: 傳遞 GT disparity 用於對齊 pol_diff 計算
                if self.args.dual_stream:
                    flow_preds = self.model(left, right, iters=self.args.iters, disparity_gt=disp_gt, noise_ratio=noise_ratio)
                else:
                    flow_preds = self.model(left, right, iters=self.args.iters)
                # flow shape: (B, 2, H, W)，只取第一個 channel 作為 disparity
                disp_preds = [-f[:, :1] for f in flow_preds]
                # Dual-Stream: 傳遞 pol_diff 給 Polarization-aware Loss
                pol_diff = None
                if self.args.dual_stream and hasattr(self.model, 'get_pol_diff'):
                    model_ref = self.model.module if hasattr(self.model, 'module') else self.model
                    pol_diff = model_ref.get_pol_diff()
                loss, metrics = self.criterion(disp_preds, disp_gt, valid_mask, glass_mask, pol_diff, glass_mask_strict)

                # NEW: Learnable Pol - Glass-aware auxiliary loss
                glass_loss = torch.tensor(0.0, device=self.device)
                if getattr(self.args, 'learnable_pol', False) and getattr(self.args, 'glass_aware_weight', 0) > 0:
                    model_ref = self.model.module if hasattr(self.model, 'module') else self.model
                    if hasattr(model_ref, 'compute_glass_aware_loss'):
                        glass_loss = model_ref.compute_glass_aware_loss(glass_mask, valid_mask)
                        loss = loss + self.args.glass_aware_weight * glass_loss
                        metrics['glass_loss'] = glass_loss.item()

                loss = loss / accum_steps

                loss.backward()

                if (batch_idx + 1) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1

                    if self.scheduler:
                        # GradualUnfreezeLRScheduler 需要 step 參數
                        if isinstance(self.scheduler, GradualUnfreezeLRScheduler):
                            self.scheduler.step(self.global_step)
                        else:
                            self.scheduler.step()

                    # DID: 在 scheduler 設定 LR 之後，應用 DID multiplier
                    # 改進 C: 參數群分離 - backbone (fnet/cnet) 只用 γ_backbone，pol/update 用完整 impulse
                    if self.did is not None:
                        for param_group in self.optimizer.param_groups:
                            group_name = param_group.get('name', '')
                            is_backbone = group_name in ['fnet', 'cnet']
                            lr_base = param_group['lr']  # 使用該 group 自己的 base LR
                            lr_multiplier = self.did.get_lr_multiplier(lr_base, self.global_step, is_backbone)
                            if abs(lr_multiplier - 1.0) > 1e-6:
                                # 注意: scheduler 已經設定了 lr，我們要乘上 multiplier
                                param_group['lr'] = lr_base * lr_multiplier

            # 更新統計
            batch_size = left.size(0)
            meters['loss'].update(metrics['loss'], batch_size)
            meters['epe'].update(metrics['epe'], batch_size)
            meters['d1'].update(metrics['d1'], batch_size)
            meters['d5'].update(metrics.get('d5', 0), batch_size)
            meters['d10'].update(metrics.get('d10', 0), batch_size)
            meters['glass_epe'].update(metrics['glass_epe'], batch_size)
            meters['glass_epe_strict'].update(metrics.get('glass_epe_strict', metrics['glass_epe']), batch_size)

            # DID: 更新 train slope 追蹤
            if self.did is not None:
                self.did.update_train(metrics['loss'], self.global_step)

            # Curriculum Learning: 分開追蹤 pol/nopol metrics
            if self.curriculum_sampler is not None:
                n_pol = sum(1 for dt in batch_data_types if dt == 'pol')
                n_nopol = batch_size - n_pol
                if n_pol > 0:
                    meters['pol_epe'].update(metrics['epe'], n_pol)
                    meters['pol_glass_epe'].update(metrics['glass_epe'], n_pol)
                if n_nopol > 0:
                    meters['nopol_epe'].update(metrics['epe'], n_nopol)
                    meters['nopol_glass_epe'].update(metrics['glass_epe'], n_nopol)

            # TensorBoard (每個 global step)
            if (batch_idx + 1) % accum_steps == 0 and self.global_step % self.args.log_freq == 0:
                self.writer.add_scalar('train/loss', metrics['loss'], self.global_step)
                self.writer.add_scalar('train/epe', metrics['epe'], self.global_step)
                self.writer.add_scalar('train/d1', metrics['d1'], self.global_step)
                self.writer.add_scalar('train/d5', metrics.get('d5', 0), self.global_step)
                self.writer.add_scalar('train/d10', metrics.get('d10', 0), self.global_step)
                self.writer.add_scalar('train/glass_epe', metrics['glass_epe'], self.global_step)
                self.writer.add_scalar('train/glass_epe_strict', metrics.get('glass_epe_strict', metrics['glass_epe']), self.global_step)

                # Exp #30: Gradual Unfreeze - 記錄 backbone/head LR
                if isinstance(self.scheduler, GradualUnfreezeLRScheduler):
                    lr_info = self.scheduler.get_last_lr()
                    self.writer.add_scalar('train/backbone_lr', lr_info.get('backbone_lr', 0), self.global_step)
                    self.writer.add_scalar('train/head_lr', lr_info.get('head_lr', 0), self.global_step)
                    self.writer.add_scalar('train/backbone_mult', lr_info.get('backbone_mult', 0), self.global_step)
                    self.writer.add_scalar('train/lr_mult', lr_info.get('lr_mult', 1), self.global_step)
                else:
                    lr = self.optimizer.param_groups[0]['lr']
                    self.writer.add_scalar('train/lr', lr, self.global_step)

                # Curriculum Learning: 記錄當前 noise_ratio
                if self.args.dual_stream:
                    self.writer.add_scalar('train/noise_ratio', noise_ratio, self.global_step)

                # V2-A/V2-B/V2-C: 記錄 residual scale (監控學習進度)
                if getattr(self.args, 'pol_volume_v2a', False) or getattr(self.args, 'pol_volume_v2b', False) or getattr(self.args, 'pol_volume_v2c', False):
                    model_ref = self.model.module if hasattr(self.model, 'module') else self.model
                    if hasattr(model_ref, 'get_residual_scale'):
                        self.writer.add_scalar('train/residual_scale', model_ref.get_residual_scale(), self.global_step)

                # Curriculum Learning (Exp #28, #29): 記錄 pol/nopol metrics 和凍結狀態
                if self.curriculum_sampler is not None:
                    phase = self.curriculum_sampler.current_phase
                    self.writer.add_scalar('curriculum/progress', self.curriculum_sampler.progress, self.global_step)
                    self.writer.add_scalar('curriculum/pol_ratio', phase.pol_ratio, self.global_step)
                    if meters['pol_epe'].count > 0:
                        self.writer.add_scalar('train/pol_epe', meters['pol_epe'].avg, self.global_step)
                        self.writer.add_scalar('train/pol_glass_epe', meters['pol_glass_epe'].avg, self.global_step)
                    if meters['nopol_epe'].count > 0:
                        self.writer.add_scalar('train/nopol_epe', meters['nopol_epe'].avg, self.global_step)
                        self.writer.add_scalar('train/nopol_glass_epe', meters['nopol_glass_epe'].avg, self.global_step)
                    # Pol Module 凍結狀態
                    if self.pol_freezer is not None:
                        self.writer.add_scalar('curriculum/pol_frozen', 1 if self.pol_freezer.is_frozen else 0, self.global_step)
                    # Backbone 凍結狀態 (Exp #29)
                    if self.backbone_freezer is not None:
                        self.writer.add_scalar('curriculum/backbone_frozen', 1 if self.backbone_freezer.is_frozen else 0, self.global_step)
                    # V2 Pol Module 凍結狀態 (Exp #31)
                    if self.v2_pol_freezer is not None:
                        self.writer.add_scalar('curriculum/v2_pol_frozen', 1 if self.v2_pol_freezer._frozen else 0, self.global_step)

            # 打印進度
            if (batch_idx + 1) % accum_steps == 0 and self.global_step % self.args.print_freq == 0:
                elapsed = time.time() - start_time
                eff_batch = self.args.batch_size * accum_steps

                # Exp #30: Gradual Unfreeze - 顯示 backbone 和 head LR
                if isinstance(self.scheduler, GradualUnfreezeLRScheduler):
                    lr_info = self.scheduler.get_last_lr()
                    bb_lr = lr_info.get('backbone_lr', 0)
                    head_lr = lr_info.get('head_lr', 0)
                    phase_name = self.scheduler.get_phase_name()
                    bb_mult = lr_info.get('backbone_mult', 0)

                    msg = (f"  Step {self.global_step:6d} | "
                           f"Loss: {meters['loss'].avg:.4f} | "
                           f"EPE: {meters['epe'].avg:.3f} | "
                           f"D1: {meters['d1'].avg:.2f}% | "
                           f"Glass EPE: {meters['glass_epe'].avg:.3f} (S:{meters['glass_epe_strict'].avg:.3f}) | "
                           f"BB_LR: {bb_lr:.6f} | Head_LR: {head_lr:.6f} | "
                           f"BB_mult: {bb_mult:.2f} | Phase: {phase_name}")
                else:
                    lr = self.optimizer.param_groups[0]['lr']
                    # 基本訊息
                    msg = (f"  Step {self.global_step:6d} | "
                           f"Loss: {meters['loss'].avg:.4f} | "
                           f"EPE: {meters['epe'].avg:.3f} | "
                           f"D1: {meters['d1'].avg:.2f}% | "
                           f"Glass EPE: {meters['glass_epe'].avg:.3f} (S:{meters['glass_epe_strict'].avg:.3f}) | "
                           f"LR: {lr:.6f}")
                    # Dual-Stream: 顯示 noise_ratio
                    if self.args.dual_stream:
                        msg += f" | NR: {noise_ratio:.2f}"
                    # Curriculum Learning: 顯示 phase 和凍結狀態
                    if self.curriculum_sampler is not None:
                        phase = self.curriculum_sampler.current_phase
                        frozen_parts = []
                        if self.backbone_freezer and self.backbone_freezer.is_frozen:
                            frozen_parts.append("BB")  # Backbone
                        if self.pol_freezer and self.pol_freezer.is_frozen:
                            frozen_parts.append("POL")
                        if self.v2_pol_freezer and self.v2_pol_freezer._frozen:
                            frozen_parts.append("V2")  # V2 Pol Modules
                        frozen_str = f" [FROZEN: {'+'.join(frozen_parts)}]" if frozen_parts else ""
                        msg += f" | Phase: {phase.name[:15]}... | Pol: {phase.pol_ratio:.0%}{frozen_str}"

                msg += f" | Time: {elapsed:.1f}s"
                print(msg)

            # 按 step 驗證 (更密集以抓 best checkpoint)
            # 同時進行 Oracle (理論上限) 和 Real (實戰能力) 驗證
            if (batch_idx + 1) % accum_steps == 0 and self.global_step % self.args.val_freq == 0 and self.global_step > 0:
                self._validate_and_log()  # 新版：同時驗證 Oracle 和 Real
                self.model.train()  # 切回訓練模式

        return {k: v.avg for k, v in meters.items()}

    @torch.no_grad()
    def validate(self, use_oracle: bool = True) -> Dict[str, float]:
        """
        驗證模型性能

        Args:
            use_oracle: True = 使用 GT disparity 對齊偏振 (Oracle Testing, 理論上限)
                        False = 使用 forward_inference (真實推論能力)

        Returns:
            metrics dict (含 robust statistics: median, trimmed_mean, p90, p95, outlier_rate)
        """
        self.model.eval()

        meters = {
            'loss': AverageMeter(),
            'epe': AverageMeter(),
            'd1': AverageMeter(),
            'd3': AverageMeter(),
            'd5': AverageMeter(),
            'd10': AverageMeter(),
            'glass_epe': AverageMeter(),
            'glass_epe_strict': AverageMeter(),
        }

        # 收集每個樣本的 glass_epe 用於 robust statistics
        all_glass_epes = []
        all_glass_epes_strict = []

        # 獲取模型引用 (處理 DataParallel)
        model_ref = self.model.module if hasattr(self.model, 'module') else self.model

        for batch in self.val_loader:
            left = batch['left'].to(self.device)
            right = batch['right'].to(self.device)
            disp_gt = batch['disparity'].to(self.device)
            valid_mask = batch['valid_mask'].to(self.device)
            # 驗證時兩種 mask 都傳入，讓 criterion 同時計算 union 和 strict 的 glass_epe
            glass_mask = batch['glass_mask'].to(self.device)
            glass_mask_strict = batch.get('glass_mask_strict', glass_mask).to(self.device)

            # 根據模式選擇推論方式
            if use_oracle:
                # ========== Oracle Testing (理論上限) ==========
                # 使用 GT disparity 對齊偏振特徵，測試偏振模組的理論貢獻
                if self.args.dual_stream:
                    flow_preds = self.model(left, right, iters=self.args.iters + 4, disparity_gt=disp_gt)
                else:
                    flow_preds = self.model(left, right, iters=self.args.iters + 4)
                # 只用最終預測計算 loss (和 Real mode 一致，否則 sequence loss 不可比)
                disp_preds = [-flow_preds[-1][:, :1]]
            else:
                # ========== Real Inference (實戰能力) ==========
                # 不使用 GT disparity，模擬真實部署情況
                if self.args.dual_stream and hasattr(model_ref, 'forward_inference'):
                    # 使用 forward_inference: 迭代中途用預測 disparity 更新偏振特徵
                    val_iters = self.args.iters + 4
                    flow_pred = model_ref.forward_inference(
                        left, right,
                        iters=val_iters,
                        pol_update_iters=[val_iters // 2]  # 中間更新一次偏振特徵
                    )
                    # forward_inference 返回單個 tensor，包成 list
                    disp_preds = [-flow_pred[:, :1]]
                else:
                    # 非 Dual-Stream 或沒有 forward_inference，用標準 forward
                    flow_preds = self.model(left, right, iters=self.args.iters + 4)
                    disp_preds = [-f[:, :1] for f in flow_preds]

            # 獲取 pol_diff (用於 loss 計算)
            pol_diff = None
            if self.args.dual_stream and hasattr(model_ref, 'get_pol_diff'):
                pol_diff = model_ref.get_pol_diff()

            # 驗證時傳入兩種 mask，criterion 同時計算 glass_epe (union) 和 glass_epe_strict
            loss, metrics = self.criterion(disp_preds, disp_gt, valid_mask, glass_mask, pol_diff, glass_mask_strict)

            batch_size = left.size(0)
            meters['loss'].update(metrics['loss'], batch_size)
            meters['epe'].update(metrics['epe'], batch_size)
            meters['d1'].update(metrics['d1'], batch_size)
            meters['d3'].update(metrics['d3'], batch_size)
            meters['d5'].update(metrics.get('d5', 0), batch_size)
            meters['d10'].update(metrics.get('d10', 0), batch_size)
            meters['glass_epe'].update(metrics['glass_epe'], batch_size)
            meters['glass_epe_strict'].update(metrics.get('glass_epe_strict', metrics['glass_epe']), batch_size)

            # 收集每個樣本的 glass_epe (兩種 mask)
            all_glass_epes.append(metrics['glass_epe'])
            all_glass_epes_strict.append(metrics.get('glass_epe_strict', metrics['glass_epe']))

        # 基本統計
        result = {k: v.avg for k, v in meters.items()}

        # ========== Robust Statistics for Glass EPE ==========
        if len(all_glass_epes) > 0:
            glass_epes = np.array(all_glass_epes)
            sorted_epes = np.sort(glass_epes)
            n = len(glass_epes)

            # Trimmed Mean: 移除 top 10%
            trim_idx = int(n * 0.9)
            trimmed_values = sorted_epes[:trim_idx] if trim_idx > 0 else sorted_epes

            # Outlier rate: > 20px
            outlier_threshold = 20.0
            outlier_count = np.sum(glass_epes > outlier_threshold)

            result['glass_epe_median'] = float(np.median(glass_epes))
            result['glass_epe_trimmed_mean'] = float(np.mean(trimmed_values))
            result['glass_epe_p90'] = float(np.percentile(glass_epes, 90))
            result['glass_epe_p95'] = float(np.percentile(glass_epes, 95))
            result['glass_epe_outlier_rate'] = float(outlier_count / n)
            result['glass_epe_outlier_count'] = int(outlier_count)

        # ========== Robust Statistics for Glass EPE Strict ==========
        if len(all_glass_epes_strict) > 0:
            glass_epes_s = np.array(all_glass_epes_strict)
            sorted_epes_s = np.sort(glass_epes_s)
            n_s = len(glass_epes_s)

            trim_idx_s = int(n_s * 0.9)
            trimmed_values_s = sorted_epes_s[:trim_idx_s] if trim_idx_s > 0 else sorted_epes_s

            outlier_count_s = np.sum(glass_epes_s > 20.0)

            result['glass_epe_strict_median'] = float(np.median(glass_epes_s))
            result['glass_epe_strict_trimmed_mean'] = float(np.mean(trimmed_values_s))
            result['glass_epe_strict_p90'] = float(np.percentile(glass_epes_s, 90))
            result['glass_epe_strict_p95'] = float(np.percentile(glass_epes_s, 95))
            result['glass_epe_strict_outlier_rate'] = float(outlier_count_s / n_s)
            result['glass_epe_strict_outlier_count'] = int(outlier_count_s)

        return result

    def _log_validation(self, val_metrics_oracle: Dict[str, float]):
        """
        記錄驗證結果並保存 checkpoint

        這是舊版本的接口，為了兼容性保留。
        新的驗證流程使用 _validate_and_log()
        """
        self._validate_and_log_internal(val_metrics_oracle, val_metrics_real=None)

    def _validate_and_log(self):
        """
        完整驗證流程：同時進行 Oracle 和 Real 驗證

        Oracle Testing: 使用 GT disparity 對齊，測試偏振模組的理論上限
        Real Inference: 不使用 GT，模擬真實部署情況

        用 Real 分數決定 best checkpoint (實戰能力才是真正標準)
        """
        # 1. Oracle 驗證 (理論上限)
        print("  [Validating Oracle mode...]")
        val_metrics_oracle = self.validate(use_oracle=True)

        # 2. Real 驗證 (實戰能力)
        # - Dual-Stream 模式: Oracle ≠ Real (依賴 GT disparity 計算 pol_diff)
        # - Pol Volume 模式: Oracle = Real (不依賴 disparity！這是新架構的核心優勢)
        # - Pol Volume V2/V2-A/V2-B/V2-C/V2-D/V2-E/V3/V3-B 模式: Oracle = Real (同 V1)
        # - 標準模式: Oracle = Real (無偏振)
        is_pol_volume = (getattr(self.args, 'pol_volume', False) or
                         getattr(self.args, 'pol_volume_v2', False) or
                         getattr(self.args, 'pol_volume_v2a', False) or
                         getattr(self.args, 'pol_volume_v2b', False) or
                         getattr(self.args, 'pol_volume_v2c', False) or
                         getattr(self.args, 'pol_volume_v2d', False) or
                         getattr(self.args, 'pol_volume_v2e', False) or
                         getattr(self.args, 'pol_volume_v3', False) or
                         getattr(self.args, 'pol_volume_v3b', False))
        if self.args.dual_stream and not is_pol_volume:
            print("  [Validating Real mode...]")
            val_metrics_real = self.validate(use_oracle=False)
        else:
            # Pol Volume 或標準模式: Oracle 和 Real 完全相同
            if is_pol_volume:
                print("  [Pol Volume mode: Oracle = Real, no gap!]")
            val_metrics_real = val_metrics_oracle

        # 3. 記錄和保存
        self._validate_and_log_internal(val_metrics_oracle, val_metrics_real)

    def _validate_and_log_internal(
        self,
        val_metrics_oracle: Dict[str, float],
        val_metrics_real: Optional[Dict[str, float]] = None
    ):
        """
        內部方法：記錄驗證結果並保存 checkpoint

        Args:
            val_metrics_oracle: Oracle 模式的驗證結果
            val_metrics_real: Real 模式的驗證結果 (None 時使用 oracle 結果)
        """
        # 如果沒有 real 結果，使用 oracle 結果 (兼容舊版本)
        if val_metrics_real is None:
            val_metrics_real = val_metrics_oracle

        # ========== 計算複合式指標 ==========
        # 使用 Real 分數計算 (實戰能力)
        # 注意: 使用 glass_epe_strict (核心玻璃區域) 作為 composite 指標
        d1_error_real = val_metrics_real['d1']
        glass_epe_strict_real = val_metrics_real.get('glass_epe_strict', val_metrics_real['glass_epe'])
        composite_score_real = glass_epe_strict_real + self.args.d1_weight * d1_error_real

        # Oracle 分數也計算 (用於對比)
        d1_error_oracle = val_metrics_oracle['d1']
        glass_epe_strict_oracle = val_metrics_oracle.get('glass_epe_strict', val_metrics_oracle['glass_epe'])
        composite_score_oracle = glass_epe_strict_oracle + self.args.d1_weight * d1_error_oracle

        # ========== 打印結果 ==========
        print(f"\n  [Val @ Step {self.global_step}]")
        print(f"  ┌──────────────────────────────────────────────────────────────────────────────────────┐")
        print(f"  │ Mode       │ Glass EPE (U) │ Glass EPE (S) │ EPE    │ D1     │ Composite │ Loss     │")
        print(f"  ├──────────────────────────────────────────────────────────────────────────────────────┤")
        print(f"  │ Oracle     │ {val_metrics_oracle['glass_epe']:13.3f} │ {glass_epe_strict_oracle:13.3f} │ {val_metrics_oracle['epe']:6.3f} │ "
              f"{val_metrics_oracle['d1']:5.2f}% │ {composite_score_oracle:9.3f} │ {val_metrics_oracle['loss']:8.4f} │")
        print(f"  │ Real       │ {val_metrics_real['glass_epe']:13.3f} │ {glass_epe_strict_real:13.3f} │ {val_metrics_real['epe']:6.3f} │ "
              f"{val_metrics_real['d1']:5.2f}% │ {composite_score_real:9.3f} │ {val_metrics_real['loss']:8.4f} │")
        print(f"  └──────────────────────────────────────────────────────────────────────────────────────┘")
        print(f"  (U) = Union mask (邊緣+核心), (S) = Strict mask (核心), Composite 使用 Strict")

        # 計算 Oracle vs Real 差距 (使用 strict)
        gap = glass_epe_strict_real - glass_epe_strict_oracle
        gap_pct = (gap / glass_epe_strict_oracle * 100) if glass_epe_strict_oracle > 0 else 0
        print(f"  Oracle-Real Gap: {gap:+.3f} px ({gap_pct:+.1f}%)")

        # Union vs Strict 差距 (監控兩種 mask 的差異)
        us_gap = val_metrics_real['glass_epe'] - glass_epe_strict_real
        print(f"  Union-Strict Gap: {us_gap:+.3f} px")

        # ========== Robust Statistics (Real Mode) ==========
        if 'glass_epe_median' in val_metrics_real:
            print(f"\n  [Glass EPE Robust Statistics (Real) — Union Mask]")
            print(f"  ┌───────────────────────────────────────────────────┐")
            print(f"  │ Mean:         {val_metrics_real['glass_epe']:8.3f} px                      │")
            print(f"  │ Median:       {val_metrics_real['glass_epe_median']:8.3f} px                      │")
            print(f"  │ Trimmed Mean: {val_metrics_real['glass_epe_trimmed_mean']:8.3f} px  (top 10% removed) │")
            print(f"  │ P90:          {val_metrics_real['glass_epe_p90']:8.3f} px                      │")
            print(f"  │ P95:          {val_metrics_real['glass_epe_p95']:8.3f} px                      │")
            print(f"  │ Outlier Rate: {val_metrics_real['glass_epe_outlier_rate']*100:7.1f}%  ({val_metrics_real['glass_epe_outlier_count']} scenes > 20px) │")
            print(f"  └───────────────────────────────────────────────────┘")
        if 'glass_epe_strict_median' in val_metrics_real:
            print(f"\n  [Glass EPE Robust Statistics (Real) — Strict Mask]")
            print(f"  ┌───────────────────────────────────────────────────┐")
            print(f"  │ Mean:         {glass_epe_strict_real:8.3f} px                      │")
            print(f"  │ Median:       {val_metrics_real['glass_epe_strict_median']:8.3f} px                      │")
            print(f"  │ Trimmed Mean: {val_metrics_real['glass_epe_strict_trimmed_mean']:8.3f} px  (top 10% removed) │")
            print(f"  │ P90:          {val_metrics_real['glass_epe_strict_p90']:8.3f} px                      │")
            print(f"  │ P95:          {val_metrics_real['glass_epe_strict_p95']:8.3f} px                      │")
            print(f"  │ Outlier Rate: {val_metrics_real['glass_epe_strict_outlier_rate']*100:7.1f}%  ({val_metrics_real['glass_epe_strict_outlier_count']} scenes > 20px) │")
            print(f"  └───────────────────────────────────────────────────┘")
        print()

        # ========== TensorBoard 記錄 ==========
        # Oracle 指標
        self.writer.add_scalar('val_oracle/loss', val_metrics_oracle['loss'], self.global_step)
        self.writer.add_scalar('val_oracle/epe', val_metrics_oracle['epe'], self.global_step)
        self.writer.add_scalar('val_oracle/d1', val_metrics_oracle['d1'], self.global_step)
        self.writer.add_scalar('val_oracle/glass_epe', val_metrics_oracle['glass_epe'], self.global_step)
        self.writer.add_scalar('val_oracle/glass_epe_strict', val_metrics_oracle.get('glass_epe_strict', val_metrics_oracle['glass_epe']), self.global_step)
        self.writer.add_scalar('val_oracle/composite_score', composite_score_oracle, self.global_step)

        # Real 指標 (實戰能力)
        self.writer.add_scalar('val_real/loss', val_metrics_real['loss'], self.global_step)
        self.writer.add_scalar('val_real/epe', val_metrics_real['epe'], self.global_step)
        self.writer.add_scalar('val_real/d1', val_metrics_real['d1'], self.global_step)
        self.writer.add_scalar('val_real/glass_epe', val_metrics_real['glass_epe'], self.global_step)
        self.writer.add_scalar('val_real/glass_epe_strict', glass_epe_strict_real, self.global_step)
        self.writer.add_scalar('val_real/composite_score', composite_score_real, self.global_step)

        # Oracle-Real Gap
        self.writer.add_scalar('val/oracle_real_gap', gap, self.global_step)
        self.writer.add_scalar('val/oracle_real_gap_pct', gap_pct, self.global_step)

        # 兼容舊版本的 val/ 前綴 (使用 Real 分數)
        self.writer.add_scalar('val/loss', val_metrics_real['loss'], self.global_step)
        self.writer.add_scalar('val/epe', val_metrics_real['epe'], self.global_step)
        self.writer.add_scalar('val/d1', val_metrics_real['d1'], self.global_step)
        self.writer.add_scalar('val/d3', val_metrics_real['d3'], self.global_step)
        self.writer.add_scalar('val/d5', val_metrics_real.get('d5', 0), self.global_step)
        self.writer.add_scalar('val/d10', val_metrics_real.get('d10', 0), self.global_step)
        self.writer.add_scalar('val/glass_epe', val_metrics_real['glass_epe'], self.global_step)
        self.writer.add_scalar('val/glass_epe_strict', glass_epe_strict_real, self.global_step)
        self.writer.add_scalar('val/composite_score', composite_score_real, self.global_step)

        # Robust Statistics — Union Mask (Real)
        if 'glass_epe_median' in val_metrics_real:
            self.writer.add_scalar('val/glass_epe_median', val_metrics_real['glass_epe_median'], self.global_step)
            self.writer.add_scalar('val/glass_epe_trimmed_mean', val_metrics_real['glass_epe_trimmed_mean'], self.global_step)
            self.writer.add_scalar('val/glass_epe_p90', val_metrics_real['glass_epe_p90'], self.global_step)
            self.writer.add_scalar('val/glass_epe_p95', val_metrics_real['glass_epe_p95'], self.global_step)
            self.writer.add_scalar('val/glass_epe_outlier_rate', val_metrics_real['glass_epe_outlier_rate'], self.global_step)

        # Robust Statistics — Strict Mask (Real)
        if 'glass_epe_strict_median' in val_metrics_real:
            self.writer.add_scalar('val/glass_epe_strict_median', val_metrics_real['glass_epe_strict_median'], self.global_step)
            self.writer.add_scalar('val/glass_epe_strict_trimmed_mean', val_metrics_real['glass_epe_strict_trimmed_mean'], self.global_step)
            self.writer.add_scalar('val/glass_epe_strict_p90', val_metrics_real['glass_epe_strict_p90'], self.global_step)
            self.writer.add_scalar('val/glass_epe_strict_p95', val_metrics_real['glass_epe_strict_p95'], self.global_step)
            self.writer.add_scalar('val/glass_epe_strict_outlier_rate', val_metrics_real['glass_epe_strict_outlier_rate'], self.global_step)

        # ========== Directional Impulse Descent (DID) ==========
        if self.did is not None:
            # 更新 DID 狀態並檢查是否觸發 (使用 strict glass_epe)
            did_result = self.did.update_val(
                val_glass_epe=glass_epe_strict_real,
                step=self.global_step,
                total_steps=self.args.num_steps,
            )

            # 獲取狀態用於 logging
            did_status = self.did.get_status()

            # 取得當前 OneCycle LR (reference)
            # 注意: DID multiplier 會在 train_epoch 的 scheduler.step() 之後應用
            lr_base = self.optimizer.param_groups[0]['lr']
            lr_multiplier = self.did.get_lr_multiplier(lr_base, self.global_step)
            lr_effective = lr_base * lr_multiplier

            # 打印 DID 狀態
            state_str = did_result['state']
            if did_result['triggered']:
                print(f"  [DID] {did_result['message']}")
            elif did_result['message']:
                print(f"  [DID] State: {state_str} | {did_result['message']}")
            else:
                print(f"  [DID] State: {state_str} | ε={did_status['epsilon']:.4f}, "
                      f"train_slope={did_status['train_slope_ema']:.4f}, "
                      f"triggers={did_status['trigger_count']}/{self.args.did_max_triggers}")

            if abs(lr_multiplier - 1.0) > 1e-6:
                print(f"  [DID] LR: {lr_base:.6f} × {lr_multiplier:.1f} = {lr_effective:.6f}")

            # TensorBoard logging
            self.writer.add_scalar('did/state', did_status['state'], self.global_step)
            self.writer.add_scalar('did/trigger_count', did_status['trigger_count'], self.global_step)
            self.writer.add_scalar('did/train_slope_ema', did_status['train_slope_ema'], self.global_step)
            self.writer.add_scalar('did/val_slope_ema', did_status['val_slope_ema'], self.global_step)
            self.writer.add_scalar('did/epsilon', did_status['epsilon'], self.global_step)
            self.writer.add_scalar('did/lr_base', lr_base, self.global_step)
            self.writer.add_scalar('did/lr_effective', lr_effective, self.global_step)
            self.writer.add_scalar('did/lr_multiplier', lr_multiplier, self.global_step)
            if did_status['val_baseline'] is not None:
                self.writer.add_scalar('did/val_baseline', did_status['val_baseline'], self.global_step)
            self.writer.add_scalar('did/epe_smooth', did_status['epe_smooth'], self.global_step)
            self.writer.add_scalar('did/gamma_eff', did_status['last_gamma_eff'], self.global_step)

        # ========== 檢查是否為最佳 (使用 Strict Mask 的 Robust Statistics) ==========
        # Primary metric: Strict Median (核心玻璃區域，更嚴格的標準)
        # Safety gate: Strict P90 (確保尾端不會太差)
        # 注意: 從本次修改起，checkpoint 選擇統一使用 strict mask 指標
        current_median = val_metrics_real.get('glass_epe_strict_median',
                         val_metrics_real.get('glass_epe_median', glass_epe_strict_real))
        current_p90 = val_metrics_real.get('glass_epe_strict_p90',
                      val_metrics_real.get('glass_epe_p90', glass_epe_strict_real))

        # 初始化 best tracking (如果尚未初始化)
        if not hasattr(self, 'best_glass_epe_median'):
            self.best_glass_epe_median = float('inf')
        if not hasattr(self, 'best_glass_epe_p90'):
            self.best_glass_epe_p90 = float('inf')

        # P90 Safety Gate: P90 不能超過 best_p90 的 1.5 倍
        p90_threshold = self.best_glass_epe_p90 * 1.5 if self.best_glass_epe_p90 < float('inf') else float('inf')
        p90_safe = current_p90 <= p90_threshold or self.best_glass_epe_p90 == float('inf')

        # Best 判斷: Strict Median 更低 且 Strict P90 安全
        is_best = (current_median < self.best_glass_epe_median) and p90_safe

        if is_best:
            self.best_glass_epe_median = current_median
            self.best_glass_epe_p90 = current_p90
            self.best_composite_score = composite_score_real
            # 同時更新 best_glass_epe (用於兼容舊 checkpoint)
            if glass_epe_strict_real < self.best_glass_epe:
                self.best_glass_epe = glass_epe_strict_real
            print(f"  *** New best! Strict Median: {current_median:.3f} px, Strict P90: {current_p90:.3f} px ***")
            print(f"      (Union Mean: {val_metrics_real['glass_epe']:.3f}, Strict Mean: {glass_epe_strict_real:.3f}, D1: {d1_error_real:.2f}%)\n")
        elif not p90_safe:
            print(f"  [Skip] Median improved but P90 ({current_p90:.3f}) > safety gate ({p90_threshold:.3f})\n")

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
            'best_glass_epe_median': getattr(self, 'best_glass_epe_median', float('inf')),
            'best_glass_epe_p90': getattr(self, 'best_glass_epe_p90', float('inf')),
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
        if self.args.dual_stream:
            print("PIDS Training - Stage I (Dual-Stream Polarization)")
        else:
            print("PIDS Training - Stage I (Synthetic Pre-training)")
        print("=" * 60)
        eff_batch = self.args.batch_size * self.args.accumulation_steps
        print(f"Effective batch size: {eff_batch} (batch={self.args.batch_size} x accum={self.args.accumulation_steps})")
        print(f"Validation every {self.args.val_freq} steps")
        if self.args.dual_stream:
            print(f"Polarization Encoder: dim={self.args.pol_dim}, threshold={self.args.pol_threshold}, sharpness={self.args.pol_sharpness}")
            print(f"Learning rate: base={self.args.lr}, pol_encoder={self.args.lr * self.args.pol_lr_mult}")
            print(f"Curriculum Learning: noise_std={self.args.disparity_noise_std}px, "
                  f"warmup={self.args.noise_warmup_steps} steps, max_ratio={self.args.max_noise_ratio}")

        # Curriculum Learning (Exp #28, #29)
        if getattr(self.args, 'curriculum', False) and self.curriculum_sampler is not None:
            print("-" * 60)
            strategy = getattr(self.args, 'curriculum_strategy', 'default')
            if strategy == 'freeze_backbone':
                print("Curriculum Learning (Exp #29): Freeze Backbone First")
            else:
                print("Curriculum Learning (Exp #28): Mixed Pol/Nopol Training")
            for phase in self.curriculum_sampler.config.phases:
                freeze_parts = []
                if phase.freeze_backbone:
                    freeze_parts.append("freeze backbone")
                if phase.freeze_pol_on_nopol:
                    freeze_parts.append("freeze pol on nopol")
                freeze_str = f" [{', '.join(freeze_parts)}]" if freeze_parts else ""
                print(f"  {phase.name}: pol={phase.pol_ratio:.0%}, nopol={phase.nopol_ratio:.0%}{freeze_str}")
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

    # Dual-Stream 偏振編碼器 (強化版)
    parser.add_argument('--dual_stream', action='store_true',
                        help='Use Dual-Stream architecture with Polarization Encoder')
    parser.add_argument('--pol_dim', type=int, default=64,
                        help='Polarization encoder output dimension (強化版: 64)')
    parser.add_argument('--pol_threshold', type=float, default=0.05,
                        help='Soft threshold for polarization difference (強化版: 0.05 捕捉更弱信號)')
    parser.add_argument('--pol_sharpness', type=float, default=20.0,
                        help='Sharpness of soft threshold sigmoid')
    parser.add_argument('--pol_lr_mult', type=float, default=5.0,
                        help='Learning rate multiplier for polarization encoder (降低避免過擬合)')
    parser.add_argument('--pol_weight', type=float, default=2.0,
                        help='Polarization-aware loss weight (偏振區域額外權重)')

    # Polarization Volume 架構 (無 Oracle/Real Gap)
    parser.add_argument('--pol_volume', action='store_true',
                        help='Use Polarization Volume architecture (No Oracle/Real Gap!)')
    parser.add_argument('--pol_levels', type=int, default=4,
                        help='Number of pyramid levels for polarization volume')
    parser.add_argument('--pol_radius', type=int, default=4,
                        help='Lookup radius for polarization volume')

    # NEW: Polarization Volume V2 架構 (Exp #31)
    parser.add_argument('--pol_volume_v2', action='store_true',
                        help='Use Polarization Volume V2 with Attention + Gated Fusion')
    parser.add_argument('--fused_dim', type=int, default=128,
                        help='Dimension of fused features in V2 architecture')

    # NEW: Polarization Volume V2-A 架構 (Exp #33)
    parser.add_argument('--pol_volume_v2a', action='store_true',
                        help='Use Polarization Volume V2-A with Corr Residual')
    parser.add_argument('--residual_hidden_dim', type=int, default=64,
                        help='Hidden dimension for PolCorrResidual network')
    parser.add_argument('--residual_init_scale', type=float, default=0.1,
                        help='Initial scale for residual (small for stable start)')

    # NEW: Polarization Volume V2-B 架構 (Exp #34)
    parser.add_argument('--pol_volume_v2b', action='store_true',
                        help='Use Polarization Volume V2-B with Scheduled Residual')
    # V2-B 使用與 V2-A 相同的 residual_hidden_dim 和 residual_init_scale 參數

    # Polarization Volume V2-C 架構 (Exp #35)
    parser.add_argument('--pol_volume_v2c', action='store_true',
                        help='Use Polarization Volume V2-C with Gradient Gating')
    parser.add_argument('--gating_hidden_dim', type=int, default=32,
                        help='Hidden dimension for GradientGating network (keep small!)')
    # V2-C 使用 residual_hidden_dim, residual_init_scale, 和 gating_hidden_dim

    # NEW: Polarization Volume V2-D 架構 (Exp #38) - 推薦！
    parser.add_argument('--pol_volume_v2d', action='store_true',
                        help='Use Polarization Volume V2-D with Disparity-Aware Pol Modulation (RECOMMENDED)')
    parser.add_argument('--pol_gate_hidden', type=int, default=8,
                        help='Hidden channels for PolGate3D network (keep small!)')
    parser.add_argument('--pol_alpha', type=float, default=0.2,
                        help='Initial modulation strength for V2-D (learnable)')
    # V2-D 是第一個讓 pol 參與 disparity 判別的設計
    # pol_volume[H,W,D] → per-disparity gate → corr modulation

    # Polarization Volume V3 架構 (Exp #36) - 已證明失敗
    parser.add_argument('--pol_volume_v3', action='store_true',
                        help='Use Polarization Volume V3 with Pol-in-Feature (6ch concat, may break pretrained)')
    # V3 最簡單：fnet 輸入從 3 channels 改成 6 channels (RGB + pol_diff)

    # NEW: Polarization Volume V3-B 架構 (Exp #37) - 推薦
    parser.add_argument('--pol_volume_v3b', action='store_true',
                        help='Use Polarization Volume V3-B with Additive Pol Fusion (preserves pretrained)')
    parser.add_argument('--pol_scale', type=float, default=0.1,
                        help='Scale factor for pol contribution in V3-B (default: 0.1)')
    # V3-B 修復 V3: 保持 3ch input，用 side branch + additive fusion

    # NEW: Polarization Volume V2-E 架構 (Exp #39) - 最後嘗試
    parser.add_argument('--pol_volume_v2e', action='store_true',
                        help='Use Polarization Volume V2-E with Pre-Corr Pol Weighting (final attempt)')
    parser.add_argument('--pol_weight_hidden', type=int, default=8,
                        help='Hidden channels for PolWeightNet (keep small!)')
    # V2-E 核心：pol 參與 hypothesis generation / pruning (pre-corr 或 corr construction)
    # 不同於 V2-A~D 的 post-corr intervention，V2-E 是在 corr 構建時就介入

    # Learnable Polarization Volume 架構 (Exp #25)
    parser.add_argument('--learnable_pol', action='store_true',
                        help='Use Learnable Polarization Volume architecture (with PolHead)')
    parser.add_argument('--glass_aware_weight', type=float, default=0.1,
                        help='Weight for glass-aware auxiliary loss (0 = disabled)')

    # NEW: Curriculum Learning for Mixed Pol/Nopol Training (Exp #28, #29, #30)
    parser.add_argument('--curriculum', action='store_true',
                        help='Enable Curriculum Learning for mixed pol/nopol training')
    parser.add_argument('--curriculum_strategy', type=str, default='default',
                        choices=['default', 'freeze_backbone', 'gradual_unfreeze'],
                        help='Curriculum strategy: default (Exp #28), freeze_backbone (Exp #29), or gradual_unfreeze (Exp #30)')
    parser.add_argument('--curriculum_seed', type=int, default=42,
                        help='Random seed for curriculum sampler')

    # Training-Inference Consistency (Curriculum Learning)
    parser.add_argument('--disparity_noise_std', type=float, default=2.0,
                        help='Disparity noise std (pixels) for training-inference consistency')
    parser.add_argument('--noise_warmup_steps', type=int, default=20000,
                        help='Steps to warmup noise_ratio from 0 to max (curriculum learning)')
    parser.add_argument('--max_noise_ratio', type=float, default=0.5,
                        help='Maximum noise ratio (0.5 = 50% samples get noisy disparity)')

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
    parser.add_argument('--glass_weight', type=float, default=5.0,
                        help='Extra weight for glass region loss (強化版: 5.0)')
    parser.add_argument('--strict_glass_weight', type=float, default=0.5,
                        help='Extra weight for strict glass mask (交集區域，保守預設 0.5)')
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

    # ========== Directional Impulse Descent (DID) ==========
    parser.add_argument('--did', action='store_true',
                        help='Enable Directional Impulse Descent for plateau escape')
    # Plateau detection
    parser.add_argument('--did_epsilon_pct', type=float, default=0.10,
                        help='Percentile for train slope threshold (ε)')
    parser.add_argument('--did_train_window', type=int, default=300,
                        help='Train slope EMA window size')
    parser.add_argument('--did_val_m', type=int, default=5,
                        help='Val oscillation detection window (cycles)')
    parser.add_argument('--did_delta', type=float, default=0.5,
                        help='Val EPE oscillation tolerance (px)')
    parser.add_argument('--did_best_stale', type=int, default=4,
                        help='Cycles without new best to consider stale (改進D)')
    # DID impulse
    parser.add_argument('--did_gamma_up', type=float, default=3.0,
                        help='LR multiplier during impulse phase (after first trigger)')
    parser.add_argument('--did_gamma_up_first', type=float, default=2.0,
                        help='LR multiplier for first impulse (more conservative, 改進B)')
    parser.add_argument('--did_gamma_backbone', type=float, default=1.2,
                        help='LR multiplier for backbone during impulse (改進C)')
    parser.add_argument('--did_gamma_down', type=float, default=0.1,
                        help='LR multiplier during cooldown phase')
    parser.add_argument('--did_t_impulse', type=int, default=200,
                        help='Impulse duration (steps)')
    parser.add_argument('--did_t_cool', type=int, default=100,
                        help='Cooldown duration (steps)')
    parser.add_argument('--did_cooldown_mode', type=str, default='relative_base',
                        choices=['relative_base', 'relative_impulse'],
                        help='Cooldown LR reference: relative_base or relative_impulse')
    # Protection
    parser.add_argument('--did_max_triggers', type=int, default=3,
                        help='Maximum DID triggers per training')
    parser.add_argument('--did_protection', type=int, default=3,
                        help='Protection period after trigger (val cycles)')
    parser.add_argument('--did_late_lock', type=float, default=0.75,
                        help='Disable DID after this progress ratio')
    # 改進 E: EPE-adaptive impulse
    parser.add_argument('--did_epe_ref', type=float, default=8.0,
                        help='EPE reference for adaptive gamma (EPE >= ref → full gamma)')
    parser.add_argument('--did_epe_smooth_alpha', type=float, default=0.4,
                        help='EMA alpha for smoothed EPE (safeguard 1)')
    parser.add_argument('--did_delta_gamma_max', type=float, default=0.3,
                        help='Max gamma change per impulse (safeguard 2)')

    return parser.parse_args()


def main():
    args = parse_args()

    # 設置隨機種子
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.benchmark = True
        # 跨平台一致性: 預設關閉 TF32 (Exp #42 結論)
        # TF32 在 Hopper/Blackwell 上導致結果偏差，FP32 三代架構一致
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # 開始訓練
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
