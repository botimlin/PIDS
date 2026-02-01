"""
Curriculum Sampler for PIDS Mixed Pol/Nopol Training
=====================================================

動態調整 pol/nopol 採樣比例，實現 Curriculum Learning 策略。

方案 A+C (原始):
- Phase 1 (0-20%):  50% nopol + 50% pol, nopol 時凍結 Pol Module
- Phase 2 (20-60%): 40% nopol + 60% pol, 全部解凍
- Phase 3 (60-100%): 30% nopol + 70% pol, 全部解凍

方案 B (Freeze Backbone First):
- Phase 1 (0-60%):  100% pol, 凍結 Backbone，只訓練 Pol Module
- Phase 2 (60-100%): 100% pol, 全部解凍，一起微調

Copyright (c) 2025-2026 Po-Ting Lin
Released under the MIT License (see LICENSE file).
"""

import numpy as np
import torch
from torch.utils.data import Sampler, Dataset
from typing import List, Dict, Iterator, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CurriculumPhase:
    """訓練階段配置"""
    name: str
    start_progress: float  # 0.0 ~ 1.0
    end_progress: float
    pol_ratio: float       # pol 數據比例
    nopol_ratio: float     # nopol 數據比例
    freeze_pol_on_nopol: bool  # nopol batch 時是否凍結 pol module
    freeze_backbone: bool = False  # 是否凍結 backbone (RAFT-Stereo 核心)


class CurriculumConfig:
    """Curriculum Learning 配置"""

    # 預設方案 A+C (原始策略)
    DEFAULT_PHASES = [
        CurriculumPhase(
            name="Phase 1: Geometry + Basic Pol",
            start_progress=0.0,
            end_progress=0.2,
            pol_ratio=0.5,
            nopol_ratio=0.5,
            freeze_pol_on_nopol=True,  # 關鍵：nopol 時凍結 pol module
            freeze_backbone=False,
        ),
        CurriculumPhase(
            name="Phase 2: Mixed Training",
            start_progress=0.2,
            end_progress=0.6,
            pol_ratio=0.6,
            nopol_ratio=0.4,
            freeze_pol_on_nopol=False,
            freeze_backbone=False,
        ),
        CurriculumPhase(
            name="Phase 3: Pol Refinement",
            start_progress=0.6,
            end_progress=1.0,
            pol_ratio=0.7,
            nopol_ratio=0.3,
            freeze_pol_on_nopol=False,
            freeze_backbone=False,
        ),
    ]

    # 方案 B: Freeze Backbone First (Exp #29)
    # 理論基礎：先讓 Pol Module 在穩定的 Backbone 上學習偏振特徵
    # 然後再解凍 Backbone 進行協同微調
    # 使用 70/30 pol/nopol 比例：獲得 nopol 幾何多樣性，但 Backbone 凍結保護
    FREEZE_BACKBONE_PHASES = [
        CurriculumPhase(
            name="Phase 1: Train Pol Module Only",
            start_progress=0.0,
            end_progress=0.6,
            pol_ratio=0.7,      # 70% pol + 30% nopol
            nopol_ratio=0.3,
            freeze_pol_on_nopol=False,
            freeze_backbone=True,  # 關鍵：凍結 Backbone，防止 nopol 破壞幾何特徵
        ),
        CurriculumPhase(
            name="Phase 2: Joint Fine-tuning",
            start_progress=0.6,
            end_progress=1.0,
            pol_ratio=0.7,      # 維持 70/30 比例
            nopol_ratio=0.3,
            freeze_pol_on_nopol=False,
            freeze_backbone=False,  # 解凍，一起訓練
        ),
    ]

    # 方案 C: Gradual Unfreeze with 5-Phase LR Schedule (Exp #30)
    # 理論基礎：
    # 1. 數據驅動：Exp #29 顯示 plateau 始於 Step 10000-12000 (16.7%-20%)
    # 2. 經驗法則：尾端低 LR 精修能帶來個位數 px 提升
    # 3. 避免空轉：主學習區用 0.5x LR 持續探索
    #
    # 5 階段 LR Schedule:
    # - 凍結期 (0-20%): backbone_mult=0, lr_mult=1.0
    # - 解凍期 (20-45%): backbone_mult=0→1, lr_mult=1.0
    # - 主學習 (45-80%): backbone_mult=1, lr_mult=0.5
    # - 過渡期 (80-85%): backbone_mult=1, lr_mult=0.2
    # - 精修期 (85-100%): backbone_mult=1, lr_mult=0.1
    #
    # 注意：這個策略使用連續 LR 調整，不是硬切換
    # freeze_backbone 欄位僅用於標記，實際由 GradualUnfreezeLRScheduler 控制
    GRADUAL_UNFREEZE_PHASES = [
        CurriculumPhase(
            name="Phase 1: Frozen Backbone",
            start_progress=0.0,
            end_progress=0.20,
            pol_ratio=0.7,
            nopol_ratio=0.3,
            freeze_pol_on_nopol=False,
            freeze_backbone=True,  # 標記用，實際由 LR scheduler 控制
        ),
        CurriculumPhase(
            name="Phase 2: Gradual Unfreezing",
            start_progress=0.20,
            end_progress=0.45,
            pol_ratio=0.7,
            nopol_ratio=0.3,
            freeze_pol_on_nopol=False,
            freeze_backbone=False,  # backbone 開始有 LR，但從 0 漸進到 1
        ),
        CurriculumPhase(
            name="Phase 3: Main Learning",
            start_progress=0.45,
            end_progress=0.80,
            pol_ratio=0.7,
            nopol_ratio=0.3,
            freeze_pol_on_nopol=False,
            freeze_backbone=False,
        ),
        CurriculumPhase(
            name="Phase 4: Transition",
            start_progress=0.80,
            end_progress=0.85,
            pol_ratio=0.7,
            nopol_ratio=0.3,
            freeze_pol_on_nopol=False,
            freeze_backbone=False,
        ),
        CurriculumPhase(
            name="Phase 5: Fine-tuning",
            start_progress=0.85,
            end_progress=1.0,
            pol_ratio=0.7,
            nopol_ratio=0.3,
            freeze_pol_on_nopol=False,
            freeze_backbone=False,
        ),
    ]

    def __init__(self, phases: Optional[List[CurriculumPhase]] = None, strategy: str = 'default'):
        """
        Args:
            phases: 自定義階段配置
            strategy: 預設策略選擇 ('default' | 'freeze_backbone' | 'gradual_unfreeze')
        """
        if phases is not None:
            self.phases = phases
        elif strategy == 'freeze_backbone':
            self.phases = self.FREEZE_BACKBONE_PHASES
        elif strategy == 'gradual_unfreeze':
            self.phases = self.GRADUAL_UNFREEZE_PHASES
        else:
            self.phases = self.DEFAULT_PHASES
        self._validate_phases()

    def _validate_phases(self):
        """驗證階段配置"""
        for phase in self.phases:
            assert 0 <= phase.start_progress < phase.end_progress <= 1.0
            assert abs(phase.pol_ratio + phase.nopol_ratio - 1.0) < 1e-6

    def get_phase(self, progress: float) -> CurriculumPhase:
        """根據訓練進度獲取當前階段"""
        for phase in self.phases:
            if phase.start_progress <= progress < phase.end_progress:
                return phase
        return self.phases[-1]  # 默認返回最後階段


class CurriculumSampler(Sampler):
    """
    Curriculum Learning Sampler

    根據訓練進度動態調整 pol/nopol 採樣比例。

    Usage:
        sampler = CurriculumSampler(
            dataset=train_dataset,
            total_steps=60000,
            batch_size=8
        )

        for step in range(total_steps):
            sampler.set_step(step)
            batch_indices = sampler.sample_batch()
            batch_info = sampler.get_batch_info()
    """

    def __init__(
        self,
        dataset: Dataset,
        total_steps: int,
        batch_size: int,
        config: Optional[CurriculumConfig] = None,
        seed: int = 42,
    ):
        """
        Args:
            dataset: PIDSSyntheticDataset 實例 (需有 scene_naming 屬性)
            total_steps: 總訓練步數
            batch_size: Batch 大小
            config: Curriculum 配置
            seed: 隨機種子
        """
        self.dataset = dataset
        self.total_steps = total_steps
        self.batch_size = batch_size
        self.config = config or CurriculumConfig()
        self.rng = np.random.RandomState(seed)

        self.current_step = 0
        self._last_batch_info = None

        # 分離 pol 和 nopol 場景索引
        self._separate_indices()

        print(f"[CurriculumSampler] Initialized")
        print(f"  Total steps: {total_steps}")
        print(f"  Batch size: {batch_size}")
        print(f"  Pol scenes: {len(self.pol_indices)}")
        print(f"  Nopol scenes: {len(self.nopol_indices)}")
        print(f"  Phases: {len(self.config.phases)}")
        for phase in self.config.phases:
            freeze_info = []
            if phase.freeze_pol_on_nopol:
                freeze_info.append("freeze_pol_on_nopol")
            if phase.freeze_backbone:
                freeze_info.append("freeze_backbone")
            freeze_str = f" [{', '.join(freeze_info)}]" if freeze_info else ""
            print(f"    - {phase.name}: pol={phase.pol_ratio:.0%}, nopol={phase.nopol_ratio:.0%}{freeze_str}")

    def _separate_indices(self):
        """分離 pol 和 nopol 場景的索引"""
        self.pol_indices = []
        self.nopol_indices = []

        # 檢查 dataset 是否有 scene_naming 屬性
        if hasattr(self.dataset, 'scene_naming') and hasattr(self.dataset, 'scenes'):
            for idx, scene_name in enumerate(self.dataset.scenes):
                naming = self.dataset.scene_naming.get(scene_name, 'pol')
                if naming == 'pol':
                    self.pol_indices.append(idx)
                else:
                    self.nopol_indices.append(idx)
        else:
            # 如果沒有 scene_naming，假設全部是 pol
            self.pol_indices = list(range(len(self.dataset)))
            print("[CurriculumSampler] Warning: dataset has no scene_naming, assuming all pol")

        self.pol_indices = np.array(self.pol_indices)
        self.nopol_indices = np.array(self.nopol_indices)

    def set_step(self, step: int):
        """設置當前訓練步數"""
        self.current_step = step

    @property
    def progress(self) -> float:
        """當前訓練進度 (0.0 ~ 1.0)"""
        return min(self.current_step / max(self.total_steps, 1), 1.0)

    @property
    def current_phase(self) -> CurriculumPhase:
        """當前訓練階段"""
        return self.config.get_phase(self.progress)

    def sample_batch(self) -> Tuple[List[int], List[str]]:
        """
        採樣一個 batch

        Returns:
            (indices, data_types): 索引列表和對應的數據類型列表
        """
        phase = self.current_phase

        # 計算這個 batch 中 pol 和 nopol 的數量
        n_pol = int(self.batch_size * phase.pol_ratio)
        n_nopol = self.batch_size - n_pol

        # 處理邊界情況
        if len(self.nopol_indices) == 0:
            n_pol = self.batch_size
            n_nopol = 0
        elif len(self.pol_indices) == 0:
            n_pol = 0
            n_nopol = self.batch_size

        # 採樣
        indices = []
        data_types = []

        if n_pol > 0 and len(self.pol_indices) > 0:
            pol_samples = self.rng.choice(self.pol_indices, size=n_pol, replace=True)
            indices.extend(pol_samples.tolist())
            data_types.extend(['pol'] * n_pol)

        if n_nopol > 0 and len(self.nopol_indices) > 0:
            nopol_samples = self.rng.choice(self.nopol_indices, size=n_nopol, replace=True)
            indices.extend(nopol_samples.tolist())
            data_types.extend(['nopol'] * n_nopol)

        # 打亂順序
        combined = list(zip(indices, data_types))
        self.rng.shuffle(combined)
        indices, data_types = zip(*combined) if combined else ([], [])

        # 保存 batch 資訊
        self._last_batch_info = {
            'step': self.current_step,
            'progress': self.progress,
            'phase': phase.name,
            'n_pol': n_pol,
            'n_nopol': n_nopol,
            'freeze_pol_on_nopol': phase.freeze_pol_on_nopol,
            'freeze_backbone': phase.freeze_backbone,  # 新增
            'data_types': list(data_types),
        }

        return list(indices), list(data_types)

    def get_batch_info(self) -> Dict:
        """獲取最後一個 batch 的資訊"""
        return self._last_batch_info or {}

    def should_freeze_pol_module(self, data_type: str) -> bool:
        """
        判斷是否應該凍結 pol module

        Args:
            data_type: 'pol' 或 'nopol'

        Returns:
            是否凍結
        """
        phase = self.current_phase
        return phase.freeze_pol_on_nopol and data_type == 'nopol'

    def get_freeze_mask(self, data_types: List[str]) -> List[bool]:
        """
        獲取 batch 中每個樣本的凍結狀態

        Args:
            data_types: 數據類型列表

        Returns:
            凍結狀態列表 (True = 凍結)
        """
        return [self.should_freeze_pol_module(dt) for dt in data_types]

    def should_freeze_backbone(self) -> bool:
        """
        判斷是否應該凍結 backbone

        Returns:
            是否凍結
        """
        return self.current_phase.freeze_backbone

    def __iter__(self) -> Iterator[int]:
        """
        標準 Sampler 接口（用於 DataLoader）
        注意：這個方法生成整個 epoch 的索引，但不考慮 curriculum
        對於 curriculum training，建議直接使用 sample_batch()
        """
        indices = []
        phase = self.current_phase

        # 計算一個 epoch 需要的樣本數
        n_samples = len(self.dataset)
        n_pol = int(n_samples * phase.pol_ratio)
        n_nopol = n_samples - n_pol

        if len(self.pol_indices) > 0:
            pol_samples = self.rng.choice(self.pol_indices, size=min(n_pol, len(self.pol_indices) * 10), replace=True)
            indices.extend(pol_samples[:n_pol].tolist())

        if len(self.nopol_indices) > 0:
            nopol_samples = self.rng.choice(self.nopol_indices, size=min(n_nopol, len(self.nopol_indices) * 10), replace=True)
            indices.extend(nopol_samples[:n_nopol].tolist())

        self.rng.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        return len(self.dataset)


class PolModuleFreezer:
    """
    Pol Module 凍結控制器

    Usage:
        freezer = PolModuleFreezer(model)

        # 訓練迴圈
        for batch in dataloader:
            data_types = batch['data_type']

            # 根據 batch 中的 nopol 比例決定是否凍結
            if curriculum_sampler.current_phase.freeze_pol_on_nopol:
                freezer.freeze()
            else:
                freezer.unfreeze()

            # forward & backward...
    """

    def __init__(self, model: torch.nn.Module, pol_keywords: List[str] = None):
        """
        Args:
            model: PIDS 模型
            pol_keywords: 識別 pol module 參數的關鍵字
        """
        self.model = model
        self.pol_keywords = pol_keywords or ['pol', 'Pol', 'POL']
        self._pol_params = self._find_pol_params()
        self._frozen = False

        print(f"[PolModuleFreezer] Found {len(self._pol_params)} pol parameters")

    def _find_pol_params(self) -> List[Tuple[str, torch.nn.Parameter]]:
        """找到所有 pol module 的參數"""
        pol_params = []
        for name, param in self.model.named_parameters():
            if any(kw in name for kw in self.pol_keywords):
                pol_params.append((name, param))
        return pol_params

    def freeze(self):
        """凍結 pol module"""
        if not self._frozen:
            for name, param in self._pol_params:
                param.requires_grad = False
            self._frozen = True

    def unfreeze(self):
        """解凍 pol module"""
        if self._frozen:
            for name, param in self._pol_params:
                param.requires_grad = True
            self._frozen = False

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def get_grad_norm(self) -> float:
        """計算 pol module 的梯度 norm (用於監控)"""
        total_norm = 0.0
        for name, param in self._pol_params:
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5


class V2PolModuleFreezer:
    """
    V2 Pol Module 凍結控制器 (專用於 PIDSStereoPolVolumeV2)

    凍結 V2 特有的偏振模組：
    - PolarizationAttention
    - GatedFusion

    這些模組只應該從 pol 數據學習，不應從 nopol 數據學習。
    因為 nopol 數據沒有偏振信號，會「稀釋」V2 模組的學習。

    Usage:
        freezer = V2PolModuleFreezer(model)

        for batch in dataloader:
            if has_nopol_in_batch:
                freezer.freeze()  # nopol batch: 凍結 V2 模組
            else:
                freezer.unfreeze()  # pol batch: V2 模組正常學習
    """

    def __init__(self, model: torch.nn.Module):
        """
        Args:
            model: PIDSStereoPolVolumeV2 模型
        """
        self.model = model
        self._frozen = False

        # 統計 V2 模組參數數量
        v2_params = 0
        for name, param in model.named_parameters():
            if 'pol_attention' in name or 'gated_fusion' in name:
                v2_params += param.numel()
        print(f"[V2PolModuleFreezer] Found {v2_params:,} V2 pol module parameters")

    def freeze(self):
        """凍結 V2 偏振模組 (PolarizationAttention + GatedFusion)"""
        if not self._frozen:
            if hasattr(self.model, 'freeze_pol_modules'):
                self.model.freeze_pol_modules()
            self._frozen = True

    def unfreeze(self):
        """解凍 V2 偏振模組"""
        if self._frozen:
            if hasattr(self.model, 'unfreeze_pol_modules'):
                self.model.unfreeze_pol_modules()
            self._frozen = False

    @property
    def is_frozen(self) -> bool:
        return self._frozen


class BackboneFreezer:
    """
    Backbone (RAFT-Stereo) 凍結控制器

    凍結 RAFT-Stereo 的核心模組：
    - fnet (Feature Network)
    - cnet (Context Network)
    - update_block (GRU Update Block)

    保持 Pol Module 可訓練。

    Usage:
        freezer = BackboneFreezer(model)

        # Phase 1: 只訓練 Pol Module
        freezer.freeze()

        # Phase 2: 一起訓練
        freezer.unfreeze()
    """

    # RAFT-Stereo Feature Encoder 關鍵字 (只凍結特徵提取，保留 GRU 可訓練)
    # 注意：對於 pol_volume 架構，update_block 需要保持可訓練
    #       因為 PolCorrBlock 是非參數的，只有 update_block 能學習如何使用 pol features
    BACKBONE_KEYWORDS = [
        'fnet',           # Feature Network
        'cnet',           # Context Network
        # 'update_block', # GRU Update Block - 保持可訓練！
        # 'corr_fn',      # Correlation Function - 通常無參數
    ]

    # Pol Module 的關鍵字 (這些不應該被凍結)
    POL_KEYWORDS = ['pol', 'Pol', 'POL']

    def __init__(self, model: torch.nn.Module, backbone_keywords: List[str] = None):
        """
        Args:
            model: PIDS 模型
            backbone_keywords: 識別 backbone 參數的關鍵字 (可選)
        """
        self.model = model
        self.backbone_keywords = backbone_keywords or self.BACKBONE_KEYWORDS
        self._backbone_params = self._find_backbone_params()
        self._frozen = False

        print(f"[BackboneFreezer] Found {len(self._backbone_params)} backbone parameters")
        if len(self._backbone_params) > 0:
            # 顯示前幾個參數名稱作為確認
            sample_names = [name for name, _ in self._backbone_params[:5]]
            print(f"  Sample params: {sample_names}")

    def _find_backbone_params(self) -> List[Tuple[str, torch.nn.Parameter]]:
        """找到所有 backbone 的參數 (排除 Pol Module)"""
        backbone_params = []
        for name, param in self.model.named_parameters():
            # 排除 Pol Module
            is_pol = any(kw in name for kw in self.POL_KEYWORDS)
            if is_pol:
                continue

            # 檢查是否屬於 backbone
            is_backbone = any(kw in name for kw in self.backbone_keywords)
            if is_backbone:
                backbone_params.append((name, param))

        return backbone_params

    def freeze(self):
        """凍結 backbone"""
        if not self._frozen:
            for name, param in self._backbone_params:
                param.requires_grad = False
            self._frozen = True
            print("[BackboneFreezer] Backbone frozen")

    def unfreeze(self):
        """解凍 backbone"""
        if self._frozen:
            for name, param in self._backbone_params:
                param.requires_grad = True
            self._frozen = False
            print("[BackboneFreezer] Backbone unfrozen")

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def get_grad_norm(self) -> float:
        """計算 backbone 的梯度 norm (用於監控)"""
        total_norm = 0.0
        for name, param in self._backbone_params:
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def get_trainable_param_count(self) -> Dict[str, int]:
        """獲取可訓練參數統計"""
        backbone_trainable = sum(1 for _, p in self._backbone_params if p.requires_grad)
        backbone_total = len(self._backbone_params)

        # 計算 Pol Module 參數
        pol_trainable = 0
        pol_total = 0
        for name, param in self.model.named_parameters():
            if any(kw in name for kw in self.POL_KEYWORDS):
                pol_total += 1
                if param.requires_grad:
                    pol_trainable += 1

        return {
            'backbone_trainable': backbone_trainable,
            'backbone_total': backbone_total,
            'pol_trainable': pol_trainable,
            'pol_total': pol_total,
        }


class GradualUnfreezeLRScheduler:
    """
    5 階段 LR Schedule for Gradual Unfreezing (Exp #30)

    設計依據：
    - 數據驅動：Exp #29 顯示 plateau 始於 Step 10000-12000 (16.7%-20%)
    - 經驗法則：尾端低 LR 精修能帶來個位數 px 提升
    - 避免空轉：主學習區用 0.5x LR 持續探索

    5 階段：
    - 凍結期 (0-20%): backbone_mult=0, lr_mult=1.0
    - 解凍期 (20-45%): backbone_mult=0→1, lr_mult=1.0
    - 主學習 (45-80%): backbone_mult=1, lr_mult=0.5
    - 過渡期 (80-85%): backbone_mult=1, lr_mult=0.2
    - 精修期 (85-100%): backbone_mult=1, lr_mult=0.1

    Usage:
        scheduler = GradualUnfreezeLRScheduler(
            optimizer=optimizer,
            base_lr=0.0002,
            total_steps=60000,
            backbone_param_groups=[0, 1],  # fnet, cnet
            head_param_groups=[2, 3],       # update_block, pol
        )

        for step in range(total_steps):
            scheduler.step(step)
            # ... training code ...
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        total_steps: int,
        backbone_param_groups: List[int] = None,
        head_param_groups: List[int] = None,
    ):
        """
        Args:
            optimizer: PyTorch optimizer with param_groups
            base_lr: Base learning rate
            total_steps: Total training steps
            backbone_param_groups: Indices of backbone param groups (default: [0, 1])
            head_param_groups: Indices of head param groups (default: [2, 3])
        """
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.backbone_groups = backbone_param_groups or [0, 1]
        self.head_groups = head_param_groups or [2, 3]

        self.current_step = 0
        self._last_lr_info = {}

        print(f"[GradualUnfreezeLRScheduler] Initialized")
        print(f"  Base LR: {base_lr}")
        print(f"  Total steps: {total_steps}")
        print(f"  Backbone param groups: {self.backbone_groups}")
        print(f"  Head param groups: {self.head_groups}")

    def get_lr_config(self, progress: float) -> Tuple[float, float]:
        """
        計算當前 progress 的 LR 配置

        Args:
            progress: 訓練進度 (0.0 ~ 1.0)

        Returns:
            (lr_mult, backbone_mult): LR 乘數和 backbone 乘數
        """
        # === Backbone multiplier ===
        if progress < 0.20:
            backbone_mult = 0.0                       # 凍結
        elif progress < 0.45:
            backbone_mult = (progress - 0.20) / 0.25  # 0→1 漸進解凍
        else:
            backbone_mult = 1.0                       # 全開

        # === Base LR multiplier ===
        if progress < 0.45:
            lr_mult = 1.0           # 前段保持 base LR
        elif progress < 0.80:
            lr_mult = 0.5           # 主學習區：半速探索
        elif progress < 0.85:
            lr_mult = 0.2           # 過渡期
        else:
            lr_mult = 0.1           # 尾端精修

        return lr_mult, backbone_mult

    def step(self, step: int = None):
        """
        更新 LR

        Args:
            step: 當前 step (如果 None，使用內部計數器)
        """
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1

        progress = min(self.current_step / max(self.total_steps, 1), 1.0)
        lr_mult, backbone_mult = self.get_lr_config(progress)

        # 計算實際 LR
        current_lr = self.base_lr * lr_mult
        backbone_lr = current_lr * backbone_mult

        # 更新 optimizer param groups
        for idx in self.backbone_groups:
            if idx < len(self.optimizer.param_groups):
                self.optimizer.param_groups[idx]['lr'] = backbone_lr

        for idx in self.head_groups:
            if idx < len(self.optimizer.param_groups):
                self.optimizer.param_groups[idx]['lr'] = current_lr

        # 保存資訊
        self._last_lr_info = {
            'step': self.current_step,
            'progress': progress,
            'lr_mult': lr_mult,
            'backbone_mult': backbone_mult,
            'head_lr': current_lr,
            'backbone_lr': backbone_lr,
        }

    def get_last_lr(self) -> Dict[str, float]:
        """獲取最後的 LR 資訊"""
        return self._last_lr_info

    def get_phase_name(self, progress: float = None) -> str:
        """獲取當前階段名稱"""
        if progress is None:
            progress = min(self.current_step / max(self.total_steps, 1), 1.0)

        if progress < 0.20:
            return "Frozen (0-20%)"
        elif progress < 0.45:
            return "Unfreezing (20-45%)"
        elif progress < 0.80:
            return "Main Learning (45-80%)"
        elif progress < 0.85:
            return "Transition (80-85%)"
        else:
            return "Fine-tuning (85-100%)"

    def state_dict(self) -> Dict:
        """返回 scheduler 狀態（用於 checkpoint 保存）"""
        return {
            'current_step': self.current_step,
            'base_lr': self.base_lr,
            'total_steps': self.total_steps,
        }

    def load_state_dict(self, state_dict: Dict):
        """載入 scheduler 狀態（用於 checkpoint 恢復）"""
        self.current_step = state_dict.get('current_step', 0)
        # base_lr 和 total_steps 應該從 args 來，這裡只是驗證
        if 'base_lr' in state_dict and state_dict['base_lr'] != self.base_lr:
            print(f"[GradualUnfreezeLRScheduler] Warning: base_lr mismatch "
                  f"(checkpoint: {state_dict['base_lr']}, current: {self.base_lr})")
        # 恢復後立即更新 LR
        self.step(self.current_step)


class MixedBatchCollator:
    """
    混合 Batch 整理器

    將 dataset 輸出的 dict 整理成包含 data_type 的 batch。

    Usage:
        collator = MixedBatchCollator()
        dataloader = DataLoader(dataset, collate_fn=collator)
    """

    def __init__(self, dataset: Dataset):
        """
        Args:
            dataset: PIDSSyntheticDataset 實例
        """
        self.dataset = dataset

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """整理 batch"""
        # 標準 collate
        keys = batch[0].keys()
        collated = {}

        for key in keys:
            if key == 'scene_name':
                collated[key] = [item[key] for item in batch]
            else:
                collated[key] = torch.stack([item[key] for item in batch])

        # 添加 data_type
        data_types = []
        for item in batch:
            scene_name = item['scene_name']
            naming = self.dataset.scene_naming.get(scene_name, 'pol')
            data_types.append(naming)

        collated['data_type'] = data_types

        return collated


def create_curriculum_dataloader(
    dataset: Dataset,
    batch_size: int,
    total_steps: int,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[torch.utils.data.DataLoader, CurriculumSampler]:
    """
    創建帶有 Curriculum Sampler 的 DataLoader

    Args:
        dataset: PIDSSyntheticDataset
        batch_size: Batch 大小
        total_steps: 總訓練步數
        num_workers: Worker 數量
        seed: 隨機種子

    Returns:
        (dataloader, sampler)
    """
    sampler = CurriculumSampler(
        dataset=dataset,
        total_steps=total_steps,
        batch_size=batch_size,
        seed=seed,
    )

    collator = MixedBatchCollator(dataset)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collator,
    )

    return dataloader, sampler


# ============================================================
# 測試程式碼
# ============================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str, default='default',
                        choices=['default', 'freeze_backbone'],
                        help='Curriculum strategy to test')
    args = parser.parse_args()

    print(f"Testing CurriculumSampler with strategy: {args.strategy}")
    print("=" * 60)

    # 模擬 dataset
    class MockDataset:
        def __init__(self, n_pol=100, n_nopol=50):
            self.scenes = [f"scene_{i:04d}" for i in range(n_pol + n_nopol)]
            self.scene_naming = {}
            for i in range(n_pol):
                self.scene_naming[f"scene_{i:04d}"] = 'pol'
            for i in range(n_pol, n_pol + n_nopol):
                self.scene_naming[f"scene_{i:04d}"] = 'nopol'

        def __len__(self):
            return len(self.scenes)

    dataset = MockDataset(n_pol=3000, n_nopol=2000)

    # 使用指定策略
    config = CurriculumConfig(strategy=args.strategy)
    sampler = CurriculumSampler(
        dataset=dataset,
        total_steps=60000,
        batch_size=8,
        config=config,
    )

    # 測試不同階段
    if args.strategy == 'freeze_backbone':
        test_steps = [0, 18000, 36000, 45000, 60000]  # 0%, 30%, 60%, 75%, 100%
    else:
        test_steps = [0, 6000, 12000, 30000, 50000, 60000]

    for step in test_steps:
        sampler.set_step(step)
        indices, data_types = sampler.sample_batch()
        info = sampler.get_batch_info()
        print(f"\nStep {step} ({info['progress']:.1%}):")
        print(f"  Phase: {info['phase']}")
        print(f"  Pol: {info['n_pol']}, Nopol: {info['n_nopol']}")
        print(f"  Freeze pol on nopol: {info['freeze_pol_on_nopol']}")
        print(f"  Freeze backbone: {info['freeze_backbone']}")
        print(f"  Data types: {data_types}")

    print("\n" + "=" * 60)
    print("Test passed!")
