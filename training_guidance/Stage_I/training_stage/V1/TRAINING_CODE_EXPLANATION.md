# PIDS Training Code 逐行說明

**版本**: 2026-01-29
**作者**: Po-Ting Lin

本文件提供 PIDS 訓練代碼的詳細解析，涵蓋所有架構版本。

---

## 目錄

1. [檔案結構](#1-檔案結構)
2. [架構總覽](#2-架構總覽)
3. [train_pids.py 詳解](#3-train_pidspy-詳解)
4. [pids_model.py 核心模組](#4-pids_modelpy-核心模組)
5. [Curriculum Learning 與凍結策略](#5-curriculum-learning-與凍結策略)
6. [訓練命令範例](#6-訓練命令範例)

---

## 1. 檔案結構

```
training_stage/
├── train_pids.py              # 主訓練腳本
├── evaluate_pids.py           # 評估腳本
├── pids_model.py              # 模型定義 (所有架構)
├── pids_dataset.py            # 數據集載入器
├── curriculum_sampler.py      # Curriculum Learning 組件
└── TRAINING_CODE_EXPLANATION.md
```

---

## 2. 架構總覽

### 2.1 架構演進

```
Baseline RAFT-Stereo
    │
    ├─→ Dual-Stream (有 Oracle-Real Gap)
    │
    └─→ Polarization Volume V1 (Exp #24)
            │ 消除 Gap
            ↓
        ┌── V2 系列 (Post-Corr) ──────────────────┐
        │   V2    : Attention + Gated Fusion      │
        │   V2-A  : Static Residual               │
        │   V2-B  : Scheduled Residual            │
        │   V2-C  : Gradient Gating               │
        │   V2-D  : Disparity-Aware Gate (3D)     │
        └─────────────────────────────────────────┘
            │
        V3 系列 (Early Fusion) → ❌ 失敗
            │
        V2-E (Pre-Corr) ← 最新
```

### 2.2 架構參數對照

| 參數 | 模型類別 | 說明 |
|------|----------|------|
| `--pol_volume` | `PIDSStereoPolVolume` | V1: 基礎 Pol Volume |
| `--pol_volume_v2` | `PIDSStereoPolVolumeV2` | Attention + Gated Fusion |
| `--pol_volume_v2a` | `PIDSStereoPolVolumeV2A` | Static Residual |
| `--pol_volume_v2b` | `PIDSStereoPolVolumeV2B` | Scheduled Residual |
| `--pol_volume_v2c` | `PIDSStereoPolVolumeV2C` | Gradient Gating |
| `--pol_volume_v2d` | `PIDSStereoPolVolumeV2D` | Disparity-Aware Gate |
| `--pol_volume_v2e` | `PIDSStereoPolVolumeV2E` | Pre-Corr Weighting |
| `--pol_volume_v3` | `PIDSStereoPolVolumeV3` | ❌ 已棄用 |
| `--pol_volume_v3b` | `PIDSStereoPolVolumeV3B` | ❌ 已棄用 |

---

## 3. train_pids.py 詳解

### 3.1 導入區塊

```python
from pids_model import (
    PIDSStereoLoss,
    PIDSStereoDualStream,
    PIDSStereoPolVolume,
    PIDSStereoPolVolumeV2,
    PIDSStereoPolVolumeV2A,
    PIDSStereoPolVolumeV2B,
    PIDSStereoPolVolumeV2C,
    PIDSStereoPolVolumeV2D,
    PIDSStereoPolVolumeV2E,  # 最新: Pre-Corr
    PIDSStereoPolVolumeV3,   # 已棄用
    PIDSStereoPolVolumeV3B,  # 已棄用
)

from curriculum_sampler import (
    CurriculumSampler,       # 動態 pol/nopol 採樣
    CurriculumConfig,        # 配置
    PolModuleFreezer,        # 凍結 pol 模組
    BackboneFreezer,         # 凍結 backbone
    V2PolModuleFreezer,      # V2 專用凍結器
    MixedBatchCollator,      # 混合 batch 整理
)
```

### 3.2 Trainer 初始化

```python
class Trainer:
    def __init__(self, args):
        # ...

        # Curriculum Learning 組件
        self.curriculum_sampler = None
        self.pol_freezer = None
        self.backbone_freezer = None
        self.v2_pol_freezer = None

        if getattr(args, 'curriculum', False):
            self._setup_curriculum()
```

### 3.3 模型建立 (_build_model)

```python
def _build_model(self):
    if getattr(self.args, 'pol_volume_v2e', False):
        # ========== V2-E: Pre-Corr Pol Weighting ==========
        model = PIDSStereoPolVolumeV2E(
            hidden_dim=self.args.hidden_dim,
            context_dim=self.args.context_dim,
            feature_dim=self.args.feature_dim,
            corr_levels=self.args.corr_levels,
            corr_radius=self.args.corr_radius,
            pol_weight_hidden=getattr(self.args, 'pol_weight_hidden', 8),
        )

    elif getattr(self.args, 'pol_volume_v2d', False):
        # ========== V2-D: Disparity-Aware Pol Modulation ==========
        model = PIDSStereoPolVolumeV2D(
            pol_gate_hidden=getattr(self.args, 'pol_gate_hidden', 8),
            pol_alpha=getattr(self.args, 'pol_alpha', 0.2),
            # ...
        )

    # ... 其他架構 ...

    # 載入預訓練權重
    if self.args.pretrained:
        self._load_pretrained_weights(model, self.args.pretrained)

    # 凍結 Feature Encoder (可選)
    if self.args.freeze_fnet:
        for name, param in model.named_parameters():
            if 'fnet' in name:
                param.requires_grad = False
```

### 3.4 Curriculum Learning 設置

```python
def _setup_curriculum(self):
    """設置 Curriculum Learning 組件"""

    # 1. 選擇策略
    strategy = self.args.curriculum_strategy  # 'default', 'freeze_backbone', 'gradual_unfreeze'

    # 2. 初始化 Sampler
    config = CurriculumConfig(strategy=strategy)
    self.curriculum_sampler = CurriculumSampler(
        dataset=self.train_dataset,
        total_steps=self.args.num_steps,
        batch_size=self.args.batch_size,
        config=config,
        seed=self.args.curriculum_seed,
    )

    # 3. 初始化 Freezer
    self.pol_freezer = PolModuleFreezer(self.model)

    if strategy == 'freeze_backbone':
        self.backbone_freezer = BackboneFreezer(self.model)

    # V2 系列專用
    if any([self.args.pol_volume_v2a, self.args.pol_volume_v2b, ...]):
        self.v2_pol_freezer = V2PolModuleFreezer(self.model)
```

### 3.5 訓練循環 (含 Curriculum)

```python
def train_epoch(self):
    for batch_idx, batch in enumerate(self.train_loader):
        # 1. Curriculum: 獲取 batch 資訊
        if self.curriculum_sampler:
            batch_info = self.curriculum_sampler.get_batch_info(self.global_step)
            data_type = batch.get('data_type', 'pol')

            # Phase 1: nopol 時凍結 pol module
            if batch_info['freeze_pol_on_nopol'] and data_type == 'nopol':
                self.pol_freezer.freeze()
            else:
                self.pol_freezer.unfreeze()

            # Freeze Backbone 策略
            if self.backbone_freezer:
                if batch_info.get('freeze_backbone', False):
                    self.backbone_freezer.freeze()
                else:
                    self.backbone_freezer.unfreeze()

        # 2. 前向傳播
        flow_preds = self.model(left, right, iters=self.args.iters)

        # 3. Loss 計算
        loss, metrics = self.criterion(...)

        # 4. 反向傳播
        loss.backward()
        self.optimizer.step()
```

### 3.6 驗證邏輯

```python
def _validate_and_log(self):
    # 1. Oracle 驗證 (所有架構)
    val_metrics_oracle = self.validate(use_oracle=True)

    # 2. Real 驗證
    is_pol_volume = (
        self.args.pol_volume or
        self.args.pol_volume_v2 or
        self.args.pol_volume_v2a or
        self.args.pol_volume_v2b or
        self.args.pol_volume_v2c or
        self.args.pol_volume_v2d or
        self.args.pol_volume_v2e or  # 新增
        self.args.pol_volume_v3 or
        self.args.pol_volume_v3b
    )

    if self.args.dual_stream and not is_pol_volume:
        # Dual-Stream: Oracle ≠ Real
        val_metrics_real = self.validate(use_oracle=False)
    else:
        # Pol Volume 系列: Oracle = Real
        val_metrics_real = val_metrics_oracle
```

---

## 4. pids_model.py 核心模組

### 4.1 V1: PolCorrBlock (基礎)

```python
class PolCorrBlock:
    """
    偏振差異體積

    pol[x, d] = left[x] - right[x-d]

    優勢: 不依賴 disparity，Oracle = Real
    """

    def __init__(self, img_left, img_right, num_levels=4, radius=4):
        pol_volume = self._compute_pol_volume(img_left, img_right)
        self.pol_pyramid = self._build_pyramid(pol_volume)

    def __call__(self, disp):
        # 用 disparity 查詢偏振差異
        return self._lookup(disp)
```

### 4.2 V2-A: PolCorrResidual (Post-Corr)

```python
class PolCorrResidual(nn.Module):
    """
    將 pol_corr 轉換為 corr 的 residual

    corr_enhanced = corr + alpha * residual(pol_corr)
    """

    def __init__(self, input_dim, hidden_dim=64, init_scale=0.1):
        self.net = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, input_dim, 1),
        )
        # 小初始化確保 residual 一開始接近 0
        nn.init.constant_(self.net[-1].weight, 0)
        nn.init.constant_(self.net[-1].bias, 0)

    def forward(self, pol_corr):
        return self.net(pol_corr) * self.scale
```

### 4.3 V2-D: DisparityAwarePolVolume + PolGate3D

```python
class DisparityAwarePolVolume:
    """
    Disparity-Aware Pol Volume

    對於每個 disparity d:
        pol_diff_d = left - shift(right, d)

    輸出: pol_volume [B, 3, D, H, W]
    """

class PolGate3D(nn.Module):
    """
    3D Gate: 判斷每個 disparity 的合理性

    輸入: pol_volume [B, 3, D, H, W]
    輸出: gate [B, 1, D, H, W] ∈ [0, 1]
    """

    def __init__(self, hidden_channels=8):
        self.net = nn.Sequential(
            nn.Conv3d(3, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
```

### 4.4 V2-E: PolWeightedCorrBlock (Pre-Corr)

```python
class PolWeightNet(nn.Module):
    """將 pol_diff 轉換為 weight"""

    def __init__(self, hidden_channels=8):
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, 1),
            nn.Sigmoid(),
        )

class PolWeightedCorrBlock:
    """
    Pre-Corr Pol Weighting

    與 V2-A~D (Post-Corr) 不同：
    - pol 在 correlation 構建時就介入
    - 不是事後修正，而是源頭介入
    """

    def __init__(self, fmap1, fmap2, left, right, pol_weight_net):
        # 1. 計算原始 correlation
        corr = self._compute_correlation(fmap1, fmap2)

        # 2. 計算 pol_weight volume
        pol_weight = self._compute_pol_weight_volume(left, right, pol_weight_net)

        # 3. Pre-Corr weighting (核心！)
        corr_weighted = corr * pol_weight

        # 4. 構建金字塔
        self.corr_pyramid = self._build_pyramid(corr_weighted)
```

### 4.5 PIDSStereoLoss

```python
class PIDSStereoLoss(nn.Module):
    """
    損失函數

    組成:
    - L1 基礎損失
    - Sequence Loss (γ 加權)
    - Glass Weight (玻璃區域加權)
    - Strict Glass Weight (交集區域額外加權)
    """

    def forward(self, disp_preds, disp_gt, valid_mask, glass_mask,
                pol_diff=None, glass_mask_strict=None):
        # 權重遮罩
        weight_mask = torch.ones_like(valid_mask)

        # 玻璃區域分層加權
        if glass_mask is not None:
            union_only = glass_mask * (1 - glass_mask_strict)  # 邊緣: w = 5.0
            strict = glass_mask_strict                          # 核心: w = 5.5
            weight_mask += (self.glass_weight - 1) * union_only
            weight_mask += (self.glass_weight - 1 + self.strict_glass_weight) * strict

        # Sequence Loss
        for i, pred in enumerate(disp_preds):
            w = self.gamma ** (n - i - 1)
            loss += w * (|pred - gt| * weight_mask).mean()
```

---

## 5. Curriculum Learning 與凍結策略

### 5.1 CurriculumSampler

```python
class CurriculumSampler:
    """動態 pol/nopol 採樣"""

    PHASES = [
        CurriculumPhase(
            name="Phase 1: Geometry",
            start_progress=0.0, end_progress=0.2,
            pol_ratio=0.5, nopol_ratio=0.5,
            freeze_pol_on_nopol=True,   # nopol 時凍結 pol
        ),
        CurriculumPhase(
            name="Phase 2: Mixed",
            start_progress=0.2, end_progress=0.6,
            pol_ratio=0.6, nopol_ratio=0.4,
            freeze_pol_on_nopol=False,
        ),
        CurriculumPhase(
            name="Phase 3: Polish",
            start_progress=0.6, end_progress=1.0,
            pol_ratio=0.7, nopol_ratio=0.3,
            freeze_pol_on_nopol=False,
        ),
    ]

    def get_batch_info(self, step):
        phase = self._get_current_phase(step)
        return {
            'phase_name': phase.name,
            'pol_ratio': phase.pol_ratio,
            'freeze_pol_on_nopol': phase.freeze_pol_on_nopol,
        }
```

### 5.2 PolModuleFreezer

```python
class PolModuleFreezer:
    """凍結 pol 相關參數"""

    POL_KEYWORDS = ['pol', 'polarization']

    def freeze(self):
        for name, param in self.model.named_parameters():
            if any(k in name.lower() for k in self.POL_KEYWORDS):
                param.requires_grad = False

    def unfreeze(self):
        for name, param in self.model.named_parameters():
            if any(k in name.lower() for k in self.POL_KEYWORDS):
                param.requires_grad = True
```

### 5.3 BackboneFreezer

```python
class BackboneFreezer:
    """凍結 backbone (fnet, cnet)"""

    # 注意: 不凍結 update_block！
    # PolCorrBlock 是非參數的，如果凍結 update_block 就沒有參數可學習
    BACKBONE_KEYWORDS = ['fnet', 'cnet']  # 不包含 update_block

    def freeze(self):
        for name, param in self.model.named_parameters():
            if any(k in name for k in self.BACKBONE_KEYWORDS):
                param.requires_grad = False
```

### 5.4 V2PolModuleFreezer

```python
class V2PolModuleFreezer:
    """V2 系列專用凍結器"""

    # 包含 V2 新增的模組
    V2_POL_KEYWORDS = [
        'pol', 'polarization',
        'pol_residual',      # V2-A
        'pol_gate',          # V2-D
        'pol_weight',        # V2-E
        'gating_net',        # V2-C
    ]
```

---

## 6. 訓練命令範例

### 6.1 V2-D (Disparity-Aware Gate)

```bash
python train_pids.py \
    --data_dir ./PIDS_dataset_exp29_mixed/data \
    --output_dir ./checkpoints_exp38_v2d \
    --pol_volume_v2d \
    --pol_gate_hidden 8 \
    --pol_alpha 0.2 \
    --curriculum \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 60000 --lr 0.0003 --iters 24 \
    --bf16
```

### 6.2 V2-E (Pre-Corr Weighting)

```bash
python train_pids.py \
    --data_dir ./PIDS_dataset_exp29_mixed/data \
    --output_dir ./checkpoints_exp39_v2e \
    --pol_volume_v2e \
    --pol_weight_hidden 8 \
    --curriculum \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 60000 --lr 0.0003 --iters 24 \
    --bf16
```

### 6.3 混合訓練 (Curriculum Learning)

```bash
# 組織混合數據集
python organize_mixed_dataset.py \
    --pol_scenes_file ./train_scenes.txt \
    --pol_data_dir ./dataset_pol \
    --nopol_data_dir ./dataset_nopol \
    --output_dir ./mixed_dataset \
    --train_pol_count 3000 \
    --nopol_count 2000 \
    --val_count 600

# 訓練
python train_pids.py \
    --data_dir ./mixed_dataset/data \
    --output_dir ./checkpoints_curriculum \
    --pol_volume_v2e \
    --curriculum \
    --curriculum_strategy default \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 60000
```

---

## 附錄: 關鍵參數說明

### A.1 架構參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--hidden_dim` | 128 | GRU hidden dimension |
| `--context_dim` | 128 | Context encoder output |
| `--feature_dim` | 128 | Feature encoder output |
| `--corr_levels` | 4 | Correlation pyramid levels |
| `--corr_radius` | 4 | Correlation lookup radius |
| `--iters` | 12-24 | GRU iterations |

### A.2 V2 專用參數

| 參數 | 版本 | 說明 |
|------|------|------|
| `--residual_hidden_dim` | V2-A/B/C | Residual network hidden dim |
| `--residual_init_scale` | V2-A/B/C | Initial residual scale (0.1) |
| `--gating_hidden_dim` | V2-C | Gating network hidden dim |
| `--pol_gate_hidden` | V2-D | 3D gate hidden channels |
| `--pol_alpha` | V2-D | Modulation strength |
| `--pol_weight_hidden` | V2-E | Weight network hidden channels |

### A.3 訓練參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--lr` | 0.0003 | Learning rate |
| `--batch_size` | 8 | Batch size |
| `--num_steps` | 60000 | Total training steps |
| `--glass_weight` | 5.0 | Glass region weight |
| `--strict_glass_weight` | 0.5 | Strict glass extra weight |
| `--bf16` | False | Use BFloat16 precision |
| `--freeze_fnet` | False | Freeze feature encoder |

### A.4 Curriculum 參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--curriculum` | False | Enable curriculum learning |
| `--curriculum_strategy` | 'default' | Strategy: default/freeze_backbone/gradual_unfreeze |
| `--curriculum_seed` | 42 | Random seed for sampler |

### A.5 DID 參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--did` | False | Enable Directional Impulse Descent |
| `--did_epsilon_pct` | 0.10 | Train slope threshold percentile |
| `--did_train_window` | 300 | Train slope EMA window |
| `--did_val_m` | 5 | Val oscillation window (cycles, 輔助證據) |
| `--did_delta` | 0.5 | Val EPE tolerance (px) |
| `--did_best_stale` | 4 | Cycles since best before trigger **[改進 D]** |
| `--did_gamma_up` | 3.0 | Impulse LR multiplier (2nd+ trigger) |
| `--did_gamma_up_first` | 2.0 | Impulse LR multiplier (1st trigger) **[改進 B]** |
| `--did_gamma_backbone` | 1.2 | Backbone impulse multiplier **[改進 C]** |
| `--did_gamma_down` | 0.1 | Cooldown LR multiplier |
| `--did_t_impulse` | 200 | Impulse duration (steps) |
| `--did_t_cool` | 100 | Cooldown duration (steps) |
| `--did_cooldown_mode` | 'relative_base' | Cooldown reference: relative_base/relative_impulse |
| `--did_max_triggers` | 3 | Max triggers per training |
| `--did_protection` | 3 | Protection period (val cycles) |
| `--did_late_lock` | 0.75 | Disable after 75% progress |

---

## 7. Directional Impulse Descent (DID)

### 7.1 概述

DID 是一個**狀態驅動**的 LR 控制機制，與 OneCycle 協同工作：

```
OneCycle = 全局退火曲線 (reference trajectory)
DID      = 狀態驅動的局部脈衝 (event-driven impulse disturbance)
```

### 7.2 狀態機

```
┌─────────┐    trigger    ┌─────────┐    T_impulse    ┌──────────┐
│ NORMAL  │──────────────►│ IMPULSE │───────────────►│ COOLDOWN │
└─────────┘               └─────────┘                └──────────┘
     ▲                                                    │
     │                     ┌───────────┐                  │ T_cool
     │         P cycles    │ PROTECTED │◄─────────────────┘
     └─────────────────────┴───────────┘
```

### 7.3 DirectionalImpulseDescent 類別

```python
class DirectionalImpulseDescent:
    """狀態驅動的局部脈衝控制器"""

    # 狀態常數
    NORMAL = 0
    IMPULSE = 1
    COOLDOWN = 2
    PROTECTED = 3

    def update_train(self, loss: float, step: int):
        """每個 training step 更新 train slope 追蹤"""
        # 計算相對斜率 s_t = |ΔL| / L
        # 維護 EMA 和歷史

    def update_val(self, val_glass_epe: float, step: int, total_steps: int) -> dict:
        """每次 validation 後更新狀態機並檢查觸發"""
        # Spike filtering
        # Moving-best tracking [改進 D]
        # 狀態轉換
        # 觸發條件檢查

    def get_lr_multiplier(self, lr_base: float, step: int, is_backbone: bool = False) -> float:
        """計算當前 LR 乘數 [改進 B, C]"""
        # NORMAL/PROTECTED: 1.0
        # IMPULSE (backbone): γ_backbone (1.2)
        # IMPULSE (pol/update, 1st): γ_up_first (2.0)
        # IMPULSE (pol/update, 2nd+): γ_up (3.0)
        # COOLDOWN: γ_down (or relative_impulse mode)
```

### 7.4 設計改進 (v2)

**改進 B: 漸進式 Impulse**
- 第一次觸發: `gamma_up_first` (×2.0)
- 後續觸發: `gamma_up` (×3.0)
- 理由: 避免 bf16 混合精度下 grad overflow

**改進 C: 參數群分離**
- Pol/Update modules: 完整 impulse
- Backbone (fnet/cnet): 輕微 impulse (×1.2)
- 理由: 保護已收斂的 ImageNet 特徵

**改進 D: 觸發條件重構**
- 主條件: moving-best staleness (連續 4 cycles 沒更新 best)
- 輔助: val oscillation (有則更確定，沒有也可觸發)
- 理由: pol-only 驗證集較「安靜」

### 7.5 整合點

1. **train_epoch**: 每 step 呼叫 `did.update_train(loss)`
2. **train_epoch**: `scheduler.step()` 後應用 `did.get_lr_multiplier(is_backbone)`
3. **_validate_and_log_internal**: 呼叫 `did.update_val()` + TensorBoard logging

### 7.6 TensorBoard 記錄

```
did/state              # 0=NORMAL, 1=IMPULSE, 2=COOLDOWN, 3=PROTECTED
did/trigger_count      # 已觸發次數
did/train_slope_ema    # train slope 的 EMA
did/val_slope_ema      # val slope 的 EMA
did/epsilon            # 當前 threshold
did/lr_base            # OneCycle 的 LR
did/lr_effective       # 應用 DID 後的 LR
did/lr_multiplier      # 當前乘數
did/val_best           # current best val EPE
did/cycles_since_best  # cycles since last best update
```

### 7.7 使用範例

```bash
python train_pids.py \
    --data_dir ./data \
    --output_dir ./checkpoints/exp40_did \
    --pol_volume_v2e \
    --did \
    --did_gamma_up 3.0 \
    --did_gamma_down 0.1 \
    --did_t_impulse 200 \
    --did_max_triggers 3 \
    --curriculum \
    --num_steps 100000 \
    --bf16
```

---

*最後更新: 2026-01-29*
