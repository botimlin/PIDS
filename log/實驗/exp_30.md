## Exp #30: Gradual Unfreezing with Continuous LR (訓練中)

**日期**: 2026-01-26 (設計 + 實作)
**目標**: 優化 Exp #29 的 phase transition，實現更平滑的 backbone 解凍

### 動機

**Exp #29 觀察到的問題**:
1. Phase 1 後期出現 mini-plateau（~16000 steps 低效訓練）
2. Phase 2 開始時可能有 loss jump（phase boundary shock）
3. 突然解凍 backbone 可能導致訓練不穩定

### 方案比較

| 方案 | 說明 | 優缺點 | 結論 |
|------|------|--------|------|
| **A: Layer-wise** | cnet → fnet → 逐層解凍 | ❌ fnet/cnet 耦合，單獨解凍 cnet 可能震盪 | 作為 Ablation |
| **B: 階段 LR** | 分段設定 backbone_lr_multiplier | ✅ 穩健、低風險 | 備案 |
| **C: 連續 LR** | backbone LR 從 0 連續增長到 base_lr | ✅ 最平滑、最符合研究假說 | **主線** |

### 方案 A 風險分析（不推薦作為主線）

RAFT-Stereo 架構特性：
```
fnet 與 cnet 並不是獨立語義模組
它們共同定義了 cost volume + recurrent state
```

單獨解凍 cnet 但 fnet frozen 的風險：
- context 想修正，但 feature space 動不了
- 可能造成短期 loss 震盪
- **結論**: 適合作為 Ablation (Exp #31-A)，不適合主線

### 數據驅動參數分析（2026-01-26 更新）

基於 Exp #29 實際訓練數據（train_glass_epe + val_glass_epe）分析 plateau 時機：

**Train/Glass_EPE 變異壓縮觀察**:
| 階段 | Step 範圍 | EPE 範圍 | 均值估計 |
|------|-----------|----------|----------|
| 初期 | 0-5000 | 20~116px | ~50px |
| 中期 | 5000-10000 | 15~98px | ~35px |
| 後期 | 10000-15000 | 10~60px | ~25px |
| 平台期 | 15000-20000 | 10~50px | ~22px |
| 現在 | 20000-29000 | 8~50px | ~18-20px |

**Val/Glass_EPE 關鍵轉折點**:
```
Step 10000: 33.39px  ← 減速開始
Step 13500: 24.86px  ← 進入平台
Step 17000: 22.58px  ← 局部最低
Step 18000: 30.88px  ← 跳升（Phase 1 極限信號）
Step 29000: 19.79px  ← 目前最佳
```

**結論**: Plateau 開始於 Step 10000~12000（16.7%~20%），應提早開始解凍。

### 方案 C 設計（主線）- 5 階段 LR Schedule

**核心理念**:
> "backbone 不是「解凍」，而是「被允許輕聲說話」"
> "尾端低 LR 精修能帶來個位數 px 的提升"（經驗證實）

**視覺化**:
```
Progress:  0%     20%      45%        80%   85%    100%
           ├──────┼────────┼──────────┼─────┼──────┤
BB mult:   0      0→1      1          1     1      1
Base LR:   1x     1x       0.5x       0.2x  0.1x   0.05x
           ├──────┼────────┼──────────┼─────┼──────┤
           [凍結]  [解凍]   [主學習]   [過渡] [精修]
```

**5 階段說明**:
| 階段 | Progress | Steps | Backbone | LR | 目的 |
|------|----------|-------|----------|-----|------|
| 凍結 | 0-20% | 0-12000 | Frozen | 1.0x | 學幾何基礎 |
| 解凍 | 20-45% | 12000-27000 | 0→1 | 1.0x | 學 pol features |
| 主學習 | 45-80% | 27000-48000 | 1.0 | 0.5x | 突破 plateau |
| 過渡 | 80-85% | 48000-51000 | 1.0 | 0.2x | 穩定過渡 |
| 精修 | 85-100% | 51000-60000 | 1.0 | 0.1x | 細節微調 |

**完整 LR Schedule 函數**:
```python
def get_lr_config(progress, base_lr):
    """
    完整 5 階段 LR Schedule (Exp #30 v2)

    設計依據：
    - 數據驅動：Exp #29 顯示 plateau 始於 Step 10000-12000
    - 經驗法則：尾端低 LR 精修能帶來個位數 px 提升
    - 避免空轉：主學習區用 0.5x LR 持續探索
    """

    # === Backbone multiplier ===
    if progress < 0.20:
        backbone_mult = 0.0                      # 凍結
    elif progress < 0.45:
        backbone_mult = (progress - 0.20) / 0.25  # 0→1 漸進解凍
    else:
        backbone_mult = 1.0                      # 全開

    # === Base LR decay ===
    if progress < 0.45:
        lr_mult = 1.0           # 前段保持 base LR
    elif progress < 0.80:
        lr_mult = 0.5           # 主學習區：半速探索
    elif progress < 0.85:
        lr_mult = 0.2           # 過渡期
    else:
        lr_mult = 0.1           # 尾端精修

    return base_lr * lr_mult, backbone_mult
```

**與原設計比較**:
| 參數 | 原設計 (v1) | 新設計 (v2) | 差異說明 |
|------|-------------|-------------|----------|
| 解凍起點 | 30% | 20% | 數據顯示 plateau 更早 |
| 完全解凍 | 60% | 45% | 減少空轉時間 |
| 主學習區 LR | 1.0x | 0.5x | 避免過度震盪 |
| 尾端精修 | 無特別設計 | 0.1x (9000 steps) | 經驗證實有效 |
| 階段數 | 3 階段 | 5 階段 | 更精細控制 |

### 方案 C 的三個關鍵優勢

**① 沒有 phase boundary shock**
- 不會在 step = X 發生 loss jump
- 表徵空間是連續演化

**② pol 特徵「自然滲透」**
- 一開始 pol 只影響 update_block
- 隨 backbone LR ↑，pol 開始影響 fnet / cnet
- 這個過程可被 EPE 曲線直接觀察

**③ 論文敘述友好**
> *"We gradually relax the backbone freezing constraint by continuously increasing its learning rate, allowing polarization cues to be progressively integrated into the feature representation without destabilizing geometric learning."*

### 實作需求

**1. Optimizer 改用 param_groups**:
```python
optimizer = torch.optim.AdamW([
    {'params': model.fnet.parameters(), 'lr': base_lr},        # group 0: fnet
    {'params': model.cnet.parameters(), 'lr': base_lr},        # group 1: cnet
    {'params': model.update_block.parameters(), 'lr': base_lr}, # group 2: update
    {'params': model.pol_corr.parameters(), 'lr': base_lr},     # group 3: pol
])
```

**2. 每個 step 動態調整 LR（5 階段版本）**:
```python
def adjust_lr(optimizer, progress, base_lr):
    """
    同時調整 backbone multiplier 和 base LR decay
    """
    current_lr, backbone_mult = get_lr_config(progress, base_lr)

    # Backbone (fnet, cnet): current_lr * backbone_mult
    optimizer.param_groups[0]['lr'] = current_lr * backbone_mult  # fnet
    optimizer.param_groups[1]['lr'] = current_lr * backbone_mult  # cnet

    # Head (update_block, pol_corr): current_lr
    optimizer.param_groups[2]['lr'] = current_lr  # update_block
    optimizer.param_groups[3]['lr'] = current_lr  # pol_corr
```

**3. Training loop 整合**:
```python
for step in range(total_steps):
    progress = step / total_steps
    adjust_lr(optimizer, progress, base_lr)

    # ... training code ...

    # Logging (optional)
    if step % log_interval == 0:
        bb_lr = optimizer.param_groups[0]['lr']
        head_lr = optimizer.param_groups[2]['lr']
        print(f"Step {step} | BB LR: {bb_lr:.6f} | Head LR: {head_lr:.6f}")
```

### 預期效果

| 指標 | Exp #29 (Hard Switch) | Exp #30 (Gradual) |
|------|----------------------|-------------------|
| Phase 1 plateau | ~16k steps | 自動縮短 |
| Phase 2 loss jump | 可能有 | 無 |
| 訓練曲線 | 階梯狀 | 平滑 |
| 論文品質 | 中 | 高 |

### 實作修改 (2026-01-26)

1. **新增 `GradualUnfreezeLRScheduler` 類別** (`curriculum_sampler.py`)
   - 實現 5-phase LR schedule
   - 支援 `state_dict()` / `load_state_dict()` 用於 checkpoint
   - 每個 step 返回 `(lr_mult, backbone_mult)`

2. **修改 optimizer 使用 4 param groups** (`train_pids.py`)
   - Group 0: fnet (backbone)
   - Group 1: cnet (backbone)
   - Group 2: update_block (head)
   - Group 3: pol_corr (head)

3. **新增 TensorBoard logging**
   - `backbone_lr`, `head_lr`, `backbone_mult`
   - 可觀察解凍進度

**訓練命令**:
```bash
nohup python train_pids.py \
    --data_dir ./PIDS_dataset_exp29_mixed/data \
    --output_dir ./checkpoints_exp30_gradual_unfreeze \
    --pol_volume \
    --curriculum \
    --curriculum_strategy gradual_unfreeze \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 60000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 500 --num_workers 4 \
    > train_exp30_gradual_unfreeze.log 2>&1 &
```

### 訓練進度 (Step 12600, ~21%)

**Phase 2 開始確認**: Step 12000 (20%) 開始解凍
```
Step 12000: BB_mult = 0.00 → 漸進上升
Step 12600: BB_mult = 0.04 (4% 解凍)
```

**Validation Metrics**:
```
Step   500:  89.50 px
Step  5000:  37.29 px
Step 10000:  33.73 px  ← Phase 1 plateau 開始
Step 12000:  30.07 px  ← Phase 2 開始
Step 12600:  ~26 px    ← 目前（仍在解凍中）
```

**關鍵觀察**:
1. ✅ **無 phase boundary shock**: 解凍過程平滑，無 loss jump
2. ✅ **BB_mult 漸進上升**: 0.00 → 0.04 → ... → 1.00
3. ⚠️ **Phase 1 也有 mini-plateau**: 與 Exp #29 類似，但解凍更早

**與 Exp #29 同期比較**:
| Step | Exp #29 (Hard) | Exp #30 (Gradual) | 差異 |
|------|---------------|-------------------|------|
| 500 | 57.38 px | 89.50 px | Exp #30 起始較高 |
| 10000 | 33.39 px | 33.73 px | 相當 |
| 12000 | ~30 px | 30.07 px | 相當（Exp #30 開始解凍）|

**最終結果 (Step 60000)**:
```
[Val @ Step 60000]
┌─────────────────────────────────────────────────────────────────┐
│ Mode       │ Glass EPE │ EPE    │ D1     │ Composite │ Loss     │
├─────────────────────────────────────────────────────────────────┤
│ Oracle     │     6.478 │  3.772 │ 24.81% │     8.958 │   5.4540 │
│ Real       │     6.478 │  3.772 │ 24.81% │     8.958 │   5.4540 │
└─────────────────────────────────────────────────────────────────┘
Best Glass EPE: 6.235 px
Train Glass EPE: 4.234 px
```

**Exp #29 vs #30 比較**:
| 策略 | Val Best Glass EPE | 特點 |
|------|-------------------|------|
| Exp #29 Hard Switch | 8.381 px | 有 phase boundary shock |
| Exp #30 Gradual | 6.235 px | 平滑過渡，**-25.6% 更好** |

**結論**: Gradual Unfreezing 優於 Hard Switch，但兩者都不如簡單訓練 (Exp #24 Val ~4.87 px)

**狀態**: ✅ 訓練完成

---

## 重大發現：測試集差異導致性能誤判 (2026-01-26)

### 問題背景

在進行多輪實驗後，發現後期實驗 (Exp #27-30) 的 Glass EPE 似乎都比 Exp #24 (2.44 px) 差。
經過深入分析，發現這是**測試集不同**造成的不公平比較。

### 關鍵發現

**Exp #24 TensorBoard Log 分析**:
- Val Glass EPE @ Step 60000: **~4.87 px** (不是 2.44 px!)
- 原先報告的 2.44 px 來自**較簡單/較小的舊評估集 (~200 場景)**

**公平重新評估** (新測試集 857 場景):

| 實驗 | 架構 | 舊 Eval (~200 scenes) | 新 Test (857 scenes) |
|------|------|----------------------|---------------------|
| Exp #22 | Baseline RAFT-Stereo | Val: 7.94 px | **93.96 px** ❌ |
| Exp #24 | Polarization Volume | 2.44 px | **4.055 px** ✅ |

### Polarization Volume 提升幅度 (公平比較)

| 指標 | Baseline (Exp #22) | Pol Volume (Exp #24) | 提升 |
|------|-------------------|---------------------|------|
| **Glass EPE** | 93.958 px | 4.055 px | **↓ 95.7%** (23.2x) |
| **Overall EPE** | 75.238 px | 2.646 px | **↓ 96.5%** (28.4x) |
| **Glass D1** | 88.81% | 33.54% | **↓ 55.3 pp** |
| **BG EPE** | 57.531 px | ~1.5 px | **↓ 97.4%** |

### 結論

1. **測試集差異是「性能退化」假象的主因**
   - 舊評估集 (~200 場景) 較簡單
   - 新測試集 (857 場景) 更多樣化、更困難

2. **Baseline 在困難場景完全失效**
   - Val 時 7.94 px → 困難測試集 93.96 px
   - 證明純幾何立體匹配無法處理透明物體

3. **Polarization Volume 保持魯棒性**
   - Val ~4.87 px → 測試集 4.055 px (甚至更好)
   - 偏振資訊不只提升性能，更保證穩定性

4. **論文可用數據**:
   - Polarization Volume 相比 Baseline 在 Glass 區域誤差降低 **95.7%**
   - 這是強有力的消融實驗證據

### 後續行動

- [x] Exp #22 (Baseline) 在新測試集評估 ✓
- [x] Exp #24 (Pol Volume) 在新測試集評估 ✓
- [ ] Exp #30 (Gradual Unfreeze) 在新測試集評估
- [ ] 更新論文數據使用統一測試集結果

---

## Nopol Test Set 補充評估 (2026-01-26)

### 問題：Pol vs Nopol 影像的公平比較

之前的比較是 Baseline 在 **pol 影像** (left_parallel, right_cross) 上測試。
但 Baseline 從未見過偏振影像，可能因強度差異而崩潰，不是公平比較。

### Nopol Test Set 建立

從 `dataset_nopol_final` 提取與 pol test set 相同的 857 個場景：
```bash
mkdir -p ./dataset_nopol_test/{stereo_pairs,ground_truth,masks}
# 複製 left.exr, right.exr, disparity.exr, glass_mask.exr
```

### 評估結果

**Exp #22 (Baseline) 在 nopol test set (859 scenes)**:
```
[Overall Metrics]
  EPE:  3.444 ± 3.369 px
  D1:   41.24 ± 17.96 %

[Glass Region Metrics]
  Glass EPE:  6.543 ± 14.607 px
  Glass D1:   75.42 ± 21.41 %

[Background Region Metrics]
  BG EPE:  2.798 ± 2.706 px
```

### 完整比較表

| 測試條件 | Baseline (Exp #22) | Pol Volume (Exp #24) | 差距 |
|----------|-------------------|---------------------|------|
| **pol 影像測試** | 93.96 px ❌ | 4.055 px ✅ | **95.7% ↓** |
| **nopol 影像測試** | 6.543 px | N/A (無偏振資訊) | - |

### 問題分析

1. **Baseline 在 nopol 上表現尚可 (6.543 px)**
   - 無偏振干擾時，純幾何立體匹配可以運作
   - 但玻璃區域仍是難題

2. **Pol Volume 優勢不夠明顯**
   - pol test: 4.055 px vs baseline nopol: 6.543 px
   - 僅 **38% 改善**，不夠有說服力
   - 需要架構優化來拉開差距

3. **結論：目前 4 px 的性能不足以展示偏振的核心價值**

---

## Polarization Volume V2 架構規劃 (2026-01-26)

### 現況問題

目前 PolCorrBlock 架構過於簡單：
```
pol_diff = left - right  (simple subtraction)
pol_corr = query(pol_volume, disp)
output = concat(corr, pol_corr, disp) → encoder → GRU
```

**缺陷**：
1. 無 learnable 偏振特徵提取
2. 無空間注意力機制（不知道哪裡是玻璃）
3. 簡單 concat 融合，沒有學習 stereo vs pol 的權重

### 架構優化方案

#### 主線 (A + C)：Polarization Attention + Gated Fusion

```
                     ┌─────────────────────────────────────────────┐
                     │           Polarization Attention            │
                     │                                             │
pol_diff ──────────→ │ [Conv] → sigmoid → pol_attention_map       │
                     │                     (where is glass?)       │
                     └──────────────────────┬──────────────────────┘
                                            ↓
                     ┌─────────────────────────────────────────────┐
                     │              Gated Fusion                   │
                     │                                             │
corr ───────────────→│                                             │
                     │  gate = σ(W_g * [corr, pol_corr, attn])    │
pol_corr ───────────→│                                             │──→ fused
                     │  fused = gate * enhanced_corr              │
pol_attention_map ──→│        + (1-gate) * enhanced_pol           │
                     │                                             │
                     └─────────────────────────────────────────────┘
                                            ↓
                            [concat(fused, disp)] → GRU
```

**方案 A: Polarization Attention (必做)**
- 從 pol_corr 生成空間注意力圖
- 高偏振差異區域（玻璃）獲得更高權重
- 讓模型知道「哪裡該信任偏振資訊」

**方案 C: Gated Fusion (必做)**
- 學習 stereo 與 pol 特徵的融合權重
- gate = 0 → 信任 pol，gate = 1 → 信任 stereo
- 玻璃區域自動降低 stereo 權重，提高 pol 權重

#### 能力放大器 (B)：Learnable Pol Encoder

```
pol_volume ──→ [Conv Encoder] ──→ deeper pol_features
```
- 不只是 raw subtraction
- 學習更豐富的偏振特徵表示

#### 論文加分項 (D)：Glass-aware Auxiliary Head

```
hidden ──→ [Glass Head] ──→ predicted_glass_mask (auxiliary loss)
```
- 預測玻璃位置作為輔助任務
- 可用於視覺化和解釋性

### 核心模組設計

```python
class PolarizationAttention(nn.Module):
    """從 pol_diff 生成空間注意力圖"""
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, pol_corr):
        return self.conv(pol_corr)  # (B, 1, H, W)


class GatedFusion(nn.Module):
    """學習 stereo vs pol 的融合權重"""
    def __init__(self, corr_dim, pol_dim, out_dim):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Conv2d(corr_dim + pol_dim + 1, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        self.corr_enhance = nn.Conv2d(corr_dim, out_dim, 1)
        self.pol_enhance = nn.Conv2d(pol_dim, out_dim, 1)

    def forward(self, corr, pol_corr, pol_attn):
        gate = self.gate_net(torch.cat([corr, pol_corr, pol_attn], dim=1))
        corr_feat = self.corr_enhance(corr)
        pol_feat = self.pol_enhance(pol_corr)
        fused = gate * corr_feat + (1 - gate) * pol_feat
        return fused, gate
```

### 預期改善

| 版本 | Glass EPE | 改善 |
|------|-----------|------|
| Baseline (nopol) | 6.543 px | - |
| Pol Volume V1 | 4.055 px | 38% ↓ |
| **Pol Volume V2 (目標)** | **< 2.5 px** | **> 60% ↓** |

### 實作計劃

- [ ] **Exp #31: Pol Volume V2 (A+C)**
  - [ ] Step 1: 實現 PolarizationAttention 模組
  - [ ] Step 2: 實現 GatedFusion 模組
  - [ ] Step 3: 建立 UpdateBlockV2
  - [ ] Step 4: 建立 PIDSStereoPolVolumeV2
  - [ ] Step 5: 訓練並評估
- [ ] (Optional) Exp #32: 加入 Learnable Pol Encoder (B)
- [ ] (Optional) Exp #33: 加入 Glass-aware Auxiliary Head (D)

**狀態**: 📋 架構設計完成，待實作

---

## Stage 2: Real-World Fine-Tuning (Kinect v1 替身法)

**日期**: 2026-01-25 (規劃)
**目標**: 使用真實世界數據微調模型，提升實際部署性能

### 替身法 (Substitute Method)

解決 Kinect 無法感測透明物體的問題：

```
拍攝流程:
┌─────────────────────────────────────────────────────────┐
│ Step 1: 偏振相機拍攝「真實玻璃」場景                      │
│         → left_parallel.exr, right_cross.exr            │
│         → 訓練輸入                                       │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Step 2: 用不透明替身替換玻璃 (相同形狀位置)               │
│         → 3D 列印 / 紙板 / 不透明壓克力                  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Step 3: Kinect v1 拍攝「替身」場景                       │
│         → depth.png (GT 深度)                           │
│         → 替身對 Kinect 結構光有效                       │
└─────────────────────────────────────────────────────────┘
```

### Kinect v1 規格

| 項目 | 規格 |
|------|------|
| 深度原理 | 結構光 (IR pattern projection) |
| 深度解析度 | 640x480 |
| RGB 解析度 | 640x480 |
| 深度範圍 | 0.8m ~ 4.0m |
| 深度精度 | ~2-4cm @ 2m |
| 優點 | 便宜、易得、室內效果好 |

### 數據採集設備

```
PIDS 採集系統:
├── 偏振相機 (IMX296 x2, 同步)
│   ├── 左: 0° 偏振片 (I∥)
│   └── 右: 90° 偏振片 (I⊥)
│
├── Kinect v1 (GT 深度)
│   └── 需與偏振相機校正對齊
│
└── 替身物體
    ├── 與玻璃相同外形
    └── 不透明材質 (白色/灰色最佳)
```

### 校正需求

1. **偏振相機立體校正**: 已完成 (Stage 1)
2. **Kinect 內參校正**: 使用 OpenCV/ROS 工具
3. **Kinect-偏振相機外參校正**:
   - 棋盤格同時可見於兩系統
   - 計算座標轉換矩陣
4. **深度對齊**: Kinect 深度 → 偏振相機視角

### 數據格式

```
real_world_dataset/
├── scene_0001/
│   ├── left_parallel.exr    # 偏振相機 (玻璃場景)
│   ├── right_cross.exr      # 偏振相機 (玻璃場景)
│   ├── depth_kinect.png     # Kinect 原始深度 (替身場景)
│   ├── depth_aligned.exr    # 對齊到偏振相機的深度
│   ├── glass_mask.png       # 手動/自動標註的玻璃區域
│   └── params.json          # 場景參數
├── scene_0002/
│   └── ...
```

### 訓練策略

```
Stage 2 Fine-tuning:
├── 基礎模型: Stage 1 最佳 checkpoint (Exp #27 或 #28)
├── 學習率: 較小 (0.00003 ~ 0.0001)
├── 數據量: ~100-500 真實場景
├── 目標: 域適應 (Synthetic → Real)
│   ├── 真實光照
│   ├── 真實紋理
│   ├── 相機噪聲
│   └── 偏振特性差異
```

### 預期改善

| 指標 | Stage 1 Only | Stage 2 Fine-tuned |
|------|--------------|-------------------|
| 合成數據 Glass EPE | ~2.0-2.5 px | ~2.0-2.5 px (維持) |
| 真實世界 Glass EPE | 可能較差 | 預期大幅改善 |
| 部署可靠性 | 中等 | 高 |

**狀態**: 📋 規劃中 (待 Stage 1 完成)

---

## 16. 待完成項目

- [x] 完成 70k steps 訓練 ✓
- [x] 新增 strict glass mask 功能 ✓
- [x] Nopol V5 數據集渲染 ✓
- [x] Nopol strict glass mask 渲染 ✓
- [x] Nopol V5 消融實驗訓練 ✓
- [x] Exp #21: Strict Glass Weight 驗證 ✓
- [x] Test Set 評估 (Pol vs Nopol 消融) ✓
- [x] 驗證邏輯改進: Oracle vs Real 雙重驗證 ✓
- [x] 驗證 Oracle-Real Gap 在 Pol vs Nopol 的差異 ✓ (Pol Real 86.7% 優於 Nopol)
- [x] Training-Inference Consistency 優化實作 ✓
- [x] train_baseline.py 乾淨版本建立 ✓
- [x] Bug Fix: Validation Loss 計算不一致 ✓
- [x] Exp #22: Baseline RAFT-Stereo 訓練 ✓ (Glass EPE: 7.52 px)
- [x] Exp #23: Curriculum Learning 訓練 ✗ (失敗，Gap 從 103% 增到 652%)
- [x] Polarization Volume 架構設計與實作 ✓
- [x] Exp #24: Polarization Volume 訓練 ✓ (**Glass EPE: 2.44 px, -67.6% vs Baseline**)
- [x] Exp #25: Learnable Pol Volume ✗ (無顯著改善，Glass EPE 2.48 px)
- [x] pids_qa.py v2.3.0: 新增 C3 玻璃偏振檢查 + 輸出 passed/failed_scenes.txt ✓
- [x] Exp #26: LR 影響分析 ✓ (LR=0.0001 導致 +120% 退化，確認 LR=0.0003 為正確配置)
- [x] **Exp #27: 嚴格 QA 100% 訓練** ✅ (Glass EPE 5.03px, 比 Exp #24 差 +106%)
- [ ] Exp #27: 5%/10%/25%/50%/75% 數據量消融 (已證明嚴格 QA 有害，暫緩)
- [x] Exp #27: 評估 C3 嚴格過濾對 Glass EPE 的影響 ✅ (結論：嚴格過濾導致性能下降)
- [x] **Exp #28: 混合 Pol/Nopol 訓練** ✅ (Glass EPE 4.88px, 結論：Raw data 最佳)
  - [x] Step 1: 生成新 OBJ (scene_15001+, 3000 個場景) ✓
  - [x] Step 2: 渲染 Nopol 數據 (18000 場景完成) ✓
  - [x] Step 3: 修改 PIDSSyntheticDataset 支援雙命名 + data_type flag ✓
  - [x] Step 4: 實現 CurriculumSampler (動態 pol/nopol 比例 + Pol Module 凍結) ✓
  - [x] Step 5: 修改 train_pids.py 分開追蹤 pol/nopol metrics + --curriculum flag ✓
  - [x] Step 6: 寫 organize_mixed_dataset.py (支援 organized + flat 結構) ✓
  - [x] Step 7: 組織混合數據集 ✓ (Train: 5000, Val: 600)
  - [x] Step 8: 訓練 Exp #28 ✓
  - [x] Step 9: 評估 Exp #28 ✓ (弱偏振 ≠ nopol，raw data 最佳)
- [x] **Exp #29: Freeze Backbone First** ✅ (完成，Best Glass EPE: 8.381 px)
  - [x] Step 1: 實現 BackboneFreezer 類別 ✓
  - [x] Step 2: 新增 FREEZE_BACKBONE_PHASES 配置 (70/30 pol/nopol) ✓
  - [x] Step 3: 更新 curriculum_sampler.py ✓
  - [x] Step 4: 修改 train_pids.py 支援 freeze_backbone 策略 ✓
  - [x] Step 5: 修正 BackboneFreezer 只凍結 fnet/cnet (保留 update_block) ✓
  - [x] Step 6: 組織混合數據集 (Train 5000, Val 600) ✓
  - [x] Step 7: 訓練 Exp #29 完成 ✓ (60000 steps)
  - [ ] Step 8: 評估並比較 (待執行)
- [x] **Exp #30: Gradual Unfreezing** ✅ (完成，Best Glass EPE: 6.235 px)
  - [x] Step 1: 實現 GradualUnfreezeLRScheduler 類別 ✓
  - [x] Step 2: 修改 optimizer 使用 4 param_groups (fnet, cnet, update_block, pol) ✓
  - [x] Step 3: 實現 5-phase LR schedule ✓
  - [x] Step 4: 更新 train_pids.py 支援 gradual_unfreeze 策略 ✓
  - [x] Step 5: 修復 state_dict() 缺失問題 ✓
  - [x] Step 6: 訓練 Exp #30 完成 ✓ (60000 steps, Best Val Glass EPE: 6.235 px)
  - [x] Step 7: 比較 Exp #29 vs #30 ✓ (Gradual 6.235 px 優於 Hard Switch 8.381 px)
- [x] **測試集公平重新評估** ✅ (2026-01-26)
  - [x] Exp #22 (Baseline) 在新測試集 (857 scenes) 評估 ✓ → **Glass EPE: 93.96 px** (pol test)
  - [x] Exp #22 (Baseline) 在 nopol test 評估 ✓ → **Glass EPE: 6.543 px**
  - [x] Exp #24 (Pol Volume) 在新測試集評估 ✓ → **Glass EPE: 4.055 px**
  - [x] 分析：Pol Volume vs Baseline (nopol) 僅 38% 改善，**不夠有說服力**
  - [ ] Exp #30 (Gradual Unfreeze) 在新測試集評估 (已跳過，預期與 Val 接近)
- [x] **Exp #31: Polarization Volume V2 (架構優化)** ✅ (完成，Best Glass EPE: 7.763 px - 退步)
  - [x] Step 1: 實現 PolarizationAttention 模組 ✓
  - [x] Step 2: 實現 GatedFusion 模組 ✓
  - [x] Step 3: 建立 UpdateBlockV2 ✓
  - [x] Step 4: 建立 PIDSStereoPolVolumeV2 ✓
  - [x] Step 5: 修復 ContextEncoder API 錯誤 ✓
  - [x] Step 6: 訓練 60000 steps ✓
  - [x] Step 7: 評估結果：**7.763 px (比 V1 的 4.055 px 退步 91%)**
  - **分析**: V2 架構本身有結構性錯誤（Spatial Attention + Multiplicative Gate 不在 Disparity Space 工作）
- [x] ~~**Exp #32: V2 + 選擇性凍結**~~ ⛔ **ABANDONED**
  - 放棄原因: V2 架構本身有結構性錯誤，Selective Freeze 無法解決
  - 預期只能改善到 5.x~6.x px，無法超越 V1
- [ ] **Exp #33: Polarization Volume V2-A (Corr Residual)** ← 當前重點
  - [x] Step 1: 設計 V2-A 架構 (Additive Bias in Disparity Space) ✓
  - [x] Step 2: 實現 PolCorrResidual 模組 ✓
  - [x] Step 3: 建立 PIDSStereoPolVolumeV2A ✓
  - [x] Step 4: 更新 train_pids.py 支援 --pol_volume_v2a ✓
  - [x] Step 5: 更新 evaluate_pids.py 支援 --pol_volume_v2a ✓
  - [ ] Step 6: 訓練並評估 (目標: Glass EPE < 3.8 px)
- [ ] (Optional) Exp #34: 加入 Learnable Pol Encoder
- [ ] (Optional) Exp #35: 加入 Glass-aware Auxiliary Head
- [ ] **Stage 2: Real-World Fine-Tuning** (規劃中)
  - [ ] Kinect v1 校正 (內參 + 與偏振相機外參)
  - [ ] 替身物體製作
  - [ ] 數據採集流程自動化
  - [ ] 真實世界數據集收集 (~100-500 場景)
  - [ ] Stage 2 fine-tuning 訓練

---

