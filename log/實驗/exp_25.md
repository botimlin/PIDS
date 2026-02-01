## Exp #25: Learnable Polarization Volume

**日期**: 2026-01-18
**目標**: 從 Glass EPE 2.44 px 進一步優化到 ~2.0 px

### 動機

Exp #24 的 Polarization Volume 使用固定公式計算偏振差異：
```python
pol_diff[x, d] = I_∥[x] - I_⊥[x-d]
```

問題：
- 固定公式無法適應不同的場景條件（亮度、角度、材質）
- 無法學習「什麼是穩健的偏振信號」
- 可能被噪聲或假陽性干擾

### 核心改進

**從「手工公式」提升到「可學習表徵」**

```
固定公式:
    pol_diff = I_∥ - I_⊥

可學習版:
    pol_feat = PolHead(fmap)           # 學習提取偏振特徵
    pol_corr = |pol_feat_L - pol_feat_R|  # L1 差異
```

### 架構設計

```
left ──→ [FeatureEncoder] ──→ fmap1 ─┬──→ [CorrBlock] ──────────┐
               │                     │                          │
               └──→ [PolHead] ───────┼──→ pol_feat1             │
                                     │                          │
right ─→ [FeatureEncoder] ──→ fmap2 ─┼──→ [CorrBlock] ──────────┤
               │                     │                          │
               └──→ [PolHead] ───────┼──→ pol_feat2             │
                                     │                          │
              [LearnablePolCorrBlock] ──────────────────────────┤
                                     │                          │
                                     ↓                          ↓
                              pol_volume              corr_volume
                                     │                          │
                                     └──────────┬───────────────┘
                                                ↓
                                        [UpdateBlockWithPol]
                                                │
                                                ↓
                                           disparity
```

### 新增組件

| 類別 | 功能 | 參數量 |
|------|------|--------|
| `LightweightPolHead` | 輕量級偏振特徵頭 (256→64→32) | ~18K |
| `LearnablePolCorrBlock` | 向量化偏振體積計算 | 0 |
| `GlassAwareLoss` | Glass-aware auxiliary loss | 0 |
| `PIDSStereoLearnablePol` | 整合模型 | 基礎 + 18K |

### LightweightPolHead 設計

```python
class LightweightPolHead(nn.Module):
    """
    輕量級偏振特徵頭 (~18K 參數)
    結構: feature_dim -> 64 -> pol_dim
    """
    def __init__(self, in_dim=256, pol_dim=32):
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, pol_dim, 1),
        )
```

### Glass-aware Auxiliary Loss

讓 PolHead 學習「偏振特徵」而非「紋理」：

```python
# 玻璃區域的 pol_volume 應該高
# 背景區域的 pol_volume 應該低
loss = max(0, margin + bg_pol - glass_pol)
```

這個 loss 強迫 PolHead 輸出的特徵在玻璃區域有高差異。

### 訓練命令

```bash
nohup python train_pids.py \
    --data_dir ./dataset/train \
    --output_dir ./checkpoints_learnable_pol_exp25 \
    --learnable_pol \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 \
    --num_steps 60000 \
    --lr 0.0003 \
    --iters 24 \
    --pol_dim 32 \
    --pol_levels 4 \
    --pol_radius 4 \
    --glass_aware_weight 0.1 \
    --val_freq 500 \
    --num_workers 4 \
    > train_learnable_pol_exp25.log 2>&1 &
```

### 實驗結果

| 指標 | Baseline | Pol Volume (Exp #24) | Learnable Pol (Exp #25) |
|------|----------|----------------------|-------------------------|
| Glass EPE | 7.520 px | 2.435 px | 2.475 px |
| Glass D1 | 81.51% | 19.64% | 19.34% |
| Overall EPE | 2.205 px | 2.286 px | 2.256 px |

### 結論

**Learnable PolHead 沒有帶來額外改善**：
- Glass EPE: 2.475 px (微幅上升 +0.04 px vs Pol Volume)
- Glass D1: 19.34% (微幅改善 -0.3%)

這表示 **固定公式 (I_sum, I_diff, DoLP) 已經足夠好**，學習額外的偏振特徵沒有顯著幫助。

可能原因：
1. 物理公式已經捕捉到最重要的偏振資訊
2. 網路容量太小無法學到更好的表徵
3. Glass-aware Loss 效果有限

**建議**: 繼續使用 Exp #24 Pol Volume 架構進行 Stage 2 fine-tuning

**狀態**: ✅ 完成 (無顯著改善)

---

