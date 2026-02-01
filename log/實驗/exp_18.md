## 實驗 #18: Cross-Attention Fusion 架構

**日期**: 2026-01-05
**狀態**: 開發完成，訓練中

### 實驗目標

將 Dual-Stream 架構的 Concatenation 融合改為 Cross-Attention 融合，驗證注意力機制能否進一步提升玻璃區域的特徵融合效果。

### 設計動機

Exp #16 的 Dual-Stream 使用簡單的 `concat + conv` 融合：
```python
# Exp #16 融合方式
fused = torch.cat([stereo_feat, pol_feat], dim=1)  # (B, 256, H, W)
fused = self.fusion_conv(fused)  # (B, 128, H, W)
```

**問題**: Concatenation 平等對待所有空間位置，但偏振信號只在玻璃區域有意義。

**解決方案**: 使用 Cross-Attention 讓 stereo 特徵「查詢」偏振特徵，自動學習在哪裡需要偏振信息。

### 架構設計

```
Pol -> Stereo Cross-Attention:
    stereo_feat ─┬─ [Q] ←─ pol_feat [K,V] ─→ attended_stereo
                 │
                 └─→ stereo_out = stereo_feat + alpha * attended_stereo

alpha = sigmoid(learnable_logit) * alpha_cap  # 可學習門控 + warmup
```

### 新增組件

#### 1. PooledCrossAttention (Memory-Efficient)

```python
# 原始 attention: O(H*W * H*W) = O(N^2)
# Pooled attention: O(H*W * pool_h*pool_w) = O(N * M), M << N
# Memory 減少約 64 倍 (pool_size=8)

Q: (B, C, H, W) - 保持原始解析度
K, V: 池化到 (B, C, H/8, W/8) - 減少記憶體
```

#### 2. SafeCrossAttentionFusion (穩定訓練)

安全機制:
- **Alpha Gate**: 初始化 sigmoid(-5) ≈ 0.007，接近 0
- **Zero Init**: 輸出投影層初始化為 0
- **Alpha Cap Warmup**: alpha 上限從 0.05 線性增長到 1.0
- **Residual Connection**: `out = stereo + alpha * attended`
- **NaN Protection**: 數值穩定的 softmax + nan_to_num

#### 3. Alpha Cap Warmup

```python
def _compute_alpha_cap(self) -> float:
    if self.global_step >= self.args.alpha_cap_warmup:
        return 1.0
    progress = self.global_step / self.args.alpha_cap_warmup
    return self.args.alpha_cap_start + progress * (1.0 - self.args.alpha_cap_start)
    # 0.05 -> 1.0 over 5000 steps
```

### 開發過程 (Fork 方式)

**重要**: 本次使用 fork 方式開發，不修改原始檔案

| 原始檔案 | Fork 檔案 | 說明 |
|----------|-----------|------|
| `pids_model.py` | `pids_model_cross_attention.py` | 新增 Cross-Attention 模型 |
| `train_pids.py` | `train_pids_cross_attention.py` | 新增 Cross-Attention 訓練器 |

### Bug 修復歷程

#### Bug 1: ImportError - PIDSTrainer
```
ImportError: cannot import name 'PIDSTrainer' from 'train_pids'
```
**修正**: `from train_pids import Trainer` (類別名稱為 `Trainer` 非 `PIDSTrainer`)

#### Bug 2: ImportError - create_dataloaders
```
ImportError: cannot import name 'create_dataloaders' from 'pids_dataset'
```
**修正**: `from pids_dataset import create_data_loaders` (有底線)

#### Bug 3: AttributeError - 缺少參數
```
AttributeError: 'Namespace' object has no attribute 'gamma'
```
**修正**: 在 `parse_args()` 中補齊所有必要參數 (gamma, adam_eps, optimizer, resume, log_freq, print_freq, save_freq)

#### Bug 4: RuntimeError - Device Mismatch
```
RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
```
**修正**: `_build_model()` 缺少 `model.to(self.device)` 和 DataParallel 處理

#### Bug 5: NaN Loss
```
Step 100 | Loss: nan | EPE: nan | D1: 42.79% | Glass EPE: nan
```
**原因**: Attention mask 全為 -inf 時，softmax 輸出 NaN (0/0)

**修正** (`pids_model_cross_attention.py:115-135`):
```python
# 1. 只在有足夠有效值時應用 mask
valid_count = (mask_pooled >= 0.5).sum(dim=-1, keepdim=True)
if valid_count.min() > 0:
    attn = attn.masked_fill(mask_pooled < 0.5, float('-inf'))

# 2. 數值穩定的 softmax
attn_max = attn.max(dim=-1, keepdim=True)[0]
attn = attn - attn_max
attn = F.softmax(attn, dim=-1)

# 3. 最後防線
attn = torch.nan_to_num(attn, nan=0.0)
```

### 訓練配置

```bash
nohup python train_pids_cross_attention.py \
    --data_dir ./dataset_pol \
    --output_dir ./checkpoints_cross_attention_exp18 \
    --pol_dim 128 \
    --pol_threshold 0.05 \
    --pol_weight 2.0 \
    --glass_weight 5.0 \
    --pol_lr_mult 5.0 \
    --hidden_dim 128 \
    --context_dim 128 \
    --feature_dim 128 \
    --iters 24 \
    --batch_size 8 \
    --num_steps 70000 \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --lr 0.0003 \
    --val_freq 500 \
    --num_heads 4 \
    --pool_size 8 \
    --alpha_cap_start 0.05 \
    --alpha_cap_warmup 5000 \
    > train_exp18.log 2>&1 &
```

### Cross-Attention 專用參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| num_heads | 4 | Multi-head attention heads 數量 |
| pool_size | 8 | K,V 池化大小 (8 = 64x 記憶體節省) |
| enable_stereo_to_pol | False | 是否啟用雙向 attention (預設單向) |
| alpha_cap_start | 0.05 | Alpha warmup 起始值 |
| alpha_cap_warmup | 5000 | Alpha warmup 步數 |

### 模型參數統計

| 模型 | 總參數 | 可訓練參數 |
|------|--------|------------|
| Exp #16 (Dual-Stream Concat) | ~4.1M | ~4.1M |
| **Exp #18 (Cross-Attention)** | **4,363,715** | **4,363,715** |

新增參數主要來自:
- PooledCrossAttention (Q,K,V projections)
- SafeCrossAttentionFusion (alpha gates, output projections)

### 預期改善

| 指標 | Exp #16 (Concat) | Exp #18 (Cross-Attn) 預期 |
|------|------------------|--------------------------|
| Glass EPE | 21.82 px | < 20 px |
| Val Loss | 160.51 | < 155 |
| 穩定性 | 穩定 | 更穩定 (alpha warmup) |

### 訓練進度 (70K / 70K = 100% 完成)

**Glass EPE 趨勢:**
```
 5K steps:    45.86 px (起始)
10K steps:    47.94 px
20K steps:    32.43 px
30K steps:    30.24 px
37.5K steps:  22.33 px (首次跌破 23)
40.5K steps:  21.62 px (首次跌破 22，超越 Exp #16)
46K steps:    20.91 px (首次跌破 21)
56.5K steps:  19.87 px (首次跌破 20！)
59.5K steps:  19.26 px
70K steps:    19.26 px ⭐ 最終結果
```

**Val Loss 趨勢:**
```
10K steps:    318.73
20K steps:    224.99
30K steps:    206.40
40K steps:    204.56
46K steps:    151.05 (跌破 Exp #16)
50K steps:    151.36
55K steps:    146.93
59.5K steps:  140.23
63K steps:    138.15
70K steps:    135.77 ⭐ 最終結果
```

### 最終結果

| 指標 | Exp #16 (Concat) | Exp #18 (Cross-Attn) | 改善 |
|------|------------------|---------------------|------|
| **Glass EPE** | 21.82 px | **19.26 px** | **-11.7%** |
| **Val Loss** | 160.51 | **135.77** | **-15.4%** |
| Best Step | 50K | 70K | - |

### 關鍵發現

1. **Cross-Attention 超越 Concatenation**: 僅改變融合機制就帶來 11.7% Glass EPE 改善
2. **整體品質提升**: Val Loss 降低 15.4%，說明沒有犧牲背景精度
3. **學習曲線較慢但後勁強**: 前期落後 Exp #16，但 40K 後開始超越並持續改善
4. **突破 20px 門檻**: 首次將 Glass EPE 降到 1 字頭 (19.26 px)
5. **最終即最佳**: 70K 結束時兩個指標都達到最佳值，說明模型仍有學習空間

### 消融實驗總結

| 方法 | Glass EPE | Val Loss | vs Baseline |
|------|-----------|----------|-------------|
| Baseline (無偏振) | 42.17 px | 248.59 | -- |
| + 隱式偏振輸入 | 39.20 px | 231.65 | -7.0% |
| + Dual-Stream Encoder | 21.82 px | 160.51 | -48.3% |
| + **Cross-Attention Fusion** | **19.26 px** | **135.77** | **-54.3%** |

### 技術貢獻

1. **PooledCrossAttention**: 64x 記憶體節省，使 attention 可行於高解析度特徵圖
2. **SafeCrossAttentionFusion**: Alpha warmup + zero init 確保訓練穩定
3. **Pol→Stereo 單向 Attention**: 讓 stereo 特徵「查詢」偏振信息，自動學習在哪裡需要偏振

**狀態:** 訓練完成，最終 Glass EPE 19.26 px，Val Loss 135.77

---

