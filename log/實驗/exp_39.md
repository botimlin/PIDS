## Exp #39: Polarization Volume V2-E (2026-01-28)

### 為什麼還需要 V2-E？

V2 系列（A/B/C/D）都是 **post-corr intervention**：
- 先計算 correlation volume
- 再用 pol 信息修正/調制

但這可能有根本限制：**錯誤的 correlation 已經形成，修正為時已晚**

### V2-E 的核心創新：Pre-Corr Pol Weighting

**關鍵轉變：**

| | V2 系列 (Post-Corr) | V2-E (Pre-Corr) |
|--|---------------------|-----------------|
| 時機 | corr 計算後 | corr 計算時 |
| 作用 | 修正已有的 corr | **影響 corr 的形成** |
| 類比 | 事後補救 | **源頭介入** |

**設計：**

```python
# V2 系列：Post-Corr
corr = dot(fmap1, fmap2)
corr_mod = corr * gate  # 事後修正

# V2-E：Pre-Corr（pol 參與 corr 計算）
for d in disparity_range:
    pol_diff_d = left - shift(right, d)
    pol_weight_d = sigmoid(f(pol_diff_d))  # [0, 1]

    # pol 直接影響 correlation 的權重
    corr[d] = dot(fmap1, fmap2_at_d) * pol_weight_d
```

**物理意義：**
- pol_weight_d 高：這個 disparity 的 pol_diff 符合「同一物理點」的特徵
- pol_weight_d 低：這個 disparity 可能是假匹配（反射/錯位）
- **在 correlation 形成時就抑制假匹配，而非事後修正**

### 訓練命令

```bash
nohup python train_pids.py \
    --data_dir /workspace/RAFT-Stereo/mixed_dataset_exp28/data \
    --output_dir ./checkpoints_exp39_v2e \
    --pol_volume_v2e \
    --pol_weight_hidden 8 \
    --curriculum \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 60000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 500 --num_workers 4 \
    > train_exp39.log 2>&1 &
```

### 目標

| 實驗 | 架構 | 介入點 | 說明 |
|------|------|--------|------|
| V2-A~D | Post-Corr | corr 計算後 | 已壓榨 |
| **V2-E** | **Pre-Corr** | **corr 計算時** | **最後嘗試** |

**如果 V2-E 也沒有提升：**
- 架構已壓榨到極致
- 轉向證明 nopol 7px 是 overfitting（用 OOD test set）

### 訓練結果

**訓練完成** (2026-01-29)

#### 訓練曲線
- Best Glass EPE: **5.468 px** @ Step 59500
- Final Glass EPE: **5.462 px** @ Step 60000
- 收斂穩定，最後 10k steps 在 5.4~5.9 px 震盪

#### 評估結果 (Test Set, 857 samples)

| Metric | V2-E (Exp39) | Baseline (Exp22) | 改善 |
|--------|--------------|------------------|------|
| Glass EPE (Mean) | **5.039 px** | 6.542 px | -23% |
| Glass EPE (Median) | **2.333 px** | 3.489 px | -33% |
| Glass D1 | **34.83%** | 75.42% | -54% |
| Glass D3 | **23.53%** | 41.31% | -43% |
| Overall EPE | **3.270 px** | 3.444 px | -5% |
| Overall D1 | **24.15%** | 41.25% | -41% |

#### Glass EPE 分佈

```
V2-E:
  Min:    0.046 px
  Q25:    1.113 px
  Median: 2.333 px
  Q75:    5.610 px
  Max:    108.182 px

Baseline:
  Min:    0.000 px
  Q25:    2.086 px
  Median: 3.489 px
  Q75:    6.441 px
  Max:    208.084 px
```

#### 關鍵發現

1. **Mean vs Median 差距大**：V2-E Mean 5.039 vs Median 2.333 (+116%)
   - 表示有 outlier 拉高平均值
   - 實際中位數表現比 Mean 好很多

2. **D1 改善顯著**：75% → 35% (減少 40 個百分點)
   - D1 = 誤差 > 1px 的比例
   - 說明偏振確實幫助減少大誤差

3. **偏振有效但 outlier 問題嚴重**
   - 需要更 robust 的評估指標
   - → 觸發 Exp #40 (Robust Checkpoint Selection)

---

### 架構演進總結

#### V2-B: Scheduled Residual

**核心改動**: 加入 iteration schedule

```python
# V2-A (current)
corr_enhanced = corr + self.pol_residual(pol_corr)

# V2-B (next)
alpha = i / (iters - 1)  # i = current iteration, 0 → 1
corr_enhanced = corr + alpha * self.pol_residual(pol_corr)
```

**設計哲學**:
- RAFT-Stereo 的核心是**逐步修正** disparity
- V2-A 的 residual 是 static，像「外掛」
- V2-B 讓 residual 前期弱、後期強，符合 RAFT 精神

**預期**: 有機會進入 3.x px

#### V2-C: Gradient-based Gating (進階)

```python
disp_grad = |∇disp|  # disparity gradient
gate = f(pol_corr, disp_grad)  # gate ∈ [0, 1]
corr_enhanced = corr + gate * pol_embedding
```

**直覺**:
- high gradient / unstable region → pol 有話語權
- flat / confident region → pol 安靜

#### 架構優化優先級

| Priority | 方向 | 預期增益 | 風險 | 說明 |
|----------|------|----------|------|------|
| ⭐ P1 | Iterative/Scheduled Residual | 高 (→ 3.x px) | 低 | V2-B |
| ⭐ P2 | Multi-scale Residual | 中 (+0.2-0.4 px) | 低 | 工程優化 |
| ⚠️ P3 | Disparity-aware Weight | 中 | 中 | 風險：feedback instability |
| 📌 P4 | Pol Feature Extractor | 高 | 高 | 成本高，歸因問題 |

### 設計陷阱警告

**❌ 危險做法**: 直接用 disparity 值作為 conditioning
```python
# 錯誤！early iteration 的 disp 是錯的
residual = pol_residual(pol_corr, disp)
```

**問題**:
- Early iteration 的 disparity 是 noise
- 會產生 feedback loop
- 訓練不穩，後期可能退化

**✅ 安全做法**:
- Option A: 使用 iteration index + fixed schedule (V2-B)
- Option B: 使用 disparity gradient/variance，不用值本身
- 關鍵: pol_residual 只學「where & how much」，不是「what disparity」
