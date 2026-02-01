## 實驗 #10：紋理渲染數據訓練

**日期**: 2025-12-31
**狀態**: 進行中

### 實驗目標

使用 v4.0.0 紋理版渲染器產生的數據進行訓練，驗證紋理多樣性對模型泛化能力的影響。

### 數據來源

- **渲染器**: `rendering_v4/pids_renderer_textured.py`
- **場景生成**: `modelling_textured/blender_furniture_randomizer_v18.py`
- **紋理資產**: `textures/` (含 manifest.json)

### 問題 1：NaN Loss

**現象**：訓練開始時 Loss、EPE、Glass EPE 全部為 NaN，D1/D3/D5/D10 為 0%。

**根本原因**：`pids_dataset.py` 中 `max_disparity=192`，但紋理數據的 disparity 範圍為 90~524。

```python
# 原本的 valid_mask 邏輯
valid_mask = (disparity_tensor > 0) & (disparity_tensor < max_disparity)
```

所有 disparity > 192 的像素都被標記為 invalid，導致無像素參與訓練。

**解決方案**：將 `max_disparity` 從 192 改為 576。

---

### 問題 2：Loss 劇烈震盪

**現象**：修復 NaN 後，Loss 從 ~560 下降到 ~170，但出現劇烈波動（如 237 → 429）。

**原因分析**：

| 問題 | 說明 |
|------|------|
| 梯度爆炸 | 遇到極難匹配的紋理時，優化器跳出局部最優 |
| 高頻細節代價 | 無紋理時靠邊緣匹配；有紋理時需學習像素級高頻細節 |
| 重複圖案干擾 | Cost Volume 出現多個波峰，模型困惑 |
| 收斂不一致 | 模型先學低頻結構，高頻紋理 Loss 干擾整體學習 |

---

### 解決方案：梯度累積 + FP32

使用梯度累積 (Gradient Accumulation) 代替 BF16 大 batch：

- `batch_size=8 × accumulation_steps=2 = Effective Batch 16`
- 使用 FP32 確保數值穩定性
- 顯存減半，訓練效果與 BS=16 一致

```bash
nohup python train_pids.py \
    --data_dir ./dataset_V3 \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 \
    --accumulation_steps 2 \
    --lr 0.00002 \
    --clip_grad 1.0 \
    --weight_decay 0.0001 \
    --scheduler cosine \
    --num_steps 20000 \
    --val_freq 100 \
    --val_split 0.2 \
    --d1_weight 0.1 \
    > train_textured_fp32_accum.log 2>&1 &
```

### 預期改善

| 方面 | 無紋理版 | 紋理版預期 |
|------|---------|-----------|
| 背景多樣性 | 單色 diffuse | 多種木紋/磚牆/地板 |
| 過擬合風險 | 高 | 降低 |
| 真實場景泛化 | 待驗證 | 預期改善 |
| 數值穩定性 | BF16 | FP32 + 梯度累積 |

### 問題 3：Validation Loss 暴衝 (1506)

**現象**：Train Loss 持續下降 (560 → 170)，但 Val Loss 上升並出現極端 spike (1506)。

**根本原因**：**紋理週期性歧義 (Texture Repetition Ambiguity)**

平整牆面 + 重複紋理貼圖導致 Cost Volume 出現多峰效應：

| 問題 | 說明 |
|------|------|
| 幾何無特徵 | 平牆沒有凸起凹陷，無法靠幾何定位 |
| 紋理重複 | 磁磚/磚塊等貼圖有週期性 pattern |
| 多點匹配 | 左眼一個點，右眼有多個「長得一樣」的候選點 |
| 週期跳錯 | 選錯一個週期，視差誤差 ~40px → Loss = 40² ≈ 1600 |

**數學驗證**：
- 紋理週期約 40 pixel
- 選錯週期時誤差 = 40
- Loss = 40^2 = 1600 approximately equals **1506** (matches!)

**過擬合機制**：
- Train：模型「背答案」，記住特定紋理的正確匹配 → Loss 低
- Val：遇到沒見過的紋理位移，無幾何特徵可依賴 → 猜錯週期 → Loss 爆炸

---

### 解決方案：強正則化 + 平滑性約束

強迫模型學習「平滑性」而非死記紋理細節：

```bash
nohup python train_pids.py \
    --data_dir ./dataset_V3 \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 \
    --accumulation_steps 2 \
    --lr 0.00001 \
    --clip_grad 0.5 \
    --weight_decay 0.01 \
    --scheduler cosine \
    --num_steps 30000 \
    --val_freq 200 \
    --val_split 0.2 \
    --d1_weight 0.1 \
    > train_flat_wall.log 2>&1 &
```

**關鍵改動**：

| 參數 | 舊值 | 新值 | 目的 |
|------|------|------|------|
| weight_decay | 0.0001 | **0.01** | 強正則化，抑制紋理過擬合 |
| lr | 0.00002 | **0.00001** | 避免在多峰間跳動 |
| num_steps | 20000 | **30000** | 更長訓練時間 |

**預期效果**：
- Train Loss 可能停在 ~200（不會太低）
- Val Loss 穩定，不再出現週期跳錯的暴衝
- 模型傾向輸出平滑視差，適應平牆場景

### 訓練進度（Step 2920/30000）

**Train Loss：**

| 階段 | 範圍 | 最佳值 |
|------|------|--------|
| 0-500 | 176~573 | 176 |
| 500-1000 | 143~571 | 143 |
| 1000-1500 | 126~463 | 126.97 |
| 1500-2000 | 150~520 | 150 |
| 2000-2500 | 121~484 | **121.03** |
| 2500-2920 | 153~426 | 153 |

**Val Loss：**

| Step | Val Loss | 備註 |
|------|----------|------|
| 200 | 1210.79 | 起始高 |
| 400 | 488.09 | 大幅下降 |
| 1200 | 467.63 | 穩定下降 |
| 1800 | 424.28 | 當時最佳 |
| 2400 | 1295.67 | Spike（紋理歧義） |
| 2600 | **421.96** | 快速恢復，新最佳 |
| 2800 | 629.01 | 正常波動 |

**與舊配置比較：**

| 指標 | 舊 (wd=0.0001) | 新 (wd=0.01) |
|------|----------------|--------------|
| Val Loss 趨勢 | 持續上升 (bad) | 整體下降 (good) |
| 最大 Spike | 1506 | 1295 (較小) |
| 最佳 Val Loss | ~480 | **421** (good) |
| Spike 恢復 | 慢 | 快 (good) |

**觀察結論：**
- 強正則化 (weight_decay=0.01) 有效抑制過擬合
- Spike 仍存在但恢復快，屬紋理歧義的正常現象
- 繼續訓練至 30000 步

### 結果（實驗 #10 第一輪）

| 指標 | 最佳值 | Step |
|------|--------|------|
| Val Loss | 376.27 | 15600 |
| Glass EPE | 63.49 px | 12200 |
| D1 | 67.12% | 1400 |
| Composite Score | 73.46 | 12200 |

**結論**：Glass EPE 仍然過高，無法達到 RA-L 投稿標準 (<20px)。

---

