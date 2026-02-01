## 實驗 #12：提高 glass_weight 至 5.0

**日期**: 2026-01-02
**狀態**: 完成

### 動機

實驗 #11 證明方向正確，趁有便宜 GPU instance 再推進一輪。

### 訓練配置

```bash
nohup python train_pids.py \
    --data_dir ./dataset_V3 \
    --output_dir ./checkpoints_v3 \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --glass_weight 5.0 \
    --lr 0.00005 \
    --batch_size 2 \
    --accumulation_steps 4 \
    --num_steps 30000 \
    --val_freq 500 \
    --iters 16 \
    --scheduler cosine \
    --bf16 \
    --d1_weight 0.2 \
    > train_v3_gw5.log 2>&1 &
```

### 結果

| 指標 | 實驗 #11 (gw=3.0) | 實驗 #12 (gw=5.0) | 變化 |
|------|-------------------|-------------------|------|
| Val Loss | 265.91 | 299.08 | +12.5% (worse) |
| Glass EPE | 46.90 px | 44.96 px | -4.1% (better) |

### 結論

Glass EPE 略有改善 (46.90 → 44.96)，但代價是 **Val Loss 惡化 12.5%**。

**分析**：
- `glass_weight=5.0` 過於激進
- 模型過度專注玻璃區域，犧牲非玻璃區域精度
- 整體泛化能力下降

**最佳配置維持 `glass_weight=3.0` (實驗 #11)**

---

## 當前最佳模型

**配置**: 實驗 #11 (`glass_weight=3.0`, `max_disp=576`)

| 指標 | 值 |
|------|-----|
| Val Loss | 265.91 |
| Glass EPE | 46.90 px |
| D1 | 58.56% |

---

## 未來優化方向

### 1. 增加訓練樣本數

目前使用約 2000 個樣本，觀察到：
- 後期 Loss 曲線非常平穩（模型已「吃飽」）
- 調整超參數（如 glass_weight）改善有限
- 這些是數據量瓶頸的典型跡象

**建議**：將樣本數從 2000 增加到 4000+，使用 v4 紋理渲染器配合隨機化增強。

### 2. 增加場景多樣性

- 更多玻璃形狀（曲面、斜面）
- 更多傢俱類型和擺放方式
- 不同光照條件

### 3. PIDS vs Baseline 對比實驗

使用現有數據完成消融實驗，量化偏振對透明物體偵測的貢獻：
- 偏振版 (PIDS): 使用 `pids_renderer_textured.py`
- 無偏振版 (Baseline): 使用 `pids_renderer_textured_nopol.py`

---

### 訓練監控標準

**Loss 震盪優先級：**
- Loss 震盪 > EPE/D1 震盪（Loss 直接影響梯度）
- Loss 穩定後，EPE/D1 會跟著穩定

**可接受的情況：**
- 整體下降趨勢
- Spike 後快速恢復（1-2 個 val_freq 內）
- Spike 幅度 < 3x 平均值
- Train/Val gap < 5x

**需要介入的情況：**
- 連續上升 3+ 次
- Spike 後不恢復
- Spike 頻率增加
- Val 持續遠離 Train

**目標終點：**
- Val Loss 穩定在 300-400
- Glass EPE < 30 px
- D1 < 20%

---

## 大規模渲染 (2026-01-01)

### 目標

為 PIDS vs Baseline 消融實驗準備 8000 張訓練數據（4000 偏振 + 4000 無偏振）。

### 渲染環境

| 項目 | 規格 |
|------|------|
| GPU | 16x NVIDIA GPU |
| 場景數 | 4000 |
| SPP | 4096 |
| 預估時間 | ~3.5 小時 |
| 預估成本 | ~$35 |

### 渲染器更新

為確保 PIDS vs Baseline 公平對比，更新了兩個渲染器：

#### 1. 確定性隨機化

將 Python 內建 `hash()` 替換為 `hashlib.md5`，確保跨會話一致：

```python
def deterministic_hash(s: str) -> int:
    """確定性 hash，跨 Python 會話一致"""
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**32)
```

#### 2. params.json 燈光參數

偏振版現在保存燈光參數到 params.json：

```json
{
  "lighting": {
    "led_intensity": 3200.5,
    "ceiling_emitter_intensity": 175.3
  }
}
```

nopol 版可從 params.json 讀取以精確匹配。

#### 3. 多 GPU 支持

兩版渲染器都支持 `--num_gpus` 參數，使用 subprocess 實現真正的多進程並行。

### 渲染流程

```bash
# Step 1: 生成 4000 場景 (Blender)
blender scene.blend --background --python blender_furniture_randomizer_v18.py -- \
    --count 4000 --output ./scenes_textured --texture_dir ./textures

# Step 2: 渲染偏振版 (16 GPU, ~3.5hr)
python pids_renderer_textured.py \
    --input_dir ./scenes_textured \
    --output ./output_pol \
    --num_gpus 16 \
    --spp 4096

# Step 3: QA 篩選，複製通過的 params.json
python copy_passed_params.py \
    --input_dir ./output_pol \
    --output_dir ./passed_params \
    --exclude failed.txt

# Step 4: 渲染無偏振版 (只渲染通過 QA 的場景)
python pids_renderer_textured_nopol.py \
    --input_dir ./scenes_textured \
    --output ./output_nopol \
    --params_dir ./passed_params \
    --num_gpus 16 \
    --spp 4096 \
    --no_preview
```

### 當前進度

- [x] 場景生成 (4000 個)
- [x] 偏振版渲染完成 (16 GPU, ~50秒/場景, ~4hr)
- [x] QA 篩選完成 (通過: 3765, 未通過: 235, 通過率: 94.1%)
- [x] 無偏振版渲染完成 (16 GPU, ~5秒/場景, 35min)
- [x] 整理數據集 pol (train 3500 / test 265)
- [x] 訓練 PIDS 偏振版 (實驗 #13, Glass EPE 39.20 px)
- [x] 整理數據集 nopol (使用 --scene_list 對齊)
- [x] 訓練 Baseline 無偏振版 (實驗 #14, Glass EPE 42.17 px)
- [x] Dual-Stream 架構開發 (實驗 #16, Glass EPE 21.82 px, -48.3% vs Baseline)
- [ ] 大規模數據訓練 (實驗 #17, 5000 樣本, 70K steps) - 訓練中
- [ ] 對比評估 PIDS vs Baseline (Test Set)

---

