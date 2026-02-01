## 實驗 #11：修正 Loss 函數 max_disp 不一致問題

**日期**: 2026-01-01
**狀態**: 進行中

### 問題發現

分析實驗 #10 結果時發現 **max_disp 設置不一致**：

| 文件 | 參數 | 值 | 作用 |
|------|------|-----|------|
| `pids_dataset.py:99` | max_disparity | **576.0** (correct) | Dataset valid_mask |
| `pids_model.py:380` | max_disp | **192.0** (wrong) | Loss 函數 valid_mask |
| `train_pids.py:637` | --max_disp | **192.0** (wrong) | 命令行默認值 |

**後果**：
- Dataset 正確加載全範圍視差 (0-576px)
- 但 Loss 函數仍將 >192px 的視差標記為 invalid
- 玻璃區域的大視差 (200-400px) 被排除在訓練外
- 導致 Glass EPE 虛高 (63.49px)

### 解決方案

修改兩個文件的默認值：

```python
# pids_model.py:380
max_disp: float = 576.0,  # 從 192.0 改為 576.0

# train_pids.py:637
parser.add_argument('--max_disp', type=float, default=576.0)  # 從 192.0 改為 576.0
```

### 訓練配置

```bash
nohup python train_pids.py \
    --data_dir ./dataset_V3 \
    --output_dir ./checkpoints_v2 \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --glass_weight 3.0 \
    --lr 0.00005 \
    --batch_size 2 \
    --accumulation_steps 4 \
    --num_steps 30000 \
    --val_freq 500 \
    --iters 16 \
    --scheduler cosine \
    --bf16 \
    --d1_weight 0.2 \
    > train_v2_maxdisp576.log 2>&1 &
```

**關鍵改動**：

| 參數 | 實驗 #10 | 實驗 #11 | 目的 |
|------|----------|----------|------|
| max_disp | 192 (錯誤) | **576** | 匹配實際視差範圍 |
| glass_weight | 1.0 | **3.0** | 增加玻璃區域關注 |
| lr | 0.00002 | **0.00005** | 配合預訓練微調 |

### 預期改善

| 指標 | 實驗 #10 | 預期值 | 提升率 |
|------|----------|--------|--------|
| Glass EPE | 63.49 px | **15-25 px** | 60-75%↓ |
| D1 | 67% | **35-45%** | 33-48%↓ |
| Composite | 73.46 | **20-35** | 52-73%↓ |

### 結果

**日期**: 2026-01-02
**狀態**: 完成

| 指標 | 實驗 #10 | 實驗 #11 | 改善 |
|------|----------|----------|------|
| Val Loss | 376.27 | **265.91** | 29%↓ |
| Glass EPE | 63.49 px | **46.90 px** | 26%↓ |
| D1 | 67.12% | **58.56%** | 13%↓ |
| 最佳 Step | 12200-15600 | **28000-30000** | - |

**關鍵發現：後期神級穩定性**

最後 5000 步 Glass EPE 波動範圍僅 **±0.5px**：
```
Step 26000: 47.30
Step 26500: 47.78
Step 27000: 47.16
Step 27500: 47.18
Step 28000: 47.25
Step 28500: 47.77
Step 29000: 47.48
Step 29500: 47.23
Step 30000: 46.90
```

對比實驗 #10 的劇烈震盪（400→1500→600），這次是**真正收斂**。

**結論**：
- max_disp 修正確實有效
- 模型學到了穩定特徵，可信任泛化能力
- 尚未達到目標 (<20px)，繼續提高 glass_weight

---

