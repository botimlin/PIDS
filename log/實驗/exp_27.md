## Exp #27: Data Scaling with Strict QA

**日期**: 2026-01-25
**目標**: 使用正確的 LR=0.0003 重新訓練，並驗證數據量對性能的影響

### 觀察與假設 (2026-01-25)

訓練過程中發現與 Exp #24 不同的現象：
1. **Val EPE 持續高於 Train EPE** (之前常出現 Val < Train)
2. **Val EPE 波動變大**

**數據差異**:
| 項目 | Exp #24 | Exp #27 |
|------|---------|---------|
| 渲染總量 | 9,000 | 15,000 |
| QA 標準 | 寬鬆 (C2+C5) | 嚴格 (C2+C3+C5) |
| 通過數量 | ~4,700 | ~5,000 |
| 通過率 | ~52% | ~33% |

**假設**: 嚴格 C3 過濾的潛在問題
- 原始數據可能混入了一些 **pol 接近 0** 的場景
- 這迫使模型**同時學習幾何特徵**（不能只依賴 pol）
- 遇到 Val 時，即使 pol 信號弱也能用幾何解決
- 嚴格 C3 後，模型**過度依賴 pol 特徵**，泛化能力下降

**後續實驗方向**:
1. 混合訓練: 80% 強 pol + 20% 弱/無 pol
2. 分階段訓練: nopol 預訓練 → pol 微調
3. 放寬 C3 閾值: 0.98~1.02 → 0.96~1.04

待 Exp #27 結果出來後決定下一步。

### 訓練配置

與 Exp #24 完全相同的超參數：
- `--lr 0.0003`
- `--iters 24`
- `--pol_levels 4`
- `--pol_radius 4`
- `--batch_size 8`

```bash
# 5% (250 scenes, 3k steps)
nohup python train_pids.py \
    --data_dir /workspace/RAFT-Stereo/dataset_pol_final/train_5pct \
    --output_dir ./checkpoints_exp27_5pct \
    --pol_volume \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 3000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 300 --num_workers 4 \
    > train_exp27_5pct.log 2>&1 &

# 10% (500 scenes, 6k steps)
nohup python train_pids.py \
    --data_dir /workspace/RAFT-Stereo/dataset_pol_final/train_10pct \
    --output_dir ./checkpoints_exp27_10pct \
    --pol_volume \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 6000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 500 --num_workers 4 \
    > train_exp27_10pct.log 2>&1 &

# 25% (1250 scenes, 15k steps)
nohup python train_pids.py \
    --data_dir /workspace/RAFT-Stereo/dataset_pol_final/train_25pct \
    --output_dir ./checkpoints_exp27_25pct \
    --pol_volume \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 15000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 500 --num_workers 4 \
    > train_exp27_15pct.log 2>&1 &

# 50% (2500 scenes, 30k steps)
nohup python train_pids.py \
    --data_dir /workspace/RAFT-Stereo/dataset_pol_final/train_50pct \
    --output_dir ./checkpoints_exp27_50pct \
    --pol_volume \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 30000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 500 --num_workers 4 \
    > train_exp27_50pct.log 2>&1 &

# 75% (3750 scenes, 45k steps)
nohup python train_pids.py \
    --data_dir /workspace/RAFT-Stereo/dataset_pol_final/train_75pct \
    --output_dir ./checkpoints_exp27_75pct \
    --pol_volume \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 45000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 500 --num_workers 4 \
    > train_exp27_75pct.log 2>&1 &

# 100% (5000 scenes, 60k steps)
nohup python train_pids.py \
    --data_dir /workspace/RAFT-Stereo/dataset_pol_final/train_100pct \
    --output_dir ./checkpoints_exp27_100pct \
    --pol_volume \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 60000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 500 --num_workers 4 \
    > train_exp27_100pct.log 2>&1 &
```

### 實驗結果

| 數據量 | Glass EPE | Glass D1 | Overall EPE | 備註 |
|--------|-----------|----------|-------------|------|
| 5% (250) | - | - | - | 待訓練 |
| 10% (500) | - | - | - | 待訓練 |
| 25% (1250) | - | - | - | 待訓練 |
| 50% (2500) | - | - | - | 待訓練 |
| 75% (3750) | - | - | - | 待訓練 |
| **100% (5000)** | **5.03 px** | **36.28%** | **3.42 px** | ⚠️ 比 Exp #24 差 |

### 關鍵發現

**對比 Exp #24 (原始數據，含弱偏振)**:
| 指標 | Exp #24 | Exp #27 | 變化 |
|------|---------|---------|------|
| Glass EPE | 2.44 px | 5.03 px | **+106%** ⚠️ |
| Glass D1 | 28.76% | 36.28% | +26% |
| Overall EPE | 1.92 px | 3.42 px | +78% |

**結論**: 嚴格 QA 過濾後性能反而大幅下降！

**原因分析**:
1. 嚴格 C3 過濾移除了所有弱偏振場景
2. 純強偏振數據 → 模型過度依賴偏振特徵
3. 缺少幾何多樣性 → 泛化能力下降
4. 測試集包含各種偏振強度場景 → 模型無法處理弱偏振

**驗證假設**: ✅ 確認需要混合 Pol/Nopol 訓練 (Exp #28)

**狀態**: ✅ 完成 (結果不佳，需 Exp #28 改進)

---

