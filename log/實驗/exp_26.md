## Exp #26: Final - Data Scaling with Strict QA

**日期**: 2026-01-24
**目標**: 使用更嚴格的 QA 標準訓練最終模型，並驗證數據量對性能的影響

### QA 改進 (pids_qa.py v2.3.0)

相比之前的 `aggregate_reports.py`（只檢查 C2 + C5），新增 **C3 玻璃偏振檢查**：

| 檢查項 | 數據來源 | 閾值 | 物理意義 |
|--------|----------|------|----------|
| C2 背景一致性 | `intensity_balance.background_ratio_mean` (跨視角) | 0.80 ~ 1.25 | Stereo pair 背景可匹配 |
| **C3 玻璃偏振** | `polarization.glass_region.stokes_ratio` (同視角) | < 0.96 或 > 1.04 | 玻璃有足夠偏振對比 |
| C5 深度有效率 | `glass_depth_validity.validity_rate` | >= 90% | 玻璃區域深度覆蓋 |

**QA 結果** (15000 渲染場景):
- C2 通過率: 56.9%
- C3 通過率: 78.1%
- C5 通過率: 97.5%
- **整體通過率: 39.1%** (5859 場景)

**假設**: 更嚴格的 C3 過濾可確保訓練數據都有明顯的偏振信號，可能進一步提升 Glass EPE。

### 數據量消融實驗設計

固定 96 epochs，不同數據量：

| 百分比 | 場景數 | Steps | 預期效果 |
|--------|--------|-------|----------|
| 5% | 250 | 3,000 | 基線，可能過擬合 |
| 10% | 500 | 6,000 | CLAUDE.md 建議的最小量 |
| 25% | 1250 | 15,000 | 中等數據量 |
| 50% | 2500 | 30,000 | 半量數據 |
| 75% | 3750 | 45,000 | 接近全量 |
| 100% | 5000 | 60,000 | 全量數據 |

### 訓練配置

```bash
# 5% (250 scenes, 3k steps)
nohup python train_pids.py \
    --data_dir /workspace/RAFT-Stereo/dataset_pol_final/train_5pct \
    --output_dir ./checkpoints_final_5pct \
    --pol_volume \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 3000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 300 --num_workers 4 \
    > train_final_5pct.log 2>&1 &

# 10% (500 scenes, 6k steps)
nohup python train_pids.py \
    --data_dir /workspace/RAFT-Stereo/dataset_pol_final/train_10pct \
    --output_dir ./checkpoints_final_10pct \
    --pol_volume \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 6000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 500 --num_workers 4 \
    > train_final_10pct.log 2>&1 &

# 25% (1250 scenes, 15k steps)
nohup python train_pids.py \
    --data_dir /workspace/RAFT-Stereo/dataset_pol_final/train_25pct \
    --output_dir ./checkpoints_final_25pct \
    --pol_volume \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 15000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 500 --num_workers 4 \
    > train_final_25pct.log 2>&1 &

# 50% (2500 scenes, 30k steps)
nohup python train_pids.py \
    --data_dir /workspace/RAFT-Stereo/dataset_pol_final/train_50pct \
    --output_dir ./checkpoints_final_50pct \
    --pol_volume \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 30000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 500 --num_workers 4 \
    > train_final_50pct.log 2>&1 &

# 75% (3750 scenes, 45k steps)
nohup python train_pids.py \
    --data_dir /workspace/RAFT-Stereo/dataset_pol_final/train_75pct \
    --output_dir ./checkpoints_final_75pct \
    --pol_volume \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 45000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 500 --num_workers 4 \
    > train_final_75pct.log 2>&1 &

# 100% (5000 scenes, 60k steps) - 正在訓練
nohup python train_pids.py \
    --data_dir /workspace/RAFT-Stereo/dataset_pol_final/train_100pct \
    --output_dir ./checkpoints_final_100pct \
    --pol_volume \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 60000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 500 --num_workers 4 \
    > train_final_100pct.log 2>&1 &
```

### 預期結果

1. **C3 過濾效果**: 更嚴格的偏振品質過濾可能讓模型學到更一致的偏振特徵
2. **數據量影響**: 預期 Glass EPE 隨數據量增加而下降，但可能在某個點飽和
3. **最終目標**: Glass EPE < 2.0 px

### 實驗結果

| Learning Rate | Glass EPE | Glass D1 | Overall EPE | 結果 |
|---------------|-----------|----------|-------------|------|
| 0.0001 (錯誤) | 5.37 px | 36.82% | 3.66 px | ❌ +120% 退化 |

**分析**:
- Learning rate 過小 (0.0001) 導致模型收斂不足
- 即使訓練完成 58500 steps，Best Glass EPE 只達到 5.66 px
- 學習率差 3 倍，性能差距超過 2 倍

**結論**:
1. **Learning rate 是關鍵超參數**，必須嚴格按照成功實驗配置
2. Polarization Volume 架構最佳 LR = 0.0003
3. 後續實驗必須使用 `--lr 0.0003`

**潛在優化方向**:
- LR Sweep: 測試 0.0002, 0.0003, 0.0004, 0.0005
- LR Schedule: Warmup + Cosine Annealing
- 可能存在更優 LR 進一步提升 Glass EPE

**狀態**: ✅ 完成 (發現 LR 影響)

---

