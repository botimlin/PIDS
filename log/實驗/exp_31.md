## Exp #31: Polarization Volume V2 (2026-01-27)

### 目標
通過架構優化拉開與 Baseline 的差距（V1 僅 38% 改善）

### V2 架構設計

**Plan A: PolarizationAttention**
```python
class PolarizationAttention(nn.Module):
    # 從 pol_corr 生成 spatial attention map
    # 識別玻璃區域 (pol_diff 高的地方)
    # 輸出: attention map [B, 1, H, W]
```

**Plan C: GatedFusion**
```python
class GatedFusion(nn.Module):
    # 學習 stereo vs pol 的動態融合權重
    # gate = sigmoid(f(corr, pol_corr, pol_attn))
    # fused = gate * stereo_feat + (1-gate) * pol_feat
```

### 訓練配置
```bash
nohup python train_pids.py \
    --data_dir ./PIDS_dataset_exp29_mixed/data \
    --output_dir ./checkpoints_exp31_polvol_v2 \
    --pol_volume_v2 \
    --curriculum \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 60000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 500 --num_workers 4 \
    > train_exp31.log 2>&1 &
```

### 訓練結果

| Step | Val Glass EPE | Val EPE | D1 | Composite |
|------|---------------|---------|-----|-----------|
| 500 | 89.xx px | - | - | - |
| 30000 | ~8.0 px | ~5.0 | ~32% | ~11.2 |
| 59500 | 7.938 px | 5.016 | 32.69% | 11.208 |
| **60000** | **7.763 px** | **4.809** | **32.23%** | **10.986** |

### 結果分析

| 實驗 | 架構 | Glass EPE | vs V1 |
|------|------|-----------|-------|
| Exp #24 | V1 (Pol Volume) | 4.055 px | 基準 |
| **Exp #31** | **V2 (Attention + Gated)** | **7.763 px** | **退步 91%** |

**關鍵發現**: V2 架構在混合數據集上反而更差

### 問題診斷（深入分析）

**初步假設**: 混合數據集稀釋了 V2 模組的學習
- 60% pol + 40% nopol 訓練
- nopol 數據沒有偏振差異 (I∥ ≈ I⊥)

**深入分析後的真正原因**: V2 架構本身有結構性錯誤

1. **Spatial Attention 問題**: PolarizationAttention 生成的是 spatial attention map [H, W]，不知道該強調哪個 disparity level
2. **Multiplicative Control 問題**: GatedFusion 使用 `gate × corr + (1-gate) × pol` 會破壞 RAFT 的 inductive bias
3. **不在 Disparity Space 工作**: V2 的 attention 和 gate 都在 spatial domain，而不是 disparity domain

**結論**: Selective Freeze 只是「讓錯的模組少犯錯」，不能解決架構本身的問題

---

