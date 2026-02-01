## 實驗 #16: Dual-Stream 修正版 (突破性進展)

**日期**: 2026-01-04
**狀態**: 訓練完成

### 修正內容

- 修正 flow->disparity 維度 bug (`[-f[:, :1] for f in flow_preds]`)
- 4 處修正: train_epoch (FP16/BF16/FP32) + validate

### 配置

```bash
nohup python train_pids.py \
    --data_dir ./dataset_pol_V4/train \
    --output_dir ./checkpoints_dual_stream_exp16 \
    --dual_stream \
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
    --num_steps 50000 \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --lr 0.0003 \
    --val_freq 500 \
    > train_exp16.log 2>&1 &
```

### 訓練進度 (50K / 50K = 100%)

**Glass EPE 趨勢:**
```
 0.5K steps:  86.61 px (起始)
 5K steps:    45.09 px
10K steps:    44.57 px
14.5K steps:  34.89 px (跌破 35)
16K steps:    33.77 px
19K steps:    30.25 px (跌破 30)
20.5K steps:  28.17 px
23.5K steps:  26.34 px
27K steps:    26.09 px
28K steps:    25.40 px
31K steps:    24.38 px
36K steps:    22.76 px (跌破 23)
42K steps:    21.53 px
45K steps:    21.67 px
50K steps:    21.82 px <- 最終結果
```

**Val Loss 趨勢:**
```
 0.5K steps:  690.34 (起始)
16K steps:    253.00 (接近 baseline)
23.5K steps:  199.86 (跌破 200)
27K steps:    193.15
31.5K steps:  182.98
36K steps:    173.86 (跌破 175)
42K steps:    162.61
45.5K steps:  161.59
50K steps:    160.51 <- 最終結果
```

### 對比歷史實驗

| 實驗 | Glass EPE | Val Loss | vs Baseline 改善 |
|------|-----------|----------|------------------|
| Exp #13 (PIDS 隱式) | 39.20 px | 231.65 | -7.0% |
| Exp #14 (Baseline 無偏振) | 42.17 px | 248.59 | -- |
| **Exp #16 (Dual-Stream)** | **21.82 px** | **160.51** | **-48.3%** |

### 關鍵發現

1. 維度 bug 修正後，Dual-Stream 架構效果顯著
2. Glass EPE: 42.17 -> 21.82 px，**改善 48.3%**
3. Val Loss: 248.59 -> 160.51，**改善 35.4%**
4. 訓練穩定，前 20K 快速收斂，後期穩定
5. 最終 Glass EPE 21.82 px 是歷史最佳

### 技術貢獻總結

1. **Dual-Stream Polarization Architecture**
   - Stereo Stream: 標準 RAFT-Stereo 特徵編碼器
   - Polarization Stream: 專用偏振特徵編碼器
     - Soft Threshold: sigmoid(kappa(P-tau)) 可學習閾值
     - ResidualBlock: 改善梯度流動
     - SpatialAttention: 自動學習關注玻璃區域

2. **Disparity-Aligned Polarization Difference**
   - 使用 GT disparity 將右圖 warp 到左圖視角
   - 計算純偏振差異，消除視差造成的假差異

3. **Polarization-Aware Loss**
   - 偏振差異大的區域額外加權 (pol_weight=2.0)
   - 玻璃區域加權 (glass_weight=5.0)

4. **Bug 修正**
   - flow->disparity 維度修正: (B,2,H,W) -> (B,1,H,W)

**狀態:** 訓練完成，最終 Glass EPE 21.82 px，Val Loss 160.51

---

