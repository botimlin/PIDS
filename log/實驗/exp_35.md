## Exp #35: Polarization Volume V2-C (2026-01-27)

### 架構改進：Gradient Gating

**核心概念**：用 disparity gradient 作為 uncertainty proxy，讓 pol 在「不確定區域」有更大話語權。

```python
# V2-C: α * gate * residual
disp_grad = compute_gradient(disp).detach()  # stop-grad 避免 feedback loop!
gate = GatingNetwork(pol_corr, disp_grad)    # 極小網絡，sigmoid → [0, 1]
alpha = i / max(iters - 1, 1)                 # 保留 V2-B 的 schedule
corr_enhanced = corr + alpha * gate * pol_residual(pol_corr)
```

**設計要點**：

1. **disp_grad.detach()** - 避免 feedback loop，讓它成為 structural cue 而非 learnable shortcut
2. **GatingNetwork 極小** (2層 conv, 無 BN, 無 attention) - mechanism proof, not black box
3. **α * gate** - 解決 early iteration disp_grad 是噪音的問題

**vs 其他版本**：
| 版本 | 公式 | 說明 |
|------|------|------|
| V2-A | `corr + residual` | static，全程相同 |
| V2-B | `corr + α * residual` | when (iteration schedule) |
| V2-C | `corr + α * gate * residual` | when + where (gradient gating) |

### 訓練命令

```bash
nohup python train_pids.py \
    --data_dir /workspace/RAFT-Stereo/mixed_dataset_exp28/data \
    --output_dir ./checkpoints_exp35_v2c \
    --pol_volume_v2c \
    --curriculum \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 60000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 500 --num_workers 4 \
    > train_exp35.log 2>&1 &
```

### 排程

- **執行目錄**: `/workspace/v2-c` (獨立目錄，避免影響 V2-B 訓練)
- **排程時間**: 19:47 (V2-B 完成後自動啟動)
- **命令**: `at 19:47`

### 目標

| 實驗 | 架構 | Glass EPE | 說明 |
|------|------|-----------|------|
| Exp #24 | V1 (Concat) | 4.055 px | 當前最佳 |
| Exp #33 | V2-A (Static) | 5.164 px | 次於 V1 |
| Exp #34 | V2-B (Scheduled) | ~5.0 px | 預估，與 V2-A 相近 |
| **Exp #35** | **V2-C (Gradient Gating)** | **< 4.0 px** | **目標：超越 V1** |

### 訓練進度

| Step | Glass EPE | 進度 |
|------|-----------|------|
| 10000 | 14.63 px | 17% |
| 20000 | 9.92 px | 33% |
| 30000 | 7.72 px | 50% |
| 40000 | 7.49 px | 67% |
| 50000 | 5.60 px | 83% |

**觀察**: V2-C 表現反而比 V2-A/V2-B 稍差（同期 ~5.6 px vs ~5.0 px）。

---

## V2 系列總結與關鍵洞察 (2026-01-28)

### Exp #34 V2-B 最終結果

| 指標 | 結果 |
|------|------|
| Best Val Glass EPE | **5.138 px** |
| Best Composite | 7.536 |

### V2 系列對比（測試集結果）

| 架構 | Val Glass EPE | Test Glass EPE | 說明 |
|------|---------------|----------------|------|
| V1 (Concat) | - | **4.055 px** | **最佳** |
| V2-A (Static Residual) | 4.869 px | 5.164 px | |
| V2-B (Scheduled Residual) | 5.138 px | 5.258 px | 與 V2-A 持平 |
| V2-C (Gradient Gating) | 5.025 px | **4.745 px** | V2 系列最佳，接近 V1 |

**觀察**: V2-C 在測試集上表現意外地好 (4.745 px)，接近 V1 (4.055 px)，差距約 0.7 px。

### 關鍵技術洞察

#### V2-B 前期領先、後期打平的含義

**現象**：V2-B 前期收斂較快，但最終與 V2-A 打平。

**技術含義**：
- **前期領先**：α schedule 讓 stereo 先穩定，不被 pol 干擾 → early EPE 下降快
- **後期打平**：一旦 corr 定型，pol 只能做 late correction → representation ceiling 相同

**結論**：
> V2-B 不是更強的模型，只是更好訓練的同一個模型。

這代表「設計是健康的，但方向已經走到牆前面了」。

#### V2-C 為什麼連 early advantage 都沒了？

**問題根源**：

1. **disp_grad 來自不穩定的中間產物**
   - Early stage 的 disp 很爛
   - grad = noise
   - gate 學不到「不確定區域」，只學到亂七八糟的 activation

2. **仍然是 post-corr intervention**
   - 不管 gate 再精準，本質仍然是：matching 做完了 → 才讓 pol 說話
   - representation ceiling 被 corr 鎖死

**結果**：
- Early：學習難度↑（比 V2-A/B 慢）
- Late：上限仍被鎖死

#### V2 系列的根本問題

```
residual / gating = too late
matching 做完了 → pol 只能做 correction
representation ceiling 被 corr 鎖死在 ~5 px
```

---

