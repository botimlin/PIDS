## 實驗 #17: 大規模數據訓練 (5000 樣本)

**日期**: 2026-01-05
**狀態**: 訓練完成

### 實驗目標

使用 5000 場景的偏振數據訓練，驗證更大數據量對模型性能的影響。

### 數據準備

- **渲染**: 使用 `pids_renderer_textured.py` (v4.0.0)
- **場景生成**: `blender_furniture_randomizer_v18.py`
- **QA 篩選**: `quality_validator.py --skip-c1`
- **最終樣本**: 5000 場景

### 訓練配置

基於 Exp #16 設定，調整步數以匹配更大數據量：

```bash
nohup python train_pids.py \
    --data_dir ./dataset_pol \
    --output_dir ./checkpoints_dual_stream_exp17 \
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
    --num_steps 70000 \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --lr 0.0003 \
    --val_freq 500 \
    > train_exp17.log 2>&1 &
```

### 與 Exp #16 對比

| 參數 | Exp #16 | Exp #17 | 說明 |
|------|---------|---------|------|
| 訓練樣本 | 3500 | **5000** | +43% |
| num_steps | 50000 | **70000** | +40% (維持相近 epoch 數) |
| 其他參數 | - | 相同 | - |

### Epoch 計算

- Exp #16: 50000 × 8 / 3500 ≈ 114 epochs
- Exp #17: 70000 × 8 / 5000 = 112 epochs (相近)

### 預期改善

| 指標 | Exp #16 | 預期值 |
|------|---------|--------|
| Glass EPE | 21.82 px | < 20 px |
| Val Loss | 160.51 | < 150 |

### 訓練進度

(待訓練開始後更新)

**狀態:** 準備中

---

