## Exp #38: Polarization Volume V2-D (2026-01-28)

### 為什麼需要 V2-D？（關鍵轉折）

經過 V3/V3-B 的失敗，確立了三條鐵律：

1. ❌ **pol 不進 fnet** (feature extraction) - V3/V3-B 已證明
2. ❌ **pol 不只做 spatial [H,W] gating** - 不知道「是哪個 disparity」
3. ✅ **pol 只能在 disparity-aware 空間作用** (cost volume, disparity index)

### V2-D 的核心創新

**關鍵轉念**：

> pol 不是告訴模型「這裡重要」(V2-C: spatial gate)
> 而是告訴模型「這個 disparity 不合理」(V2-D: per-disparity gate)

這是 V2 系列的質變：

| | V2-C | V2-D |
|--|------|------|
| gate 維度 | [H,W] | [H,W,D] |
| 能否區分 disparity | ❌ | ✅ |
| pol 角色 | importance mask | **validity judge** |
| 是否符合 stereo 物理 | 部分 | 完全符合 |

### 架構設計

**Step 1: Disparity-Aware Pol Volume**

```python
# 對於每個 disparity 候選 d:
right_at_d = sample(right, x - d)  # 右圖在 disparity d 處的值
pol_diff_d = left - right_at_d     # disparity-aware pol_diff

# 物理意義：
# - d = d_gt (正確): pol_diff 反映 material 特性
# - d ≠ d_gt (錯誤/假匹配): pol_diff 反映幾何錯位
```

**Step 2: Per-Disparity Gate (3D Conv)**

```python
self.pol_gate = nn.Sequential(
    nn.Conv3d(3, 8, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv3d(8, 1, kernel_size=1),
    nn.Sigmoid()
)
# 輸入: pol_volume [B, 3, D, H, W]
# 輸出: gate [B, D, H, W]
```

**Step 3: Residual Modulation**

```python
# 不破壞 RGB stereo baseline
corr_mod = corr * (1.0 + alpha * (gate - 0.5) * 2)
# gate=0.5 為中性，<0.5 抑制，>0.5 增強
```

### 為什麼這有機會破 4 → 3.x px？

1. **解決 gross error (D1/D3)**：不是 smooth refinement，而是「抑制假匹配」
2. **物理一致**：pol 提供的是「這個 disparity 假設下，左右是否來自同一物理點」
3. **回答 reviewer 致命問題**：「pol 到底在哪一步提供了 RGB 做不到的資訊？」

### 訓練命令

```bash
nohup python train_pids.py \
    --data_dir /workspace/RAFT-Stereo/mixed_dataset_exp28/data \
    --output_dir ./checkpoints_exp38_v2d \
    --pol_volume_v2d \
    --pol_gate_hidden 8 \
    --pol_alpha 0.2 \
    --curriculum \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 60000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 500 --num_workers 4 \
    > train_exp38.log 2>&1 &
```

### 目標

| 實驗 | 架構 | Glass EPE | 說明 |
|------|------|-----------|------|
| Exp #24 | V1 (Concat) | 4.055 px | 當前最佳 |
| Exp #35 | V2-C | 4.745 px | V2 系列最佳 |
| Exp #36-37 | V3/V3-B | ❌ 失敗 | early fusion 方向錯誤 |
| **Exp #38** | **V2-D** | **< 4.0 px** | **目標：disparity 判別** |

### 訓練進度

**Step 23500 Val 結果：**
- Glass EPE: **9.879 px**
- EPE: 6.017 px
- D1: 35.53%
- Composite: 13.432 (New best)

**觀察：**
1. Train-Val gap 後期改善（Train ~10-11 px, Val ~9.9 px）
2. Val Glass EPE 從早期 40+ px 降到 ~10 px
3. **但曲線與 V2-A 後期重疊度很高**

### 結論：提早停止 @ Step 23500

**決定**：在 Step 23500 提早停止訓練，原因如下：

1. **曲線收斂趨勢明確**：V2-D 與 V2-A 後期曲線重疊，表明兩者天花板相同
2. **Post-Corr 系列已到極限**：V2-A/B/C/D 四個變體本質相同（post-corr intervention），只是修正方式不同，但 ceiling 一致
3. **GPU 時間更值得給 V2-E**：V2-E 是 pre-corr 介入，是真正不同的假說

**V2-D 的貢獻**：
- 驗證了 disparity-aware gating（3D Conv）的可行性
- 但未能突破 post-corr intervention 的天花板
- 進一步確認：**V2 系列的瓶頸不在「怎麼修」，而在「什麼時候修」**

---

