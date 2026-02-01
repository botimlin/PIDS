## Exp #36: Polarization Volume V3 (2026-01-28)

### 為什麼需要 V3？

V2 系列證明了：**post-corr intervention 有天花板**。

唯一的突破方向是讓 pol 進入 feature extraction，**直接影響 matching 本身**。

### 架構改進：Pol-in-Feature (Early Fusion)

**核心改動**：

```python
# 原本 (V1/V2 系列)
fmap1 = fnet(left)   # 3 channels
fmap2 = fnet(right)  # 3 channels
# pol 只能在 corr 形成後介入

# V3 (Pol-in-Feature)
pol_diff = left - right
fmap1 = fnet(concat(left, pol_diff))   # 6 channels
fmap2 = fnet(concat(right, pol_diff))  # 6 channels
# pol 直接影響 feature extraction
```

**設計原則**：
- ✅ 不加新 branch
- ✅ 不加 attention
- ✅ 不加 residual
- ✅ 不加 gating
- ✅ 只問一件事：**pol 是否能影響 feature matching**

**預期效果**：
- 如果 V3 > V1：證明 early fusion 有效
- 如果 V3 ≤ V1：說明 pol 信息本身不足以改善 feature matching

### 訓練命令

```bash
nohup python train_pids.py \
    --data_dir /workspace/RAFT-Stereo/mixed_dataset_exp28/data \
    --output_dir ./checkpoints_exp36_v3 \
    --pol_volume_v3 \
    --curriculum \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 60000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 500 --num_workers 4 \
    > train_exp36.log 2>&1 &
```

**注意**：fnet 輸入從 3→6 channels，pretrained 的 `fnet.conv1` 不匹配會跳過，需要重新學習。

### 目標

| 實驗 | 架構 | Glass EPE | 說明 |
|------|------|-----------|------|
| Exp #24 | V1 (Concat) | 4.055 px | 當前最佳 |
| Exp #33-35 | V2 系列 | ~5.1-5.6 px | post-corr 天花板 |
| **Exp #36** | **V3 (Pol-in-Feature)** | **< 4.0 px** | **目標：超越 V1** |

### 訓練進度

**Step 10000 觀察到嚴重問題：**

| 指標 | Train | Val |
|------|-------|-----|
| Glass EPE | ~18-20 px | 24→29 px (不穩定) |
| EPE | ~10-11 px | 17→21→36 px (劇烈波動) |
| D1/D3 | 48-49% | 91-92% (居高不下) |

**對比 V2-C 同期（Step 10000）：**
- V2-C Val Glass EPE: ~14.6 px
- V3 Val Glass EPE: ~29 px ❌

### 問題診斷

**V3 失敗原因**：6ch concat 破壞了 pretrained weights

1. `conv1` 輸入從 3ch → 6ch
2. pretrained 的 `conv1` weights 無法使用
3. 必須重新學習整個 feature encoder
4. 導致 D1/D3 居高不下（基礎幾何能力喪失）
5. Val 極不穩定（pretrained 主導能力被破壞）

**結論**：V3 (6ch concat) 的設計是錯誤的方向，需要修正。

### 結果

| 狀態 | 說明 |
|------|------|
| ❌ 失敗 | 訓練在 Step ~10000 時中止 |
| 原因 | Val 不健康，D1/D3 居高不下 |

---

