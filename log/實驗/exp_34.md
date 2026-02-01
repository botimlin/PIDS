## Exp #34: Polarization Volume V2-B (2026-01-27)

### 架構改進：Scheduled Residual

**核心改動**：讓 residual 強度隨 iteration 增加

```python
# V2-A (static)
corr_enhanced = corr + self.pol_residual(pol_corr)

# V2-B (scheduled)
alpha = i / max(iters - 1, 1)  # 0 → 1
corr_enhanced = corr + alpha * self.pol_residual(pol_corr)
```

**設計哲學**：V2-B 的 α schedule 等於做了三件事（V1 concat 隱性做到的）

| Phase | α 值 | 作用 | 說明 |
|-------|------|------|------|
| **Early** | α ≈ 0 | 保護 stereo geometry | 等價於純 RAFT-Stereo，pretrained stereo 不被偏振破壞 |
| **Mid** | α 漸增 | pol 成為輔助證據 | stereo 已有 reasonable disparity，pol 只做 nudging（邊界、specular） |
| **Late** | α → 1 | pol = refinement prior | RAFT 本來就在做 small correction，此時 pol 的 scale/semantic/timing 都對 |

👉 **這修復了 V2-A 的核心問題**：讓 pol 從「全程干擾項」變成「正確時機的 refinement tool」

### 訓練命令

```bash
nohup python train_pids.py \
    --data_dir ./mixed_dataset_exp28/data \
    --output_dir ./checkpoints_exp34_v2b \
    --pol_volume_v2b \
    --curriculum \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 60000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 500 --num_workers 4 \
    > train_exp34.log 2>&1 &
```

### 目標

| 實驗 | 架構 | Glass EPE | 說明 |
|------|------|-----------|------|
| Exp #24 | V1 (Concat) | 4.055 px | 當前最佳 |
| Exp #33 | V2-A (Static) | 5.164 px | 次於 V1 |
| **Exp #34** | **V2-B (Scheduled)** | **< 4.0 px** | **目標：超越 V1** |

### 訓練進度

| Step | Val Glass EPE | 進度 | 備註 |
|------|---------------|------|------|
| 25000 | 8.214 px | 42% | |
| 25500 | 8.588 px | 43% | |
| 48500 | 5.895 px | 81% | 與 V2-A 同期相近 |

**觀察**: V2-B 與 V2-A 表現接近，α schedule 未帶來顯著突破。預估最終約 5.0-5.2 px。

---

