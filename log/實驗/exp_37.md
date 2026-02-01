## Exp #37: Polarization Volume V3-B (2026-01-28)

### 為什麼需要 V3-B？

V3 的核心想法是對的（pol 進入 feature extraction），但實作方式錯誤：

| | V3 (失敗) | V3-B (修正) |
|--|--|--|
| fnet input | 6ch (concat) | 3ch (保留) |
| pretrained conv1 | ❌ 破壞 | ✅ 完整保留 |
| pol 注入方式 | 強制融合 | soft additive |
| 預期 D1/D3 | 高（不穩定） | 正常（穩定） |

### 架構改進：Additive Pol Fusion

**核心設計**：

```
原始 3ch input (保留 pretrained)
      │
      ▼
   conv1 ──────────────► 64ch ─┐
                               │ + pol_scale * pol_feat
   pol_diff (3ch)              │
      │                        │
      ▼                        │
   pol_conv1 (random init) ──► 64ch ─┘
                                    │
                                    ▼
                              layer1 → layer2 → layer3 → out
```

**程式碼**：

```python
class FeatureEncoderWithPolFusion(nn.Module):
    def __init__(self, output_dim=128, pol_scale=0.1):
        # 主分支（保留 pretrained）
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)

        # Pol side branch（獨立學習）
        self.pol_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)

        # 共享後續層
        self.layer1, self.layer2, self.layer3 = ...

    def forward(self, img, pol_diff):
        x = self.relu1(self.norm1(self.conv1(img)))  # pretrained
        pol_feat = self.pol_relu1(self.pol_norm1(self.pol_conv1(pol_diff)))  # random init
        x = x + self.pol_scale * pol_feat  # soft additive fusion
        return self.conv_out(self.layer3(self.layer2(self.layer1(x))))
```

**設計原則**：
- ✅ 完整保留 pretrained fnet.conv1（幾何能力）
- ✅ 獨立 pol_conv1 學習 pol 表示
- ✅ soft additive fusion（pol_scale=0.1）讓 pretrained 主導
- ✅ 後續層共享（layer1/2/3），pol 信息逐漸融入

### 訓練命令

```bash
nohup python train_pids.py \
    --data_dir /workspace/RAFT-Stereo/mixed_dataset_exp28/data \
    --output_dir ./checkpoints_exp37_v3b \
    --pol_volume_v3b \
    --pol_scale 0.1 \
    --curriculum \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 60000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 500 --num_workers 4 \
    > train_exp37.log 2>&1 &
```

### 目標

| 實驗 | 架構 | Glass EPE | 說明 |
|------|------|-----------|------|
| Exp #24 | V1 (Concat) | 4.055 px | 當前最佳 |
| Exp #35 | V2-C | 4.745 px | V2 系列最佳 |
| Exp #36 | V3 | ❌ 失敗 | 6ch concat 破壞 pretrained |
| **Exp #37** | **V3-B** | **< 4.0 px** | **目標：超越 V1** |

### 訓練進度

**已終止** - Val 表現不健康，與 V3 相同的問題。

Early fusion 方向被徹底否定：
- pol_diff 在 feature extraction 階段不是有用的信號
- Feature encoder 學的是「匹配特徵」
- pol_diff 是像素級偏振差異，強行融合 = 注入噪音

---

