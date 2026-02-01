## Exp #33: Polarization Volume V2-A (2026-01-27)

### 設計理念

**錯誤方向** (V2):
```
pol_corr → Spatial Attention → Multiplicative Gate
         (不知道哪個 d 重要)    (破壞 RAFT bias)
```

**正確方向** (V2-A):
```
pol_corr → PolCorrResidual → Δcorr
corr_enhanced = corr + α × Δcorr  (Additive Bias in Disparity Space)
```

### 架構圖

```
left ──→ [FeatureEncoder] ──→ fmap1 ─┐
                                      ├──→ [CorrBlock] ─────┐
right ─→ [FeatureEncoder] ──→ fmap2 ─┘                      │
                                                             │
left ──→ [Downsample 1/4] ───────────┐                      │
                                      ├──→ [PolCorrBlock] ──┼──→ [PolCorrResidual] ──┐
right ─→ [Downsample 1/4] ───────────┘                      │                        │
                                                             ▼                        ▼
                                                    ┌────────────────────────────────────┐
                                                    │  corr_enhanced = corr + α × Δcorr  │
                                                    └────────────────────────────────────┘
                                                                      │
                                                                      ▼
                                                    ┌─────────────────────────────────┐
                                                    │      UpdateBlock (原始 RAFT)     │
                                                    └─────────────────────────────────┘
```

### PolCorrResidual 模組

```python
class PolCorrResidual(nn.Module):
    def __init__(self, pol_dim, corr_dim, hidden_dim=64, init_scale=0.1):
        self.net = nn.Sequential(
            nn.Conv2d(pol_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, corr_dim, 1),  # 投影到 corr 維度
        )
        self.scale = nn.Parameter(torch.tensor(init_scale))
        # 最後一層初始化為 0 → 初始行為接近 V1
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, pol_corr):
        return self.scale * self.net(pol_corr)
```

### 核心優勢

| 特性 | V1 | V2 | V2-A |
|------|----|----|------|
| Pol 注入 | Concat | Multiplicative | **Additive** |
| 工作空間 | Feature | Spatial | **Disparity** |
| UpdateBlock | 需改動 | 大改動 | **原始 RAFT** |
| RAFT bias | 部分破壞 | 嚴重破壞 | **完全保留** |
| Mixed 穩定性 | 中 | 差 | **極穩** |

### 訓練配置

```bash
nohup python train_pids.py \
    --data_dir ./PIDS_dataset_exp29_mixed/data \
    --output_dir ./checkpoints_exp33_v2a \
    --pol_volume_v2a \
    --curriculum \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 60000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 500 --num_workers 4 \
    > train_exp33.log 2>&1 &
```

### 預期效果

| 實驗 | 架構 | Glass EPE | 說明 |
|------|------|-----------|------|
| Exp #24 | V1 (Concat) | 4.055 px | 基準 |
| Exp #31 | V2 (Attention+Gate) | 7.763 px | 架構錯誤 |
| **Exp #33** | **V2-A (Corr Residual)** | **< 3.8 px** | **目標** |

### 訓練進度

**訓練中觀察** (Step ~14600):

`curriculum/pol_frozen` 分析：
| Step 範圍 | pol_frozen | 說明 |
|-----------|------------|------|
| 10 - 12000 | 1.0 | Phase 1: PolCorrResidual 被凍結 |
| 12020+ | 0.0 | Phase 2+: PolCorrResidual 解凍開始學習 |

**發現問題**: `PolModuleFreezer` 的 `pol_keywords=['pol', 'Pol', 'POL']` 會匹配到 V2-A 的 `pol_residual` 模組，導致 Phase 1 (0-20%) 期間 PolCorrResidual 被意外凍結。

**影響分析**:
- Phase 1 (Step 0-12000): stereo backbone 單獨訓練，pol_residual 凍結
- Phase 2+ (Step 12000+): pol_residual 開始學習

**這可能不是壞事**:
- Phase 1 讓 stereo 先穩定
- Phase 2 再引入 pol residual

**但與 V2-A 設計理念有衝突**:
- V2-A 的 scale 初始化 0.1，最後一層初始化 0
- 理論上初始行為接近 V1，應該可以從頭一起學習

### 後續優化方案

**Option A**: 讓 V2-A 的 `pol_residual` 不被 PolModuleFreezer 凍結
```python
# 在 PolModuleFreezer 中排除 pol_residual
# 或在 train_pids.py 中對 V2-A 跳過 PolModuleFreezer
```

**Option B**: 保持現狀，如果效果好就不改

**決定**: 先繼續實驗，觀察結果再決定。

### 訓練進度追蹤

| Step | Val Glass EPE | 進度 | 備註 |
|------|---------------|------|------|
| 17000 | 11.611 px | 28% | 持續下降 |
| 21000 | 9.791 px | 35% | |
| 24000 | 8.611 px | 40% | |
| 30500 | 7.159 px | 51% | 已超越 V2 (7.763 px) |
| 33500 | 6.878 px | 56% | 訓練中 |

---

## 方法論討論 (2026-01-27)

### Polarization 方法的合理預期

#### 公平的 Baseline 對比

| 實驗 | 架構 | 測試集 | Glass EPE |
|------|------|--------|-----------|
| Exp #22 | RAFT-Stereo (無 pol) | nopol test | **6.543 px** |
| Exp #24 | V1 Pol Volume | pol test | **4.055 px** |

**Polarization 的實際增益**: 6.543 → 4.055 px = **降低 38%**

#### 38% 是否足夠？

**問題**: 對於一個需要額外硬體（偏振片、特殊光源配置）的方法，38% 的提升說服力不足。

**可能的解釋**:
1. V1 (Concat) 是最簡單的 fusion，沒有充分利用 pol 信息
2. nopol baseline 可能 overfitting 到訓練集紋理（訓練時波動大）
3. Polarization 的真正價值可能在於**泛化能力**而非同分布測試集上的提升

**待驗證假設**: nopol 模型可能記住了特定紋理，換到 OOD (Out-of-Distribution) 場景時會崩潰，而 pol 模型依賴物理信號應該更穩定。

#### 目標設定

| 目標等級 | Glass EPE | vs Baseline | 說明 |
|----------|-----------|-------------|------|
| 保守 | < 4.0 px | -39% | 持平 V1 |
| 合理 | < 3.5 px | **-46%** | V2-A/V2-B 目標 |
| 理想 | < 3.0 px | **-54%** | 需要更多架構優化 |

**現階段判斷**: V2-A 持平 V1 (~4 px) 是可能的，但要達到 3 px 可能需要進一步架構更新。

---

## Pol Volume 架構演進路線圖

### 已完成

| 版本 | 架構 | Glass EPE | 狀態 |
|------|------|-----------|------|
| V1 | Concat | 4.055 px | ✅ Baseline |
| V2 | Attention + Gate | 7.763 px | ❌ 失敗 |
| V2-A | Static Residual | 5.164 px | ✅ 完成 (次於 V1) |
| **V2-B** | **Scheduled Residual** | **?** | **🔄 訓練中** |

---


## Exp #33 最終結果 (2026-01-27)

### 訓練完成

| 指標 | Val (Best) | Test |
|------|------------|------|
| Glass EPE | 4.869 px | **5.164 px** |
| D1 | 22.62% | 24.11% |
| EPE | 2.874 px | 3.341 px |

### 結論

- V2-A 比 V2 (7.763 px) 大幅改善
- 但**未超越 V1** (4.055 px)，差距約 1.1 px

### 失敗原因分析（關鍵洞察）

**❌ 次要原因**：Phase 1 意外凍結 pol_residual
- 這會拖慢收斂，但不足以解釋 1.1 px 的落差

**❌ 真正致命點**：Static residual = 在錯的時間，用錯的力道

V2-A 的問題不是「pol 不夠強」，而是：

| Iteration | 問題 |
|-----------|------|
| Early | stereo 還在對齊 coarse geometry，pol residual 直接加進來 = **放大 early noise** |
| Late | RAFT 已經進入 refinement，pol residual 沒有「更重要」，只是「一樣重要」 |

👉 結果：pol 在整個過程中都只是**干擾項**，而不是 **refinement tool**

---

