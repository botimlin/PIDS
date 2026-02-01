## Exp #42: Numerical Stability Test (2026-01-30)

### 動機

在不同 GPU 上評估 Baseline (RAFT-Stereo) 時發現結果大幅不同（後經 Part B 確認為 TF32 行為差異）。

### 實驗結果

#### Part A: 同卡穩定性測試 (RTX 4090)

5 組精度設定 × 2 架構 = 10 次 evaluation：

| 組別 | TF32 | Benchmark | AMP | 說明 |
|------|------|-----------|-----|------|
| C1 | ON | ON | OFF | 預設 (FP32) |
| C2 | OFF | ON | OFF | 關 TF32 |
| C3 | ON | OFF | OFF | 關 benchmark |
| C4 | OFF | OFF | OFF | 全確定性 |
| C5 | ON | ON | ON | Mixed Precision (FP16) |

**結果 — Glass EPE Median (px)：**

```
Method              C1       C2       C3       C4       C5        σ    CV(%)
────────────────────────────────────────────────────────────────────────────
RAFT-Stereo       5.264    5.251    5.258    5.251    5.262    0.006   0.12%
PIDS (Ours)       2.376    2.377    2.335    2.377    2.331    0.024   1.01%
────────────────────────────────────────────────────────────────────────────
```

**結論**：4090 上 TF32、benchmark、AMP 均不影響結果（CV < 1.1%）。

#### Part B: H200 穩定性測試（根因確認）

在**另一台 H200**（非訓練用的原 H200）上跑同樣的 C1-C5：

```
Method              C1       C2       C3       C4       C5        σ    CV(%)
────────────────────────────────────────────────────────────────────────────
RAFT-Stereo       3.504    5.250    3.501    5.250    3.501    0.958  22.79%
────────────────────────────────────────────────────────────────────────────
                  TF32✓    TF32✗    TF32✓    TF32✗    TF32✓
```

**根因確認：TF32 在 Hopper 架構上的行為**

| 條件 | TF32 | Bench | H200 結果 | 4090 結果 | 分析 |
|------|------|-------|-----------|-----------|------|
| C1 | ON | ON | 3.504 px | 5.264 px | **TF32 ON: H200 與 4090 差異巨大** |
| C2 | OFF | ON | 5.250 px | 5.251 px | **TF32 OFF: 完全一致** |
| C3 | ON | OFF | 3.501 px | 5.258 px | Benchmark 無影響 |
| C4 | OFF | OFF | 5.250 px | 5.251 px | **FP32 基準: 跨平台一致** |
| C5 | ON | ON | 3.501 px | 5.262 px | AMP 跟隨 TF32 行為 |

關鍵結論：

1. **Benchmark 完全無影響**（C1≈C3, C2≈C4）
2. **TF32 是唯一變數**：H200 上 TF32 ON → 3.50, OFF → 5.25（差 50%）
3. **4090 上 TF32 無影響**（CV 0.12%）
4. **FP32 (TF32=OFF) 跨平台完全一致**：H200 5.250 ≈ 4090 5.251
5. Outlier Rate: TF32 ON → 4.1% (35 scenes), TF32 OFF → 11.4% (98 scenes)

#### Part C: RTX 5090 (Blackwell) 測試 — 完成

```
Method              C1       C2       C3       C4       C5        σ    CV(%)
────────────────────────────────────────────────────────────────────────────
RAFT-Stereo       3.498    5.250    3.515    5.250    3.504    0.956  22.73%
────────────────────────────────────────────────────────────────────────────
                  TF32✓    TF32✗    TF32✓    TF32✗    TF32✓
```

**結論：Blackwell = Hopper 行為**（CV 22.73% ≈ 22.79%），Ada Lovelace 是唯一 TF32 不影響的架構。

#### 三代架構完整對照

```
                C1       C2       C3       C4       C5        σ    CV(%)
               TF32✓    TF32✗    TF32✓    TF32✗    TF32✓
────────────────────────────────────────────────────────────────────────
4090 (Ada)    5.264    5.251    5.258    5.251    5.262    0.006   0.12%
H200 (Hopper) 3.504    5.250    3.501    5.250    3.501    0.958  22.79%
5090 (Black.) 3.498    5.250    3.515    5.250    3.504    0.956  22.73%
────────────────────────────────────────────────────────────────────────
FP32 基準:    全部 5.250 ± 0.001 — 三代架構完全一致
```

### 根因分析

```
問題鏈:
TF32 Tensor Core (Hopper / Blackwell)
  → matmul 精度: FP32 mantissa 23-bit 截斷為 10-bit
  → conv 結果微小差異
  → GRU × 32 iterations 放大
  → Baseline 最終結果從 5.25 → 3.50 px (Hopper + Blackwell)

為什麼 4090 (Ada Lovelace) 不受影響:
  → Ada 的 TF32 實作與 Hopper/Blackwell 不同
  → 在 Ada 上 TF32 的擾動幅度遠小於 Hopper/Blackwell
  → CV 0.12% vs 22.79%/22.73%
  → Ada Lovelace 是唯一不受 TF32 影響的架構

為什麼 Hopper TF32 結果反而更好 (3.50 < 5.25):
  → TF32 精度截斷起到「隱式正則化」效果
  → 壓制了 GRU 迭代中的數值發散
  → 不是系統性改善，是特定硬體+模型的巧合
  → 證據: Outlier 從 98 降到 35，不像隨機擾動
```

### 原始假說 vs 實際根因

| | 原始假說 | 實際根因 |
|---|---|---|
| **原因** | GPU 架構差異 (cuDNN 算法選擇) | TF32 在 Hopper 上的行為 |
| **機制** | 不同卷積算法 → 不同累加順序 | TF32 精度截斷 → 隱式正則化 |
| **解法** | 難以統一 | **關閉 TF32 即可** |
| **跨平台** | 不確定 | FP32 完全一致 (5.250 vs 5.251) |

### 論文呈現

**方案（已更新）**：在 evaluation 程式碼中預設關閉 TF32：

```python
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

統一使用 FP32 結果：所有平台 Baseline Median ≈ 5.25 px。

Supplementary 放三代架構對照表（Ada / Hopper / Blackwell）。

### 與其他實驗的關係

```
Exp #39 (V2-E baseline)
    │
    ├── Exp #40 (+ Robust Checkpoint)
    │       │
    │       └── Exp #41 (+ DID) [🧊 凍結]
    │
    └── Exp #42 (Stability Test)  ← 本實驗
            │
            ├── Part A: 4090 同卡 (完成)
            ├── Part B: H200 跨卡 + TF32 根因 (完成)
            └── Part C: RTX 5090 Blackwell (完成)
```

### 測試命令

```bash
# Baseline only (無 V2-E)
python run_stability_test.py \
    --nopol_data_dir ./dataset_nopol_test \
    --baseline_ckpt ./checkpoints_baseline_exp22/baseline_best.pth \
    --output_dir ./eval_results/stability_test

# 含 V2-E
python run_stability_test.py \
    --pol_data_dir ./dataset_pol_test \
    --nopol_data_dir ./dataset_nopol_test \
    --baseline_ckpt ./checkpoints_baseline_exp22/baseline_best.pth \
    --v2e_ckpt ./checkpoints_exp39_v2e/checkpoint_best.pth \
    --output_dir ./eval_results/stability_test
```

### 狀態

**全部完成。根因：TF32。解法：evaluation 預設關閉 TF32。三代架構驗證完畢。**

---

*建立日期: 2026-01-30*
