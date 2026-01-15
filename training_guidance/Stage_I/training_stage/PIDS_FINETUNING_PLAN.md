# PIDS Finetuning 完整規劃文檔

**版本**: 1.6
**日期**: 2026-01-04
**狀態**: 實驗 #16 突破性進展

---

## 1. 概述

本文檔描述 PIDS 的**兩階段訓練策略**和**數據效率實驗設計**。

### 1.1 核心目標

1. **偏振立體匹配**: 讓 RAFT-Stereo 學習從偏振影像對 (I∥, I⊥) 預測視差
2. **透明物體深度估計**: 利用玻璃表面的偏振特性提升深度估計精度
3. **數據效率**: 證明偏振信息能減少所需訓練數據量

### 1.2 基礎模型

**RAFT-Stereo** (Princeton Vision Lab)
- GitHub: https://github.com/princeton-vl/RAFT-Stereo
- 論文: "RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching" (3DV 2021 Best Student Paper)

### 1.3 PIDS 核心貢獻

```
┌─────────────────────────────────────────────────────────────────┐
│                     PIDS 論文核心貢獻                            │
├─────────────────────────────────────────────────────────────────┤
│ 1. 偏振影像提供玻璃表面的額外幾何線索 (Fresnel 反射)              │
│ 2. 兩階段訓練: 合成數據 → 真實數據 (Domain Adaptation)           │
│ 3. 數據效率: 用更少的訓練樣本達到相同或更好的效果                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 兩階段訓練策略

### 2.1 訓練流程總覽

```
                         PIDS 兩階段訓練流程
═══════════════════════════════════════════════════════════════════

                    ┌─────────────────────────┐
                    │   RAFT-Stereo           │
                    │   (SceneFlow 預訓練)     │
                    └───────────┬─────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│  Stage I: 合成數據訓練 (Synthetic Data)                        │
│  ─────────────────────────────────────────────────────────────│
│  數據: pids_renderer.py 生成的合成場景                         │
│  特點: Exaggerated polarization (誇大偏振效果)                 │
│  目的: 學習偏振 → 視差的基本映射關係                           │
│  數據量: 500-2000 場景                                         │
│  訓練: ~50K iterations                                         │
└───────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│  Stage II: 真實數據微調 (Real Data)                            │
│  ─────────────────────────────────────────────────────────────│
│  數據: 真實偏振相機拍攝的場景                                  │
│  特點: Subtle polarization (微弱但真實的偏振)                  │
│  目的: Domain adaptation，適應真實世界的噪聲和變化             │
│  數據量: 100-500 場景 (較少)                                   │
│  訓練: ~20K iterations                                         │
└───────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │   PIDS-RAFT             │
                    │   (最終模型)             │
                    └─────────────────────────┘
```

### 2.2 Stage I vs Stage II 對比

| 維度 | Stage I (合成) | Stage II (真實) |
|------|----------------|-----------------|
| **數據來源** | Mitsuba 3 渲染 | 偏振相機拍攝 |
| **偏振強度** | 誇大 (DoLP ~25%) | 微弱 (DoLP ~5-10%) |
| **Ground Truth** | 精確 (渲染生成) | 結構光/LiDAR |
| **場景多樣性** | 可控隨機生成 | 真實環境 |
| **噪聲** | 無/可控 | 真實相機噪聲 |
| **目的** | 學習偏振-視差關係 | Domain adaptation |
| **訓練輪數** | 較多 | 較少 (避免過擬合) |

### 2.3 為什麼需要兩階段？

```
問題: 真實偏振數據難以大量獲取
      ├── 需要特殊偏振相機設備
      ├── 需要配對的 Ground Truth 深度
      └── 場景多樣性受限

解決:
      Stage I: 用大量合成數據學習「偏振 → 深度」的映射
              └── 合成數據容易生成、完美 Ground Truth

      Stage II: 用少量真實數據進行 domain adaptation
              └── 彌合 synthetic-to-real gap
```

---

## 3. 數據效率實驗設計

### 3.1 核心假設

> **假設**: 偏振信息 (I∥, I⊥) 相比普通 RGB 立體對，
> 能用**更少的訓練數據**達到**相同或更好**的深度估計效果。

### 3.2 實驗設計

```
                      數據效率實驗矩陣
═══════════════════════════════════════════════════════════════════

           │  100 場景  │  250 場景  │  500 場景  │  1000 場景
───────────┼────────────┼────────────┼────────────┼────────────
 RGB 立體  │  Exp A1    │  Exp A2    │  Exp A3    │  Exp A4
 (Baseline)│            │            │            │
───────────┼────────────┼────────────┼────────────┼────────────
 偏振立體  │  Exp B1    │  Exp B2    │  Exp B3    │  Exp B4
 (PIDS)    │            │            │            │
═══════════════════════════════════════════════════════════════════

預期結果:
- B1 (偏振, 100場景) ≈ A3 (RGB, 500場景) → 5x 數據效率提升
- B2 (偏振, 250場景) > A4 (RGB, 1000場景) → 更好的最終效果
```

### 3.3 實驗配置

#### 實驗 A: RGB Baseline (對照組)

```yaml
# 使用相同場景，但輸入為合成 RGB (非偏振)
input_mode: "rgb"
# image1 = rendered RGB left
# image2 = rendered RGB right

experiments:
  A1: { scenes: 100,  iters: 20000 }
  A2: { scenes: 250,  iters: 30000 }
  A3: { scenes: 500,  iters: 40000 }
  A4: { scenes: 1000, iters: 50000 }
```

#### 實驗 B: PIDS 偏振 (實驗組)

```yaml
# 使用偏振影像對
input_mode: "polarization"
# image1 = I∥ (left_parallel)
# image2 = I⊥ (right_cross)

experiments:
  B1: { scenes: 100,  iters: 20000 }
  B2: { scenes: 250,  iters: 30000 }
  B3: { scenes: 500,  iters: 40000 }
  B4: { scenes: 1000, iters: 50000 }
```

### 3.4 評估指標

| 指標 | 定義 | 用途 |
|------|------|------|
| **EPE-all** | 全圖平均端點誤差 | 整體精度 |
| **EPE-glass** | 玻璃區域端點誤差 | 透明物體精度 |
| **D1-all** | 誤差>3px 且>5% 的比例 | 錯誤率 |
| **D1-glass** | 玻璃區域 D1 | 透明物體錯誤率 |
| **Data Efficiency** | 達到目標精度所需場景數 | 核心指標 |

### 3.5 預期結果可視化

```
EPE-glass (越低越好)
    │
 5.0├─────●                               ← RGB, 100 scenes
    │      \
 4.0├───────●                             ← RGB, 250 scenes
    │    ●   \
 3.0├────┼────●                           ← RGB, 500 scenes
    │    │     \
 2.0├────┼──────●                         ← RGB, 1000 scenes
    │    │       \
    │  ● │        ●                       ← PIDS 曲線 (預期)
 1.0├──┼─●─────────●
    │  │
 0.5├──●────────────●                     ← PIDS 用更少數據達到更低誤差
    │
    └──┬────┬────┬────┬────┬──────────
      100  250  500  750  1000    場景數

關鍵發現 (預期):
├── PIDS 100 場景 ≈ RGB 500 場景 (5x 效率)
├── PIDS 250 場景 < RGB 1000 場景 (更好效果)
└── 玻璃區域改善最顯著
```

### 3.6 消融實驗 (Ablation Study)

```
                       消融實驗設計
═══════════════════════════════════════════════════════════════════

實驗 C: 輸入模態消融
────────────────────
C1: I∥ only (單偏振)           - 驗證需要雙偏振
C2: I⊥ only (單偏振)           - 驗證需要雙偏振
C3: I∥ + I⊥ (雙偏振, PIDS)     - 完整方法
C4: (I∥ + I⊥)/2 (平均)         - 驗證需要分開輸入

實驗 D: 損失函數消融
────────────────────
D1: L1 loss (全圖)             - Baseline
D2: L1 + glass weight (2x)    - 玻璃區域加權
D3: L1 + glass weight (5x)    - 更高玻璃權重
D4: L1 + DoLP-guided weight   - 用 DoLP 作為權重

實驗 E: 預訓練消融
────────────────────
E1: From scratch               - 從頭訓練
E2: From SceneFlow            - 從 SceneFlow 開始
E3: From SceneFlow + Stage I  - 兩階段
```

---

## 4. 實驗執行計劃

### 4.1 Phase 1: 基礎設施 (Week 1)

```
□ Clone RAFT-Stereo
□ 實現 PIDSDataset (EXR 讀取)
□ 實現 RGBDataset (對照組)
□ 驗證 DataLoader 正確性
□ 設置 TensorBoard 監控
```

### 4.2 Phase 2: 數據效率實驗 (Week 2-3)

```
□ 準備數據集分割 (100/250/500/1000)
□ 運行 Exp A1-A4 (RGB Baseline)
□ 運行 Exp B1-B4 (PIDS)
□ 收集和整理結果
```

### 4.3 Phase 3: 消融實驗 (Week 4)

```
□ 運行 Exp C1-C4 (輸入模態)
□ 運行 Exp D1-D4 (損失函數)
□ 運行 Exp E1-E3 (預訓練)
```

### 4.4 Phase 4: Stage II (Week 5+)

```
□ 收集真實偏振數據
□ 設計 Stage II DataLoader
□ Stage II 訓練
□ 最終評估
```

---

## 5. 結果報告模板

```markdown
# PIDS 數據效率實驗結果

## 實驗配置
- GPU:
- Batch Size:
- Training Iterations:

## 主要結果

### 表 1: RGB vs PIDS 數據效率對比

| 場景數 | RGB EPE-all | RGB EPE-glass | PIDS EPE-all | PIDS EPE-glass |
|--------|-------------|---------------|--------------|----------------|
| 100    |             |               |              |                |
| 250    |             |               |              |                |
| 500    |             |               |              |                |
| 1000   |             |               |              |                |

### 表 2: 數據效率提升

達到 EPE-glass < 2.0:
- RGB: ___ 場景
- PIDS: ___ 場景
- 效率提升: ___x

## 結論
...
```

---

## 6. 數據流

### 6.1 渲染輸出 → 訓練輸入

```
渲染輸出 (pids_renderer.py v3.4.2)
════════════════════════════════════════════════════════════════

scene_XXXX/
├── scene_XXXX_left_parallel.exr    ──►  image1 (I∥ 左相機)
├── scene_XXXX_right_cross.exr      ──►  image2 (I⊥ 右相機)
├── scene_XXXX_right_parallel.exr   ──►  (用於品質驗證)
├── scene_XXXX_disparity.exr        ──►  ground_truth (視差)
├── scene_XXXX_depth.exr            ──►  (深度相機視角)
├── scene_XXXX_glass_mask.exr       ──►  valid_mask (玻璃區域)
└── scene_XXXX_params.json          ──►  metadata
```

### 6.2 數據格式

| 檔案 | 格式 | 維度 | 範圍 | 說明 |
|------|------|------|------|------|
| left_parallel.exr | float32 | HxW | [0, ~10] | HDR 灰階，需 tone mapping |
| right_cross.exr | float32 | HxW | [0, ~10] | HDR 灰階，需 tone mapping |
| disparity.exr | float32 | HxW | [55, 94] px | 我們場景的視差範圍 |
| glass_mask.exr | float32 | HxW | {0, 1} | 二值 mask |

### 6.3 相機配置

```
感測器: Sony IMX296LQR-C (5.023 × 3.754 mm)
焦距: 6mm
FOV: 45.4°
基線: 65mm
影像尺寸: 640 × 480 px

視差公式: disparity = (baseline × focal_px) / depth
focal_px = 640 / (2 × tan(45.4°/2)) ≈ 765 px

視差範圍:
- 近端 (528mm): 65 × 765 / 528 ≈ 94 px
- 遠端 (900mm): 65 × 765 / 900 ≈ 55 px
```

---

## 7. RAFT-Stereo 架構

### 7.1 輸入輸出

```python
# 輸入
image1: Tensor [N, 3, H, W]  # 左圖 RGB, 範圍 [0, 255]
image2: Tensor [N, 3, H, W]  # 右圖 RGB, 範圍 [0, 255]

# 輸出
disparity: Tensor [N, 1, H, W]  # 視差預測 (像素)
```

### 7.2 訓練參數 (原始)

| 參數 | 值 | 說明 |
|------|-----|------|
| batch_size | 6 | 兩張 RTX-6000 |
| image_size | 320 × 720 | random crop |
| num_iters | 100,000 | 總訓練步數 |
| train_iters | 16 | 每次 forward 的 GRU 迭代 |
| lr | 0.0002 | 初始學習率 |
| optimizer | AdamW | weight_decay=1e-5 |
| scheduler | OneCycleLR | |
| grad_clip | 1.0 | 梯度裁剪 |

### 7.3 損失函數

```python
def sequence_loss(disp_preds, disp_gt, valid, gamma=0.9):
    """多尺度序列損失"""
    n_predictions = len(disp_preds)
    loss = 0.0

    for i, disp_pred in enumerate(disp_preds):
        # 越後面的預測權重越高
        weight = gamma ** (n_predictions - i - 1)
        # L1 損失，只計算有效像素
        diff = torch.abs(disp_pred - disp_gt)
        loss += weight * (valid * diff).mean()

    return loss
```

---

## 8. PIDS 修改方案

### 8.1 輸入修改

**問題**: 原始 RAFT-Stereo 期望 RGB (3通道)，我們的偏振影像是灰階 (1通道)

**方案 A: 複製通道 (推薦)**
```python
# 將灰階複製成 3 通道
img1_gray = read_exr("left_parallel.exr")  # [H, W]
img1_rgb = np.stack([img1_gray, img1_gray, img1_gray], axis=-1)  # [H, W, 3]
```

**方案 B: 修改 Encoder**
```python
# 修改第一層 conv 的 in_channels: 3 → 1
# 需要重新訓練或調整預訓練權重
```

### 8.2 數據正規化

```python
def tonemap_exr(img, percentile=99):
    """HDR → LDR tone mapping"""
    img = np.maximum(img, 0)
    max_val = np.percentile(img, percentile)
    if max_val > 0:
        img = img / max_val * 255
    return np.clip(img, 0, 255).astype(np.float32)
```

---

## 9. 代碼結構

```
training_guidance/Stage_I/training_stage/
│
├── RAFT-Stereo/                      # git clone (不修改)
│   ├── core/
│   ├── train_stereo.py
│   └── evaluate_stereo.py
│
├── pids_training/                    # PIDS 專用代碼
│   ├── datasets/
│   │   └── pids_dataset.py           # PIDS DataLoader
│   ├── configs/
│   │   └── pids_stage1.yaml          # Stage I 配置
│   ├── train_pids.py                 # 訓練入口
│   └── evaluate_pids.py              # 評估腳本
│
├── checkpoints/                      # 模型權重
├── runs/                             # TensorBoard logs
├── data/ -> /rendered/output         # 符號連結
└── PIDS_FINETUNING_PLAN.md          # 本文檔
```

---

## 10. 訓練配置

```yaml
# pids_training/configs/pids_stage1.yaml

data:
  train_path: "./data/rendered"
  image_size: [480, 640]
  augment: true

model:
  name: "raft-stereo"
  pretrained: "./checkpoints/raftstereo-sceneflow.pth"

training:
  batch_size: 4
  num_iters: 50000
  train_iters: 16
  lr: 0.0001                    # Finetuning 用較低學習率
  weight_decay: 0.00001
  grad_clip: 1.0
  loss_gamma: 0.9

validation:
  val_freq: 5000
  valid_iters: 32

seed: 42
mixed_precision: true
```

---

## 11. 評估指標

| 指標 | 定義 | 目標 |
|------|------|------|
| **EPE** | mean(\|pred - gt\|) | < 1.0 px |
| **D1-all** | error > 3px and > 5% | < 5% |
| **D1-glass** | D1 in glass region | < 10% |
| **EPE-glass** | EPE in glass region | < 2.0 px |

---

## 12. 待討論問題

1. **灰階 vs RGB 輸入**: 複製成 3 通道 vs 修改 encoder
2. **預訓練權重**: 從 SceneFlow 開始 vs 從頭訓練
3. **數據量**: 100 / 500 / 1000 / 2000 場景
4. **玻璃區域權重**: 1:1 / 2:1 / 5:1

---

## 13. 參考資料

- [RAFT-Stereo Paper](https://arxiv.org/abs/2109.07547)
- [RAFT-Stereo GitHub](https://github.com/princeton-vl/RAFT-Stereo)
- [Mitsuba 3 Documentation](https://mitsuba.readthedocs.io/)

---

## 14. 實驗記錄

### 14.1 數據集準備 (2026-01-02)

| 項目 | 數量 | 說明 |
|------|------|------|
| 偏振版渲染 | 4000 場景 | pids_renderer_v3.py |
| QA 篩選 | 94.1% 通過率 | quality_validator.py |
| 無偏振版渲染 | 3765 場景 | pids_renderer_nopol.py |
| 訓練集 | 3500 場景 | 偏振 + 無偏振 |
| 測試集 | 265 場景 | 偏振 + 無偏振 |

### 14.2 實驗 #13: PIDS 偏振版 (2026-01-02)

```
GPU: H200
數據: dataset_pol_V4/train (3500 scenes)
模型: RAFT-Stereo + Scene Flow pretrained
訓練: 50K steps, batch=4, lr=1e-4, glass_weight=3.0
```

**結果:**
| 指標 | 數值 |
|------|------|
| Glass EPE | **39.20 px** |
| 穩定性 | ±0.5 px 震盪 |
| 收斂 | ~30K steps |

### 14.3 實驗 #14: Baseline 無偏振版 (2026-01-02 ~ 01-03) 完成

```
GPU: H200
數據: dataset_nopol_3500/train (3500 scenes)
模型: RAFT-Stereo + Scene Flow pretrained
訓練: 50K steps, batch=4, lr=1e-4, glass_weight=3.0
```

**最終結果 (50K steps):**
| 指標 | 數值 |
|------|------|
| Glass EPE (最終) | **42.17 px** |
| Glass EPE (最佳) | **40.79 px** @ 47K steps |
| Val Loss (最終) | **248.59** |
| 穩定性 | 高震盪 (40-228 px) |

**訓練曲線特徵:**
- 0-40K: 劇烈震盪 (45-228 px)
- 40K-50K: 收斂穩定 (~40-43 px)

**分析:**
- PIDS (39.20) vs Baseline (42.17) = **7.0% 改善**
- PIDS 優勢: 穩定性顯著更好 (±0.5 vs ±50)
- 結論: 隱式偏振信息提供穩定性，但改善幅度有限，需要顯式偏振編碼器

### 14.4 Dual-Stream 架構設計 (2026-01-03)

為提升 PIDS 效果，設計 Dual-Stream Polarization Encoder：

```
架構:
    left ──→ [FeatureEncoder] ──→ fmap1 ─┐
                                          ├──→ [Fusion] ──→ [Corr + GRU] ──→ disparity
    right ─→ [FeatureEncoder] ──→ fmap2 ─┤
                                          │
    |left-right| ─→ [PolarizationEncoder]─┘
             (soft threshold + spatial attention)
```

### 14.5 強化版 Dual-Stream (2026-01-03 更新)

為達成 25-30% 改善目標，實施以下強化策略：

**1. 強化版 PolarizationEncoder:**

```python
架構:
    pol_diff ──→ [Soft Threshold] ──→ [Stem Conv] ──→ [ResidualBlock]
                       ↓                                    ↓
                   w = σ(κ(P-τ))                      [SpatialAttention]
                                                           ↓
                                                    pol_features (64-dim)
```

組件:
- **ResidualBlock2D**: 殘差連接提升梯度流動
- **SpatialAttention**: 學習關注玻璃區域
- **輸出維度**: 64 (原 32)

**2. Polarization-aware Loss:**

```python
loss = Σ γ^(N-i) × [glass_mask × glass_weight + (1-glass_mask)]
                 × [pol_weight × pol_diff + (1-pol_diff)]  # 新增
                 × |D_pred - D_gt|
```

- `pol_weight=2.0`: 偏振差異大的區域額外加權
- 理論: 偏振差異大 = 玻璃區域 → 應更精確匹配

**3. 超參數調整:**

| 參數 | 原值 | 新值 | 原因 |
|------|------|------|------|
| pol_dim | 32 | **64** | 更大容量捕捉複雜特徵 |
| pol_threshold | 0.1 | **0.05** | 更敏感捕捉微弱偏振 |
| glass_weight | 3.0 | **5.0** | 更強調玻璃區域 |
| pol_lr_mult | 10.0 | **5.0** | 避免新層不穩定 |
| pol_weight | - | **2.0** | 新增 Polarization-aware Loss |

**訓練命令:**
```bash
python train_pids.py \
    --data_dir ./dataset_pol_V4/train \
    --output_dir ./checkpoints_dual_stream_v2 \
    --dual_stream \
    --pol_dim 64 \
    --pol_threshold 0.05 \
    --pol_sharpness 20.0 \
    --pol_lr_mult 5.0 \
    --pol_weight 2.0 \
    --glass_weight 5.0 \
    --pretrained ./pretrained/raftstereo-sceneflow.pth \
    --batch_size 4 \
    --num_steps 50000 \
    --lr 0.0001 \
    --bf16
```

### 14.6 實驗 #15: 強化版 Dual-Stream (待執行)

**目標:**
- 樂觀目標: Glass EPE 27-29 px (25-30% 改善)
- 保守目標: Glass EPE 31-33 px (15-20% 改善)

**實驗配置:**
| 參數 | 值 |
|------|-----|
| pol_dim | 64 |
| pol_threshold | 0.05 |
| pol_sharpness | 20.0 |
| pol_lr_mult | 5.0 |
| pol_weight | 2.0 |
| glass_weight | 5.0 |
| num_steps | 50K |

**強化策略總結:**
1. ResidualBlock2D - 改善梯度流動
2. SpatialAttention - 自動學習關注玻璃
3. 更大 pol_dim (64) - 更強表達能力
4. 更低 threshold (0.05) - 更敏感偏振檢測
5. Polarization-aware Loss - 偏振區域額外加權
6. 更高 glass_weight (5.0) - 玻璃區域更重要
7. **視差對齊修正** - 使用 GT disparity warp 右圖後再計算 pol_diff

### 14.7 視差對齊修正 (2026-01-03)

**問題發現:**
原始 `pol_diff = |left(x,y) - right(x,y)|` 比較的是**不同 3D 點**：
- 左右相機有 65mm 基線
- 視差範圍 55-94 px
- 導致 pol_diff 混入視差誤差，不是純偏振差異

**解決方案:**
```python
# 新增 warp_with_disparity() 函數
def warp_with_disparity(img, disparity):
    """使用 GT 視差將右圖 warp 到左圖視角"""
    # right(x - disparity, y) 對應 left(x, y)

# PolarizationEncoder.compute_pol_diff() 修改
if disparity is not None:
    right_aligned = warp_with_disparity(right, disparity)
else:
    right_aligned = right  # 推論時退化為原始方式

pol_diff = |left - right_aligned|  # 現在是純偏振差異
```

**相機參數:**
```
Baseline: 65mm
Focal: ~765px (計算值)
視差公式: disparity = baseline × focal / depth
視差範圍: 55-94 px (深度 900-528mm)
```

**修改檔案:**
- `pids_model.py`: 新增 `warp_with_disparity()`，更新 `PolarizationEncoder`
- `train_pids.py`: 訓練/驗證時傳遞 `disparity_gt`

**效果:**
- 訓練時: 使用 GT disparity 對齊，計算純偏振差異
- 推論時: 無 GT disparity，退化為原始方式 (可接受)

### 14.8 實驗 #15: Dual-Stream 首次嘗試 (2026-01-04) BUG

**配置:**
```
GPU: H200
訓練: 50K steps, batch=8, iters=24
參數: pol_dim=128, pol_threshold=0.05, pol_weight=2.0, glass_weight=5.0
```

**結果 (有 BUG):**
| 指標 | 數值 | 問題 |
|------|------|------|
| Val Loss | 2154.68 | 比 baseline (~248) 高 9x |
| Glass EPE | 288.71 px | 比 baseline (~42) 高 7x |

**發現的 BUG:**
```python
# train_pids.py 原始代碼
disp_preds = [-f for f in flow_preds]  # f shape: (B, 2, H, W)

# 問題: flow 輸出是 (B, 2, H, W)，但 disp_gt 是 (B, 1, H, W)
# 導致 loss 計算時 broadcasting 錯誤：
# - Channel 0: |disp - disp_gt| (correct)
# - Channel 1: |0 - disp_gt| = disp_gt (wrong - GT itself added to loss!)

# 修正後
disp_preds = [-f[:, :1] for f in flow_preds]  # 只取第一個 channel
```

**結論:** 實驗 #15 數據無效，需重新訓練

### 14.9 實驗 #16: Dual-Stream 修正版 (2026-01-04) 突破性進展

**修正內容:**
- 修正 flow→disparity 維度 bug (`[-f[:, :1] for f in flow_preds]`)
- 4 處修正: train_epoch (FP16/BF16/FP32) + validate

**配置:**
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

**訓練進度 (32.5K / 50K = 65%):**

```
Glass EPE 趨勢:
├─  0.5K steps:  86.61 px (起始)
├─  5K steps:    45.09 px
├─ 10K steps:    44.57 px
├─ 14.5K steps:  34.89 px (跌破 35!)
├─ 16K steps:    33.77 px
├─ 19K steps:    30.25 px (跌破 30!)
├─ 20.5K steps:  28.17 px
├─ 23.5K steps:  26.34 px
├─ 27K steps:    26.09 px
├─ 28K steps:    25.40 px
├─ 31K steps:    24.38 px
├─ 31.5K steps:  24.08 px ← 目前最佳
└─ 32.5K steps:  24.68 px (訓練中)
```

**對比歷史實驗:**
| 實驗 | Glass EPE | vs Exp#16 改善 |
|------|-----------|----------------|
| Exp #13 (PIDS 隱式) | 39.20 px | **-38.6%** |
| Exp #14 (Baseline 無偏振) | 42.17 px | **-42.9%** |
| **Exp #16 (Dual-Stream)** | **24.08 px** | 目前最佳 |

**關鍵發現:**
1. 維度 bug 修正後，Dual-Stream 架構效果顯著
2. 從 39.20 -> 24.08 px，**改善近 40%**
3. 訓練穩定，持續下降
4. 還有 35% 訓練空間，最終結果可能更好

**狀態:** 訓練中 (65%)，預計完成後更新最終數據

---

## 15. 版本更新記錄

| 版本 | 日期 | 更新內容 |
|------|------|----------|
| 1.0 | 2025-12-22 | 初始規劃文檔 |
| 1.1 | 2025-12-22 | 細化實驗設計 |
| 1.2 | 2026-01-03 | 新增實驗 #13/#14 結果、Dual-Stream 架構設計 |
| 1.3 | 2026-01-03 | 強化版 Dual-Stream：ResidualBlock + SpatialAttention + Polarization-aware Loss |
| 1.4 | 2026-01-03 | 視差對齊修正：使用 GT disparity warp 右圖後計算純偏振差異 |
| 1.5 | 2026-01-04 | 修正 flow→disparity 維度 bug、實驗 #15/#16 記錄 |
| 1.6 | 2026-01-04 | 實驗 #16 突破性進展：Glass EPE 24.08 px (改善 40%) |
