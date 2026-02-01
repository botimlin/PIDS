## 實驗 #14：Baseline 無偏振版訓練

**日期**: 2026-01-03
**狀態**: 完成

### 實驗目標

使用相同的 3500 場景訓練無偏振版本，作為 PIDS 消融實驗的 Baseline。

### 數據準備

使用 `--scene_list` 參數確保與實驗 #13 使用完全相同的場景：

```bash
python organize_dataset.py \
    --input_dir ./output_nopol \
    --output_dir ./dataset_nopol \
    --scene_list ./dataset_pol_V4 \
    --copy
```

### 訓練配置

對齊實驗 #13 參數，唯一差異是輸入數據：

```bash
nohup python train_pids.py \
    --data_dir ./dataset_nopol \
    --output_dir ./checkpoints_nopol_3500 \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --glass_weight 3.0 \
    --lr 0.00005 \
    --batch_size 8 \
    --accumulation_steps 1 \
    --num_steps 50000 \
    --val_freq 500 \
    --iters 16 \
    --scheduler cosine \
    --d1_weight 0.2 \
    > train_nopol_3500.log 2>&1 &
```

### 對齊項目

| 參數 | PIDS (實驗#13) | Baseline (實驗#14) |
|------|----------------|-------------------|
| 場景列表 | train_scenes.txt | **相同** |
| train_size | 3500 | **相同** |
| test_size | 265 | **相同** |
| num_steps | 50000 | **相同** |
| batch_size | 8 | **相同** |
| lr | 0.00005 | **相同** |
| 輸入格式 | `_left_parallel.exr` | `_left.exr` |

### 預期結果

如果偏振確實有助於透明物體檢測，預期：
- Baseline Glass EPE > PIDS Glass EPE
- Baseline D1 > PIDS D1

### 最終結果 (50K steps)

| 指標 | 數值 |
|------|------|
| Glass EPE (最終) | **42.17 px** |
| Glass EPE (最佳) | **40.79 px** @ 47K steps |
| Val Loss (最終) | **248.59** |
| 穩定性 | 高震盪 (40-228 px) |

**訓練曲線特徵:**
- 0-40K: 劇烈震盪 (45-228 px)
- 40K-50K: 收斂穩定 (~40-43 px)

### 訓練觀察

#### 1. 訓練穩定性差異

| 指標 | PIDS (實驗#13) | Baseline (實驗#14) |
|------|----------------|-------------------|
| Glass EPE 全程最高 | < 143 px | 220+ px |
| 學習穩定性 | 穩定收斂 | 劇烈跳動 |
| Val Loss 峰值 | 正常範圍 | 2500+ |

#### 2. 關鍵發現：背景學習 vs 玻璃學習

Baseline 出現有趣的分離現象：
- 全域 EPE：~50 px（相對正常）
- Glass EPE：200+ px（完全失敗）

**解讀**：網路能夠學習背景（漫反射表面）的 stereo matching，但對玻璃區域幾乎無法學習。這是因為：
- 無偏振時，玻璃區域左右影像幾乎相同
- 網路沒有可利用的 signal 來區分玻璃
- 全域 EPE 低只是因為背景像素佔多數

> "Without polarization, the network learns background disparity normally but treats glass regions as noise, resulting in near-random predictions on transparent surfaces."

#### 3. 結論

- PIDS (39.20) vs Baseline (42.17) = **7.0% 改善**
- PIDS 優勢: 穩定性顯著更好 (±0.5 vs ±50)
- 結論: 隱式偏振信息提供穩定性，但改善幅度有限，需要顯式偏振編碼器

---

### 渲染速度對比

| | Polarized | Non-Polarized | 差異 |
|---|---|---|---|
| Mitsuba variant | `spectral_polarized` | `rgb` | - |
| Integrator | Stokes | Path | - |
| 每場景時間 | ~58 秒 | ~5 秒 | **6.5x 更快** |
| 總時間 (16 GPU) | ~4 小時 | 35 分鐘 | - |

**結論**：偏振模擬的計算成本是普通渲染的 **6.5 倍**。這也解釋了為何現有研究較少使用大規模偏振渲染訓練數據。

### QA 工具更新

新增 `--failed` 參數自動輸出未通過場景列表：

```bash
python quality_validator.py --input_dir ./output_pol --output report.md --skip-c1
# 自動生成 failed.txt（235 個未通過場景）
```

### nopol 渲染器邏輯更新

改為「以 params.json 為主導」：
- 掃描 `--params_dir` 中的 `*_params.json`
- 只渲染有對應 params.json 的場景
- 不需要刪除 OBJ 檔案，只需刪除不通過的 params.json

### 訓練工具更新

#### organize_dataset.py 新增 Train/Test 分割

```bash
python organize_dataset.py \
    --input_dir ./output_pol \
    --output_dir ./dataset_pol \
    --report ./Quality_Assurance/quality_report.md \
    --train_size 3500 \
    --copy
```

輸出結構：
```
dataset_pol/
├── train/
│   ├── stereo_pairs/
│   ├── ground_truth/
│   └── masks/
├── test/
│   ├── stereo_pairs/
│   ├── ground_truth/
│   └── masks/
├── train_scenes.txt
└── test_scenes.txt
```

- 從 3765 場景中隨機選取 3500 作為訓練集
- 剩餘 265 場景作為測試集
- 使用固定 seed=42 確保可重現

#### pids_dataset.py 支持新目錄結構

自動偵測 `dataset/train/stereo_pairs` 結構，只讀取訓練集：

```python
# 自動檢測並只讀取 train/ 子目錄
dataset = PIDSSyntheticDataset(data_dir="./dataset_pol", split='train')
# [PIDSDataset] Using train/ subdirectory (ignoring test/)
```

#### check_nopol_completeness.py 新增

簡單的 nopol 輸出完整性檢查（不做偏振 QA）：

```bash
python check_nopol_completeness.py --input_dir ./output_nopol
```

檢查 5 個必要文件是否齊全：`_left.exr`, `_right.exr`, `_disparity.exr`, `_depth.exr`, `_glass_mask.exr`

#### organize_dataset.py 新增 --scene_list 參數

為確保 PIDS vs Baseline 公平對比，新增場景列表對齊功能：

```bash
# 使用 pol 數據集的場景列表來整理 nopol 數據集
python organize_dataset.py \
    --input_dir ./output_nopol \
    --output_dir ./dataset_nopol \
    --scene_list ./dataset_pol_V4 \
    --copy
```

**功能**：
- 讀取 `--scene_list` 目錄中的 `train_scenes.txt` 和 `test_scenes.txt`
- 確保 nopol 使用完全相同的 3500 train + 265 test 場景
- 避免因 QA 篩選差異導致不公平對比

**支援的檔案命名**：
- pol: `_left_parallel.exr`, `_right_cross.exr`
- nopol: `_left.exr`, `_right.exr`

---

# 參考資料

### 論文訓練策略 (PIDS.pdf)

- Stage I: AdamW, ε=10⁻⁶, OneCycle, 大 batch
- Loss: L = Σ γ^(N-i) ||Di - Dgt||₁
- 預訓練權重: Scene Flow

### 調優建議

- batch: 16（不行就 8）
- lr: 1e-4（還抖就 5e-5 或 2e-5）
- optimizer: AdamW（eps=1e-6 保留）
- grad clip: 0.1（高反光數據必要）
- valid 頻率: 拉密一點

---

# 第四部分：Dual-Stream 架構開發

## Dual-Stream Polarization Encoder 設計

**日期**: 2026-01-03
**動機**: 實驗 #13/#14 顯示隱式偏振 (直接輸入 I_parallel, I_cross) 改善有限 (7%)，需要顯式偏振編碼器

### 架構設計

```
架構:
    left ──> [FeatureEncoder] ──> fmap1 ─┐
                                          ├──> [Fusion] ──> [Corr + GRU] ──> disparity
    right ─> [FeatureEncoder] ──> fmap2 ─┤
                                          │
    |left-right| ─> [PolarizationEncoder]─┘
             (soft threshold + spatial attention)
```

### PolarizationEncoder 組件

```python
架構:
    pol_diff ──> [Soft Threshold] ──> [Stem Conv] ──> [ResidualBlock]
                       |                                    |
                   w = sigmoid(kappa(P-tau))          [SpatialAttention]
                                                           |
                                                    pol_features (64-dim)
```

組件:
- **ResidualBlock2D**: 殘差連接提升梯度流動
- **SpatialAttention**: 學習關注玻璃區域
- **輸出維度**: 64 (原 32)

### Polarization-aware Loss

```python
loss = Sum gamma^(N-i) * [glass_mask * glass_weight + (1-glass_mask)]
                       * [pol_weight * pol_diff + (1-pol_diff)]
                       * |D_pred - D_gt|
```

- `pol_weight=2.0`: 偏振差異大的區域額外加權
- 理論: 偏振差異大 = 玻璃區域 -> 應更精確匹配

### 超參數配置

| 參數 | 原值 | 新值 | 原因 |
|------|------|------|------|
| pol_dim | 32 | **64** | 更大容量捕捉複雜特徵 |
| pol_threshold | 0.1 | **0.05** | 更敏感捕捉微弱偏振 |
| glass_weight | 3.0 | **5.0** | 更強調玻璃區域 |
| pol_lr_mult | 10.0 | **5.0** | 避免新層不穩定 |
| pol_weight | - | **2.0** | 新增 Polarization-aware Loss |

---

## 視差對齊修正

**日期**: 2026-01-03

### 問題發現

原始 `pol_diff = |left(x,y) - right(x,y)|` 比較的是**不同 3D 點**：
- 左右相機有 65mm 基線
- 視差範圍 55-94 px
- 導致 pol_diff 混入視差誤差，不是純偏振差異

### 解決方案

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

**效果:**
- 訓練時: 使用 GT disparity 對齊，計算純偏振差異
- 推論時: 無 GT disparity，退化為原始方式 (可接受)

---

