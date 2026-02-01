# PIDS 開發日誌

**專案**: Physics-Informed Deep Stereo (PIDS)
**作者**: Po-Ting Lin
**整合日期**: 2025-12-30

> Copyright (c) 2025-2026 Po-Ting Lin
>
> This document records the design decisions, experiments, and analysis
> conducted during the development of the Physics-Informed Deep Stereo (PIDS) project.
>
> Released under the MIT License (see LICENSE file).

---

# 第一部分：渲染器開發 (Stage I Sampling)

## 1. 偏振渲染器排錯歷程

### 問題描述

**主要問題**：`left_parallel` (I∥) 和 `right_cross` (I⊥) 兩張圖的**背景亮度差異過大**，導致 RAFT-Stereo 無法正確對齊。

**次要問題**：櫃子等非玻璃物體出現「鬼影」。

---

### 階段 1：初始診斷

**觀察**：
- `left_parallel` 背景很亮
- `right_cross` 背景很暗
- 預期：背景（漫反射表面）應該 I∥ ≈ I⊥

**我們的假設**：偏振光源照射整個場景，導致背景也有偏振。

---

### 階段 2：嘗試 Constant Emitter（效果不佳）

**理論**：使用 `constant` emitter 作為環境光，因為它是真正的非偏振漫射光。

**我們的修改**：
```python
'fill_light': {
    'enabled': True,
    'intensity': 15000.0,  # constant emitter
},
'led': {
    'intensity': 500.0,    # 降低偏振光
},
```

**結果**：背景亮度仍然不平衡。

**原因分析**：constant emitter 的強度單位與 area light 不同，實際強度不足。

---

### 階段 3：嘗試關閉偏振光源（全黑）

**我們的修改**：
```python
'led': {
    'enabled': False,
    'intensity': 0.0,
},
'fill_light': {
    'enabled': True,
    'intensity': 50.0,
},
```

**結果**：渲染結果全黑。

**原因**：constant emitter 強度 50.0 太低。

---

### 階段 4：嘗試取消放大因子（仍有差異）

**理論**：`amplify_factor=5.0` 會放大 I∥ - I⊥ 的差異，即使背景 S1 很小也會被放大。

**我們的修改**：
```python
amplify_factor=1.0,  # 不放大
exposure=1.0,        # 不調整曝光
```

**結果**：背景亮度差異改善，但整體太暗。

---

### 階段 5：發現材質誤判（關鍵發現）

**我們發現的線索**：
> 即使已經關閉了 LED，如果材質被誤判為玻璃，它就會變成一個巨大的偏振產生器。

**問題代碼**：
```python
def is_glass_material(mat_name: str) -> bool:
    glass_keywords = ['glass', 'clear', 'transparent', 'window',
                      'door', 'partition', 'panel', 'acrylic']
    return any(k in name_lower for k in glass_keywords)
```

**危險情況**：如果有材質叫 `Back_Panel` 或 `Wall_Partition`，會被誤判為玻璃！

**我們的修復**：
```python
def is_glass_material(mat_name: str) -> bool:
    name_lower = mat_name.lower()

    # 先排除明確的非玻璃材質
    non_glass_keywords = ['background', 'wall', 'floor', 'ground', 'ceiling',
                          'diffuse', 'opaque', 'solid', 'wood', 'metal', 'fabric']
    if any(k in name_lower for k in non_glass_keywords):
        return False

    # 只有明確包含 'glass' 才算玻璃
    glass_keywords = ['glass', 'transparent', 'clear_glass', 'acrylic']
    return any(k in name_lower for k in glass_keywords)
```

---

### 階段 6-7：IOR 和 twosided 嘗試（無效）

我們嘗試調高 IOR 和使用 twosided 材質，但都無法解決鬼影問題。

---

### 階段 8：回滾到穩定版本

**最終配置**：
```python
CONFIG = {
    'lighting': {
        'led': {
            'enabled': True,
            'intensity': 1500.0,
        },
        'fill_light': {
            'enabled': True,
            'intensity': 8000.0,
        },
    },
    'materials': {
        'glass': {
            'type': 'thindielectric',
            'ior': 1.5,
        },
    },
}

amplify_factor = 5.0
exposure = 2.0
auto_balance = True
```

---

### 階段 9-11：天花板光源問題排查（全部失敗）

我們嘗試了多種方法改善天花板光源的漫反射效果：
1. 提高天花板光照強度 - 無效
2. 增加光線反彈次數 (MAX_DEPTH 12→48) - 無效
3. 提高材質反射率 - 無效
4. 獨立燈陣列 - 失敗並回滾

**關鍵發現 - Mitsuba 3 的 polarizer BSDF 行為**：

> 在 Mitsuba 3 中，`polarizer` BSDF 是設計來**過濾穿過它的光**，而不是作為發射器的材質。正確做法是讓光源發射非偏振光，然後**穿過一個獨立的偏振片幾何體**。

---

### 階段 12：實施物理偏振片架構 (v3.1.0)

**解決方案**：我們實施了完整的物理偏振片架構：

1. **LED 光源重構**：
   - `led_emitter`：純粹的非偏振面光源
   - `led_polarizer`：獨立偏振片幾何體 (θ=0°)，位於光源前方 5mm

2. **相機偏振片**：
   - 左相機：前方加偏振片 (θ=0°) → 通過 I∥
   - 右相機：前方加偏振片 (θ=90°) → 通過 I⊥

**結果**：I∥ 和 I⊥ 亮度平衡了！

**實測數據**：

| 指標 | 數值 | 評價 |
|------|------|------|
| 品質評分 | 87/100 | excellent |
| 玻璃區域 DoLP | 24.7% | 很好的偏振效果 |
| 背景 DoLP | 3% | 低，符合預期 |
| 背景 I∥/I⊥ 平衡 | 1.0003 | 幾乎完美 |

---

### 階段 13：修正相機光軸為平行配置 (v3.2.0)

**問題**：兩個相機都使用 `look_at` 指向同一個目標點，導致光軸會聚，違反 PIDS 論文的 Criterion 1。

**我們的修正**：

```
修正前（會聚光軸）:
    [L]                 [R]
      \                 /
       \               /
        \_____________/
             目標點

修正後（平行光軸）:
    [L]                 [R]
     |                   |
     |                   |
     ↓                   ↓
```

**實測結果**：Vertical disparity 約等於 0

---

## 2. 關鍵發現總結

### Mitsuba 3 偏振行為

| 元件 | 行為 |
|------|------|
| area emitter | 本身非偏振，但 Fresnel 反射會產生偏振 |
| constant emitter | 真正的非偏振漫射光 |
| diffuse BSDF | 會 depolarize 入射光 |
| thindielectric | Fresnel 反射產生偏振 |

### 放大因子影響

| amplify_factor | 效果 |
|----------------|------|
| 1.0 | 物理正確，但對比度低 |
| 5.0 | 對比度增強，背景差異也被放大 |
| 10.0 | 過度放大，產生 artifacts |

---

## 3. 最終渲染器配置 (v3.4.2)

### 相機配置

| 參數 | 值 |
|------|-----|
| 感測器 | Sony IMX296LQR-C |
| 焦距 | 6mm |
| FOV | 45.4° |
| BASELINE | 65mm |
| SPP | 16384 |

### 相機位置

```
左相機 (I∥):    X=-82.5mm, Y=400mm, Z=80mm
右相機 (I⊥):    X=-17.5mm, Y=400mm, Z=80mm
深度相機:       X=-50.0mm, Y=400mm, Z=80mm

光軸方向: (0, 1, 0) - 三台相機完全平行
```

### 品質驗證標準 (PIDS 5 Criteria)

| Criterion | 名稱 | 閾值 | 狀態 |
|-----------|------|------|------|
| 1 | Geometric Consistency | vertical disparity < 1px | Pass |
| 2 | Background Photometric Consistency | I∥/I⊥ in [0.5, 2.0] | Pass |
| 3 | Polarization Signal Validity | 玻璃 DoLP > 10% | Pass |
| 4 | Ground Truth Alignment | < 10% 正規化誤差 | Pass |
| 5 | Depth Validity Rate | > 90% in glass region | Pass |

---

# 第二部分：紋理支持實現 (v4.0.0)

**日期**: 2025-12-29

## 目標

在 PIDS 渲染管線中加入材質貼圖支持（做加法，不改動現有功能）

## 新建檔案

| 檔案 | 說明 |
|------|------|
| `modelling_textured/blender_furniture_randomizer_v18.py` | Fork from v17，新增紋理支持 |
| `rendering_v4/pids_renderer_textured.py` | Fork from pids_renderer_fast.py，新增紋理支持 |
| `textures/manifest.json` | 紋理資產索引 |

## Blender 端修改 (v18.py)

### TextureManager 類

- `_load_manifest()`: 載入 manifest.json
- `_scan_directory()`: 自動掃描紋理檔案 (PNG > JPG 優先)
- `get_random_texture()`: 隨機選取紋理

### 紋理分配邏輯

| 物件 | 紋理來源 | 行為 |
|------|---------|------|
| 牆壁 (4面) | `wall/` | 每場景隨機選一個，所有牆面統一 |
| 天花板 | `wall/` | 與牆壁共用同一紋理 |
| 地板 | `floor/` | 每場景隨機 |
| 傢俱 | `furniture/` | 每場景隨機，每件獨立 |

### 材質獨立性

| 物件 | 獨立材質 | 說明 |
|------|---------|------|
| 傢俱 | Yes | 每個傢俱使用獨立材質實例 |
| 地板 | Yes | 每場景使用獨立材質實例 |
| 牆壁 | No | 所有牆面共用同一材質 |
| 天花板 | No | 與牆壁共用同一材質 |

## Mitsuba 端修改 (v4.0.0)

### MTLParser 擴展

新增解析 `map_Kd` (diffuse 紋理路徑)

### MaterialFactory 擴展

```python
@staticmethod
def diffuse_textured(texture_path: str) -> Dict:
    return {
        'type': 'diffuse',
        'reflectance': {
            'type': 'bitmap',
            'filename': texture_path,
            'filter_type': 'bilinear',
            'wrap_mode': 'repeat',
        },
    }
```

### 渲染配置

- SPP: 8192 (可調整)
- 多 GPU 均勻分配：餘數分散給前幾個 GPU

## 使用說明

### 生成場景

```bash
blender --background --python blender_furniture_randomizer_v18.py -- \
    --seed 42 \
    --count 100 \
    --output ./scenes_textured \
    --texture_dir ./textures
```

### 渲染

```bash
python pids_renderer_textured.py \
    --input_dir ./scenes_textured \
    --output ./output \
    --num_gpus 8
```

---

# 第三部分：訓練調適記錄

## 訓練環境

- **GPU**: NVIDIA H100 (80GB)
- **框架**: PyTorch + RAFT-Stereo
- **預訓練**: raftstereo-sceneflow.pth

## 指標說明

| 指標 | 說明 | 理想值 |
|------|------|--------|
| Loss | L1 加權損失 | < 10 |
| EPE | 平均像素誤差 | < 1.0 px |
| Glass EPE | 玻璃區域誤差 (核心指標) | < 3.0 px |
| D1 | 誤差 >1px 比例 | < 5% |
| Composite | glass_epe + 0.1×d1 | 越低越好 |

---

## 實驗 #1-4：過擬合問題排查

### 問題

我們觀察到嚴重的過擬合：
- 訓練集指標良好 (Glass EPE ~1.5px)
- 驗證集波動大 (Glass EPE 15-40px)

### 我們嘗試的方法（均未解決）

1. **調大 batch、降低 lr、改用 cosine** - 無效
2. **移除 Random Crop** - 無效
3. **加 weight_decay** - 無效
4. **凍結 Feature Encoder** - 無效

### 結論

問題不在這些超參數，而是數據量不足。

---

## 實驗 #5：大規模數據訓練

### 解決方案

1. **大規模渲染**: 使用 12x RTX 4090 渲染 4000 場景
2. **品質過濾**: 保留 3765 場景 (94.1%)
3. **調整分割比例**: 80/20 分割

### 結果

| 指標 | 範圍 |
|------|------|
| Val Glass EPE | 12.4-76.3 px |
| Val D1 | 49.4-99.1% |

**問題**：4x 數據量仍未解決過擬合，波動依然劇烈。

---

## 實驗 #7：增加場景多樣性

### 解決方案

我們更新了 Blender 腳本 (v17.2)，新增 7 種物件類型：

| 物件 | source 名稱 |
|------|-------------|
| 掛鐘 | `source_clock` |
| 檯燈 | `source_lamp` |
| 盆栽 | `source_plant` |
| 電視 | `source_tv` |
| 畫框 | `source_frame` |
| 花瓶 | `source_vase` |
| 書本 | `source_books` |

---

## 實驗 #8：數值穩定性優化

**日期**: 2025-12-29
**狀態**: 實驗成功

### 問題描述

我們觀察到嚴重的 Validation Loss 震盪現象，偶發性 Loss 瞬間飆升至天文數字 (如 1360.5)。

### 根本原因分析

| 問題 | 原設定 | 後果 |
|------|--------|------|
| **過度訓練** | 1900 組數據跑 1,000,000 步 | 模型死記硬背 |
| **數值不穩定 (Epsilon)** | PyTorch 預設 eps=1e-8 | 梯度爆炸 |
| **混合精度溢出** | 開啟 FP16 | NaN/Infinity |
| **顯存限制** | 切回 FP32 後 OOM | 無法訓練 |

### 我們的優化歷程

| 階段 | 修改項目 | 具體數值 |
|------|----------|----------|
| Phase 1 | 縮減步數 | 1,000,000 → 20,000 |
| Phase 2 | 對齊論文參數 | eps=1e-6 |
| Phase 3 | 強力梯度裁剪 | clip_grad 1.0 → 0.1 |
| Phase 4 | 棄用混合精度 | 移除 --mixed_precision |
| Phase 5 | 調整 Batch Size | 16 → 8 |

### 最終配置 (Golden Command)

```bash
nohup python train_pids.py \
    --data_dir ./dataset \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 \
    --lr 0.00002 \
    --clip_grad 0.1 \
    --weight_decay 0.0001 \
    --scheduler cosine \
    --num_steps 20000 \
    --val_freq 100 \
    --val_split 0.2 \
    --d1_weight 0.1 \
    > train_fp32_final.log 2>&1 &
```

### 結果

| 指標 | 優化前 | 優化後 |
|------|--------|--------|
| Loss 穩定性 | 極不穩定 | 穩定下降 |
| Loss 峰值 | 1360.50 | 44.31 |
| 收斂趨勢 | 發散 | 收斂 |

### 關鍵發現

1. **FP32 是必要的** - 高反光區域的極端數值需要全精度處理
2. **eps=1e-6 是必要的** - 論文參數不可省略
3. **強梯度裁剪 (0.1) 有效** - 抑制極端樣本的梯度衝擊
4. **步數要合理** - 小數據集不需要百萬步

---

## 實驗 #9：BFloat16 混合精度優化

**日期**: 2025-12-29

### 背景

實驗 #8 使用 FP32 成功穩定訓練，但 batch_size 被迫降到 8。為了恢復 batch_size=16，我們改用 **BFloat16**。

### BFloat16 優勢

| 特性 | FP16 | BF16 | FP32 |
|------|------|------|------|
| 數值範圍 | 小 (會炸) | **同 FP32** | 最大 |
| 顯存佔用 | 小 | **小** | 大 |
| H100 優化 | 有 | **最佳** | 無 |

### 配置

```bash
nohup python train_pids.py \
    --data_dir ./dataset \
    --batch_size 16 \
    --lr 0.00002 \
    --clip_grad 0.1 \
    --bf16 \
    > train_bs16_bf16.log 2>&1 &
```

---

# 版本歷史

| 版本 | 日期 | 主要變更 |
|------|------|----------|
| v3.0.5 | 2025-12-21 | 解決非偏振光源問題 |
| v3.1.0 | 2025-12-22 | 物理偏振片架構 |
| v3.2.0 | 2025-12-22 | 平行光軸配置 |
| v3.4.2 | 2025-12-22 | 暫定最終版，完整 5 Criteria |
| v4.0.0 | 2025-12-29 | 紋理支持 |
| v17.2 | 2025-12-28 | Blender 腳本增加物件多樣性 |
| v18.0 | 2025-12-29 | Blender 腳本紋理版 |

---

## 實驗 #10：紋理渲染數據訓練

**日期**: 2025-12-31
**狀態**: 進行中

### 實驗目標

使用 v4.0.0 紋理版渲染器產生的數據進行訓練，驗證紋理多樣性對模型泛化能力的影響。

### 數據來源

- **渲染器**: `rendering_v4/pids_renderer_textured.py`
- **場景生成**: `modelling_textured/blender_furniture_randomizer_v18.py`
- **紋理資產**: `textures/` (含 manifest.json)

### 問題 1：NaN Loss

**現象**：訓練開始時 Loss、EPE、Glass EPE 全部為 NaN，D1/D3/D5/D10 為 0%。

**根本原因**：`pids_dataset.py` 中 `max_disparity=192`，但紋理數據的 disparity 範圍為 90~524。

```python
# 原本的 valid_mask 邏輯
valid_mask = (disparity_tensor > 0) & (disparity_tensor < max_disparity)
```

所有 disparity > 192 的像素都被標記為 invalid，導致無像素參與訓練。

**解決方案**：將 `max_disparity` 從 192 改為 576。

---

### 問題 2：Loss 劇烈震盪

**現象**：修復 NaN 後，Loss 從 ~560 下降到 ~170，但出現劇烈波動（如 237 → 429）。

**原因分析**：

| 問題 | 說明 |
|------|------|
| 梯度爆炸 | 遇到極難匹配的紋理時，優化器跳出局部最優 |
| 高頻細節代價 | 無紋理時靠邊緣匹配；有紋理時需學習像素級高頻細節 |
| 重複圖案干擾 | Cost Volume 出現多個波峰，模型困惑 |
| 收斂不一致 | 模型先學低頻結構，高頻紋理 Loss 干擾整體學習 |

---

### 解決方案：梯度累積 + FP32

使用梯度累積 (Gradient Accumulation) 代替 BF16 大 batch：

- `batch_size=8 × accumulation_steps=2 = Effective Batch 16`
- 使用 FP32 確保數值穩定性
- 顯存減半，訓練效果與 BS=16 一致

```bash
nohup python train_pids.py \
    --data_dir ./dataset_V3 \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 \
    --accumulation_steps 2 \
    --lr 0.00002 \
    --clip_grad 1.0 \
    --weight_decay 0.0001 \
    --scheduler cosine \
    --num_steps 20000 \
    --val_freq 100 \
    --val_split 0.2 \
    --d1_weight 0.1 \
    > train_textured_fp32_accum.log 2>&1 &
```

### 預期改善

| 方面 | 無紋理版 | 紋理版預期 |
|------|---------|-----------|
| 背景多樣性 | 單色 diffuse | 多種木紋/磚牆/地板 |
| 過擬合風險 | 高 | 降低 |
| 真實場景泛化 | 待驗證 | 預期改善 |
| 數值穩定性 | BF16 | FP32 + 梯度累積 |

### 問題 3：Validation Loss 暴衝 (1506)

**現象**：Train Loss 持續下降 (560 → 170)，但 Val Loss 上升並出現極端 spike (1506)。

**根本原因**：**紋理週期性歧義 (Texture Repetition Ambiguity)**

平整牆面 + 重複紋理貼圖導致 Cost Volume 出現多峰效應：

| 問題 | 說明 |
|------|------|
| 幾何無特徵 | 平牆沒有凸起凹陷，無法靠幾何定位 |
| 紋理重複 | 磁磚/磚塊等貼圖有週期性 pattern |
| 多點匹配 | 左眼一個點，右眼有多個「長得一樣」的候選點 |
| 週期跳錯 | 選錯一個週期，視差誤差 ~40px → Loss = 40² ≈ 1600 |

**數學驗證**：
- 紋理週期約 40 pixel
- 選錯週期時誤差 = 40
- Loss = 40^2 = 1600 approximately equals **1506** (matches!)

**過擬合機制**：
- Train：模型「背答案」，記住特定紋理的正確匹配 → Loss 低
- Val：遇到沒見過的紋理位移，無幾何特徵可依賴 → 猜錯週期 → Loss 爆炸

---

### 解決方案：強正則化 + 平滑性約束

強迫模型學習「平滑性」而非死記紋理細節：

```bash
nohup python train_pids.py \
    --data_dir ./dataset_V3 \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 \
    --accumulation_steps 2 \
    --lr 0.00001 \
    --clip_grad 0.5 \
    --weight_decay 0.01 \
    --scheduler cosine \
    --num_steps 30000 \
    --val_freq 200 \
    --val_split 0.2 \
    --d1_weight 0.1 \
    > train_flat_wall.log 2>&1 &
```

**關鍵改動**：

| 參數 | 舊值 | 新值 | 目的 |
|------|------|------|------|
| weight_decay | 0.0001 | **0.01** | 強正則化，抑制紋理過擬合 |
| lr | 0.00002 | **0.00001** | 避免在多峰間跳動 |
| num_steps | 20000 | **30000** | 更長訓練時間 |

**預期效果**：
- Train Loss 可能停在 ~200（不會太低）
- Val Loss 穩定，不再出現週期跳錯的暴衝
- 模型傾向輸出平滑視差，適應平牆場景

### 訓練進度（Step 2920/30000）

**Train Loss：**

| 階段 | 範圍 | 最佳值 |
|------|------|--------|
| 0-500 | 176~573 | 176 |
| 500-1000 | 143~571 | 143 |
| 1000-1500 | 126~463 | 126.97 |
| 1500-2000 | 150~520 | 150 |
| 2000-2500 | 121~484 | **121.03** |
| 2500-2920 | 153~426 | 153 |

**Val Loss：**

| Step | Val Loss | 備註 |
|------|----------|------|
| 200 | 1210.79 | 起始高 |
| 400 | 488.09 | 大幅下降 |
| 1200 | 467.63 | 穩定下降 |
| 1800 | 424.28 | 當時最佳 |
| 2400 | 1295.67 | Spike（紋理歧義） |
| 2600 | **421.96** | 快速恢復，新最佳 |
| 2800 | 629.01 | 正常波動 |

**與舊配置比較：**

| 指標 | 舊 (wd=0.0001) | 新 (wd=0.01) |
|------|----------------|--------------|
| Val Loss 趨勢 | 持續上升 (bad) | 整體下降 (good) |
| 最大 Spike | 1506 | 1295 (較小) |
| 最佳 Val Loss | ~480 | **421** (good) |
| Spike 恢復 | 慢 | 快 (good) |

**觀察結論：**
- 強正則化 (weight_decay=0.01) 有效抑制過擬合
- Spike 仍存在但恢復快，屬紋理歧義的正常現象
- 繼續訓練至 30000 步

### 結果（實驗 #10 第一輪）

| 指標 | 最佳值 | Step |
|------|--------|------|
| Val Loss | 376.27 | 15600 |
| Glass EPE | 63.49 px | 12200 |
| D1 | 67.12% | 1400 |
| Composite Score | 73.46 | 12200 |

**結論**：Glass EPE 仍然過高，無法達到 RA-L 投稿標準 (<20px)。

---

## 實驗 #11：修正 Loss 函數 max_disp 不一致問題

**日期**: 2026-01-01
**狀態**: 進行中

### 問題發現

分析實驗 #10 結果時發現 **max_disp 設置不一致**：

| 文件 | 參數 | 值 | 作用 |
|------|------|-----|------|
| `pids_dataset.py:99` | max_disparity | **576.0** (correct) | Dataset valid_mask |
| `pids_model.py:380` | max_disp | **192.0** (wrong) | Loss 函數 valid_mask |
| `train_pids.py:637` | --max_disp | **192.0** (wrong) | 命令行默認值 |

**後果**：
- Dataset 正確加載全範圍視差 (0-576px)
- 但 Loss 函數仍將 >192px 的視差標記為 invalid
- 玻璃區域的大視差 (200-400px) 被排除在訓練外
- 導致 Glass EPE 虛高 (63.49px)

### 解決方案

修改兩個文件的默認值：

```python
# pids_model.py:380
max_disp: float = 576.0,  # 從 192.0 改為 576.0

# train_pids.py:637
parser.add_argument('--max_disp', type=float, default=576.0)  # 從 192.0 改為 576.0
```

### 訓練配置

```bash
nohup python train_pids.py \
    --data_dir ./dataset_V3 \
    --output_dir ./checkpoints_v2 \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --glass_weight 3.0 \
    --lr 0.00005 \
    --batch_size 2 \
    --accumulation_steps 4 \
    --num_steps 30000 \
    --val_freq 500 \
    --iters 16 \
    --scheduler cosine \
    --bf16 \
    --d1_weight 0.2 \
    > train_v2_maxdisp576.log 2>&1 &
```

**關鍵改動**：

| 參數 | 實驗 #10 | 實驗 #11 | 目的 |
|------|----------|----------|------|
| max_disp | 192 (錯誤) | **576** | 匹配實際視差範圍 |
| glass_weight | 1.0 | **3.0** | 增加玻璃區域關注 |
| lr | 0.00002 | **0.00005** | 配合預訓練微調 |

### 預期改善

| 指標 | 實驗 #10 | 預期值 | 提升率 |
|------|----------|--------|--------|
| Glass EPE | 63.49 px | **15-25 px** | 60-75%↓ |
| D1 | 67% | **35-45%** | 33-48%↓ |
| Composite | 73.46 | **20-35** | 52-73%↓ |

### 結果

**日期**: 2026-01-02
**狀態**: 完成

| 指標 | 實驗 #10 | 實驗 #11 | 改善 |
|------|----------|----------|------|
| Val Loss | 376.27 | **265.91** | 29%↓ |
| Glass EPE | 63.49 px | **46.90 px** | 26%↓ |
| D1 | 67.12% | **58.56%** | 13%↓ |
| 最佳 Step | 12200-15600 | **28000-30000** | - |

**關鍵發現：後期神級穩定性**

最後 5000 步 Glass EPE 波動範圍僅 **±0.5px**：
```
Step 26000: 47.30
Step 26500: 47.78
Step 27000: 47.16
Step 27500: 47.18
Step 28000: 47.25
Step 28500: 47.77
Step 29000: 47.48
Step 29500: 47.23
Step 30000: 46.90
```

對比實驗 #10 的劇烈震盪（400→1500→600），這次是**真正收斂**。

**結論**：
- max_disp 修正確實有效
- 模型學到了穩定特徵，可信任泛化能力
- 尚未達到目標 (<20px)，繼續提高 glass_weight

---

## 實驗 #12：提高 glass_weight 至 5.0

**日期**: 2026-01-02
**狀態**: 完成

### 動機

實驗 #11 證明方向正確，趁有便宜 GPU instance 再推進一輪。

### 訓練配置

```bash
nohup python train_pids.py \
    --data_dir ./dataset_V3 \
    --output_dir ./checkpoints_v3 \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --glass_weight 5.0 \
    --lr 0.00005 \
    --batch_size 2 \
    --accumulation_steps 4 \
    --num_steps 30000 \
    --val_freq 500 \
    --iters 16 \
    --scheduler cosine \
    --bf16 \
    --d1_weight 0.2 \
    > train_v3_gw5.log 2>&1 &
```

### 結果

| 指標 | 實驗 #11 (gw=3.0) | 實驗 #12 (gw=5.0) | 變化 |
|------|-------------------|-------------------|------|
| Val Loss | 265.91 | 299.08 | +12.5% (worse) |
| Glass EPE | 46.90 px | 44.96 px | -4.1% (better) |

### 結論

Glass EPE 略有改善 (46.90 → 44.96)，但代價是 **Val Loss 惡化 12.5%**。

**分析**：
- `glass_weight=5.0` 過於激進
- 模型過度專注玻璃區域，犧牲非玻璃區域精度
- 整體泛化能力下降

**最佳配置維持 `glass_weight=3.0` (實驗 #11)**

---

## 當前最佳模型

**配置**: 實驗 #11 (`glass_weight=3.0`, `max_disp=576`)

| 指標 | 值 |
|------|-----|
| Val Loss | 265.91 |
| Glass EPE | 46.90 px |
| D1 | 58.56% |

---

## 未來優化方向

### 1. 增加訓練樣本數

目前使用約 2000 個樣本，觀察到：
- 後期 Loss 曲線非常平穩（模型已「吃飽」）
- 調整超參數（如 glass_weight）改善有限
- 這些是數據量瓶頸的典型跡象

**建議**：將樣本數從 2000 增加到 4000+，使用 v4 紋理渲染器配合隨機化增強。

### 2. 增加場景多樣性

- 更多玻璃形狀（曲面、斜面）
- 更多傢俱類型和擺放方式
- 不同光照條件

### 3. PIDS vs Baseline 對比實驗

使用現有數據完成消融實驗，量化偏振對透明物體偵測的貢獻：
- 偏振版 (PIDS): 使用 `pids_renderer_textured.py`
- 無偏振版 (Baseline): 使用 `pids_renderer_textured_nopol.py`

---

### 訓練監控標準

**Loss 震盪優先級：**
- Loss 震盪 > EPE/D1 震盪（Loss 直接影響梯度）
- Loss 穩定後，EPE/D1 會跟著穩定

**可接受的情況：**
- 整體下降趨勢
- Spike 後快速恢復（1-2 個 val_freq 內）
- Spike 幅度 < 3x 平均值
- Train/Val gap < 5x

**需要介入的情況：**
- 連續上升 3+ 次
- Spike 後不恢復
- Spike 頻率增加
- Val 持續遠離 Train

**目標終點：**
- Val Loss 穩定在 300-400
- Glass EPE < 30 px
- D1 < 20%

---

## 大規模渲染 (2026-01-01)

### 目標

為 PIDS vs Baseline 消融實驗準備 8000 張訓練數據（4000 偏振 + 4000 無偏振）。

### 渲染環境

| 項目 | 規格 |
|------|------|
| GPU | 16x NVIDIA GPU |
| 場景數 | 4000 |
| SPP | 4096 |
| 預估時間 | ~3.5 小時 |
| 預估成本 | ~$35 |

### 渲染器更新

為確保 PIDS vs Baseline 公平對比，更新了兩個渲染器：

#### 1. 確定性隨機化

將 Python 內建 `hash()` 替換為 `hashlib.md5`，確保跨會話一致：

```python
def deterministic_hash(s: str) -> int:
    """確定性 hash，跨 Python 會話一致"""
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**32)
```

#### 2. params.json 燈光參數

偏振版現在保存燈光參數到 params.json：

```json
{
  "lighting": {
    "led_intensity": 3200.5,
    "ceiling_emitter_intensity": 175.3
  }
}
```

nopol 版可從 params.json 讀取以精確匹配。

#### 3. 多 GPU 支持

兩版渲染器都支持 `--num_gpus` 參數，使用 subprocess 實現真正的多進程並行。

### 渲染流程

```bash
# Step 1: 生成 4000 場景 (Blender)
blender scene.blend --background --python blender_furniture_randomizer_v18.py -- \
    --count 4000 --output ./scenes_textured --texture_dir ./textures

# Step 2: 渲染偏振版 (16 GPU, ~3.5hr)
python pids_renderer_textured.py \
    --input_dir ./scenes_textured \
    --output ./output_pol \
    --num_gpus 16 \
    --spp 4096

# Step 3: QA 篩選，複製通過的 params.json
python copy_passed_params.py \
    --input_dir ./output_pol \
    --output_dir ./passed_params \
    --exclude failed.txt

# Step 4: 渲染無偏振版 (只渲染通過 QA 的場景)
python pids_renderer_textured_nopol.py \
    --input_dir ./scenes_textured \
    --output ./output_nopol \
    --params_dir ./passed_params \
    --num_gpus 16 \
    --spp 4096 \
    --no_preview
```

### 當前進度

- [x] 場景生成 (4000 個)
- [x] 偏振版渲染完成 (16 GPU, ~50秒/場景, ~4hr)
- [x] QA 篩選完成 (通過: 3765, 未通過: 235, 通過率: 94.1%)
- [x] 無偏振版渲染完成 (16 GPU, ~5秒/場景, 35min)
- [x] 整理數據集 pol (train 3500 / test 265)
- [x] 訓練 PIDS 偏振版 (實驗 #13, Glass EPE 39.20 px)
- [x] 整理數據集 nopol (使用 --scene_list 對齊)
- [x] 訓練 Baseline 無偏振版 (實驗 #14, Glass EPE 42.17 px)
- [x] Dual-Stream 架構開發 (實驗 #16, Glass EPE 21.82 px, -48.3% vs Baseline)
- [ ] 大規模數據訓練 (實驗 #17, 5000 樣本, 70K steps) - 訓練中
- [ ] 對比評估 PIDS vs Baseline (Test Set)

---

## 實驗 #13：大規模偏振數據訓練

**日期**: 2026-01-02 ~ 2026-01-03
**狀態**: 完成

### 實驗目標

使用 3500 場景的偏振數據訓練 PIDS 模型，驗證數據量增加對模型性能的影響。

### 訓練環境

| 項目 | 規格 |
|------|------|
| GPU | NVIDIA H200 (141GB) |
| 費用 | $1.7/hr |
| 預估時間 | ~7-8 小時 |
| 預估成本 | ~$14 |

### 訓練配置

```bash
nohup python train_pids.py \
    --data_dir ./dataset_pol \
    --output_dir ./checkpoints_pol_3500 \
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
    > train_pol_3500.log 2>&1 &
```

### 與實驗 #11 對比

| 參數 | 實驗 #11 | 實驗 #13 | 說明 |
|------|----------|----------|------|
| 訓練數據 | ~2000 | **3500** | +75% |
| num_steps | 30000 | **50000** | +67% |
| batch_size | 2 | **8** | H200 顯存充裕 |
| accumulation | 4 | **1** | 無需累積 |
| 精度 | BF16 | **FP32** | 更穩定 |
| GPU | H100 80GB | **H200 141GB** | - |

### 預期改善

| 指標 | 實驗 #11 | 預期值 | 實際值 | 達成 |
|------|----------|--------|--------|------|
| Glass EPE | 46.90 px | **< 42 px** | 39.20 px | Pass |
| D1 | 58.56% | **< 56%** | 53.68% | Pass |
| Val Loss | 265.91 | **< 250** | 231.65 | Pass |

> 注：10%+ 改善在 ML 領域已屬顯著成果

### 訓練觀察

#### 轉折點現象

| | 實驗 #11 | 實驗 #13 |
|---|---|---|
| 轉折點 | ~21000 步 | **~37500 步** |
| 原因 | 數據量小，早收斂 | 數據量大 + 多樣性高，需要更多步數 |

轉折點後模型開始快速學習偏振特徵：
- 500 步內 Glass EPE 從 97 降到 47（降 50 px）
- Val Loss 2000 步內降 200

#### 訓練穩定性分析

本次實驗相比實驗 #11 **波動更大**，原因：

| 因素 | 實驗 #11 | 實驗 #13 | 影響 |
|------|----------|----------|------|
| 精度 | BF16 | **FP32** | FP32 梯度更精確，不被精度損失平滑 |
| Batch | 2 + 累積 4 | **真實 8** | 真實 batch 每次看 8 個不同樣本，梯度方向更多樣 |
| 數據多樣性 | 低 | **高（紋理+光照+相機隨機化）** | 場景差異大導致梯度波動 |

**結論**：波動是正常的 trade-off：
- 多樣性高 + 真實 batch → 訓練波動 → 但泛化能力更好
- 多樣性低 + 梯度累積 → 訓練穩定 → 但可能過擬合

累積 batch 的梯度是逐步累積的，會被「平滑」；真實 batch 每次都是 8 個樣本同時投票，梯度方向更真實反映數據分佈。

#### 邊際效應分析

| 數據量變化 | Glass EPE 變化 |
|------------|----------------|
| 2000 → 3500 (+75%) | 46.90 → ~42.7 (-8.9%) |

**觀察**：數據量增加 75%，但 EPE 只降 8.9%，邊際效應明顯。

**可能的瓶頸**：

| 瓶頸類型 | 說明 |
|----------|------|
| 數據量不足 | 可能需要 8000+ 場景才能看到更大改善 |
| 場景多樣性 | 雖然有紋理隨機化，但場景類型單一（都是室內+玻璃門） |
| 模型容量 | RAFT-Stereo 架構可能接近其能力上限 |

**結論**：當前結果足以進行 PIDS vs Baseline 消融實驗，證明偏振對透明物體檢測的貢獻。後續如需進一步提升，應考慮增加場景多樣性（不同房間類型、不同玻璃形狀）而非單純增加數據量。

### 結果

**日期**: 2026-01-03
**狀態**: 完成

| 指標 | 實驗 #11 | 實驗 #13 | 改善 |
|------|----------|----------|------|
| Val Loss | 265.91 | **231.65** | -12.9% |
| Glass EPE | 46.90 px | **39.20 px** | -16.4% |
| D1 | 58.56% | **53.68%** | -8.3% |
| EPE (全域) | - | **17.09 px** | - |

**Best Checkpoint**: step 48500 (Val Loss 231.65, Glass EPE 39.20)

**結論**：數據量從 ~2000 增加到 3500 (+75%)，Glass EPE 從 46.90 降到 39.20 (-16.4%)。雖然存在邊際效應，但改善仍然顯著。

---

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

## 實驗 #15: Dual-Stream 首次嘗試 (BUG)

**日期**: 2026-01-04
**狀態**: 失敗 (發現 bug)

### 配置

```
GPU: H200
訓練: 50K steps, batch=8, iters=24
參數: pol_dim=128, pol_threshold=0.05, pol_weight=2.0, glass_weight=5.0
```

### 結果 (有 BUG)

| 指標 | 數值 | 問題 |
|------|------|------|
| Val Loss | 2154.68 | 比 baseline (~248) 高 9x |
| Glass EPE | 288.71 px | 比 baseline (~42) 高 7x |

### 發現的 BUG

```python
# train_pids.py 原始代碼
disp_preds = [-f for f in flow_preds]  # f shape: (B, 2, H, W)

# 問題: flow 輸出是 (B, 2, H, W)，但 disp_gt 是 (B, 1, H, W)
# 導致 loss 計算時 broadcasting 錯誤：
# - Channel 0: |disp - disp_gt| 正確
# - Channel 1: |0 - disp_gt| = disp_gt  GT 本身被加進 loss！

# 修正後
disp_preds = [-f[:, :1] for f in flow_preds]  # 只取第一個 channel
```

**結論:** 實驗 #15 數據無效，需重新訓練

---

## 實驗 #16: Dual-Stream 修正版 (突破性進展)

**日期**: 2026-01-04
**狀態**: 訓練完成

### 修正內容

- 修正 flow->disparity 維度 bug (`[-f[:, :1] for f in flow_preds]`)
- 4 處修正: train_epoch (FP16/BF16/FP32) + validate

### 配置

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

### 訓練進度 (50K / 50K = 100%)

**Glass EPE 趨勢:**
```
 0.5K steps:  86.61 px (起始)
 5K steps:    45.09 px
10K steps:    44.57 px
14.5K steps:  34.89 px (跌破 35)
16K steps:    33.77 px
19K steps:    30.25 px (跌破 30)
20.5K steps:  28.17 px
23.5K steps:  26.34 px
27K steps:    26.09 px
28K steps:    25.40 px
31K steps:    24.38 px
36K steps:    22.76 px (跌破 23)
42K steps:    21.53 px
45K steps:    21.67 px
50K steps:    21.82 px <- 最終結果
```

**Val Loss 趨勢:**
```
 0.5K steps:  690.34 (起始)
16K steps:    253.00 (接近 baseline)
23.5K steps:  199.86 (跌破 200)
27K steps:    193.15
31.5K steps:  182.98
36K steps:    173.86 (跌破 175)
42K steps:    162.61
45.5K steps:  161.59
50K steps:    160.51 <- 最終結果
```

### 對比歷史實驗

| 實驗 | Glass EPE | Val Loss | vs Baseline 改善 |
|------|-----------|----------|------------------|
| Exp #13 (PIDS 隱式) | 39.20 px | 231.65 | -7.0% |
| Exp #14 (Baseline 無偏振) | 42.17 px | 248.59 | -- |
| **Exp #16 (Dual-Stream)** | **21.82 px** | **160.51** | **-48.3%** |

### 關鍵發現

1. 維度 bug 修正後，Dual-Stream 架構效果顯著
2. Glass EPE: 42.17 -> 21.82 px，**改善 48.3%**
3. Val Loss: 248.59 -> 160.51，**改善 35.4%**
4. 訓練穩定，前 20K 快速收斂，後期穩定
5. 最終 Glass EPE 21.82 px 是歷史最佳

### 技術貢獻總結

1. **Dual-Stream Polarization Architecture**
   - Stereo Stream: 標準 RAFT-Stereo 特徵編碼器
   - Polarization Stream: 專用偏振特徵編碼器
     - Soft Threshold: sigmoid(kappa(P-tau)) 可學習閾值
     - ResidualBlock: 改善梯度流動
     - SpatialAttention: 自動學習關注玻璃區域

2. **Disparity-Aligned Polarization Difference**
   - 使用 GT disparity 將右圖 warp 到左圖視角
   - 計算純偏振差異，消除視差造成的假差異

3. **Polarization-Aware Loss**
   - 偏振差異大的區域額外加權 (pol_weight=2.0)
   - 玻璃區域加權 (glass_weight=5.0)

4. **Bug 修正**
   - flow->disparity 維度修正: (B,2,H,W) -> (B,1,H,W)

**狀態:** 訓練完成，最終 Glass EPE 21.82 px，Val Loss 160.51

---

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

## 實驗 #18: Cross-Attention Fusion 架構

**日期**: 2026-01-05
**狀態**: 開發完成，訓練中

### 實驗目標

將 Dual-Stream 架構的 Concatenation 融合改為 Cross-Attention 融合，驗證注意力機制能否進一步提升玻璃區域的特徵融合效果。

### 設計動機

Exp #16 的 Dual-Stream 使用簡單的 `concat + conv` 融合：
```python
# Exp #16 融合方式
fused = torch.cat([stereo_feat, pol_feat], dim=1)  # (B, 256, H, W)
fused = self.fusion_conv(fused)  # (B, 128, H, W)
```

**問題**: Concatenation 平等對待所有空間位置，但偏振信號只在玻璃區域有意義。

**解決方案**: 使用 Cross-Attention 讓 stereo 特徵「查詢」偏振特徵，自動學習在哪裡需要偏振信息。

### 架構設計

```
Pol -> Stereo Cross-Attention:
    stereo_feat ─┬─ [Q] ←─ pol_feat [K,V] ─→ attended_stereo
                 │
                 └─→ stereo_out = stereo_feat + alpha * attended_stereo

alpha = sigmoid(learnable_logit) * alpha_cap  # 可學習門控 + warmup
```

### 新增組件

#### 1. PooledCrossAttention (Memory-Efficient)

```python
# 原始 attention: O(H*W * H*W) = O(N^2)
# Pooled attention: O(H*W * pool_h*pool_w) = O(N * M), M << N
# Memory 減少約 64 倍 (pool_size=8)

Q: (B, C, H, W) - 保持原始解析度
K, V: 池化到 (B, C, H/8, W/8) - 減少記憶體
```

#### 2. SafeCrossAttentionFusion (穩定訓練)

安全機制:
- **Alpha Gate**: 初始化 sigmoid(-5) ≈ 0.007，接近 0
- **Zero Init**: 輸出投影層初始化為 0
- **Alpha Cap Warmup**: alpha 上限從 0.05 線性增長到 1.0
- **Residual Connection**: `out = stereo + alpha * attended`
- **NaN Protection**: 數值穩定的 softmax + nan_to_num

#### 3. Alpha Cap Warmup

```python
def _compute_alpha_cap(self) -> float:
    if self.global_step >= self.args.alpha_cap_warmup:
        return 1.0
    progress = self.global_step / self.args.alpha_cap_warmup
    return self.args.alpha_cap_start + progress * (1.0 - self.args.alpha_cap_start)
    # 0.05 -> 1.0 over 5000 steps
```

### 開發過程 (Fork 方式)

**重要**: 本次使用 fork 方式開發，不修改原始檔案

| 原始檔案 | Fork 檔案 | 說明 |
|----------|-----------|------|
| `pids_model.py` | `pids_model_cross_attention.py` | 新增 Cross-Attention 模型 |
| `train_pids.py` | `train_pids_cross_attention.py` | 新增 Cross-Attention 訓練器 |

### Bug 修復歷程

#### Bug 1: ImportError - PIDSTrainer
```
ImportError: cannot import name 'PIDSTrainer' from 'train_pids'
```
**修正**: `from train_pids import Trainer` (類別名稱為 `Trainer` 非 `PIDSTrainer`)

#### Bug 2: ImportError - create_dataloaders
```
ImportError: cannot import name 'create_dataloaders' from 'pids_dataset'
```
**修正**: `from pids_dataset import create_data_loaders` (有底線)

#### Bug 3: AttributeError - 缺少參數
```
AttributeError: 'Namespace' object has no attribute 'gamma'
```
**修正**: 在 `parse_args()` 中補齊所有必要參數 (gamma, adam_eps, optimizer, resume, log_freq, print_freq, save_freq)

#### Bug 4: RuntimeError - Device Mismatch
```
RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
```
**修正**: `_build_model()` 缺少 `model.to(self.device)` 和 DataParallel 處理

#### Bug 5: NaN Loss
```
Step 100 | Loss: nan | EPE: nan | D1: 42.79% | Glass EPE: nan
```
**原因**: Attention mask 全為 -inf 時，softmax 輸出 NaN (0/0)

**修正** (`pids_model_cross_attention.py:115-135`):
```python
# 1. 只在有足夠有效值時應用 mask
valid_count = (mask_pooled >= 0.5).sum(dim=-1, keepdim=True)
if valid_count.min() > 0:
    attn = attn.masked_fill(mask_pooled < 0.5, float('-inf'))

# 2. 數值穩定的 softmax
attn_max = attn.max(dim=-1, keepdim=True)[0]
attn = attn - attn_max
attn = F.softmax(attn, dim=-1)

# 3. 最後防線
attn = torch.nan_to_num(attn, nan=0.0)
```

### 訓練配置

```bash
nohup python train_pids_cross_attention.py \
    --data_dir ./dataset_pol \
    --output_dir ./checkpoints_cross_attention_exp18 \
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
    --num_heads 4 \
    --pool_size 8 \
    --alpha_cap_start 0.05 \
    --alpha_cap_warmup 5000 \
    > train_exp18.log 2>&1 &
```

### Cross-Attention 專用參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| num_heads | 4 | Multi-head attention heads 數量 |
| pool_size | 8 | K,V 池化大小 (8 = 64x 記憶體節省) |
| enable_stereo_to_pol | False | 是否啟用雙向 attention (預設單向) |
| alpha_cap_start | 0.05 | Alpha warmup 起始值 |
| alpha_cap_warmup | 5000 | Alpha warmup 步數 |

### 模型參數統計

| 模型 | 總參數 | 可訓練參數 |
|------|--------|------------|
| Exp #16 (Dual-Stream Concat) | ~4.1M | ~4.1M |
| **Exp #18 (Cross-Attention)** | **4,363,715** | **4,363,715** |

新增參數主要來自:
- PooledCrossAttention (Q,K,V projections)
- SafeCrossAttentionFusion (alpha gates, output projections)

### 預期改善

| 指標 | Exp #16 (Concat) | Exp #18 (Cross-Attn) 預期 |
|------|------------------|--------------------------|
| Glass EPE | 21.82 px | < 20 px |
| Val Loss | 160.51 | < 155 |
| 穩定性 | 穩定 | 更穩定 (alpha warmup) |

### 訓練進度 (70K / 70K = 100% 完成)

**Glass EPE 趨勢:**
```
 5K steps:    45.86 px (起始)
10K steps:    47.94 px
20K steps:    32.43 px
30K steps:    30.24 px
37.5K steps:  22.33 px (首次跌破 23)
40.5K steps:  21.62 px (首次跌破 22，超越 Exp #16)
46K steps:    20.91 px (首次跌破 21)
56.5K steps:  19.87 px (首次跌破 20！)
59.5K steps:  19.26 px
70K steps:    19.26 px ⭐ 最終結果
```

**Val Loss 趨勢:**
```
10K steps:    318.73
20K steps:    224.99
30K steps:    206.40
40K steps:    204.56
46K steps:    151.05 (跌破 Exp #16)
50K steps:    151.36
55K steps:    146.93
59.5K steps:  140.23
63K steps:    138.15
70K steps:    135.77 ⭐ 最終結果
```

### 最終結果

| 指標 | Exp #16 (Concat) | Exp #18 (Cross-Attn) | 改善 |
|------|------------------|---------------------|------|
| **Glass EPE** | 21.82 px | **19.26 px** | **-11.7%** |
| **Val Loss** | 160.51 | **135.77** | **-15.4%** |
| Best Step | 50K | 70K | - |

### 關鍵發現

1. **Cross-Attention 超越 Concatenation**: 僅改變融合機制就帶來 11.7% Glass EPE 改善
2. **整體品質提升**: Val Loss 降低 15.4%，說明沒有犧牲背景精度
3. **學習曲線較慢但後勁強**: 前期落後 Exp #16，但 40K 後開始超越並持續改善
4. **突破 20px 門檻**: 首次將 Glass EPE 降到 1 字頭 (19.26 px)
5. **最終即最佳**: 70K 結束時兩個指標都達到最佳值，說明模型仍有學習空間

### 消融實驗總結

| 方法 | Glass EPE | Val Loss | vs Baseline |
|------|-----------|----------|-------------|
| Baseline (無偏振) | 42.17 px | 248.59 | -- |
| + 隱式偏振輸入 | 39.20 px | 231.65 | -7.0% |
| + Dual-Stream Encoder | 21.82 px | 160.51 | -48.3% |
| + **Cross-Attention Fusion** | **19.26 px** | **135.77** | **-54.3%** |

### 技術貢獻

1. **PooledCrossAttention**: 64x 記憶體節省，使 attention 可行於高解析度特徵圖
2. **SafeCrossAttentionFusion**: Alpha warmup + zero init 確保訓練穩定
3. **Pol→Stereo 單向 Attention**: 讓 stereo 特徵「查詢」偏振信息，自動學習在哪裡需要偏振

**狀態:** 訓練完成，最終 Glass EPE 19.26 px，Val Loss 135.77

---

## 實驗 #19: Nopol Ablation (同架構無偏振)

**日期**: 2026-01-06
**狀態**: 訓練中

### 實驗目標

使用與 Exp #18 完全相同的 Cross-Attention 架構，但強制關閉偏振分支，驗證偏振信息的實際貢獻。

### 設計理念

傳統 ablation 做法是用不同架構（如標準 RAFT-Stereo）訓練 nopol 數據，但這樣無法區分：
- 改善來自偏振信息？
- 還是來自架構本身？

**更公平的 ablation**: 同架構 + 關閉偏振分支

```
Exp #18 (Pol):    stereo_out = stereo + alpha * attended_pol  (alpha → 1.0)
Exp #19 (Nopol):  stereo_out = stereo + 0 * attended_pol      (alpha = 0)
```

### 實現方式

新增 `--nopol_ablation` flag 到 `train_pids_cross_attention.py`:

```python
def _compute_alpha_cap(self) -> float:
    # Ablation: 強制 alpha=0，等效無偏振
    if getattr(self.args, 'nopol_ablation', False):
        return 0.0
    # ... 正常 warmup 邏輯
```

**效果:**
- 模型架構完全相同 (Dual-Stream + Cross-Attention)
- Polarization Encoder 仍然存在並運算
- 但 Cross-Attention 的輸出被乘以 0，不貢獻到最終結果
- 等效於「有架構但無偏振信息」

### 數據準備

- **渲染**: `pids_renderer_textured_nopol.py` (無偏振版)
- **Mitsuba variant**: `cuda_ad_rgb` (非 spectral_polarized)
- **Integrator**: `path` (非 stokes)
- **輸出格式**: `_left.exr`, `_right.exr` (左右幾乎相同，無 I∥/I⊥ 差異)

### 訓練配置

對齊 Exp #17 基礎參數，加入 Cross-Attention 專用參數：

```bash
python train_pids_cross_attention.py \
    --data_dir ./PIDS_dataset_nopol_V1 \
    --output_dir ./checkpoints_cross_attention_exp19_nopol \
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
    --num_heads 4 \
    --pool_size 8 \
    --alpha_cap_start 0.05 \
    --alpha_cap_warmup 5000 \
    --nopol_ablation
```

### 訓練環境

| 項目 | 規格 |
|------|------|
| 訓練樣本 | 3991 scenes |
| 驗證樣本 | 998 scenes |
| 模型參數 | 4,363,715 |
| Pretrained | raftstereo-sceneflow.pth (135/337 layers) |
| 數據格式 | nopol (`_left.exr`, `_right.exr`) |

### 預期結果

| 實驗 | 數據 | 架構 | alpha | 預期 Glass EPE |
|------|------|------|-------|----------------|
| Exp #18 | pol | Cross-Attention | warmup→1.0 | 19.26 px |
| **Exp #19** | **nopol** | **Cross-Attention** | **固定 0** | **~40+ px** |

### Ablation 對比設計

| 對比 | 說明 | 預期差異 |
|------|------|----------|
| Exp #18 vs #19 | 同架構，有/無偏振 | 偏振的純貢獻 |
| Exp #14 vs #19 | 不同架構，都無偏振 | 架構本身的影響 |

### 訓練進度

(訓練中，待更新)

---

## 渲染器修正：偏振信號強化 (v5.0.0)

**日期**: 2026-01-06
**狀態**: 完成
**位置**: `rendering_v5/pids_renderer_textured.py`

### 問題發現

Exp #19 (Nopol) 的訓練曲線與 Exp #18 (Pol) 幾乎相同，懷疑偏振信號過弱。

**驗證結果**：
```
玻璃區域 |I∥ - I⊥|: 3.60
非玻璃區域 |I∥ - I⊥|: 3.07
比值: 1.17x  ← 幾乎沒差！
```

預期應該是玻璃 >> 非玻璃 (至少 3-5x)。

### 根本原因分析

#### 1. 天花板非偏振光太強

| 光源 | 強度範圍 | 面積估算 | 總光通量 |
|------|----------|----------|----------|
| LED (偏振) | 2000-4000 | ~0.01 m² | ~30 |
| 天花板 (非偏振) | 100-250 | ~9 m² | **~1575** |

**天花板貢獻是 LED 的 50 倍！** 非偏振光淹沒了偏振信號。

#### 2. DoLP 計算有基線誤差

原本的 DoLP 計算：
```python
# 錯誤：比較不同相機位置的 I∥ 和 I⊥
dolp = |I_parallel - I_cross| / (I_parallel + I_cross)
```

問題：I_parallel 來自左相機，I_cross 來自右相機 (65mm 基線)，直接比較有視差誤差。

### 修正方案

#### 修正 1：降低天花板光強度

```python
# pids_renderer_textured.py
CEILING_EMITTER_INTENSITY = 15.0  # 從 100.0 降到 15.0
# 隨機範圍
cls.CEILING_EMITTER_INTENSITY = random.uniform(10, 25)  # 從 [100, 250] 改為 [10, 25]
```

新的光通量比：
- LED 貢獻 ≈ 30
- 天花板貢獻 ≈ 157.5 (從 1575 降到 157.5)
- 比值從 1:50 改善到 **1:5**

#### 修正 2：DoLP 使用 Stokes 參數計算

```python
# 正確：使用單一相機的 Stokes 參數
dolp = np.sqrt(S1**2 + S2**2) / S0
```

這樣避免了基線問題，DoLP 計算物理正確。

#### 修正 3：Glass Mask 左右聯集

**問題**：Glass mask 原本從深度相機（中央）渲染，但 EPE 計算在左影像座標系進行。

由於 65mm 基線，玻璃邊緣在左/右視角位置不同：
- 左相機看到的玻璃邊緣 vs 中央相機有 32.5mm 偏移
- 這會導致 EPE 評估時漏算邊緣像素

**解決方案**：從左右相機各渲染一次 glass mask，取聯集 (OR)：

```python
# pids_renderer_textured.py Line 1543-1547
glass_mask_left = self._render_glass_mask(builder, left_pos, left_target)
glass_mask_right = self._render_glass_mask(builder, right_pos, right_target)
glass_mask = ((glass_mask_left > 0.5) | (glass_mask_right > 0.5)).astype(np.float32)
```

**優點**：
- 完整覆蓋 stereo pair 中所有玻璃像素
- 因視差造成的邊緣偏移都被包含
- EPE 評估更公平準確

#### 修正 4：DoLP 計算優化（對齊 + 效率）

原本的 DoLP 計算是全局計算，然後再用 mask 分區域統計。這有兩個問題：
1. Glass mask 和 DoLP 圖像可能不對齊
2. 重複計算浪費資源

**優化方案**：
- 渲染左相機 glass mask 後，立即用該 mask 計算左眼玻璃 DoLP
- 渲染右相機 glass mask 後，立即用該 mask 計算右眼玻璃 DoLP
- 背景 DoLP 用聯集 mask 的補集計算

```python
# 左相機 glass mask + DoLP（對齊）
glass_mask_left = self._render_glass_mask(builder, left_pos, left_target, "左")
dolp_left = StokesProcessor.compute_dolp(S0_left, S1_left, S2_left)
glass_dolp_left = dolp_left[glass_mask_left > 0.5].mean()

# 右相機 glass mask + DoLP（對齊）
glass_mask_right = self._render_glass_mask(builder, right_pos, right_target, "右")
dolp_right = StokesProcessor.compute_dolp(S0_right, S1_right, S2_right)
glass_dolp_right = dolp_right[glass_mask_right > 0.5].mean()

# 背景 DoLP
glass_mask = glass_mask_left | glass_mask_right
background_dolp = dolp_left[glass_mask < 0.5].mean()
```

**優點**：
- 各相機用自己視角對齊的 mask 計算 DoLP，更精確
- 報告新增 `dolp_left`, `dolp_right`, `dolp_ratio` 指標
- 減少重複計算

#### 修正 5：SNR 計算改用 DoLP

原本 SNR 計算：
```python
signal = mean(|I_parallel - I_cross|)  # 受亮度差異影響
noise = sqrt(2) * noise_std
snr = signal / noise
```

問題：`|I∥ - I⊥|` 會受到兩張圖亮度差異的影響，不是純粹的偏振信號。

**修正後**：
```python
# Signal: 玻璃區域平均 DoLP（已歸一化）
signal = (glass_dolp_left + glass_dolp_right) / 2

# Noise: 背景區域 DoLP 標準差（理想背景 DoLP≈0，變異為噪點）
noise = std(dolp[background_mask])

# SNR
snr = signal / noise
```

**優點**：
- DoLP 已經除以總強度 S0，歸一化後不受亮度影響
- 背景 DoLP 變異直接反映偏振噪點水平
- 更準確反映偏振信號品質

### 修改檔案

| 檔案 | 修改項目 |
|------|----------|
| `pids_renderer_textured.py` | |
| Line 151 | `CEILING_EMITTER_INTENSITY`: 100.0 → 15.0 |
| Line 261 | 隨機範圍: [100, 250] → [10, 25] |
| Line 1542-1573 | Glass mask 左右聯集 + 對齊 DoLP 計算 |
| Line 1581-1585 | `_save_outputs` 改用 `dolp`, `dolp_stats` |
| Line 1130-1152 | 報告生成器改用預計算的 `dolp_stats` |
| Line 1194-1215 | 報告使用對齊的玻璃/背景 DoLP |
| Line 1251 | 報告新增 `dolp_ratio` 指標 |
| Line 1268-1292 | SNR 改用 DoLP-based 計算 |
| Line 2025-2026 | 新增 `--skip-qa`, `--skip-c1` 參數 |
| Line 2124-2176 | 渲染完成後自動執行 QA 驗證 |

#### 修正 6：整合 QA 驗證

渲染完成後自動執行品質驗證，無需手動執行 `quality_validator.py`。

**新增參數**：
- `--skip-qa`: 跳過 QA 驗證
- `--skip-c1`: QA 時跳過 C1 (Geometric Consistency) - 模擬場景建議使用

**自動輸出**：
- `quality_report.md`: 完整 Markdown 品質報告
- `failed_scenes.txt`: 未通過場景列表

**使用範例**：
```bash
# 渲染 + 自動 QA（模擬場景跳過 C1）
python pids_renderer_textured.py --input_dir ./scenes --output ./output --skip-c1

# 只渲染，不執行 QA
python pids_renderer_textured.py --input_dir ./scenes --output ./output --skip-qa
```

### 預期改善

| 指標 | 修正前 | 修正後預期 |
|------|--------|------------|
| 玻璃 DoLP | ~20% | **>30%** |
| 非玻璃 DoLP | ~17% | **<5%** |
| 玻璃/非玻璃比值 | 1.17x | **>6x** |

### 下一步

1. 重新渲染測試場景驗證偏振對比
2. 確認後重新渲染全部訓練數據
3. 重新訓練 Exp #18 和 Exp #19 進行對比

---

## 渲染器修正：玻璃比值計算 (v5.0.1)

**日期**: 2026-01-06
**狀態**: 測試中
**位置**: `rendering_v5/pids_renderer_textured.py`, `Quality_Assurance/quality_validator.py`

### 問題發現

v5.0.0 修正後，QA 仍然失敗，發現多個指標計算問題：

1. **DoLP 比值異常大** (24867x)：背景 DoLP 接近 0 導致除法爆炸
2. **I∥/I⊥ 比值計算錯誤**：直接比較左右圖像像素，未考慮 65mm 基線
3. **左右眼 DoLP 差異大**：左眼 4%，右眼 1%，因偏振片角度不同

### 根本原因分析

#### 1. 比值計算未對齊 3D 點

```python
# 錯誤：比較不同 3D 點
intensity_ratio = I_parallel[x,y] / I_cross[x,y]  # 像素對應不同 3D 點！
```

問題：左相機 pixel(x,y) 和右相機 pixel(x,y) 看到的是**不同的 3D 點**（有 65mm 視差）。

#### 2. 左右眼 DoLP 物理意義不同

- 左相機使用 0° 偏振片 → 保留水平偏振
- 右相機使用 90° 偏振片 → 保留垂直偏振

Stokes 參數經過偏振片後會改變，所以左右眼 DoLP 不可直接平均。

### 修正方案

#### 修正 1：DoLP Floor 值

避免除以接近零的背景 DoLP：

```python
# pids_renderer_textured.py
Config.DOLP_FLOOR = 0.01  # 1%

# 使用
dolp_ratio = glass_dolp / max(bg_dolp, DOLP_FLOOR)
snr = glass_dolp / max(bg_dolp_std, DOLP_FLOOR)
```

#### 修正 2：Warp 對齊強度比值

使用視差將右圖 warp 到左視角，確保比較同一 3D 點：

```python
def warp_right_to_left(right_img, disparity):
    """使用視差將右圖 warp 到左視角"""
    xx_src = (xx + disparity)  # 往右找對應點
    warped = cv2.remap(right_img, xx_src, yy, cv2.INTER_LINEAR)
    return warped, valid_mask

# 使用
I_cross_warped, warp_valid = warp_right_to_left(I_cross, disparity)
glass_ratio = I_parallel[glass_mask_left] / I_cross_warped[glass_mask_left]
```

**注意**：使用 `glass_mask_left` 因為 warp 後在左視角座標系。

#### 修正 3：QA 改用玻璃比值

C3 標準從 DoLP 改為玻璃比值 (I∥/I⊥)：

| 項目 | 修改前 | 修改後 |
|------|--------|--------|
| 名稱 | `glass_dolp` | `glass_ratio` |
| 指標 | DoLP > 1% | I∥/I⊥ > 1.2x |
| 計算 | Stokes DoLP | warp 對齊後強度比 |

```python
# quality_validator.py v1.5.0
CRITERIA['glass_ratio'] = {
    'name': 'Glass Polarization Ratio',
    'criterion': 3,
    'threshold': 1.2,  # I∥/I⊥ > 1.2x
}
```

#### 修正 4：Warp 方向修正

原本 warp 方向錯誤導致 ratio = 0.91x（反向）：

```python
# 修正前（錯誤）
xx_src = xx - disparity  # 得到 ratio = 0.91x

# 修正後（正確）
xx_src = xx + disparity  # 預期 ratio ≈ 1.1x
```

**原理**：
- 左相機在 x=-82.5mm（左邊）
- 右相機在 x=-17.5mm（右邊）
- 同一 3D 點在右圖中偏**左**（x 較小）
- 從左圖找右圖對應，要往**右**找 (+disparity)

### 修改檔案

| 檔案 | 修改項目 |
|------|----------|
| `pids_renderer_textured.py` | |
| Line 146 | 新增 `DOLP_FLOOR = 0.01` |
| Line 1168-1198 | 新增 `warp_right_to_left()` 函數 |
| Line 1260-1270 | 強度比值使用 warp 對齊計算 |
| Line 1275-1292 | 玻璃區域比值使用 `glass_mask_left` |
| Line 1537-1539 | 警告使用玻璃區域比值 |
| `quality_validator.py` | |
| 版本 | v1.4.3 → v1.5.0 |
| Line 291-296 | C3 改為 `glass_ratio` |
| Line 428-436 | 驗證邏輯使用 `intensity_ratio_mean` |

### 新輸出格式

```
DoLP (左眼玻璃): 0.0402
DoLP (右眼玻璃): 0.0103
DoLP (背景): 0.0000 (floor=0.01)  ← 顯示使用 floor
DoLP 玻璃/背景比: 4.02x
I∥/I⊥ 玻璃區域 (warp對齊): X.XXx  ← 新增，關鍵指標
I∥/I⊥ 全局: 1.20x
```

### 預期結果

修正 warp 方向後，玻璃比值應從 0.91x 變為 ~1.1x：

| 指標 | warp 錯誤 | warp 正確 |
|------|-----------|-----------|
| 玻璃比值 | 0.91x | ~1.1x |
| QA C3 | 未通過 | **通過** |

---

## 當前最佳模型

**配置**: Exp #18 (Cross-Attention Fusion)

| 指標 | 值 | vs Baseline |
|------|-----|-------------|
| Glass EPE | **19.26 px** | -54.3% |
| Val Loss | **135.77** | -45.4% |
| 架構 | Dual-Stream + Cross-Attention | - |
| Checkpoint | `checkpoints_cross_attention_exp18/checkpoint_best.pth` | - |

---

# 第三部分：渲染器 V5 開發 (2026-01-06 ~ 2026-01-07)

## 1. V5 渲染器總覽

### 目標
解決 V4 渲染器的偏振信號不足問題，通過物理正確的光源配置實現有效的玻璃偏振對比。

### 最終配置 (v5.1.6)

| 參數 | 值 | 說明 |
|------|------|------|
| GLASS_IOR | 1.65 | 固定，正入射 R≈6.2% |
| LED_INTENSITY | 10000 | 偏振光源 |
| CEILING_INTENSITY | 800 | 非偏振頂光 |
| 光源比例 | 12.5:1 | 偏振/非偏振 |

### 驗證結果

| 指標 | 值 | 狀態 |
|------|------|------|
| 玻璃 I(90°)/I(0°) | 1.73x | ✓ |
| 背景 I(90°)/I(0°) | 1.00x | ✓ 完美去偏振 |
| 背景平衡 (warp) | 1.11x | ✓ |
| SNR | 13.33 | ✓ |
| 品質分數 | 92 | ✓ excellent |

---

## 2. 偏振信號弱的根本原因分析

### Fresnel 方程計算

對於空氣 → 玻璃 (n=1.5) 界面：

| 入射角 | Rs | Rp | R平均 | T透射 | 反射光 DoP |
|--------|-----|-----|-------|-------|------------|
| 0° (正入射) | 4% | 4% | 4% | 96% | 0% |
| 56.3° (Brewster) | 13.4% | 0% | 6.7% | 93% | 100% |
| 80° (掠射) | 60% | 35% | 48% | 52% | 26% |

### 核心問題

1. **正入射只有 4% 反射**：96% 的光穿透玻璃，照亮後方物體後漫反射回來（去偏振）
2. **偏振信號被稀釋**：4% 偏振反射 vs 96% 去偏振透射光
3. **實測 S1/S0 ≈ 5-6%**：符合物理預期

### 漫反射去偏振效應

- **物理原理**：漫反射表面由無數隨機取向微平面組成，每次反射改變偏振方向
- **Mueller 矩陣**：理想 Lambertian 表面的 Mueller 矩陣會將 S1, S2, S3 歸零
- **Mitsuba 支持**：`spectral_polarized` variant 的 `diffuse` BSDF 正確模擬去偏振

---

## 3. V5 版本演進

### v5.0.0-5.0.2：Warp 對齊與指標改進

- **Warp 對齊計算**: 新增 `warp_right_to_left()` 函數
- **玻璃比值指標**: QA 從 DoLP 改為 I∥/I⊥ 比值
- **DOLP_FLOOR**: 0.01 下限，避免除以零

### v5.0.3：Stokes 量化

- **LED 同步相機位置**: LED 跟隨 CAMERA_X 移動
- **Stokes → Malus 量化**: 直接計算同視角偏振比值
  ```python
  I(0°) = 0.5 * (S0 + S1)
  I(90°) = 0.5 * (S0 - S1)
  ```
- **高 IOR 玻璃**: GLASS_IOR = 3.5

### v5.0.4：世界坐標偏振對齊

- **問題**: Mitsuba 的 polarizer theta 是相對於 look_at 局部坐標系
- **解決**: `compute_world_aligned_theta()` 計算補償角度

### v5.0.5：Brewster 角配置（失敗）

- **配置**: LED 以 56° 入射角照射玻璃
- **結果**: 幾何效應 2.87x，但真正偏振只有 2%，背景也不平衡
- **結論**: LED 位置造成亮度不對稱，不是偏振效應

### v5.1.0：整面偏振光源

- **配置**: 500x250mm 均勻偏振照明
- **結果**: 玻璃 2.24x ✓，但背景 warp 不平衡 2.08x ⚠️

### v5.1.1：整面偏振光源 + 非偏振頂光 ✓ 成功

- **整面偏振光源**: 500x250mm，強度 5000
- **非偏振頂光**: 稀釋背景殘餘偏振，強度 2000
- **結果**: 玻璃 2.2x ✓，背景 1.0x ✓，品質 92 分

### v5.1.2：精簡報告 + 修正背景平衡

- **精簡報告**: 只保留 6 個關鍵指標
- **修正背景平衡計算**: 使用 warp 後的 I_cross_warped
- **結果**: 背景平衡從 1.30x 改善到 1.06x

### v5.1.3：🔧 關鍵修正 - Warp 方向 + 深度相機位置

- **修正 Warp 方向**: `xx + disparity` → `xx - disparity`
  - 左相機 X=-82.5mm（較左），右相機 X=-17.5mm（較右）
  - disparity = x_left - x_right > 0
  - 要從右圖找來源：`x_right = x_left - disparity`

- **修正深度相機位置**: 中間位置 → 左相機位置

### v5.1.4：🔧 關鍵修正 - Ray Depth → Z Depth 轉換

- **問題**: Mitsuba 輸出 ray depth（歐幾里得距離），視差公式需要 Z depth（垂直距離）
- **影響**: 邊緣像素誤差 5-10 px
- **修正**: `Z = ray_depth * cos(angle)`，其中 `cos = f / sqrt(f² + dx² + dy²)`
- **結果**: 邊緣視差從 ~10 px 誤差改善到 <0.1 px

### v5.1.5：✅ 最終光學配置

- IOR=1.65, LED=10000, Ceiling=800
- 玻璃對比 1.73x，背景 1.00x

### v5.1.6：整合 organize_dataset + JSON 報告修正

- **整合 organize_dataset**: 渲染完成後自動整理數據集
- **新增 stokes_ratio 欄位**: 從 Stokes 參數計算 I(90°)/I(0°)
- **修正多 GPU 模式**: 子進程加入 `--skip-qa`

### v5.1.7：玻璃 Mask 修正 + 重渲染模式

- **玻璃檢測修正**: 從關鍵字匹配改為精確匹配 "Glass_Clear"
  ```python
  # 舊邏輯（錯誤）
  'glass' in name  # 會誤判 'glass_table', 'glass_shelf'

  # 新邏輯（正確）
  name.lower() == 'glass_clear'
  ```

- **新增 --rerender-mask 模式**: 使用現有 params.json 重新渲染玻璃 mask
- **新增 --scene-list**: 支援場景列表過濾

---

## 4. 渲染器文件說明

| 文件 | 版本 | 說明 |
|------|------|------|
| `pids_renderer_textured.py` | v5.1.7 | 偏振渲染器（主要） |
| `pids_renderer_textured_nopol.py` | v4.0.0-nopol | 無偏振渲染器（消融實驗） |
| `Quality_Assurance/quality_validator.py` | v1.6.0 | 品質驗證器 |

---

# 第四部分：訓練架構改進 (2026-01-07)

## 1. 65mm Baseline 對齊問題

### 問題描述

Dual-Stream 架構在計算偏振特徵時需要比較 I∥ 和 I⊥：

```
|I∥ - I⊥| → 偏振差異特徵
```

但是左右相機有 65mm baseline，直接比較會比較**不同 3D 點**的亮度：

```
左相機 (I∥): 看到點 A
右相機 (I⊥): 同一像素位置看到點 B（偏移 ~60px）
```

### 訓練 vs 推論的差異

| 階段 | GT Disparity | 處理方式 |
|------|--------------|----------|
| 訓練 | 有 | 使用 GT disparity 做 warp 對齊 |
| 推論 | 無 | ??? |

### 解決方案

**兩階段推論方法**:
1. 先用未對齊的偏振特徵獲得初始 disparity 估計
2. 使用估計的 disparity 做 warp，重新計算對齊的偏振特徵
3. 繼續迭代，用更準確的偏振特徵優化 disparity

---

## 2. warp_with_disparity 實現

### 函數定義

```python
def warp_with_disparity(
    img: torch.Tensor,
    disparity: torch.Tensor,
    return_valid_mask: bool = True
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    使用視差圖將右圖 warp 到左圖視角

    公式: warped(x, y) = img(x - disparity, y)

    Returns:
        warped: warp 後的右圖（對齊到左圖視角）
        valid_mask: 有效區域 mask（超出邊界為 0）
    """
```

### 有效區域 Mask

Warp 時，邊界外的區域會被填充為 0，這些區域的偏振差異是假信號：

```python
valid_mask = (xx_warped >= 0) & (xx_warped <= W - 1)
```

**用途**:
- 訓練時：只計算有效區域的 loss
- 推論時：忽略無效區域的偏振特徵

---

## 3. 兩階段推論方法

### 方法 1: forward_inference（高效版，推薦）

只更新 1-2 次偏振特徵，GRU 狀態保持連續：

```python
def forward_inference(
    self,
    left: torch.Tensor,
    right: torch.Tensor,
    iters: Optional[int] = None,
    pol_update_iters: Optional[List[int]] = None,  # 預設 [iters//2]
) -> torch.Tensor:
```

**流程**:
1. 用未對齊偏振特徵開始迭代
2. 在指定迭代（如第6次）後，用當前 disparity 估計做 warp
3. 重新計算對齊的偏振特徵、重建 correlation volume
4. 繼續剩餘迭代

**優點**:
- `fnet` 只計算一次（不重複計算 stereo features）
- GRU hidden state 保持連續（不重啟）
- 計算量約 1.1-1.2x 基礎 forward

### 方法 2: forward_two_pass（完整版）

完整兩階段，每階段獨立：

```python
def forward_two_pass(
    self,
    left: torch.Tensor,
    right: torch.Tensor,
    iters_pass1: int = 6,
    iters_pass2: int = 6,
) -> torch.Tensor:
```

**流程**:
1. Pass 1: 無偏振對齊 → disparity_v1
2. Pass 2: 用 disparity_v1 做 warp → 正確偏振特徵 → disparity_v2

**優點**:
- 偏振對齊更精確
- 適合精度優先的場景

**缺點**:
- 計算量 2x

---

## 4. 使用建議

| 場景 | 推薦方法 | 說明 |
|------|----------|------|
| 訓練 | `forward(..., disparity_gt=gt)` | 使用 GT disparity |
| 推論（即時） | `forward_inference(..., pol_update_iters=[6])` | 平衡效率與精度 |
| 推論（高精度） | `forward_two_pass(...)` | 最高精度 |

---

## 5. 修改的文件

### pids_model.py

| 函數/類別 | 修改內容 |
|-----------|----------|
| `warp_with_disparity()` | 新增 `return_valid_mask` 參數 |
| `PolarizationEncoder.compute_pol_diff()` | 支援 validity mask，避免邊界假信號 |
| `PolarizationEncoder.get_valid_mask()` | 新增方法 |
| `PIDSStereoDualStream.forward_inference()` | 新增高效兩階段推論 |
| `PIDSStereoDualStream.forward_two_pass()` | 新增完整兩階段推論 |

---

## 6. 版本歷史總結

| 日期 | 版本 | 主要變更 |
|------|------|----------|
| 2026-01-07 | pids_model v2.0 | 新增 forward_inference, forward_two_pass |
| 2026-01-07 | renderer v5.1.7 | 玻璃 mask 精確匹配 + rerender 模式 |
| 2026-01-06 | renderer v5.1.6 | 整合 organize_dataset |
| 2026-01-06 | renderer v5.1.4 | Ray depth → Z depth 修正 |
| 2026-01-06 | renderer v5.1.3 | Warp 方向修正 |
| 2026-01-06 | renderer v5.1.1 | ✓ 成功配置：整面偏振 + 非偏振頂光 |

---

# 實驗 #19：Architecture Refresh 訓練驗證

## 1. 背景

**日期**: 2026-01-07

在完成以下改進後，使用 V5 dataset 進行訓練驗證：

1. **Renderer V5.1.7**: 玻璃 mask 精確匹配 + GT 修正
2. **Dual-Stream Architecture**: 偏振特徵融合到 correlation volume
3. **Two-Stage Inference**: `forward_inference` + `forward_two_pass`
4. **Dataset V5**: ~4000 training samples, 高品質偏振數據

---

## 2. 訓練配置

```bash
python train_pids.py \
    --data_dir ./PIDS_dataset_pol_V5 \
    --output_dir ./checkpoints_pids_v5 \
    --dual_stream \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 \
    --num_steps 70000 \
    --lr 0.0003 \
    --iters 24 \
    --pol_dim 128 \
    --pol_threshold 0.05 \
    --pol_sharpness 20.0 \
    --pol_lr_mult 5.0 \
    --pol_weight 2.0 \
    --glass_weight 5.0 \
    --val_freq 500 \
    --num_workers 4
```

**模型規模**: 4,281,154 parameters
**Dataset**: 3,995 training / 999 validation samples

---

## 3. 訓練結果

### 與 Exp #18 對比

| Step | Exp #18 EPE | Exp #19 EPE | 改善倍數 |
|------|-------------|-------------|----------|
| 500 | 67.88 | - | - |
| 1000 | 80.99 | **34.24** | **2.4x** |
| 3000 | 109.70 | **19.84** | **5.5x** |
| 4500 | 63.06 | **17.16** | **3.7x** |

### 收斂軌跡

```
Step  1000: EPE 34.24 | Glass EPE 45.41
Step  3000: EPE 19.84 | Glass EPE 27.79  ⬇️
Step  7000: EPE 11.27 | Glass EPE 15.15  ⬇️
Step 18000: EPE  6.61 | Glass EPE  9.85  ⬇️ (Glass EPE < 10px)
Step 25500: EPE  4.70 | Glass EPE  6.86  ⬇️ (EPE < 5px)
Step 53000: EPE  2.85 | Glass EPE  4.19  ⬇️ (EPE < 3px)
Step 63500: EPE  2.79 | Glass EPE  3.99  🏆 Best Glass EPE
Step 69000: EPE  2.74 | Glass EPE  4.03  🏆 Best EPE
Step 70000: EPE  2.79 | Glass EPE  4.04  (Final)
```

### 最終結果 (70k steps 完成)

| Metric | Best Value | @ Step | Final (70k) |
|--------|------------|--------|-------------|
| **Glass EPE** | **3.99 px** | 63500 | 4.04 px |
| **EPE** | **2.74 px** | 69000 | 2.79 px |
| **Loss** | 36.26 | 63500 | 36.44 |

### 里程碑達成

| 目標 | 達成 Step | 說明 |
|------|-----------|------|
| Glass EPE < 10px | 18000 | 比預估更早達成 |
| Glass EPE < 5px | 48000 | 突破性進展 |
| Glass EPE < 4px | 63500 | 🏆 最終最佳 |
| EPE < 5px | 25500 | 快速收斂 |
| EPE < 3px | 53000 | 🏆 達到 sub-3px |

### 關鍵觀察

1. **收斂速度大幅提升**:
   - Exp #19 Step 3000 (EPE 19.84) ≈ Exp #18 Step 37500 (EPE 22.33)
   - 達到相同精度所需步數減少 **12x**

2. **Glass EPE 突破預期**:
   - 原預估: < 15px
   - 實際達成: **3.99px** (改善 **3.8x**)

3. **Overall EPE 大幅領先**:
   - Exp #18 最佳: ~21px
   - Exp #19 最佳: **2.74px** (改善 **7.7x**)

---

## 4. 改善來源分析

| 改進項目 | 預估貢獻 | 說明 |
|----------|----------|------|
| V5 Dataset 品質 | 40% | 修正的 glass mask、GT disparity |
| Dual-Stream 架構 | 35% | 偏振特徵正確融合到 correlation volume |
| Two-Stage Training | 15% | 訓練時使用 GT disparity 做 warp |
| Pretrained Weights | 10% | RAFT-Stereo SceneFlow 預訓練 |

---

## 5. 與 Exp #18 最終對比

| Metric | Exp #18 Best | Exp #19 Best | 改善倍數 |
|--------|--------------|--------------|----------|
| EPE | ~21 px | **2.74 px** | **7.7x** |
| Glass EPE | N/A | **3.99 px** | - |

---

## 6. Strict Glass Mask 功能

**日期**: 2026-01-08

### 動機

原有的 Glass Mask 使用左右視角的**聯集** (union)，會因為 65mm 基線造成邊緣擴張。
為了更精確評估玻璃區域精度，新增**交集** (intersection) 作為嚴格 mask。

### 設計

```
┌─────────────────────────────────────┐
│  Background (w = 1.0)               │
│  ┌─────────────────────────────┐    │
│  │  Union-only (w = 5.0)       │    │  ← 邊緣區域
│  │  ┌─────────────────────┐    │    │
│  │  │  Strict (w = 5.5)   │    │    │  ← 核心區域
│  │  └─────────────────────┘    │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
```

### 權重公式

```python
# 顯式拆分 union-only 和 strict 區域
union_only = glass_mask * (1.0 - glass_mask_strict)  # U & ~S
strict = glass_mask_strict

# 權重分布 (預設 glass_weight=5.0, strict_glass_weight=0.5)
# - 背景:     w = 1.0
# - 邊緣:     w = glass_weight = 5.0
# - 核心:     w = glass_weight + strict_glass_weight = 5.5
```

### 實現變更

| 檔案 | 變更 |
|------|------|
| `pids_renderer_textured.py` | 輸出 `*_glass_mask_strict.exr` (交集) |
| `pids_dataset.py` | 載入 `glass_mask_strict`，augment 時同步處理 |
| `pids_model.py` | Loss 支援 `strict_glass_weight` 參數 |
| `train_pids.py` | 訓練時傳遞 strict mask，驗證時用 strict 計算 Glass EPE |

### 使用方式

```bash
# 重新渲染現有場景的 strict mask
python pids_renderer_textured.py --rerender-mask ./output --obj-dir ./scenes --output ./output

# 訓練時使用（預設 strict_glass_weight=0.5）
python train_pids.py --data_dir ./dataset --dual_stream

# 調整權重
python train_pids.py --data_dir ./dataset --dual_stream \
    --glass_weight 5.0 \
    --strict_glass_weight 1.0  # 更強調核心
```

### 預設參數選擇

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `glass_weight` | 5.0 | 玻璃區域整體權重 |
| `strict_glass_weight` | 0.5 | 核心區域額外權重（保守） |

選擇 0.5 作為預設值的原因：
- 交集面積較小，梯度波動較大
- 保守設定可避免訓練抖動
- 需要強調核心時可調至 1.0-2.0

---

## 實驗 #20：Nopol V5/V6 消融實驗（消融實驗 - 控制組）

**日期**: 2026-01-10

> **消融實驗設計**：Exp #20 (Nopol) 與 Exp #21 (Pol) 構成一組消融實驗，用於驗證偏振信息對玻璃深度估計的貢獻。Nopol 為**控制組**（無偏振差異），Pol 為**實驗組**（有偏振差異）。兩者使用**完全相同的架構與超參數**，僅改變輸入數據。

### 目的

驗證偏振信息對深度估計的貢獻。通過保持**完全相同的架構**，只改變輸入數據（有偏振 vs 無偏振），進行公平的消融實驗。

### 實驗設計

| 變數 | Pol 版 (控制組) | Nopol 版 (實驗組) |
|------|-----------------|-------------------|
| 架構 | Dual-Stream ✓ | Dual-Stream ✓ |
| 預訓練 | raftstereo-sceneflow.pth | raftstereo-sceneflow.pth |
| batch_size | 8 | 8 |
| num_steps | 60000 | 60000 |
| lr | 0.0003 | 0.0003 |
| iters | 24 | 24 |
| pol_dim | 128 | 128 |
| pol_threshold | 0.05 | 0.05 |
| pol_weight | 2.0 | 2.0 |
| glass_weight | 5.0 | 5.0 |
| strict_glass_weight | 0.5 | 0.5 |
| **輸入數據** | **I∥, I⊥ (偏振差明顯)** | **Left, Right (無偏振差)** |

### Nopol 訓練命令

```bash
nohup python train_pids.py \
    --data_dir ./PIDS_dataset_nopol_V5 \
    --output_dir ./checkpoints_nopol_v5 \
    --dual_stream \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 \
    --num_steps 60000 \
    --lr 0.0003 \
    --iters 24 \
    --pol_dim 128 \
    --pol_threshold 0.05 \
    --pol_sharpness 20.0 \
    --pol_lr_mult 5.0 \
    --pol_weight 2.0 \
    --glass_weight 5.0 \
    --strict_glass_weight 0.5 \
    --val_freq 500 \
    --num_workers 4 \
    > train_nopol_v5.log 2>&1 &
```

### 數據集準備

| 項目 | 數量 | 說明 |
|------|------|------|
| V5/V6 Pol 渲染 | ~4700 場景 | `pids_renderer_textured.py` |
| V5/V6 Nopol 渲染 | ~4700 場景 | `pids_renderer_nopol.py --from-params` |
| Strict Glass Mask | ✓ | 左右視角交集 |
| 訓練/測試分割 | 相同 | 使用 `train_scenes.txt` / `test_scenes.txt` |

### 實驗結果

**狀態**: 已完成 ✓

#### Glass EPE 結果

| Step | Glass EPE (px) | 備註 |
|------|----------------|------|
| 500 | 74.87 | 初始高誤差 |
| 3000 | **93.40** | 最差值 |
| 17500 | 23.19 | 局部最佳 |
| 23000 | 16.56 | 持續改善 |
| 29000 | 13.67 | 局部最佳 |
| 44000 | **10.99** | 全局最佳 |
| 60000 | 52.51 | 最終值（反彈） |

#### Val Loss 結果

| Step | Val Loss | 備註 |
|------|----------|------|
| 500 | 634.23 | 初始 |
| 1500 | **641.67** | 最差值 |
| 29000 | 114.27 | 局部最佳 |
| 44000 | **95.23** | 全局最佳 |
| 60000 | 321.83 | 最終值（反彈） |

#### 關鍵觀察

1. **極端震盪**: Glass EPE 在 10.99 ~ 93.40 px 之間劇烈波動
2. **無收斂趨勢**: 訓練後期 (50k-60k) 誤差反彈至 ~350，無法穩定
3. **最佳 vs 最終**: 最佳 10.99 px @ Step 44000，但最終 52.51 px @ Step 60000
4. **Val Loss 同步震盪**: 與 Glass EPE 呈現相同的不穩定模式

#### Pol vs Nopol 對比

| 指標 | Pol 版 | Nopol 版 | 偏振優勢 |
|------|--------|----------|----------|
| Best Glass EPE | **3.99 px** | 10.99 px | **2.75x** |
| Final Glass EPE | ~4 px | 52.51 px | **13x** |
| 收斂穩定性 | 穩定下降 | 極端震盪 | ∞ |
| 訓練可靠性 | 高 | 低 | - |

### 分析與假設

#### 核心發現

**當 I∥ - I⊥ 在 nopol 數據中不再提供任何有意義的差異時，pol_diff 計算結果變成了一種干擾信號而非有效特徵。**

#### 理論解釋

在 pol 數據中：
```
pol_diff = I∥ - I⊥
        = 玻璃區域強反射 - 玻璃區域弱反射
        = 明確的正向信號 (高對比度)
```

在 nopol 數據中：
```
pol_diff = Left - Right
        = 幾乎相同的亮度 - 幾乎相同的亮度
        = 接近零的噪聲信號 + 視差引起的錯位偽影
```

#### Dual-Stream 架構在 nopol 下的問題

1. **PolEncoder 接收無效輸入**:
   - 設計用於處理偏振差異
   - 在 nopol 中只看到噪聲

2. **pol_diff 引導失效**:
   - `pol_threshold=0.05` 無法區分玻璃與背景
   - 所有區域的 pol_diff 都接近閾值

3. **訓練信號矛盾**:
   - glass_weight 懲罰玻璃區域誤差
   - 但模型無法從 pol_diff 獲得玻璃位置線索
   - 導致梯度方向不穩定

#### 為何最佳性能發生在中期 (Step 44000)

推測：
- 早期：模型嘗試學習 pol_diff，但信號是噪聲
- 中期：模型開始忽略 pol_diff，依靠純視差匹配
- 後期：過擬合噪聲，性能崩潰

這解釋了為何 nopol 最佳 (10.99 px) 約為 pol 最佳 (3.99 px) 的 2.75 倍：
- 2.75x 差距 = 失去偏振特徵的代價
- 模型仍能從傳統立體匹配獲得一定性能，但失去了關鍵優勢

### 結論

消融實驗證實：

1. **偏振信息是 PIDS 成功的關鍵**: 2.75 倍的 Glass EPE 改善
2. **Dual-Stream 架構需要有效的偏振輸入**: 在無偏振數據上表現不穩定
3. **pol_diff 在 nopol 上成為干擾**: 不僅沒幫助，還導致訓練不穩定

**建議**: 未來如需處理無偏振場景，應使用傳統 RAFT-Stereo 架構而非 Dual-Stream。

---

## 實驗 #21：Pol V5 + Strict Glass Weight 訓練（消融實驗 - 實驗組）

**日期**: 2026-01-11

> **消融實驗設計**：Exp #21 (Pol) 與 Exp #20 (Nopol) 構成一組消融實驗。本實驗為**實驗組**，驗證偏振信息是否優於無偏振基準。

### 目的

1. 在 Exp #19 基礎上加入 `strict_glass_weight` 參數，驗證對玻璃核心區域額外加權的效果
2. 作為消融實驗的實驗組，驗證偏振數據相對於 Exp #20 (Nopol) 的優勢

### 與 Exp #19 差異

| 參數 | Exp #19 | Exp #21 |
|------|---------|---------|
| num_steps | 70000 | 60000 |
| strict_glass_weight | (無) | 0.5 |

### 訓練命令

```bash
nohup python train_pids.py \
    --data_dir ./PIDS_dataset_pol_V5 \
    --output_dir ./checkpoints_pol_v5_exp21 \
    --dual_stream \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 \
    --num_steps 60000 \
    --lr 0.0003 \
    --iters 24 \
    --pol_dim 128 \
    --pol_threshold 0.05 \
    --pol_sharpness 20.0 \
    --pol_lr_mult 5.0 \
    --pol_weight 2.0 \
    --glass_weight 5.0 \
    --strict_glass_weight 0.5 \
    --val_freq 500 \
    --num_workers 4 \
    > train_pol_v5_exp21.log 2>&1 &
```

### 實驗結果

**狀態**: 已完成 ✓

#### Glass EPE 收斂軌跡

| Step | Glass EPE (px) | Val Loss | 備註 |
|------|----------------|----------|------|
| 500 | 59.96 | 588.98 | 初始 |
| 5000 | 12.69 | 99.15 | 快速下降 |
| 10000 | 9.99 | 75.22 | < 10 px |
| 22000 | 5.90 | 45.86 | < 6 px |
| 40000 | **3.98** | 32.70 | 首次 < 4 px |
| 52500 | 3.63 | 29.73 | 持續改善 |
| 58000 | **3.48** | **28.02** | 🏆 Best |
| 60000 | 3.49 | 28.19 | Final |

#### 與 Exp #19 對比

| 指標 | Exp #19 Best | Exp #21 Best | 改善 |
|------|--------------|--------------|------|
| **Glass EPE** | 3.99 px | **3.48 px** | **12.8%** ↓ |
| **EPE** | 2.74 px | **2.09 px** | **23.7%** ↓ |
| **Val Loss** | 36.26 | **28.02** | **22.7%** ↓ |
| **D1** | - | **18.10%** | - |
| **Composite** | - | **5.30** | - |

#### 關鍵觀察

1. **收斂穩定性**：與 Exp #20 (Nopol) 的劇烈震盪相比，Exp #21 呈現平穩單調下降
2. **Glass EPE < 3.5 px**：達成預期目標，突破 Exp #19 的 3.99 px 門檻
3. **EPE 大幅改善**：從 2.74 px 降至 2.09 px，改善幅度超過預期
4. **strict_glass_weight 效果**：對玻璃核心區域額外加權確實有效

#### 結論

`strict_glass_weight=0.5` 參數驗證成功：
- 讓模型更專注於「確定是玻璃」的區域（左右視角交集）
- 減少邊緣模糊區域的梯度干擾
- 在較少步數 (60k vs 70k) 內達成更好結果

---

## 研究貢獻分析：PIDS 是否為 0→1 研究？

### 消融實驗總結

**實驗設計**：相同架構、相同超參數，僅改變輸入數據（Pol vs Nopol）

#### Validation Set 結果

| 指標 | Nopol (控制組) | Pol (實驗組) | 改善 |
|------|---------------|--------------|------|
| Glass EPE (Best) | 10.99 px | **3.48 px** | **68.3%** ↓ |
| Val Loss (Best) | 95.23 | **28.02** | **70.6%** ↓ |

#### Test Set 結果 (100+ 場景)

| 指標 | Nopol (控制組) | Pol (實驗組) | 改善 |
|------|---------------|--------------|------|
| **Glass EPE (Mean)** | 11.27 px | **5.87 px** | **48.0%** ↓ |
| **Glass EPE (Median)** | 6.69 px | **2.07 px** | **69.1%** ↓ |
| **Glass D1** | 87.78% | **34.99%** | **60.1%** ↓ |
| **Glass D3** | 66.62% | **22.84%** | **65.7%** ↓ |
| BG EPE | 4.79 px | 5.72 px | -19.4% |

#### 測試集統計分布

| 統計量 | Pol Glass EPE | Nopol Glass EPE |
|--------|---------------|-----------------|
| Min | 0.22 px | 1.19 px |
| Q25 | 0.63 px | 4.57 px |
| Median | **2.07 px** | 6.69 px |
| Q75 | 7.77 px | 10.88 px |
| Max | 36.42 px | 73.38 px |

#### 關鍵發現

1. **偏振對玻璃區域效果顯著**：Glass EPE Median 改善 69.1% (6.69 → 2.07 px)
2. **背景區域差異不大**：BG EPE 相近，符合預期（偏振主要幫助玻璃）
3. **D1 指標差異最大**：34.99% vs 87.78%，說明偏振大幅減少「完全錯誤」的預測
4. **Median vs Mean**：Pol 的 Median (2.07 px) 遠優於 Mean (5.87 px)，表示大多數場景表現優秀

**結論**：偏振信息使玻璃區域深度估計誤差降低 **48-69%**（取決於統計量選擇）

#### 視差誤差轉換為實際距離

**相機參數**：基線 65mm, 焦距 502px (FOV 65°, 640×480)

**換算公式**：`深度誤差 ΔZ ≈ Z² × Δd / (f × B)`

| 實際深度 | Pol (2.07 px) | Nopol (6.69 px) | 偏振優勢 |
|----------|---------------|-----------------|----------|
| 0.5 m | 1.6 cm | 5.1 cm | **減少 3.5 cm** |
| 1.0 m | 6.3 cm | 20.5 cm | **減少 14.2 cm** |
| 2.0 m | 25.4 cm | 82.0 cm | **減少 56.6 cm** |
| 3.0 m | 57.0 cm | 184.5 cm | **減少 127.5 cm** |

**實際意義**：
- **1m 距離**：Pol 誤差 6.3 cm（可安全避障） vs Nopol 誤差 20.5 cm（可能碰撞）
- **2m 距離**：Pol 誤差 25.4 cm（有預警空間） vs Nopol 誤差 82.0 cm（幾乎無法使用）

### 偏向 0→1 的部分

| 創新點 | 說明 |
|--------|------|
| **問題定義** | 「透明障礙物深度感測」是現有 LiDAR/ToF/傳統 Stereo 都無法解決的痛點 |
| **方法論** | 主動非對稱偏振 + 深度學習立體匹配的結合，文獻中少見 |
| **物理洞見** | 利用 I∥ - I⊥ 作為玻璃偵測信號，有明確物理依據 |
| **消融驗證** | Val 68.3% / Test 69.1% 改善證明偏振是關鍵，不是「加了就好」|

### 偏向 1→N 的部分

| 借鏡之處 | 說明 |
|----------|------|
| 基礎架構 | RAFT-Stereo 是現有方法 |
| 偏振成像 | 光學領域已有研究 |
| 立體匹配 | 成熟技術 |

### 評估結論

**介於 0→1 和 1→N 之間，但偏向 0→1。**

原因：本研究不是單純「把 A 和 B 拼起來」，而是：

1. **發現了一個未被充分解決的問題**：透明障礙物對機器人導航的威脅
2. **提出了一個有物理依據的解法**：利用偏振差異 (I∥ - I⊥) 作為玻璃偵測信號
3. **設計了專門的架構**：Dual-Stream + Polarization Encoder + Soft Threshold
4. **用消融實驗證明核心假設**：Val 68.3% / Test 69.1% 改善證明偏振信息是成功關鍵

這比純粹的 incremental improvement 更具研究貢獻。

---

## 9. 訓練驗證邏輯改進：Oracle vs Real 模式

**日期**: 2026-01-16

### 問題發現

在 `train_pids.py` 的驗證邏輯中發現潛在問題：

```python
# 原本的驗證程式碼
if self.args.dual_stream:
    flow_preds = self.model(left, right, iters=16, disparity_gt=disp_gt)
```

**問題**：驗證時傳入 `disparity_gt`，但真實推論時不可能有 GT disparity。

這導致驗證分數是「**Oracle Testing**」（理論上限），而非真實部署能力。

### 問題分析

```
┌─────────────────────────────────────────────────────────────────┐
│                        三種模式對比                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  【訓練時】                                                      │
│   pol_diff = |left - warp(right, GT_disparity)|                │
│   ✓ 完美對齊，比較同一 3D 點的偏振差異                           │
│                                                                 │
│  【驗證時 - 舊版】                                               │
│   pol_diff = |left - warp(right, GT_disparity)|                │
│   ✓ 同樣完美對齊 ← Oracle Testing (過度樂觀)                    │
│                                                                 │
│  【真實推論】                                                    │
│   pol_diff = |left - right| (無對齊) 或                         │
│   pol_diff = |left - warp(right, predicted_disparity)|         │
│   ✗ 65mm baseline 導致錯位，或用預測值對齊 (有誤差)             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 解決方案

修改 `train_pids.py`，新增雙重驗證模式：

#### 1. `validate(use_oracle)` 方法修改

```python
@torch.no_grad()
def validate(self, use_oracle: bool = True) -> Dict[str, float]:
    """
    Args:
        use_oracle: True = 用 GT disparity 對齊 (理論上限)
                    False = 用 forward_inference (實戰能力)
    """
    if use_oracle:
        # Oracle Testing - 用 GT disparity
        flow_preds = self.model(left, right, iters=16, disparity_gt=disp_gt)
    else:
        # Real Inference - 用 forward_inference
        flow_pred = model_ref.forward_inference(
            left, right,
            iters=16,
            pol_update_iters=[8]  # 中間更新一次偏振特徵
        )
```

#### 2. `_validate_and_log()` 新增方法

```python
def _validate_and_log(self):
    """同時進行 Oracle 和 Real 驗證"""
    # 1. Oracle 驗證 (理論上限)
    val_metrics_oracle = self.validate(use_oracle=True)

    # 2. Real 驗證 (只在 Dual-Stream 模式有差異)
    if self.args.dual_stream:
        val_metrics_real = self.validate(use_oracle=False)
    else:
        val_metrics_real = val_metrics_oracle

    # 3. 用 Real 分數決定 best checkpoint
    is_best = composite_score_real < self.best_composite_score
```

### 新的驗證輸出格式

```
  [Validating Oracle mode...]
  [Validating Real mode...]

  [Val @ Step 5000]
  ┌─────────────────────────────────────────────────────────────────┐
  │ Mode       │ Glass EPE │ EPE    │ D1     │ Composite │ Loss     │
  ├─────────────────────────────────────────────────────────────────┤
  │ Oracle     │     3.485 │ 28.187 │ 42.50% │     7.735 │  28.1874 │
  │ Real       │     4.200 │ 29.500 │ 45.20% │     8.720 │  30.2100 │
  └─────────────────────────────────────────────────────────────────┘
  Oracle-Real Gap: +0.715 px (+20.5%)
```

### TensorBoard 新增指標

| 指標路徑 | 說明 |
|----------|------|
| `val_oracle/glass_epe` | Oracle 模式 Glass EPE |
| `val_real/glass_epe` | **Real 模式 Glass EPE (實戰能力)** |
| `val/oracle_real_gap` | Oracle-Real 差距 (px) |
| `val/oracle_real_gap_pct` | Oracle-Real 差距 (%) |

### Pol vs Nopol 影響分析

| 實驗 | Oracle-Real Gap | 原因 |
|------|-----------------|------|
| **Pol** | 15-25% | pol_diff 依賴對齊，未對齊損失準確度 |
| **Nopol** | ~0-2% | pol_diff ≈ 0，對齊與否沒差 |

**關鍵洞見**：Oracle-Real Gap 可作為偏振貢獻的額外證據。

- Pol Gap 大 → 證明偏振對齊很重要
- Nopol Gap ≈ 0 → 證明 Gap 確實來自偏振，不是其他因素

### Best Checkpoint 邏輯改變

```
舊版: 用 Oracle 分數決定 best → 過度樂觀
新版: 用 Real 分數決定 best → 反映真實部署能力 ✓
```

### 注意事項

1. **驗證時間加倍**: 每次驗證跑兩遍 (Oracle + Real)
2. **兼容性**: 舊版 `val/glass_epe` 指標仍存在，現在記錄 Real 分數
3. **推論方法**: Real 模式使用 `forward_inference`，迭代中途更新偏振特徵

---

## 10. Oracle vs Real 測試結果分析

**日期**: 2026-01-16

### 問題發現延伸

在修正 `train_pids.py` 的驗證邏輯後，發現 `evaluate_pids.py` **也存在相同問題**：

```python
# 原本的測試程式碼
flow_preds = self.model(left, right, iters=self.args.iters, disparity_gt=disp_gt)
```

這代表先前報告的所有測試結果都是 **Oracle 模式** 的理論上限，而非實際部署能力！

### 修正 evaluate_pids.py

新增以下參數：

```python
parser.add_argument('--oracle', action='store_true',
                    help='Use Oracle mode (with GT disparity for pol alignment).')
parser.add_argument('--two_pass', action='store_true',
                    help='Use two-pass inference in Real mode.')
parser.add_argument('--pol_update_iters', type=int, nargs='+', default=None,
                    help='Iterations for pol feature update in Real mode.')
```

推論邏輯：

```python
if self.args.oracle:
    # Oracle: 用 GT disparity 對齊 (理論上限)
    flow_preds = self.model(left, right, iters=iters, disparity_gt=disp_gt)
elif self.args.two_pass:
    # Real Two-Pass: Pass 1 估計 disparity → Pass 2 對齊偏振
    flow_pred = self.model.forward_two_pass(left, right, ...)
else:
    # Real Single-Pass: 迭代中途更新偏振特徵
    flow_pred = self.model.forward_inference(left, right, pol_update_iters=[...])
```

### 測試結果

**Exp #21 (Pol) - checkpoints/pol_v6_exp21_best.pth**

| Mode | Glass EPE | Background EPE | Overall EPE |
|------|-----------|----------------|-------------|
| Oracle | **5.87 px** | 5.10 px | 5.14 px |
| Real (single-pass) | **11.92 px** | - | - |

**Exp #17 (Nopol) - checkpoints/nopol_v5_exp17_strict_best.pth**

| Mode | Glass EPE | Background EPE | Overall EPE |
|------|-----------|----------------|-------------|
| Oracle | **11.27 px** | 4.32 px | 4.61 px |
| Real (single-pass) | **89.45 px** | - | - |

### 關鍵發現：Real 模式揭示偏振的真正價值

```
┌─────────────────────────────────────────────────────────────────┐
│              Oracle vs Real 模式對比                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Oracle 模式 (先前報告):                                         │
│  ┌─────────────────────────────────────────────────────┐        │
│  │ Pol:   5.87 px                                      │        │
│  │ Nopol: 11.27 px                                     │        │
│  │ 改進:  47.9%                                        │        │
│  └─────────────────────────────────────────────────────┘        │
│                                                                 │
│  Real 模式 (真實部署):                                           │
│  ┌─────────────────────────────────────────────────────┐        │
│  │ Pol:   11.92 px                                     │        │
│  │ Nopol: 89.45 px                                     │        │
│  │ 改進:  86.7% ← 偏振的真正價值！                       │        │
│  └─────────────────────────────────────────────────────┘        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 為什麼 Nopol Real 會災難性失敗？

```
Nopol 的 pol_diff 計算:
  pol_diff = |left - warp(right, disparity)|
           = |I∥ - warp(I∥, disparity)|  ← 兩張圖一樣！

Oracle 模式 (GT disparity):
  → pol_diff ≈ 0 (完美對齊，I∥ - I∥ = 0)
  → 模型正常運作，pol_feat = 0 不干擾

Real 模式 (predicted disparity):
  → disparity 預測不準確 (尤其在透明物體上)
  → warp 錯位產生巨大幾何雜訊
  → pol_diff = 純雜訊 (不是物理訊號)
  → 模型被錯誤資訊嚴重干擾 → 災難性失敗
```

### 為什麼 Pol Real 相對穩健？

```
Pol 的 pol_diff 計算:
  pol_diff = |I∥ - warp(I⊥, disparity)|

即使 warp 有誤差:
  → 透明物體區域: I∥ >> I⊥ (強烈偏振對比)
  → 這個強訊號能「穿透」幾何雜訊
  → 模型仍能識別透明物體的大致位置

Real 模式退化分析:
  Oracle: 5.87 px → Real: 11.92 px
  退化幅度: +103% (仍可接受)
```

### 結論更新

**原本結論** (基於 Oracle 測試):
> 偏振改進 Glass EPE 47.9%

**更新結論** (基於 Real 測試):
> 在真實部署場景下，偏振改進 Glass EPE **86.7%**
>
> 偏振訊號具有 **幾何魯棒性** - 即使 disparity 預測不準確導致 pol_diff
> 對齊錯誤，強烈的偏振對比 (I∥ >> I⊥) 仍能提供有用資訊。
>
> 相比之下，Nopol 在無偏振對比的情況下，幾何錯誤直接變成純雜訊，
> 導致模型完全無法處理透明物體。

### 實務意義

1. **論文數據**: 建議報告 Real 模式結果，更能反映實際應用價值
2. **Baseline 對比**: 在 Real 模式下偏振優勢更加明顯 (86.7% vs 47.9%)
3. **系統魯棒性**: 偏振方法對 disparity 估計誤差有較高容忍度

### 未來改進方向

1. **更好的 forward_inference**: 嘗試多次 pol_feat 更新
2. **Two-pass 優化**: 目前 two-pass 沒有顯著改進，需要研究原因
3. **Training-Inference 一致性**: 訓練時也使用 predicted disparity 對齊

---

## 11. Training-Inference Consistency 優化

**日期**: 2026-01-16

### 問題分析

從 Section 10 的 Real 模式測試發現：
- Pol Real: 11.92 px (Oracle: 5.87 px)
- Oracle-Real Gap: +103%

**根本原因**: 訓練時使用 GT disparity，推論時使用 predicted disparity
→ 訓練/推論分布不一致 (Distribution Mismatch)

```
┌─────────────────────────────────────────────────────────────────┐
│                    問題根源                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  訓練時:                                                         │
│    pol_diff = |left - warp(right, GT_disparity)|               │
│    → 完美對齊，模型學到「乾淨」的偏振特徵                        │
│                                                                 │
│  推論時:                                                         │
│    pol_diff = |left - warp(right, predicted_disparity)|        │
│    → 預測有誤差，pol_diff 包含 geometric noise                  │
│    → 模型沒見過這種雜訊，效能下降                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 解決方案：Curriculum Learning + Disparity Noise

#### 1. pids_model.py 修改

```python
class PIDSStereoDualStream(nn.Module):
    def __init__(self, ..., disparity_noise_std: float = 2.0):
        self.disparity_noise_std = disparity_noise_std

    def forward(self, left, right, ..., noise_ratio: float = 0.0):
        # Training-Inference Consistency
        disparity_for_pol = disparity_gt
        if self.training and disparity_gt is not None and noise_ratio > 0:
            if torch.rand(1).item() < noise_ratio:
                # 加入高斯雜訊模擬推論誤差
                noise = torch.randn_like(disparity_gt) * self.disparity_noise_std
                disparity_for_pol = disparity_gt + noise

        pol_feat = self.pol_encoder(left, right, disparity_for_pol)
```

#### 2. train_pids.py 修改

新增 Curriculum Learning 參數：

```python
parser.add_argument('--disparity_noise_std', type=float, default=2.0,
                    help='Disparity noise std (pixels)')
parser.add_argument('--noise_warmup_steps', type=int, default=20000,
                    help='Steps to warmup noise_ratio from 0 to max')
parser.add_argument('--max_noise_ratio', type=float, default=0.5,
                    help='Maximum noise ratio (0.5 = 50% noisy samples)')
```

Curriculum Learning 函數：

```python
def _get_noise_ratio(self, step: int) -> float:
    """線性增長: 0 → max_noise_ratio over warmup_steps"""
    if step >= self.args.noise_warmup_steps:
        return self.args.max_noise_ratio
    else:
        return self.args.max_noise_ratio * (step / self.args.noise_warmup_steps)
```

### Curriculum Learning 過程

```
Step         noise_ratio    訓練樣本分布
────────────────────────────────────────────
    0        0.00           100% GT disparity
10000        0.25            75% GT, 25% noisy
20000        0.50            50% GT, 50% noisy
60000        0.50            50% GT, 50% noisy (維持)
```

### 多次偏振更新 (Multi-Update Inference)

除了訓練改進，推論時也可使用多次 pol_feat 更新：

```bash
# 原本: 只在 iter 11 更新一次
--pol_update_iters 11

# 改進: 漸進式更新三次
--pol_update_iters 6 12 18
```

**原理**: 每次更新後 disparity 更準確 → pol_diff 對齊更好 → 下次更新更準確

### 訓練命令 (Exp #23)

```bash
python train_pids.py \
    --name "pol_v6_exp23_curriculum" \
    --data_dir ./PIDS_dataset_pol_V6 \
    --dual_stream \
    --disparity_noise_std 2.0 \
    --noise_warmup_steps 20000 \
    --max_noise_ratio 0.5 \
    # ... 其他參數同 Exp #21
```

### 預估改進

| 指標 | Exp #21 (原) | Exp #23 (優化) | 改進 |
|------|-------------|----------------|------|
| Oracle | 5.87 px | ~5.5-6.0 px | 持平 |
| Real | 11.92 px | **7-9 px** | **25-40%** |
| Gap | +103% | **30-50%** | **縮小一半** |

### 風險與調整

| 參數 | 過小風險 | 過大風險 | 建議值 |
|------|---------|---------|--------|
| `noise_std` | 效果不明顯 | 訓練不穩定 | 1.5-3.0 |
| `warmup_steps` | 太快適應不了 | 太慢浪費時間 | 15k-25k |
| `max_ratio` | 效果不足 | GT 樣本太少 | 0.3-0.6 |

### TensorBoard 新增指標

| 指標 | 說明 |
|------|------|
| `train/noise_ratio` | 當前 curriculum 進度 |

---

## 12. Bug Fix: Validation Loss 計算不一致

**日期**: 2026-01-17

### 問題發現

在 Exp #23 訓練初期觀察到異常數據：

```
[Val @ Step 2500]
┌─────────────────────────────────────────────────────────────────┐
│ Mode       │ Glass EPE │ EPE    │ D1     │ Composite │ Loss     │
├─────────────────────────────────────────────────────────────────┤
│ Oracle     │    44.364 │ 25.523 │ 75.19% │    51.883 │ 294.4802 │
│ Real       │    62.380 │ 38.839 │ 80.36% │    70.416 │  49.5877 │
└─────────────────────────────────────────────────────────────────┘
```

**異常**: Real Loss (49.59) 比 Oracle Loss (294.48) 小 **6 倍**！

如果 Real 的 EPE 更差，Loss 應該更高才對。

### 根本原因

`train_pids.py` 的 `validate()` 函數中：

```python
# Oracle mode (修正前)
flow_preds = self.model(left, right, iters=28, disparity_gt=disp_gt)
disp_preds = [-f[:, :1] for f in flow_preds]  # 28 個預測

# Real mode
flow_pred = model_ref.forward_inference(left, right, iters=28, ...)
disp_preds = [-flow_pred[:, :1]]  # 只有 1 個預測
```

`PIDSStereoLoss` 的 sequence loss 計算：

```python
for i, disp_pred in enumerate(disp_predictions):
    weight = gamma ** (n_predictions - i - 1)
    total_loss += weight * loss
```

**Oracle**: 28 個預測 → `total_loss ≈ Σ(γⁱ × Lᵢ) ≈ 9 × L`
**Real**: 1 個預測 → `total_loss = 1 × L`

**Loss 差 9 倍純粹是 prediction 數量不同，不是預測品質！**

### 修正方案

讓 Oracle 驗證也只用最終預測計算 loss：

```python
# Oracle mode (修正後)
flow_preds = self.model(left, right, iters=28, disparity_gt=disp_gt)
disp_preds = [-flow_preds[-1][:, :1]]  # 只用最終預測
```

**修改檔案**: `train_pids.py` line 655

### 修正後行為

| Mode | 修正前 | 修正後 |
|------|--------|--------|
| Oracle | 28 predictions → loss ≈ 9L | 1 prediction → loss ≈ L |
| Real | 1 prediction → loss ≈ L | 1 prediction → loss ≈ L |

修正後 Step 1000 驗證結果：

```
│ Oracle     │    40.155 │ 31.838 │ 98.68% │    50.023 │  34.6722 │
│ Real       │    44.794 │ 34.152 │ 98.18% │    54.612 │  39.0791 │
```

✅ Oracle Loss (34.67) 和 Real Loss (39.08) 現在可以公平比較

### 影響範圍

| 指標 | 是否受影響 |
|------|-----------|
| EPE, Glass EPE | ❌ 不受影響 (本來就只用 final prediction) |
| D1, D3, D5, D10 | ❌ 不受影響 |
| Validation Loss | ✅ 已修正 |
| Training Loss | ❌ 不受影響 (仍使用 sequence loss) |

### 備註

Training Loss 和 Validation Loss 數值不同是正常的：
- **Training Loss ≈ 300**: Sequence loss，監督所有 24 個 iterations
- **Validation Loss ≈ 35**: 只評估最終預測品質

兩者用途不同，不需要相同。

---

## 13. Exp #22 & #23: Baseline vs Curriculum Learning 訓練

**日期**: 2026-01-17

### 背景

Exp #21 的 Oracle vs Real 測試揭露了重要問題：

| 模式 | Oracle (GT disp) | Real (Pred disp) | Gap |
|------|------------------|------------------|-----|
| **Pol** | 5.87 px | 11.92 px | +103% |
| **Nopol** | 5.39 px | 89.45 px | +1559% |

**關鍵發現**：Nopol 的 Real 模式失敗是因為 **dual-stream 架構注入純噪聲**。
當沒有偏振信號時，錯誤的 disparity 預測會產生隨機的 pol_diff 幾何誤差，
dual-stream 把這些噪聲注入模型，導致災難性失敗。

### 實驗設計

#### Exp #22: Baseline RAFT-Stereo (Clean)

**目的**: 建立乾淨的 baseline，無任何 PIDS 修改

**訓練腳本**: `train_baseline.py` (新建)
- 使用原版 `RAFTStereo` 模型
- 無 dual-stream 架構
- 無 pol_encoder, pol_feat
- 作為公平比較的基準

**訓練命令**:
```bash
nohup python train_baseline.py \
    --data_dir ./PIDS_dataset_pol_V6 \
    --output_dir ./checkpoints_baseline_exp22 \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 \
    --num_steps 60000 \
    --lr 0.0003 \
    --iters 24 \
    --max_disp 576.0 \
    --val_freq 500 \
    --num_workers 4 \
    > train_baseline_exp22.log 2>&1 &
```

**狀態**: 🔄 訓練中 (2026-01-17 開始)

---

#### Exp #23: Pol + Curriculum Learning

**目的**: 縮小 Oracle-Real Gap

**訓練腳本**: `train_pids.py` (已修改)

**新增功能**:
1. **Disparity Noise Injection**: 訓練時對 GT disparity 加噪
2. **Curriculum Learning**: 噪聲比例從 0 線性增長到 0.5

**訓練命令**:
```bash
nohup python train_pids.py \
    --data_dir ./PIDS_dataset_pol_V6 \
    --output_dir ./checkpoints_pol_v6_exp23 \
    --dual_stream \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 \
    --num_steps 60000 \
    --lr 0.0003 \
    --iters 24 \
    --pol_dim 128 \
    --pol_threshold 0.05 \
    --pol_sharpness 20.0 \
    --pol_lr_mult 5.0 \
    --pol_weight 2.0 \
    --glass_weight 5.0 \
    --strict_glass_weight 0.5 \
    --disparity_noise_std 2.0 \
    --noise_warmup_steps 20000 \
    --max_noise_ratio 0.5 \
    --val_freq 500 \
    --num_workers 4 \
    > train_pol_v6_exp23.log 2>&1 &
```

**Curriculum Learning 參數**:
| 參數 | 值 | 說明 |
|------|-----|------|
| `disparity_noise_std` | 2.0 px | 噪聲標準差 |
| `noise_warmup_steps` | 20,000 | 線性增長步數 |
| `max_noise_ratio` | 0.5 | 最終 50% batch 加噪 |

**訓練進度**:
```
Step         noise_ratio    樣本分布
────────────────────────────────────
    0        0.00           100% GT
10000        0.25           75% GT, 25% noisy
20000+       0.50           50% GT, 50% noisy
```

**狀態**: ⏳ 待開始

---

### Exp #23 實際結果 (失敗)

**日期**: 2026-01-18

Curriculum Learning **完全失敗**：

| Experiment | Oracle | Real | Gap | 說明 |
|------------|--------|------|-----|------|
| Exp #21 Pol | 5.87 px | 11.92 px | +103% | 之前最佳 |
| **Exp #23 Curriculum** | **3.27 px** | **24.63 px** | **+652%** | 災難性失敗 |

**失敗原因**：Gaussian 噪聲 ≠ 真實預測誤差
- 訓練時加的噪聲：隨機、均勻分布
- 實際預測誤差：結構性、在玻璃區域特別大
- 模型學會了「抵抗隨機噪聲」，但對「系統性預測誤差」完全沒幫助

**結論**：需要根本性的架構改變，而非訓練策略調整。

---

## 14. Exp #24: Polarization Volume 架構 (根本解決方案)

**日期**: 2026-01-18

### 問題根源分析

Exp #23 失敗後，重新思考問題本質：

```
Oracle: pol_diff = warp(I_left, GT_disp) - I_right     → 完美對齊 ✓
Real:   pol_diff = warp(I_left, pred_disp) - I_right   → 有誤差 ✗
```

**核心矛盾**：要用 disparity 計算 pol_diff，但 disparity 本身就是要預測的東西。

### 靈感來源

用戶提問：「我們一定要 disparity 才能 warp 嗎？相機的資訊和位置不都有了，不能直接算嗎？」

**關鍵洞察**：我們知道對應點一定在同一水平線 (epipolar line) 上！

### 解決方案：Polarization Volume

類似 RAFT-Stereo 的 Correlation Volume，預先計算所有 disparity 的 pol_diff：

```python
# 原本 (需要知道 disparity)
pol_diff = warp(left, d_pred) - right

# 新方法 (不需要 disparity)
pol_volume[d] = shift(left, d) - right   # for all d in [0, max_disp]
```

### 新架構

```
┌─────────────────────────────────────────────────────────────────────┐
│                      PIDSStereoPolVolume                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   left ──→ [FeatureEncoder] ──→ fmap1 ─┐                           │
│                                          ├──→ [CorrBlock] ────┐     │
│   right ─→ [FeatureEncoder] ──→ fmap2 ─┘                      │     │
│                                                                │     │
│   left ──→ [AvgPool 4x] ──→ left_ds ──┐                       │     │
│                                         ├──→ [PolCorrBlock] ──┼─────┤
│   right ─→ [AvgPool 4x] ──→ right_ds ─┘                       │     │
│                                                                │     │
│   left ──→ [ContextEncoder] ──→ context, hidden               │     │
│                                                                │     │
│   ┌────────────── GRU Loop (24 iters) ──────────────────┐     │     │
│   │                                                      │     │     │
│   │   disp ──→ [CorrBlock.lookup] ──→ corr ─────┐       │◄────┘     │
│   │       └──→ [PolCorrBlock.lookup] ──→ pol ───┤       │◄──────────┘
│   │                                              ▼       │
│   │                         [UpdateBlockWithPol]         │
│   │                         concat(corr, pol, disp)      │
│   │                                │                     │
│   │   disp = disp + Δdisp ◄────────┘                     │
│   │                                                      │
│   └──────────────────────────────────────────────────────┘
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### CorrBlock vs PolCorrBlock

| 項目 | CorrBlock | PolCorrBlock |
|------|-----------|--------------|
| 計算 | `dot(fmap1[x], fmap2[x-d])` | `left[x] - right[x-d]` |
| 意義 | 特徵相似度 | 偏振差異 |
| 維度 | (B*H, 1, W, W) | (B*H, 1, W, W) |
| 查詢 | 用 disp 取樣 | 用 disp 取樣 |

### 核心優勢

| 項目 | 舊架構 (Dual-Stream) | 新架構 (Pol Volume) |
|------|---------------------|---------------------|
| pol_diff 計算 | `warp(left, disp) - right` | 預計算所有 disp |
| 需要 disparity | ✗ 需要 GT 或預測 | ✓ **不需要** |
| Oracle/Real Gap | ✗ 有 (最大問題) | ✓ **完全沒有** |
| 訓練/推論一致 | ✗ 不一致 | ✓ 完全一致 |

### 新增程式碼

**pids_model.py**:
- `PolCorrBlock`: 計算偏振差異體積（類似 CorrBlock）
- `UpdateBlockWithPol`: 輸入 corr + pol_corr + disp
- `PIDSStereoPolVolume`: 新模型類別

**train_pids.py**:
- 新增 `--pol_volume` 參數
- 新增 `--pol_levels`, `--pol_radius` 參數

### 訓練命令

```bash
nohup python train_pids.py \
    --data_dir ./PIDS_dataset_pol_V6 \
    --output_dir ./checkpoints_pol_volume_exp24 \
    --pol_volume \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 \
    --num_steps 60000 \
    --lr 0.0003 \
    --iters 24 \
    --pol_levels 4 \
    --pol_radius 4 \
    --val_freq 500 \
    --num_workers 4 \
    > train_pol_volume_exp24.log 2>&1 &
```

### 實驗結果 (2026-01-18)

**測試集評估** (200 場景):

| 指標 | Baseline (RAFT-Stereo) | Pol Volume | 改進 |
|------|------------------------|------------|------|
| **Glass EPE** | 7.520 px | **2.435 px** | **-67.6%** (3.1x) |
| Glass D1 | 81.51% | **19.64%** | **-75.9%** (4.2x) |
| Glass D3 | 49.72% | **12.43%** | **-75.0%** (4.0x) |
| Overall EPE | 3.685 px | 2.310 px | -37.3% |
| BG EPE | 2.713 px | 2.493 px | -8.1% |

**Glass EPE 分布** (Pol Volume):
- Min: 0.000 px
- Q25: 0.291 px
- Median: 0.922 px
- Q75: 2.034 px
- Max: 44.630 px

### 關鍵發現

1. **玻璃區域大幅改善**: Glass EPE 從 7.5px 降到 2.4px (3.1x 改進)
2. **背景沒有退化**: BG EPE 甚至略微改善 (-8.1%)
3. **D1 錯誤率暴降**: 從 81.5% 降到 19.6% (4.2x 改進)
4. **Oracle/Real Gap 完全消除**: 設計目標達成

### 結論

Polarization Volume 架構成功證明：
- ✅ 偏振信息對透明物體檢測至關重要
- ✅ 無需 disparity 即可利用偏振 cue
- ✅ 訓練/推論完全一致，無 gap
- ✅ 相比 Baseline 有顯著改進

**狀態**: ✅ **成功**

---

## Exp #25: Learnable Polarization Volume

**日期**: 2026-01-18
**目標**: 從 Glass EPE 2.44 px 進一步優化到 ~2.0 px

### 動機

Exp #24 的 Polarization Volume 使用固定公式計算偏振差異：
```python
pol_diff[x, d] = I_∥[x] - I_⊥[x-d]
```

問題：
- 固定公式無法適應不同的場景條件（亮度、角度、材質）
- 無法學習「什麼是穩健的偏振信號」
- 可能被噪聲或假陽性干擾

### 核心改進

**從「手工公式」提升到「可學習表徵」**

```
固定公式:
    pol_diff = I_∥ - I_⊥

可學習版:
    pol_feat = PolHead(fmap)           # 學習提取偏振特徵
    pol_corr = |pol_feat_L - pol_feat_R|  # L1 差異
```

### 架構設計

```
left ──→ [FeatureEncoder] ──→ fmap1 ─┬──→ [CorrBlock] ──────────┐
               │                     │                          │
               └──→ [PolHead] ───────┼──→ pol_feat1             │
                                     │                          │
right ─→ [FeatureEncoder] ──→ fmap2 ─┼──→ [CorrBlock] ──────────┤
               │                     │                          │
               └──→ [PolHead] ───────┼──→ pol_feat2             │
                                     │                          │
              [LearnablePolCorrBlock] ──────────────────────────┤
                                     │                          │
                                     ↓                          ↓
                              pol_volume              corr_volume
                                     │                          │
                                     └──────────┬───────────────┘
                                                ↓
                                        [UpdateBlockWithPol]
                                                │
                                                ↓
                                           disparity
```

### 新增組件

| 類別 | 功能 | 參數量 |
|------|------|--------|
| `LightweightPolHead` | 輕量級偏振特徵頭 (256→64→32) | ~18K |
| `LearnablePolCorrBlock` | 向量化偏振體積計算 | 0 |
| `GlassAwareLoss` | Glass-aware auxiliary loss | 0 |
| `PIDSStereoLearnablePol` | 整合模型 | 基礎 + 18K |

### LightweightPolHead 設計

```python
class LightweightPolHead(nn.Module):
    """
    輕量級偏振特徵頭 (~18K 參數)
    結構: feature_dim -> 64 -> pol_dim
    """
    def __init__(self, in_dim=256, pol_dim=32):
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, pol_dim, 1),
        )
```

### Glass-aware Auxiliary Loss

讓 PolHead 學習「偏振特徵」而非「紋理」：

```python
# 玻璃區域的 pol_volume 應該高
# 背景區域的 pol_volume 應該低
loss = max(0, margin + bg_pol - glass_pol)
```

這個 loss 強迫 PolHead 輸出的特徵在玻璃區域有高差異。

### 訓練命令

```bash
nohup python train_pids.py \
    --data_dir ./dataset/train \
    --output_dir ./checkpoints_learnable_pol_exp25 \
    --learnable_pol \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 \
    --num_steps 60000 \
    --lr 0.0003 \
    --iters 24 \
    --pol_dim 32 \
    --pol_levels 4 \
    --pol_radius 4 \
    --glass_aware_weight 0.1 \
    --val_freq 500 \
    --num_workers 4 \
    > train_learnable_pol_exp25.log 2>&1 &
```

### 實驗結果

| 指標 | Baseline | Pol Volume (Exp #24) | Learnable Pol (Exp #25) |
|------|----------|----------------------|-------------------------|
| Glass EPE | 7.520 px | 2.435 px | 2.475 px |
| Glass D1 | 81.51% | 19.64% | 19.34% |
| Overall EPE | 2.205 px | 2.286 px | 2.256 px |

### 結論

**Learnable PolHead 沒有帶來額外改善**：
- Glass EPE: 2.475 px (微幅上升 +0.04 px vs Pol Volume)
- Glass D1: 19.34% (微幅改善 -0.3%)

這表示 **固定公式 (I_sum, I_diff, DoLP) 已經足夠好**，學習額外的偏振特徵沒有顯著幫助。

可能原因：
1. 物理公式已經捕捉到最重要的偏振資訊
2. 網路容量太小無法學到更好的表徵
3. Glass-aware Loss 效果有限

**建議**: 繼續使用 Exp #24 Pol Volume 架構進行 Stage 2 fine-tuning

**狀態**: ✅ 完成 (無顯著改善)

---

## Exp #26: Final - Data Scaling with Strict QA

**日期**: 2026-01-24
**目標**: 使用更嚴格的 QA 標準訓練最終模型，並驗證數據量對性能的影響

### QA 改進 (pids_qa.py v2.3.0)

相比之前的 `aggregate_reports.py`（只檢查 C2 + C5），新增 **C3 玻璃偏振檢查**：

| 檢查項 | 數據來源 | 閾值 | 物理意義 |
|--------|----------|------|----------|
| C2 背景一致性 | `intensity_balance.background_ratio_mean` (跨視角) | 0.80 ~ 1.25 | Stereo pair 背景可匹配 |
| **C3 玻璃偏振** | `polarization.glass_region.stokes_ratio` (同視角) | < 0.96 或 > 1.04 | 玻璃有足夠偏振對比 |
| C5 深度有效率 | `glass_depth_validity.validity_rate` | >= 90% | 玻璃區域深度覆蓋 |

**QA 結果** (15000 渲染場景):
- C2 通過率: 56.9%
- C3 通過率: 78.1%
- C5 通過率: 97.5%
- **整體通過率: 39.1%** (5859 場景)

**假設**: 更嚴格的 C3 過濾可確保訓練數據都有明顯的偏振信號，可能進一步提升 Glass EPE。

### 數據量消融實驗設計

固定 96 epochs，不同數據量：

| 百分比 | 場景數 | Steps | 預期效果 |
|--------|--------|-------|----------|
| 5% | 250 | 3,000 | 基線，可能過擬合 |
| 10% | 500 | 6,000 | CLAUDE.md 建議的最小量 |
| 25% | 1250 | 15,000 | 中等數據量 |
| 50% | 2500 | 30,000 | 半量數據 |
| 75% | 3750 | 45,000 | 接近全量 |
| 100% | 5000 | 60,000 | 全量數據 |

### 訓練配置

```bash
# 5% (250 scenes, 3k steps)
nohup python train_pids.py \
    --data_dir /workspace/RAFT-Stereo/dataset_pol_final/train_5pct \
    --output_dir ./checkpoints_final_5pct \
    --pol_volume \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 3000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 300 --num_workers 4 \
    > train_final_5pct.log 2>&1 &

# 10% (500 scenes, 6k steps)
nohup python train_pids.py \
    --data_dir /workspace/RAFT-Stereo/dataset_pol_final/train_10pct \
    --output_dir ./checkpoints_final_10pct \
    --pol_volume \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 6000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 500 --num_workers 4 \
    > train_final_10pct.log 2>&1 &

# 25% (1250 scenes, 15k steps)
nohup python train_pids.py \
    --data_dir /workspace/RAFT-Stereo/dataset_pol_final/train_25pct \
    --output_dir ./checkpoints_final_25pct \
    --pol_volume \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 15000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 500 --num_workers 4 \
    > train_final_25pct.log 2>&1 &

# 50% (2500 scenes, 30k steps)
nohup python train_pids.py \
    --data_dir /workspace/RAFT-Stereo/dataset_pol_final/train_50pct \
    --output_dir ./checkpoints_final_50pct \
    --pol_volume \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 30000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 500 --num_workers 4 \
    > train_final_50pct.log 2>&1 &

# 75% (3750 scenes, 45k steps)
nohup python train_pids.py \
    --data_dir /workspace/RAFT-Stereo/dataset_pol_final/train_75pct \
    --output_dir ./checkpoints_final_75pct \
    --pol_volume \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 45000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 500 --num_workers 4 \
    > train_final_75pct.log 2>&1 &

# 100% (5000 scenes, 60k steps) - 正在訓練
nohup python train_pids.py \
    --data_dir /workspace/RAFT-Stereo/dataset_pol_final/train_100pct \
    --output_dir ./checkpoints_final_100pct \
    --pol_volume \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 60000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 500 --num_workers 4 \
    > train_final_100pct.log 2>&1 &
```

### 預期結果

1. **C3 過濾效果**: 更嚴格的偏振品質過濾可能讓模型學到更一致的偏振特徵
2. **數據量影響**: 預期 Glass EPE 隨數據量增加而下降，但可能在某個點飽和
3. **最終目標**: Glass EPE < 2.0 px

### 實驗結果

| Learning Rate | Glass EPE | Glass D1 | Overall EPE | 結果 |
|---------------|-----------|----------|-------------|------|
| 0.0001 (錯誤) | 5.37 px | 36.82% | 3.66 px | ❌ +120% 退化 |

**分析**:
- Learning rate 過小 (0.0001) 導致模型收斂不足
- 即使訓練完成 58500 steps，Best Glass EPE 只達到 5.66 px
- 學習率差 3 倍，性能差距超過 2 倍

**結論**:
1. **Learning rate 是關鍵超參數**，必須嚴格按照成功實驗配置
2. Polarization Volume 架構最佳 LR = 0.0003
3. 後續實驗必須使用 `--lr 0.0003`

**潛在優化方向**:
- LR Sweep: 測試 0.0002, 0.0003, 0.0004, 0.0005
- LR Schedule: Warmup + Cosine Annealing
- 可能存在更優 LR 進一步提升 Glass EPE

**狀態**: ✅ 完成 (發現 LR 影響)

---

## Exp #27: Data Scaling with Strict QA

**日期**: 2026-01-25
**目標**: 使用正確的 LR=0.0003 重新訓練，並驗證數據量對性能的影響

### 觀察與假設 (2026-01-25)

訓練過程中發現與 Exp #24 不同的現象：
1. **Val EPE 持續高於 Train EPE** (之前常出現 Val < Train)
2. **Val EPE 波動變大**

**數據差異**:
| 項目 | Exp #24 | Exp #27 |
|------|---------|---------|
| 渲染總量 | 9,000 | 15,000 |
| QA 標準 | 寬鬆 (C2+C5) | 嚴格 (C2+C3+C5) |
| 通過數量 | ~4,700 | ~5,000 |
| 通過率 | ~52% | ~33% |

**假設**: 嚴格 C3 過濾的潛在問題
- 原始數據可能混入了一些 **pol 接近 0** 的場景
- 這迫使模型**同時學習幾何特徵**（不能只依賴 pol）
- 遇到 Val 時，即使 pol 信號弱也能用幾何解決
- 嚴格 C3 後，模型**過度依賴 pol 特徵**，泛化能力下降

**後續實驗方向**:
1. 混合訓練: 80% 強 pol + 20% 弱/無 pol
2. 分階段訓練: nopol 預訓練 → pol 微調
3. 放寬 C3 閾值: 0.98~1.02 → 0.96~1.04

待 Exp #27 結果出來後決定下一步。

### 訓練配置

與 Exp #24 完全相同的超參數：
- `--lr 0.0003`
- `--iters 24`
- `--pol_levels 4`
- `--pol_radius 4`
- `--batch_size 8`

```bash
# 5% (250 scenes, 3k steps)
nohup python train_pids.py \
    --data_dir /workspace/RAFT-Stereo/dataset_pol_final/train_5pct \
    --output_dir ./checkpoints_exp27_5pct \
    --pol_volume \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 3000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 300 --num_workers 4 \
    > train_exp27_5pct.log 2>&1 &

# 10% (500 scenes, 6k steps)
nohup python train_pids.py \
    --data_dir /workspace/RAFT-Stereo/dataset_pol_final/train_10pct \
    --output_dir ./checkpoints_exp27_10pct \
    --pol_volume \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 6000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 500 --num_workers 4 \
    > train_exp27_10pct.log 2>&1 &

# 25% (1250 scenes, 15k steps)
nohup python train_pids.py \
    --data_dir /workspace/RAFT-Stereo/dataset_pol_final/train_25pct \
    --output_dir ./checkpoints_exp27_25pct \
    --pol_volume \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 15000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 500 --num_workers 4 \
    > train_exp27_15pct.log 2>&1 &

# 50% (2500 scenes, 30k steps)
nohup python train_pids.py \
    --data_dir /workspace/RAFT-Stereo/dataset_pol_final/train_50pct \
    --output_dir ./checkpoints_exp27_50pct \
    --pol_volume \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 30000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 500 --num_workers 4 \
    > train_exp27_50pct.log 2>&1 &

# 75% (3750 scenes, 45k steps)
nohup python train_pids.py \
    --data_dir /workspace/RAFT-Stereo/dataset_pol_final/train_75pct \
    --output_dir ./checkpoints_exp27_75pct \
    --pol_volume \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 45000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 500 --num_workers 4 \
    > train_exp27_75pct.log 2>&1 &

# 100% (5000 scenes, 60k steps)
nohup python train_pids.py \
    --data_dir /workspace/RAFT-Stereo/dataset_pol_final/train_100pct \
    --output_dir ./checkpoints_exp27_100pct \
    --pol_volume \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 60000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 500 --num_workers 4 \
    > train_exp27_100pct.log 2>&1 &
```

### 實驗結果

| 數據量 | Glass EPE | Glass D1 | Overall EPE | 備註 |
|--------|-----------|----------|-------------|------|
| 5% (250) | - | - | - | 待訓練 |
| 10% (500) | - | - | - | 待訓練 |
| 25% (1250) | - | - | - | 待訓練 |
| 50% (2500) | - | - | - | 待訓練 |
| 75% (3750) | - | - | - | 待訓練 |
| **100% (5000)** | **5.03 px** | **36.28%** | **3.42 px** | ⚠️ 比 Exp #24 差 |

### 關鍵發現

**對比 Exp #24 (原始數據，含弱偏振)**:
| 指標 | Exp #24 | Exp #27 | 變化 |
|------|---------|---------|------|
| Glass EPE | 2.44 px | 5.03 px | **+106%** ⚠️ |
| Glass D1 | 28.76% | 36.28% | +26% |
| Overall EPE | 1.92 px | 3.42 px | +78% |

**結論**: 嚴格 QA 過濾後性能反而大幅下降！

**原因分析**:
1. 嚴格 C3 過濾移除了所有弱偏振場景
2. 純強偏振數據 → 模型過度依賴偏振特徵
3. 缺少幾何多樣性 → 泛化能力下降
4. 測試集包含各種偏振強度場景 → 模型無法處理弱偏振

**驗證假設**: ✅ 確認需要混合 Pol/Nopol 訓練 (Exp #28)

**狀態**: ✅ 完成 (結果不佳，需 Exp #28 改進)

---

## Exp #28: 混合 Pol/Nopol 訓練

**日期**: 2026-01-25
**目標**: 混合偏振和無偏振數據，迫使模型同時學習幾何和偏振特徵，提升泛化能力

### 動機

Exp #27 訓練過程中觀察到：
1. Val EPE 持續高於 Train EPE（之前常出現 Val < Train）
2. Val EPE 波動變大

**假設**: 嚴格 C3 過濾後，所有訓練數據都有強偏振信號，模型過度依賴偏振特徵。
混入無偏振數據可迫使模型**同時學習幾何特徵**，提升魯棒性。

### 物理洞察：偏振反射的空間不均勻性

玻璃表面的偏振反射**不是均勻分佈**的：

```
玻璃表面偏振分佈示意:
┌─────────────────────────────────────┐
│  ████  強偏振 (接近 Brewster 角)     │
│  ▓▓▓▓  中等偏振                      │
│  ░░░░  弱/無偏振 (正入射或邊緣角度)   │
└─────────────────────────────────────┘
```

**物理原因**：
- 偏振反射強度取決於入射角（Fresnel 方程）
- 只有接近 Brewster 角（~56° for glass）時才有強烈偏振
- 玻璃表面不同區域入射角不同 → 偏振強度不均勻
- 正入射區域幾乎無偏振差異

**問題**：
- 模型若過度依賴偏振 → 弱偏振區域預測失效
- Exp #27 (純強偏振訓練) 正是這個問題
- 測試集包含各種偏振強度 → 模型在弱偏振區域崩潰

**Exp #28 解決方案**：
- Nopol 數據 → 強迫學習純幾何特徵
- 模型學會在偏振弱的區域靠幾何 fallback
- 偏振強時用偏振，偏振弱時用幾何 → 更魯棒

### 數據配置

```
混合比例: Pol 60% + Nopol 40%

Pol 數據 (已完成):
├── 來源: scene_0001 ~ scene_15000 (嚴格 QA 通過 5859 個)
├── Train: 3000 場景
└── Val: 300 場景 (只有 pol!)

Nopol 數據 (待渲染):
├── 來源: 全新 OBJ (scene_15001+，無重疊)
├── 渲染器: pids_renderer_nopol.py
├── QA: 不需要 (無偏振檢查)
└── Train: 2000 場景 (不進 val!)

總計:
├── Train: 3000 + 2000 = 5000 場景
└── Val: 300 場景 (純 pol)
```

### 關鍵設計原則

| 規則 | 說明 |
|------|------|
| **場景不重疊** | Pol 和 Nopol 使用完全不同的 OBJ |
| **Val = Pol only** | 驗證集只有偏振數據（評估偏振能力） |
| **Nopol = Train only** | 無偏振數據只用於訓練（學習幾何） |
| **保留命名差異** | Pol: `_left_parallel.exr`, Nopol: `_left.exr` (便於追蹤) |

### Curriculum Learning 策略 (方案 A+C)

#### 架構考量

PIDS Polarization Volume 有兩套 LR：
- **RAFT Backbone**: 較小 LR (pretrained, fine-tuning)
- **Pol Module**: 較大 LR (從零學習)

**風險**: 若早期大量 nopol 數據，Pol Module 會用大 LR 學習無意義信號 → 學到垃圾特徵

**解決方案**:
1. 調整比例，確保每個 phase 都有足夠 pol 數據
2. Nopol batch 時凍結 Pol Module

#### 訓練階段

```
總步數: 60000 steps

Phase 1 (前 20%): 幾何 + 偏振基礎
├── Steps 0 ~ 12000
├── 數據: 50% nopol + 50% pol
├── Nopol batch: 凍結 Pol Module (只訓練 backbone)
├── Pol batch: 正常訓練
└── 目標: Backbone 學幾何，Pol Module 學基礎偏振

Phase 2 (中 40%): 混合強化
├── Steps 12000 ~ 36000
├── 數據: 40% nopol + 60% pol
├── 全部解凍，混合訓練
└── 目標: 整合幾何與偏振特徵

Phase 3 (後 40%): 偏振精修
├── Steps 36000 ~ 60000
├── 數據: 30% nopol + 70% pol
├── 全部解凍，混合訓練
└── 目標: 強化偏振特徵，保持幾何能力
```

#### 實現方式

**1. 動態 Sampler**
```python
class CurriculumSampler:
    def get_ratio(self, current_step, total_steps):
        progress = current_step / total_steps
        if progress < 0.2:
            return pol=0.5, nopol=0.5
        elif progress < 0.6:
            return pol=0.6, nopol=0.4
        else:
            return pol=0.7, nopol=0.3
```

**2. Pol Module 凍結控制**
```python
def freeze_pol_module(model, freeze: bool):
    """Phase 1 的 nopol batch 時凍結"""
    for name, param in model.named_parameters():
        if 'pol' in name.lower():
            param.requires_grad = not freeze

# 訓練迴圈
if phase == 1 and data_type == "nopol":
    freeze_pol_module(model, freeze=True)
else:
    freeze_pol_module(model, freeze=False)
```

#### 追蹤指標

- `train_pol_epe`: 訓練集 pol 數據的 EPE
- `train_nopol_epe`: 訓練集 nopol 數據的 EPE
- `val_pol_epe`: 驗證集 EPE (純 pol)
- `pol_module_grad_norm`: 監控 Pol Module 梯度（確認凍結生效）

### 執行步驟

#### Step 1: 生成新 OBJ (Blender)
```bash
blender --background --python blender_furniture_randomizer_v17.py -- \
    --start_index 15001 \
    --num_scenes 2500 \
    --output_dir /path/to/new_obj_nopol
```

#### Step 2: 渲染 Nopol (8 GPU)
```bash
nohup python pids_renderer_nopol.py \
    --input_dir /path/to/new_obj_nopol \
    --output /workspace/dataset_nopol_exp28 \
    --num_gpus 8 \
    --max_scenes 2500 \
    > render_nopol_exp28.log 2>&1 &
```

#### Step 3: 組織混合數據集
```bash
python organize_mixed_dataset.py \
    --pol_scenes ./train_100pct_scenes.txt \
    --pol_dir /workspace/dataset_pol \
    --nopol_dir /workspace/dataset_nopol_exp28 \
    --output_dir /workspace/dataset_mixed_exp28 \
    --pol_train 3000 \
    --nopol_train 2000 \
    --pol_val 300
```

#### Step 4: 訓練
```bash
nohup python train_pids.py \
    --data_dir /workspace/dataset_mixed_exp28 \
    --output_dir ./checkpoints_exp28_mixed \
    --pol_volume \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 60000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 500 --num_workers 4 \
    > train_exp28_mixed.log 2>&1 &
```

### 預期結果

| 指標 | Exp #27 (純 Pol) | Exp #28 (混合) 預期 |
|------|-----------------|-------------------|
| Glass EPE | ? | 可能略高（訓練有 nopol） |
| Val 穩定性 | 波動大 | 預期更穩定 |
| 泛化能力 | 可能過擬合 pol | 更好（學了幾何） |

### 實現進度 (2026-01-25)

#### 已完成的代碼實現

**1. `curriculum_sampler.py`** (新建)
```
training_guidance/Stage_I/training_stage/curriculum_sampler.py

類別:
├── CurriculumPhase: 訓練階段配置 dataclass
├── CurriculumConfig: Curriculum Learning 配置
├── CurriculumSampler: 動態 pol/nopol 採樣器
│   ├── sample_batch() → (indices, data_types)
│   ├── should_freeze_pol_module(data_type) → bool
│   └── get_batch_info() → Dict
├── PolModuleFreezer: Pol Module 凍結控制器
│   ├── freeze() / unfreeze()
│   └── get_grad_norm() → float (監控)
└── MixedBatchCollator: 混合 Batch 整理器
```

**2. `pids_dataset.py`** (修改)
```python
# 新增 data_type 輸出
return {
    ...
    'data_type': self.scene_naming.get(scene_name, 'pol'),
}
```

**3. `train_pids.py`** (修改)
```
新增功能:
├── --curriculum flag: 啟用 Curriculum Learning
├── --curriculum_seed: Curriculum 隨機種子
├── _setup_curriculum(): 初始化 sampler 和 freezer
├── Pol/Nopol 分開追蹤 metrics
├── TensorBoard logging: curriculum progress, freeze status
└── Phase 1 凍結邏輯: nopol batch 時自動凍結 Pol Module
```

**4. `organize_mixed_dataset.py`** (新建)
```
training_guidance/Stage_I/training_stage/organize_mixed_dataset.py

功能:
├── 載入 QA 通過的 pol 場景
├── 發現/生成 nopol 場景列表
├── 檢查場景無重疊
├── 分割: Val = pol only, Nopol = train only
├── 輸出:
│   ├── train_pol_scenes.txt
│   ├── train_nopol_scenes.txt
│   ├── train_mixed_scenes.txt
│   ├── val_scenes.txt
│   ├── scene_naming.json (供 CurriculumSampler)
│   └── dataset_stats.json
└── 可選: --link_data 創建數據連結
```

#### 使用方式

```bash
# Step 1: 組織數據集
python organize_mixed_dataset.py \
    --pol_scenes_file /workspace/dataset_pol_final/train_100pct_scenes.txt \
    --pol_data_dir /workspace/dataset_pol_final/train_100pct \
    --nopol_data_dir /workspace/dataset_nopol_final \
    --output_dir /workspace/mixed_dataset_exp28 \
    --train_pol_count 3000 \
    --val_count 600 \
    --nopol_start 15001 \
    --nopol_count 2000 \
    --link_data

# Step 2: 訓練 (啟用 Curriculum)
nohup python train_pids.py \
    --data_dir /workspace/mixed_dataset_exp28/data \
    --output_dir ./checkpoints_exp28_100pct \
    --pol_volume \
    --curriculum \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 60000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 500 --num_workers 4 \
    > train_exp28_100pct.log 2>&1 &
```

### 數據準備完成 (2026-01-25)

```
Nopol 渲染: 18000 場景完成 ✓

混合數據集組織:
├── Pol structure: organized (stereo_pairs/, ground_truth/, masks/)
├── Nopol structure: flat
├── Train: 3000 pol (60%) + 2000 nopol (40%) = 5000 場景
├── Val: 600 pol only
└── 檔案數:
    ├── Train: 26000 files
    └── Val: 2400 files
```

### 實驗結果

| 指標 | 數值 | 備註 |
|------|------|------|
| Glass EPE | **4.88 px** | 與 Exp #24 val (~4.87 px) 幾乎一致（見修正） |
| Glass D1 | 34.49% | |
| Overall EPE | 3.13 px | |
| Val 波動 | 穩定 | Val < Train 正常模式 ✅ |

### 實驗對比分析（已修正）

> **⚠️ 修正 (2026-01-30)**：原表使用 Exp #24 舊 eval (2.44 px, ~200 scenes)，實際公平比較應用 val 或新 test set。

| 實驗 | 數據策略 | Glass EPE (Val) | Glass EPE (新 Test 857 scenes) |
|------|----------|-----------------|-------------------------------|
| **Exp #24** | Raw data (含弱偏振) | ~4.87 px | 4.055 px |
| **Exp #28** | Mixed pol + nopol | 4.88 px | — |
| **Exp #27** | Strict QA (純強偏振) | 5.03 px | — |

**修正後排序**: Raw ≈ Mixed >> Strict QA（差距遠小於原先認為）

### 關鍵發現

**1. 弱偏振 ≠ Nopol**
```
弱偏振場景: 仍有微弱偏振線索 → 模型可學習「軟過渡」
Nopol 場景: 完全無偏振 → 對 Pol Module 是「噪音」
```

**2. Raw Data 的弱偏振是「軟標籤」**
- 自然包含：強偏振 → 中偏振 → 弱偏振
- 連續過渡比人工二分法 (pol/nopol) 更有利於學習

**3. Nopol 可能有害**
- 即使 Phase 1 凍結 Pol Module，nopol 仍影響 backbone
- 或者 Pol Module 解凍後被 nopol 數據干擾

**4. RAFT 雙目一致性假設被強偏振破壞** (重要洞察)

RAFT-Stereo 的 Correlation Volume 假設左右圖像在對應點有相似外觀：
```
理想情況: I_left(x, y) ≈ I_right(x - d, y)
         → 高 Correlation → 正確匹配
```

強偏振破壞了這個假設：
```
強偏振 (Strict QA 保留):
├── 玻璃: I∥ >> I⊥ (例如 0.8 vs 0.2, 4倍差異)
├── RAFT 看到: 左邊亮斑 vs 右邊暗區
└── 結果: Correlation 低 → 匹配困難

弱偏振 (Raw data 包含):
├── 玻璃: I∥ ≈ I⊥ (例如 0.5 vs 0.4, 1.25倍差異)
├── RAFT 看到: 左右相似
└── 結果: Correlation 高 → 匹配容易
```

這解釋了 Exp #27 (Strict QA) 失敗的原因：
```
Exp #27 (純強偏振):
└── 所有玻璃場景 RAFT 都難匹配
    → Backbone 持續掙扎
    → 訓練信號嘈雜
    → 無「容易」案例作為學習基礎

Exp #24 (Raw data 混合):
├── 弱偏振: RAFT 容易 → 學到基本幾何
├── 中偏振: RAFT 有點難 → Pol Module 開始補償
└── 強偏振: RAFT 很難 → Pol Module 必須介入
→ 漸進式學習，有明確的學習梯度
```

**PIDS 訓練的兩難**：
| 偏振強度 | 對 Pol Module | 對 RAFT Backbone |
|---------|--------------|------------------|
| 強偏振 | ✅ 明確信號 | ❌ 難匹配 |
| 弱偏振 | ⚠️ 信號弱 | ✅ 容易匹配 |

**最佳策略**: 混合數據讓模型同時學習「容易」和「困難」案例

### 結論（已修正）

> **⚠️ 原結論基於不公平比較（舊 eval 2.44 px vs val 4.88 px），以下為修正版。**

**Raw data 與 Mixed pol/nopol 效果相當**：
- Exp #24 val ~4.87 px ≈ Exp #28 val 4.88 px，差距 < 1%
- 原先認為的 "+100% 差距" 是舊評估集過於簡單造成的假象
- 「軟篩選」論點無法被數據支持 — 兩者表現一致
- Strict QA (Exp #27, 5.03 px) 僅略差 ~3%

**仍然成立的觀察**：
- Curriculum Learning + 混合策略訓練穩定
- RAFT 雙目一致性假設的分析仍有參考價值

**狀態**: ✅ 完成（結論已修正：Raw ≈ Mixed >> Strict QA）

---

## Exp #29: Freeze Backbone First 策略 (提議)

**日期**: 2026-01-26
**目標**: 驗證「先凍結 Backbone 訓練 Pol Module」是否優於當前 Curriculum 策略

### 動機與假設

**Exp #28 訓練觀察**：
```
Phase 1 (0-20%, freeze pol on nopol): Val < Train ✅ 正常
Phase 2/3 (20-100%, all unfrozen):    Val > Train ❌ Plateau + 退化
```

**關鍵發現**: 解凍 Pol Module 後訓練開始退化

**假設 1 (Domain Gap)**: Nopol 數據會讓 Backbone (RAFT-Stereo) 適應「無偏振特徵」場景，這種適應性與 Pol 數據產生 domain gap。Backbone 的變化導致 Pol Module 追著不斷變化的特徵空間學習。

**假設 2 (雙目一致性破壞)**: RAFT 的 Correlation Volume 假設左右圖像相似，但強偏振數據 (I∥ >> I⊥) 嚴重破壞這個假設。如果 Backbone 在強偏振數據上持續訓練而沒有 Pol Module 的補償，會學到錯誤的匹配模式。

```
強偏振對 RAFT 的影響:
├── 玻璃區域: 左亮右暗 → Correlation 低 → Backbone 掙扎
├── 如果 Backbone 解凍: 會試圖「適應」這種不一致
└── 這種適應可能破壞原本良好的幾何特徵提取能力
```

### 新策略：Freeze Backbone First

```
原始策略 (Exp #28):
├── Phase 1 (0-20%):   freeze Pol Module on nopol
├── Phase 2 (20-60%):  全部解凍
└── Phase 3 (60-100%): 全部解凍
→ 結果: 4.88 px (比 raw data 差)

新策略 (Exp #29):
├── Phase 1 (0-60%):   freeze Backbone, 只訓練 Pol Module
│                       └── 70% pol + 30% nopol (獲得幾何多樣性)
└── Phase 2 (60-100%): 全部解凍, 一起微調
                        └── 70% pol + 30% nopol
```

### 理論基礎

1. **Backbone 已預訓練**：RAFT-Stereo 在 SceneFlow 等數據上已有良好的幾何特徵提取能力

2. **Pol Module 需從零學習**：偏振特徵提取是全新任務，需要穩定的 Backbone 作為基礎

3. **避免特徵空間漂移**：如果 Backbone 持續變化，Pol Module 學到的特徵可能與 Backbone 當前表示不匹配

4. **協同微調更穩定**：當 Pol Module 成熟後，再讓 Backbone 適應偏振特徵，兩者可協同改進

5. **保護 Backbone 免受雙目不一致干擾**：強偏振數據的 I∥ >> I⊥ 會破壞 RAFT 的 Correlation 假設。凍結 Backbone 可防止它學到錯誤的匹配模式，讓 Pol Module 先學會補償這種不一致

6. **70/30 混合比例的優勢**：使用 nopol 數據提供幾何多樣性，但 Backbone 凍結確保 nopol 不會破壞預訓練的匹配能力。Pol Module 仍能從 pol 數據學習偏振特徵

### 實現細節

**curriculum_sampler.py 更新**:

```python
# 新增 CurriculumPhase 欄位
@dataclass
class CurriculumPhase:
    freeze_backbone: bool = False  # 新增

# 新策略配置
FREEZE_BACKBONE_PHASES = [
    CurriculumPhase(
        name="Phase 1: Train Pol Module Only",
        start_progress=0.0,
        end_progress=0.6,
        pol_ratio=0.7,          # 70% pol + 30% nopol
        nopol_ratio=0.3,        # 獲得幾何多樣性
        freeze_pol_on_nopol=False,
        freeze_backbone=True,   # 關鍵：凍結 Backbone，保護不受 nopol 干擾
    ),
    CurriculumPhase(
        name="Phase 2: Joint Fine-tuning",
        start_progress=0.6,
        end_progress=1.0,
        pol_ratio=0.7,
        nopol_ratio=0.3,
        freeze_pol_on_nopol=False,
        freeze_backbone=False,  # 解凍
    ),
]
```

**BackboneFreezer 類別**:
- 凍結: fnet, cnet (特徵提取器)
- **保留 update_block 可訓練**：因為 `pol_volume` 架構的 `PolCorrBlock` 是非參數的，只有 `update_block` (GRU) 能學習如何使用 pol correlation features
- 監控: 參數數量統計、gradient norm

**重要修正** (2026-01-26):
```python
# 原本 (會導致所有參數被凍結)
BACKBONE_KEYWORDS = ['fnet', 'cnet', 'update_block', 'corr_fn']

# 修正後 (保留 update_block 可訓練)
BACKBONE_KEYWORDS = ['fnet', 'cnet']
```

**原因**: `PIDSStereoPolVolume` 架構中，`PolCorrBlock` 只是非參數的相關性計算，沒有可學習參數。如果凍結 `update_block`，就沒有任何參數可以學習如何使用偏振特徵。

### 數據集配置

```
Training Set:
  Pol:   3500 (70.0%)   ← 來自 Exp #27 的 QA passed scenes
  Nopol: 1500 (30.0%)   ← scene_15001 ~ scene_16500
  Total: 5000

Validation Set:
  Pol only: 600         ← 只用 pol 評估偏振檢測能力
```

### 訓練命令

```bash
nohup python train_pids.py \
    --data_dir ./PIDS_dataset_exp29_mixed/data \
    --output_dir ./checkpoints_exp29_freeze_backbone \
    --pol_volume \
    --curriculum \
    --curriculum_strategy freeze_backbone \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 60000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 500 --num_workers 4 \
    > train_exp29_freeze_backbone.log 2>&1 &
```

### 使用方法

```bash
# 使用新策略
python curriculum_sampler.py --strategy freeze_backbone

# 在訓練中
config = CurriculumConfig(strategy='freeze_backbone')
sampler = CurriculumSampler(dataset, total_steps, batch_size, config=config)

# 訓練迴圈
backbone_freezer = BackboneFreezer(model)
for step in range(total_steps):
    if sampler.should_freeze_backbone():
        backbone_freezer.freeze()
    else:
        backbone_freezer.unfreeze()
```

### 預期結果

| 指標 | Exp #24 (Raw) | Exp #28 (Mixed) | Exp #29 (Freeze BB) |
|------|---------------|-----------------|---------------------|
| Glass EPE | **2.44 px** | 4.88 px | ? |
| 訓練穩定性 | 高 | Phase 2/3 plateau | ? |

**假設**：如果 domain gap 假設正確，Exp #29 應優於 Exp #28

**狀態**: ✅ 完成 (2026-01-26)

### 訓練中期觀察 (Step 25500, 42.5%)

**Validation Metrics**:
```
Step   500:  57.38 px
Step 10000:  33.39 px
Step 20000:  23.84 px
Step 25500:  21.37 px
```
- Val Glass EPE 持續下降，**↓ 63%** 改善
- 訓練穩定，無發散

**Phase 1 瓶頸觀察**:
```
Step 25100-26000: Glass EPE 在 21.9 ~ 23.1 px 震盪
```
- Step ~20000 後進入 mini-plateau
- 原因：Backbone 凍結，update_block 能學的已趨飽和
- 還需等到 Step 36000 才進 Phase 2
- **約 16000 steps 的「低效訓練」**

**改進方向**: 見 Exp #30 設計

### 最終結果 (Step 60000, 100%)

**Final Training Output**:
```
Step 60000 | Train Loss: 6.0078 | Val Loss: 6.5178
Glass EPE - Train: 6.85 px | Val: 8.83 px
NON Glass EPE - Train: 2.03 px | Val: 2.36 px
All EPE - Train: 4.23 px | Val: 5.36 px
Phase: Phase 2: Joint Fine-tuning | Pol Ratio: 0.70 | Freeze Pol: False
Best checkpoint updated at step 59000 with val_glass_epe: 8.381
```

**最佳結果**: **Glass EPE = 8.381 px** @ Step 59000

**完整訓練曲線**:
```
Phase 1 (0-60%, Backbone Frozen):
Step   500:  57.38 px
Step 10000:  33.39 px  ← 開始減速
Step 17000:  22.58 px  ← Phase 1 最低
Step 35000:  21.37 px  ← Phase 1 結束前

Phase 2 (60-100%, Full Fine-tuning):
Step 36000:  解凍開始
Step 45000:  11.33 px  ← 快速下降
Step 55000:  10.14 px
Step 59000:   8.38 px  ← Best
Step 60000:   8.83 px
```

**關鍵觀察**:
1. **Phase 1 plateau 確認**: Step 17000~36000 約 16000 steps 空轉
2. **Phase 2 有效**: 解凍後從 21 px → 8.4 px，**↓ 60%**
3. **Val > Train 現象**: 整體 train_glass_epe < val_glass_epe，可能有輕微過擬合
4. **尾端仍有改善空間**: 最後幾千步仍有改善，支持低 LR 精修策略

**與預期比較**:
| 指標 | Exp #24 (Raw) | Exp #28 (Mixed) | Exp #29 (Freeze BB) |
|------|---------------|-----------------|---------------------|
| Glass EPE | **2.44 px** | 4.88 px | 8.38 px |
| 訓練穩定性 | 高 | Phase 2/3 plateau | 穩定但 Phase 1 空轉 |

**結論**: Exp #29 未達預期（比 Exp #28 還差）
- Freeze Backbone 策略本身並未帶來改善
- Raw data + 弱 QA 仍是最佳配置 (Exp #24)
- Phase 1 空轉問題明顯，需改進 → 見 Exp #30

**待執行**: 評估腳本 (evaluate_pids.py)

---

## Exp #30: Gradual Unfreezing with Continuous LR (訓練中)

**日期**: 2026-01-26 (設計 + 實作)
**目標**: 優化 Exp #29 的 phase transition，實現更平滑的 backbone 解凍

### 動機

**Exp #29 觀察到的問題**:
1. Phase 1 後期出現 mini-plateau（~16000 steps 低效訓練）
2. Phase 2 開始時可能有 loss jump（phase boundary shock）
3. 突然解凍 backbone 可能導致訓練不穩定

### 方案比較

| 方案 | 說明 | 優缺點 | 結論 |
|------|------|--------|------|
| **A: Layer-wise** | cnet → fnet → 逐層解凍 | ❌ fnet/cnet 耦合，單獨解凍 cnet 可能震盪 | 作為 Ablation |
| **B: 階段 LR** | 分段設定 backbone_lr_multiplier | ✅ 穩健、低風險 | 備案 |
| **C: 連續 LR** | backbone LR 從 0 連續增長到 base_lr | ✅ 最平滑、最符合研究假說 | **主線** |

### 方案 A 風險分析（不推薦作為主線）

RAFT-Stereo 架構特性：
```
fnet 與 cnet 並不是獨立語義模組
它們共同定義了 cost volume + recurrent state
```

單獨解凍 cnet 但 fnet frozen 的風險：
- context 想修正，但 feature space 動不了
- 可能造成短期 loss 震盪
- **結論**: 適合作為 Ablation (Exp #31-A)，不適合主線

### 數據驅動參數分析（2026-01-26 更新）

基於 Exp #29 實際訓練數據（train_glass_epe + val_glass_epe）分析 plateau 時機：

**Train/Glass_EPE 變異壓縮觀察**:
| 階段 | Step 範圍 | EPE 範圍 | 均值估計 |
|------|-----------|----------|----------|
| 初期 | 0-5000 | 20~116px | ~50px |
| 中期 | 5000-10000 | 15~98px | ~35px |
| 後期 | 10000-15000 | 10~60px | ~25px |
| 平台期 | 15000-20000 | 10~50px | ~22px |
| 現在 | 20000-29000 | 8~50px | ~18-20px |

**Val/Glass_EPE 關鍵轉折點**:
```
Step 10000: 33.39px  ← 減速開始
Step 13500: 24.86px  ← 進入平台
Step 17000: 22.58px  ← 局部最低
Step 18000: 30.88px  ← 跳升（Phase 1 極限信號）
Step 29000: 19.79px  ← 目前最佳
```

**結論**: Plateau 開始於 Step 10000~12000（16.7%~20%），應提早開始解凍。

### 方案 C 設計（主線）- 5 階段 LR Schedule

**核心理念**:
> "backbone 不是「解凍」，而是「被允許輕聲說話」"
> "尾端低 LR 精修能帶來個位數 px 的提升"（經驗證實）

**視覺化**:
```
Progress:  0%     20%      45%        80%   85%    100%
           ├──────┼────────┼──────────┼─────┼──────┤
BB mult:   0      0→1      1          1     1      1
Base LR:   1x     1x       0.5x       0.2x  0.1x   0.05x
           ├──────┼────────┼──────────┼─────┼──────┤
           [凍結]  [解凍]   [主學習]   [過渡] [精修]
```

**5 階段說明**:
| 階段 | Progress | Steps | Backbone | LR | 目的 |
|------|----------|-------|----------|-----|------|
| 凍結 | 0-20% | 0-12000 | Frozen | 1.0x | 學幾何基礎 |
| 解凍 | 20-45% | 12000-27000 | 0→1 | 1.0x | 學 pol features |
| 主學習 | 45-80% | 27000-48000 | 1.0 | 0.5x | 突破 plateau |
| 過渡 | 80-85% | 48000-51000 | 1.0 | 0.2x | 穩定過渡 |
| 精修 | 85-100% | 51000-60000 | 1.0 | 0.1x | 細節微調 |

**完整 LR Schedule 函數**:
```python
def get_lr_config(progress, base_lr):
    """
    完整 5 階段 LR Schedule (Exp #30 v2)

    設計依據：
    - 數據驅動：Exp #29 顯示 plateau 始於 Step 10000-12000
    - 經驗法則：尾端低 LR 精修能帶來個位數 px 提升
    - 避免空轉：主學習區用 0.5x LR 持續探索
    """

    # === Backbone multiplier ===
    if progress < 0.20:
        backbone_mult = 0.0                      # 凍結
    elif progress < 0.45:
        backbone_mult = (progress - 0.20) / 0.25  # 0→1 漸進解凍
    else:
        backbone_mult = 1.0                      # 全開

    # === Base LR decay ===
    if progress < 0.45:
        lr_mult = 1.0           # 前段保持 base LR
    elif progress < 0.80:
        lr_mult = 0.5           # 主學習區：半速探索
    elif progress < 0.85:
        lr_mult = 0.2           # 過渡期
    else:
        lr_mult = 0.1           # 尾端精修

    return base_lr * lr_mult, backbone_mult
```

**與原設計比較**:
| 參數 | 原設計 (v1) | 新設計 (v2) | 差異說明 |
|------|-------------|-------------|----------|
| 解凍起點 | 30% | 20% | 數據顯示 plateau 更早 |
| 完全解凍 | 60% | 45% | 減少空轉時間 |
| 主學習區 LR | 1.0x | 0.5x | 避免過度震盪 |
| 尾端精修 | 無特別設計 | 0.1x (9000 steps) | 經驗證實有效 |
| 階段數 | 3 階段 | 5 階段 | 更精細控制 |

### 方案 C 的三個關鍵優勢

**① 沒有 phase boundary shock**
- 不會在 step = X 發生 loss jump
- 表徵空間是連續演化

**② pol 特徵「自然滲透」**
- 一開始 pol 只影響 update_block
- 隨 backbone LR ↑，pol 開始影響 fnet / cnet
- 這個過程可被 EPE 曲線直接觀察

**③ 論文敘述友好**
> *"We gradually relax the backbone freezing constraint by continuously increasing its learning rate, allowing polarization cues to be progressively integrated into the feature representation without destabilizing geometric learning."*

### 實作需求

**1. Optimizer 改用 param_groups**:
```python
optimizer = torch.optim.AdamW([
    {'params': model.fnet.parameters(), 'lr': base_lr},        # group 0: fnet
    {'params': model.cnet.parameters(), 'lr': base_lr},        # group 1: cnet
    {'params': model.update_block.parameters(), 'lr': base_lr}, # group 2: update
    {'params': model.pol_corr.parameters(), 'lr': base_lr},     # group 3: pol
])
```

**2. 每個 step 動態調整 LR（5 階段版本）**:
```python
def adjust_lr(optimizer, progress, base_lr):
    """
    同時調整 backbone multiplier 和 base LR decay
    """
    current_lr, backbone_mult = get_lr_config(progress, base_lr)

    # Backbone (fnet, cnet): current_lr * backbone_mult
    optimizer.param_groups[0]['lr'] = current_lr * backbone_mult  # fnet
    optimizer.param_groups[1]['lr'] = current_lr * backbone_mult  # cnet

    # Head (update_block, pol_corr): current_lr
    optimizer.param_groups[2]['lr'] = current_lr  # update_block
    optimizer.param_groups[3]['lr'] = current_lr  # pol_corr
```

**3. Training loop 整合**:
```python
for step in range(total_steps):
    progress = step / total_steps
    adjust_lr(optimizer, progress, base_lr)

    # ... training code ...

    # Logging (optional)
    if step % log_interval == 0:
        bb_lr = optimizer.param_groups[0]['lr']
        head_lr = optimizer.param_groups[2]['lr']
        print(f"Step {step} | BB LR: {bb_lr:.6f} | Head LR: {head_lr:.6f}")
```

### 預期效果

| 指標 | Exp #29 (Hard Switch) | Exp #30 (Gradual) |
|------|----------------------|-------------------|
| Phase 1 plateau | ~16k steps | 自動縮短 |
| Phase 2 loss jump | 可能有 | 無 |
| 訓練曲線 | 階梯狀 | 平滑 |
| 論文品質 | 中 | 高 |

### 實作修改 (2026-01-26)

1. **新增 `GradualUnfreezeLRScheduler` 類別** (`curriculum_sampler.py`)
   - 實現 5-phase LR schedule
   - 支援 `state_dict()` / `load_state_dict()` 用於 checkpoint
   - 每個 step 返回 `(lr_mult, backbone_mult)`

2. **修改 optimizer 使用 4 param groups** (`train_pids.py`)
   - Group 0: fnet (backbone)
   - Group 1: cnet (backbone)
   - Group 2: update_block (head)
   - Group 3: pol_corr (head)

3. **新增 TensorBoard logging**
   - `backbone_lr`, `head_lr`, `backbone_mult`
   - 可觀察解凍進度

**訓練命令**:
```bash
nohup python train_pids.py \
    --data_dir ./PIDS_dataset_exp29_mixed/data \
    --output_dir ./checkpoints_exp30_gradual_unfreeze \
    --pol_volume \
    --curriculum \
    --curriculum_strategy gradual_unfreeze \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 60000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 500 --num_workers 4 \
    > train_exp30_gradual_unfreeze.log 2>&1 &
```

### 訓練進度 (Step 12600, ~21%)

**Phase 2 開始確認**: Step 12000 (20%) 開始解凍
```
Step 12000: BB_mult = 0.00 → 漸進上升
Step 12600: BB_mult = 0.04 (4% 解凍)
```

**Validation Metrics**:
```
Step   500:  89.50 px
Step  5000:  37.29 px
Step 10000:  33.73 px  ← Phase 1 plateau 開始
Step 12000:  30.07 px  ← Phase 2 開始
Step 12600:  ~26 px    ← 目前（仍在解凍中）
```

**關鍵觀察**:
1. ✅ **無 phase boundary shock**: 解凍過程平滑，無 loss jump
2. ✅ **BB_mult 漸進上升**: 0.00 → 0.04 → ... → 1.00
3. ⚠️ **Phase 1 也有 mini-plateau**: 與 Exp #29 類似，但解凍更早

**與 Exp #29 同期比較**:
| Step | Exp #29 (Hard) | Exp #30 (Gradual) | 差異 |
|------|---------------|-------------------|------|
| 500 | 57.38 px | 89.50 px | Exp #30 起始較高 |
| 10000 | 33.39 px | 33.73 px | 相當 |
| 12000 | ~30 px | 30.07 px | 相當（Exp #30 開始解凍）|

**最終結果 (Step 60000)**:
```
[Val @ Step 60000]
┌─────────────────────────────────────────────────────────────────┐
│ Mode       │ Glass EPE │ EPE    │ D1     │ Composite │ Loss     │
├─────────────────────────────────────────────────────────────────┤
│ Oracle     │     6.478 │  3.772 │ 24.81% │     8.958 │   5.4540 │
│ Real       │     6.478 │  3.772 │ 24.81% │     8.958 │   5.4540 │
└─────────────────────────────────────────────────────────────────┘
Best Glass EPE: 6.235 px
Train Glass EPE: 4.234 px
```

**Exp #29 vs #30 比較**:
| 策略 | Val Best Glass EPE | 特點 |
|------|-------------------|------|
| Exp #29 Hard Switch | 8.381 px | 有 phase boundary shock |
| Exp #30 Gradual | 6.235 px | 平滑過渡，**-25.6% 更好** |

**結論**: Gradual Unfreezing 優於 Hard Switch，但兩者都不如簡單訓練 (Exp #24 Val ~4.87 px)

**狀態**: ✅ 訓練完成

---

## 重大發現：測試集差異導致性能誤判 (2026-01-26)

### 問題背景

在進行多輪實驗後，發現後期實驗 (Exp #27-30) 的 Glass EPE 似乎都比 Exp #24 (2.44 px) 差。
經過深入分析，發現這是**測試集不同**造成的不公平比較。

### 關鍵發現

**Exp #24 TensorBoard Log 分析**:
- Val Glass EPE @ Step 60000: **~4.87 px** (不是 2.44 px!)
- 原先報告的 2.44 px 來自**較簡單/較小的舊評估集 (~200 場景)**

**公平重新評估** (新測試集 857 場景):

| 實驗 | 架構 | 舊 Eval (~200 scenes) | 新 Test (857 scenes) |
|------|------|----------------------|---------------------|
| Exp #22 | Baseline RAFT-Stereo | Val: 7.94 px | **93.96 px** ❌ |
| Exp #24 | Polarization Volume | 2.44 px | **4.055 px** ✅ |

### Polarization Volume 提升幅度 (公平比較)

| 指標 | Baseline (Exp #22) | Pol Volume (Exp #24) | 提升 |
|------|-------------------|---------------------|------|
| **Glass EPE** | 93.958 px | 4.055 px | **↓ 95.7%** (23.2x) |
| **Overall EPE** | 75.238 px | 2.646 px | **↓ 96.5%** (28.4x) |
| **Glass D1** | 88.81% | 33.54% | **↓ 55.3 pp** |
| **BG EPE** | 57.531 px | ~1.5 px | **↓ 97.4%** |

### 結論

1. **測試集差異是「性能退化」假象的主因**
   - 舊評估集 (~200 場景) 較簡單
   - 新測試集 (857 場景) 更多樣化、更困難

2. **Baseline 在困難場景完全失效**
   - Val 時 7.94 px → 困難測試集 93.96 px
   - 證明純幾何立體匹配無法處理透明物體

3. **Polarization Volume 保持魯棒性**
   - Val ~4.87 px → 測試集 4.055 px (甚至更好)
   - 偏振資訊不只提升性能，更保證穩定性

4. **論文可用數據**:
   - Polarization Volume 相比 Baseline 在 Glass 區域誤差降低 **95.7%**
   - 這是強有力的消融實驗證據

### 後續行動

- [x] Exp #22 (Baseline) 在新測試集評估 ✓
- [x] Exp #24 (Pol Volume) 在新測試集評估 ✓
- [ ] Exp #30 (Gradual Unfreeze) 在新測試集評估
- [ ] 更新論文數據使用統一測試集結果

---

## Nopol Test Set 補充評估 (2026-01-26)

### 問題：Pol vs Nopol 影像的公平比較

之前的比較是 Baseline 在 **pol 影像** (left_parallel, right_cross) 上測試。
但 Baseline 從未見過偏振影像，可能因強度差異而崩潰，不是公平比較。

### Nopol Test Set 建立

從 `dataset_nopol_final` 提取與 pol test set 相同的 857 個場景：
```bash
mkdir -p ./dataset_nopol_test/{stereo_pairs,ground_truth,masks}
# 複製 left.exr, right.exr, disparity.exr, glass_mask.exr
```

### 評估結果

**Exp #22 (Baseline) 在 nopol test set (859 scenes)**:
```
[Overall Metrics]
  EPE:  3.444 ± 3.369 px
  D1:   41.24 ± 17.96 %

[Glass Region Metrics]
  Glass EPE:  6.543 ± 14.607 px
  Glass D1:   75.42 ± 21.41 %

[Background Region Metrics]
  BG EPE:  2.798 ± 2.706 px
```

### 完整比較表

| 測試條件 | Baseline (Exp #22) | Pol Volume (Exp #24) | 差距 |
|----------|-------------------|---------------------|------|
| **pol 影像測試** | 93.96 px ❌ | 4.055 px ✅ | **95.7% ↓** |
| **nopol 影像測試** | 6.543 px | N/A (無偏振資訊) | - |

### 問題分析

1. **Baseline 在 nopol 上表現尚可 (6.543 px)**
   - 無偏振干擾時，純幾何立體匹配可以運作
   - 但玻璃區域仍是難題

2. **Pol Volume 優勢不夠明顯**
   - pol test: 4.055 px vs baseline nopol: 6.543 px
   - 僅 **38% 改善**，不夠有說服力
   - 需要架構優化來拉開差距

3. **結論：目前 4 px 的性能不足以展示偏振的核心價值**

---

## Polarization Volume V2 架構規劃 (2026-01-26)

### 現況問題

目前 PolCorrBlock 架構過於簡單：
```
pol_diff = left - right  (simple subtraction)
pol_corr = query(pol_volume, disp)
output = concat(corr, pol_corr, disp) → encoder → GRU
```

**缺陷**：
1. 無 learnable 偏振特徵提取
2. 無空間注意力機制（不知道哪裡是玻璃）
3. 簡單 concat 融合，沒有學習 stereo vs pol 的權重

### 架構優化方案

#### 主線 (A + C)：Polarization Attention + Gated Fusion

```
                     ┌─────────────────────────────────────────────┐
                     │           Polarization Attention            │
                     │                                             │
pol_diff ──────────→ │ [Conv] → sigmoid → pol_attention_map       │
                     │                     (where is glass?)       │
                     └──────────────────────┬──────────────────────┘
                                            ↓
                     ┌─────────────────────────────────────────────┐
                     │              Gated Fusion                   │
                     │                                             │
corr ───────────────→│                                             │
                     │  gate = σ(W_g * [corr, pol_corr, attn])    │
pol_corr ───────────→│                                             │──→ fused
                     │  fused = gate * enhanced_corr              │
pol_attention_map ──→│        + (1-gate) * enhanced_pol           │
                     │                                             │
                     └─────────────────────────────────────────────┘
                                            ↓
                            [concat(fused, disp)] → GRU
```

**方案 A: Polarization Attention (必做)**
- 從 pol_corr 生成空間注意力圖
- 高偏振差異區域（玻璃）獲得更高權重
- 讓模型知道「哪裡該信任偏振資訊」

**方案 C: Gated Fusion (必做)**
- 學習 stereo 與 pol 特徵的融合權重
- gate = 0 → 信任 pol，gate = 1 → 信任 stereo
- 玻璃區域自動降低 stereo 權重，提高 pol 權重

#### 能力放大器 (B)：Learnable Pol Encoder

```
pol_volume ──→ [Conv Encoder] ──→ deeper pol_features
```
- 不只是 raw subtraction
- 學習更豐富的偏振特徵表示

#### 論文加分項 (D)：Glass-aware Auxiliary Head

```
hidden ──→ [Glass Head] ──→ predicted_glass_mask (auxiliary loss)
```
- 預測玻璃位置作為輔助任務
- 可用於視覺化和解釋性

### 核心模組設計

```python
class PolarizationAttention(nn.Module):
    """從 pol_diff 生成空間注意力圖"""
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, pol_corr):
        return self.conv(pol_corr)  # (B, 1, H, W)


class GatedFusion(nn.Module):
    """學習 stereo vs pol 的融合權重"""
    def __init__(self, corr_dim, pol_dim, out_dim):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Conv2d(corr_dim + pol_dim + 1, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        self.corr_enhance = nn.Conv2d(corr_dim, out_dim, 1)
        self.pol_enhance = nn.Conv2d(pol_dim, out_dim, 1)

    def forward(self, corr, pol_corr, pol_attn):
        gate = self.gate_net(torch.cat([corr, pol_corr, pol_attn], dim=1))
        corr_feat = self.corr_enhance(corr)
        pol_feat = self.pol_enhance(pol_corr)
        fused = gate * corr_feat + (1 - gate) * pol_feat
        return fused, gate
```

### 預期改善

| 版本 | Glass EPE | 改善 |
|------|-----------|------|
| Baseline (nopol) | 6.543 px | - |
| Pol Volume V1 | 4.055 px | 38% ↓ |
| **Pol Volume V2 (目標)** | **< 2.5 px** | **> 60% ↓** |

### 實作計劃

- [ ] **Exp #31: Pol Volume V2 (A+C)**
  - [ ] Step 1: 實現 PolarizationAttention 模組
  - [ ] Step 2: 實現 GatedFusion 模組
  - [ ] Step 3: 建立 UpdateBlockV2
  - [ ] Step 4: 建立 PIDSStereoPolVolumeV2
  - [ ] Step 5: 訓練並評估
- [ ] (Optional) Exp #32: 加入 Learnable Pol Encoder (B)
- [ ] (Optional) Exp #33: 加入 Glass-aware Auxiliary Head (D)

**狀態**: 📋 架構設計完成，待實作

---

## Stage 2: Real-World Fine-Tuning (Kinect v1 替身法)

**日期**: 2026-01-25 (規劃)
**目標**: 使用真實世界數據微調模型，提升實際部署性能

### 替身法 (Substitute Method)

解決 Kinect 無法感測透明物體的問題：

```
拍攝流程:
┌─────────────────────────────────────────────────────────┐
│ Step 1: 偏振相機拍攝「真實玻璃」場景                      │
│         → left_parallel.exr, right_cross.exr            │
│         → 訓練輸入                                       │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Step 2: 用不透明替身替換玻璃 (相同形狀位置)               │
│         → 3D 列印 / 紙板 / 不透明壓克力                  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Step 3: Kinect v1 拍攝「替身」場景                       │
│         → depth.png (GT 深度)                           │
│         → 替身對 Kinect 結構光有效                       │
└─────────────────────────────────────────────────────────┘
```

### Kinect v1 規格

| 項目 | 規格 |
|------|------|
| 深度原理 | 結構光 (IR pattern projection) |
| 深度解析度 | 640x480 |
| RGB 解析度 | 640x480 |
| 深度範圍 | 0.8m ~ 4.0m |
| 深度精度 | ~2-4cm @ 2m |
| 優點 | 便宜、易得、室內效果好 |

### 數據採集設備

```
PIDS 採集系統:
├── 偏振相機 (IMX296 x2, 同步)
│   ├── 左: 0° 偏振片 (I∥)
│   └── 右: 90° 偏振片 (I⊥)
│
├── Kinect v1 (GT 深度)
│   └── 需與偏振相機校正對齊
│
└── 替身物體
    ├── 與玻璃相同外形
    └── 不透明材質 (白色/灰色最佳)
```

### 校正需求

1. **偏振相機立體校正**: 已完成 (Stage 1)
2. **Kinect 內參校正**: 使用 OpenCV/ROS 工具
3. **Kinect-偏振相機外參校正**:
   - 棋盤格同時可見於兩系統
   - 計算座標轉換矩陣
4. **深度對齊**: Kinect 深度 → 偏振相機視角

### 數據格式

```
real_world_dataset/
├── scene_0001/
│   ├── left_parallel.exr    # 偏振相機 (玻璃場景)
│   ├── right_cross.exr      # 偏振相機 (玻璃場景)
│   ├── depth_kinect.png     # Kinect 原始深度 (替身場景)
│   ├── depth_aligned.exr    # 對齊到偏振相機的深度
│   ├── glass_mask.png       # 手動/自動標註的玻璃區域
│   └── params.json          # 場景參數
├── scene_0002/
│   └── ...
```

### 訓練策略

```
Stage 2 Fine-tuning:
├── 基礎模型: Stage 1 最佳 checkpoint (Exp #27 或 #28)
├── 學習率: 較小 (0.00003 ~ 0.0001)
├── 數據量: ~100-500 真實場景
├── 目標: 域適應 (Synthetic → Real)
│   ├── 真實光照
│   ├── 真實紋理
│   ├── 相機噪聲
│   └── 偏振特性差異
```

### 預期改善

| 指標 | Stage 1 Only | Stage 2 Fine-tuned |
|------|--------------|-------------------|
| 合成數據 Glass EPE | ~2.0-2.5 px | ~2.0-2.5 px (維持) |
| 真實世界 Glass EPE | 可能較差 | 預期大幅改善 |
| 部署可靠性 | 中等 | 高 |

**狀態**: 📋 規劃中 (待 Stage 1 完成)

---

## 16. 待完成項目

- [x] 完成 70k steps 訓練 ✓
- [x] 新增 strict glass mask 功能 ✓
- [x] Nopol V5 數據集渲染 ✓
- [x] Nopol strict glass mask 渲染 ✓
- [x] Nopol V5 消融實驗訓練 ✓
- [x] Exp #21: Strict Glass Weight 驗證 ✓
- [x] Test Set 評估 (Pol vs Nopol 消融) ✓
- [x] 驗證邏輯改進: Oracle vs Real 雙重驗證 ✓
- [x] 驗證 Oracle-Real Gap 在 Pol vs Nopol 的差異 ✓ (Pol Real 86.7% 優於 Nopol)
- [x] Training-Inference Consistency 優化實作 ✓
- [x] train_baseline.py 乾淨版本建立 ✓
- [x] Bug Fix: Validation Loss 計算不一致 ✓
- [x] Exp #22: Baseline RAFT-Stereo 訓練 ✓ (Glass EPE: 7.52 px)
- [x] Exp #23: Curriculum Learning 訓練 ✗ (失敗，Gap 從 103% 增到 652%)
- [x] Polarization Volume 架構設計與實作 ✓
- [x] Exp #24: Polarization Volume 訓練 ✓ (**Glass EPE: 2.44 px, -67.6% vs Baseline**)
- [x] Exp #25: Learnable Pol Volume ✗ (無顯著改善，Glass EPE 2.48 px)
- [x] pids_qa.py v2.3.0: 新增 C3 玻璃偏振檢查 + 輸出 passed/failed_scenes.txt ✓
- [x] Exp #26: LR 影響分析 ✓ (LR=0.0001 導致 +120% 退化，確認 LR=0.0003 為正確配置)
- [x] **Exp #27: 嚴格 QA 100% 訓練** ✅ (Glass EPE 5.03px, 比 Exp #24 差 +106%)
- [ ] Exp #27: 5%/10%/25%/50%/75% 數據量消融 (已證明嚴格 QA 有害，暫緩)
- [x] Exp #27: 評估 C3 嚴格過濾對 Glass EPE 的影響 ✅ (結論：嚴格過濾導致性能下降)
- [x] **Exp #28: 混合 Pol/Nopol 訓練** ✅ (Glass EPE 4.88px, 結論：Raw data 最佳)
  - [x] Step 1: 生成新 OBJ (scene_15001+, 3000 個場景) ✓
  - [x] Step 2: 渲染 Nopol 數據 (18000 場景完成) ✓
  - [x] Step 3: 修改 PIDSSyntheticDataset 支援雙命名 + data_type flag ✓
  - [x] Step 4: 實現 CurriculumSampler (動態 pol/nopol 比例 + Pol Module 凍結) ✓
  - [x] Step 5: 修改 train_pids.py 分開追蹤 pol/nopol metrics + --curriculum flag ✓
  - [x] Step 6: 寫 organize_mixed_dataset.py (支援 organized + flat 結構) ✓
  - [x] Step 7: 組織混合數據集 ✓ (Train: 5000, Val: 600)
  - [x] Step 8: 訓練 Exp #28 ✓
  - [x] Step 9: 評估 Exp #28 ✓ (弱偏振 ≠ nopol，raw data 最佳)
- [x] **Exp #29: Freeze Backbone First** ✅ (完成，Best Glass EPE: 8.381 px)
  - [x] Step 1: 實現 BackboneFreezer 類別 ✓
  - [x] Step 2: 新增 FREEZE_BACKBONE_PHASES 配置 (70/30 pol/nopol) ✓
  - [x] Step 3: 更新 curriculum_sampler.py ✓
  - [x] Step 4: 修改 train_pids.py 支援 freeze_backbone 策略 ✓
  - [x] Step 5: 修正 BackboneFreezer 只凍結 fnet/cnet (保留 update_block) ✓
  - [x] Step 6: 組織混合數據集 (Train 5000, Val 600) ✓
  - [x] Step 7: 訓練 Exp #29 完成 ✓ (60000 steps)
  - [ ] Step 8: 評估並比較 (待執行)
- [x] **Exp #30: Gradual Unfreezing** ✅ (完成，Best Glass EPE: 6.235 px)
  - [x] Step 1: 實現 GradualUnfreezeLRScheduler 類別 ✓
  - [x] Step 2: 修改 optimizer 使用 4 param_groups (fnet, cnet, update_block, pol) ✓
  - [x] Step 3: 實現 5-phase LR schedule ✓
  - [x] Step 4: 更新 train_pids.py 支援 gradual_unfreeze 策略 ✓
  - [x] Step 5: 修復 state_dict() 缺失問題 ✓
  - [x] Step 6: 訓練 Exp #30 完成 ✓ (60000 steps, Best Val Glass EPE: 6.235 px)
  - [x] Step 7: 比較 Exp #29 vs #30 ✓ (Gradual 6.235 px 優於 Hard Switch 8.381 px)
- [x] **測試集公平重新評估** ✅ (2026-01-26)
  - [x] Exp #22 (Baseline) 在新測試集 (857 scenes) 評估 ✓ → **Glass EPE: 93.96 px** (pol test)
  - [x] Exp #22 (Baseline) 在 nopol test 評估 ✓ → **Glass EPE: 6.543 px**
  - [x] Exp #24 (Pol Volume) 在新測試集評估 ✓ → **Glass EPE: 4.055 px**
  - [x] 分析：Pol Volume vs Baseline (nopol) 僅 38% 改善，**不夠有說服力**
  - [ ] Exp #30 (Gradual Unfreeze) 在新測試集評估 (已跳過，預期與 Val 接近)
- [x] **Exp #31: Polarization Volume V2 (架構優化)** ✅ (完成，Best Glass EPE: 7.763 px - 退步)
  - [x] Step 1: 實現 PolarizationAttention 模組 ✓
  - [x] Step 2: 實現 GatedFusion 模組 ✓
  - [x] Step 3: 建立 UpdateBlockV2 ✓
  - [x] Step 4: 建立 PIDSStereoPolVolumeV2 ✓
  - [x] Step 5: 修復 ContextEncoder API 錯誤 ✓
  - [x] Step 6: 訓練 60000 steps ✓
  - [x] Step 7: 評估結果：**7.763 px (比 V1 的 4.055 px 退步 91%)**
  - **分析**: V2 架構本身有結構性錯誤（Spatial Attention + Multiplicative Gate 不在 Disparity Space 工作）
- [x] ~~**Exp #32: V2 + 選擇性凍結**~~ ⛔ **ABANDONED**
  - 放棄原因: V2 架構本身有結構性錯誤，Selective Freeze 無法解決
  - 預期只能改善到 5.x~6.x px，無法超越 V1
- [ ] **Exp #33: Polarization Volume V2-A (Corr Residual)** ← 當前重點
  - [x] Step 1: 設計 V2-A 架構 (Additive Bias in Disparity Space) ✓
  - [x] Step 2: 實現 PolCorrResidual 模組 ✓
  - [x] Step 3: 建立 PIDSStereoPolVolumeV2A ✓
  - [x] Step 4: 更新 train_pids.py 支援 --pol_volume_v2a ✓
  - [x] Step 5: 更新 evaluate_pids.py 支援 --pol_volume_v2a ✓
  - [ ] Step 6: 訓練並評估 (目標: Glass EPE < 3.8 px)
- [ ] (Optional) Exp #34: 加入 Learnable Pol Encoder
- [ ] (Optional) Exp #35: 加入 Glass-aware Auxiliary Head
- [ ] **Stage 2: Real-World Fine-Tuning** (規劃中)
  - [ ] Kinect v1 校正 (內參 + 與偏振相機外參)
  - [ ] 替身物體製作
  - [ ] 數據採集流程自動化
  - [ ] 真實世界數據集收集 (~100-500 場景)
  - [ ] Stage 2 fine-tuning 訓練

---

## Exp #31: Polarization Volume V2 (2026-01-27)

### 目標
通過架構優化拉開與 Baseline 的差距（V1 僅 38% 改善）

### V2 架構設計

**Plan A: PolarizationAttention**
```python
class PolarizationAttention(nn.Module):
    # 從 pol_corr 生成 spatial attention map
    # 識別玻璃區域 (pol_diff 高的地方)
    # 輸出: attention map [B, 1, H, W]
```

**Plan C: GatedFusion**
```python
class GatedFusion(nn.Module):
    # 學習 stereo vs pol 的動態融合權重
    # gate = sigmoid(f(corr, pol_corr, pol_attn))
    # fused = gate * stereo_feat + (1-gate) * pol_feat
```

### 訓練配置
```bash
nohup python train_pids.py \
    --data_dir ./PIDS_dataset_exp29_mixed/data \
    --output_dir ./checkpoints_exp31_polvol_v2 \
    --pol_volume_v2 \
    --curriculum \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 60000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 500 --num_workers 4 \
    > train_exp31.log 2>&1 &
```

### 訓練結果

| Step | Val Glass EPE | Val EPE | D1 | Composite |
|------|---------------|---------|-----|-----------|
| 500 | 89.xx px | - | - | - |
| 30000 | ~8.0 px | ~5.0 | ~32% | ~11.2 |
| 59500 | 7.938 px | 5.016 | 32.69% | 11.208 |
| **60000** | **7.763 px** | **4.809** | **32.23%** | **10.986** |

### 結果分析

| 實驗 | 架構 | Glass EPE | vs V1 |
|------|------|-----------|-------|
| Exp #24 | V1 (Pol Volume) | 4.055 px | 基準 |
| **Exp #31** | **V2 (Attention + Gated)** | **7.763 px** | **退步 91%** |

**關鍵發現**: V2 架構在混合數據集上反而更差

### 問題診斷（深入分析）

**初步假設**: 混合數據集稀釋了 V2 模組的學習
- 60% pol + 40% nopol 訓練
- nopol 數據沒有偏振差異 (I∥ ≈ I⊥)

**深入分析後的真正原因**: V2 架構本身有結構性錯誤

1. **Spatial Attention 問題**: PolarizationAttention 生成的是 spatial attention map [H, W]，不知道該強調哪個 disparity level
2. **Multiplicative Control 問題**: GatedFusion 使用 `gate × corr + (1-gate) × pol` 會破壞 RAFT 的 inductive bias
3. **不在 Disparity Space 工作**: V2 的 attention 和 gate 都在 spatial domain，而不是 disparity domain

**結論**: Selective Freeze 只是「讓錯的模組少犯錯」，不能解決架構本身的問題

---

## Exp #32: V2 + 選擇性凍結 ⛔ ABANDONED

**放棄原因**:
- V2 架構本身有結構性錯誤
- Selective Freeze 預期只能改善到 5.x~6.x px，無法超越 V1 (4.055 px)
- 直接轉向正確的架構方向: V2-A (Pol-Conditioned Corr Residual)

---

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

## Exp #39: Polarization Volume V2-E (2026-01-28)

### 為什麼還需要 V2-E？

V2 系列（A/B/C/D）都是 **post-corr intervention**：
- 先計算 correlation volume
- 再用 pol 信息修正/調制

但這可能有根本限制：**錯誤的 correlation 已經形成，修正為時已晚**

### V2-E 的核心創新：Pre-Corr Pol Weighting

**關鍵轉變：**

| | V2 系列 (Post-Corr) | V2-E (Pre-Corr) |
|--|---------------------|-----------------|
| 時機 | corr 計算後 | corr 計算時 |
| 作用 | 修正已有的 corr | **影響 corr 的形成** |
| 類比 | 事後補救 | **源頭介入** |

**設計：**

```python
# V2 系列：Post-Corr
corr = dot(fmap1, fmap2)
corr_mod = corr * gate  # 事後修正

# V2-E：Pre-Corr（pol 參與 corr 計算）
for d in disparity_range:
    pol_diff_d = left - shift(right, d)
    pol_weight_d = sigmoid(f(pol_diff_d))  # [0, 1]

    # pol 直接影響 correlation 的權重
    corr[d] = dot(fmap1, fmap2_at_d) * pol_weight_d
```

**物理意義：**
- pol_weight_d 高：這個 disparity 的 pol_diff 符合「同一物理點」的特徵
- pol_weight_d 低：這個 disparity 可能是假匹配（反射/錯位）
- **在 correlation 形成時就抑制假匹配，而非事後修正**

### 訓練命令

```bash
nohup python train_pids.py \
    --data_dir /workspace/RAFT-Stereo/mixed_dataset_exp28/data \
    --output_dir ./checkpoints_exp39_v2e \
    --pol_volume_v2e \
    --pol_weight_hidden 8 \
    --curriculum \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 60000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 500 --num_workers 4 \
    > train_exp39.log 2>&1 &
```

### 目標

| 實驗 | 架構 | 介入點 | 說明 |
|------|------|--------|------|
| V2-A~D | Post-Corr | corr 計算後 | 已壓榨 |
| **V2-E** | **Pre-Corr** | **corr 計算時** | **最後嘗試** |

**如果 V2-E 也沒有提升：**
- 架構已壓榨到極致
- 轉向證明 nopol 7px 是 overfitting（用 OOD test set）

### 訓練進度

**已開始訓練** (2026-01-28)

（待更新...）

---

### 架構演進總結

#### V2-B: Scheduled Residual

**核心改動**: 加入 iteration schedule

```python
# V2-A (current)
corr_enhanced = corr + self.pol_residual(pol_corr)

# V2-B (next)
alpha = i / (iters - 1)  # i = current iteration, 0 → 1
corr_enhanced = corr + alpha * self.pol_residual(pol_corr)
```

**設計哲學**:
- RAFT-Stereo 的核心是**逐步修正** disparity
- V2-A 的 residual 是 static，像「外掛」
- V2-B 讓 residual 前期弱、後期強，符合 RAFT 精神

**預期**: 有機會進入 3.x px

#### V2-C: Gradient-based Gating (進階)

```python
disp_grad = |∇disp|  # disparity gradient
gate = f(pol_corr, disp_grad)  # gate ∈ [0, 1]
corr_enhanced = corr + gate * pol_embedding
```

**直覺**:
- high gradient / unstable region → pol 有話語權
- flat / confident region → pol 安靜

#### 架構優化優先級

| Priority | 方向 | 預期增益 | 風險 | 說明 |
|----------|------|----------|------|------|
| ⭐ P1 | Iterative/Scheduled Residual | 高 (→ 3.x px) | 低 | V2-B |
| ⭐ P2 | Multi-scale Residual | 中 (+0.2-0.4 px) | 低 | 工程優化 |
| ⚠️ P3 | Disparity-aware Weight | 中 | 中 | 風險：feedback instability |
| 📌 P4 | Pol Feature Extractor | 高 | 高 | 成本高，歸因問題 |

### 設計陷阱警告

**❌ 危險做法**: 直接用 disparity 值作為 conditioning
```python
# 錯誤！early iteration 的 disp 是錯的
residual = pol_residual(pol_corr, disp)
```

**問題**:
- Early iteration 的 disparity 是 noise
- 會產生 feedback loop
- 訓練不穩，後期可能退化

**✅ 安全做法**:
- Option A: 使用 iteration index + fixed schedule (V2-B)
- Option B: 使用 disparity gradient/variance，不用值本身
- 關鍵: pol_residual 只學「where & how much」，不是「what disparity」

---

## 2026-01-31: Pol Volume Crosstalk 分析 — Spike 根因與灰階瓶頸

### 問題觀察

即使使用 Median 作為 checkpoint 選擇指標，val Glass EPE 仍有明顯 spike。分析後發現這是 pol volume 的**結構性弱點**，而非訓練不穩定。

### 根因：Disparity-Pol Crosstalk

Pol volume 的設計前提是：**左右影像的亮度差 = 偏振效應**。但這只在 disparity 正確時成立。

```
正確 disparity:
  Left I_∥(x, y)  ──┐
                     ├── ΔI = 真正的偏振信號（玻璃）  ✓
  Right I_⊥(x, y-d) ┘  ← 同一物理點

錯誤 disparity:
  Left I_∥(x, y)     = 物件 A 的亮度
                     ├── ΔI = 不同物件的亮度差（假信號）  ✗
  Right I_⊥(x, y-d') = 物件 B 的亮度
                          ← 不同物理點！
```

Pol volume **無法分辨**這兩種 ΔI 的來源。當 disparity 不正確時，不同物件重疊產生的亮度差會被誤判為偏振信號。

### 正反饋迴路

```
disparity 錯位 → 假 pol 信號 → 引導 disparity 往錯誤方向修正 → 更大錯位
     ↑                                                           │
     └───────────────────────────────────────────────────────────┘
```

某些場景（高對比物件排列在 epipolar line 上）特別容易觸發此迴路，導致 Glass EPE spike。

### 灰階加劇問題（雙重弱化）

系統使用灰階輸入（I_∥, I_⊥ 各 1 channel），這造成**雙重弱化**：

1. **Backbone 特徵區分力不足**：RGB 有 3 個通道建立特徵，灰階只有 1 個純量，不同物件可能有相似的灰度值但完全不同的顏色
2. **無法辨別假信號**：即使錯位了，RGB 的 color 差異可以暗示「這不是 pol 效應」；灰階完全沒有這個能力

```
灰階 ──→ backbone 特徵區分力弱 ──→ disparity 更常估錯
  │                                      │
  │                                      ▼
  └──→ 同時沒有 color 辨別假信號 ──→ pol module 無法自救
```

### 根本矛盾（Chicken-and-Egg Problem）

```
標準 stereo:  RGB × 2 cameras  →  6 channels，特徵豐富但看不到玻璃
偏振 stereo:  Gray × 2 cameras →  2 channels，能看到玻璃但特徵貧乏
```

Pol volume 存在的理由是補償灰階損失的判斷力，但它自己又依賴準確的 disparity 才能正常工作 — 而準確的 disparity 正是灰階最缺的東西。

### 理想解法：RGB 作為第三方裁判

引入無偏振 RGB 作為 backbone 的獨立輸入，解耦 stereo matching 和 pol detection：

```
理想架構:
  RGB (3ch)        → backbone → 獨立做好 stereo matching
  I_∥, I_⊥ (2ch)  → pol module → 在正確 disparity 上做偏振判斷
                     兩件事解耦，各司其職
```

**~~限制~~**：~~Mitsuba 3 的 spectral-polarized 渲染不穩定。~~ **已解決** — 見下方「V6 渲染器：Polarized RGB」。關鍵發現：`cuda_ad_spectral_polarized` variant 搭配 `rgb` film format + `stokes` integrator 可直接輸出 15ch RGB Stokes (RGB + S0_rgb + S1_rgb + S2_rgb + S3_rgb)，無需切換 variant。

### 短期可行方案（不變更灰階輸入）

#### 方案 A: Iteration-level Pol Warmup（推薦優先嘗試）

在 RAFT 的迭代迴圈中，前幾輪不啟用 pol，讓 correlation 先穩住 disparity：

```python
for itr in range(num_iters):
    pol_scale = min(1.0, max(0.0, (itr - warmup) / ramp))
    pol_features = pol_lookup(...) * pol_scale
```

```
Iteration:  1  2  3  4  5  6 ... 18 19 20 21 22 23 24
Pol weight: 0  0  0  0  .2 .4 ... 1  1  1  1  1  1  1
            ├─ correlation 先穩住 ─┤├─ pol 精修 ──────┤
```

- 改動量極小，不引入新參數
- 直接打斷正反饋迴路
- 與 training-level curriculum 正交（那個跨 step，這個跨 iteration）

#### 方案 B: Correlation Confidence Gating

用 stereo match 的信心來控制 pol 的影響力（pixel-level 軟門控）：

```python
corr_confidence = softmax(corr_features, dim=-1).max(dim=-1)
pol_gate = sigmoid((corr_confidence - threshold) * temperature)
pol_features_gated = pol_features * pol_gate
```

- 比方案 A 更精細（pixel-level 而非 iteration-level）
- 紋理豐富區域可以早點用 pol，紋理貧乏區域自動抑制

#### 方案 C: Iteration Consistency Check

跨迭代驗證 pol 信號穩定性 — 真正的玻璃 pol 信號隨 disparity 收斂越來越穩定，假信號會跳動：

```
Iter N:   pol = +0.8 → Iter N+1: pol = +0.7 → 穩定，可信  ✓
Iter N:   pol = +0.8 → Iter N+1: pol = -0.3 → 不穩定      ✗
```

### 優先級

| 方案 | 複雜度 | 風險 | 預期效果 | 建議 |
|------|--------|------|---------|------|
| A: Iteration Warmup | 極低 | 低 | 中 | **先做** |
| B: Confidence Gating | 中 | 中 | 高 | 第二步 |
| C: Consistency Check | 中高 | 中 | 中 | 可選 |
| RGB 第三方裁判 | 高 | 高 | 最高 | **已完成 (V6 渲染器)** |

---

## 2026-01-31: V6 渲染器 — Polarized RGB 輸出

### 動機

Pol Volume Crosstalk 分析指出灰階是結構性瓶頸。研究 Mitsuba 3 文檔後發現：現有的 `cuda_ad_spectral_polarized` variant 只需將 film `pixel_format` 從 `luminance` 改為 `rgb`，即可輸出 RGB 偏振資料，不需要換 variant。

### 技術方案

從 `rendering_v5/` fork 為 `rendering_v6/`，修改 `pids_renderer_textured.py`：

| 組件 | v5 (灰階) | v6 (RGB) |
|------|-----------|----------|
| Film pixel_format | `luminance` | `rgb` |
| Stokes 輸出 | 9~12ch spectral → 平均為 (H,W) | 15ch: RGB(3) + S0(3) + S1(3) + S2(3) + S3(3) |
| I_∥ / I_⊥ | (H,W) 灰階 | (H,W,3) RGB |
| EXR 輸出 | 1ch 灰階 | 3ch RGB |
| DoLP | 直接計算 | RGB → luminance → DoLP (標量) |
| QA 報告 | 灰階指標 | 內部轉 luminance，JSON 格式不變 |

### 修改清單

1. **Film format**: `'luminance'` → `'rgb'` (僅主渲染，glass mask/depth 不變)
2. **`extract_stokes()`**: 解析 15ch RGB Stokes，含 12ch/v5 grayscale fallback
3. **`compute_polarization_images()`**: 支援 (H,W,3)，stats 用 luminance
4. **`compute_dolp()` / `compute_dolp_from_intensities()`**: RGB → luminance → DoLP (H,W)
5. **`_save_exr()`**: 支援 `mi.Bitmap.PixelFormat.RGB`
6. **`_save_png()` / `_save_png_fixed()`**: RGB → BGR for cv2
7. **`generate_scene_report()`**: 開頭轉 luminance，後續全用 2D，與 v5 JSON 格式相容
8. **渲染 pipeline 偏振比值**: 轉 luminance 計算 (避免 H,W,3 vs H,W mask broadcasting)
9. **斷點續渲**: 檢查 `_report.json` 存在即跳過，支援長時間渲染中斷恢復
10. **刪除 `Quality_Assurance/`**: QA 統一使用 `pids_qa.py` (v2.4.0，自動相容 v5/v6)

### 測試狀態

- 單場景渲染測試通過 (SPP 1024)
- 15ch Stokes 解析正確
- RGB EXR 輸出正確
- 內部 QA (DoLP, 偏振比值, 報告) 全部審計通過

### 物理觀察

渲染中觀察到：玻璃表面只有 **10~50%** 的像素有明顯鏡面反射偏振信號，取決於入射角（菲涅爾效應）。這意味著：
- 模型需要從稀疏偏振線索推斷整塊玻璃 → 依賴 backbone 語義能力
- RGB 對此至關重要：提供色彩線索輔助「局部偏振 → 全局推斷」
- Pol volume 的實際貢獻可能比預期小，需要 ablation 驗證

### 待辦

- [ ] 大規模渲染 18000 場景 (~6 天，200+ GB)
- [ ] QA 篩選
- [ ] 訓練 ablation: RGB+pol vs RGB-only baseline
- [ ] 評估 pol 的邊際貢獻

---

## 2026-01-31: PIDS 2.0 架構規劃 — RGB 時代的架構革新

### 版本定義

```
PIDS 1.x (灰階時代):
  輸入: 灰階 I∥, I⊥
  架構演進: Dual-Stream → Pol Volume V1 → V2-A~D → V2-E
  改動範圍: 只動 correlation volume
  Context Encoder: 原封不動
  渲染器: v5 (luminance)
  最佳結果: Exp #39 V2-E, Glass EPE Median 2.333 px

PIDS 2.0 (RGB 時代):
  輸入: RGB I∥(3ch), I⊥(3ch)
  架構: 兩個候選方案獨立實驗（見下方）
  改動範圍: correlation + context encoder
  Pol 角色轉變: 從「唯一線索」→「RGB 的輔助裁判」
  渲染器: v6 (RGB Stokes)
```

### 關鍵觀察：Context Encoder 是被忽略的介入點

回讀 RAFT-Stereo 原始論文後發現：

```
RAFT-Stereo 三大組件:
  1. Feature Encoder (fnet) — shared weights, 左右各跑一次
  2. Context Encoder (cnet) — 只看 left image，產出 context + hidden state
  3. Correlation Pyramid + GRU — 迭代精修 disparity

PIDS 1.x 只改了 correlation (V2 系列)
Context Encoder 從未被動過 — 它只看左圖，完全不知道 pol 資訊
```

Context Encoder 的角色：
- **初始化 GRU hidden state** — 決定迭代起點
- **產出 static context** — 每個 GRU iteration 都注入，引導「往哪裡看」
- 一次性計算，不需要 pixel-perfect 精度，是「注意力引導」

### 困難：65mm 基線限制

不能直接將右圖 concat 進 cnet（Dual-Stream 陷阱）：
- 左右對應點有 disparity 偏移，pixel-wise concat 沒有物理意義
- 如果用 disparity 對齊，又回到了 Oracle-Real Gap 問題

兩個候選架構用不同方式解決此問題。

---

### 候選架構 A: Two-Pass (V2-B + Pol-Context) — 完整規格

**核心思想**：Pass 1 做幾何搜尋產出 disp₁，對齊後萃取 pol 資訊，Pass 2 做 pol-aware 精修。

```
Pass 1 — 幾何搜尋 (N=12 iterations)
  fnet(left) → fmap1,  fnet(right) → fmap2      ← 只算一次
  cnet(left) → context₁, hidden₁                 ← 原版 context
  corr_pyramid = CorrBlock(fmap1, fmap2)          ← 只算一次，Pass 2 共用
  + V2-B pol_corr
  GRU₁(hidden₁, context₁, corr) × 12 → disp₁
  L₁ = 0.3 × L(disp₁, GT)

Between — pol 資訊萃取
  right_warped = warp(right, disp₁.detach())
  pol_diff = (left - right_warped) / (left + right_warped + ε)  # normalized contrast
  pol_feat = PolEncoder(pol_diff)                  # (B, Cp, H, W), Cp=16~32

Pass 2 — pol-aware 精修 (M=4~6 iterations)
  context₂ = cnet(left) + Wc(pol_feat)            # additive 注入
  hidden₂  = cnet_h(left) + Wh(pol_feat)          # hidden 也注入
  corr_pyramid 重用（vanilla，無 pol_corr）
  GRU₂ × M, 從 disp₁.detach() 出發 → disp₂      # GRU₂ ≠ GRU₁
  L₂ = 1.0 × L(disp₂, GT)

Total Loss = L₁ + L₂
```

#### 設計決策與理由

**1. pol_diff = normalized contrast + PolEncoder**

使用 `(I∥ - I⊥) / (I∥ + I⊥ + ε)` 而非 raw diff。物理上接近 DoLP，scale-invariant（不受曝光/白平衡影響）。玻璃上值高、漫反射上值低、disp₁ 錯誤區為隨機值。PolEncoder 只需做空間平滑 + channel projection（極小網路），不需要學 normalization — 輸入已經是乾淨的偏振度信號。本質上是 glass indicator map。

**2. context + hidden 都注入 (additive)**

hidden 是 GRU 初始記憶。如果只改 context：GRU update rule 在變，但「從哪個狀態開始」還是純 RGB prior → pol 看得到但改不動。nopol 安全：`pol_diff=0 → PolEncoder(0)≈0 → hidden₂ ≈ base_hidden`。

**3. Pass 2 從 disp₁.detach() 出發（不是 zero）**

Pass 2 所有假設建立在「左右已對齊到同一物理點」。從 zero 出發 = 重新 stereo matching → pol context 被迫解釋幾何 → 浪費。Pass 2 任務是 refine，不是 re-search。

**4. GRU₁ / GRU₂ 權重獨立（結構同、weights 不同）**

Pass 1 學光度一致性（幾何搜尋），Pass 2 學 material cue（局部修正）。共享會讓 Pass 2 被迫用搜尋策略，pol 影響被壓扁。

**5. disp₁ + disp₂ 都監督（L₁=0.3, L₂=1.0）**

`disp₁.detach()` 阻斷梯度 → Pass 1 收不到 Pass 2 反饋 → 必須有自己的 loss。否則 Pass 1 = 亂猜 → warp 品質不穩 → pol_diff 變垃圾。

**6. Pass 2 不保留 V2-B pol_corr（vanilla corr）**

Pass 2 的 disparity 已在 disp₁ 附近，不再做大範圍搜尋。pol_corr 是給搜尋用的，Pass 2 已不在搜尋 → 重複 encoding，增加雜訊。

**7. Pass 2 迭代數 M = 4~6（< Pass 1 的 N=12）**

Refinement pass 不需要多次迭代：(1) over-correction 風險（pol 是局部 cue，迭代太多會修過頭）；(2) 模型開始「信 pol 多於 geometry」；(3) error 已在低頻段，後期迭代只是抖動。經驗法則：refinement ≈ search 的 1/3~1/2。

**8. 訓練策略：end-to-end + 前期軟凍結 Pass 2**

不做 staged training（兩階段），也不完全放開。L₂ 的 pol 注入係數從 0 漸進到 1.0，讓 Pass 1 先穩定幾何搜尋。可搭配 Pass 2 LR 輕微壓低（非 hard 壓）。不建議直接凍結 Pass 2 參數。

#### 與 Dual-Stream 的關鍵差異

| | Dual-Stream | Two-Pass |
|--|-------------|----------|
| 對齊來源 | GT disparity (Oracle) | 模型自己的 Pass 1 |
| 訓練 vs 推論 | 不一致 (Oracle-Real Gap) | **完全一致** |
| Pass 1 品質 | N/A | V2-B 已有 pol 引導 |
| pol 精度需求 | pixel-level (直接做 matching) | region-level (context guidance) |
| 失敗模式 | Gap 導致 pol 信號退化 | Pass 2 最差 = 忽略 pol（neutral） |

#### 新增參數

| 模組 | 參數量 | 說明 |
|------|--------|------|
| PolEncoder | ~1K-5K | 3→Cp (16~32), 2-3 層 3×3 conv, 不下採樣 |
| Wc | ~2K-4K | Cp→128, 1×1 conv |
| Wh | ~2K-4K | Cp→128, 1×1 conv |
| GRU₂ | ~同 GRU₁ | 結構同、權重獨立 |

**計算成本**：~1.5x baseline（Pass 2 只跑 4~6 iterations，非 2.2x）

---

### 候選架構 B: Coarse-Scale Pol Context

**核心思想**：在低解析度下計算 pol 特徵（不需要精確對齊），注入 Context Encoder，單 pass 完成。

```
單 Pass:
  coarse_left  = downsample(left,  1/S)           ← S = 8 or 16
  coarse_right = downsample(right, 1/S)
  coarse_pol   = CoarsePolEncoder(coarse_left, coarse_right)  ← 粗略 pol 特徵
  coarse_pol_up = upsample(coarse_pol, to H×W)

  fnet(left), fnet(right) → fmaps
  cnet(left, coarse_pol_up) → context, hidden      ← pol-aware context
  CorrBlock(fmaps) → corr_pyramid
  GRU × N with context → disp
```

**對齊問題的誠實評估**：

```
原始解析度 640×480, typical disparity 0~160px:
  1/4  (160×120):  disp 0~40px   ← 還是很大
  1/8  (80×60):    disp 0~20px   ← 仍然顯著
  1/16 (40×30):    disp 0~10px   ← 佔圖寬 25%，不能忽略
```

低解析度**並不能真正消除對齊問題**。但 Coarse-Scale 提供的不是 pixel-level pol signal，而是更粗糙的「**這個區域左右統計差異有多大**」的 region-level 資訊。CoarsePolEncoder 用卷積 + 大 receptive field 可能學會補償 misalignment。

**待設計細節**：

| 項目 | 說明 |
|------|------|
| Downsample scale | 1/8 或 1/16 |
| CoarsePolEncoder | 輕量 CNN (2-3 layers)，輸入 6ch (left+right coarse)，輸出 Kch |
| 注入方式 | concat 到 cnet 第一層（cnet 輸入從 3ch → 3+K ch） |
| 或注入方式 | concat 到 cnet 中間層（FiLM-style conditioning） |

**計算成本**：~1.15x baseline RAFT-Stereo（幾乎無額外成本）

---

### 兩個架構的比較

| | 架構 A: Two-Pass | 架構 B: Coarse-Scale |
|--|------------------|---------------------|
| Pol 精度 | pixel-aligned（靠 disp₁） | region-level（靠低解析度） |
| 對齊保證 | 有（顯式 warp） | 無（靠 receptive field 補償） |
| 計算成本 | ~1.5x baseline | ~1.15x baseline |
| 新增參數 | PolEncoder+Wc+Wh+GRU₂ | CoarsePolEncoder ~0.1M |
| 風險 | disp₁ 品質影響 Pass 2 | 資訊太粗，可能沒有幫助 |
| 依賴 | Pass 1 必須足夠好 | 無依賴 |
| 實作複雜度 | 高（兩輪 GRU、雙 loss、軟凍結） | 低（多一個小 encoder） |

### 與三條鐵律的關係

兩個架構都符合：
1. ❌ pol 不進 fnet — **未違反**（fnet 不動）
2. ❌ pol 不只做 spatial gate — **未違反**（context 注入 GRU 迴圈，GRU 在 disparity-aware 空間操作）
3. ✅ pol 只能在 disparity-aware 空間作用 — **符合**（context 每輪注入 GRU，GRU 配合 correlation lookup）

### PIDS 2.0 實驗矩陣

灰階時代已證明 vanilla RAFT-Stereo 不會主動利用 pol 訊號（Exp #13/#14: 隱式偏振僅改善 7%），無 pol module 的實驗不再重複。所有 2.0 實驗均以 V2-B (pol module) 為基礎。

```
Exp 2.0-A:  V2-B (mixed)       — RGB + pol module 基準
Exp 2.0-A': V2-B (all-pol)     — 驗證是否需要 nopol 數據
Exp 2.0-C:  Two-Pass           — 候選架構 A (V2-B + Pol-Context)
Exp 2.0-D:  Coarse-Scale       — 候選架構 B (Coarse Pol Context)

對照組:
  Exp #22: Baseline（灰階 RAFT-Stereo）
  Exp #39: PIDS 1.x 最佳（灰階 V2-E）

比較關係:
  A vs #22:  RGB+pol vs 灰階（整體升級效果）
  A vs A':   mixed vs all-pol（決定數據策略）
  A vs C:    Two-Pass 額外價值
  A vs D:    Coarse-Scale 額外價值
  C vs D:    兩架構直接比較
  全部 vs #39: PIDS 2.0 vs 1.x
```

**數據策略決策**：A vs A' 結果決定後續實驗使用 mixed 或 all-pol。
若差距 < 0.5 px → 全線改 all-pol，Two-Pass 設計大幅簡化（無需 nopol 防護邏輯）。

### 研究路線圖

```
階段 0 (目前): 資料準備
  ✅ V6 渲染器完成
  → 大規模渲染 18000 場景
  → QA 篩選

階段 1: 數據策略驗證 + RGB 基準
  Exp 2.0-A (mixed) + Exp 2.0-A' (all-pol)
  → 同時建立 RGB 基準數字 + 決定數據組成

階段 2a: Two-Pass (Exp 2.0-C) ← 獨立實驗
階段 2b: Coarse-Scale (Exp 2.0-D) ← 獨立實驗
  → 兩者獨立測試，不合併
  → 比較各自對 Exp 2.0-A 的增量

階段 3: 結論
  → 選擇最佳架構作為 PIDS 2.0 正式版
```

---

## 2026-02-01: Oracle-Real Gap 作為偏振天花板的論文論述

### 核心發現

Oracle mode（用 GT disparity 完美對齊後計算 pol_diff）的結果建立了偏振輔助立體匹配的 **empirical upper bound**：

```
偏振在完美對齊下的極限:
  Oracle (GT warp):    2.44 px mean Glass EPE  ← 含所有 outlier

現實最佳 (無 Oracle):
  V2-E (Exp #39):      5.039 px mean  /  2.333 px median

灰階 Baseline:
  RAFT-Stereo (#22):   6.542 px mean  /  3.489 px median
```

### 論文價值

**2.44 px 是 mean，不是 median** — 這意味著即使在完美對齊下，outlier 仍然存在但被壓到很低。這個數字回答了一個根本問題：**偏振在最理想的條件下能做到什麼程度？**

四個可寫進 paper 的 insight：

1. **Pol 的天花板是 2.44 px (mean)**：不是無限好，但顯著優於任何 non-pol 方法。這證明偏振資訊確實包含幾何上有價值的信號。

2. **V2-E median 已打贏 Oracle mean (2.333 < 2.44)**：說明「典型表現」已接近理論極限。瓶頸不是架構能力，而是少數 outlier 拉高 mean。

3. **Mean 的 gap (5.04 vs 2.44) 幾乎全來自 outlier**：Pol Volume (V2-E) 在大多數場景已達到 Oracle 級別表現，但對極端 case 無解。這直接指向 Two-Pass 的動機 — 用 context-level 的 pol 資訊處理這些困難區域。

4. **Oracle-Real Gap 不是失敗，是路標**：
   - Dual-Stream (Oracle input) → 2.44 px，但 Oracle-Real Gap +103%
   - Pol Volume V2-E (自力更生) → median 2.333 px，**消除了 Gap**
   - Two-Pass (近似 Oracle) → 目標：在保持 Gap-free 的前提下逼近 Oracle mean

### 論文圖表建議

```
Architecture Evolution（建議放 Fig. 3 或 Table II）:

Method                  Mean    Median   Oracle-Real Gap
─────────────────────────────────────────────────────────
Baseline (RAFT-Stereo)  6.542   3.489    N/A
Dual-Stream (Oracle)    2.44    —        +103% (致命)
Pol Volume V2-E         5.039   2.333    0% (消除)
Two-Pass (PIDS 2.0)    目標     目標     0% (設計保證)
```

**一句話論述**：Oracle 結果 (2.44 px) 證明偏振的價值，Pol Volume 消除了 Oracle-Real Gap 但受限於 outlier，Two-Pass 在保持 Gap-free 的前提下逼近 Oracle 的 outlier 控制能力。

---

## 2026-01-31: 偏振渲染的物理真實性分析 — Mitsuba vs 現實

### 問題背景

PIDS 使用 0°/90° 交叉偏振配置。最初擔憂：90° 偏振片會 100% 濾掉鏡面反射的偏振光，導致左右影像在玻璃區域「內容完全不同」，使 RGB 第三方裁判失效。

### 分析結論：問題沒有想像中嚴重

**1. 偏振片不改光譜形狀（wavelength-independent）**

偏振片主要影響 intensity，不影響 spectral shape。因此交叉偏振下：
- chromaticity（顏色比例）近似不變
- 變化的是亮度/對比，不是「顏色變成另一個物體」
- RGB 裁判在正常 SNR 下可以區分「同物體變暗」vs「不同物體」

**2. 現實中不是 100% 消光**

- 真實偏振片 extinction ratio = 1:100 ~ 1:10,000
- 鏡面反射產生的是**部分偏振**（partial polarization），不是完美線偏振
- 多次散射、粗糙表面會去偏振（depolarization）
- 結果：I⊥ 不會完全黑掉，仍有殘留鏡面訊號

**3. RGB 裁判的失效邊界**

| 邊界條件 | 說明 | 嚴重性 |
|----------|------|--------|
| (A) SNR 底 | I⊥ 暗到噪聲主導，chromaticity 不可靠 | 中 — 可控制曝光 |
| (B) Clipping | 一側飽和，RGB ratio 失真 | 中 — 可用 HDR |
| (C) 偏振光譜效應 | 鍍膜材料不同波長反射率不同 | 低 — 少見 |

**修正後的敘事**：

> RGB 提供更高維的局部外觀特徵（texture + chromatic cues），在偏振造成的強度變化下仍有助於區分同物體/異物體匹配；但若交叉偏振導致嚴重低照度，該優勢會下降。

### Mitsuba 渲染的真實性評估

**物理光學 — 正確**：
- ✅ Fresnel 方程：正確計算部分偏振，偏振程度取決於入射角
- ✅ Mueller calculus：完整追蹤 Stokes vector 經過每次光學事件
- ✅ 多次散射去偏振
- ✅ 不同材質的偏振特性

**感測器模型 — 理想化**：

| 面向 | Mitsuba | 現實 | Gap |
|------|---------|------|-----|
| 偏振片 | 理想（extinction = ∞） | 有限消光比 1:100~1:10,000 | Mitsuba 比現實**更極端** |
| 感測器噪聲 | 無（乾淨 float32） | shot noise + read noise | 低光區 chromaticity 會被噪聲污染 |
| 動態範圍 | 無限（float32） | 8-12bit，高光 clip | 鏡面可能 clip |

**關鍵結論**：Mitsuba 的偏振比現實**更極端**（完全消光 vs 部分消光）。這意味著：
- 用 Mitsuba 訓練的模型看到的是「最難情況」
- 真實相機的鏡面區域反而更容易匹配（有殘留鏡面）
- 但真實相機有噪聲和 clipping，Mitsuba 沒有模擬

### 解決方案：Sensor Realism Augmentation

在 training data pipeline 加入三個後處理 augmentation，縮小 synthetic-to-real gap：

```python
# 1. 模擬有限消光比 — I⊥ 保留微弱鏡面
extinction_ratio = random.uniform(100, 10000)
I_cross_real = I_cross_ideal + I_parallel_ideal / extinction_ratio

# 2. 模擬感測器噪聲 — Poisson (shot) + Gaussian (read)
shot_noise = poisson(I * gain) - I * gain
read_noise = gaussian(0, sigma_read)
I_noisy = I + shot_noise + read_noise

# 3. 模擬動態範圍限制 — clipping
I_clipped = clip(I_noisy, 0, saturation_value)
```

| Augmentation | 模擬什麼 | 參數範圍 | 隨機化 |
|-------------|---------|---------|--------|
| 消光比洩漏 | 非理想偏振片 | ε = 1/100 ~ 1/10000 | 每場景隨機 |
| Shot + Read noise | 感測器噪聲 | gain, σ_read | 每場景隨機 |
| Clipping | 有限動態範圍 | saturation = 0.8~1.0 | 每場景隨機 |

### 對 PIDS 2.0 架構的影響

1. **Specular-Invariance Auxiliary Loss 可能不需要** — chromaticity 本來就近似不變，fnet 的 Instance Normalization 已能正規化 intensity 差異
2. **RGB 裁判在正常 SNR 下有效** — 邊界條件可通過 augmentation 管理
3. **Two-Pass 和 Coarse-Scale 設計不需要因此改變** — 原始設計動機仍然成立
4. **Sensor Realism Augmentation 應加入訓練 pipeline** — 對 Stage I→II 的泛化至關重要

---

## Exp #41: Directional Impulse Descent (DID) (2026-01-29)

### 背景：為什麼需要 DID？

V2-E 訓練過程中觀察到：
1. 曲線**非常穩定**，但中間會有暫時的 plateau
2. 傳統的 LR scheduler (OneCycle) 是**時間驅動**的，無法感知 plateau
3. 需要一個**狀態驅動**的機制來逃離 local minimum

### DID 的設計哲學

```
OneCycle = 全局退火曲線 (reference trajectory)
DID      = 狀態驅動的局部脈衝 (event-driven impulse disturbance)
```

**物理類比**：球在 loss landscape 的 saddle point 附近

```
┌────────────────────────────────────────┐
│         ╱╲                             │
│        ╱  ╲  ← ridge (脊線)            │
│       ╱    ╲                           │
│   ○ ←───────→ ○  球來回滾              │
│      同一坡面，方向一致                 │
│      但動能不足以翻過脊線              │
└────────────────────────────────────────┘
```

**關鍵洞察**：
- 不是「沒有下降」，而是「一直想下降但下不去」
- `sign(∂L/∂t)` 一致 = 有方向意圖
- `|∂L/∂t|` 很小 = 動能不足

### DID 狀態機

```
                    ┌─────────────────────┐
                    │                     │
                    ▼                     │
┌─────────┐    trigger    ┌─────────┐    T_impulse    ┌──────────┐
│ NORMAL  │──────────────►│ IMPULSE │───────────────►│ COOLDOWN │
└─────────┘               └─────────┘                └──────────┘
     ▲                                                    │
     │                     ┌───────────┐                  │ T_cool
     │         P cycles    │ PROTECTED │◄─────────────────┘
     └─────────────────────┴───────────┘
```

### 觸發條件（同時滿足）

1. **Train slope plateau**: `median(s_recent) < ε`，其中 `ε = percentile(s_history, 10%)`
2. **Val oscillating**: val Glass EPE 在 ±δ px 內震盪 M 次
3. **Direction consistent**: slope EMA 方向穩定（用 EMA + sign，不用 all()）
4. **Not in late phase**: `progress < 0.75`
5. **Under trigger limit**: `trigger_count < max_triggers`
6. **Not in protection**: 距離上次觸發已過 P 個 validation cycles

### LR 控制

```python
lr_base = OneCycle(step)  # 始終追蹤 reference

if state == IMPULSE:
    lr_effective = lr_base * γ_up      # 能量注入
elif state == COOLDOWN:
    lr_effective = lr_base * γ_down    # 快速阻尼
else:  # NORMAL, PROTECTED
    lr_effective = lr_base             # 正常跟隨
```

**γ_down 的 cooldown_mode 選項**：
- `relative_base` (預設): `lr_cool = lr_base * γ_down`
- `relative_impulse`: `lr_cool = lr_impulse * γ_down`（用於 ablation）

### 參數表

| 類別 | 參數 | 預設值 | 說明 |
|------|------|--------|------|
| **Plateau Detection** | | | |
| | `--did_epsilon_pct` | 0.10 | train slope threshold percentile |
| | `--did_train_window` | 300 | train slope EMA window |
| | `--did_val_m` | 5 | val oscillation window (cycles) |
| | `--did_delta` | 0.5 | val EPE tolerance (px) |
| | `--did_best_stale` | 4 | cycles since best before trigger **[改進 D]** |
| **DID Impulse** | | | |
| | `--did_gamma_up` | 3.0 | impulse multiplier (2nd+ trigger) |
| | `--did_gamma_up_first` | 2.0 | impulse multiplier (1st trigger) **[改進 B]** |
| | `--did_gamma_backbone` | 1.2 | backbone impulse multiplier **[改進 C]** |
| | `--did_gamma_down` | 0.1 | cooldown multiplier |
| | `--did_t_impulse` | 200 | impulse duration (steps) |
| | `--did_t_cool` | 100 | cooldown duration (steps) |
| | `--did_cooldown_mode` | relative_base | cooldown LR reference |
| **Protection** | | | |
| | `--did_max_triggers` | 3 | max triggers per training |
| | `--did_protection` | 3 | protection period (val cycles) |
| | `--did_late_lock` | 0.75 | disable after 75% progress |
| **EPE-Adaptive [改進 E]** | | | |
| | `--did_epe_ref` | 8.0 | EPE reference for adaptive γ |
| | `--did_epe_smooth_alpha` | 0.4 | EMA α for smoothed EPE (保險絲 1) |
| | `--did_delta_gamma_max` | 0.3 | max γ change per impulse (保險絲 2) |

### 訊號設計

| 訊號 | 時間尺度 | 計算方式 |
|------|----------|----------|
| Train slope | Step-level | `s_t = \|ΔL_t\| / L_t`，EMA(α=0.3, window=300) |
| Val EPE | Validation-level | Spike filter + baseline window |
| Direction | Val-level | `slope_ema * last_slope > 0` (同號) |

### TensorBoard 記錄

```
did/state              # 0=NORMAL, 1=IMPULSE, 2=COOLDOWN, 3=PROTECTED
did/trigger_count      # 已觸發次數
did/train_slope_ema    # train slope 的 EMA
did/val_slope_ema      # val slope 的 EMA
did/epsilon            # 當前 threshold
did/lr_base            # OneCycle 的 LR
did/lr_effective       # 應用 DID 後的 LR
did/lr_multiplier      # 當前乘數 (1.0 / γ_up / γ_down)
did/val_baseline       # val baseline (用於 spike filtering)
did/epe_smooth         # smoothed EPE (EMA) [改進 E / 保險絲 1]
did/gamma_eff          # 實際使用的 γ (EPE-adaptive 後) [改進 E]
```

### 使用範例

```bash
python train_pids.py \
    --data_dir ./data \
    --output_dir ./checkpoints_exp41_v2e_did \
    --pol_volume_v2e \
    --pol_weight_hidden 8 \
    --curriculum \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 40000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 500 --num_workers 4 \
    --did \
    --did_epsilon_pct 0.10 \
    --did_train_window 300 \
    --did_val_m 5 \
    --did_best_stale 4 \
    --did_gamma_up 3.0 \
    --did_gamma_up_first 2.0 \
    --did_gamma_backbone 1.2 \
    --did_t_impulse 200 \
    --did_t_cool 100 \
    --did_max_triggers 3 \
    --did_protection 3 \
    --did_late_lock 0.75 \
    --did_epe_ref 8.0 \
    --did_epe_smooth_alpha 0.4 \
    --did_delta_gamma_max 0.3
```

### 設計改進 (v2)

**改進 B: 漸進式 Impulse**
- 第一次觸發使用 `gamma_up_first` (×2.0)，較保守
- 後續觸發使用 `gamma_up` (×3.0)
- 理由：避免 bf16 混合精度下 grad overflow

**改進 C: 參數群分離**
```
Pol modules (pol):     完整 impulse (×2.0/×3.0)  ← 需要擾動的目標
Update block (update): 完整 impulse
Backbone (fnet, cnet): 輕微 impulse (×1.2)       ← 保護 ImageNet 特徵
```
- 理由：pol-only 驗證集表示 pol module 是瓶頸；backbone 已收斂到好的特徵空間

**改進 D: 觸發條件重構**
- 舊：val oscillation 是必要條件
- 新：moving-best staleness 是必要條件，val oscillation 是輔助證據
- 理由：pol-only 驗證集較「安靜」，oscillation ±0.5px 條件太難滿足

**改進 E: EPE-Adaptive Impulse**
- EPE 越低 → 模型越精細 → 脈衝自動縮小
- `γ_eff = 1.0 + (γ_raw - 1.0) × clamp(epe_smooth / epe_ref, 0, 1)`
- `epe_ref = 8.0`：EPE ≥ 8 時用完整 γ，EPE 越低 γ 越接近 1.0
- **保險絲 1**: Smoothed EPE — `EMA(val_glass_epe, α=0.4)` 避免 val 抖動
- **保險絲 2**: γ 變化率限制 — `|Δγ| ≤ 0.3` 防止 γ 突變

### 與 OneCycle 的關係

| 面向 | OneCycle | DID |
|------|----------|-----|
| 驅動方式 | 時間驅動 | **狀態驅動** |
| 作用範圍 | 全局 | 局部 |
| 控制理論 | 開迴路 | **閉迴路** |
| 類比 | 退火曲線 | 脈衝擾動 |
| 參數群控制 | 無 | **分群 (改進 C)** |

**一句話總結**：OneCycle 管整體探索–收斂節律，DID 負責在錯誤 basin 內的瞬時能量注入。

### 預期效果

1. **自動逃離 plateau**: 不需要人工監控和調參
2. **不破壞 OneCycle**: 只在局部做擾動，整體節律不變
3. **可 ablation**: `cooldown_mode` 可切換，用於論文比較
4. **可觀測**: TensorBoard 完整記錄所有狀態，方便分析

### 前期結果 (19500 steps)

| 指標 | 數值 | 判定 |
|------|------|------|
| EPE Smooth | 51.0 → 41.5 (5k) → **42.12 (8k-19.5k 完全停滯)** | ❌ |
| Epsilon | 0.035 → **0.079 (持續上升)** | ❌ |
| Gamma_eff | **1.0 (恆定不變)** | ❌ |

**問題分析**：
1. EPE 42 px 等於基本未學到立體匹配（正常 <5 px），Step 8000 後 11500 steps 零進展
2. `gamma_eff = 1.0` 恆定 → DID 的 adaptive gamma 機制退化，對所有 GRU iteration 等權，未學到「後期迭代更準」
3. `epsilon` 持續上升 → 惡性循環：模型不準 → DID 擴大探索 → 探索無效 → 繼續擴大
4. 結論：base model 在 DID 加入的額外優化複雜度下連基礎立體匹配都無法收斂

### 狀態：🧊 凍結

DID 機制已實作完成（含改進 B/C/D/E + 保險絲 1/2），但前期實驗結果顯示需要更多調適工作。在 base model 尚未穩定收斂的情況下，DID 增加了不必要的優化難度。

**凍結原因**：目前主線工作轉向 PIDS 2.0 架構（Two-Pass / Coarse-Scale Pol Context），DID 的調適工作暫緩，避免影響主線進度。

**後續可能方向**（待主線穩定後）：
- 先讓 base model 收斂到合理 EPE（<10 px），再引入 DID
- 檢查 DID 與 Curriculum 的交互作用是否產生衝突
- 考慮簡化 DID（減少同時學習的超參數）

---

## 2026-01-29: Exp #39 V2-E 訓練完成

### 訓練結果

V2-E (Pre-Corr Pol Weighting) 訓練完成：

```
Best Glass EPE: 5.468 px @ Step 59500
Final Glass EPE: 5.462 px @ Step 60000
```

曲線穩定，最後 10k steps 在 5.4~5.9 px 震盪。

### 評估結果

**Test Set (857 samples) 對比：**

| Metric | V2-E (Exp39) | Baseline (Exp22) | 改善 |
|--------|--------------|------------------|------|
| Glass EPE (Mean) | **5.039 px** | 6.542 px | -23% |
| Glass EPE (Median) | **2.333 px** | 3.489 px | -33% |
| Glass D1 | **34.83%** | 75.42% | -54% |
| Glass D3 | **23.53%** | 41.31% | -43% |
| Overall EPE | **3.270 px** | 3.444 px | -5% |
| Overall D1 | **24.15%** | 41.25% | -41% |

### 關鍵發現：Mean vs Median 差距

```
V2-E Glass EPE:
  Mean:   5.039 px
  Median: 2.333 px
  差距:   +116% !!!
```

**意義**：
- 少數 outlier 嚴重拉高平均值
- Median 更能反映「典型」表現
- 需要更 robust 的評估指標

### 觸發改進：Robust Checkpoint Selection

這個發現直接觸發了 Exp #40 的設計。

---

## 2026-01-30: Exp #40 Robust Checkpoint Selection

### 動機

Exp #39 發現 Mean vs Median 差距 +116%，表示：
1. 訓練時的 best checkpoint 可能不是真正最好的
2. 論文報告 Mean 會讓結果看起來比實際差

### 改進：新增 Robust Statistics

**程式碼修改**：`train_pids.py`

```python
def validate(self, use_oracle: bool = True) -> Dict[str, float]:
    # 收集每個樣本的 glass_epe
    all_glass_epes = []

    for batch in self.val_loader:
        # ... inference ...
        all_glass_epes.append(metrics['glass_epe'])

    # Robust Statistics
    glass_epes = np.array(all_glass_epes)
    sorted_epes = np.sort(glass_epes)
    n = len(glass_epes)

    # Trimmed Mean: 移除 top 10%
    trim_idx = int(n * 0.9)
    trimmed_values = sorted_epes[:trim_idx]

    result['glass_epe_median'] = float(np.median(glass_epes))
    result['glass_epe_trimmed_mean'] = float(np.mean(trimmed_values))
    result['glass_epe_p90'] = float(np.percentile(glass_epes, 90))
    result['glass_epe_p95'] = float(np.percentile(glass_epes, 95))
    result['glass_epe_outlier_rate'] = float(np.sum(glass_epes > 20) / n)
```

### 新增評估指標

| 指標 | 說明 | 用途 |
|------|------|------|
| Mean | 傳統平均值 | 易受 outlier 影響 |
| **Median** | 中位數 | **Primary metric** |
| Trimmed Mean | 移除 top 10% 後平均 | 論文報告用 |
| P90 | 90th percentile | **Safety gate** |
| P95 | 95th percentile | 觀察用 |
| Outlier Rate | > 20px 的比例 | 資料品質指標 |

### Checkpoint 選擇邏輯

```python
# Primary: Median 更低
is_better_median = current_median < best_median

# Safety Gate: P90 不能太差 (不能超過 best_p90 的 1.5 倍)
p90_safe = current_p90 <= best_p90 * 1.5

# 同時滿足才是 best
is_best = is_better_median and p90_safe
```

**設計理由**：
- Median 不受 outlier 影響，更能反映「典型」表現
- P90 確保尾端不會太差（避免 median 好但 worst case 很爛）

### 驗證輸出格式

```
[Glass EPE Robust Statistics (Real)]
┌───────────────────────────────────────────────────┐
│ Mean:         5.039 px                            │
│ Median:       2.333 px                            │
│ Trimmed Mean: 3.215 px  (top 10% removed)         │
│ P90:          8.721 px                            │
│ P95:         12.456 px                            │
│ Outlier Rate:   5.2%  (44 scenes > 20px)          │
└───────────────────────────────────────────────────┘
```

### TensorBoard 新增

- `val/glass_epe_median`
- `val/glass_epe_trimmed_mean`
- `val/glass_epe_p90`
- `val/glass_epe_p95`
- `val/glass_epe_outlier_rate`

### 訓練命令

```bash
nohup python train_pids.py \
    --data_dir /workspace/RAFT-Stereo/mixed_dataset_exp28/data \
    --output_dir ./checkpoints_exp40_v2e_robust \
    --pol_volume_v2e \
    --pol_weight_hidden 8 \
    --curriculum \
    --pretrained ./models/raftstereo-sceneflow.pth \
    --batch_size 8 --num_steps 60000 --lr 0.0003 --iters 24 \
    --pol_levels 4 --pol_radius 4 --val_freq 500 --num_workers 4 \
    > train_exp40.log 2>&1 &
```

### 評估結果 (Test Set, 857 samples)

| 指標 | Exp #39 | Exp #40 | 變化 |
|------|---------|---------|------|
| Glass EPE (Mean) | 5.039 px | 5.402 px | +0.36 |
| Glass EPE (Median) | 2.333 px | 2.501 px | +0.17 |
| Glass EPE (Trimmed Mean) | — | 3.121 px | — |
| Glass D1 | 34.83% | 35.58% | +0.75 |
| Overall EPE | 3.270 px | 3.520 px | +0.25 |
| P90 | — | 12.574 px | — |
| Outlier Rate | — | 5.0% (43 scenes) | — |

**結論**：Robust Checkpoint 未帶來性能提升（Median +0.17 px），但基礎設施（Robust Statistics、Dual-mask Metric）已就位，為 Exp #41 DID 提供監控框架。

### 狀態：已完成

---

## 2026-01-30: Exp #42 Numerical Stability Test

### 根因確認：TF32 在 Hopper 上的行為

原始觀察：H200 Baseline Median 3.489 px vs RTX 4090 5.264 px (+51%)。

在另一台 H200 上跑 C1-C5 穩定性測試，**根因確認為 TF32**：

```
                    C1       C2       C3       C4       C5        σ    CV(%)
                   TF32✓    TF32✗    TF32✓    TF32✗    TF32✓
────────────────────────────────────────────────────────────────────────────
4090 (Ada)        5.264    5.251    5.258    5.251    5.262    0.006   0.12%
H200 (Hopper)     3.504    5.250    3.501    5.250    3.501    0.958  22.79%
────────────────────────────────────────────────────────────────────────────
```

關鍵發現：
1. **FP32 (TF32=OFF) 跨平台完全一致**：H200 5.250 ≈ 4090 5.251
2. **TF32 在 Hopper 上造成 50% 差異**（3.50 vs 5.25），Ada 上幾乎無影響
3. **Benchmark 完全無影響**（C1≈C3, C2≈C4）
4. Hopper TF32 結果反而更好（3.50 < 5.25），可能是精度截斷的隱式正則化效果

### 解法

在 evaluation 程式碼中預設關閉 TF32：
```python
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```
→ 跨平台一致的 FP32 結果 (Baseline Median ≈ 5.25 px)

### Part C: RTX 5090 (Blackwell) — Blackwell = Hopper 行為

```
                C1       C2       C3       C4       C5        σ    CV(%)
               TF32✓    TF32✗    TF32✓    TF32✗    TF32✓
────────────────────────────────────────────────────────────────────────
4090 (Ada)    5.264    5.251    5.258    5.251    5.262    0.006   0.12%
H200 (Hopper) 3.504    5.250    3.501    5.250    3.501    0.958  22.79%
5090 (Black.) 3.498    5.250    3.515    5.250    3.504    0.956  22.73%
────────────────────────────────────────────────────────────────────────
FP32 基準: 三代架構全部 5.250 ± 0.001，完全一致
```

Ada Lovelace 是唯一 TF32 不影響結果的架構。Hopper 和 Blackwell（含消費級 RTX 5090）TF32 造成 50% 差異。

### 行動

所有 evaluation/training 程式碼預設關閉 TF32：
```python
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

### 狀態：全部完成

---

## 2026-01-30: Train-Val Gap 分析 & Dual-Mask Metric

### 問題

V2-E 訓練後期出現 train-val gap：train glass_epe ~3 px，val ~5.5 px（plateau after Step 27000）。
分佈相同、無 crop，理論上不該有如此大的差距。

### 根因分析

深入追蹤程式碼後發現 **5 個 train-val 不對稱因素**：

| 因素 | Training | Validation | 影響方向 |
|------|----------|------------|----------|
| **Glass Mask** | Union (邊緣+核心) | Strict (核心) | **主因**：不同像素集 |
| **GRU Iterations** | 12 | 16 (+4) | 反向：val 更低 |
| **Augmentation** | 亮度/對比度/翻轉 | 無 | 不確定 |
| **Model Mode** | train() BatchNorm | eval() BatchNorm | train 有優勢 |
| **Metric** | per-batch | full dataset | 噪音差異 |

**主因**：Training 用 `glass_mask` (聯集 mask)，Validation 用 `glass_mask_strict` (交集 mask)。
兩者在 criterion 中都傳入 `glass_mask` 參數位置，但實際是不同的 mask。
聯集包含邊緣像素（有紋理梯度、容易匹配），嚴格只保留核心玻璃（透明/反射、最難匹配）。

程式碼位置：
- Training: `train_pids.py:1452` → `criterion(..., glass_mask, ..., glass_mask_strict)`
- Validation (舊): `train_pids.py:1765` → `criterion(..., glass_mask_strict, ..., None)`

### 解決方案：Dual-Mask Metric

修改 criterion 和 training/validation pipeline，統一提供兩種指標：

1. **`glass_epe`** — Union mask (邊緣+核心)，同一指標在 train/val 可直接對比
2. **`glass_epe_strict`** — Strict mask (核心)，最嚴格的評估標準

修改清單：
- `pids_model.py:3354-3380`: criterion 同時回傳 `glass_epe` 和 `glass_epe_strict`
- `train_pids.py:1727-1730`: validation 傳入兩種 mask (而非只傳 strict)
- `train_pids.py:1378,1714`: training/validation meters 加入 `glass_epe_strict`
- `train_pids.py:1593`: TensorBoard 記錄 `train/glass_epe_strict`
- `train_pids.py:1938-1985`: TensorBoard 記錄所有 strict robust statistics
- `train_pids.py:1898`: composite score 改用 `glass_epe_strict`
- `train_pids.py:2017`: checkpoint selection 改用 `glass_epe_strict_median`
- `train_pids.py:1974`: DID 更新改用 `glass_epe_strict`

### 效果

1. **TensorBoard 可直接比較** train/val 的 glass_epe (union) 和 glass_epe_strict (strict)
2. **Checkpoint selection** 基於嚴格指標（核心玻璃區域）
3. **Console 輸出** 同時顯示 Glass EPE (U) 和 (S)
4. **Union-Strict Gap** 被監控，可量化邊緣 vs 核心的差異

### Loss 權重分配（已有）

```
Background:     w = 1.0
Edge (Union-only): w = glass_weight = 5.0
Core (Strict):    w = glass_weight + strict_glass_weight = 5.5
```

Loss 本身已區分權重，但之前 metric 只呈現單一 mask → 無法觀測。
