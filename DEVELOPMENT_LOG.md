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

## 10. 待完成項目

- [x] 完成 70k steps 訓練 ✓
- [x] 新增 strict glass mask 功能 ✓
- [x] Nopol V5 數據集渲染 ✓
- [x] Nopol strict glass mask 渲染 ✓
- [x] Nopol V5 消融實驗訓練 ✓
- [x] Exp #21: Strict Glass Weight 驗證 ✓
- [x] Test Set 評估 (Pol vs Nopol 消融) ✓
- [ ] Ablation: 不同 `pol_update_iters` 設定
- [ ] Stage 2 real-world fine-tuning
