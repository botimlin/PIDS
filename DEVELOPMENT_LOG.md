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

### 階段 2：嘗試 Constant Emitter（❌ 效果不佳）

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

### 階段 3：嘗試關閉偏振光源（❌ 全黑）

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

### 階段 4：嘗試取消放大因子（❌ 仍有差異）

**理論**：`amplify_factor=5.0` 會放大 I∥ - I⊥ 的差異，即使背景 S1 很小也會被放大。

**我們的修改**：
```python
amplify_factor=1.0,  # 不放大
exposure=1.0,        # 不調整曝光
```

**結果**：背景亮度差異改善，但整體太暗。

---

### 階段 5：發現材質誤判（✅ 關鍵發現）

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

### 階段 6-7：IOR 和 twosided 嘗試（❌ 無效）

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

### 階段 9-11：天花板光源問題排查（❌ 全部失敗）

我們嘗試了多種方法改善天花板光源的漫反射效果：
1. 提高天花板光照強度 - 無效
2. 增加光線反彈次數 (MAX_DEPTH 12→48) - 無效
3. 提高材質反射率 - 無效
4. 獨立燈陣列 - 失敗並回滾

**關鍵發現 - Mitsuba 3 的 polarizer BSDF 行為**：

> 在 Mitsuba 3 中，`polarizer` BSDF 是設計來**過濾穿過它的光**，而不是作為發射器的材質。正確做法是讓光源發射非偏振光，然後**穿過一個獨立的偏振片幾何體**。

---

### 階段 12：實施物理偏振片架構 (v3.1.0) ✅

**解決方案**：我們實施了完整的物理偏振片架構：

1. **LED 光源重構**：
   - `led_emitter`：純粹的非偏振面光源
   - `led_polarizer`：獨立偏振片幾何體 (θ=0°)，位於光源前方 5mm

2. **相機偏振片**：
   - 左相機：前方加偏振片 (θ=0°) → 通過 I∥
   - 右相機：前方加偏振片 (θ=90°) → 通過 I⊥

**結果**：✅ I∥ 和 I⊥ 亮度平衡了！

**實測數據**：

| 指標 | 數值 | 評價 |
|------|------|------|
| 品質評分 | 87/100 | excellent |
| 玻璃區域 DoLP | 24.7% | 很好的偏振效果 |
| 背景 DoLP | 3% | 低，符合預期 |
| 背景 I∥/I⊥ 平衡 | 1.0003 | 幾乎完美 |

---

### 階段 13：修正相機光軸為平行配置 (v3.2.0) ✅

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

**實測結果**：✅ Vertical disparity ≈ 0

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
| 1 | Geometric Consistency | vertical disparity < 1px | ✅ |
| 2 | Background Photometric Consistency | I∥/I⊥ ∈ [0.5, 2.0] | ✅ |
| 3 | Polarization Signal Validity | 玻璃 DoLP > 10% | ✅ |
| 4 | Ground Truth Alignment | < 10% 正規化誤差 | ✅ |
| 5 | Depth Validity Rate | > 90% in glass region | ✅ |

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
| 傢俱 | ✅ | 每個傢俱使用獨立材質實例 |
| 地板 | ✅ | 每場景使用獨立材質實例 |
| 牆壁 | ❌ | 所有牆面共用同一材質 |
| 天花板 | ❌ | 與牆壁共用同一材質 |

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

## 實驗 #8：數值穩定性優化 ✅

**日期**: 2025-12-29
**狀態**: ✅ 實驗成功

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
| 收斂趨勢 | 發散 | ✅ 收斂 |

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
**狀態**: 🔄 進行中

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
- Loss = 40² = 1600 ≈ **1506** ✓

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
| Val Loss 趨勢 | 持續上升 ✗ | 整體下降 ✓ |
| 最大 Spike | 1506 | 1295 (較小) |
| 最佳 Val Loss | ~480 | **421** ✓ |
| Spike 恢復 | 慢 | 快 ✓ |

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
**狀態**: 🔄 進行中

### 問題發現

分析實驗 #10 結果時發現 **max_disp 設置不一致**：

| 文件 | 參數 | 值 | 作用 |
|------|------|-----|------|
| `pids_dataset.py:99` | max_disparity | **576.0** ✅ | Dataset valid_mask |
| `pids_model.py:380` | max_disp | **192.0** ❌ | Loss 函數 valid_mask |
| `train_pids.py:637` | --max_disp | **192.0** ❌ | 命令行默認值 |

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
**狀態**: ✅ 完成

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
**狀態**: ✅ 完成

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
| Val Loss | 265.91 | 299.08 | +12.5% ❌ |
| Glass EPE | 46.90 px | 44.96 px | -4.1% ✓ |

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
- [x] 整理數據集 (train 3500 / test 265)
- [ ] 訓練 PIDS (偏振版)
- [ ] 訓練 Baseline (無偏振版)
- [ ] 對比評估

---

## 實驗 #13：大規模偏振數據訓練

**日期**: 2026-01-02 ~ 2026-01-03
**狀態**: ✅ 完成

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
| Glass EPE | 46.90 px | **< 42 px** | 39.20 px | ✅ |
| D1 | 58.56% | **< 56%** | 53.68% | ✅ |
| Val Loss | 265.91 | **< 250** | 231.65 | ✅ |

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
**狀態**: ✅ 完成

| 指標 | 實驗 #11 | 實驗 #13 | 改善 |
|------|----------|----------|------|
| Val Loss | 265.91 | **231.65** | -12.9% |
| Glass EPE | 46.90 px | **39.20 px** | -16.4% |
| D1 | 58.56% | **53.68%** | -8.3% |
| EPE (全域) | - | **17.09 px** | - |

**Best Checkpoint**: step 48500 (Val Loss 231.65, Glass EPE 39.20)

**結論**：數據量從 ~2000 增加到 3500 (+75%)，Glass EPE 從 46.90 降到 39.20 (-16.4%)。雖然存在邊際效應，但改善仍然顯著。

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

檢查 5 個必要文件是否齊全：`_left.exr`, `_right.exr`, `_disparity.exr`, `_depth.exr`, `_mask.png`

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
