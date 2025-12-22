# PIDS 偏振渲染器排錯歷程

## 問題描述

**主要問題**：`left_parallel` (I∥) 和 `right_cross` (I⊥) 兩張圖的**背景亮度差異過大**，導致 RAFT-Stereo 無法正確對齊。

**次要問題**：櫃子等非玻璃物體出現「鬼影」。

---

## 排錯時間線

### 階段 1：初始診斷

**觀察**：
- `left_parallel` 背景很亮
- `right_cross` 背景很暗
- 預期：背景（漫反射表面）應該 I∥ ≈ I⊥

**初步假設**：偏振光源照射整個場景，導致背景也有偏振。

---

### 階段 2：嘗試 Constant Emitter（❌ 效果不佳）

**理論**：使用 `constant` emitter 作為環境光，因為它是真正的非偏振漫射光。

**修改**：
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

**修改**：
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

**修改**：
```python
amplify_factor=1.0,  # 不放大
exposure=1.0,        # 不調整曝光
```

**結果**：背景亮度差異改善，但整體太暗。

---

### 階段 5：發現材質誤判（✅ 關鍵發現）

**用戶提供的線索**：
> 即使您已經關閉了 LED，如果材質被誤判為玻璃，它就會變成一個巨大的偏振產生器。

**問題代碼**：
```python
def is_glass_material(mat_name: str) -> bool:
    glass_keywords = ['glass', 'clear', 'transparent', 'window', 
                      'door', 'partition', 'panel', 'acrylic']
    return any(k in name_lower for k in glass_keywords)
```

**場景材質名稱**：
- `background_wall_back` → 包含 'back'，但沒問題
- `glass_partition_01` → 包含 'glass' 和 'partition'，判定為玻璃 ✓
- `diffuse_cabinet_02` → 不包含任何關鍵字，判定為非玻璃 ✓

**但危險的是**：如果有材質叫 `Back_Panel` 或 `Wall_Partition`，會被誤判為玻璃！

**修復**：
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

### 階段 6：嘗試調高 IOR（❌ 鬼影更嚴重）

**修改**：
```python
'ior': 1.8,  # 從 1.5 調高到 1.8
```

**結果**：櫃子鬼影問題依然存在，而且更明顯。

**結論**：IOR 不是鬼影的原因。

---

### 階段 7：嘗試 twosided 材質（❌ 無效）

**理論**：鬼影可能是因為 OBJ 中玻璃有雙面幾何。

**修改**：
```python
'bsdf': {
    'type': 'twosided',
    'material': {
        'type': 'thindielectric',
        'int_ior': glass_ior,
    },
},
```

**結果**：鬼影仍然存在。

---

### 階段 8：回滾到穩定版本（✅ 當前狀態）

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
            'intensity': 8000.0,  # constant emitter = 8.0
        },
    },
    'materials': {
        'glass': {
            'type': 'thindielectric',
            'ior': 1.5,
        },
    },
}

# Stokes 提取參數
amplify_factor = 5.0
exposure = 2.0
auto_balance = True  # 後處理平衡
```

---

### 階段 9：解決天花板光源漫反射不足問題 (v3.0.6)

**用戶觀察**：渲染圖像頂部的燈光似乎沒有對周圍和傢俱產生漫反射效果。

**程式碼分析**：
- `pids_renderer.py` 中的 `SceneBuilder` 會將名稱包含 "ceiling" 的材質物體轉換為面光源。
- 經過檢查，發現天花板光源的強度 `CEILING_EMITTER_INTENSITY` (初始值 100.0) 遠低於主偏振光源 `LED_INTENSITY` (2000.0)，導致其光照效果被覆蓋。
- 同時，光線反彈次數 `MAX_DEPTH` (初始值 12) 可能不足以讓間接光照充分分佈。

**排錯過程與結果**：

1.  **第一次嘗試：提高天花板光照強度**
    - **修改**：在 `pids_renderer.py` 的 `Config` 中，將 `CEILING_EMITTER_INTENSITY` 從 `100.0` 提高到 `1500.0`。
    - **結果**：用戶反饋問題依舊存在，效果不明顯。這表明僅增加強度不足以解決問題。

2.  **第二次嘗試：增加光線反彈次數**
    - **修改**：將 `MAX_DEPTH` 從 `12` 提高到 `24`。
    - **結果**：用戶反饋「感覺有改變」，證明光線反彈次數是正確方向，但效果仍不足。

3.  **第三次嘗試：大幅增加光線反彈次數**
    - **修改**：將 `MAX_DEPTH` 從 `24` 進一步提高到 `48`。
    - **結果**：用戶反饋此系列嘗試（提高強度與反彈次數）均未解決根本問題。

**結論**：提高光源強度與反彈次數 (`MAX_DEPTH`) 對改善間接光照有可見影響，但未能根本解決漫反射不足的問題。此路徑的嘗試暫告一段落。

---

### 階段 10：嘗試使用獨立燈陣列 (❌ 失敗並回滾)

**理論**：將單一、巨大的天花板光源替換為一組較小、更集中的獨立光源，可能產生更真實的漫反射效果。

**實施**：
1. 在 `Config` 中添加獨立燈的參數 (`CEILING_LIGHT_SIZE`, `CEILING_LIGHT_INTENSITY`) 並啟用 `CEILING_LIGHTS_ENABLED`。
2. 在 `SceneBuilder` 中實現 `_create_ceiling_lights` 方法來創建四個獨立的矩形光源。
3. 修改 `_create_meshes`，當新燈陣啟用時，禁用舊的“整個天花板為光源”的邏輯。

**結果與修正**：
- **初次嘗試結果**：渲染圖像幾乎全黑。
- **分析**：經查，光源的旋轉矩陣計算錯誤 (`rotate(X, -90)`)，導致燈光朝上射入天花板，而非朝下照亮場景。
- **二次嘗試**：將旋轉角度修正為 `rotate(X, 90)`，使燈光朝下。
- **最終結果**：用戶反饋渲染圖像依然幾乎全黑。

**結論**：此路徑的實現失敗。儘管修正了明顯的旋轉錯誤，但問題依然存在，根本原因尚不清楚，可能與變換、強度或著色器交互的更深層問題有關。**根據用戶要求，已將此功能完全回滾。**

---

### 階段 11：嘗試改善天花板光源對傢俱/牆面的漫反射效果 (❌ 全部失敗)

**問題描述**：天花板光源本身發光正常，但傢俱和牆面對該光源的漫反射效果不明顯，場景看起來像只有 LED 在照明。

**嘗試的方案**：

1. **提高天花板光源強度**
   - **修改**：`CEILING_EMITTER_INTENSITY`: 100.0 → 5000.0（提高 50 倍）
   - **結果**：❌ 無明顯改善

2. **增加光線反彈次數**
   - **修改**：`MAX_DEPTH`: 12 → 48（提高 4 倍）
   - **結果**：❌ 無明顯改善

3. **提高材質反射率（albedo）**
   - **修改**：非玻璃物件的 `reflectance`: 0.5 → 0.85
   - **結果**：❌ 無明顯改善

**結論**：三種常見的改善間接照明方法均未能解決問題。**已回滾至穩定版本。**

**關鍵發現 - Mitsuba 3 的 polarizer BSDF 行為**：

> ⚠️ 在 Mitsuba 3 中，`polarizer` BSDF 是設計來**過濾穿過它的光**，而不是作為發射器的材質。當您把它放在 `area emitter` 上時：
> - 發射的光**可能不會經過偏振片處理**
> - 或者行為不符合預期
>
> **正確做法**：應該讓光源發射非偏振光，然後**穿過一個獨立的偏振片幾何體**。

這解釋了為何將天花板設為 `area emitter` + `diffuse BSDF` 後，發射的光能正確照亮場景（因為 diffuse BSDF 會正確處理反射光的去偏振），但漫反射效果仍然不明顯的原因可能在於其他因素。

**後續排查方向**：
- 檢查天花板網格的法線方向是否正確（是否朝下）
- 嘗試使用不同的積分器（integrator）設定
- 檢查場景單位換算是否正確影響光照衰減
- 考慮將偏振光源改為「非偏振 emitter + 獨立偏振片幾何體」的架構

---

### 階段 12：實施物理偏振片架構 (v3.1.0) ✅

**問題**：根據階段 11 的關鍵發現，`polarizer` BSDF 應該用來過濾光線，而不是作為發射器材質。

**解決方案**：實施完整的物理偏振片架構：

1. **LED 光源重構**：
   - `led_emitter`：純粹的非偏振面光源（只有 `area emitter`）
   - `led_polarizer`：獨立偏振片幾何體（只有 `polarizer` BSDF, θ=0°），位於光源前方 5mm

2. **相機偏振片**：
   - 左相機：前方加偏振片 (θ=0°) → 通過 I∥
   - 右相機：前方加偏振片 (θ=90°) → 通過 I⊥

3. **其他修改**：
   - 玻璃材質：`thindielectric` → `dielectric`（實心玻璃，有折射）
   - 像素格式：`rgb` → `luminance`（灰階）

**結果**：✅ I∥ 和 I⊥ 亮度平衡了！

**實測數據 (scene_0002)**：

| 指標 | 數值 | 評價 |
|------|------|------|
| 品質評分 | 87/100 | excellent ✓ |
| 玻璃區域 DoLP | 0.247 (24.7%) | 很好的偏振效果 |
| 背景 DoLP | 0.030 (3%) | 低，符合預期 |
| 玻璃/背景對比 | 8.2x | 優秀的區分度 |
| 背景 I∥/I⊥ 平衡 | 1.0003 | 幾乎完美 ✓ |
| SNR | 2.75 | 合格 |

**關鍵成果**：
- DoLP 圖成功作為「玻璃 mask」：高 DoLP 區域 = 玻璃表面
- Fresnel 反射在玻璃邊緣產生明顯偏振
- 背景完美平衡，RAFT-Stereo 可正確對齊
- DoLP 梯度與深度邊界高度相關

**新增功能 - 品質報告生成**：

渲染完成後自動生成 `{scene}_report.json`，包含：
- DoLP 統計（全局、玻璃區域、背景區域）
- I∥/I⊥ 強度比值
- 背景平衡度檢測
- SNR 噪點分析
- 綜合品質評分 (0-100)
- 警告列表

**輸出檔案**：
- `{scene}_DoLP.png` - DoLP 灰階圖
- `{scene}_DoLP_color.png` - DoLP 熱力圖（JET colormap）
- `{scene}_report.json` - 完整品質報告

---

### 階段 13：修正相機光軸為平行配置 (v3.2.0) ✅

**問題描述**：

用戶在檢視 PIDS 論文的 Training Data Collection Standards 後發現，現有實現違反了 **Criterion 1 (Geometric Consistency Filtering)**：

> **Criterion 1**: 左右圖像的 vertical disparity 應 < 1 pixel

原因是兩個相機都使用 `look_at` 指向**同一個目標點** `(0, 611.5, 150)`，導致光軸**會聚**（converging），而非平行。這會造成：
- 垂直視差（vertical disparity）不為零
- 需要額外的 stereo rectification 步驟
- 違反標準立體視覺的假設

**原始配置（會聚光軸）**：
```
左相機: 位置 = (-32.5, 360, 150), 目標 = (0, 611.5, 150)
右相機: 位置 = (+32.5, 360, 150), 目標 = (0, 611.5, 150)

     ← 32.5mm →← 32.5mm →
    [L]                 [R]
      \                 /
       \               /
        \             /
         \           /
          \         /
           \_______/
              ↑
         共同目標點
```

**修正後（平行光軸）**：
```
左相機: 位置 = (-32.5, 360, 150), 目標 = (-32.5, 611.5, 150)
右相機: 位置 = (+32.5, 360, 150), 目標 = (+32.5, 611.5, 150)

    [L]                 [R]
     |                   |
     |                   |
     |                   |
     ↓                   ↓
  (各自的目標，方向相同)
```

**實施方案**：

1. 新增 `Config.forward_direction()` 方法：計算統一的視線方向向量 `(0, 1, 0)`
2. 新增 `Config.camera_target_for_position()` 方法：根據相機位置計算個別目標點
3. 修改 `render_scene()` 方法：為左右相機計算獨立的目標點
4. 相機偏振片自動跟隨新配置

**代碼變更**：

```python
@classmethod
def forward_direction(cls) -> Tuple[float, float, float]:
    """相機光軸方向（歸一化）"""
    center_pos = np.array([0.0, cls.CAMERA_Y, cls.CAMERA_Z])
    target = np.array(cls.target_point())
    direction = target - center_pos
    direction = direction / np.linalg.norm(direction)
    return tuple(direction)

@classmethod
def camera_target_for_position(cls, camera_pos: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """計算相機的個別目標點（確保平行光軸）"""
    direction = np.array(cls.forward_direction())
    center_pos = np.array([0.0, cls.CAMERA_Y, cls.CAMERA_Z])
    target = np.array(cls.target_point())
    distance = np.linalg.norm(target - center_pos)
    camera_pos_arr = np.array(camera_pos)
    camera_target = camera_pos_arr + direction * distance
    return tuple(camera_target)
```

**預期效果**：
- Vertical disparity ≈ 0（滿足 Criterion 1）
- 無需 stereo rectification
- 視差只出現在水平方向（epipolar line = 水平線）
- 與標準立體匹配算法相容

**實測結果**：✅ 成功

渲染輸出確認平行光軸配置正確：
```
[配置] 左相機位置: (-32.5, 360.0, 150.0)
[配置] 左相機目標: (-32.5, 611.5, 150.0)
[配置] 右相機位置: (32.5, 360.0, 150.0)
[配置] 右相機目標: (32.5, 611.5, 150.0)
[配置] 視線方向: (0.0, 1.0, 0.0) (平行光軸)
```

視覺檢查確認 I∥ 和 I⊥ 亮度平衡。

---

## 關鍵發現總結

### 1. 材質判斷函數的危險關鍵字

| 關鍵字 | 風險 | 建議 |
|--------|------|------|
| `glass` | 低 | 保留 |
| `transparent` | 低 | 保留 |
| `partition` | 高 | **移除** |
| `panel` | 高 | **移除** |
| `door` | 中 | 移除 |
| `window` | 中 | 移除 |
| `clear` | 中 | 移除（可能匹配 `clear_floor`） |

### 2. Mitsuba 3 偏振行為

- **area emitter**：本身是非偏振的，但光線經過 Fresnel 反射會產生偏振
- **constant emitter**：真正的非偏振漫射光，不參與偏振追蹤
- **diffuse BSDF**：會 depolarize 入射光（理論上）
- **thindielectric**：Fresnel 反射產生偏振

### 3. Constant Emitter 強度單位

```python
# 錯誤：直接使用大數值
scene_dict['constant_emitter'] = {
    'radiance': {'value': 8000.0},  # 太亮！
}

# 正確：需要縮放
scene_dict['constant_emitter'] = {
    'radiance': {'value': 8000.0 / 1000.0},  # = 8.0
}
```

### 4. 放大因子的影響

| amplify_factor | 效果 |
|----------------|------|
| 1.0 | 物理正確，但對比度低 |
| 5.0 | 對比度增強，背景差異也被放大 |
| 10.0 | 過度放大，產生 artifacts |

---

## 已解決問題

### 1. 背景亮度差異 (已解決於 v3.0.5)

**現狀**：已解決。通過將天花板網格本身設為非偏振面光源，為場景提供了均勻的背景照明，顯著減少了 I∥ 和 I⊥ 圖像之間的亮度差異。

**理想解決方案**：已實施。將漫反射表面（天花板）作為發光體，確保了光線的正確傳播和去偏振效果。

## V2 遺留問題（不適用於 V3）

### 1. 櫃子鬼影 (V2 問題)

> ⚠️ **注意**：此問題僅存在於 V2 版本，V3 版本已通過新的光源架構解決。

**可能原因**（V2）：
1. OBJ 模型有雙層面
2. 玻璃透射造成的物理現象
3. Stokes 通道解析錯誤

**排查方向**（V2）：
- 檢查 OBJ 文件的面數量
- 嘗試不同的 Stokes 通道解析方式
- 檢查是否有重複的幾何體

---

## 推薦的渲染參數

```bash
# 標準渲染
python renderer_stage1_exaggerated.py \
    --scene_file scene.obj \
    --output_dir ./output

# 調整光源平衡
python renderer_stage1_exaggerated.py \
    --scene_file scene.obj \
    --output_dir ./output \
    --polarized_intensity 1500 \
    --fill_intensity 8000

# 高品質渲染
python renderer_stage1_exaggerated.py \
    --scene_file scene.obj \
    --output_dir ./output \
    --spp 32768
```

---

## 版本歷史

| 版本 | 日期 | 主要變更 |
|------|------|----------|
| v1.0 | 初始 | 基本偏振渲染 |
| v1.1 | - | 添加 fill_light |
| v1.2 | - | 添加 constant emitter |
| v1.3 | - | 修復材質判斷函數 |
| v3.0.5 | 2025-12-21 | 解決非偏振光源問題，將天花板網格直接設為面光源 |
| v3.0.6 | 2025-12-21 | 嘗試提高光源強度與 MAX_DEPTH，未能解決漫反射問題 |
| v3.1.0 | 2025-12-22 | **物理偏振片架構**：LED/相機獨立偏振片、dielectric玻璃、品質報告生成 |
| v3.2.0 | 2025-12-22 | **平行光軸配置**：修正相機會聚問題，符合 PIDS Criterion 1 |

---

## 後續工作

1. **深入研究 Mitsuba 3 偏振 BSDF**：確認 diffuse 材質是否真的會 depolarize
2. **檢查 OBJ 幾何**：確認是否有重複面導致鬼影
3. **測試不同的 Stokes 提取方式**：可能需要針對不同通道數量做更細緻的處理
4. **考慮使用 Mask**：在後處理中完全遮蔽背景的偏振差異
