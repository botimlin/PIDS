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

## 未解決的問題

### 1. 背景亮度差異

**現狀**：使用 `auto_balance` 後處理來平衡背景。

**理想解決方案**：
- 讓漫反射表面真正 depolarize 光線
- 可能需要更深入研究 Mitsuba 3 的偏振 BSDF 實現

### 2. 櫃子鬼影

**可能原因**：
1. OBJ 模型有雙層面
2. 玻璃透射造成的物理現象
3. Stokes 通道解析錯誤

**排查方向**：
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
| v1.4 | 當前 | 回滾到穩定配置 + 保留材質修復 |

---

## 後續工作

1. **深入研究 Mitsuba 3 偏振 BSDF**：確認 diffuse 材質是否真的會 depolarize
2. **檢查 OBJ 幾何**：確認是否有重複面導致鬼影
3. **測試不同的 Stokes 提取方式**：可能需要針對不同通道數量做更細緻的處理
4. **考慮使用 Mask**：在後處理中完全遮蔽背景的偏振差異
