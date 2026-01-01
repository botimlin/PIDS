# PIDS Stage 1 偏振渲染器開發記錄

## 📋 文件信息

- **項目**: PIDS (Polarized Image Depth Sensing)
- **階段**: Stage 1 - 誇張偏振 (Exaggerated Polarization)
- **渲染器**: `renderer_stage1_exaggerated.py`
- **日期**: 2025-12-21
- **狀態**: ✅ 偏振信號成功產生

---

## 🎯 目標

Stage 1 的目標是產生**誇張的偏振差異**，讓神經網路能夠輕鬆學習偏振特徵，之後再通過 Stage 2 的 domain adaptation 轉移到真實物理世界。

### 論文設置 (Fig. 1)
- 光源通過 0° 線性偏振片（水平偏振）
- 左相機前 0° 偏振片（與光源平行）→ I∥
- 右相機前 90° 偏振片（與光源垂直）→ I⊥
- 玻璃表面的 Fresnel 反射產生偏振差異

---

## 🔧 關鍵配置

### 最終工作配置

```python
CONFIG = {
    'render': {
        'width': 640,
        'height': 480,
        'spp': 4096,
        'spp_per_batch': 1024,
        'max_depth': 8,  # 透明玻璃需要足夠的 bounce
    },
    'lighting': {
        'lightbox': {'enabled': False},  # 必須關閉！
        'led': {
            'intensity': 5000.0,
            'polarization_angle': 0.0,
        },
        'ambient': {'intensity': 0.5},
    },
}
```

### Mitsuba Variant

```python
# ✅ 正確
'cuda_ad_spectral_polarized'

# ❌ 錯誤（會丟棄偏振信息）
'cuda_ad_mono_polarized'  # + luminance film = S1/S2/S3 全部丟失！
```

### 玻璃材質

```python
# ✅ Stage 1 最終選擇
'bsdf': {
    'type': 'roughdielectric',
    'alpha': 0.05,  # 低粗糙度 = 強 Fresnel 偏振
    'int_ior': 1.5,
    'ext_ior': 1.0,
}
```

### Integrator

```python
'integrator': {
    'type': 'stokes',  # 輸出 Stokes Vector
    'integrator': {
        'type': 'path',
        'max_depth': 8,
    },
}
```

---

## 🐛 Debug 歷程

### 問題 1：兩張圖完全相同

**症狀**: `left_parallel.png` 和 `right_cross.png` 看起來一模一樣

**根因**: 13 通道 Stokes 解析錯誤
- 錯誤假設：通道 0-3 = [S0, S1, S2, S3]
- 實際格式：通道 0-3 相同（S0 的波長副本）

**診斷輸出**:
```
通道 0: [0.0206, 2.5299]
通道 1: [0.0206, 2.5299]  ← 相同！
通道 2: [0.0206, 2.5299]  ← 相同！
通道 3: [0.0206, 2.5299]  ← 相同！
通道 4: [0.0000, 0.1161]  ← 不同 = S1
```

**修正**: 13 通道格式為 `[4×S0, 4×S1, 4×S2, 1×S3]`
```python
S0 = stokes_image[:,:,0]   # 通道 0
S1 = stokes_image[:,:,4]   # 通道 4
S2 = stokes_image[:,:,8]   # 通道 8
```

---

### 問題 2：S1/S2 全為 0

**症狀**: 
```
通道 4-12: [0.0000, 0.0000]
DoLP mean: 0.0000
```

**根因**: Mitsuba 警告
```
WARN [HDRFilm] Monochrome mode enabled, setting film output pixel format to 'luminance'
```

`cuda_ad_mono_polarized` + 自動 `luminance` film = **偏振信息被丟棄**！

**修正**: 改用 `cuda_ad_spectral_polarized`
```python
polarized_variants = [
    'cuda_ad_spectral_polarized',   # ✅ 保留完整 Stokes
    # 'cuda_ad_mono_polarized',     # ❌ 丟棄偏振
]
```

---

### 問題 3：偏振信號極弱 (DoLP ≈ 0.01)

**症狀**: DoLP mean 接近 0，|I∥ - I⊥| 幾乎為 0

**根因**: 三個物理問題疊加
1. **被動偏振太弱**: `dielectric` 材質的 Fresnel 反射很弱
2. **燈箱模式消掉偏振**: 多方向光源 = 偏振平均掉
3. **path depth 太深**: 多次反射導致 depolarization

**修正**:
1. 關閉燈箱：`lightbox.enabled = False`
2. 改用 `conductor` 或 `roughdielectric` 材質
3. 限制 `max_depth`（但透明材質需要足夠深度）

---

### 問題 4：玻璃全黑

**症狀**: 玻璃區域完全黑色

**根因 A**: 使用 `conductor` 材質
- Conductor = 金屬 = 完全反射（不透明）
- 反射的是黑色環境背景

**修正**: 改用 `roughdielectric`（透明 + 偏振）

**根因 B**: `max_depth = 2` 太低
- 光線穿透玻璃需要至少 4 次 bounce
- `進入(1) → 穿過(2) → 出去(3) → 背景(4)`

**修正**: `max_depth = 8`

---

### 問題 5：matplotlib/PIL 依賴缺失

**症狀**: `ModuleNotFoundError: No module named 'matplotlib'`

**修正**: 全部改用 OpenCV
```python
# ❌ 之前
import matplotlib.pyplot as plt
from PIL import Image

# ✅ 現在
import cv2
cv2.applyColorMap(img, cv2.COLORMAP_JET)
cv2.imwrite(path, img)
```

---

## 📊 成功指標

### 偏振信號確認

```
# ✅ 成功的輸出
DoLP mean: 0.6585
S1 範圍: [0.0000, 78.8130]
S2 範圍: [0.0000, 74.5111]
I_parallel 範圍: [0.0000, 149.4125]
I_cross 範圍: [0.0000, 53.9793]
|I∥ - I⊥| mean: 14.1817
```

### 預期效果
- **玻璃反射區域**: I_parallel >> I_cross（明顯差異）
- **非偏振區域**: I_parallel ≈ I_cross（相似）
- **DoLP**: 0.1 ~ 0.5（玻璃表面）

---

## 📁 輸出文件

| 文件 | 描述 |
|------|------|
| `*_left_parallel.exr` | I∥ 強度 (EXR float32) |
| `*_right_cross.exr` | I⊥ 強度 (EXR float32) |
| `*_left_parallel.png` | I∥ 預覽 (統一範圍) |
| `*_right_cross.png` | I⊥ 預覽 (統一範圍) |
| `*_polarization_diff.png` | |I∥ - I⊥| 差異圖 |
| `*_DoLP.png` | 偏振度 (灰階) |
| `*_DoLP_color.png` | 偏振度 (JET colormap) |
| `*_AoLP.png` | 偏振角 (HSV) |
| `*_depth.exr` | 深度圖 |
| `*_disparity.exr` | 視差圖 |

---

## 🧮 物理公式

### Stokes Vector
```
S0 = 總強度
S1 = I_H - I_V (水平 vs 垂直偏振差)
S2 = I_+45 - I_-45 (±45° 偏振差)
S3 = I_RCP - I_LCP (圓偏振，通常忽略)
```

### 偏振度 (DoLP)
```
DoLP = √(S1² + S2²) / S0
範圍: [0, 1]
```

### 偏振角 (AoLP)
```
AoLP = 0.5 × atan2(S2, S1)
範圍: [0°, 180°]
```

### 偏振片透射 (Malus 定律)
```
I(θ) = 0.5 × (S0 + S1×cos(2θ) + S2×sin(2θ))
I_parallel (θ=0°) = 0.5 × (S0 + S1)
I_cross (θ=90°) = 0.5 × (S0 - S1)
```

---

## 🔑 關鍵教訓

### 1. Mitsuba Variant 選擇
> `mono_polarized` 在 `luminance` film 下會**丟棄偏振信息**！必須使用 `spectral_polarized`。

### 2. Stage 1 的哲學
> 「Stage-1 偏振不明顯不是失敗，而是證明之前的設定是物理正確的。」
> 
> Stage 1 要做的是：**刻意不物理正確，讓 network 先學會看偏振**。這就是 **representation forcing**。

### 3. 物理限制
- `conductor` = 不透明，但偏振最強
- `dielectric` = 透明，但偏振較弱
- `roughdielectric` = 透明 + 中等偏振（推薦）

### 4. path depth 的權衡
- 太低 (2)：光線無法穿透玻璃 → 全黑
- 太高 (32)：多次反射 → depolarization
- 推薦：8（透明玻璃的最佳平衡）

---

## 🚀 下一步

1. **驗證透明玻璃的偏振效果**（max_depth = 8）
2. **調整參數達到最佳偏振對比**
3. **批量渲染所有場景**
4. **準備 Stage 2 的 domain adaptation**

---

## 📚 參考

- Mitsuba 3 Documentation: https://mitsuba.readthedocs.io/
- Stokes Vector: https://en.wikipedia.org/wiki/Stokes_parameters
- Fresnel Equations: https://en.wikipedia.org/wiki/Fresnel_equations
- Brewster's Angle: ~56° for glass (n=1.5)
