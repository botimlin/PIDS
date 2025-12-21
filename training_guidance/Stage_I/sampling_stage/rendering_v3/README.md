# PIDS Renderer V3 文檔

## 概述

PIDS Renderer V3 是專為 PIDS (Physics-Informed Deep Stereo) 專案設計的偏振立體渲染器，用於生成 Stage 1 合成訓練數據。

**版本**: 3.0
**日期**: 2025-12-21
**檔案**: `pids_renderer.py`

---

## 座標系統

### Blender/OBJ 座標系

```
        Z (高度)
        │
        │   天花板 Z=300mm
        │
        │   ┌─────────────────────────────┐
        │   │                             │
        │   │  Chamber 內部               │
        │   │                             │
        └───┼─────────────────────────────┼──── X (左右)
            │                             │
           -300mm                       +300mm


        俯視圖 (從 Z 軸往下看):

        Y (深度/前方)
        │
        │   後牆 Y=900mm ─────────────────────────
        │   │                                    │
        │   │  傢俱區 Y=700-800mm                │
        │   │                                    │
        │   │  玻璃區 Y=528-695mm (景深範圍)      │
        │   │                                    │
        │   │  📷 相機 Y=360mm                   │  ← 相機在 chamber 內！
        │   │                                    │
        │   前牆 Y=350mm ─────────────────────────
        │
        └────────────────────────────────────────── X
                      相機朝向 +Y 方向
```

### 座標轉換 (OBJ → Mitsuba)

OBJ 匯出設定為 `forward=-Y, up=Z`，載入 Mitsuba 時需要 -90° X 軸旋轉：

| OBJ 座標 | Mitsuba 座標 | 說明 |
|----------|--------------|------|
| X | X | 左右 (不變) |
| Y | Z | 深度/前後 |
| Z | Y | 高度 |

同時需要 mm → m 單位轉換 (×0.001)。

---

## 配置參數

### Chamber 尺寸

| 參數 | 值 | 說明 |
|------|-----|------|
| `CHAMBER_WIDTH` | 600mm | X 方向 (-300 ~ +300) |
| `CHAMBER_HEIGHT` | 300mm | Z 方向 (0 ~ 300) |
| `CHAMBER_Y_FRONT` | 350mm | 前牆 Y 位置 |
| `CHAMBER_Y_BACK` | 900mm | 後牆 Y 位置 |

### 相機配置

| 參數 | 值 | 說明 |
|------|-----|------|
| `CAMERA_Y` | 360mm | 相機深度 (在 chamber 內) |
| `CAMERA_Z` | 150mm | 相機高度 (chamber 中心) |
| `BASELINE` | 65mm | 立體基線 |
| `FOV` | 65° | 水平視場角 |

### 相機位置計算

```python
左相機: (-32.5mm, 360mm, 150mm)  # X=-BASELINE/2
右相機: (+32.5mm, 360mm, 150mm)  # X=+BASELINE/2
目標點: (0, 611.5mm, 150mm)      # 玻璃區中心
```

### 渲染配置

| 參數 | 值 | 說明 |
|------|-----|------|
| `WIDTH` | 640 | 輸出寬度 (px) |
| `HEIGHT` | 480 | 輸出高度 (px) |
| `SPP` | 4096 | 每像素樣本數 |
| `SPP_PER_BATCH` | 1024 | 分批渲染 |
| `MAX_DEPTH` | 12 | 光線反彈次數 |

### 光源配置

| 參數 | 值 | 說明 |
|------|-----|------|
| `LED_INTENSITY` | 2000.0 | LED 強度 |
| `LED_SIZE` | (180, 100)mm | 面光源尺寸 |
| `LED_POSITION_Y` | 280mm | LED 高度 |
| `LED_POSITION_Z` | 420mm | LED 深度 |

### 材質配置

| 參數 | 值 | 說明 |
|------|-----|------|
| `GLASS_IOR` | 1.5 | 玻璃折射率 |
| `GLASS_ROUGHNESS` | 0.02 | 玻璃粗糙度 (低值增強偏振) |

---

## 使用方式

### 基本用法

```bash
# 單一場景渲染
python pids_renderer.py --scene scene_0001.obj --output ./output

# 批次渲染
python pids_renderer.py --input_dir ./scenes_output --output ./rendered

# 限制場景數量
python pids_renderer.py --input_dir ./scenes_output --output ./rendered --max_scenes 10
```

### 進階選項

```bash
# 調整 SPP (品質)
python pids_renderer.py --scene scene.obj --output ./output --spp 8192

# 快速預覽 (低 SPP)
python pids_renderer.py --scene scene.obj --output ./output --spp 512

# 不保存預覽 PNG
python pids_renderer.py --scene scene.obj --output ./output --no_preview
```

### 命令行參數

| 參數 | 必需 | 說明 |
|------|------|------|
| `--scene` | 二選一 | 單一 OBJ 場景路徑 |
| `--input_dir` | 二選一 | OBJ 場景目錄 |
| `--output` | 是 | 輸出目錄 |
| `--spp` | 否 | SPP (預設: 4096) |
| `--max_scenes` | 否 | 最大渲染場景數 |
| `--no_preview` | 否 | 不保存預覽 PNG |

---

## 輸出檔案

每個場景生成以下檔案：

### 主要輸出 (EXR)

| 檔案 | 格式 | 說明 |
|------|------|------|
| `{scene}_left_parallel.exr` | float32 灰階 | I∥ (0° 偏振) |
| `{scene}_right_cross.exr` | float32 灰階 | I⊥ (90° 偏振) |
| `{scene}_depth.exr` | float32 | 深度圖 (米) |
| `{scene}_disparity.exr` | float32 | 視差圖 (像素) |

### 預覽圖 (PNG)

| 檔案 | 說明 |
|------|------|
| `{scene}_left_parallel.png` | I∥ 預覽 (統一範圍) |
| `{scene}_right_cross.png` | I⊥ 預覽 (統一範圍) |
| `{scene}_polarization_diff.png` | \|I∥ - I⊥\| 差異圖 |
| `{scene}_depth.png` | 深度 colormap |
| `{scene}_DoLP.png` | 偏振度 (Degree of Linear Polarization) |

### 參數檔案

| 檔案 | 說明 |
|------|------|
| `{scene}_params.json` | 渲染參數和統計資訊 |

---

## 物理原理

### 偏振成像系統

```
偏振 LED (0°)
     │
     ▼ 線性偏振光
     │
     ├──────────────────────────────┐
     │                              │
     ▼                              ▼
  玻璃表面                      漫反射背景
  (Fresnel 反射)                (無偏振)
     │                              │
     ├──────────┬───────────────────┤
     │          │                   │
     ▼          ▼                   ▼
 左相機(0°)  右相機(90°)
   I∥            I⊥
```

### Stokes Vector

Mitsuba 3 偏振渲染輸出 Stokes Vector：

```
S0 = 總強度
S1 = I_H - I_V (水平 vs 垂直偏振差)
S2 = I_+45 - I_-45 (±45° 偏振差)
S3 = I_RCP - I_LCP (圓偏振，通常忽略)
```

### 偏振片透射 (Malus 定律)

```
I(θ) = 0.5 × (S0 + S1×cos(2θ) + S2×sin(2θ))

I∥ (θ=0°)  = 0.5 × (S0 + S1)  → 保留水平偏振
I⊥ (θ=90°) = 0.5 × (S0 - S1)  → 保留垂直偏振
```

### 偏振度 (DoLP)

```
DoLP = √(S1² + S2²) / S0
範圍: [0, 1]

DoLP ≈ 0: 非偏振 (漫反射)
DoLP > 0: 部分偏振 (玻璃反射)
```

### 預期效果

| 區域 | I∥ | I⊥ | 差異 |
|------|-----|-----|------|
| 玻璃表面 (Fresnel 反射) | 強 | 弱 | I∥ >> I⊥ |
| 漫反射背景 | 中 | 中 | I∥ ≈ I⊥ |

---

## 視差計算

### 公式

```
focal_length (px) = width / (2 × tan(FOV/2))
                  = 640 / (2 × tan(32.5°))
                  ≈ 502 px

disparity = (baseline × focal_length) / depth
```

### 視差範圍

| 深度 (mm) | 視差 (px) | 說明 |
|-----------|-----------|------|
| 528 | ~62 | 景深近端 |
| 600 | ~54 | 對焦點 |
| 695 | ~47 | 景深遠端 |
| 900 | ~36 | 後牆 |

---

## 模組架構

```
pids_renderer.py
│
├── Config                 # 集中配置管理
│
├── MaterialFactory        # 材質創建
│   ├── glass()           # 玻璃材質 (roughdielectric)
│   ├── diffuse()         # 漫反射材質
│   └── diffuse_rgb()     # RGB 轉漫反射
│
├── MTLParser              # MTL 檔案解析
│   ├── parse()           # 解析材質定義
│   └── _is_glass_name()  # 判斷玻璃材質
│
├── SceneBuilder           # Mitsuba 場景建構
│   ├── build()           # 建構完整場景
│   ├── _create_integrator()
│   ├── _create_sensor()
│   ├── _create_led_light()
│   ├── _create_mesh()
│   └── _transform_point() # 座標轉換
│
├── StokesProcessor        # Stokes 向量處理
│   ├── extract_stokes()          # 提取 S0, S1, S2
│   ├── compute_polarization_images()  # 計算 I∥, I⊥
│   └── compute_dolp()            # 計算偏振度
│
└── PIDSRenderer           # 主渲染器
    ├── render_scene()     # 渲染場景入口
    ├── _render_camera()   # 渲染單一相機
    ├── _render_depth()    # 渲染深度圖
    ├── _compute_disparity() # 計算視差
    └── _save_outputs()    # 保存輸出
```

---

## Mitsuba 配置

### Variant 選擇

**必須使用 `spectral_polarized` variant！**

```python
# ✅ 正確
'cuda_ad_spectral_polarized'
'cuda_spectral_polarized'
'llvm_ad_spectral_polarized'

# ❌ 錯誤 (會丟棄偏振信息)
'cuda_ad_mono_polarized'  # luminance film 會丟棄 S1/S2/S3
```

### Integrator

```python
'integrator': {
    'type': 'stokes',  # 輸出 Stokes Vector
    'integrator': {
        'type': 'path',
        'max_depth': 12,  # 透明物體需要足夠深度
    },
}
```

### 玻璃材質

```python
'bsdf': {
    'type': 'roughdielectric',
    'alpha': 0.02,     # 低粗糙度 = 強偏振反射
    'int_ior': 1.5,
    'ext_ior': 1.0,
    'distribution': 'ggx',
}
```

---

## 常見問題

### Q: 兩張圖 (I∥ 和 I⊥) 看起來一樣？

**可能原因**：
1. 使用了 `mono_polarized` variant → 改用 `spectral_polarized`
2. Stokes 通道解析錯誤 → 檢查 13 通道格式
3. 光源未產生偏振 → 確認 LED 配置

### Q: 玻璃區域全黑？

**可能原因**：
1. `max_depth` 太低 → 增加到 8-12
2. 光源位置不對 → 檢查 LED 配置
3. 材質設定錯誤 → 使用 `roughdielectric`

### Q: DoLP 接近 0？

**可能原因**：
1. 被動偏振太弱 → Stage 1 本來就是誇大偏振
2. 多方向光源 → 關閉 lightbox 模式
3. 多次反射 depolarization → 限制 max_depth

### Q: 座標/位置不對？

**檢查項目**：
1. OBJ 匯出設定：`forward=-Y, up=Z`
2. 相機在 chamber 內：Y > 350mm
3. 玻璃在景深範圍：Y = 528-695mm

### Q: 背景亮度差異過大？(I∥ 很亮，I⊥ 很暗)

**可能原因**：材質被誤判為玻璃

**解決方案**：已在 v3 修復，使用保守的材質判斷策略：
```python
# 非玻璃關鍵字（優先排除）
NON_GLASS_KEYWORDS = ['background', 'wall', 'floor', 'ground', 'ceiling',
                      'diffuse', 'opaque', 'solid', 'wood', 'metal', 'fabric']

# 玻璃關鍵字（只使用明確的）
GLASS_KEYWORDS = ['glass', 'transparent', 'acrylic']
```

**詳細排錯歷程**：參考 `PIDS_DEBUG_HISTORY.md`

### Q: 櫃子等非玻璃物體出現「鬼影」？

**可能原因**：
1. OBJ 模型有雙層面
2. 材質被誤判為玻璃
3. Stokes 通道解析問題

**排查方向**：
1. 檢查 OBJ 檔案的面數量
2. 確認材質名稱不含 `glass`、`transparent` 等關鍵字

---

## 與 Blender 腳本的對應

本渲染器的配置與 `blender_furniture_randomizer_v17.py` 對應：

| Blender 配置 | 渲染器配置 | 值 |
|--------------|-----------|-----|
| `front_wall.y_position` | `CHAMBER_Y_FRONT` | 350mm |
| `back_wall.y_offset` | `CHAMBER_Y_BACK` | ~900mm |
| `glass_y_min/max` | `GLASS_Y_MIN/MAX` | 528-695mm |
| `furniture_y` | - | 700-800mm |
| `back_wall.width` | `CHAMBER_WIDTH` | 600mm |
| `back_wall.height` | `CHAMBER_HEIGHT` | 300mm |

---

## 工作流程

```
1. Blender 生成場景
   │  blender_furniture_randomizer_v17.py
   ▼
2. 匯出 OBJ
   │  scenes_output/scene_XXXX.obj
   ▼
3. 渲染
   │  python pids_renderer.py --input_dir ./scenes_output --output ./rendered
   ▼
4. 輸出訓練數據
   │  rendered/
   │  ├── scene_XXXX_left_parallel.exr
   │  ├── scene_XXXX_right_cross.exr
   │  ├── scene_XXXX_depth.exr
   │  ├── scene_XXXX_disparity.exr
   │  └── ...
   ▼
5. 質量檢查
   │  python training_data_quality_check.py --input_dir ./rendered
   ▼
6. 訓練 RAFT-Stereo
```

---

## 開發記錄

### 2025-12-21: OBJ 分離方案成功

**問題**: Mitsuba 3 的 OBJ loader 限制每個 shape 只能有一個 BSDF，無法在單一 OBJ 中同時指定玻璃和漫反射材質。

**錯誤訊息**:
```
RuntimeError: [Shape] Only a single BSDF child object can be specified per shape.
```

**解決方案**: `OBJSplitter` 類別
1. 解析 OBJ 檔案，讀取所有頂點、法線、UV 和面
2. 按材質分離面：
   - 玻璃材質 → `{scene}_glass.obj`
   - 其他材質 → `{scene}_other.obj`
3. 分別載入並指定不同 BSDF：
   - `mesh_other`: `diffuse` BSDF
   - `mesh_glass`: `thindielectric` BSDF

**結果**: 玻璃正確顯示為透明

---

### 2025-12-21: 偏振處理問題 - 環境光分析

**問題**: 背景亮度差異過大，I∥ 和 I⊥ 應該在背景區域接近一致

**根本原因分析**:

1. **Constant Emitter 被 Chamber 擋住**
   ```
           ☀️ constant emitter（無限遠環境球）
                 ↓ 光線
           ┌─────────────┐
           │   Chamber   │  ← 光線被外牆擋住！
           │  ┌───────┐  │
           │  │ 內部  │  │  ← 環境光進不來
           │  └───────┘  │
           └─────────────┘
   ```
   - `AMBIENT_INTENSITY = 5.0` 完全無效，因為 chamber 是封閉的
   - 場景 90%+ 的光來自偏振 LED

2. **Mitsuba 3 的 diffuse BSDF 未完全 depolarize**
   - 理論上漫反射應該消除偏振 (S1 → 0)
   - 實際上 S1 仍有微小值，導致 `0.5*(S0+S1)` 和 `0.5*(S0-S1)` 有差異

**嘗試的解決方案**:

| 方案 | 結果 |
|------|------|
| 提高 `AMBIENT_INTENSITY` | 無效 (被 chamber 擋住) |
| 天花板設為 emitter | 效果不佳 |
| 獨立天花板區域光源 | 測試中 |

**當前配置** (v3.0.3):
```
光源配置:
├── LED (偏振光源)
│   ├── 位置: 頂部前方 (Y=420, Z=280)
│   ├── 尺寸: 180×100mm
│   └── 強度: 2000
│
└── 天花板光 (非偏振環境光)
    ├── 位置: 頂部中央 (Y=625, Z=295)
    ├── 尺寸: 400×350mm
    └── 強度: 1500
```

**OBJ 分離** (新增天花板分離):
```
OBJSplitter 現在分離三種幾何:
├── mesh_glass    → thindielectric BSDF
├── mesh_ceiling  → diffuse BSDF (反射率 0.85)
└── mesh_other    → diffuse BSDF (反射率 0.5)
```

---

## 版本歷史

| 版本 | 日期 | 說明 |
|------|------|------|
| 3.0.3 | 2025-12-21 | 添加獨立天花板光源，解決 chamber 封閉問題 |
| 3.0.2 | 2025-12-21 | 回滾偏振處理，保留 OBJ 分離方案 |
| 3.0.1 | 2025-12-21 | 嘗試 V2 偏振放大方法 (失敗) |
| 3.0 | 2025-12-21 | 從頭設計，OBJ 分離解決材質問題 |
| 2.0 | - | 物理正確偏振版 |
| 1.0 | - | 初始版本 |
