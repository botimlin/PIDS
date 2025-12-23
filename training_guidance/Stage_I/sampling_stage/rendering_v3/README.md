# PIDS Renderer V3 文檔

## 概述

PIDS Renderer V3 是專為 PIDS (Physics-Informed Deep Stereo) 專案設計的偏振立體渲染器，用於生成 Stage 1 合成訓練數據。

**版本**: 3.4.2 (暫定最終版)
**日期**: 2025-12-22

### 目錄結構

```
rendering_v3/
├── pids_renderer.py              # 偏振渲染器 (主要)
├── pids_renderer_nopol.py        # 無偏振渲染器 (對比實驗用)
├── README.md                     # 本文檔
├── PIDS_DEBUG_HISTORY.md         # 開發除錯歷史
│
├── Quality_Assurance/            # 品質檢測工具
│   ├── quality_validator.py      # 品質驗證器 v1.4
│   ├── aggregate_reports.py      # 報告匯總腳本
│   └── report/                   # 生成的報告
│       ├── summary_report.md     # 場景匯總報告
│       └── quality_report.md     # 品質驗證報告
│
└── output/                       # 渲染輸出
    └── output/                   # 場景渲染結果
        ├── scene_XXXX_*.exr      # EXR 檔案
        ├── scene_XXXX_*.png      # 預覽圖
        └── scene_XXXX_*.json     # 參數/報告
```

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

### 感測器配置 (v3.4.0)

| 參數 | 值 | 說明 |
|------|-----|------|
| 感測器 | Sony IMX296LQR-C | 1.58 MP 彩色 |
| `SENSOR_WIDTH` | 5.023mm | 1456 × 3.45μm |
| `SENSOR_HEIGHT` | 3.754mm | 1088 × 3.45μm |
| `FOCAL_LENGTH` | 6mm | 鏡頭焦距 |
| `FOV` | 45.4° | 水平視場角 = 2×arctan(5.023/(2×6)) |

### 相機配置

| 參數 | 值 | 說明 |
|------|-----|------|
| `CAMERA_X` | -50mm | X 偏移（向左）|
| `CAMERA_Y` | 400mm | 相機深度 (在 chamber 內) |
| `CAMERA_Z` | 80mm | 相機高度 (較低) |
| `BASELINE` | 65mm | 立體基線 |

### 相機位置計算（平行光軸配置 v3.4.1）

```python
# 立體相機位置
左相機: (CAMERA_X - BASELINE/2, CAMERA_Y, CAMERA_Z) = (-82.5mm, 400mm, 80mm)
右相機: (CAMERA_X + BASELINE/2, CAMERA_Y, CAMERA_Z) = (-17.5mm, 400mm, 80mm)

# 深度相機位置（在立體相機中央，同水平）
深度相機: (CAMERA_X, CAMERA_Y, CAMERA_Z) = (-50mm, 400mm, 80mm)

# 統一視線方向
光軸方向: (0, 1, 0)  # 三台相機完全平行，無會聚
```

**重要**:
- v3.2.0 修正了光軸會聚問題，各相機有獨立目標點，確保光軸完全平行
- v3.4.0 新增獨立深度相機，位於立體相機中央，FOV 與立體相機一致

### 渲染配置

| 參數 | 值 | 說明 |
|------|-----|------|
| `WIDTH` | 640 | 輸出寬度 (px) |
| `HEIGHT` | 480 | 輸出高度 (px) |
| `SPP` | 16384 | 每像素樣本數 (16K 高品質) |
| `SPP_PER_BATCH` | 1024 | 分批渲染 |
| `MAX_DEPTH` | 12 | 光線反彈次數 |

### 光源配置

| 參數 | 值 | 說明 |
|------|-----|------|
| `LED_INTENSITY` | 2000.0 | LED 強度 |
| `LED_SIZE` | (180, 100)mm | 面光源尺寸 |
| `LED_POSITION_Y` | 280mm | LED 高度 |
| `LED_POSITION_Z` | 420mm | LED 深度 |
| `CEILING_EMITTER_INTENSITY` | 20.0 | 天花板面光源強度 (非偏振) |

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
| `--spp` | 否 | SPP (預設: 16384) |
| `--max_scenes` | 否 | 最大渲染場景數 |
| `--no_preview` | 否 | 不保存預覽 PNG |

---

## 無偏振版本 (pids_renderer_nopol.py)

### 用途

用於論文中的**消融實驗 (Ablation Study)**，比較偏振 vs 無偏振對透明物體偵測的影響。

### 與偏振版的差異

| 項目 | 偏振版 (`pids_renderer.py`) | 無偏振版 (`pids_renderer_nopol.py`) |
|------|---------------------------|-------------------------------------|
| Mitsuba variant | `cuda_ad_spectral_polarized` | `cuda_ad_rgb` |
| Integrator | `stokes` (輸出 S0,S1,S2,S3) | `path` (標準渲染) |
| LED 光源 | 有偏振片 (0° 水平偏振) | 無偏振片 |
| 左相機 | 0° 偏振片 (I∥) | 無偏振片 |
| 右相機 | 90° 偏振片 (I⊥) | 無偏振片 |
| 輸出檔名 | `*_left_parallel.exr`, `*_right_cross.exr` | `*_left_nopol.exr`, `*_right_nopol.exr` |
| DoLP 輸出 | 有 | 無 |

### 保持完全相同的設置

以下參數與偏振版**完全相同**，確保對比實驗的公平性：

- **相機位置**: 左 (-82.5, 400, 80)mm，右 (-17.5, 400, 80)mm
- **Baseline**: 65mm
- **FOV**: 45.4°
- **LED 位置**: (0, 420, 280)mm
- **LED 強度**: 2000
- **LED 尺寸**: 180×100mm
- **天花板發光強度**: 100
- **玻璃 IOR**: 1.5
- **材質設定**: dielectric (玻璃), diffuse (其他)
- **座標轉換**: OBJ → Mitsuba (-90° X 軸旋轉, mm→m)

### 使用方式

```bash
# 單一場景
python pids_renderer_nopol.py --scene scene_0001.obj --output ./output_nopol

# 批次渲染（使用與偏振版相同的場景）
python pids_renderer_nopol.py --input_dir ./scenes --output ./output_nopol --max_scenes 100

# 調整 SPP
python pids_renderer_nopol.py --scene scene.obj --output ./output_nopol --spp 8192
```

### 無偏振版輸出檔案

| 檔案 | 格式 | 說明 |
|------|------|------|
| `{scene}_left_nopol.exr` | float32 灰階 | 左相機 (無偏振) |
| `{scene}_right_nopol.exr` | float32 灰階 | 右相機 (無偏振) |
| `{scene}_depth.exr` | float32 | 深度圖 (米) |
| `{scene}_disparity.exr` | float32 | 視差圖 (像素) |
| `{scene}_glass_mask.exr` | float32 | 玻璃區域 mask |
| `{scene}_stereo_diff.png` | PNG | 左右圖像差異 (應該很小) |
| `{scene}_report_nopol.json` | JSON | 品質報告 |
| `{scene}_params_nopol.json` | JSON | 渲染參數 |

### 預期結果差異

| 指標 | 偏振版 | 無偏振版 |
|------|--------|----------|
| 玻璃區域對比度 | 高 (I∥ >> I⊥) | 低 (左 ≈ 右) |
| DoLP | 10-30% (玻璃區域) | N/A |
| 立體匹配難度 | 較低 (偏振輔助) | 較高 (純紋理) |

### 論文對比實驗建議

1. 使用**相同的 OBJ 場景**分別渲染偏振版和無偏振版
2. 使用**相同的訓練配置**分別訓練兩個模型
3. 在相同的測試集上比較 EPE、3px error 等指標
4. 特別關注**玻璃區域**的深度估計精度差異

---

## 輸出檔案

每個場景生成以下檔案：

### 主要輸出 (EXR)

| 檔案 | 格式 | 說明 |
|------|------|------|
| `{scene}_left_parallel.exr` | float32 灰階 | 左相機 I∥ (0° 偏振) |
| `{scene}_right_parallel.exr` | float32 灰階 | 右相機 I∥ (0° 偏振) - v3.4.0 新增 |
| `{scene}_right_cross.exr` | float32 灰階 | 右相機 I⊥ (90° 偏振) |
| `{scene}_depth.exr` | float32 | 深度圖 (米) - 深度相機視角 |
| `{scene}_disparity.exr` | float32 | 視差圖 (像素) |
| `{scene}_glass_mask.exr` | float32 | 玻璃區域 mask - v3.3.0 新增 |

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

### 2025-12-21: 解決非偏振光源無法照亮場景問題 (v3.0.5)

**問題**: 在 Mitsuba 3 `spectral_polarized` 變體中，獨立的非偏振光源（如天花板燈陣列）無法有效照亮場景，導致背景亮度差異過大。

**根本原因分析**:
1.  **Chamber 封閉**: 外部 `constant` emitter 完全被場景的牆壁阻擋，無法提供內部照明。
2.  **獨立光源問題**: 獨立的矩形 `area` emitter 雖然可見，但其光線在 `spectral_polarized` 模式下未被路徑追蹤器正確採樣，導致傢俱等物體仍然很暗。

**解決方案**:
1.  **將天花板網格作為面光源**: 修改 `SceneBuilder._create_meshes` 方法，為 `mesh_ceiling` 幾何體直接添加 `emitter` 屬性。這樣，天花板本身成為一個大型的非偏振面光源，為整個室內空間提供均勻的漫反射照明。
2.  **禁用舊有光源**: 移除了 `SceneBuilder.build` 中對獨立天花板燈陣列 (`CEILING_LIGHTS_ENABLED`) 和 `ambient` 光源的引用，避免冗餘和潛在的光線追蹤問題。
3.  **光源設定**:
    *   `pids_renderer.py` 的 `Config` 類別中新增 `CEILING_EMITTER_INTENSITY = 20.0` 來控制天花板光源的亮度。
    *   天花板材質的 `bsdf` 保持為 `diffuse`，反射率為 `0.85`，同時添加 `emitter` 屬性。

**結果**:
-   背景獲得了均勻的非偏振光照，顯著改善了 I∥ 和 I⊥ 圖像之間的亮度差異。
-   傢俱和室內其他物體現在也被正確照亮，提升了整體渲染質量。

---

### 2025-12-22: 物理偏振片架構 (v3.1.0) ✅

**問題**: `polarizer` BSDF 設計用來過濾穿過的光，而不是作為發射器材質。將它放在 `area emitter` 上時，發射的光可能不會正確經過偏振處理。

**解決方案**: 實施完整的物理偏振片架構：

1. **LED 光源重構**:
   - `led_emitter`: 純粹的非偏振面光源
   - `led_polarizer`: 獨立偏振片幾何體 (θ=0°)，位於光源前方 5mm

2. **相機偏振片**:
   - 左相機: 前方加偏振片 (θ=0°) → 通過 I∥
   - 右相機: 前方加偏振片 (θ=90°) → 通過 I⊥

3. **其他修改**:
   - 玻璃材質: `thindielectric` → `dielectric`（實心玻璃）
   - 像素格式: `rgb` → `luminance`（灰階）

**實測結果 (scene_0002)**:

| 指標 | 數值 | 評價 |
|------|------|------|
| 品質評分 | 87/100 | excellent ✓ |
| 玻璃區域 DoLP | 24.7% | 很好的偏振效果 |
| 背景 DoLP | 3% | 低，符合預期 |
| 玻璃/背景對比 | 8.2x | 優秀的區分度 |
| 背景 I∥/I⊥ 平衡 | 1.0003 | 幾乎完美 ✓ |

**新增功能**:
- `{scene}_DoLP.png` - DoLP 灰階圖
- `{scene}_DoLP_color.png` - DoLP 熱力圖
- `{scene}_report.json` - 完整品質報告（含 DoLP 統計、SNR、品質評分）

---

### 2025-12-22: 平行光軸配置 (v3.2.0) ✅

**問題**: 用戶在檢視 PIDS 論文的 Training Data Collection Standards 後發現，現有實現違反了 Criterion 1 (Geometric Consistency Filtering)。原因是兩個相機都使用 `look_at` 指向同一個目標點，導致光軸會聚（converging），而非平行。

**影響**:
- 垂直視差（vertical disparity）不為零
- 需要額外的 stereo rectification 步驟
- 違反標準立體視覺的假設

**解決方案**:

1. 新增 `Config.forward_direction()`: 計算統一的視線方向向量
2. 新增 `Config.camera_target_for_position()`: 根據相機位置計算個別目標點
3. 修改渲染流程: 為左右相機使用獨立的目標點

**修正後**:
```
左相機: 位置 = (-32.5, 360, 150), 目標 = (-32.5, 611.5, 150)
右相機: 位置 = (+32.5, 360, 150), 目標 = (+32.5, 611.5, 150)
光軸方向: (0, 1, 0) - 兩相機完全平行
```

**預期效果**:
- Vertical disparity ≈ 0（滿足 Criterion 1）
- 無需 stereo rectification
- 視差只出現在水平方向（epipolar line = 水平線）

---

## 版本歷史

| 版本 | 日期 | 說明 |
|------|------|------|
| 3.4.2-nopol | 2025-12-23 | **無偏振版本**：用於消融實驗，移除所有偏振特性但保持相同場景設置 |
| 3.4.2 | 2025-12-22 | **暫定最終版**：完整 5 Criteria 驗證、詳細失敗診斷報告、SPP=16384 |
| 3.4.1 | 2025-12-22 | **深度相機同水平**：深度相機與立體相機同高度，光軸平行 |
| 3.4.0 | 2025-12-22 | **獨立深度相機 + 感測器配置**：Sony IMX296LQR-C, FOV 45.4°, right_parallel 輸出 |
| 3.3.0 | 2025-12-22 | **玻璃 mask 渲染**：計算 Criterion 5 深度有效率 |
| 3.2.0 | 2025-12-22 | **平行光軸配置**：修正相機會聚問題，符合 PIDS Criterion 1 ✅ |
| 3.1.0 | 2025-12-22 | **物理偏振片架構**：獨立偏振片、dielectric玻璃、品質報告 ✅ |
| 3.0.5 | 2025-12-21 | 解決非偏振光源問題，將天花板網格直接設為面光源 |
| 3.0.4 | 2025-12-21 | 嘗試天花板燈陣列，發現光線追蹤問題 |
| 3.0.3 | 2025-12-21 | 添加獨立天花板光源，解決 chamber 封閉問題 |
| 3.0.2 | 2025-12-21 | 回滾偏振處理，保留 OBJ 分離方案 |
| 3.0.1 | 2025-12-21 | 嘗試 V2 偏振放大方法 (失敗) |
| 3.0 | 2025-12-21 | 從頭設計，OBJ 分離解決材質問題 |
| 2.0 | - | 物理正確偏振版 |
| 1.0 | - | 初始版本 |

---

## 品質驗證器 (quality_validator.py v1.4)

### PIDS 品質標準

| Criterion | 名稱 | 閾值 | 驗證方式 |
|-----------|------|------|----------|
| 1 | Geometric Consistency | vertical disparity < 1px | 比較 left_parallel vs right_parallel |
| 2 | Background Photometric Consistency | I∥/I⊥ ∈ [0.5, 2.0] | 從 JSON 報告讀取 |
| 3 | Polarization Signal Validity | 玻璃 DoLP > 10% | 從 JSON 報告讀取 |
| 4 | Ground Truth Alignment | < 10% 正規化誤差 | 用 disparity warp 驗證 |
| 5 | Depth Validity Rate | > 90% in glass region | 渲染時計算，從 JSON 讀取 |

> ⚠️ **模擬場景注意事項**: 對於模擬場景，C1 (Geometric Consistency) 檢查可能產生高誤判率，因為相位相關法在低紋理區域無法正確匹配。模擬場景的相機位置由渲染器精確定義，使用 `--skip-c1` 參數可跳過此檢查。

### 各標準實施方法

#### Criterion 1: Geometric Consistency (幾何一致性)

**目的**: 確保立體相機對的垂直對齊精度

**實施方法**:
1. 讀取左右相機的平行偏振圖像 (I∥_L, I∥_R)
2. 使用相位相關法 (Phase Correlation) 計算全局位移
3. 透過 FFT 交叉功率譜計算亞像素級偏移量
4. 提取垂直分量 (vertical disparity)
5. 判定: |vertical_disparity| < 1.0 pixel

```python
# 相位相關法計算垂直視差
f1, f2 = np.fft.fft2(left), np.fft.fft2(right)
cross_power = (f1 * np.conj(f2)) / |f1 * np.conj(f2)|
correlation = np.fft.ifft2(cross_power)
peak_y, peak_x = find_subpixel_peak(correlation)
vertical_disparity = peak_y  # 應 < 1.0 px
```

#### Criterion 2: Background Photometric Consistency (背景光度一致性)

**目的**: 驗證非偏振光源在背景區域的平衡性

**實施方法**:
1. 使用玻璃遮罩 (glass_mask) 識別背景區域
2. 在背景區域計算 I∥ 和 I⊥ 的平均強度
3. 計算比值 ratio = mean(I∥) / mean(I⊥)
4. 判定: 0.5 ≤ ratio ≤ 2.0

```python
# 背景區域光度比值
bg_mask = ~glass_mask
bg_parallel = img_parallel[bg_mask].mean()
bg_cross = img_cross[bg_mask].mean()
ratio = bg_parallel / bg_cross  # 應在 [0.5, 2.0]
```

#### Criterion 3: Polarization Signal Validity (偏振信號有效性)

**目的**: 確認玻璃區域產生足夠的偏振信號

**實施方法**:
1. 計算每個像素的線偏振度 DoLP = (I∥ - I⊥) / (I∥ + I⊥)
2. 使用玻璃遮罩提取玻璃區域
3. 計算玻璃區域的平均 DoLP
4. 判定: mean(DoLP_glass) > 10%

```python
# 玻璃區域偏振度
dolp = (I_parallel - I_cross) / (I_parallel + I_cross + eps)
glass_dolp = dolp[glass_mask].mean()  # 應 > 0.10
```

#### Criterion 4: Ground Truth Alignment (深度對齊)

**目的**: 驗證視差圖與影像的幾何對應關係

**實施方法**:
1. 讀取左右圖像和視差圖 (disparity)
2. 使用視差對右圖進行 warp 到左視角
3. 計算 warped_right 與 left 的正規化誤差
4. 在有效視差區域計算平均誤差
5. 判定: mean_error < 10% (正規化誤差)

```python
# 視差 warp 對齊檢查
for x in range(width):
    x_src = x + disparity[y, x]
    warped_right[y, x] = right[y, x_src]
error = |warped_right - left| / max_intensity
mean_error = error[valid_mask].mean()  # 應 < 0.10
```

#### Criterion 5: Depth Validity Rate (深度有效率)

**目的**: 確保玻璃區域有足夠的有效深度值

**實施方法**:
1. 讀取深度圖 (depth) 和玻璃遮罩
2. 統計玻璃區域的總像素數
3. 統計玻璃區域中深度值有效 (> 0 且 < ∞) 的像素數
4. 計算有效率 = valid_count / total_count
5. 判定: validity_rate > 90%

```python
# 玻璃區域深度有效率
glass_depth = depth[glass_mask]
valid = (glass_depth > 0) & (glass_depth < inf)
validity_rate = valid.sum() / glass_mask.sum()  # 應 > 0.90
```

### 使用方式

```bash
# 進入 Quality_Assurance 目錄
cd Quality_Assurance

# 驗證單一目錄（完整檢查）
python quality_validator.py --input_dir ../output/output --output ./report/quality_report.md

# 模擬場景模式（跳過 C1 Geometric Consistency）
python quality_validator.py --input_dir ../output/output --output ./report/quality_report.md --skip-c1

# 同時輸出 JSON
python quality_validator.py --input_dir ../output/output --output ./report/quality_report.md --json ./report/quality_report.json

# 跳過 EXR 讀取（快速驗證，不計算 C1 和 C4）
python quality_validator.py --input_dir ../output/output --output ./report/quality_report.md --skip-exr

# 匯總所有場景報告
python aggregate_reports.py --input_dir ../output/output --output ./report/summary_report.md
```

### 命令行參數

| 參數 | 說明 |
|------|------|
| `--input_dir` | 包含 `*_report.json` 的目錄 (必需) |
| `--output` | 輸出 Markdown 報告路徑 (預設: `quality_report.md`) |
| `--json` | 同時輸出 JSON 格式報告 |
| `--skip-c1` | 跳過 C1 (Geometric Consistency) 檢查 - **模擬場景專用** |
| `--skip-exr` | 跳過 EXR 讀取（不計算 vertical disparity 和 ground truth alignment）|

### v1.4 更新說明

1. **新增 `--skip-c1` 參數**: 模擬場景相機位置由渲染器精確定義，C1 檢查不適用
2. **自動過濾中間檔案**: 自動排除 `*_glass_report.json` 和 `*_other_report.json`（OBJ 分離的中間產物）
3. **改進報告格式**: 顯示跳過的檢查項，清楚標示模擬場景模式

### 報告格式

報告會產生：
1. **總覽表格**: 所有場景的 5 個 Criterion 結果
2. **不合規詳細資訊**: 只對失敗場景顯示詳細診斷，包含：
   - 目的說明
   - 測量結果（實際值、閾值、差距）
   - 計算方法
   - 可能原因與建議
