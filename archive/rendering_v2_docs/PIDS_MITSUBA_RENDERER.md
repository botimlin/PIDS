# PIDS Mitsuba 3 渲染器

基於 Mitsuba 3 的偏振立體渲染器，用於生成 PIDS (Polarization-based Intelligent Depth Sensing) 訓練資料。

## 系統需求

- Python 3.8+
- Mitsuba 3
- CUDA GPU（推薦 RTX 5090 或同等級）
- 依賴套件：numpy, opencv-python, pillow, matplotlib

## 安裝

```bash
pip install mitsuba drjit numpy opencv-python pillow matplotlib
```

## 快速開始

```bash
# 單一場景渲染
python pids_mitsuba_renderer.py \
  --scene_file /path/to/scene.obj \
  --output_dir /path/to/output

# 批次渲染
python pids_mitsuba_renderer.py \
  --input_dir /path/to/scenes \
  --output_dir /path/to/output \
  --max_scenes 100
```

## 偏振物理原理

### Brewster 角效應

當光線以 Brewster 角（~56°，對於 n=1.5 的玻璃）入射時：

| 偏振方向 | 符號 | 反射率 | 玻璃外觀 |
|----------|------|--------|----------|
| p-偏振 (平行) | I∥ | Rp ≈ 0% | 幾乎透明 |
| s-偏振 (垂直) | I⊥ | Rs ≈ 15% | 有反射 |

### 渲染實現

使用 `thindielectric` 材質，通過不同 IOR 模擬偏振效果：

```
左相機 (I∥): IOR = 1.1  → 低反射（模擬 p-偏振）
右相機 (I⊥): IOR = 1.5  → 正常反射（模擬 s-偏振）
```

## 立體相機配置

```
        ← 65mm (基線) →
    [左相機]         [右相機]
    (-32.5mm)       (+32.5mm)
       ↓               ↓
      I∥              I⊥
   (parallel)       (cross)
```

- **基線**: 65mm
- **視場角**: 65°
- **解析度**: 640×480
- **光軸**: 平行（非 toe-in）

## 輸出檔案

每個場景生成以下檔案：

| 檔案 | 格式 | 說明 |
|------|------|------|
| `{scene}_left_parallel.exr` | EXR | 左相機 I∥（p-偏振，透明玻璃）|
| `{scene}_left_parallel.png` | PNG | 預覽圖 |
| `{scene}_right_cross.exr` | EXR | 右相機 I⊥（s-偏振，有反射）|
| `{scene}_right_cross.png` | PNG | 預覽圖 |
| `{scene}_depth.exr` | EXR | 深度圖（米）|
| `{scene}_depth.png` | PNG | 深度熱力圖 |
| `{scene}_disparity.exr` | EXR | 視差圖（像素）|
| `{scene}_disparity.png` | PNG | 視差熱力圖 |
| `{scene}_mask.png` | PNG | 透明物體遮罩 |
| `{scene}_params.json` | JSON | 渲染參數 |

## 命令行參數

### 輸入選項
| 參數 | 說明 |
|------|------|
| `--scene_file` | 單一 OBJ 場景檔案路徑 |
| `--input_dir` | 包含 OBJ 檔案的目錄（批次渲染）|

### 輸出選項
| 參數 | 說明 |
|------|------|
| `--output_dir` | 輸出目錄（必需）|

### 渲染選項
| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--spp` | 65536 | 每像素樣本數 |
| `--spp_per_batch` | 2048 | 每批次最大 SPP（避免 GPU 內存溢出）|
| `--width` | 640 | 輸出寬度 |
| `--height` | 480 | 輸出高度 |
| `--max_scenes` | None | 最大渲染場景數（批次模式）|

### 其他選項
| 參數 | 說明 |
|------|------|
| `--no_preview` | 不儲存預覽 PNG 圖 |
| `--denoise` | 使用降噪（預設關閉以保持細節）|
| `--variant` | 指定 Mitsuba variant（如 `cuda_ad_rgb`）|

## 渲染設定

### 預設配置

```python
CONFIG = {
    'render': {
        'width': 640,
        'height': 480,
        'spp': 65536,           # 高採樣率減少噪點
        'spp_per_batch': 2048,  # 分批渲染避免 OOM
        'max_depth': 6,         # 光線反彈次數
    },
    'camera': {
        'baseline': 65.0,       # 基線 (mm)
        'fov': 65.0,            # 視場角 (度)
    },
    'lighting': {
        'led': {
            'intensity': 25.0,
            'size_x': 180.0,    # 光源尺寸 (mm)
            'size_y': 100.0,
        },
        'ambient': {
            'intensity': 0.05,
        },
    },
    'materials': {
        'glass': {
            'ior': 1.5,
        },
    },
}
```

### 分批渲染

為避免 GPU 內存溢出，高 SPP 渲染會自動分批：

```
總 SPP = 65536
每批 SPP = 2048
批次數 = 32

渲染過程：
[渲染] 批次 1/32, SPP=2048
[渲染] 批次 2/32, SPP=2048
...
[渲染] 批次 32/32, SPP=2048
[渲染] 完成: 總 SPP=65536
```

## 座標系統

### 轉換關係（OBJ → Mitsuba）

```
OBJ/Blender          Mitsuba (經過 -90° X 軸旋轉)
    Z (高度)    →        Y (高度)
    Y (深度)    →        Z (深度)
    X (左右)    →        X (左右)
```

### 場景佈局

```
                    後牆 (Z = 900mm)
                         │
    ┌────────────────────┴────────────────────┐
    │                                          │
    │              [物體區域]                   │
    │           (Z = 528~695mm)                │
    │                                          │
    │    [光源]                                │
    │   (Y=290mm, Z=450mm)                    │
    │         ↘                               │
    │           ↘  ~56° (Brewster 角)         │
    │             ↘                           │
    │    [左相機]  ↘  [右相機]                 │
    │   (-32.5mm)     (+32.5mm)               │
    │        (Y=100mm, Z=360mm)               │
    └──────────────────────────────────────────┘
                    前牆 (Z = 350mm)
```

## 材質識別

渲染器自動識別 OBJ/MTL 中的材質類型：

| 關鍵字 | 材質類型 |
|--------|----------|
| `glass`, `transparent`, `clear` | 玻璃 (thindielectric) |
| 其他 | 漫反射 (diffuse) |

## 使用範例

### 高品質渲染

```bash
python pids_mitsuba_renderer.py \
  --spp 65536 \
  --scene_file scene.obj \
  --output_dir ./output
```

### 快速預覽

```bash
python pids_mitsuba_renderer.py \
  --spp 4096 \
  --scene_file scene.obj \
  --output_dir ./output
```

### GPU 內存不足時

```bash
python pids_mitsuba_renderer.py \
  --spp 65536 \
  --spp_per_batch 1024 \
  --scene_file scene.obj \
  --output_dir ./output
```

### 批次渲染 100 個場景

```bash
python pids_mitsuba_renderer.py \
  --input_dir /workspace/scenes \
  --output_dir /workspace/output \
  --max_scenes 100
```

## 質量檢測

配套的質量檢測腳本 `training_data_quality_check.py` 可以驗證渲染結果：

```bash
python training_data_quality_check.py \
  --input_dir /workspace/output \
  --report_only
```

檢測項目：
1. **幾何一致性**: Y 軸對齊（允許 X 軸視差）
2. **深度有效性**: 深度值有效率 > 80%

## 已知限制

1. **焦散噪點**: 玻璃下方地面可能有噪點（路徑追蹤的固有問題）
2. **偏振模擬**: 使用 IOR 差異近似偏振效果，非物理正確的偏振渲染
3. **內存需求**: 高 SPP 渲染需要大量 GPU 內存

## 版本資訊

- **版本**: 1.0
- **日期**: 2025-12-20
- **Mitsuba**: 3.x
- **CUDA**: 支援 RTX 5090
