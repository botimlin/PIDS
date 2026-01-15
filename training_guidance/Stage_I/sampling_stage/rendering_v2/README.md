# PIDS Mitsuba 3 渲染系統

> Copyright (c) 2025-2026 Po-Ting Lin
> Released under the MIT License (see LICENSE file).

用於 Physics-Informed Deep Stereo (PIDS) 專案的 Stage I 訓練數據生成。

## 📁 檔案結構

```
├── pids_mitsuba_renderer.py    # 主渲染腳本
├── pids_renderer_advanced.py   # 進階版 (更精確的偏振模型)
├── pids_config.py              # 配置檔案
├── test_mitsuba_setup.py       # 環境測試腳本
└── README.md                   # 本說明文件
```

## 🚀 快速開始

### 1. 安裝依賴

```bash
# 安裝 Mitsuba 3 (CUDA 版本)
pip install mitsuba

# 安裝其他依賴
pip install numpy Pillow
```

### 2. 測試環境

```bash
python test_mitsuba_setup.py
```

確認輸出包含:
- ✓ Mitsuba 已安裝
- ✓ CUDA 加速可用
- ✓ 渲染成功

### 3. 渲染場景

```bash
# 單一場景
python pids_mitsuba_renderer.py --scene_file ./scenes/scene_0001.obj --output_dir ./output

# 批次渲染
python pids_mitsuba_renderer.py --input_dir ./scenes_output --output_dir ./rendered_output

# 調整品質
python pids_mitsuba_renderer.py --scene_file scene.obj --output_dir ./output --spp 512
```

## 📋 輸出檔案

每個場景會生成以下檔案 (**灰階影像**):

| 檔案 | 格式 | 說明 |
|------|------|------|
| `*_I_parallel.exr` | 32-bit HDR 灰階 | 左相機影像 (I∥, 0° 偏振) |
| `*_I_cross.exr` | 32-bit HDR 灰階 | 右相機影像 (I⊥, 90° 偏振) |
| `*_depth.exr` | 32-bit float | 深度圖 (meters) |
| `*_disparity.exr` | 32-bit float | 視差圖 (pixels) |
| `*_params.json` | JSON | 場景參數 |
| `*.png` | 8-bit 灰階 | 預覽圖 (可選) |

> **注意**: 所有影像輸出為**單通道灰階格式**，符合實際偏振相機的輸出特性。

## ⚙️ 配置說明

### 相機配置

```python
# pids_config.py

CAMERA_CONFIG = {
    'baseline': 65.0,         # 基線 65mm
    'fov': 65.0,              # 視場角 65°
    'focus_distance': 600.0,  # 對焦距離 600mm
    'position_y': 150.0,      # 相機高度 (chamber 中心)
    'position_z': 0.0,        # 相機深度位置
}
```

### Chamber 尺寸 (對應 Blender 腳本)

```python
CHAMBER_CONFIG = {
    'width': 600.0,       # 寬度 600mm
    'height': 300.0,      # 高度 300mm
    'z_start': 350.0,     # 前牆 (Blender Y=350)
    'z_end': 900.0,       # 後牆 (Blender Y=900)
    'dof_near': 528.0,    # 景深近端
    'dof_far': 695.0,     # 景深遠端
}
```

### 光源配置 (Stage I)

```python
LIGHTING_CONFIG = {
    'led': {
        'enabled': True,
        'intensity': 5.0,
        'offset_y': 200.0,    # 上方 200mm
        'offset_z': -100.0,   # 後方 100mm
        # 入射角 ≈ 55° (Brewster angle for glass)
    },
    'ambient': {
        'enabled': False,     # Stage I 關閉環境光
    },
}
```

## 🔬 偏振物理模型

### 理論基礎

根據論文公式:

```
I∥(u) = α · (Id + Ib) + Is    # 平行偏振 (保留鏡面反射)
I⊥(u) = α · (Id + Ib)          # 交叉偏振 (抑制鏡面反射)
```

- `α ≈ 0.5`: 偏振片透射係數
- `Id`: 漫反射分量
- `Ib`: 背景光分量
- `Is`: 鏡面反射分量 (偏振)

### 渲染方法

**方法 A: 材質修改法** (預設，較快)
- I∥: 使用 `dielectric` 材質
- I⊥: 使用 `roughdielectric` 材質 (增加粗糙度抑制反射)

**方法 B: 分離渲染法** (更精確)
- 分別渲染完整場景和純漫反射場景
- 計算鏡面分量並組合

```bash
# 使用分離渲染法
python pids_renderer_advanced.py --scene_file scene.obj --output_dir ./output --method separate
```

## 📐 座標系統轉換

### Blender → Mitsuba

| Blender | Mitsuba | 說明 |
|---------|---------|------|
| X | X | 右 |
| Y | Z | 深度/前 |
| Z | Y | 上 |

OBJ 匯出設定: `forward=-Y, up=Z`

### 單位轉換

- Blender 單位: mm
- Mitsuba 單位: m
- 轉換: `to_world: scale([0.001, 0.001, 0.001])`

## 🎯 視差計算

```python
# 計算公式
focal_length_px = width / (2 × tan(FOV/2))
                = 640 / (2 × tan(32.5°))
                ≈ 502 pixels

disparity = (focal_length × baseline) / depth
```

| 深度 (mm) | 視差 (px) | 說明 |
|-----------|-----------|------|
| 528 | 62 | 景深近端 |
| 600 | 54 | 對焦點 |
| 695 | 47 | 景深遠端 |
| 900 | 36 | 背景牆 |

## 🔧 進階用法

### 自訂配置

```python
from pids_renderer_advanced import PIDSRenderer
from pids_config import get_stage1_config

# 取得配置並修改
config = get_stage1_config()
config['render']['spp'] = 512
config['render']['width'] = 1280
config['render']['height'] = 960

# 建立渲染器
renderer = PIDSRenderer(config=config, method='separate')

# 渲染
renderer.render('scene.obj', './output')
```

### 批次處理

```python
from pids_renderer_advanced import render_batch

render_batch(
    input_dir='./scenes_output',
    output_dir='./rendered_output',
    method='material',
    max_scenes=100
)
```

## ⚠️ 常見問題

### Q: CUDA 不可用?

確認安裝了 CUDA toolkit 並安裝 CUDA 版 Mitsuba:
```bash
pip install mitsuba
```

### Q: OBJ 載入失敗?

確認 OBJ 匯出設定:
- Forward: `-Y`
- Up: `Z`
- Scale: `0.001` (或在 Mitsuba 中轉換)

### Q: 深度圖全黑?

檢查場景是否有光源，以及物體是否在相機視野內。

## 📚 參考資料

- [PIDS 論文](./PIDS__17_.pdf)
- [Blender 場景指南](./blender_guide_v2.md)
- [Mitsuba 3 文檔](https://mitsuba.readthedocs.io/)

## 📝 版本歷史

- v1.0: 初始版本，支援 Stage I 渲染
