# PIDS Renderer V4

> Copyright (c) 2025-2026 Po-Ting Lin
> Released under the MIT License (see LICENSE file).

## 概述

PIDS Renderer V4 是 PIDS (Physics-Informed Deep Stereo) 專案的最新渲染器版本，用於生成 Stage 1 合成訓練數據。

**版本**: 4.0.0
**日期**: 2026-01-01

## 文件說明

| 文件 | 版本 | 說明 |
|------|------|------|
| `pids_renderer_textured.py` | v4.0.0 | 偏振渲染器（主要）- 支援紋理貼圖 |
| `pids_renderer_textured_nopol.py` | v4.0.0-nopol | 無偏振渲染器 - 用於消融實驗 |

## V4 相比 V3 的改進

- **紋理支持**: 支援 MTL 的 `map_Kd` 紋理貼圖
- **更強的隨機化**: LED 強度、環境光、相機位置的數據增強
- **更完整的報告**: 場景品質報告包含更多診斷信息
- **更清晰的架構**: 模組化設計，更易維護

## 使用方式

### 偏振渲染（主要用途）

```bash
# 單一場景
python pids_renderer_textured.py --scene scene_0001.obj --output ./output

# 批次渲染
python pids_renderer_textured.py --input_dir ./scenes --output ./output --max_scenes 100

# 多 GPU 並行
python pids_renderer_textured.py --input_dir ./scenes --output ./output --num_gpus 4
```

### 無偏振渲染（消融實驗用）

```bash
# 單一場景
python pids_renderer_textured_nopol.py --scene scene_0001.obj --output ./output_nopol

# 批次渲染
python pids_renderer_textured_nopol.py --input_dir ./scenes --output ./output_nopol --max_scenes 100
```

## 輸出文件

### 偏振版輸出

| 檔案 | 說明 |
|------|------|
| `*_left_parallel.exr` | 左相機 I∥ (0° 偏振) |
| `*_right_cross.exr` | 右相機 I⊥ (90° 偏振) |
| `*_depth.exr` | 深度圖 |
| `*_disparity.exr` | 視差圖 |
| `*_glass_mask.exr` | 玻璃區域 mask |
| `*_params.json` | 渲染參數 |
| `*_report.json` | 品質報告 |

### 無偏振版輸出

| 檔案 | 說明 |
|------|------|
| `*_left.exr` | 左相機 (無偏振) |
| `*_right.exr` | 右相機 (無偏振) |
| `*_depth.exr` | 深度圖 |
| `*_disparity.exr` | 視差圖 |
| `*_glass_mask.exr` | 玻璃區域 mask |
| `*_params.json` | 渲染參數 |

## 偏振 vs 無偏振對比

| 特性 | 偏振版 | 無偏振版 |
|------|--------|----------|
| Mitsuba variant | `spectral_polarized` | `rgb` |
| Integrator | `stokes` | `path` |
| LED 偏振片 | 有 (0°) | 無 |
| 相機偏振片 | 有 (0°/90°) | 無 |
| 玻璃對比度 | 高 (I∥ >> I⊥) | 低 |
| 用途 | 訓練 PIDS 模型 | 消融實驗基線 |

## 配置參數

與 V3 相同，詳見 `rendering_v3/README.md`。

主要參數：
- 感測器: Sony IMX296LQR-C
- FOV: 45.4°
- Baseline: 65mm
- SPP: 8192 (預設)

## 座標系統

與 V3 相同，OBJ 匯出設定 `forward=-Y, up=Z`。

## 版本歷史

| 版本 | 日期 | 說明 |
|------|------|------|
| 4.0.0 | 2026-01-01 | 紋理支持、隨機化增強、nopol 版本 |
| 4.0.0-nopol | 2026-01-01 | 無偏振版本，用於消融實驗 |
