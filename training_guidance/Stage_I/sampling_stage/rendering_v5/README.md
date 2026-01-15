# PIDS Renderer V5

> Copyright (c) 2025-2026 Po-Ting Lin
> Released under the MIT License (see LICENSE file).

## 概述

PIDS Renderer V5 是 PIDS (Physics-Informed Deep Stereo) 專案的最新渲染器版本，用於生成 Stage 1 合成訓練數據。

**版本**: 5.1.6
**日期**: 2026-01-06

## 文件說明

| 文件 | 版本 | 說明 |
|------|------|------|
| `pids_renderer_textured.py` | v5.1.6 | 偏振渲染器（主要）- 整合 organize_dataset + 修正 JSON 報告 |
| `pids_renderer_textured_nopol.py` | v4.0.0-nopol | 無偏振渲染器 - 用於消融實驗 |
| `Quality_Assurance/quality_validator.py` | v1.6.0 | 品質驗證器 - 玻璃比值雙向對比 |

## V5 相比 V4 的改進

### v5.1.6 新增 (2026-01-06) - 整合 organize_dataset + 修正報告

- **整合 organize_dataset 功能**:
  - 渲染完成後自動整理數據集
  - 新增參數：`--organize`, `--train_size`, `--organize_seed`, `--organize_move`
  - 預設使用複製模式（保留原始檔案）
  - 自動排除 QA 不合格場景

- **修正 JSON 報告**:
  - 新增 `stokes_ratio` 欄位：I(90°)/I(0°) 從 Stokes 參數計算
  - `glass_region.stokes_ratio`: 玻璃區域偏振比值
  - `background_region.stokes_ratio`: 背景區域偏振比值

- **修正多 GPU 模式**:
  - 支援 `--skip-c1`, `--organize` 等新參數
  - organize 只在主進程執行一次（所有 GPU 完成後）

### v5.1.5 新增 (2026-01-06) - ✅ 最終配置，準備批渲染

- **最終光學配置**:
  | 參數 | 值 | 說明 |
  |------|------|------|
  | GLASS_IOR | 1.65 | 固定，正入射 R≈6.2% |
  | LED_INTENSITY | 10000 | 偏振光源 |
  | CEILING_INTENSITY | 800 | 非偏振頂光 |
  | 光源比例 | 12.5:1 | 偏振/非偏振 |

- **驗證結果**:
  | 指標 | 值 | 狀態 |
  |------|------|------|
  | 玻璃 I(90°)/I(0°) | 1.73x | ✓ |
  | 背景 I(90°)/I(0°) | 1.00x | ✓ 完美去偏振 |
  | 背景平衡 (warp) | 1.11x | ✓ |
  | SNR | 13.33 | ✓ |
  | 品質分數 | 92 | ✓ excellent |

- **隨機化範圍**:
  - LED: [8000, 12000]
  - Ceiling: [600, 1000]
  - CAMERA_X: [-120, 20]

### v5.1.4 新增 (2026-01-06) - 🔧 關鍵修正：Ray Depth → Z Depth 轉換

- **修正深度類型**: Mitsuba 輸出 ray depth (歐幾里得距離)，視差公式需要 Z depth (垂直距離)
  - 問題：邊緣像素 ray depth 比 Z depth 大 ~8%，造成視差誤差 5-10 px
  - 修正：`Z = ray_depth * cos(angle)`，其中 `cos = f / sqrt(f² + dx² + dy²)`
  - 影響：邊緣視差從 ~10 px 誤差改善到 <0.1 px

- **Sanity check 驗證**:
  - `sanity_check_warp.py`: 驗證 warp 方向正確性
  - `sanity_check_depth_cam.py`: 分析 ray depth vs Z depth 差異

### v5.1.3 新增 (2026-01-06) - 🔧 關鍵修正：Warp 方向 + 深度相機位置

- **修正 Warp 方向**: `xx + disparity` → `xx - disparity`
  - 問題分析：
    - 左相機 X=-82.5mm（較左），右相機 X=-17.5mm（較右）
    - 同一 3D 點：在左圖 x 座標較大，在右圖 x 座標較小
    - 因此 disparity = x_left - x_right > 0
  - 修正：要從右圖找來源像素，應該是 `x_right = x_left - disparity`
  - 影響：確保 warp 對齊正確，I∥/I⊥ 比較同一 3D 點

- **修正深度相機位置**: 中間位置 → 左相機位置
  - 問題：原本深度相機在中間 X=-50mm，但視差 GT 應該從左相機視角計算
  - 修正：`depth_camera_position()` 現在返回 `left_camera_position()`
  - 影響：確保視差 GT 與訓練時的左圖視角一致

- **試用正常玻璃 IOR**: GLASS_IOR 從 3.5 改為 1.5（測試用）

### v5.1.2 新增 (2026-01-06) - 精簡報告 + 修正背景平衡

- **精簡報告輸出**: 只保留 6 個關鍵指標
  ```
  ┌─────────────────────────────────────────┐
  │ 玻璃 I(90°)/I(0°)  :  2.201x            │  ← 偏振對比度
  │ 背景 I(90°)/I(0°)  :  1.000x            │  ← 背景去偏振
  │ 背景平衡 (warp)    :   1.06x ✓          │  ← RAFT 需要
  │ SNR               :  16.84              │  ← 信噪比
  │ 玻璃深度有效率    : 100.0% ✓            │  ← 深度品質
  │ 品質分數          :  92 (excellent)     │  ← 總評
  └─────────────────────────────────────────┘
  ```

- **修正背景平衡計算**: 使用 warp 後的 I_cross_warped
  - 原本：比較不同 3D 點（左相機位置 vs 右相機位置）
  - 修正：比較同一 3D 點（warp 對齊後）
  - 結果：背景平衡從 1.30x 改善到 **1.06x**

- **刪除冗餘指標**: DoLP 相關、I∥/I⊥ warp/全局 等被 Stokes 計算取代

### v5.1.1 新增 (2026-01-06) - 整面偏振光源 + 非偏振頂光 ✓ 成功

- **整面偏振光源**: 大面積偏振光源（500x250mm）均勻照明
  - 原理：漫反射會去偏振，鏡面反射保持偏振
  - 位置：相機同側（Y=380），面向場景
  - 強度：5000（隨機範圍 4000-6000）

- **非偏振頂光**: 稀釋背景殘餘偏振
  - 原理：背景漫反射需要多次 bounce 才能完全去偏振，加入非偏振光稀釋
  - 強度：2000（隨機範圍 1500-2500）

- **實測結果**:
  | 指標 | 值 |
  |------|-----|
  | 玻璃 I(90°)/I(0°) | **2.2x** ✓ |
  | 背景 I(90°)/I(0°) | **1.0x** ✓ |
  | 背景平衡 (warp) | **1.30x** ✓ |
  | 品質分數 | **92分** |

- **PNG 預覽修復**:
  - 使用 99.5 percentile 代替 max，避免極端亮點
  - 添加 gamma correction (γ=2.2)

### v5.1.0 嘗試 (2026-01-06) - 純整面偏振光源

- **整面偏振光源**: 500x250mm 均勻照明，無頂光
  - **結果**：
    - 玻璃 I(90°)/I(0°) = 2.238x ✓
    - 背景 I(90°)/I(0°) = 1.000x ✓
    - 背景平衡 (warp) = 2.08x ⚠️ 不平衡
  - **問題**：背景殘餘偏振未完全去除，需要加入非偏振頂光稀釋

### v5.0.5 嘗試 (2026-01-06) - Brewster 角配置（失敗）

- **Brewster 角配置**: LED 以 56° 入射角照射玻璃
  - 幾何：LED 在 (300, 400, 150)，玻璃中心 (0, 600, 150)
  - 入射角 = arctan(300/200) = 56.3° = Brewster 角 (n=1.5)
  - **結果**：
    - I∥/I⊥ 玻璃 (warp): 2.87x ← 幾何效應（LED 在右側）
    - I(90°)/I(0°) 玻璃: 0.980x ← 真正偏振只有 2%
    - 背景平衡: 2.79x ⚠️ 也不平衡！
  - **問題**：LED 位置造成整體亮度不對稱，不是偏振效應

## 偏振信號弱的根本原因分析 (2026-01-06)

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

### 嘗試的解決方案

| 方案 | 結果 | 問題 |
|------|------|------|
| 縮小 LED (30x30mm) | S1/S0 ≈ 5.8% | 無明顯改善 |
| Brewster 角 (56°) | 幾何 2.87x，偏振 2% | 背景也不平衡 |
| 整面偏振光源 | 測試中 | - |

### v5.0.4 新增 (2026-01-06)

- **世界坐標對齊偏振角度**: 新增 `compute_world_aligned_theta()` 函數，解決相機/LED 移動時偏振角度 shift 的問題
  - 問題：Mitsuba 的 polarizer theta 是相對於 look_at 局部坐標系，當位置改變時局部坐標系旋轉
  - 解決：計算補償角度，確保世界坐標中的偏振方向保持一致
  - LED 和相機偏振片都使用此函數，保證不同 CAMERA_X 下偏振一致

### v5.0.3 新增 (2026-01-06)

- **LED 同步相機位置**: LED 跟隨 CAMERA_X 移動並指向相同目標
- **Stokes → Malus 量化**: 使用 Stokes 參數直接計算同視角偏振比值 I(90°)/I(0°)，無需額外渲染
  - `I(0°) = 0.5 * (S0 + S1)`
  - `I(90°) = 0.5 * (S0 - S1)`
- **高 IOR 玻璃**: GLASS_IOR 提升至 3.5，增加正入射 Fresnel 反射（R ≈ 31%）
- **純偏振照明**: 完全關閉頂光 (CEILING_EMITTER_INTENSITY = 0)，只用偏振 LED

### v5.0.0-5.0.2 改進

- **Warp 對齊計算**: 新增 `warp_right_to_left()` 函數，使用視差將右圖 warp 到左視角，確保 I∥/I⊥ 比較同一 3D 點
- **玻璃比值指標**: QA 從 DoLP 改為 I∥/I⊥ 比值 (`glass_ratio`)，更直觀且更穩定
- **高強度偏振光**: LED 強度提升至 20000，環境光降至 1，偏振/環境光比 ~20000:1
- **DOLP_FLOOR**: 新增 0.01 下限，避免除以零導致的異常比值
- **正確的 mask 使用**: warp 後使用 `glass_mask_left` 而非聯合 mask

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

主要參數：
- 感測器: Sony IMX296LQR-C
- FOV: 45.4°
- Baseline: 65mm
- SPP: 8192 (預設)

### 光源設定 (v5.1.1)

| 參數 | 預設值 | 隨機範圍 | 說明 |
|------|--------|----------|------|
| 偏振光源強度 | 5000 | 4000-6000 | 整面偏振光源 (500x250mm) |
| 偏振光源位置 | (0, 380, 150) | - | 相機同側，面向場景 |
| 非偏振頂光強度 | 2000 | 1500-2500 | 稀釋背景殘餘偏振 |
| GLASS_IOR | 3.5 | - | 高 IOR 增加 Fresnel 反射 |

### QA 閾值 (quality_validator v1.6.0)

| 指標 | 閾值 | 說明 |
|------|------|------|
| C3 glass_ratio | < 0.96x 或 > 1.04x | 玻璃區域 I∥/I⊥ 比值（雙向對比）|
| DOLP_FLOOR | 0.01 | 背景 DoLP 下限，避免除零 |

## 座標系統

與 V3 相同，OBJ 匯出設定 `forward=-Y, up=Z`。

## 版本歷史

| 版本 | 日期 | 說明 |
|------|------|------|
| 5.1.6 | 2026-01-06 | 整合 organize_dataset + JSON 報告新增 stokes_ratio + 多 GPU 參數修正 |
| 5.1.5 | 2026-01-06 | ✅ 最終配置：IOR=1.65, LED=10000, Ceiling=800, 玻璃對比 1.73x |
| 5.1.4 | 2026-01-06 | 🔧 關鍵修正：ray depth → Z depth 轉換，修正邊緣視差 5-10 px 誤差 |
| 5.1.3 | 2026-01-06 | 🔧 關鍵修正：warp 方向 `xx-disparity`、深度相機改用左相機位置 |
| 5.1.2 | 2026-01-06 | 精簡報告（6 指標）+ 修正背景平衡計算（warp 對齊），1.30x → 1.06x |
| 5.1.1 | 2026-01-06 | ✓ 成功：整面偏振光源 + 非偏振頂光，玻璃 2.2x，背景 1.0x，品質 92 分 |
| 5.1.0 | 2026-01-06 | 整面偏振光源：玻璃 2.24x 但背景 warp 不平衡 2.08x |
| 5.0.5 | 2026-01-06 | Brewster 角配置（失敗）：56° 入射角，幾何 2.87x 但偏振只有 2%，背景不平衡 |
| 5.0.4 | 2026-01-06 | 世界坐標對齊偏振角度，解決 CAMERA_X 移動導致的 theta shift |
| 5.0.3 | 2026-01-06 | LED 同步相機、Stokes→Malus 量化、IOR=3.5、關閉頂光 |
| 5.0.2 | 2026-01-06 | 高強度偏振光：LED 20000, 環境光 1，最大化 I∥/I⊥ 比值 |
| 5.0.1 | 2026-01-06 | 修正 warp 方向：`xx + disparity`，玻璃比值 1.22x |
| 5.0.0 | 2026-01-06 | Warp 對齊、DOLP_FLOOR、QA 改用 glass_ratio 指標 |
| 4.0.0 | 2026-01-01 | 紋理支持、隨機化增強、nopol 版本 |
| 4.0.0-nopol | 2026-01-01 | 無偏振版本，用於消融實驗 |
