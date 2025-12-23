# Training Data Quality Check

## 概述

本腳本根據論文中的 **Training Data Collection Standards** 實現五項自動化質量控制標準，用於篩選高品質的訓練數據。

## 五項質量控制標準

### 1. Geometric Consistency Filtering（幾何一致性過濾）

**目的**：確保立體對的幾何校正正確

**方法**：
- 使用 ORB 特徵檢測器提取背景區域的關鍵點
- 計算左右圖像匹配點的垂直視差
- 閾值：平均垂直視差 ≤ 1 pixel

**失敗原因**：
- 相機未正確校正
- 立體對存在旋轉偏差

```python
# 檢測邏輯
avg_vertical_disparity = mean(|y_left - y_right|)
passed = avg_vertical_disparity <= 1.0
```

---

### 2. Background Photometric Consistency Filtering（背景光度一致性過濾）

**目的**：驗證光照穩定性和輻射校準

**方法**：
- 評估背景區域在偏振通道間的亮度差異
- 漫反射背景應該幾乎沒有偏振效果
- 閾值：|I∥ - I⊥| ≤ 0.05（正規化後）

**失敗原因**：
- 光源不穩定
- 曝光設定不一致
- 偏振片角度錯誤

```python
# 檢測邏輯
background_diff = mean(|I_parallel - I_cross|) in background_region
passed = background_diff <= 0.05
```

---

### 3. Polarization Signal Validity Filtering（偏振信號有效性過濾）

**目的**：確保透明表面有足夠的偏振對比度

**方法**：
- 檢查透明物體高光區域
- 確保 I∥ > I⊥（平行偏振應該有更強的鏡面反射）
- 閾值：對比度 ≥ 0.02

**失敗原因**：
- 入射角遠離 Brewster angle
- 偏振片對齊錯誤
- 物體材質問題

```python
# 檢測邏輯
polarization_contrast = mean(I_parallel - I_cross) in specular_region
polarization_ratio = mean(I_parallel) / mean(I_cross)
passed = (polarization_contrast >= 0.02) and (polarization_ratio > 1.0)
```

---

### 4. Ground Truth Alignment Filtering（Ground Truth 對齊過濾）

**目的**：確保輸入圖像與深度 Ground Truth 精確對齊

**方法**：
- 比較輸入圖像與深度傳感器 RGB 參考圖的邊緣結構
- 使用 Canny 邊緣檢測
- 閾值：位移誤差 ≤ 1 pixel

**失敗原因**：
- proxy 替換時對齊不準確
- 相機移動
- 時間同步問題

```python
# 檢測邏輯
edges_input = Canny(I_parallel)
edges_ref = Canny(RGB_reference)
alignment_error = compute_displacement(edges_input, edges_ref)
passed = alignment_error <= 1.0
```

---

### 5. Depth Validity Rate Filtering（深度有效率過濾）

**目的**：確保深度 Ground Truth 有足夠的覆蓋率

**方法**：
- 計算透明物體遮罩內的有效深度像素比例
- 閾值：有效率 ≥ 90%

**失敗原因**：
- 深度傳感器噪聲
- 透明表面反射導致深度缺失
- proxy 遮擋不完整

```python
# 檢測邏輯
valid_depth = (depth > 0.001) & (depth < 10.0)
validity_rate = sum(valid_depth & mask) / sum(mask)
passed = validity_rate >= 0.90
```

---

## 檔案命名規則

腳本會自動識別以下命名格式的檔案：

### 新格式（推薦）- 立體偏振系統
| 檔案類型 | 命名格式 | 說明 |
|----------|----------|------|
| 左相機平行偏振 | `{scene}_left_parallel.exr` | I∥ (左相機) |
| 右相機交叉偏振 | `{scene}_right_cross.exr` | I⊥ (右相機) |
| 深度圖 | `{scene}_depth.exr` | Ground Truth |
| 視差圖 | `{scene}_disparity.exr` | 視差 |
| 透明物體遮罩 | `{scene}_mask.png` | 二值遮罩 |

**渲染器輸出檔案（pids_mitsuba_renderer.py）**：
```
scene_0001_left_parallel.exr    # 左相機 I∥
scene_0001_left_parallel.png    # 預覽
scene_0001_right_cross.exr      # 右相機 I⊥
scene_0001_right_cross.png      # 預覽
scene_0001_depth.exr            # 深度 GT
scene_0001_depth.png            # 深度熱力圖
scene_0001_disparity.exr        # 視差圖
scene_0001_disparity.png        # 視差熱力圖
scene_0001_mask.png             # 透明物體遮罩
scene_0001_params.json          # 參數檔案
```

### 舊格式（仍支援）
| 檔案類型 | 命名格式 |
|----------|----------|
| 左相機平行偏振 | `{scene}_left_parallel.exr` |
| 左相機交叉偏振 | `{scene}_left_cross.exr` |
| 右相機平行偏振 | `{scene}_right_parallel.exr` |
| 右相機交叉偏振 | `{scene}_right_cross.exr` |

---

## 使用方式

### 基本使用

```bash
# 檢測並複製通過的樣本
python training_data_quality_check.py \
    --input_dir ./raw_dataset \
    --output_dir ./filtered_dataset

# 僅生成報告
python training_data_quality_check.py \
    --input_dir ./raw_dataset \
    --report_only
```

### 自定義閾值

```bash
python training_data_quality_check.py \
    --input_dir ./raw_dataset \
    --output_dir ./filtered_dataset \
    --max_vertical_disparity 0.5 \
    --max_background_diff 0.03 \
    --min_polarization_contrast 0.03 \
    --max_alignment_error 0.5 \
    --min_depth_validity 0.95
```

### 參數說明

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--max_vertical_disparity` | 1.0 | 最大允許垂直視差 (pixels) |
| `--max_background_diff` | 0.05 | 背景最大亮度差異 (normalized) |
| `--min_polarization_contrast` | 0.02 | 最小偏振對比度 |
| `--max_alignment_error` | 1.0 | 最大對齊誤差 (pixels) |
| `--min_depth_validity` | 0.90 | 最小深度有效率 (90%) |

---

## 輸出報告

腳本會生成 JSON 格式的質量報告：

```json
{
  "config": {
    "max_vertical_disparity": 1.0,
    "max_background_diff": 0.05,
    ...
  },
  "timestamp": "2025-01-01T12:00:00",
  "total_scenes": 100,
  "passed_scenes": 85,
  "failed_scenes": 15,
  "statistics": {
    "geometric_failures": 3,
    "photometric_failures": 5,
    "polarization_failures": 4,
    "alignment_failures": 2,
    "depth_failures": 6
  },
  "reports": [
    {
      "scene_name": "scene_0001",
      "passed": true,
      "geometric_consistency": true,
      "geometric_vertical_disparity": 0.32,
      ...
    },
    ...
  ]
}
```

---

## 檢測流程圖

```
                    ┌─────────────────┐
                    │   載入場景檔案   │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ I_parallel│  │ I_cross  │  │  depth   │
        └────┬─────┘  └────┬─────┘  └────┬─────┘
             │              │              │
             └──────┬───────┘              │
                    │                      │
    ┌───────────────┼───────────────┐      │
    │               │               │      │
    ▼               ▼               ▼      ▼
┌────────┐    ┌──────────┐    ┌────────┐ ┌────────┐
│Geometric│    │Photometric│    │Polariz.│ │ Depth  │
│ Check  │    │  Check   │    │ Check  │ │ Check  │
└────┬───┘    └────┬─────┘    └───┬────┘ └───┬────┘
     │              │              │          │
     └──────────────┴──────────────┴──────────┘
                         │
                         ▼
                 ┌───────────────┐
                 │ Alignment Check│
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │  ALL PASSED?  │
                 └───────┬───────┘
                    YES  │  NO
              ┌──────────┴──────────┐
              ▼                     ▼
       ┌────────────┐        ┌────────────┐
       │ 複製到輸出  │        │ 記錄失敗原因│
       └────────────┘        └────────────┘
```

---

## 常見問題

### Q: 為什麼大量樣本因幾何一致性失敗？
**A**: 可能原因：
1. 相機未正確校正 → 重新進行立體校正
2. 特徵點不足 → 增加場景紋理
3. 閾值過嚴 → 調整 `--max_vertical_disparity`

### Q: 偏振對比度不足怎麼辦？
**A**: 可能原因：
1. 入射角不在 Brewster angle 附近 → 調整光源位置
2. 偏振片角度錯誤 → 檢查偏振片對齊
3. 物體材質反射率低 → 更換物體或增強光源

### Q: 深度有效率太低？
**A**: 可能原因：
1. 深度傳感器對透明物體失效 → 使用 proxy 方法
2. 遮罩不準確 → 重新生成遮罩
3. 閾值過嚴 → 調整 `--min_depth_validity`

### Q: 合成數據該如何調整閾值？
**A**: 合成數據與真實數據有所不同：

```bash
# 合成數據推薦閾值
python training_data_quality_check.py \
    --input_dir ./synthetic_dataset \
    --max_background_diff 0.1 \
    --min_polarization_contrast 0.01 \
    --min_depth_validity 0.8 \
    --report_only
```

主要差異：
- 合成數據特徵點較少 → 降低 `min_feature_matches`
- 合成數據遮罩生成方式不同 → 降低 `min_depth_validity`
- 合成數據背景完全相同 → `max_background_diff` 可以較嚴格

---

## 依賴套件

```bash
pip install opencv-python numpy
```

---

## 版本歷史

| 版本 | 日期 | 說明 |
|------|------|------|
| 1.0 | 2025-01 | 初始版本 |
