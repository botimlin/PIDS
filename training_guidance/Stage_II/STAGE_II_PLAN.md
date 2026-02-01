# Stage II: Real-World Fine-tuning 規劃

## 1. 硬體配置

### 1.1 計算架構
```
[Raspberry Pi 5] ──以太網──> [Jetson Nano] ──(或)──> [雲端]
   (相機控制)                  (本地推理)            (遠端推理)
```

- **Raspberry Pi 5**: 控制雙相機、數據採集
- **Jetson Nano**: 本地推理（RAFT-Stereo 在 Pi 上跑不動）
- **雲端**: 備選方案

### 1.2 相機系統
| 項目 | 規格 |
|------|------|
| 型號 | IMX296 (Raspberry Pi Global Shutter Camera) |
| 感測器 | RGB Bayer (RGGB) |
| 原生解析度 | 1456 x 1088 |
| 輸出格式 | YUV420 (取 Y 通道作為灰階) |
| 焦距 | 6mm (與模擬一致) |
| FOV | 45.4° |
| Baseline | 65mm |

### 1.3 深度感測器 (GT 採集用)
| 項目 | 規格 |
|------|------|
| 深度解析度 | 320 x 240 @ 30fps |
| 深度 FOV | 55°(H) x 72°(V) ✓ 覆蓋相機 FOV |
| 檢測距離 | 0.2 ~ 2m |
| 介面 | USB2.0 (Type-C) |

---

## 2. 灰階策略

### 2.1 問題
- Stage 1 訓練使用灰階合成數據
- IMX296 是 RGB Bayer 感測器
- 需要對齊兩者

### 2.2 解決方案
**使用 YUV420 的 Y 通道作為灰階**

```python
# Picamera2 配置
config = cam.create_video_configuration(
    main={"size": (640, 480), "format": "YUV420"}
)

# 提取 Y 通道
y_channel = frame[:480, :640, 0]  # Y plane
```

### 2.3 理由
- Y = 0.299R + 0.587G + 0.114B (標準灰階轉換)
- 偏振強度差異 (I∥ vs I⊥) 在 Y 通道完全保留
- 簡單高效，ISP 已處理 demosaic

---

## 3. Ground Truth 來源

### 3.1 挑戰
- 結構光/LiDAR 無法測量透明物體
- 需要 disparity GT，不是 depth

### 3.2 Proxy GT 方法
```
步驟 1: 放置透明玻璃 → 拍攝偏振立體對 (I∥, I⊥)
步驟 2: 替換成 3D 列印 1:1 不透明模型 → 深度感測器取 GT
步驟 3: depth → disparity 轉換
步驟 4: 對齊標註
```

### 3.3 Disparity 計算
```python
# 從 depth 轉換為 disparity
focal_px = (width / 2) / np.tan(np.radians(FOV) / 2)
disparity = (baseline_mm * focal_px) / depth_mm
```

### 3.4 注意事項
- 使用定位機制確保替換位置一致
- 替換過程中場景其他物體不能移動
- 3D 列印精度需足夠

---

## 4. 數據量規劃

### 4.1 目標
- **總樣本數**: 200 - 500
- **訓練集**: 80% (160 - 400)
- **驗證集**: 20% (40 - 100)

### 4.2 多樣性要求
| 維度 | 建議數量 |
|------|----------|
| 透明物體種類 | 3-5 種 (玻璃、壓克力、不同厚度) |
| 背景場景 | 5-10 種 |
| 擺放角度 | 多角度 |
| 距離範圍 | 0.3m - 1.5m |

### 4.3 採集效率
每個場景設置後可產出多個樣本：
1. 微調透明物體位置
2. 稍微移動相機角度
3. 一個場景 → 5-10 個樣本

---

## 5. 數據格式

### 5.1 檔案命名
```
{scene_id}/
├── left_parallel.png      # I∥ (左相機, 0° 偏振)
├── right_cross.png        # I⊥ (右相機, 90° 偏振)
├── disparity_gt.npy       # Ground truth disparity
├── glass_mask.png         # 透明物體遮罩
└── metadata.json          # 場景參數
```

### 5.2 metadata.json
```json
{
  "scene_id": "scene_001",
  "glass_type": "acrylic_5mm",
  "distance_m": 0.8,
  "capture_time": "2026-01-19T10:30:00",
  "exposure_us": 10000,
  "notes": ""
}
```

---

## 6. Stage 1 → Stage 2 對齊

| 項目 | Stage 1 (合成) | Stage 2 (真實) |
|------|----------------|----------------|
| 圖像類型 | 灰階 | YUV420 Y 通道 |
| 解析度 | 640 x 480 | 640 x 480 |
| 焦距 | 6mm | 6mm |
| FOV | 45.4° | 45.4° |
| Baseline | 65mm | 65mm |
| 偏振配置 | I∥ / I⊥ | I∥ / I⊥ |

---

## 7. 待完成項目

- [ ] 雙相機同步設置
- [ ] Fine-tuning 訓練腳本
- [ ] 數據採集 SOP
- [ ] 校準流程 (相機內外參、深度感測器對齊)
- [ ] 數據品質檢查工具

---

## 8. 時間軸

```
Phase 1: 硬體設置
  - 雙相機安裝與同步
  - 深度感測器校準
  - 定位夾具製作

Phase 2: 採集系統
  - 採集腳本開發
  - 品質檢查工具
  - 試拍驗證

Phase 3: 數據採集
  - 場景設置
  - 採集 200-500 樣本
  - 標註與檢查

Phase 4: Fine-tuning
  - 訓練腳本調整
  - 超參數調優
  - 評估與迭代
```
