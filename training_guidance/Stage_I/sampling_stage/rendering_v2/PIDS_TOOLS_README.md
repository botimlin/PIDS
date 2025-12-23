# PIDS Stage 1 工具集使用說明

> **Polarization-based Invisible Depth Sensing (PIDS)**  
> 偏振立體視覺透明物體深度估計系統

---

## 📁 工具總覽

| 工具 | 功能 | 使用時機 |
|------|------|---------|
| `renderer_stage1_exaggerated.py` | 偏振渲染器 | 渲染訓練數據 |
| `pids_summarize_reports.py` | 報告匯總 | 渲染完成後生成總報告 |
| `pids_data_checker.py` | 綜合檢測 | 訓練前檢查（獨立模式） |
| `pids_sanity_check.py` | 偏振信號檢測 | 快速驗證 |
| `pids_noise_check.py` | 噪點檢測 | 評估渲染品質 |
| `exr_to_png.py` | EXR 轉換 | 視覺化檢查 |
| `exr_tool.py` | EXR 工具箱 | 進階分析 |

---

## 🔄 工作流程（新架構）

```
┌─────────────────────────────────────────────────────────┐
│  渲染階段                                                │
│  ┌──────────────────────────────────────────────────┐  │
│  │  renderer_stage1_exaggerated.py                   │  │
│  │                                                   │  │
│  │  對每個場景:                                       │  │
│  │    1. 渲染 I∥, I⊥, depth, disparity              │  │
│  │    2. 分析偏振信號（分區域：玻璃 vs 背景）          │  │
│  │    3. 計算噪點 SNR                                │  │
│  │    4. 輸出 scene_xxxx_report.json ← 品質報告      │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  匯總階段                                                │
│  ┌──────────────────────────────────────────────────┐  │
│  │  pids_summarize_reports.py                        │  │
│  │                                                   │  │
│  │  讀取所有 *_report.json                           │  │
│  │  生成 SUMMARY.md / .html                          │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          ↓
                    檢查報告
                    通過? → 開始訓練
                    未通過 → 調整參數重渲染
```

---

## 1️⃣ 渲染器 (`renderer_stage1_exaggerated.py`)

### 基本用法

```bash
# 渲染單一場景（使用預設：薄玻璃 + 背景光）
python renderer_stage1_exaggerated.py \
    --scene_file /path/to/scene.obj \
    --output_dir ./output_stage1

# 批量渲染目錄
python renderer_stage1_exaggerated.py \
    --input_dir /path/to/scenes/ \
    --output_dir ./output_stage1

# 🔥 調整光源平衡（如果背景亮度差異大）
python renderer_stage1_exaggerated.py \
    --input_dir ./scenes \
    --output_dir ./output \
    --polarized_intensity 1500 \
    --fill_intensity 8000

# 使用標準玻璃材質（有厚度）
python renderer_stage1_exaggerated.py \
    --scene_file scene.obj \
    --output_dir ./output \
    --glass_type roughdielectric
```

### 重要參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--spp` | 16384 | 每像素樣本數，越高越清晰但越慢 |
| `--max_depth` | 16 | 光線反彈次數，多層玻璃需要較高值 |
| `--glass_type` | thindielectric | 玻璃材質：`thindielectric`(薄) / `roughdielectric`(標準) |
| `--glass_ior` | 1.5 | 玻璃折射率 |
| `--polarized_intensity` | 2000 | 偏振光強度 |
| `--fill_intensity` | 5000 | 非偏振背景光強度（🔥 重要！） |
| `--no_fill_light` | - | 關閉非偏振背景光 |
| `--no_preview` | - | 不保存預覽 PNG |

### 玻璃材質選擇

| 材質 | 說明 | 適用場景 |
|------|------|---------|
| `thindielectric` | 薄玻璃，假設厚度為0 | 薄壁玻璃容器、窗戶 |
| `roughdielectric` | 標準玻璃，有折射 | 實心玻璃、厚壁容器 |

### 🔥 光源強度調整

為了讓背景區域 I∥ ≈ I⊥，需要適當調整光源比例：

```bash
# 如果背景還是不平衡，增加背景光強度
python renderer_stage1_exaggerated.py \
    --scene_file scene.obj \
    --output_dir ./output \
    --polarized_intensity 1500 \
    --fill_intensity 10000
```

建議比例：`fill_intensity` ≈ 2-5x `polarized_intensity`

### 輸出文件（每個場景）

```
output_stage1/
├── scene_0001_left_parallel.exr    # 左視角 I∥ (訓練輸入)
├── scene_0001_right_cross.exr      # 右視角 I⊥ (訓練輸入)
├── scene_0001_depth.exr            # 深度圖
├── scene_0001_disparity.exr        # 視差圖 (訓練 GT)
├── scene_0001_report.json          # 🔥 品質報告（新增）
├── scene_0001_params.json          # 渲染參數
├── scene_0001_left_parallel.png    # 預覽圖
└── ...
```

### 品質報告內容 (`*_report.json`)

```json
{
  "scene_name": "scene_0001",
  "polarization": {
    "glass_region": {
      "pixel_ratio": 0.15,
      "dolp_mean": 0.65
    },
    "background_region": {
      "pixel_ratio": 0.85,
      "dolp_mean": 0.02
    },
    "intensity_ratio": {
      "mean": 42.5
    }
  },
  "noise": {
    "snr_polarization": 35.2
  },
  "quality": {
    "score": 85,
    "level": "excellent",
    "valid": true
  },
  "warnings": []
}
```

---

## 2️⃣ 報告匯總 (`pids_summarize_reports.py`)

### 基本用法

```bash
# 渲染完成後，匯總所有報告
python pids_summarize_reports.py --input_dir ./output_stage1

# 指定輸出路徑
python pids_summarize_reports.py -i ./output_stage1 -o ./summary.md

# 生成 HTML 報告
python pids_summarize_reports.py -i ./output_stage1 --format html

# 生成 JSON 報告
python pids_summarize_reports.py -i ./output_stage1 --format json
```

### 輸出報告內容

- **總覽**: 通過率、平均分數
- **品質分佈**: excellent/good/acceptable/poor 各多少
- **偏振統計**: 玻璃區域 DoLP、背景區域 DoLP、I∥/I⊥ 比值
- **噪點統計**: 平均 SNR、範圍
- **警告彙總**: 哪些警告最常出現
- **各場景詳情**: 每個場景的具體數值

---

## 3️⃣ 分區域 DoLP 分析（重要！）

### 為什麼要分區域？

```
整張圖的組成：
┌─────────────────────────────────┐
│  背景（牆壁、地板）              │  ← DoLP ≈ 0（漫反射）
│  佔比：~85%                     │
│                                 │
│      ┌─────────┐                │
│      │  玻璃   │ ← DoLP > 0.3   │
│      │ 佔比15% │  （Fresnel反射）│
│      └─────────┘                │
└─────────────────────────────────┘

如果只看全局平均：
  DoLP_mean = 0.85×0.02 + 0.15×0.65 = 0.11 ← 被稀釋了！

分區域看：
  玻璃區域 DoLP = 0.65 ← 這才是重點！
  背景區域 DoLP = 0.02 ← 這是正常的
```

### 判斷標準

| 區域 | 預期 DoLP | 說明 |
|------|-----------|------|
| 玻璃區域 | > 0.3 | ✓ 偏振效果正確 |
| 背景區域 | < 0.1 | ✓ 漫反射正確 |
| 兩者都高 | - | ⚠️ 檢查光源設置 |
| 兩者都低 | - | ⚠️ 偏振效果不足 |

---

## 4️⃣ 完整工作流程範例

```bash
# 步驟 1: 批量渲染
python renderer_stage1_exaggerated.py \
    --input_dir ./scenes \
    --output_dir ./output_stage1 \
    --spp 16384 \
    --max_depth 16

# 步驟 2: 匯總報告
python pids_summarize_reports.py \
    --input_dir ./output_stage1 \
    --format html

# 步驟 3: 查看報告
# 打開 ./output_stage1/SUMMARY.html

# 步驟 4: 如果通過率 > 80%，開始訓練
# 如果未通過，根據報告調整參數重新渲染
```

---

## 📊 品質評分標準

### 綜合評分 (0-100)

| 等級 | 分數 | 說明 |
|------|------|------|
| 🟢 excellent | ≥ 80 | 可直接訓練 |
| 🟢 good | ≥ 60 | 可直接訓練 |
| 🟡 acceptable | ≥ 40 | 可以訓練，但建議改進 |
| 🔴 poor | < 40 | 需要修復 |

### 評分組成

| 項目 | 權重 | 評分依據 |
|------|------|---------|
| 偏振分數 | 33% | 玻璃區域 DoLP |
| 噪點分數 | 33% | SNR |
| 對比分數 | 33% | 玻璃 DoLP / 背景 DoLP |

---

## 🔧 常見問題

### Q: 背景 DoLP 很高怎麼辦？

**A**: 這表示背景區域也有偏振反射，可能原因：
1. 光源設置導致整體都有偏振
2. 背景材質不是純漫反射

解決方案：
- 檢查環境光設置
- 確認背景材質為 diffuse

### Q: 玻璃區域 DoLP 很低怎麼辦？

**A**: 偏振效果不明顯，可能原因：
1. 入射角不在 Brewster 角附近
2. 玻璃材質設置問題
3. max_depth 不夠

解決方案：
- 調整光源入射角（~56° 為 Brewster 角）
- 確認材質為 roughdielectric
- 增加 max_depth

### Q: I∥/I⊥ 比值接近 1 怎麼辦？

**A**: 偏振片沒有產生差異，可能原因：
1. 偏振光源角度錯誤
2. 偏振片角度設置錯誤

### 🔥 Q: 背景區域 I∥/I⊥ 不平衡（不接近 1）怎麼辦？

**A**: 這是很重要的問題！如果背景區域 I∥ >> I⊥ 或 I∥ << I⊥：
- 左圖和右圖的背景亮度會差很多
- RAFT-Stereo 無法做雙目對齊
- 訓練會失敗

**原因**：只有偏振光源，沒有非偏振背景光

**解決方案**：

```python
# 在 CONFIG 中啟用非偏振背景光
'lighting': {
    'led': {
        'intensity': 3000.0,  # 偏振光（照射玻璃）
        ...
    },
    'fill_light': {
        'enabled': True,       # 🔥 啟用非偏振背景光
        'intensity': 2000.0,   # 背景光強度
        'polarized': False,    # 非偏振
        ...
    },
}
```

**預期效果**：
- 玻璃區域：I∥ >> I⊥（偏振效果）
- 背景區域：I∥ ≈ I⊥（正常漫反射）

---

## ⚖️ 強度平衡檢測（新功能）

渲染器現在會自動檢測背景區域的強度平衡：

```
報告輸出：
    I∥/I⊥ 比值 (全局): 42.0x      ← 玻璃區域偏振效果
    I∥/I⊥ 比值 (背景): 1.05x ✓ 平衡  ← 背景區域應該接近 1
```

### 判斷標準

| 背景 I∥/I⊥ | 狀態 | 說明 |
|------------|------|------|
| 0.5 ~ 2.0x | ✓ 平衡 | RAFT 可正常對齊 |
| 0.3 ~ 3.0x | ⚠️ 警告 | 可能有問題 |
| < 0.3 或 > 3.0 | ❌ 失敗 | 需要啟用 fill_light |

---

## 📚 參考資料

- [PIDS 論文](./PIDS__17_.pdf)
- [Mitsuba 3 文檔](https://mitsuba.readthedocs.io/)
- [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo)

---

## 📝 版本記錄

| 日期 | 版本 | 更新 |
|------|------|------|
| 2024-12-21 | 1.3 | 添加薄玻璃材質（thindielectric）、光源強度命令行參數 |
| 2024-12-21 | 1.2 | 新增非偏振背景光（fill_light）、強度平衡檢測 |
| 2024-12-21 | 1.1 | 新增渲染時品質報告、分區域 DoLP 分析 |
| 2024-12-21 | 1.0 | 初始版本 |

