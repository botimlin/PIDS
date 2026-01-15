# PIDS Multi-GPU Renderer

> Copyright (c) 2025-2026 Po-Ting Lin
> Released under the MIT License (see LICENSE file).

多 GPU 並行渲染 PIDS 訓練數據。

## 文件說明

| 文件 | 說明 |
|------|------|
| `multi_gpu_launcher.py` | 多 GPU 調度器，分配場景到各 GPU |
| `pids_renderer_random.py` | 渲染器主程式 (v3.5.2)，支援隨機化數據增強 |

## 快速開始

### 1. 上傳到雲端機器

```bash
scp -P <PORT> multi_gpu_launcher.py pids_renderer_random.py root@<IP>:~/rendering/
```

### 2. 啟動多 GPU 渲染

```bash
# 4 GPU 並行
nohup python multi_gpu_launcher.py \
    --input_dir ../scenes_output \
    --output ./output \
    --num_gpus 4 \
    --no_preview \
    > render.log 2>&1 &
```

### 3. 監控進度

```bash
# 查看主日誌
tail -f render.log

# 查看各 GPU 日誌
tail -f gpu0.log gpu1.log gpu2.log gpu3.log

# 查看 GPU 使用情況
watch -n 1 nvidia-smi

# 統計已完成場景數
ls output/*_left_parallel.exr | wc -l
```

## 參數說明

### multi_gpu_launcher.py

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--input_dir` | OBJ 場景目錄 | (必填) |
| `--output` | 輸出目錄 | (必填) |
| `--num_gpus` | GPU 數量 | 4 |
| `--spp` | 每像素樣本數 | 16384 |
| `--max_scenes` | 最大場景數 | 全部 |
| `--skip` | 跳過前 N 個場景 | 0 |
| `--no_preview` | 不保存預覽 PNG | False |

### pids_renderer_random.py (單 GPU)

```bash
# 單 GPU 渲染
CUDA_VISIBLE_DEVICES=0 python pids_renderer_random.py \
    --input_dir ../scenes_output \
    --output ./output \
    --spp 16384
```

## 輸出文件

每個場景會產生以下文件：

| 文件 | 說明 |
|------|------|
| `*_left_parallel.exr` | 左相機 I∥ (0° 偏振) |
| `*_right_parallel.exr` | 右相機 I∥ (用於 vertical disparity 驗證) |
| `*_right_cross.exr` | 右相機 I⊥ (90° 偏振) |
| `*_depth.exr` | 深度圖 |
| `*_disparity.exr` | 視差圖 |
| `*_glass_mask.exr` | 玻璃區域 mask |
| `*_params.json` | 渲染參數 |
| `*_report.json` | 品質報告 |

## 隨機化數據增強

v3.5.0+ 新增隨機化功能，每個場景使用不同的：

| 參數 | 範圍 | 說明 |
|------|------|------|
| LED_INTENSITY | [1200, 3500] | 偏振光強度 |
| CEILING_EMITTER_INTENSITY | [50, 200] | 環境光強度 |
| CAMERA_X | [-120, 20] mm | 相機水平位置 |

固定不變：
- 基線: 65mm
- FOV: 45.4°
- 偏振片角度: 0°/90°

## 常見問題

### 進程管理

```bash
# 殺掉所有渲染進程
pkill -9 -f python

# 查看進程
ps aux | grep python
```

### 清理分離文件

渲染過程會產生臨時分離文件，可以清理：

```bash
rm ../scenes_output/*_glass.obj
rm ../scenes_output/*_ceiling.obj
rm ../scenes_output/*_other.obj
```

### SPP 建議

| SPP | 品質 | 速度 |
|-----|------|------|
| 16384 | 高品質，低噪點 | 慢 |
| 8192 | 夠用 | 中等 |
| 4096 | 有些噪點 | 快 |

## 版本歷史

- v3.5.2: Multi-GPU 支援，獨立 launcher
- v3.5.1: 修復 CUDA_VISIBLE_DEVICES 問題
- v3.5.0: 隨機化數據增強
- v3.4.2: 基礎偏振渲染
