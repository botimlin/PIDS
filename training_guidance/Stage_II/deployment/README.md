# PIDS Deployment (Vast.ai + Docker)

## 架構

```
[Raspberry Pi]              [Vast.ai GPU Server]
  雙相機採集                    Docker Container
      ↓                            ↓
  pids_client.py ──ZeroMQ──→ realtime_server.py
      ↓                            ↓
  接收 disparity ←─────────── PIDS 推理
```

## 快速開始

### 1. Vast.ai 設置

1. 註冊 https://vast.ai/
2. 租用 GPU (建議 RTX 3060 以上)
3. 選擇 Docker Image: `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime`
4. 開放端口: `5555`

### 2. 伺服器端 (Vast.ai)

```bash
# SSH 進入 Vast.ai 實例
ssh -p <PORT> root@<IP>

# 安裝依賴
pip install pyzmq opencv-python-headless

# 上傳程式碼 (用 scp 或 git clone)
scp -P <PORT> -r deployment/ root@<IP>:/app/

# 啟動伺服器
cd /app
python realtime_server.py --port 5555 --iters 12 --fp16
```

### 3. 客戶端 (Raspberry Pi)

```bash
# 安裝依賴
pip install pyzmq opencv-python numpy

# 執行客戶端
python pids_client.py --server <VAST_AI_IP> --port 5555
```

## Docker 部署 (可選)

如果想用 Docker 一鍵部署：

```bash
# 建置
docker build -t pids-server .

# 執行
docker run --gpus all -p 5555:5555 pids-server
```

## 檔案說明

| 檔案 | 說明 |
|------|------|
| `Dockerfile` | Docker 容器定義 |
| `realtime_server.py` | GPU 推理伺服器 (ZeroMQ) |
| `pids_client.py` | Pi 客戶端 |
| `pi_capture_client.py` | Pi 相機 + 客戶端整合 |

## 性能預期

| 配置 | 延遲 | FPS |
|------|------|-----|
| RTX 3060, 12 iters, FP16 | ~50ms 推理 | 15-20 |
| + 網路延遲 (良好) | ~150ms 總計 | 5-7 |
| + 網路延遲 (一般) | ~250ms 總計 | 3-4 |

## 故障排除

### 連接超時
- 確認 Vast.ai 端口已開放
- 確認防火牆設置
- 測試: `nc -zv <IP> 5555`

### GPU 記憶體不足
- 減少 batch size
- 使用 FP16: `--fp16`
- 減少迭代: `--iters 8`

### 延遲過高
- 檢查網路帶寬
- 考慮降低圖像品質 (PNG compression)
- 考慮降低解析度
