# PIDS Real-time Inference Server
# Build: docker build -t pids-server .
# Run: docker run --gpus all -p 5555:5555 pids-server

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 安裝 Python 依賴
RUN pip install --no-cache-dir \
    pyzmq==25.1.0 \
    opencv-python-headless==4.8.0.74 \
    numpy==1.24.3 \
    scipy==1.11.1 \
    timm==0.9.2

# 複製程式碼和模型
COPY core/ ./core/
COPY pids_model.py .
COPY realtime_server.py .
COPY checkpoints/ ./checkpoints/

EXPOSE 5555

CMD ["python", "realtime_server.py", "--port", "5555", "--iters", "12", "--fp16"]
