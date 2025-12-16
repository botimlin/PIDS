FROM dtcooper/raspberrypi-os:bookworm

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-picamera2 \
    python3-opencv \
    python3-numpy \
    python3-flask \
    libcamera-apps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

CMD ["python3", "calibration_capture.py"]