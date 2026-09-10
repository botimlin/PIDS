### Active Asymmetric Polarization for Data-Efficient Transparent Obstacle Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

**Po-Ting Lin**

> Copyright (c) 2025-2026 Po-Ting Lin
> Released under the MIT License (see [LICENSE](LICENSE) file).

---

## 📖 Abstract

Transparent obstacles—such as glass doors, acrylic panels, and plastic shields—pose a fundamental challenge for autonomous navigation. Traditional RGB-D, LiDAR, and depth-completion approaches often fail to reconstruct reliable geometry due to light transmission and specular reflection.

We propose **PIDS**, a **physics-informed deep stereo framework** that leverages **active asymmetric polarization**. By capturing cross-polarized stereo pairs (I∥ and I⊥), we convert the transparency detection problem into a robust photometric discrepancy learning task.


---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔬 **Physics-Informed** | Exploits polarization difference (I∥ ≫ I⊥) to detect "invisible" transparent obstacles |
| 📊 **Data-Efficient** | Achieves high accuracy with minimal training samples (tens instead of thousands) |
| 🔌 **Plug-and-Play** | Compatible with standard deep stereo backbones (e.g., RAFT-Stereo) without architectural changes |
| ⚡ **Real-Time Capable** | Supports both cloud inference and local LAN deployment for <100ms latency |

---

## 🔧 System Architecture

### Why It Works

| Region | Parallel View (I∥) | Cross View (I⊥) | Result |
|--------|-------------------|-----------------|--------|
| **Diffuse Background** | α·(Id + Ib) | α·(Id + Ib) | I∥ ≈ I⊥ → Standard stereo matching works |
| **Transparent Surface** | α·(Id + Ib) + Is | α·(Id + Ib) | I∥ ≫ I⊥ → Strong discriminative cue |

---

## 🛠️ Hardware Requirements

### Bill of Materials (BOM)

| Component | Specification | Qty | Notes |
|-----------|--------------|-----|-------|
| **Compute Unit** | Raspberry Pi 5 (8GB) | 1 | Pi 4B compatible; Pi 5 recommended |
| **Camera** | Raspberry Pi Global Shutter Camera | 2 | Sony IMX296 sensor, essential for sync |
| **Lens** | 6mm CS-Mount Lens | 2 | Must be identical focal lengths |
| **Polarizer** | Linear Polarization Film | 3 | High extinction ratio recommended |
| **Light Source** | High-intensity LED Panel | 1 | Mount near cameras |
| **Inference Server** | NVIDIA GPU (RTX 3060+ / Tesla P4) | 1 | For running RAFT-Stereo |
| **Mounting** | 3D Printed Bracket | 1 | See `appendix/Duo_Cam_Bracket.step` |

### Optical Setup Procedure

1. **Light Source (0°)**: Attach a linear polarizer to the LED panel. This defines the **reference angle**.

2. **Left Camera (0° - Parallel View)**: Rotate the lens polarizer to **maximize** brightness when viewing the light's reflection on a mirror.
   - *Physics*: Captures strong specular highlights on glass

3. **Right Camera (90° - Cross View)**: Rotate the lens polarizer to **minimize** brightness when viewing the reflection.
   - *Physics*: Suppresses highlights, seeing "through" the reflection

4. **Hardware Sync**: Connect the **XVS** pins of both Global Shutter cameras. Configure one as `Master` and the other as `Slave`.

---

## 💻 Installation

### Prerequisites

- Python 3.8+
- CUDA 11.7+ (for GPU inference)
- Docker & Docker Compose (for Raspberry Pi deployment)

### Clone Repository

```bash
git clone https://github.com/your-username/PIDS.git
cd PIDS
```

### Server Setup (GPU Workstation)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download pretrained RAFT-Stereo weights
bash scripts/download_models.sh
```

### Client Setup (Raspberry Pi)

```bash
cd docker_setup

# Build and start container
docker-compose up -d

# Enter container
docker exec -it pids_container bash
```

---

## 🚀 Usage

### Step 1: Radiometric Calibration

Before first use, perform sensor calibration to correct for dark current and vignetting:

```bash
# Inside Docker container on Raspberry Pi
cd /app/codes
python complex_calibration_tool.py
```

Follow the on-screen instructions:
1. **Dark Field**: Cover lens caps → Capture 30 frames
2. **Flat Field**: Point at uniform white surface → Capture 30 frames

Output files will be saved to `./output/calibration_data/`:
- `master_dark_L.npy`, `master_dark_R.npy`
- `gain_map_L.npy`, `gain_map_R.npy`

### Step 2: Run Inference

#### Option A: Local LAN Mode (Real-time, <100ms latency)

**Server Side (GPU Workstation):**
```bash
python server_local.py --port 8000
```

**Client Side (Raspberry Pi):**
```bash
python robot_client.py --mode lan --ip "SERVER_IP" --port 8000
```

#### Option B: Cloud Mode (Demo/Testing)

```bash
# Start ngrok tunnel on server
ngrok http 8000

# On Raspberry Pi
python robot_client.py --mode cloud --url "YOUR_NGROK_URL"
```

### Step 3: Visualization

Access the live stereo stream via web browser:
```
http://<raspberry_pi_ip>:5000
```

---

## 📂 Project Structure

```
PIDS/
├── appendix/
│   ├── Duo_Cam_Bracket.step      # 3D printable camera mount
│   └── Schematic_Optical_Path.drawio
├── codes/
│   └── calibration_tools/
│       ├── calibration_capture_zh-TW.py
│       ├── complex_calibration_tool.py
│       └── exposure_calibrate_tool_zh-TW.py
├── docker_setup/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── codes/
│       ├── complex_calibration_tool.py
│       └── web_stream.py
├── models/                        # Pretrained weights (download separately)
├── scripts/
│   └── download_models.sh
├── assets/                        # Images for documentation
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 🧪 Training

### Two-Stage Fine-Tuning Strategy


#### Stage I: Synthetic Pre-training
```bash
python train.py \
    --stage 1 \
    --data_path ./data/synthetic \
    --batch_size 8 \
    --num_steps 100000 \
    --lr 0.0001
```

#### Stage II: Real-World Fine-tuning
```bash
python train.py \
    --stage 2 \
    --data_path ./data/real_world \
    --checkpoint checkpoints/stage1_best.pth \
    --batch_size 4 \
    --num_steps 5000 \
    --lr 0.00001
```

### Data Quality Criteria

Training data must pass 5 quality filters:
1. ✅ Geometric consistency (vertical disparity < 1px)
2. ✅ Background photometric consistency (|I∥ - I⊥| ≈ 0)
3. ✅ Polarization signal validity (I∥ > I⊥ on transparent surfaces)
4. ✅ Ground truth alignment (< 1px displacement)
5. ✅ Depth validity rate (> 90% valid pixels in ROI)

---

## 📊 Evaluation

```bash
python evaluate.py \
    --checkpoint checkpoints/pids_best.pth \
    --data_path ./data/test_set \
    --metrics EPE RMSE
```

### Metrics

| Metric | Description |
|--------|-------------|
| **EPE** | End-Point Error - Average absolute disparity difference (pixels) |
| **RMSE** | Root Mean Squared Error - Sensitive to large deviations |

---




## 📜 License

This project is released under the [MIT License](LICENSE).

**Attribution Requirement**: If you use this code, hardware design, or derivative works in academic publications or commercial products, you must cite the paper above and attribute the original authorship.

---

## 🙏 Acknowledgments

- [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) - Base stereo matching architecture
- [Picamera2](https://github.com/raspberrypi/picamera2) - Raspberry Pi camera interface
- The open-source robotics community

---

## 📮 Contact

For questions or collaboration inquiries:

- **Author**: Po-Ting Lin
- **Email**: [botimlinlin@gmail.com]
- **WebSite**: potinglin.org
- **Issues**: Please use [GitHub Issues](https://github.com/your-username/PIDS/issues)

---

## 🗺️ Roadmap

- [ ] Release full dataset upon paper acceptance
- [ ] Add ROS2 integration
- [ ] Support for additional stereo backbones (PSMNet, AANet)
- [ ] Web-based calibration GUI
- [ ] Multi-camera array support

---

<p align="center">
  <b>⭐ If you find this project useful, please consider giving it a star! ⭐</b>
</p>
