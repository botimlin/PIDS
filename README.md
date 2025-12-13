# Physics-Informed Deep Stereo: Active Asymmetric Polarization for Data-Efficient Transparent Obstacle Avoidance

This repository contains the official implementation, datasets, and supplementary materials for our paper:

**Physics-Informed Deep Stereo: Active Asymmetric Polarization for Data-Efficient Transparent Obstacle Avoidance**  
*Po-Ting.Lin et al., 2025*  
📄 **arXiv preprint:** https://arxiv.org/abs/XXXX.XXXXX  
📌 RA-L submission

---

## 🔧 Overview

Transparent obstacles—such as glass doors, acrylic panels, and plastic shields—pose a unique challenge for robotic vision systems due to light transmission, refraction distortion, and specular reflection. Traditional RGB-D, LiDAR, and depth-completion approaches often fail to reconstruct reliable geometry in these scenarios.

We propose a **physics-informed deep stereo framework** that captures **cross-polarized stereo pairs (0° and 90°)**. Transparent objects generate **strong polarized reflections**, enabling:

- High-contrast feature cues  
- Texture-independent detection  
- Extremely low training data requirement  
- Compatibility with any existing deep stereo network (e.g., PSMNet, GwcNet, RAFT-Stereo)

This repository includes:

✔ Dataset  
✔ Asymmetric Polarization stereo capture pipeline  
✔ Training code  
✔ Evaluation scripts  
✔ Reproducible experiments

---

## 🚀 Features

- **Active Asymmetric Polarization stereo capture**
- **Minimal data learning** (tens of samples instead of tens of thousands)
- **Plug-and-play input representation** compatible with all stereo networks
- **High-performance detection under complex lighting**
- **Fully open-source reproduction**

---

## 🗂 Repository Structure
- 📁 dataset/
- 📁 calibration/
- 📁 scripts/
- 📁 models/
- 📁 results/
- 📁 paper/
- 📄 train.py
- 📄 evaluate.py
- 📄 requirements.txt
- 📄 LICENSE
- 📄 README.md


---
## 🛠️ Reproduction Guide

This repository provides two modes for reproducing our PIDS framework:
1. **☁️ Cloud Mode (Google Colab):** For quick demonstration without a local GPU.
2. **⚡ LAN Mode (Local Server):** High-performance setup for real-time avoidance (as described in the paper).

### 🔧 Hardware Setup
The PIDS framework relies on a specific Active Asymmetric Polarization setup. Unlike standard stereo vision, we require a polarized light source and polarization filters on the cameras.1. Bill of Materials (BOM)
Here is the exact hardware configuration used in our experiments:
ComponentSpecification / ModelQtyNotesCompute (Robot)Raspberry Pi 5 (8GB)1Pi 4B is compatible but lower FPS.CameraRaspberry Pi Global Shutter Camera2Based on Sony IMX296 sensor. Essential for moving platforms.Lens6mm or 16mm CS-Mount Lens2Ensure both lenses are identical.PolarizerLinear Polarization Film/Filter3High extinction ratio recommended.Light SourceHigh-intensity LED / Flashlight1Must be mounted near the cameras.Compute (Server)NVIDIA GPU (Tesla P4 / RTX 3060+)1For running RAFT-Stereo (min 4GB VRAM).

---

### 🚀 Mode 1: Quick Start (Google Colab)
The easiest way to test the inference pipeline using free cloud GPUs (Tesla T4).

1.  **Open the Notebook:**
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](LINK_TO_YOUR_NOTEBOOK)
2.  **Start the Server:** Run the cells to install dependencies and start the `ngrok` tunnel. Copy the generated public URL (e.g., `https://xxxx.ngrok-free.app`).
3.  **Run the Client (Pi 5):**
    ```bash
    # On your Raspberry Pi
    python robot_client.py --mode cloud --url "YOUR_NGROK_URL"
    ```

---

### ⚡ Mode 2: Local LAN Setup (Low Latency)
Recommended for real-time experiments (<100ms latency).

#### 1. Server Side (PC/Workstation)
```bash
# Clone the repository
git clone [https://github.com/your-username/PIDS.git](https://github.com/your-username/PIDS.git)
cd PIDS

# Create environment
conda create -n pids python=3.8
conda activate pids
pip install -r requirements.txt

# Download Pre-trained Models
bash scripts/download_models.sh

# Start the Inference Server
python server_local.py --port 8000

## 📦 Installation

```bash
git clone https://github.com/yourname/polar-stereo.git
cd polar-stereo
pip install -r requirements.txt
```
---

## 📘 Usage
Training
python train.py --config configs/polar.yaml

Evaluation
python evaluate.py --checkpoint checkpoints/best.pth

Data preparation

Instructions for collecting cross-polarized stereo pairs are provided in
docs/data_collection.md.

---

## 📊 Dataset

Our dataset includes:

Stereo image pairs (0° / 90° polarization)

Ground-truth depth

Polarization-difference maps

Transparent object annotations

Dataset download link (Google Drive / HuggingFace):
👉 Coming soon after RA-L review.

---

## 🔬 Citation

If you use this repository, please cite the paper:

BibTeX
@article{your_arxiv_2025,
  title={Physics-Informed Deep Stereo: Active Asymmetric Polarization for Data-Efficient Transparent Obstacle Avoidance},
  author={Your Name and Others},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}


(Replace with official RA-L citation once accepted.)

---

## 📜 License

This project is released under the MIT License.

Attribution Requirement

If you use this code or any derivative works in academic publications,
products, or research, you must cite the paper above and attribute the
original authorship.

---

## ⭐ Acknowledgments

This work was supported by:

Your institution

Robotics community open-source contributors

---

## 🤝 Contributions

Pull requests and issues are welcome.
If you want to extend this project (e.g., add more networks or datasets),
please open an issue first to discuss the plan.
