# PIDS: Physics-Informed Deep Stereo
### Active Asymmetric Polarization for Data-Efficient Transparent Obstacle Avoidance

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](LINK_TO_YOUR_NOTEBOOK_URL)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7660046.svg)](https://doi.org/10.5281/zenodo.7660046)
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](LINK_TO_YOUR_ARXIV_URL)

**Po-Ting Lin**, et al., 2025  
📄 **arXiv preprint:** https://arxiv.org/abs/XXXX.XXXXX  
📌 *Submitted to IEEE Robotics and Automation Letters (RA-L) / ICRA*

---

## 📖 Overview

Transparent obstacles—such as glass doors, acrylic panels, and plastic shields—pose a fundamental challenge for autonomous navigation. Traditional RGB-D, LiDAR, and depth-completion approaches often fail to reconstruct reliable geometry due to light transmission and specular reflection.

We propose **PIDS**, a **physics-informed deep stereo framework** that leverages **active asymmetric polarization**. By capturing cross-polarized stereo pairs ($I_{\parallel}$ and $I_{\perp}$), we convert the transparency detection problem into a robust stereo matching task.

### Key Features
* **Physics-Informed:** Utilizes polarization difference ($I_{\parallel} \gg I_{\perp}$) to detect "invisible" obstacles.
* **Data-Efficient:** Achieves high accuracy with minimal training samples (tens instead of thousands).
* **Plug-and-Play:** Compatible with standard deep stereo backbones (e.g., RAFT-Stereo).
* **Dual-Mode Deployment:** Supports **Cloud (Colab)** for easy access and **LAN (Local Server)** for real-time performance.

---

## 🛠️ Hardware Setup (The Physical Rig)

> **⚠️ Critical:** This algorithm relies on a specific physical optical configuration. Standard stereo cameras will NOT work without modification.

### 1. Bill of Materials (BOM)

| Component | Specification / Model | Qty | Notes |
| :--- | :--- | :--- | :--- |
| **Compute (Robot)** | Raspberry Pi 5 (8GB) | 1 | Pi 4B compatible; Pi 5 recommended for high FPS. |
| **Camera** | **Raspberry Pi Global Shutter Camera** | 2 | **Sony IMX296**. Essential for hardware sync & avoiding motion artifacts. |
| **Lens** | 6mm or 16mm CS-Mount Lens | 2 | Must be identical focal lengths. |
| **Polarizer** | Linear Polarization Film | 3 | High extinction ratio recommended. |
| **Light Source** | High-intensity LED / Flashlight | 1 | Must be mounted near the cameras. |
| **Server** | NVIDIA GPU (Tesla P4 / RTX 3060+) | 1 | For running RAFT-Stereo inference. |

### 2. Optical Configuration (The "Secret Sauce")
The system uses an **Active** setup where the light source and cameras have specific polarization angles relative to each other:

1.  **Light Source ($0^{\circ}$)**: Place a linear polarizer in front of the light. This defines the **Reference Angle**.
2.  **Left Camera ($0^{\circ}$ - Parallel View)**: Rotate the lens polarizer to **maximize** brightness when viewing the light's reflection on a mirror.
    * *Physics:* Captures strong specular highlights on glass.
3.  **Right Camera ($90^{\circ}$ - Cross View)**: Rotate the lens polarizer to **minimize** brightness (darkest) when viewing the reflection.
    * *Physics:* Suppresses highlights, seeing "through" the reflection.

### 3. Hardware Synchronization
* Connect the **XVS** pins of both Global Shutter cameras.
* Configure one camera as `Master` and the other as `Slave` to ensure microsecond-level synchronization.

---

## 💻 Installation

```bash
# Clone the repository
git clone [https://github.com/your-username/PIDS.git](https://github.com/your-username/PIDS.git)
cd PIDS

# Install dependencies
pip install -r requirements.txt
```
---

## 🚀 Usage: Two Modes
We provide two modes to ensure reproducibility for everyone, from students with just a laptop to researchers with lab equipment.

☁️ Mode 1: Cloud Inference (Google Colab)
Best for quick demonstration without local GPU hardware.

Click the Open in Colab badge at the top of this README.

Run the notebook cells to start the ngrok tunnel.

Copy the generated public URL (e.g., https://xxxx.ngrok-free.app).

On your Raspberry Pi:

```bash

python robot_client.py --mode cloud --url "YOUR_NGROK_URL"
```
⚡ Mode 2: Local LAN (Real-time)
Best for physical experiments and low latency (<100ms).
1. Server Side (PC/Workstation with GPU):
```bash
# Download pretrained RAFT models
bash scripts/download_models.sh

# Start the inference server
python server_local.py --port 8000
```
2. Client Side (Raspberry Pi): Ensure the Pi and Server are on the same Wi-Fi network.
```bash
python robot_client.py --mode lan --ip "SERVER_IP" --port 8000
```
---

## 📊 Evaluation
To reproduce the quantitative results (RMSE / AbsRel) reported in Table 1 of our paper:
```bash
python evaluate.py --checkpoint checkpoints/pids_best.pth --data_path ./dataset/test_set
```
Note: The full dataset including polarization-difference maps will be released upon paper acceptance.
---

## 🔬 Citation
If you use this hardware configuration, code, or method in your research, please cite:
```bash
@article{lin2025pids,
  title={Physics-Informed Deep Stereo: Active Asymmetric Polarization for Data-Efficient Transparent Obstacle Avoidance},
  author={Lin, Po-Ting and [Add Co-authors Here]},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```
---

## 📜 License & Attribution
This project is released under the MIT License.

Attribution Requirement: If you use this code, hardware design, or derivative works in academic publications or commercial products, you must cite the paper above and attribute the original authorship.
---
## ⭐ Acknowledgments
This work is dedicated to the open-source robotics community. We aim to provide a robust perception layer for future End-to-End Pure Vision Navigation systems.

Contributions: Pull requests are welcome! Please open an issue to discuss proposed changes.
