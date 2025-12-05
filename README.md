# PIDS-ACP
Physics-Informed Deep Stereo: Active Cross-Polarization  for Data-Efficient Transparent Obstacle Avoidance
# Physics-Informed Deep Stereo: Active Cross-Polarization for Data-Efficient Transparent Obstacle Avoidance

This repository contains the official implementation, datasets, and supplementary materials for our paper:

**Physics-Informed Deep Stereo: Active Cross-Polarization for Data-Efficient Transparent Obstacle Avoidance**  
*Your Name et al., 2025*  
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
✔ Cross-polarization stereo capture pipeline  
✔ Training code  
✔ Evaluation scripts  
✔ Reproducible experiments

---

## 🚀 Features

- **Active cross-polarization stereo capture**
- **Minimal data learning** (tens of samples instead of tens of thousands)
- **Plug-and-play input representation** compatible with all stereo networks
- **High-performance detection under complex lighting**
- **Fully open-source reproduction**

---

## 🗂 Repository Structure

