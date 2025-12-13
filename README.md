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
