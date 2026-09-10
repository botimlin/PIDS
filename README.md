# PIDS — Physics-Informed Deep Stereo

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Concluded-lightgrey.svg)](#project-status)

**Po-Ting Lin**

> Copyright (c) 2025–2026 Po-Ting Lin  
> Released under the MIT License. See [LICENSE](LICENSE).

---

## Project Status

> **CONCLUDED — February 26, 2026**

PIDS was a research investigation into whether **active asymmetric polarization**
could make stereo vision capable of recovering the surface depth of transparent
obstacles such as glass doors and acrylic panels.

After approximately **five months**, **45+ training experiments**, and multiple
generations of model and rendering architectures, the project reached a negative
but important conclusion:

> **Polarized stereo vision does not provide a reliable path to measuring the
> surface depth of transparent objects through conventional stereo
> correspondence.**

The limiting factor is not primarily model capacity, optimization, dataset size,
or network architecture. It is the underlying optics.

This repository is therefore maintained as a **research archive**, containing
hardware designs, calibration tools, polarized rendering code, stereo-model
experiments, data-quality infrastructure, and engineering knowledge produced
during the investigation.

---

## Abstract

Transparent objects violate several assumptions on which conventional stereo
vision depends.

For opaque Lambertian surfaces, corresponding points observed by the left and
right cameras have approximately consistent appearance. Transparent surfaces
behave differently: reflection, refraction, internal reflection, and
view-dependent specular components cause the two cameras to observe different
photometric patterns.

PIDS investigated whether **active asymmetric polarization** could introduce a
useful physical cue:

- the illumination is linearly polarized;
- the left camera observes a parallel-polarized view (`I∥`);
- the right camera observes a cross-polarized view (`I⊥`);
- transparent surfaces can exhibit a measurable difference between the two
  polarization states.

The original hypothesis was that this difference could help a deep stereo
network recover transparent-surface disparity.

Experiments showed, however, that polarization changes the **relative intensity
of reflected components** without creating stable, geometrically corresponding
features between the stereo views.

As a result, the polarization signal did not form a sharp matchable peak in
correlation space, while conventional stereo tended to recover the
**background depth behind the transparent surface rather than the depth of the
surface itself**.

The project was therefore concluded as a negative-result study.

---

## Research Question

The original goal was:

> Can a synchronized stereo camera pair using active asymmetric polarization
> recover the depth of transparent obstacles?

The optical configuration was:

| Component | Polarization |
|---|---:|
| Active illumination | 0° |
| Left camera | 0° / Parallel (`I∥`) |
| Right camera | 90° / Cross (`I⊥`) |

The intended cue was:

| Region | Parallel View (`I∥`) | Cross View (`I⊥`) |
|---|---|---|
| Diffuse / background region | Similar appearance | Similar appearance |
| Polarization-sensitive reflection | Stronger reflected component | Suppressed reflected component |

This difference is physically observable.

The critical question was whether it was also **stereo-matchable**.

Experiments showed that these are not the same thing.

---

## Why the Approach Fails

### 1. Transparent surfaces violate stereo photometric consistency

Conventional stereo matching assumes that the same physical point produces
sufficiently similar observable features in the left and right images.

For transparent surfaces, this assumption breaks down.

The observed image may contain combinations of:

- specular reflection;
- transmitted background light;
- refraction;
- internal reflection;
- view-dependent illumination effects.

These components vary strongly with viewing direction.

Therefore:

> The left and right cameras may observe different optical paths rather than
> two projections of the same surface feature.

On glass, stereo matching consequently tends to converge to the **background
disparity**, not the disparity of the glass surface.

---

### 2. Polarization does not create correspondence

Polarization can alter the ratio of reflected and transmitted light.

It can suppress or enhance particular reflection components.

However, it does **not** create a geometrically corresponding feature at the
same glass-surface point in both cameras.

The central lesson from PIDS is:

> **A discriminative signal is not necessarily a matchable signal.**

A polarization difference may indicate that transparent material is present,
while still providing insufficient information to determine stereo disparity.

---

## Correlation-Space Evidence

Representative measurements from the experiments showed:

| Metric | Observed Value | Interpretation |
|---|---:|---|
| Peak-to-Mean Ratio | **0.963** | Ground-truth location was not a dominant correlation peak |
| Peak Sharpness | **Negative** | Response was effectively inverted |
| `I∥ / I⊥` Ratio | **1.168** | Polarization difference existed, but was too weak / non-localized for correspondence |

The important result was not that polarization produced *no signal*.

It did.

The problem was that the signal produced **no sharp, spatially matchable
correlation peak** corresponding to the transparent surface.

---

## Why Larger Models Do Not Solve the Problem

The failure was reproduced across several generations of architecture.

The project explored a progression including:

```text
Baseline RAFT-Stereo
    ↓
Dual-Stream architectures
    ↓
Polarization Volume V1 / V2-E
    ↓
PIDS 2.0 — Two-Pass RGB
    ↓
PIDS V3 — Dual Volume + FiLM
    ↓
V4 — True Dual-Stream
    ↓
V5 — Cost Concatenation
    ↓
V6 — Glass-Aware
    ↓
PIDS 3.0 / S2M2
    — Transformer
    — 6-channel input
    — ComplexCNNEncoder
```

Increasing model capacity or changing the fusion strategy did not remove the
underlying left-right inconsistency.

This led to the conclusion that the primary limitation was **optical rather
than architectural**.

---

## Surface Reconstruction Is Not a Complete Escape Route

Once stereo correspondence becomes unreliable, the problem effectively shifts
from:

> finding corresponding points between two images

to:

> inferring the geometry of the transparent surface from indirect optical cues.

That is a **surface reconstruction** problem.

PIDS identified two major limitations in that direction.

### A. Incomplete reflective-surface coverage

Polarization-based active probing depends strongly on specular reflection.

For a planar glass surface, strong reflection only occurs over particular
incidence-angle regions.

Therefore the system obtains scattered informative regions rather than complete
surface coverage.

This prevents direct depth measurement over the entire transparent obstacle.

### B. Surface-normal inference remains underconstrained

Surface reconstruction would typically require reliable surface-normal
estimation.

A neural network can be trained to infer these normals, but doing so does not
remove the underlying ambiguity; it transfers the unsolved problem into learned
shape inference.

This carries a theoretical risk similar to other transparent-object
reconstruction approaches.

---

## Closure Judgment

After **45+ experiments** spanning three major generations:

1. **RAFT-Stereo based PIDS 1.x**
2. **Two-Pass RGB / PIDS 2.0**
3. **S2M2 / PIDS 3.0**

the project concluded that:

> **“polarization + stereo matching” is the wrong formulation for direct
> transparent-surface depth recovery.**

The failure is rooted primarily in optical principles rather than an isolated
implementation defect.

Closing the project was therefore considered a valid research outcome rather
than an unfinished engineering milestone.

---

## What PIDS Did Establish

Although the original depth-recovery objective was unsuccessful, the project
produced several useful findings.

### Transparent-object stereo

Stereo matching on transparent surfaces frequently estimates the geometry of
the visible background rather than the transparent interface itself.

### Polarization

Polarization is useful for modifying and analyzing reflection components, but
does not by itself solve stereo correspondence.

### Deep learning

A sufficiently large neural network cannot reliably recover information that is
systematically absent or inconsistent in the measurement process.

### Experimental methodology

Negative results are valuable when repeated architecture changes converge on
the same physical limitation.

---

## Legacy Assets

The repository preserves the engineering and research infrastructure developed
during PIDS.

### Hardware

- Dual Raspberry Pi Global Shutter Camera system
- Sony IMX296 sensors
- Linear polarization system:
  - illumination: 0°
  - left camera: 0°
  - right camera: 90°
- Camera calibration fixtures
- Mechanical mounting hardware
- XVS Master/Slave synchronization circuitry

### Software

- Dark-field radiometric calibration
- Flat-field calibration
- Exposure calibration tools
- Polarization image-processing utilities
- Stereo capture tools
- Dataset validation utilities
- Experimental stereo architectures
- Training and evaluation scripts

### Synthetic Data Infrastructure

The project also produced multiple generations of polarized synthetic-data
rendering tools, including:

- Mitsuba 3 polarized rendering scripts;
- physical-polarizer geometry;
- parallel optical-axis configurations;
- textured RGB Stokes rendering;
- Blender scene randomization;
- sensor-realism augmentation.

### Data Quality Infrastructure

Datasets were evaluated using a five-point quality process covering:

1. geometric consistency;
2. background photometric consistency;
3. polarization-signal validity;
4. ground-truth alignment;
5. valid-depth coverage.

---

## Architecture & Training Knowledge

The project produced extensive development documentation covering:

- renderer architecture evolution;
- depth-network architecture evolution;
- polarization injection strategies;
- curriculum sampling;
- staged freezing and unfreezing;
- Directional Impulse Descent;
- sensor-realism augmentation;
- normalization effects on physical signals;
- cross-GPU numerical consistency.

One particularly important engineering finding was that normalization methods
such as:

- BatchNorm;
- LayerNorm;
- L2 normalization;
- per-image percentile normalization

can unintentionally destroy or distort physically meaningful absolute
radiometric signals.

---

## Hardware Reference

The following hardware was used during the investigation.

| Component | Specification | Qty |
|---|---|---:|
| Compute Unit | Raspberry Pi 5, 8 GB | 1 |
| Camera | Raspberry Pi Global Shutter Camera / Sony IMX296 | 2 |
| Lens | Identical 6 mm CS-Mount lenses | 2 |
| Camera Polarizer | Linear polarizing film | 2 |
| Illumination Polarizer | Linear polarizing film | 1 |
| Active Light | High-intensity LED panel | 1 |
| GPU Workstation | NVIDIA CUDA-capable GPU | 1 |
| Camera Mount | Custom / 3D-printed stereo bracket | 1 |

These specifications are retained for **experimental reproduction and archival
purposes**. They should not be interpreted as a recommended production system
for transparent-obstacle depth measurement.

---

## Optical Setup

### Illumination — 0°

Attach a linear polarizer to the active light source.

This defines the reference polarization direction.

### Left Camera — Parallel

Set the left-camera polarizer approximately parallel to the illumination
polarization.

The reflected component should be relatively strong.

### Right Camera — Cross

Rotate the right-camera polarizer approximately 90° relative to the
illumination polarization.

The polarized reflected component should be suppressed.

### Synchronization

The two global-shutter cameras should be hardware synchronized.

The experimental system used the IMX296 synchronization interface with a
Master/Slave XVS configuration.

---

## Radiometric Calibration

Calibration tools are retained because polarization experiments are highly
sensitive to sensor bias, exposure differences, vignetting, and nonlinear image
processing.

Example:

```bash
cd /app/codes
python complex_calibration_tool.py
```

The calibration procedure includes:

### Dark Field

Cover the lenses and capture multiple frames to estimate sensor offset and dark
current.

### Flat Field

Observe a uniform field to estimate spatial gain and vignetting.

Example outputs:

```text
master_dark_L.npy
master_dark_R.npy
gain_map_L.npy
gain_map_R.npy
```

---

## Repository Structure

```text
PIDS/
├── appendix/
│   ├── Duo_Cam_Bracket.step
│   └── Schematic_Optical_Path.drawio
│
├── codes/
│   └── calibration_tools/
│       ├── calibration_capture_zh-TW.py
│       ├── complex_calibration_tool.py
│       └── exposure_calibrate_tool_zh-TW.py
│
├── docker_setup/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── codes/
│       ├── complex_calibration_tool.py
│       └── web_stream.py
│
├── models/
├── scripts/
├── assets/
├── LICENSE
├── README.md
└── requirements.txt
```

> The exact contents of the repository may evolve as legacy material is
> organized and released.

---

## Reproduction / Historical Experiments

Some training scripts and checkpoints may be retained to reproduce historical
PIDS experiments.

They are provided for **research and archival purposes**, not as a validated
transparent-depth solution.

Example historical workflow:

```bash
python train.py \
    --data_path ./data/synthetic \
    --batch_size 8 \
    --num_steps 100000 \
    --lr 0.0001
```

Evaluation:

```bash
python evaluate.py \
    --checkpoint checkpoints/pids_best.pth \
    --data_path ./data/test_set \
    --metrics EPE RMSE
```

The meaning of these metrics must be interpreted carefully: good stereo
performance on opaque/background regions does not imply correct recovery of the
transparent surface itself.

---

## Research Lessons

PIDS produced several broader lessons that may be useful for future
physics-informed vision research.

### 1. Verify observability before scaling the network

Before increasing model complexity, determine whether the desired physical
quantity is actually represented in the sensor measurements.

### 2. Detection and depth recovery are different problems

A cue may reveal that glass exists without specifying where the glass surface
lies in 3D.

### 3. Photometric asymmetry can hurt correspondence

A cue intentionally introduced to make an object more visible may
simultaneously violate the assumptions required by stereo matching.

### 4. Physical signals should be preserved

Image normalization and learned feature normalization can destroy meaningful
radiometric relationships.

### 5. Negative results should be documented

Demonstrating that a plausible approach fails for fundamental reasons prevents
future work from repeatedly paying the same experimental cost.

---

## Future Directions

PIDS itself is concluded.

Possible future research directions should reformulate the measurement problem
rather than simply enlarge the stereo model.

Potential directions include:

- transparent-object **detection** rather than direct stereo depth;
- active structured optical measurements;
- multi-view observations;
- controlled illumination over multiple angles;
- temporal measurements;
- polarization as an auxiliary material cue;
- sensor fusion with non-RGB modalities;
- reconstruction methods based on explicit optical models.

These directions are **not claims of solved approaches**; they are possible
research questions motivated by the limitations observed in PIDS.

---

## Project Timeline

```text
Late 2025
│
├── Initial polarized stereo hypothesis
├── RAFT-Stereo / PIDS 1.x
├── Polarization-volume experiments
├── Renderer and calibration development
│
├── PIDS 2.0
│   └── Two-Pass RGB experiments
│
├── Advanced dual-stream / fusion architectures
│
├── PIDS 3.0 / S2M2
│   ├── Transformer experiments
│   ├── 6-channel representations
│   └── ComplexCNNEncoder
│
└── 2026-02-26
    Research project concluded
```

---

## License

This repository is released under the [MIT License](LICENSE).

If you use or discuss this project in academic work, please cite or otherwise
reference the project and its author where appropriate.

---

## Acknowledgments

This work builds on and benefited from open-source research and engineering
projects including:

- [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo)
- [Picamera2](https://github.com/raspberrypi/picamera2)
- [Mitsuba 3](https://mitsuba-renderer.org/)
- the open-source robotics and computer-vision communities

---

## Contact

**Po-Ting Lin**

- Website: [potinglin.org](https://potinglin.org)
- Email: botimlinlin@gmail.com
- GitHub Issues: use the issue tracker associated with this repository

---

## Final Note

PIDS did not produce the transparent-surface depth sensor originally envisioned.

It produced something different:

**experimental evidence identifying why a plausible physics-informed stereo
approach does not work.**

The central conclusion is:

> Transparent-object depth cannot be recovered reliably by assuming that
> polarization will restore conventional left-right stereo correspondence.
> Transparent surfaces remain strongly view-dependent, while asymmetric
> polarization introduces additional photometric inconsistency.

Knowing where a path ends is also a research result.
