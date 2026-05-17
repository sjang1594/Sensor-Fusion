# Sensor Fusion Portfolio

**Seungho Jang** — GPU/rendering engineer with 4 years of industry experience.
This portfolio covers neural sensor simulation, BEV perception, and classical sensor fusion,
from raw radar waveform physics to deep multi-sensor architectures.

---

## Research

> *For PhD admissions and research roles: novel methods, contributions, and research arc.*

### NeuralSensorSim → BEVFormerRadar

A two-stage research pipeline: generate physics-accurate synthetic radar data with 3D Gaussian Splatting, then use it to improve BEV perception training.

**The core question:** *Can synthetic radar data generated from neural scene representations substitute for scarce real radar data?*

| Stage | What it does | Key result |
|---|---|---|
| [NeuralSensorSim](NeuralSensorSim/) | 3DGS scene fitting → novel view synthesis → physics-based radar simulation + dynamic object paste | PSNR **30.94 dB**, 7.17M Gaussians; 120 synthetic samples/scene |
| [BEVFormerRadar](BEVFormerRadar/) | BEVFormer (ResNet50 + SCA) trained on real + synthetic mix | Real+Synth 1:1 → IDSW **3** (real-only: 13, −77%); MOTA **-0.054** at thr=0.1 |

**Research arc:**
```
nuScenes LiDAR/Camera/Radar
         │
         ▼
NeuralSensorSim
  Phase 1: 3DGS static scene fitting     → PSNR 30.94 dB, 7.17M Gaussians
  Phase 2: Novel view synthesis          → 120 samples × 12 pose perturbations
  Phase 3: Physics-based radar sim       → Radial Doppler noise, dynamic copy-paste
         │
         ▼
BEVFormerRadar  (score_threshold=0.1, nuScenes v1.0-mini val)
  Exp A: real-only baseline              → MOTA -0.048, FP/f 2.8, IDSW 13
  Exp B: real + synth 1:1               → MOTA -0.054, FP/f 2.2, IDSW  3  (IDSW −77%)
  PMBM upper bound (radar GT)           → MOTA -0.177, FP/f 0.8, IDSW  3
```

**Key finding:** Doppler-corrected synthetic radar augmentation reduces IDSW by 77% (13→3),
matching PMBM-level tracking consistency. MOTA and FP/frame also marginally improve.

Docs: [SUMMARY.md](Docs/SUMMARY.md) — architecture, implementation steps, experiment analysis, [research roadmap](Docs/neural-sensor-sim/roadmap.md)

---

## Engineering

> *For AV/robotics industry roles: working systems, classical algorithms, benchmarks.*

### FMCW Radar + EO/IR Isaac Sim Extension [`fmcw-radar-sensor/`](fmcw-radar-sensor/)

GPU-accelerated radar and dual-band EO/IR sensor extensions for NVIDIA Isaac Sim,
alongside a 29-phase multi-target tracker progression evaluated on nuScenes v1.0-mini.

**Tracker benchmark (nuScenes scene-0061, 5 radars):**

| Phase | Algorithm | MOTA | FP | IDSW | Contribution |
|---|---|---|---|---|---|
| 18 | CV-EKF | -0.183 | 161 | — | Baseline |
| 19 | IMM | -0.170 | ~140 | — | Multi-model blending |
| 22 | GM-PHD | -0.363 | 792 | 22 | RFS, no explicit association |
| 23 | LMB | -0.033 | 122 | 5 | Persistent labels |
| 24 | LMB+Camera | -0.023 | 104 | 5 | 360° radar-camera gate |
| 27 | ETT (Random Matrix) | -0.865 | 1636 | 10 | Extent estimation, −52% FP |
| 29 | **PMBM** | **-0.177** | **383** | **3** | Theoretically complete RFS |

- PMBM: 85% FP reduction vs LMB (2569 → 383) via data-driven Bayesian birth
- ETT: 80% IDSW reduction (49 → 10) by absorbing multi-reflection clusters
- GPU LWIR + VIS dual-band Isaac Sim extension; EO/IR fusion: 97% RMSE reduction

→ Full algorithm writeup: [`fmcw-radar-sensor/PORTFOLIO.md`](fmcw-radar-sensor/PORTFOLIO.md)

---

### PointPainting — Camera-LiDAR Deep Fusion [`PointPainting/`](PointPainting/)

Reimplementation of *PointPainting* (Vora et al., CVPR 2020).
LiDAR points are painted with YOLOv8-seg class scores before 3D detection.

```
Camera → YOLOv8-seg (H×W×6)
                      ↓ project LiDAR → camera, sample score
LiDAR → (N,9) painted cloud → BEV voxelizer → DBSCAN → 3D boxes
```

→ [`PointPainting/README.md`](PointPainting/README.md)

---

### Radar-Camera EKF Tracking [`RadarCameraFusion/`](RadarCameraFusion/)

4-state EKF with sequential sensor updates: radar (sparse depth + Doppler) and camera (YOLOv8 detections).

```
Radar ──→ coordinate projection ──┐
                                   ├─→ EKF [x, y, vx, vy] → tracks
Camera ──→ YOLOv8 2D detection ───┘
```

→ [`RadarCameraFusion/README.md`](RadarCameraFusion/README.md)

---

### Camera-LiDAR Extrinsic Calibration [`CameraLidarCalib/`](CameraLidarCalib/)

Target-based 6-DoF rigid transform estimation: `findChessboardCorners` → RANSAC plane fit → `solvePnP` → Nelder-Mead refinement.

| Metric | Result |
|---|---|
| Intrinsic reprojection error | **0.035 px** |
| Extrinsic reprojection RMSE | **0.707 px** |
| Rotation error vs GT | **0.15°** |
| Translation error vs GT | **0.64 cm** |

→ [`CameraLidarCalib/README.md`](CameraLidarCalib/README.md)

---

## Theory Foundations [`tutorials/`](tutorials/)

7 notebooks covering the math behind the projects above (LiDAR, Camera, Radar basics; Calibration; Kalman/EKF/UKF; Sensor Fusion; Advanced MOT).
Based on Udacity ND313 — study references, not original research.

---

## Skills Map

| Area | Projects |
|---|---|
| Neural rendering (3DGS, gsplat 1.x) | NeuralSensorSim |
| Synthetic data generation (radar simulation) | NeuralSensorSim |
| BEV perception (BEVFormer, SCA, heatmap head) | BEVFormerRadar |
| GPU acceleration (NVIDIA Warp, Isaac Sim) | fmcw-radar-sensor |
| Radar signal processing (FMCW, CFAR, AoA) | fmcw-radar-sensor |
| EO/IR sensor physics (Planck, NETD, attenuation) | fmcw-radar-sensor |
| RFS tracking (GM-PHD, LMB, PMBM) | fmcw-radar-sensor |
| Sensor calibration (PnP, RANSAC, Nelder-Mead) | CameraLidarCalib |
| Multi-sensor EKF (radar + camera, Doppler Jacobian) | RadarCameraFusion |
| Camera-LiDAR deep fusion (PointPainting) | PointPainting |

---

## Setup

```bash
# Core
pip install numpy scipy matplotlib opencv-python open3d nuscenes-devkit ultralytics

# NeuralSensorSim / BEVFormerRadar
pip install "gsplat>=1.0.0" torch torchvision

# GPU phases (fmcw-radar-sensor)
pip install warp-lang
# Isaac Sim phases: NVIDIA Isaac Sim 2023.1+
```

Windows CUDA build notes: [Docs/reference/windows-build.md](Docs/reference/windows-build.md)
