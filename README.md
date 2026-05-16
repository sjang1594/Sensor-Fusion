# Sensor Fusion Portfolio

A portfolio of autonomous driving sensor fusion projects, organized as a progression from
physics-level signal simulation through classical filtering to deep learning and neural scene
reconstruction. All projects share the nuScenes v1.0-mini dataset and a common sensor
coordinate convention.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     THEORY FOUNDATION                               │
│  Docs/BasicSensorTutorial — notebooks mapping math to each project  │
└────────────────────────────┬────────────────────────────────────────┘
                             │ informs
┌────────────────────────────▼────────────────────────────────────────┐
│                     SIGNAL PHYSICS                                  │
│                                                                     │
│  fmcw-radar-sensor ── Physics-level 77 GHz FMCW radar simulator    │
│      │                NVIDIA Isaac Sim + Warp GPU ray casting       │
│      │                ADC → Range-Doppler → CA-CFAR detections      │
│      │                                                              │
│      └── data/v1.0-mini/ ◄─── nuScenes mini (shared by all)        │
└──────────────┬──────────────────────────────────────────────────────┘
               │ nuScenes data + sensor transform chain
┌──────────────▼──────────────────────────────────────────────────────┐
│                     CALIBRATION                                     │
│                                                                     │
│  CameraLiDARLib ── 6-DoF Camera↔LiDAR extrinsic calibration        │
│      └── solvePnP + Nelder-Mead, RANSAC plane fitting, noise study  │
└──────────────┬──────────────────────────────────────────────────────┘
               │ calibrated sensor frames
┌──────────────▼──────────────────────────────────────────────────────┐
│                     CLASSICAL FUSION                                │
│                                                                     │
│  RadarCameraFusion ── Radar + Camera → EKF tracking                │
│      └── YOLOv8m detections + nuScenes RADAR_FRONT + EKF [x,y,vx,vy]│
└──────────────┬──────────────────────────────────────────────────────┘
               │ fused object tracks
┌──────────────▼──────────────────────────────────────────────────────┐
│                     DEEP FUSION                                     │
│                                                                     │
│  PointPainting ── LiDAR + Camera → semantic painting → 3D detect   │
│      └── YOLOv8-seg scores painted onto LiDAR → BEV + DBSCAN       │
│                                                                     │
│  BevFormerRadar ── Multi-camera + Radar → BEV transformer          │
│      └── ResNet50-FPN + Spatial Cross-Attention + CenterHead        │
└──────────────┬──────────────────────────────────────────────────────┘
               │ scene understanding
┌──────────────▼──────────────────────────────────────────────────────┐
│                     NEURAL SCENE                                    │
│                                                                     │
│  NeuralRadarSim ── 3D Gaussian Splatting → novel-view synthesis     │
│      └── LiDAR-initialized Gaussians + dynamic masking + radar sim  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Sub-Projects

| Project | Sensors | Technique | Key Output |
|---------|---------|-----------|------------|
| [fmcw-radar-sensor](https://github.com/sjang1594/fmcw-radar-sensor) | Radar (simulated) | FMCW physics, Warp BVH, GPU FFT, CA-CFAR, MUSIC DOA, Kalman tracking | Range-Doppler map, CFAR detections, ROS2 PointCloud2 |
| [CameraLiDARLib](https://github.com/sjang1594/CameraLiDARLib) | Camera + LiDAR | OpenCV intrinsic, RANSAC plane fit, solvePnP + Nelder-Mead | 6-DoF extrinsic T (0.15° / 0.64 cm error) |
| RadarCameraFusion | Radar + Camera | nuScenes devkit, YOLOv8m, EKF [x,y,vx,vy] | Per-object depth + Doppler-fused tracks |
| [PointPainting](https://github.com/sjang1594/PointPainting) | LiDAR + Camera | YOLOv8-seg, BEV voxelization, DBSCAN | Semantically-painted point clouds, 3D boxes |
| BevFormerRadar | Camera × 6 + Radar × 5 | ResNet50-FPN, Spatial Cross-Attn, CenterHead | BEV heatmap, 3D bounding boxes (vehicle/pedestrian/cyclist) |
| [NeuralRadarSim](https://github.com/sjang1594/NeuralRadarSim) | Camera × 6 + LiDAR | 3D Gaussian Splatting, dynamic masking, pose perturbation | Novel-view images + synthetic radar detections |
| Docs/BasicSensorTutorial | — | Theory notebooks (Udacity ND313) | Math foundations mapped to each project above |

---

## How the Projects Relate

### Shared Data

All projects consume **nuScenes v1.0-mini**, stored once inside `fmcw-radar-sensor/data/v1.0-mini/`
and referenced by other projects via relative path:

```
fmcw-radar-sensor/data/v1.0-mini/   ← canonical location
BevFormerRadar/config.yaml          dataroot: "../fmcw-radar-sensor/data/v1.0-mini"
NeuralRadarSim/config.yaml          dataroot: "../fmcw-radar-sensor/data/v1.0-mini"
PointPainting/configs/config.yaml   nuscenes_root: "../RadarCameraFusion/data/nuscenes"
```

> nuScenes v1.0-mini is not committed to git. Download separately from
> https://www.nuscenes.org/nuscenes#download and extract into
> `fmcw-radar-sensor/data/`.

### Sensor Transform Chain

All projects use the same nuScenes sensor-frame convention:

```
Radar sensor frame  → (R_radar, t_radar) →  Ego vehicle frame
                                                    ↓ (R_cam, t_cam)
                                             Camera sensor frame
                                                    ↓ K (intrinsic)
                                             Image pixel (u, v)
```

`CameraLiDARLib` calibrates this transform from scratch using synthetic checkerboard data.
`RadarCameraFusion`, `PointPainting`, and `BevFormerRadar` load it directly from
nuScenes `calibrated_sensor.json`.

### Learning Progression

The projects are sequenced from signal fundamentals to end-to-end deep learning:

1. **fmcw-radar-sensor** — understand radar physics at ADC level (beat frequency, Doppler,
   CFAR, DOA). Phases 1–12 build the full signal chain incrementally.
2. **CameraLiDARLib** — understand how sensors are spatially aligned. Foundation for all
   projection code in the other projects.
3. **RadarCameraFusion** — classical multi-sensor fusion: explicit association + Kalman filter.
4. **PointPainting** — sequential deep fusion: camera semantics enriching 3D geometry.
5. **BevFormerRadar** — attention-based fusion: transformer queries over BEV space.
6. **NeuralRadarSim** — neural scene representations: synthesize novel sensor views for
   data augmentation.
7. **Docs/BasicSensorTutorial** — theory companion to all of the above.

### Cross-Project Dependencies

```
fmcw-radar-sensor
  ├── Phase 8  calibrate_cam_radar.py   ← same algorithm as CameraLiDARLib/03
  ├── Phase 9  nuscenes_radar_projection ← same transform chain as RadarCameraFusion/01
  ├── Phase 10 fusion_ekf.py            ← same EKF design as RadarCameraFusion/04
  └── Phase 11 lidar_camera_projection  ← same RANSAC + projection as PointPainting/02

BevFormerRadar
  └── config.yaml bev grid              ← same 200×200 0.5m/cell as PointPainting/03

NeuralRadarSim
  └── use_lidar_init: true              ← reuses LIDAR_TOP from fmcw-radar-sensor/data
```

---

## Setup

### 1. Clone with submodules

```bash
git clone --recurse-submodules https://github.com/sjang1594/Sensor-Fusion.git
cd Sensor-Fusion
```

Or, if already cloned:

```bash
git submodule update --init --recursive
```

### 2. Download nuScenes mini

Register at https://www.nuscenes.org/sign-up, download `v1.0-mini`, and extract to:

```
fmcw-radar-sensor/data/
└── v1.0-mini/
    ├── maps/
    ├── samples/
    ├── sweeps/
    └── v1.0-mini/
        ├── scene.json
        └── ...
```

### 3. Per-project setup

Each sub-project has its own `requirements.txt` and `README.md`:

```bash
# Example: RadarCameraFusion
cd RadarCameraFusion
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python src/05_run_pipeline.py
```

`fmcw-radar-sensor` requires **NVIDIA Isaac Sim 5.1.0** (see its README for setup).
All other projects run on standard Python 3.9+.

---

## Repository Structure

```
Sensor-Fusion/
├── fmcw-radar-sensor/        [submodule] Physics FMCW radar simulator
├── CameraLiDARLib/           [submodule] Camera-LiDAR extrinsic calibration
├── PointPainting/            [submodule] LiDAR-camera sequential deep fusion
├── NeuralRadarSim/           [submodule] 3D Gaussian Splatting neural scene
├── RadarCameraFusion/        Classical radar-camera EKF fusion pipeline
├── BevFormerRadar/           BEV transformer with radar fusion
├── Docs/
│   └── BasicSensorTutorial/  Theory notebooks (LiDAR/Camera/Radar/KF/Fusion)
└── README.md                 This file
```

---

## Hardware Used

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 4090 (24 GB, sm_89) |
| CUDA | 12.8 |
| OS | Windows 11 |
| Isaac Sim | 5.1.0 (for fmcw-radar-sensor) |
