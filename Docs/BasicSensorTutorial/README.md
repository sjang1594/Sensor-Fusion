# Sensor Fusion — Theory Notes

Study notebooks covering the mathematical foundations of the sensor fusion projects in this repository.
These are learning references derived from the Udacity Sensor Fusion Nanodegree (ND313) curriculum,
not original implementations. Each notebook links to the project where the theory is applied.

---

## Notebooks

| # | Notebook | Core Topics | Applied In |
|---|----------|------------|------------|
| 01 | [LiDAR Basics](01_lidar_basics.ipynb) | Point cloud I/O, Voxel Grid downsampling, RANSAC ground removal, Euclidean clustering, AABB | `CameraLidarCalib` step 2 |
| 02 | [Camera Basics](02_camera_basics.ipynb) | Pinhole model, Harris / FAST / SIFT / ORB, ratio test, Lucas-Kanade optical flow, TTC | `RadarCameraFusion` camera leg |
| 03 | [Radar Signal Processing](03_radar_basics.ipynb) | FMCW chirp design, Range-Doppler 2D FFT, CA-CFAR, MUSIC AoA | `fmcw-radar-sensor` Phase 1–3 |
| 04 | [Sensor Calibration](04_calibration.ipynb) | Intrinsic calibration, LiDAR plane fitting, solvePnP, Bundle Adjustment, reprojection error | `CameraLidarCalib` full pipeline |
| 05 | [Kalman Filter](05_kalman_filter.ipynb) | Bayes filter, linear KF, EKF Jacobian, UKF sigma points, CTRV model, NIS consistency | `fmcw-radar-sensor` Phase 9, 25 |
| 06 | [Multi-Sensor Fusion](06_sensor_fusion.ipynb) | LiDAR-camera ROI association, TTC (camera + LiDAR), Hungarian algorithm, track lifecycle | `fmcw-radar-sensor` Phase 10, 18 |
| 07 | [Advanced MOT](07_advanced.ipynb) | IMM, JPDA, GM-PHD, PMBM, PointPainting pipeline | `fmcw-radar-sensor` Phase 19–29 |

---

## Relationship to the Projects

```
tutorials/                      →  production implementations
──────────────────────────────────────────────────────────────────
01 RANSAC plane fitting         →  CameraLidarCalib/src/02_lidar_plane_fitting.py
02 Pinhole projection           →  RadarCameraFusion/src/01_radar_projection.py
03 FMCW / CA-CFAR               →  fmcw-radar-sensor/standalone/cfar_roc.py
04 solvePnP + Nelder-Mead       →  CameraLidarCalib/src/03_extrinsic_calibration.py
05 EKF Jacobian                 →  fmcw-radar-sensor/standalone/fusion_ekf.py
05 UKF / CTRV                   →  fmcw-radar-sensor/standalone/nuscenes_ctrv_ukf.py
06 Hungarian association        →  fmcw-radar-sensor/standalone/nuscenes_ekf_fusion.py
07 GM-PHD predict/update/prune  →  fmcw-radar-sensor/standalone/nuscenes_gm_phd.py
07 PMBM PPP + Bernoulli         →  fmcw-radar-sensor/standalone/nuscenes_pmbm_tracker.py
07 PointPainting paint()        →  PointPainting/src/
```

---

## Setup

```bash
pip install numpy scipy matplotlib opencv-python open3d nuscenes-devkit ultralytics
```

KITTI `.bin` files are required for notebooks 01–02. nuScenes mini is required for notebooks 03, 06–07.
All Python code blocks in each notebook can be run independently if the relevant dataset is available.