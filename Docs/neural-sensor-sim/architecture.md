# 시스템 아키텍처

## 전체 파이프라인

```
nuScenes v1.0-mini
(LiDAR, Camera, Radar, GT)
         │
         ▼
┌─────────────────────────────────────┐
│         NeuralSensorSim             │
│                                     │
│  Phase 1: 3DGS Fitting             │
│    LiDAR 초기화 → gsplat 1.x 학습  │
│    PSNR 30.94dB / 7.17M Gaussians  │
│                 │                   │
│  Phase 2: Novel View Synthesis     │
│    PosePerturber → 12가지 섭동      │
│    10 frames × 12 = 120 samples    │
│                 │                   │
│  Phase 3: Radar Simulation         │
│    ┌─ RadarSynthesizer ────────┐   │
│    │  FOV/Range 필터링         │   │
│    │  Opacity 기반 샘플링      │   │
│    │  Radial Doppler noise     │   │
│    └───────────────────────────┘   │
│    ┌─ DynamicCompositor ───────┐   │
│    │  GT bbox → 실제 레이더 pts│   │
│    │  orig_ego → world → new   │   │
│    │  GT velocity compensated  │   │
│    └───────────────────────────┘   │
└─────────────────────────────────────┘
         │
         │  outputs/synthetic/scene_00/
         │  120 × .npz (imgs, radar_pts, gt_boxes)
         ▼
┌─────────────────────────────────────┐
│         BEVFormerRadar              │
│                                     │
│  MixedDataset                       │
│    Real (242) + Synth (120~360)    │
│                 │                   │
│  BEVFormer                          │
│    ResNet50 Backbone (frozen)       │
│    FPN (C3/C4/C5, 256ch)           │
│    BEV Encoder (2L, 8-head SCA)    │
│    Radar Fusion Branch              │
│                 │                   │
│  Detection Head → Heatmap          │
│                 │                   │
│  Evaluation: MOTA / MOTP / FP / IDSW│
└─────────────────────────────────────┘
```

---

## 데이터 흐름

### 합성 데이터 포맷 (.npz)

```python
{
    "imgs":            (6, 3, H, W)  uint8    # 6-camera images
    "cam_intrinsics":  (6, 3, 3)     float32  # Camera K matrices
    "cam_extrinsics":  (6, 4, 4)     float32  # Camera extrinsics
    "ego_to_world":    (4, 4)        float32  # Ego pose
    "gt_boxes":        (M, 7)        float32  # [x,y,z,w,l,h,yaw]
    "gt_labels":       (M,)          int64    # 0=vehicle, 1=ped, 2=cyclist
    "radar_pts":       (N, 5)        float32  # [x,y,z,vx_comp,vy_comp]
}
```

### 레이더 포인트 좌표계

모든 레이더 포인트는 **new ego frame** 기준:
- `x, y, z`: 3D 위치 [m]
- `vx_comp, vy_comp`: ego-compensated velocity [m/s]
  - 정적 배경: 0 + radial 방향 noise
  - 동적 객체: GT velocity (world→new_ego 회전)

---

## 모듈 구성

```
NeuralSensorSim/
├── src/
│   ├── scene/
│   │   ├── data_extractor.py      # nuScenes → frame dict
│   │   ├── gaussian_trainer.py    # gsplat 1.x 학습 루프
│   │   └── dynamic_masker.py      # GT bbox 마스킹
│   ├── synthesis/
│   │   ├── pose_perturber.py      # Ego pose 섭동
│   │   ├── view_synthesizer.py    # 3DGS 렌더링
│   │   ├── radar_synthesizer.py   # 배경 레이더 시뮬레이션
│   │   └── dynamic_compositor.py  # 동적 객체 copy-paste
│   └── utils/
│       └── build_env.py           # Windows CUDA 환경 설정
├── scripts/
│   ├── fit_scene.py               # Phase 1: 3DGS 학습
│   └── synthesize_views.py        # Phase 2+3: 합성 데이터 생성
└── config.yaml

BEVFormerRadar/
├── src/
│   ├── data/
│   │   ├── dataset.py             # nuScenes BEV dataset
│   │   ├── synthetic_dataset.py   # 합성 데이터 로더
│   │   └── mixed_dataset.py       # 혼합 데이터셋
│   └── model/
│       ├── bevformer.py           # BEVFormer 모델
│       └── detection_head.py      # Heatmap detection head
├── train_mixed.py                 # Exp A/B/C 학습
└── evaluate_all.py                # 전체 평가
```
