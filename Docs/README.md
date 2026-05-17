# Portfolio — Sensor Fusion & Neural Rendering Engineer

> **목표:** AV Perception / Sensor Simulation 분야 Senior Engineer
> **배경:** 2D Computer Vision 전공 → MORAI 자율주행 시뮬레이터 Virtual Sensor 개발 (Radar, Camera)
> **현재:** 제조 분야 VFBuilderV2Plugin 개발 + 포트폴리오 구축 병행

---

## Technical Stack

| Domain | Skills |
|---|---|
| **Sensor Fusion** | FMCW Radar, Camera, LiDAR, Extrinsic Calibration, EKF/UKF/IMM, GM-PHD, LMB, JPDA, PMBM |
| **BEV Perception** | BEVFormer, Multi-camera attention, Radar-Camera fusion, nuScenes |
| **Neural Rendering** | 3D Gaussian Splatting (CUDA), NeRF, Novel View Synthesis, gsplat |
| **3D Vision / CV** | SfM, MVS, Point Cloud, OpenCV, Open3D, Depth Estimation |
| **GPU Programming** | CUDA kernel optimization, DirectX 12 (DXR), Vulkan, HLSL/GLSL |
| **Systems** | C++20, Python, PyTorch, real-time rendering, deferred rendering |

---

## Projects

---

### 1. NeuralSensorSim + BEVFormerRadar *(Ongoing — Flagship)*

> **한 줄 요약:** 3D Gaussian Splatting으로 생성한 합성 레이더 데이터로 BEV 인식 성능을 향상시키는 Neural Sensor Simulation 파이프라인

**배경:** nuScenes 실제 레이더 데이터 부족 문제 → 3DGS 기반 novel view synthesis로 합성 데이터 생성 → BEV 모델 학습에 활용

**경로:** `SensorFusion/NeuralSensorSim` + `SensorFusion/BEVFormerRadar`

#### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     NeuralSensorSim Pipeline                    │
│                                                                 │
│  nuScenes                                                       │
│  v1.0-mini ──► Phase 1: 3DGS Static Background Fitting         │
│   (10 frames)       │   gsplat 1.x + DefaultStrategy           │
│   LiDAR 83K pts     │   PSNR: 24.5dB → 30.94dB                │
│                     │   Gaussians: 83K → 7.17M                 │
│                     ▼                                           │
│             Phase 2: Novel View Synthesis                       │
│                     │   PosePerturber (±2m lat, ±4m lon, ±8°)  │
│                     │   12 perturbs × 10 frames = 120 samples  │
│                     ▼                                           │
│             Phase 3: Synthetic Radar Generation                 │
│                     │   ┌─────────────────────────────┐        │
│                     │   │ RadarSynthesizer             │        │
│                     │   │  • 5채널 FOV/range 필터링   │        │
│                     │   │  • Opacity 기반 샘플링       │        │
│                     │   │  • Doppler: radial noise     │        │
│                     │   │    vx = noise_r × r_hat_x   │        │
│                     │   └─────────────────────────────┘        │
│                     │   ┌─────────────────────────────┐        │
│                     │   │ DynamicCompositor (NEW)      │        │
│                     │   │  • Real radar GT bbox 추출  │        │
│                     │   │  • orig_ego→world→new_ego   │        │
│                     │   │  • GT velocity compensated  │        │
│                     │   └─────────────────────────────┘        │
│                     │   395 pts/sample → +dynamic objects      │
│                     ▼                                           │
│        outputs/synthetic/scene_00/  (120 .npz samples)         │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     BEVFormerRadar Pipeline                     │
│                                                                 │
│  Real Data          Synthetic Data                              │
│  (nuScenes)    +    (3DGS generated)                           │
│   242 samples        120 samples                                │
│       │                   │                                     │
│       └──── MixedDataset ─┘   ratio=1.0 (1:1) or 3.0 (1:3)    │
│                   │                                             │
│                   ▼                                             │
│       ┌───────────────────────┐                                 │
│       │      BEVFormer        │                                 │
│       │  ResNet50 Backbone    │                                 │
│       │  + FPN (C3/C4/C5)    │                                 │
│       │  + BEV Encoder        │                                 │
│       │    (2L, 8-head SCA)  │                                 │
│       │  + Radar Fusion       │                                 │
│       └───────────────────────┘                                 │
│                   │                                             │
│                   ▼                                             │
│          Detection Head → MOTA / MOTP / FP / IDSW              │
└─────────────────────────────────────────────────────────────────┘
```

#### 실험 설계

| Experiment | 데이터 구성 | 목표 |
|---|---|---|
| Exp A | Real only (242 samples) | 공정한 baseline |
| Exp B | Real + Synth 1:1 (362 samples) | 합성 데이터 효과 검증 |
| Exp C | Real + Synth 1:3 (606 samples) | 비율 최적화 |

#### 현재 결과 (Exp B — 구버전 데이터 기준)

| Metric | Baseline | Exp B (1:1) | 변화 |
|---|---|---|---|
| FP/frame | 49.6 | 20.7 | **-58%** ✅ |
| MOTP | 1.23m | 1.20m | ✅ |
| MOTA | -7.664 | -9.666 | ❌ (IDSW 폭증) |

> **IDSW 원인 분석 및 수정:** 합성 레이더 Doppler(vx≈0)가 물리적으로 부정확 → radial noise model 수정 + 동적 객체 copy-paste 추가 → 재실험 진행 중

#### 향후 계획 (Simple → Complex)

```
SIMPLE   ✅ gsplat 1.x + densification (PSNR 30.94dB)
         ✅ Doppler velocity 수정 (radial noise model)
         ✅ Dynamic object copy-paste augmentation
MEDIUM   🔄 Exp A/B/C 재실험 (진행 중)
         □  데이터 효율성 곡선 (X synth = Y real)
COMPLEX  □  Sim-to-Real Curriculum
         □  Radar Scene Completion (3DGS prior)
         □  Doppler-Aware BEV Attention
         □  Radar Occupancy Intermediate Task
```

---

### 2. FMCW Radar Multi-Object Tracking *(Complete — Phase 29)*

> **한 줄 요약:** EKF부터 PMBM까지 전체 MOT 알고리즘 스택을 구현하고 nuScenes에서 벤치마크

**경로:** `fmcw-radar-sensor/`

#### Algorithm Evolution

```
Phase 18  EKF (5-radar 360° fusion)
    │
Phase 19  IMM (3-model: CV-SLOW / CV-FAST / STOP)
    │
Phase 20  4D MIMO Radar Simulation
    │
Phase 21  JPDAF (Salmonds' approximation)
    │
Phase 22  GM-PHD (Vo & Ma 2006)
    │
Phase 23  LMB (Labeled Multi-Bernoulli)     FP 792→122 (-85%)
    │                                        IDSW 22→5 (-77%)
Phase 24  LMB + Camera Gate (6-cam PnP)
    │
Phase 29  PMBM (Poisson Multi-Bernoulli Mixture)  ← Best
```

#### PMBM Final Results (nuScenes v1.0-mini)

| Tracker | MOTA | MOTP | FP/frame | IDSW |
|---|---|---|---|---|
| EKF baseline | -1.312 | 1.35m | 49.6 | 22 |
| GM-PHD | -0.312 | 1.28m | 8.4 | 8 |
| LMB | -0.231 | 1.23m | 2.1 | 5 |
| **PMBM** | **-0.177** | **1.20m** | **0.8** | **3** |

#### PMBM Architecture

```
┌──────────────────────────────────────────────┐
│              PMBM Filter                     │
│                                              │
│  PPP (Poisson Point Process) Birth           │
│  r_birth = pD × l_PPP / D_j                 │
│                 │                            │
│                 ▼                            │
│  Bernoulli Update                            │
│  r_new = (δ_i + r(1-pD)) / (1+δ_i-r×pD)   │
│                 │                            │
│  Multi-Bernoulli Mixture                     │
│  → Confirmed tracks (r > 0.5)               │
└──────────────────────────────────────────────┘
```

---

### 3. Camera-LiDAR Extrinsic Calibration *(Complete)*

> **한 줄 요약:** 체커보드 없이 자연 코너를 이용한 카메라-LiDAR 외부 파라미터 자동 보정

**경로:** `CameraLidarCalib/`

#### Pipeline

```
Camera Images ──► SIFT Feature Detection
                        │
LiDAR Point Cloud ──► Plane Segmentation (RANSAC)
                        │
                        ▼
              Edge Correspondence Matching
                        │
                        ▼
              PnP + Bundle Adjustment
                        │
                        ▼
              Extrinsic Matrix T_cam_lidar
```

#### Results

| Metric | Value |
|---|---|
| Intrinsic reproj error | **0.035 px** |
| Extrinsic RMSE | **0.707 px** |
| Rotation error | **0.15°** |
| Translation error | **0.64 cm** |
| Pipeline runtime | **11.8 sec** |

---

### 4. AerialPhotogrammetry3D *(Complete — Naver Labs Digital Twin)*

> **한 줄 요약:** 드론 항공 영상 → SfM → MVS → Poisson 메쉬 → UV Atlas → 디지털 트윈 8단계 파이프라인

**경로:** `AerialPhotogrammetry3D/`

#### Pipeline Architecture

```
Synthetic Aerial Images
        │
        ▼
Phase 1: Data Generation (synthetic scene)
        │
Phase 2: Feature Matching (SIFT + FLANN)
        │
Phase 3: Incremental SfM + Bundle Adjustment
        │   Camera poses + sparse point cloud
        ▼
Phase 4: Dense MVS (StereoSGBM)
        │   Dense depth maps → point cloud
        ▼
Phase 5: Poisson Surface Reconstruction
        │   Watertight mesh
        ▼
Phase 6: UV Atlas (xatlas)
        │   Texture coordinates
        ▼
Phase 7: Occlusion Inpainting (LaMa)
        │   Fill texture holes
        ▼
Phase 8: Evaluation
            Chamfer distance / Pose error / Reproj error
```

#### Extension Roadmap

```
Direction B: DL Depth (Depth Anything v2 / UniDepth)
Direction C: SfM → COLMAP 변환 → 3DGS 연결
Direction D: 실제 드론 데이터 (UrbanScene3D / DJI)
```

---

### 5. LunaEngine *(C++20 — DX12 + Vulkan Dual Backend)*

> **한 줄 요약:** DirectX 12 + Vulkan 듀얼 백엔드 실시간 렌더링 엔진 (PBR + DXR + Deferred + TAA + Bloom + SSR)

**경로:** `LunaEngine-source/`

#### Rendering Architecture

```
┌─────────────────────────────────────────────┐
│              LunaEngine HAL                 │
│  IRenderBackend / IRenderContext / IBuffer  │
└────────────────┬────────────────────────────┘
                 │
       ┌─────────┴─────────┐
       │                   │
┌──────▼──────┐    ┌───────▼──────┐
│  DX12       │    │  Vulkan       │
│  Backend    │    │  Backend      │
│             │    │               │
│ • D3D12MA   │    │ • GPU-driven  │
│ • DXR RT    │    │   indirect    │
│ • CSM (4x)  │    │ • Bindless    │
│ • G-buffer  │    │ • IBL         │
│   Deferred  │    │ • TAA + Bloom │
│ • SSR       │    │ • SSR         │
│ • HLSL SM6  │    │ • GLSL/SPIRV  │
└─────────────┘    └───────────────┘
```

#### Feature Milestone

| Phase | Feature | Status |
|---|---|---|
| 1–4 | DX12 triangle → PBR → DXR Shadows | ✅ |
| 5 | API correctness + PBR material pipeline | ✅ |
| 6 | Render Graph (ImportTexture / AddPass) | ✅ |
| 7 | Deferred Rendering (G-buffer + Lighting) | ✅ |
| 8 | Cascaded Shadow Maps (4-cascade 2K array) | ✅ |
| 9 | SSAO (구현 예정) | 🔲 |
| 10–11 | Vulkan GPU-driven indirect + IBL | ✅ |
| 12–14 | Vulkan TAA + Bloom + SSR + HDR | ✅ |
| 15–18 | HLSL→GLSL migration (19 shaders) | ✅ |

#### Render Graph Flow (DX12)

```
BeginFrame
    │
    ├─► CSM Pass (4 cascades, depth-only)
    │       csm_depth.vert.hlsl
    │
    ├─► G-buffer Pass (3 MRT: albedo/normal/material)
    │       gbuffer.frag.hlsl
    │
    ├─► Deferred Lighting Pass
    │       deferred_lighting.frag.hlsl
    │       Cook-Torrance PBR + 5-tap PCF shadow
    │
    ├─► SSR Pass (view-space DDA march)
    │
    └─► ImGui Overlay
```

---

### 6. NeuralGraphicsPortfolio — 3DGS CUDA *(NVIDIA Neural Graphics 목표)*

> **한 줄 요약:** CUDA C++로 3D Gaussian Splatting 순전파 렌더러 직접 구현

**경로:** `NeuralGraphicsPortfolio/GaussianSplattingCUDA/`

#### Architecture

```
PLY Point Cloud (5000 Gaussians)
        │
        ▼
CUDA Tile Rasterizer
  • Project 3D Gaussians → 2D screen splats
  • Tile assignment (16×16 tiles)
  • Depth sort per tile (133K tile pairs)
  • Alpha blending front-to-back
        │
        ▼
Output Image (out.png)
  • 80% pixel coverage confirmed
  • Colored sphere scene
```

#### Build System (Windows CUDA)

```
PowerShell: build_ninja.ps1
  → vcvars64 env capture
  → CMake -G Ninja
  → set_source_files_properties(main.cpp LANGUAGE CUDA)
     ← required: main.cpp uses <cub/cub.cuh>
```

---

## Architecture Overview — 전체 포트폴리오

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Sensor Perception Stack                          │
│                                                                     │
│  Physical Layer    Simulation Layer       Perception Layer          │
│  ──────────────    ────────────────       ────────────────          │
│                                                                     │
│  FMCW Radar   ──► Radar Simulation  ──► MOT (PMBM)                │
│  Camera       ──► 3DGS Renderer     ──► BEV Detection              │
│  LiDAR        ──► Point Cloud       ──► Calibration                │
│                                                                     │
│  [fmcw-radar-sensor]  [NeuralSensorSim]  [BEVFormerRadar]         │
│  [CameraLidarCalib]                                                 │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    Rendering & Neural Graphics                      │
│                                                                     │
│  Real-time Rendering          Neural Rendering                      │
│  ──────────────────           ───────────────                       │
│                                                                     │
│  LunaEngine                   GaussianSplattingCUDA                │
│  • DX12 + Vulkan              • CUDA rasterizer                    │
│  • PBR + DXR + CSM            • Forward pass confirmed             │
│  • Deferred + TAA             • Backward (WIP)                     │
│  • IBL + Bloom + SSR                                                │
│                               AerialPhotogrammetry3D               │
│                               • SfM → 3DGS 연결 계획              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 이루고자 하는 것

### 단기 (2026 Q2)

```
□ NeuralSensorSim Exp A/B/C 완료 → 비교 표 작성
□ 데이터 효율성 곡선 (핵심 포트폴리오 결과)
□ LunaEngine DamagedHelmet 스크린샷 (CSM + G-buffer 시각화)
□ GaussianSplatting CUDA backward pass
```

### 중기 (2026 Q3)

```
□ NeuralSensorSim Scene 1~5 확장 (720 samples)
□ Doppler-Aware BEV Attention 구현
□ Sim-to-Real Curriculum 실험
□ AerialPhotogrammetry → COLMAP → 3DGS 연결
```

### 장기 목표

```
□ nuScenes Full Dataset (700 scenes, 28K samples)
□ nuScenes 공식 평가 (NDS / mAP)
□ 논문 작성 준비
   → "3DGS Neural Radar Simulation for BEV Perception"
```

### 지원 목표 포지션

| 회사 | 포지션 | 연관 포트폴리오 |
|---|---|---|
| **42dot** | AV Perception / Sensor Fusion | NeuralSensorSim + PMBM |
| **Waymo** | Sensor Simulation | NeuralSensorSim |
| **NAVER Labs** | 3D Vision / Digital Twin | AerialPhotogrammetry3D |
| **NVIDIA** | Neural Graphics | GaussianSplattingCUDA + LunaEngine |

---

## Key Technical Highlights

> 면접 / 기술 발표에서 강조할 포인트

1. **Radar 물리 이해 + BEV Fusion + 3DGS Simulation의 희소한 조합**
   - 대부분의 엔지니어는 하나만 함. 세 가지를 연결한 시스템 설계 경험

2. **PMBM까지의 MOT 알고리즘 스택 전체 구현**
   - EKF → IMM → GM-PHD → LMB → JPDA → PMBM 직접 구현 및 비교

3. **실제 문제 진단 및 수정 경험**
   - IDSW 폭증 원인을 Doppler 물리 오류로 진단 → radial noise model 수정
   - gsplat 1.x Windows MSVC 컴파일 문제 3단계 해결 (cp949 / MSVC path / GCC flags)

4. **End-to-End 파이프라인 설계**
   - nuScenes raw data → 3DGS 학습 → novel view → 레이더 생성 → BEV 학습 → 평가

5. **DX12 + Vulkan 듀얼 백엔드 렌더링 엔진**
   - HAL 추상화로 두 API를 동시 지원하는 실제 동작 엔진
