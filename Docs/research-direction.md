# Research Direction

> 작성: 2026-05-07

---

## 현재 포트폴리오 구조

| 프로젝트 | 영역 |
|---|---|
| CameraLidarCalib | 센서 캘리브레이션 |
| FMCW Radar (Phase 29) | 센서 퓨전, MOT (EKF/PHD/PMBM) |
| AerialPhotogrammetry3D | 3D 복원, SfM, 3DGS |
| NeuralGraphicsPortfolio | 3DGS CUDA 구현 |
| NeuralNetworkStudy | Transformer + NeRF/3DGS 이론 |
| LunaEngine | DX12/Vulkan 렌더링 |

```
Radar MOT (EKF/PHD/PMBM)  ─┐
Camera-LiDAR Calib         ─┤→ Sensor Fusion 축
3D Reconstruction (SfM)    ─┘

Transformer (ViT/Swin)     ─┐
NeRF / 3DGS               ─┤→ Neural Perception 축
NeuralNetworkStudy         ─┘
```

두 축이 현재 분리된 상태. **이 둘을 연결하는 지점**이 핵심 방향.

---

## Direction 1: BEV Multi-modal Fusion

### 흐름

```
BEVFormer          BEVFusion           CRN / RCBEVDet      OccFormer
(Camera BEV) → (Camera+LiDAR) → (Camera+Radar) → (Occupancy)
   2022              2022               2023           2023~
```

### 핵심 논문 순서

| 순서 | 논문 | 핵심 개념 |
|---|---|---|
| 1 | **BEVFormer** | BEV query + deformable attention, temporal self-attention |
| 2 | **BEVDepth** | explicit depth supervision으로 camera BEV 정확도 향상 |
| 3 | **BEVFusion (MIT)** | camera + LiDAR를 unified BEV space에서 fusion |
| 4 | **CenterFusion** | radar + camera, frustum association |
| 5 | **CRN** | camera + radar BEV, radar를 BEV feature로 변환 |
| 6 | **OccFormer / SurroundOcc** | BEV → 3D occupancy prediction |

### 기존 작업과 연결

```
FMCW Radar PMBM/EKF   →  CRN/RCBEVDet (Radar BEV)
Transformer (ViT/Swin) →  BEVFormer backbone 이해
Camera-LiDAR Calib     →  BEVFusion 센서 정렬 이해
MORAI 경험             →  nuScenes/Waymo 데이터셋 감각
```

### Research 각도 (논문 기여 포인트)

- **4D Radar + BEV** — Doppler velocity를 BEV feature에 통합하는 방식이 아직 덜 탐구됨
- **Radar uncertainty in BEV** — sparse하고 noisy한 radar를 BEV에서 어떻게 표현할지
- **Temporal radar-camera fusion** — 시간축 정보 활용

### 포트폴리오 프로젝트

```
nuScenes 데이터셋 기반
Camera + 4D Radar → BEV Detection
(BEVFormer backbone + radar branch 추가)
```

### 산업 타겟

42dot, 카카오모빌리티, Hyundai ADAS, Samsung SDC, Waymo, Mobileye

---

## Direction 2: City-scale Neural Reconstruction (Aerial)

### 흐름

```
Block-NeRF/MegaNeRF → Scaffold-GS → VastGaussian → Semantic-GS → 실시간 Digital Twin
   (2022, NeRF)        (2023)         (2024)          (2024)
```

### 핵심 논문 순서

| 순서 | 논문 | 핵심 개념 |
|---|---|---|
| 1 | **Scaffold-GS** | 3DGS에 계층 구조 추가, 대규모 장면 안정화 |
| 2 | **Octree-GS** | octree 기반 LOD, 메모리 효율 |
| 3 | **VastGaussian** | 도시 단위 장면 분할(partition) + 병렬 학습 + merge |
| 4 | **CityGaussian** | 레벨 오브 디테일, 실시간 렌더링 |
| 5 | **LangSplat** | language feature를 Gaussian에 삽입 → semantic query |
| 6 | **GaussianCity** | 항공 + 지상 multi-altitude fusion |

### 기존 작업과 연결

```
AerialPhotogrammetry3D (SfM + 3DGS export) → VastGaussian 진입 장벽 낮음
NeuralGraphicsPortfolio (3DGS CUDA 구현)   → 논문 코드 수정/이해 빠름
LunaEngine (DX12 렌더링)                    → 실시간 렌더러 연결 가능
Camera-LiDAR Calib                          → LiDAR init으로 3DGS 품질 향상
```

### Research 각도 (논문 기여 포인트)

- **항공 특화 partition 전략** — 고도별 스케일 변화를 고려한 scene 분할
- **LiDAR initialization for aerial 3DGS** — SfM sparse init 대신 LiDAR로 품질 향상
- **Temporal aerial 3DGS** — 시간대별 장면 변화 (공사, 식생) 반영
- **Semantic aerial 3DGS** — 건물/도로/식생 분류를 3DGS에 직접 통합

### 포트폴리오 프로젝트

```
실제 항공 이미지 (Aukerman 또는 공개 데이터셋)
→ VastGaussian 방식 partition 적용
→ Semantic label 통합
→ 실시간 뷰어 (LunaEngine 연결)
```

### 산업 타겟

Naver Labs, 카카오, 현대건설/ENG (디지털 트윈), HERE Technologies

---

## 비교 요약

| | BEV Fusion | City-scale 3DGS |
|---|---|---|
| **코어 기술** | Transformer + 센서 퓨전 | Neural Rendering + 3D 재구성 |
| **기존 강점 활용도** | Radar 경험 + Transformer 이론 | SfM/3DGS 구현 + Graphics 엔진 |
| **Research 난이도** | 경쟁 치열 (주류 AV 연구) | 상대적으로 틈새, 차별화 쉬움 |
| **포트폴리오 연속성** | Radar → BEV로 점프 필요 | 기존 작업 바로 확장 |
| **산업 수요** | 매우 넓음 | 특화, 디지털 트윈 한정 |
| **논문 기여 난이도** | 높음 (빅랩 경쟁) | 중간 (aerial 특화 틈새) |

---

## 결론

**단기 포트폴리오 임팩트** → City-scale 3DGS
지금 만든 것들이 직접 이어지고, Naver Labs 타겟에 일관된 스토리가 생김.

**장기 Research + 커리어 확장성** → BEV Fusion
시장이 넓고, 4D Radar + BEV 조합은 논문 각도도 있고 희소성도 있음.

**두 방향을 동시에 가져가는 구조:**

```
City-scale 3DGS         BEV Fusion
(Aerial/Digital Twin)   (AV Perception)
        ↘                  ↙
   공통 기반: 3D Scene Understanding
   + Multi-modal Sensor Fusion
```
