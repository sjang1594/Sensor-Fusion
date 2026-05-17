# 프로젝트 개요

## 한 줄 요약

> **3D Gaussian Splatting으로 생성한 합성 레이더 데이터로 BEV 인식 성능을 향상시키는 Neural Sensor Simulation 파이프라인**

---

## 배경 및 동기

nuScenes 데이터셋은 자율주행 연구에 널리 사용되지만, **레이더 샘플 수가 제한적**이다. 특히 v1.0-mini는 학습용 시퀀스가 6개 씬(약 242 샘플)에 불과하여 BEV 인식 모델 학습에 부족하다.

**핵심 아이디어:** 3D Gaussian Splatting(3DGS)으로 학습된 정적 배경 모델에서 novel view를 합성하고, 해당 viewpoint에서의 레이더 포인트를 물리적으로 시뮬레이션하여 학습 데이터를 보강한다.

---

## 프로젝트 구성

```
SensorFusion/
├── NeuralSensorSim/     ← 합성 데이터 생성 (3DGS + Radar)
└── BEVFormerRadar/      ← BEV 인식 모델 학습 및 평가
```

---

## 기술 스택

| 컴포넌트 | 기술 |
|---|---|
| 3D 장면 표현 | 3D Gaussian Splatting (gsplat 1.x) |
| 레이더 시뮬레이션 | Physics-based ray casting + Doppler model |
| 동적 객체 | nuScenes GT copy-paste augmentation |
| BEV 인식 | BEVFormer (ResNet50 + SCA) |
| 데이터셋 | nuScenes v1.0-mini |

---

## 현재 완료된 Phase

| Phase | 내용 | 결과 |
|---|---|---|
| 1 | nuScenes → 3DGS Static BG Fitting | PSNR **30.94dB**, 7.17M Gaussians |
| 2 | Novel View Synthesis | 120 samples (10 frames × 12 perturb) |
| 3 | Synthetic Radar Generation | 물리 기반 Doppler, 동적 객체 paste |
| 4 | Mixed Training (Exp A/B/C) | Exp C 최적: MOTA -3.75, IDSW 80 |
| 5 | Evaluation & Analysis | threshold=1e-3 → MOTA -0.51, IDSW 1 |
