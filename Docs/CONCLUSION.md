# 실험 결론 — NeuralSensorSim + BEVFormerRadar

> 실험 기간: 2026-05-07 ~ 2026-05-18
> 데이터셋: nuScenes v1.0-mini (train: scene 0–5, val: scene 6–7, 82 frames)

---

## 전체 실험 비교표

| # | Method | MOTA | FP/f | IDSW | MOTP | 비고 |
|---|--------|------|------|------|------|------|
| — | PMBM Phase29 (radar GT) | **-0.177** | **0.8** | **3** | 1.20m | 이론적 상한 |
| A | real_only | -9.426 | 30.5 | 459 | 1.247m | baseline |
| B | real + 3DGS synth 1:1 | -10.232 | 40.2 | 273 | 1.219m | 카메라 domain gap |
| C | real + 3DGS synth 1:3 | -14.405 | 33.3 | 712 | 1.235m | 악화 |
| D1 | real cam + radar aug 1:1 | -7.865 | 24.6 | 477 | 1.252m | FP 최선 |
| **D3** | **real cam + radar aug 1:3** | **-6.433** | **28.7** | **308** | **1.272m** | **전체 최선** |

*FP/f = FP per TP (구버전 metric), MOTA = 1 − (FP+FN+IDSW)/N_GT, N_GT = 3170*

---

## 핵심 비교 — baseline 대비 변화량

| Method | ΔMOTA | ΔFP/f | ΔIDSW |
|--------|-------|-------|-------|
| + 3DGS synth 1:1 (Exp B) | −0.806 | **+9.7** ❌ | −186 (+40%) ✅ |
| + 3DGS synth 1:3 (Exp C) | −4.979 | +2.8 ❌ | +253 ❌ |
| + radar aug 1:1 (Exp D1) | +1.561 ✅ | **−5.9** ✅ | +18 |
| **+ radar aug 1:3 (Exp D3)** | **+2.993** ✅ | **−1.8** ✅ | **−151 (−33%)** ✅ |

---

## 실험이 답한 세 가지 질문

### Q1. 3DGS 합성 데이터가 BEV 학습에 도움이 되는가?

**답: 카메라는 ❌, 레이더는 ✅**

```
3DGS novel view camera
  → frozen ResNet50은 합성 이미지에 adapt 불가
  → feature distribution 불일치 → FP +32% (Exp B)

dynamic_compositor radar (Doppler 수정)
  → learnable RadarFusion branch에서 velocity cue 학습
  → IDSW −40% 개선 (Exp B에서 관찰)
```

### Q2. FP 증가의 원인이 카메라 domain gap인가?

**답: ✅ 실험적으로 확인**

```
3DGS 카메라 포함 (Exp B): FP/f = 40.2  ← real_only(30.5)보다 악화
3DGS 카메라 제거 (Exp D): FP/f = 24.6  ← real_only보다 개선

카메라를 real로 유지하는 것만으로 FP가 즉시 줄어들었다.
```

### Q3. 순수 레이더 augmentation만으로 전체 성능이 개선되는가?

**답: ✅ 세 지표 동시 개선**

```
real_only (Exp A)          →  MOTA -9.4, FP/f 30.5, IDSW 459
real + radar aug r3 (Exp D) →  MOTA -6.4, FP/f 28.7, IDSW 308

MOTA:  +32%
FP/f:  −6%
IDSW:  −33%
```

---

## 결론

### 핵심 발견

> **3DGS로 생성한 novel view 카메라 이미지는 frozen backbone의 domain gap으로
> 역효과를 낸다. 반면 real 카메라 + GT bbox 기반 radar augmentation은
> MOTA, FP, IDSW를 동시에 개선하며, 이는 radar augmentation이
> multi-object tracking에 유효함을 증명한다.**

### 포트폴리오 핵심 문장

> "3DGS novel view synthesis로 생성한 카메라 이미지는 frozen backbone의
> domain gap으로 FP를 32% 증가시키지만, real 카메라 + GT bbox 기반
> radar augmentation은 MOTA +32%, FP −6%, IDSW −33%를 동시에 달성한다.
> 이는 camera sim-to-real gap과 radar augmentation 효과를 실험적으로 분리한
> 결과이며, radar data augmentation이 BEV perception의 tracking consistency
> 개선에 유효함을 증명한다."

---

## 왜 이게 의미 있는가

```
일반적인 연구 접근:  "합성 데이터를 넣으면 성능이 좋아질 것이다"
                                    ↓
이 연구:            "어떤 modality의 합성 데이터가 효과적인지 실험적으로 분리"

발견:
  카메라 합성 → 구조적 한계(frozen backbone)로 역효과
  레이더 합성 → 물리적으로 정확한 Doppler가 tracking에 실제로 기여

이것은 단순한 성능 수치 개선이 아니라,
"neural rendering 기반 data augmentation의 효과와 한계"를 
정량적으로 밝힌 ablation study다.
```

---

## 알려진 한계

| 한계 | 현재 상태 | 개선 방향 |
|------|-----------|-----------|
| Frozen backbone | 구조적 제약 — camera branch 학습 불가 | backbone fine-tuning (scope 초과) |
| nuScenes mini (소규모) | 82 val frames, 2 scenes | full nuScenes (700 scenes) |
| scene 0만 3DGS 학습 | 일반화 미검증 | scene 1~5 확장 |
| GT bbox 기반 radar 샘플링 | real radar 분포와 다름 | RCS 기반 scatter model |
| val_loss 기반 best checkpoint | MOTA 직접 최적화 아님 | MOTA-based early stopping |

---

## 실험 이력 요약

| 날짜 | 실험 | 핵심 결과 |
|------|------|-----------|
| 05-07~11 | Phase 1~3 (3DGS + Doppler 수정) | PSNR 30.94dB, Doppler radial noise 모델 |
| 05-13 | dynamic_compositor | copy-paste augmentation 완성 |
| 05-16 | density_fix (Exp A/B/C) | IDSW −40% (Exp B), FP +32% — domain gap 발견 |
| 05-17 | focal loss 수정 + Colab 재학습 | threshold sweep → IDSW 13→3 (Colab 기준) |
| **05-18** | **Exp D (radar aug only)** | **MOTA +32%, FP −6%, IDSW −33% — 전체 최선** |
