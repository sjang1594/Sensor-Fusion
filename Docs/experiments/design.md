# 실험 설계 및 결과

## 실험 설계

| 실험 | 학습 데이터 | 목적 |
|---|---|---|
| **Exp A** | Real only (242 samples) | 공정한 baseline 확립 |
| **Exp B** | Real + Synth 1:1 (362 samples) | 합성 데이터 효과 검증 |
| **Exp C** | Real + Synth 1:3 (726 samples) | 최적 비율 탐색 |

**공통 설정:**
- Model: BEVFormer (ResNet50 frozen + 2L SCA + CenterPoint head)
- Epochs: 30, LR: 2e-4 (cosine decay), AdamW
- Val: nuScenes mini val (2 scenes, 82 frames)

---

## 평가 지표

| 지표 | 설명 | 방향 |
|---|---|---|
| **MOTA** | Multi-Object Tracking Accuracy = 1 − (FP+FN+IDSW)/N_GT | ↑ |
| **MOTP** | Mean distance error for matched detections [m] | ↓ |
| **FP/frame** | 프레임당 False Positive | ↓ |
| **IDSW** | ID Switch 누적 횟수 | ↓ |

---

## 최종 결과 (2026-05-13)

### Best Checkpoint 기준 (val_loss 최소 → epoch sweep으로 수정)

| Method | MOTA | MOTP | FP/frame | IDSW | best_ep |
|---|---|---|---|---|---|
| **PMBM (radar GT)** | **-0.177** | **1.20m** | **0.8** | **3** | — |
| Exp A — Real only | -13.23 | 1.26m | 36.4 | 514 | 30 |
| Exp B — Real+Synth 1:1 | -11.69 | 1.23m | 35.5 | 422 | 20 |
| **Exp C — Real+Synth 1:3** | **-3.75** | **1.24m** | **30.9** | **80** | **15** |

> **FP 개선:** 49.6 (old baseline) → 30.9 (-37%)
> **IDSW 개선:** 514 (Exp A) → 80 (Exp C, -84%)

---

## Epoch별 MOTA 분석 (Exp C)

> val_loss 최소 epoch(5)와 실제 MOTA peak(15)이 불일치 — 소규모 데이터셋 과적합 현상

| Epoch | MOTA | FP/frame | IDSW | 비고 |
|---|---|---|---|---|
| 5  | -9.05 | 27.8 | 409 | ← val_loss 최소 (best.pt 저장됨) |
| 10 | -7.07 | 34.7 | 208 | |
| **15** | **-3.75** | **30.9** | **80** | ← **실제 MOTA peak** |
| 20 | -5.65 | 26.9 | 220 | |
| 25 | -8.58 | 31.1 | 355 | |
| 30 | -7.32 | 28.2 | 344 | |

**결론:** val_loss 기반 checkpoint 선택은 5.3 MOTA point 손실 유발. MOTA 직접 모니터링 또는 early stopping 기준 변경 필요.

---

## Score Threshold 분석 (Exp C / ep15)

> 모든 score < 2e-3 (score calibration 미흡)

| Threshold | MOTA | MOTP | FP/frame | IDSW |
|---|---|---|---|---|
| 1e-4 (default) | -3.75 | 1.24m | 30.9 | 80 |
| 5e-4 | -1.11 | 1.26m | 63.5 | 2 |
| **1e-3** | **-0.51** | **1.18m** | 56.3 | **1** |
| ≥5e-3 | 0.00 | — | 0.0 | 0 |

thr=1e-3에서 **MOTA -0.51, IDSW 1** → PMBM 대비 0.33 MOTA 차이.
잔여 과제: FP/frame=56.3 (PMBM=0.8) → score calibration 또는 focal loss 튜닝.

---

## 재현 방법

```powershell
# NeuralSensorSim: 3DGS 학습
cd NeuralSensorSim
python scripts/fit_scene.py --scene 0

# NeuralSensorSim: 합성 데이터 생성 (Doppler + Dynamic)
python scripts/synthesize_views.py --scene 0

# BEVFormerRadar: 전체 실험 (A/B/C)
cd ../BEVFormerRadar
python train_mixed.py --ratio 0                    # Exp A
python train_mixed.py --ratio 1                    # Exp B
python train_mixed.py --ratio 3                    # Exp C

# 평가 — epoch별 MOTA curve
python evaluate_epochs.py

# 평가 — threshold sweep
python sweep_threshold.py --ckpt checkpoints/real+synth_r3/epoch_015.pt
```

---

## 다음 단계

| Step | 내용 | 예상 효과 |
|---|---|---|
| Step 5 | 데이터 효율성 곡선 (real_fraction × ratio) | "합성 N개 = 실제 M개" 정량화 |
| Step 6 | Sim-to-Real Curriculum | 합성 pre-train → 실제 fine-tune |
| Step 7 | Score Calibration (temperature scaling) | FP/frame 감소 |
