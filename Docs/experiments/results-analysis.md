# 결과 심층 분석

## 1. val_loss vs MOTA Decoupling 문제

### 현상

BEVFormerRadar 학습 시 val_loss 최소 epoch에 best.pt를 저장하는 전략이 실제 MOTA peak을 놓침.

**Exp C (Real+Synth 1:3) 사례:**

```
ep05  MOTA=-9.053  IDSW=409  ← best.pt 저장 (val_loss 최소)
ep10  MOTA=-7.065  IDSW=208
ep15  MOTA=-3.753  IDSW= 80  ← 실제 MOTA peak (+5.3 point 손실)
ep20  MOTA=-5.647  IDSW=220
ep25  MOTA=-8.576  IDSW=355
ep30  MOTA=-7.318  IDSW=344
```

### 원인

- nuScenes mini: 학습 6 scenes (242 samples), val 2 scenes (82 frames) — 극소 규모
- ep15 이후 heatmap loss는 계속 감소하지만 detection 분포가 변화 → FP/IDSW 악화
- `score_threshold=0.0001` (매우 낮음) → 모든 낮은 confidence 예측이 FP로 계산됨

### 대응

1. `evaluate_epochs.py`로 모든 epoch_*.pt 평가 후 MOTA 기준 선택
2. 장기적: `train_mixed.py`에 MOTA 기반 early stopping 추가
3. score_threshold를 하이퍼파라미터로 공동 튜닝

---

## 2. Score Calibration 분석

### Score 분포 (Exp C / ep15, 82 frames)

```
전체 raw detection: 136,394개
  p50.0 : 2.75e-05   ← 중간값이 매우 낮음
  p75.0 : 1.26e-04
  p90.0 : 4.02e-04
  p95.0 : 8.28e-04
  p99.0 : 1.37e-03
  p99.9 : 1.95e-03   ← 최고 confidence도 0.002 미만
```

모든 score가 0.005 미만 → 표준 threshold(0.3~0.5) 사용 불가.

### Threshold별 성능

| thr | MOTA | FP/f | IDSW | TP | FP 누적 |
|---|---|---|---|---|---|
| 1e-5 | -8.61 | 31.4 | 320 | 889 | 27,874 |
| 1e-4 | -3.75 | 30.9 | 80 | 395 | 12,211 |
| 5e-4 | -1.11 | 63.5 | 2 | 56 | 3,558 |
| 1e-3 | **-0.51** | 56.3 | **1** | 29 | 1,632 |

**thr=1e-3**: TP=29, FP=1,632 (82 frames) → Precision=1.7%, Recall=0.9%

IDSW=1은 PMBM 수준(IDSW=3)이지만, FP가 압도적으로 높음.

### 해석

고신뢰 예측(thr=1e-3 이상)이 실제 객체 위치와 일치하지 않음 — 모델의 heatmap peak이 공간적으로 분산되어 있음. Score calibration (temperature scaling, focal loss weight 조정)이 필요.

---

## 3. 합성 데이터 비율 효과 요약

| Exp | 합성/실제 | Best MOTA | Best ep | IDSW |
|---|---|---|---|---|
| A (real only) | 0:1 | -13.23 | 30 | 514 |
| B (1:1) | 1:1 | -11.69 | 20 | 422 |
| C (1:3) | 3:1 | **-3.75** | **15** | **80** |

- 합성 비율이 높을수록 MOTA 개선, IDSW 감소
- Exp C: Exp A 대비 MOTA +9.5 point, IDSW -84%
- FP는 비율과 무관하게 유사 (30~36 /frame) — FP는 model capacity 문제

---

## 4. PMBM 대비 최종 격차

| 지표 | PMBM (GT) | Exp C ep15 thr=1e-4 | 격차 |
|---|---|---|---|
| MOTA | -0.177 | -3.753 | 3.58 |
| MOTP | 1.20m | 1.24m | 0.04m |
| FP/frame | 0.8 | 30.9 | 30.1 |
| IDSW | 3 | 80 | 77 |

threshold=1e-3 적용 시:

| 지표 | PMBM (GT) | Exp C ep15 thr=1e-3 | 격차 |
|---|---|---|---|
| MOTA | -0.177 | **-0.506** | **0.33** |
| IDSW | 3 | **1** | **-2** |
| FP/frame | 0.8 | 56.3 | 55.5 |

MOTA/IDSW 관점에서는 PMBM에 근접. **FP가 유일한 주요 bottleneck**.
