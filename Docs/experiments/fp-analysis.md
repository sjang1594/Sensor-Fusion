# FP 증가 원인 분석 (Exp B/C)

작성: 2026-05-14

---

## 현상

합성 데이터를 추가하면 FP/frame이 2배 이상 증가.

```
Exp A (real only)   FP/frame  38.3  MOTA -11.75
Exp B (r=1)         FP/frame  89.4  MOTA  -1.03
Exp C (r=3)         FP/frame  90.5  MOTA  -2.26
```

IDSW는 403 → 2로 극적으로 줄었지만 FP 폭증이 상쇄.

---

## 후보 원인

### 1. Score threshold 문제 (가장 먼저 확인)
합성 데이터 학습 후 모델 confidence 분포가 바뀜.
기본 threshold(1e-4)가 너무 낮아 low-confidence 예측이 전부 FP로 잡힘.
→ `sweep_threshold.py`로 threshold 올리면 FP 줄어드는지 확인.

### 2. 합성 레이더 밀도 문제
합성 레이더 395 pts/sample — 실제 nuScenes보다 밀도 높음.
모델이 "dense radar = object"로 과학습 → 배경 dense region에서도 검출.

### 3. 합성 배경 리턴 문제
3DGS depth에서 뽑은 정적 배경 포인트(벽, 건물)가 실제 레이더엔 없는 패턴.
모델이 그 패턴도 object로 인식 → FP 증가.

---

## 다음 액션

1. `sweep_threshold.py` 로 Exp B/C threshold 곡선 확인 → 원인 1 검증
2. 합성 vs 실제 레이더 point count 비교 → 원인 2 검증
3. 배경 레이더 포인트 시각화 → 원인 3 검증
