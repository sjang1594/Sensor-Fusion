# MOT Metrics Guide

작성: 2026-05-14

---

## Detection Metrics

| Metric | 의미 | 방향 |
|--------|------|------|
| **TP** | True Positive — GT 있고 검출도 됨 | ↑ |
| **FP** | False Positive — GT 없는데 검출함 | ↓ |
| **FN** | False Negative — GT 있는데 못 잡음 | ↓ |
| **FP/frame** | FP ÷ 전체 프레임 수 | ↓ |
| **FN/frame** | FN ÷ 전체 프레임 수 | ↓ |

---

## Tracking Metrics

| Metric | 의미 | 방향 |
|--------|------|------|
| **IDSW** | ID Switch — 같은 GT 객체에 다른 트래커 ID가 붙은 횟수 | ↓ |
| **MOTP** | Matched TP 쌍의 평균 거리(m) — 위치 정확도 | ↓ |
| **MOTA** | 종합 추적 정확도 (아래 수식) | ↑ (1.0이 perfect) |

---

## MOTA 수식

```
MOTA = 1 - (FP + FN + IDSW) / N_GT
```

- `N_GT`: 전체 프레임에 걸친 GT 객체 수 (분모)
- FP, FN, IDSW 중 하나라도 크면 MOTA가 크게 떨어짐
- **음수도 흔함** — FP가 N_GT보다 많으면 당연히 음수
- 완벽하면 1.0, 아무것도 못 잡으면 FN = N_GT → MOTA = 0 (FP 없을 때)

---

## 실험별 문제 진단 (efficiency.json 기준)

```
Exp A  FP/frame 38, FN/frame  2, IDSW 403  → ID 추적이 문제 (Doppler 부정확)
Exp B  FP/frame 89, FN/frame 85, IDSW   2  → FP/FN 둘 다 높음
Exp C  FP/frame 90, FN/frame 39, IDSW   4  → FP 폭증이 문제
```

- **Exp A**: 검출은 어느 정도 하지만 트래커가 ID를 못 유지
- **Exp B/C**: Doppler 수정으로 IDSW는 잡았지만 FP가 2배 이상 증가

---

## TP=0 Degenerate Case 판별법

```
TP=0, FP≈0, FN=N_GT, IDSW=0 → 모델이 아무것도 예측 안 함
MOTP=0.0 (matched pair 없으면 0 또는 기본값)
MOTA ≈ 1 - N_GT/N_GT = 0 (FP 없으면) or 약간 음수
```

→ MOTA가 나쁘지 않아 보여도 실제론 완전히 collapse된 모델.
→ 반드시 TP/FN도 함께 확인할 것.
