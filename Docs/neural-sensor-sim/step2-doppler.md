# Step 2: Doppler Velocity 수정

## 문제: IDSW 폭증의 원인

Exp B (real+synth 1:1) 결과에서 IDSW가 143 → 811로 폭증했다.

**근본 원인:** 합성 레이더 포인트의 Doppler 모델이 물리적으로 부정확했다.

---

## 기존 구현의 문제

```python
# ❌ 기존 코드 — 물리적으로 부정확
vx = np.random.randn(n_sample) * cfg.VEL_NOISE_MPS   # x 방향 독립 노이즈
vy = np.random.randn(n_sample) * cfg.VEL_NOISE_MPS   # y 방향 독립 노이즈
```

**왜 문제인가?**

레이더는 **radial velocity(시선 방향 속도)만** 측정한다. 노이즈도 radial 방향에서만 발생하는데, 기존 코드는 x, y를 독립적으로 추가해서 물리적으로 불가능한 방향의 속도 노이즈가 생겼다. 모델이 "레이더 포인트에는 velocity가 없다"고 잘못 학습 → 트래커 ID 일관성 파괴 → IDSW 폭증.

---

## 수정: Radial Noise Model

```python
# ✅ 수정된 코드 — radial 방향으로만 노이즈 추가
# 정적 배경: ego-compensated velocity = 0 (물체가 정지 → 보상 후 0)
# 노이즈는 레이더 측정 방향(radial)으로만 존재

# 2D radial 단위 벡터 (new_ego frame xy 평면)
pts_ne_2d = pts_noisy_ne[:, :2]                                   # (n, 2)
r_2d      = np.linalg.norm(pts_ne_2d, axis=1, keepdims=True).clip(min=1e-6)
r_hat_2d  = pts_ne_2d / r_2d                                      # (n, 2) 단위벡터

# radial 방향으로만 노이즈
v_radial_noise = np.random.randn(n_sample) * cfg.VEL_NOISE_MPS   # scalar noise
vx = v_radial_noise * r_hat_2d[:, 0]                              # x 성분
vy = v_radial_noise * r_hat_2d[:, 1]                              # y 성분
```

---

## 물리적 배경

nuScenes 레이더 포맷에서 `vx_comp`, `vy_comp`는 **ego-compensated velocity**:

```
raw Doppler = (v_object - v_ego) · r_hat

정적 배경에서:
  v_object = 0 (물체 정지)
  compensated = 0    ← ego 이동 보상 후 0

동적 물체에서:
  compensated = v_object_in_ego_frame
```

따라서 정적 배경 합성 포인트의 compensated velocity는 항상 0이고, 노이즈는 레이더 측정 방향(radial)에서만 발생한다.

---

## Ego Velocity 추가 (Step 3 준비)

동적 객체 처리를 위해 `data_extractor.py`에 ego velocity 계산 추가:

```python
def _get_ego_velocity(self, sample_token: str) -> np.ndarray:
    """연속 ego_pose로부터 ego velocity 추정. Returns (2,) [vx, vy] world frame m/s."""
    sample = self.nusc.get("sample", sample_token)

    if sample["next"]:
        # 다음 프레임과의 차분으로 속도 계산
        sd_nxt = self.nusc.get("sample_data", ...)
        dt   = (sd_nxt["timestamp"] - t0) * 1e-6   # μs → s
        vel  = (pos1 - pos0) / max(dt, 1e-6)
        return vel[:2].astype(np.float32)

    return np.zeros(2, dtype=np.float32)
```

`frame["ego_velocity"]` 키로 frame dict에 포함 → 동적 객체 Doppler 계산에 사용.
