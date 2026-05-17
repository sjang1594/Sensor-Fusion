# Step 3: 동적 객체 Copy-Paste Augmentation

## 문제

3DGS는 **정적 배경만** 표현한다. 차량/보행자 등 동적 객체는 학습 중 마스킹되어 제거되었기 때문에 합성 레이더 포인트에 동적 객체가 전혀 없다.

결과: 모델이 "레이더 포인트 = 정적 배경"으로 학습 → 동적 객체 탐지 능력 저하.

---

## 접근 방법: Copy-Paste Augmentation

실제 nuScenes 프레임의 GT bbox 내 레이더 포인트를 추출하여 합성 ego pose에 재투영한다.

```
원본 프레임 (real nuScenes)
  GT bbox 내 실제 레이더 포인트 추출
              │
              ▼
  orig_ego → world → new_ego 변환
              │
              ▼
  GT velocity (world) → new_ego frame compensated velocity
              │
              ▼
  합성 배경 포인트에 합성
```

---

## 구현: DynamicCompositor

```python
class DynamicCompositor:
    def compose(self,
                sample_token: str,
                orig_ego_to_world: np.ndarray,   # (4,4)
                new_ego_to_world: np.ndarray,    # (4,4) 섭동 포즈
                background_pts: np.ndarray,      # (N,5) 배경 포인트
                ) -> np.ndarray:                  # (N+M,5)
```

### 1. 실제 레이더 포인트 로드

```python
# 5채널 RadarPointCloud → sensor frame → ego frame 변환
pc = RadarPointCloud.from_file(pc_path)
pts = pc.points.T           # (N, 18)
xyz = pts[:, :3]            # sensor frame 위치
vx_comp = pts[:, 8]        # ego-compensated vx (sensor frame)
vy_comp = pts[:, 9]        # ego-compensated vy (sensor frame)

# sensor → ego
R_s2e = calibrated_sensor["rotation_matrix"]
xyz_ego = (R_s2e @ xyz.T).T + t_s2e
v_ego   = (R_s2e @ [vx, vy, 0].T).T  # velocity도 회전
```

### 2. GT Bbox 필터링 (회전 bbox)

```python
# bbox-aligned frame으로 변환하여 필터링
cos_y = cos(-yaw); sin_y = sin(-yaw)
dx = pts[:, 0] - cx; dy = pts[:, 1] - cy
local_x = cos_y * dx - sin_y * dy
local_y = sin_y * dx + cos_y * dy

in_box = (|local_x| <= l/2 + margin) & (|local_y| <= w/2 + margin)
```

### 3. 새 Ego Frame으로 변환

```python
# orig_ego → world → new_ego
R_orig2w = orig_ego_to_world[:3, :3]
t_orig2w = orig_ego_to_world[:3, 3]
R_w2new  = new_ego_to_world[:3, :3].T
t_w2new  = -R_w2new @ new_ego_to_world[:3, 3]

pts_world   = (R_orig2w @ pts_sel[:, :3].T).T + t_orig2w
pts_new_ego = (R_w2new  @ pts_world.T).T       + t_w2new
```

### 4. Compensated Velocity 계산

```python
# GT velocity (world frame) → new_ego frame으로 회전
v_obj_world  = nusc.box_velocity(ann_token)[:2]   # world frame
v_obj_ne     = R_w2new[:2, :2] @ v_obj_world      # new_ego frame

# radial 방향 노이즈 추가 (Step 2와 동일 물리 모델)
r_hat = pts_ne[:, :2] / r[:, None].clip(min=1e-6)
noise = np.random.randn(n) * 0.2   # [m/s]
vx_comp = v_obj_ne[0] + noise * r_hat[:, 0]
vy_comp = v_obj_ne[1] + noise * r_hat[:, 1]
```

---

## 결과 포인트 구조

```
배경 포인트 (RadarSynthesizer)    +    동적 객체 포인트 (DynamicCompositor)
  [x, y, z, vx≈0, vy≈0]                [x, y, z, vx_gt, vy_gt]
  정적 배경에서 radial noise              GT velocity 기반 실제 속도
```

---

## 핵심 설계 결정

**Isaac Sim을 쓰지 않은 이유:**

Isaac Sim으로 동적 객체를 시뮬레이션하면 더 사실적이지만, 프로젝트의 핵심 기여는 **3DGS 기반 Neural Sensor Simulation**이다. Isaac Sim을 동적 객체에 사용하면 "왜 Isaac Sim 전체를 안 쓰나?"라는 질문이 생긴다.

**Copy-paste 방식의 장점:**
- 3DGS 파이프라인의 기여가 명확하게 유지됨
- 실제 nuScenes 레이더 특성(노이즈, RCS 등)이 자연스럽게 보존됨
- 구현이 단순하고 GT 정보를 직접 활용 가능
</content>
</invoke>