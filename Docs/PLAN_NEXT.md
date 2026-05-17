# NeuralSensorSim — 전체 계획 (Simple → Complex)

> 작성: 2026-05-11
> 현재 완료: Phase 1~5 (Exp B 1:1만)

---

## 현재 상태 요약

```
Phase 1  3DGS Static Background Fitting     ✅  PSNR 24.5dB, 83K Gaussians
Phase 2  Novel View Synthesis               ✅  120 samples (10 frames × 12 perturb)
Phase 3  Synthetic Radar Generation         ✅  395 pts/sample (vx≈0, vy≈0 — 미수정)
Phase 4  Mixed Training                     🔵  Exp B(1:1)만 완료
Phase 5  Evaluation                         🔵  Exp B만 평가
```

**현재 결과 (Exp B: real+synth 1:1):**
- FP/frame: 49.6 → 20.7 (-58%) ✅
- MOTP: 1.23 → 1.20m ✅
- MOTA: -7.664 → -9.666 ❌ (IDSW 811로 폭증)

**IDSW 폭증 근본 원인:** 합성 데이터의 Doppler(vx≈0, vy≈0)가 잘못되어
모델이 속도 특징을 학습하지 못함 → 트래커 ID 일관성 붕괴

---

## SIMPLE — 데이터 수정

### Step 1: gsplat 1.x 업그레이드 + Densification 활성화 ✅

**현재 문제:**
- `gsplat==0.1.11` — `project_gaussians` / `rasterize_gaussians` API 사용
- Densification 비활성화 (0.1.x backward shape 버그)

**gsplat 1.x API 변경 사항:**
```python
# 0.1.x (현재)
from gsplat import project_gaussians, rasterize_gaussians
xys, depths, radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(...)
result = rasterize_gaussians(...)

# 1.x (목표)
from gsplat import rasterization
renders, alphas, meta = rasterization(
    means=model.means,
    quats=model.get_quats,
    scales=model.get_scales,
    opacities=model.get_opacities.squeeze(-1),
    colors=model.colors,
    viewmats=viewmat.unsqueeze(0),   # (1, 4, 4)
    Ks=K.unsqueeze(0),               # (1, 3, 3)
    width=W, height=H,
)
# renders: (1, H, W, 3)
```

**Densification (1.x):**
```python
from gsplat import DefaultStrategy
strategy = DefaultStrategy()
# train loop 안에서:
strategy.step_pre_backward(params, optimizers, state, step, info)
loss.backward()
strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
```

**완료 내용:**
- `requirements.txt`: `gsplat==0.1.11` → `gsplat>=1.0.0` ✅
- `src/scene/gaussian_trainer.py`: `_render()` 재작성 + `DefaultStrategy` 연결 ✅
- `src/synthesis/view_synthesizer.py`: `GaussianScene.render()` 1.x API로 수정 ✅
- `scripts/render_check.py`: `render_view()` 1.x API로 수정 ✅
- `test_gsplat.py`: 1.x API 테스트로 교체 ✅
  - `project_gaussians` + `rasterize_gaussians` → `rasterization()` 단일 호출
  - 단일 Adam → per-param optimizers dict
  - `strategy_state = {}` → DefaultStrategy가 자동 populate
  - `packed=False` 설정 (DefaultStrategy 호환)
  - `_densify_and_prune` 제거 (DefaultStrategy가 대체)

**설치 필요:** `pip install "gsplat>=1.0.0"` ✅ (설치 완료, JIT 컴파일 성공)

**실제 결과 (15K iters, scene 0):**
- PSNR peak: **30.94dB** (기존 24.5dB → +6.4dB) ✅
- Gaussians: 83K → **7.17M** (densification 10K iter에서 중단, 이후 고정)
- config: `iterations: 15000`, `refine_stop_iter: 10000`

---

### Step 2: Doppler Velocity 수정 ✅

**완료 내용:**
- 핵심 수정: 레이더는 radial velocity만 측정 → 노이즈도 radial 방향으로만 추가
  - 기존: `vx = noise_x, vy = noise_y` (물리적으로 부정확 — 독립 xy 노이즈)
  - 수정: `r_hat = pts_2d / |pts_2d|`, `vx = noise_r * r_hat_x`, `vy = noise_r * r_hat_y`
- `data_extractor.py`: `_get_ego_velocity()` 추가, `frame["ego_velocity"]` 포함
- `radar_synthesizer.py`:
  - `_get_ego_velocity_world()` 추가 (자체 조회 가능)
  - `_simulate_one_channel()` 에 `ego_velocity_world` 파라미터 추가
  - `synthesize()` 에 `ego_velocity_world` 옵션 파라미터 추가 (None이면 자동 조회)
- `synthesize_views.py`: `frame["ego_velocity"]` 전달

**효과:** 속도 노이즈 방향이 포인트 위치와 물리적으로 일관됨 (Step 3 동적 객체 준비 완료)

---

### Step 3: 동적 객체 Copy-Paste Augmentation ✅

**현재:** 3DGS는 정적 배경만 — 차량/보행자 레이더 리턴 없음

**방법:**
1. 소스 선택: ego pose 유사도 기준으로 실제 nuScenes 프레임 선택
2. 변환: GT bbox 내 레이더 포인트를 합성 ego pose로 재투영
   ```python
   # R_synth @ R_real.T @ (pts - t_real) + t_synth
   pts_new = R_synth @ R_real.T @ (pts_obj - t_real) + t_synth
   ```
3. Doppler: nuScenes GT object velocity → radial 투영
   ```python
   v_obj = gt_annotation["velocity"]   # (vx, vy) in global frame
   v_rel = v_obj - v_ego[:2]           # ego-relative
   v_radial = (v_rel * r_hat[:, :2]).sum(axis=1)
   ```
4. GT bbox도 새 ego frame으로 변환하여 함께 저장

**완료 내용:** `src/synthesis/dynamic_compositor.py` 신규 생성
- `_load_radar_ego()`: 5채널 RadarPointCloud → sensor→ego → (N,5) [x,y,z,vx_c,vy_c]
- `_load_gt_with_velocity()`: GT bbox + `nusc.box_velocity()` (world frame)
- `_filter_in_bbox_2d()`: 회전 bbox 필터 (margin=0.5m)
- `compose()`: orig_ego→world→new_ego 변환, GT velocity compensated, radial 노이즈
- `synthesize_views.py`: 배경 포인트 뒤에 `compose()` 연결

---

## MEDIUM — 실험

### Step 4: Exp A / B / C 재실험

Step 1~3 완료 후 공정한 비교:

```powershell
cd BEVFormerRadar

# Exp A: Real only baseline (train_mixed.py 기준)
python train_mixed.py --ratio 0
# → checkpoints/real_only/best.pt

# Exp B: Real + Synth 1:1
python train_mixed.py --ratio 1
# → checkpoints/real+synth_r1/best.pt

# Exp C: Real + Synth 1:3
python train_mixed.py --ratio 3
# → checkpoints/real+synth_r3/best.pt

python evaluate_all.py
```

목표 비교 표:

| Method | MOTA | MOTP | FP/frame | IDSW |
|---|---|---|---|---|
| PMBM Phase29 (radar GT) | -0.177 | 1.20m | 0.8 | 3 |
| Real only (Exp A) | TBD | TBD | TBD | TBD |
| Real + Synth 1:1 (Exp B) | TBD | TBD | TBD | TBD |
| Real + Synth 1:3 (Exp C) | TBD | TBD | TBD | TBD |

---

### Step 5: Idea 3 — 데이터 효율성 곡선

**포트폴리오 핵심 결과.** 합성 데이터가 실제 데이터를 얼마나 대체하는가?

```
실험 설계:
  - real_fraction: [10%, 25%, 50%, 75%, 100%] × Exp A/B/C
  - 각 조합에서 MOTA 측정
  - 결과: "X개 합성 샘플 = Y개 실제 샘플"
```

```python
# train_mixed.py --real-fraction 0.25 --ratio 1
# → 실제 60샘플 + 합성 60샘플로 학습
```

**출력:** learning curve plot (x=실제 데이터 비율, y=MOTA, 라인=Exp A/B/C)

---

## MEDIUM-HIGH — 일반화

### Step 6: Idea 1 — Sim-to-Real Curriculum

합성 데이터로 pre-train → 실제 데이터로 fine-tune.

```
Stage 1: synthetic only (720 samples) — 50 epochs
Stage 2: real only (242 samples) — fine-tune 20 epochs
비교: scratch부터 real-only 학습 vs curriculum
```

**목표:** "합성 pre-training이 실제 데이터만큼 효과적인가?"

---

### Step 7: Idea 2 — Radar Scene Completion (3DGS Prior)

실제 레이더는 희소하고 가려짐 있음. 3DGS 기하학으로 누락된 레이더 리턴 예측.

```
3DGS depth map → 레이더 가시성 추정
가려진 영역 → depth 기반 합성 포인트 추가
BEVFormer 입력: 실제 레이더 + 3DGS 보완 포인트
```

**신규 파일:** `src/synthesis/scene_completer.py`

---

## COMPLEX — 아키텍처

### Step 8: Idea 4 — Doppler-Aware BEV Attention

BEVFormer의 radar branch에 Doppler velocity를 BEV attention에 직접 주입.

```
현재: radar [x,y,z,vx,vy] → linear projection → BEV bias
개선: velocity magnitude + direction → SCA cross-attention 조건 추가
```

**신규 파일:** `BEVFormerRadar/src/model/radar_bev_encoder.py`

---

### Step 9: Idea 5 — Radar Occupancy Intermediate Task

BEVFormer encoder 위에 occupancy head 추가. 합성 데이터로 무제한 label 생성.

```
BEV feature (B, 200×200, 256)
    ↓
Occupancy head (vehicle/ped/free)
    ↓
Detection head
```

**신규 파일:** `BEVFormerRadar/src/model/occupancy_head.py`

---

## 우선순위 요약

```
즉시 (이번 주):
  ✅ Step 1: gsplat 1.x 업그레이드 + _render() 재작성 + densification 활성화
  ✅ Step 2: Doppler velocity 수정 (radial 노이즈 모델)
  ✅ Step 3: dynamic_compositor.py (copy-paste augmentation) — synthesize_views.py 성공

단기 (다음 주):
  □ Step 4: Exp A / B / C 재실험  ← 현재
  □ Step 5: 데이터 효율성 곡선

중기 (이번 달):
  □ Step 6: Sim-to-Real Curriculum
  □ Step 7: Radar Scene Completion

장기 (다음 달):
  □ Step 8: Doppler-Aware BEV Attention
  □ Step 9: Radar Occupancy Intermediate Task
```

---

## 파일 구조 목표

```
NeuralSensorSim/
├── src/
│   ├── scene/           ✅ data_extractor, dynamic_masker
│   │                    ✅ gaussian_trainer (gsplat 1.x 마이그레이션)
│   ├── synthesis/       ✅ pose_perturber, view_synthesizer
│   │                    ✅ radar_synthesizer (Doppler 수정)
│   │                    ✅ dynamic_compositor (copy-paste)
│   │                    🔲 scene_completer (Step 7)
│   └── utils/           ✅ build_env

BEVFormerRadar/
├── src/
│   ├── data/            ✅ synthetic_dataset, mixed_dataset
│   ├── model/           ✅ bevformer, detection_head
│   │                    🔲 radar_bev_encoder (Step 8)
│   │                    🔲 occupancy_head (Step 9)
│   └── evaluation/      ✅ mot_evaluator, nms
├── train_mixed.py       ✅ (--real-fraction 파라미터 추가 필요)
└── evaluate_all.py      ✅
```
