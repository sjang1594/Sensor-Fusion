# Training & Fine-Tuning Idealization

> 작성: 2026-05-16
> 현재 상태: Exp A/B/C 완료 (scene 0, scratch training)

---

## 1. 핵심 연구 질문

> **"nuScenes에서 레이더 데이터가 부족할 때, 3DGS로 생성한 합성 레이더가 실제 데이터를 얼마나 대체할 수 있는가?"**

nuScenes v1.0-mini는 학습용 시퀀스가 6 scenes(242 샘플)에 불과하다.
이 프로젝트는 3DGS로 Novel View를 합성하고 물리 기반 레이더를 시뮬레이션하여
부족한 학습 데이터를 보강하는 파이프라인을 검증한다.

---

## 2. 데이터 파이프라인

```
nuScenes v1.0-mini
├── Train  : scene 0~5  →  242 real 샘플
├── Val    : scene 6~7  →   82 샘플
└── Test   : scene 8~9  →   80 샘플

          ↓  NeuralSensorSim (scene당 ~40분, RTX 4090 기준)

합성 데이터 생성 (scene 0 기준)
├── Phase 1  3DGS 학습        → 7.17M Gaussians, PSNR 30.94dB
├── Phase 2  Novel View Syn.  → 10 frames × 12 perturb = 120 샘플
└── Phase 3  Radar Sim        → Doppler 물리 모델 + Dynamic copy-paste

          ↓  BEVFormerRadar

실험 비교
├── Exp A  Real only              242 샘플
├── Exp B  Real + Synth 1:1       362 샘플
└── Exp C  Real + Synth 1:3       726 샘플  ← 현재 최고
```

### 합성 샘플 포맷 (.npz)

```python
{
    "imgs"           : (6, 3, H, W)  uint8    # 6-camera images
    "cam_intrinsics" : (6, 3, 3)     float32
    "cam_extrinsics" : (6, 4, 4)     float32
    "ego_to_world"   : (4, 4)        float32
    "gt_boxes"       : (M, 7)        float32  # [x,y,z,w,l,h,yaw]
    "gt_labels"      : (M,)          int64    # 0=vehicle, 1=ped, 2=cyclist
    "radar_pts"      : (N, 5)        float32  # [x,y,z,vx_comp,vy_comp]
}
```

---

## 3. 현재 Training 방식

### 모델 구조

```
입력: 6-camera images (448×800) + radar points
        ↓
ResNet50 backbone  (ImageNet pretrained, frozen)
        ↓
FPN (C3/C4/C5, 256ch)
        ↓
BEV Encoder (2 layers, 8-head SCA, 200×200 grid)
  ├── Spatial Cross-Attention  ← camera feature → BEV query
  └── Temporal Self-Attention  ← 이전 BEV feature 재사용
        ↓
Radar Fusion Branch  (radar pts → BEV grid에 scatter)
        ↓
Detection Head (Heatmap + Offset + Size + Yaw)
        ↓
평가: MOTA / MOTP / FP/frame / IDSW
```

### 학습 설정

| 항목 | 값 |
|---|---|
| Optimizer | AdamW |
| LR | 2e-4 (cosine decay) |
| Epochs | 30 |
| Batch size | 1 (메모리 제한) |
| Backbone | Frozen (ImageNet weights) |
| BEV Encoder | Scratch 학습 |

### 현재 결과

| Method | MOTA | MOTP | FP/frame | IDSW |
|---|---|---|---|---|
| PMBM (radar GT 상한선) | -0.177 | 1.20m | 0.8 | 3 |
| Exp A — Real only | -13.23 | 1.26m | 36.4 | 514 |
| Exp B — Real+Synth 1:1 | -11.69 | 1.23m | 35.5 | 422 |
| **Exp C — Real+Synth 1:3** | **-3.75** | **1.24m** | **30.9** | **80** |
| Exp C + thr=1e-3 | **-0.51** | 1.18m | 56.3 | **1** |

**핵심 발견:**
- 합성 1:3 혼합 → MOTA +9.5 point, IDSW -84% (vs real-only)
- FP/frame이 유일한 주요 bottleneck (30.9 vs PMBM 0.8)
- Score calibration 미흡: 모든 confidence < 0.002 → threshold 튜닝 필수

### 현재 방식의 한계

1. **Scratch training** — BEVFormer는 nuScenes full (28,000 샘플) 기준 설계. 242개로는 부족
2. **Score uncalibrated** — heatmap peak이 공간 분산 → FP 폭증
3. **단일 scene** — scene 0만 3DGS 학습. 다양성 부족
4. **val_loss ≠ MOTA** — best checkpoint 선택 기준 오류 (ep5 저장, ep15가 실제 peak)

---

## 4. Idealization — 실험 로드맵

### TIER 1: 즉시 가능 (Low cost, High impact)

#### A. Official BEVFormer Weights Fine-Tuning

**왜 중요한가:**
현재 BEV Encoder를 242개로 scratch 학습 중.
mmdet3d에 nuScenes full로 pre-trained된 공식 가중치를 사용하면
수렴이 훨씬 빠르고 성능 상한선이 올라감.

**실험 설계:**
```
Base: official BEVFormer (ResNet50, nuScenes full trained)
        ↓  fine-tune BEV Encoder + Detection Head only
Exp A': fine-tune on Real 242
Exp C': fine-tune on Real+Synth 1:3

비교:
  scratch Exp A  (-13.23)  vs  fine-tune Exp A'  (?)
  scratch Exp C  ( -3.75)  vs  fine-tune Exp C'  (?)
```

**스토리:**
"pre-trained 없이도 합성 데이터가 9.5 MOTA point 향상 → pre-trained + 합성 조합은 얼마나?"

#### B. Score Calibration (Temperature Scaling)

**문제:** 모든 confidence < 0.002 → threshold 정상 사용 불가

```python
# Detection Head 출력 이후 temperature scaling 추가
logit_scale = nn.Parameter(torch.ones(1))   # 학습 가능
score = torch.sigmoid(raw_logit / logit_scale)
```

**또는:** Focal Loss gamma 조정 (현재 gamma=2.0 → gamma=0.5)으로 high-confidence 예측 강화

**예상 효과:** thr=0.3~0.5 정상 사용 가능 → FP/frame 대폭 감소

---

### TIER 2: 포트폴리오 핵심 (이번 달)

#### C. 데이터 효율성 곡선

**핵심 질문:** "합성 데이터 N개가 실제 데이터 M개와 동등한가?"

```
실험 격자:
  real_fraction : [10%, 25%, 50%, 75%, 100%]
  synth_ratio   : [0 (A), 1 (B), 3 (C)]
  → 15개 실험 조합

커맨드:
  python train_mixed.py --real-fraction 0.25 --ratio 3
  → 실제 61샘플 + 합성 182샘플

출력: Learning curve plot
  x축 = 실제 데이터 비율
  y축 = MOTA
  라인 = Exp A / B / C
```

**논문 Figure 1 후보.** "X개 합성 = Y개 실제" 수치 정량화.

#### D. Sim-to-Real Curriculum Learning

```
Stage 1: Synth 360개 → 50 epoch pre-train   (합성만으로 feature 초기화)
Stage 2: Real 242개  → 20 epoch fine-tune   (실제 도메인 적응)

비교:
  scratch real-only (Exp A)  vs  curriculum (Stage1 → Stage2)
```

**질문:** "합성 pre-training이 실제 데이터 없이도 유효한 초기화를 제공하는가?"

---

### TIER 3: Research Contribution (중장기)

#### E. Doppler-Aware BEV Attention

**현재:** radar [x,y,z,vx,vy] → linear projection → BEV bias (속도 정보 손실)

**개선:**
```python
# 속도 크기 + 방향을 SCA cross-attention에 직접 conditioning
v_mag = torch.norm(radar_vel, dim=-1, keepdim=True)   # 속도 크기
v_dir = radar_vel / (v_mag + 1e-6)                    # 방향 unit vector
doppler_feat = mlp(torch.cat([pos, v_mag, v_dir], -1))
# → SCA attention key에 추가
```

신규 파일: `BEVFormerRadar/src/model/radar_bev_encoder.py`

#### F. Radar Scene Completion (3DGS Prior)

실제 레이더는 희소 + 가려짐. 3DGS depth map으로 누락 리턴 예측.

```
3DGS depth → 가시성 추정 → 합성 포인트 보완 → BEVFormer 입력
```

신규 파일: `NeuralSensorSim/src/synthesis/scene_completer.py`

#### G. Radar Occupancy Intermediate Task

합성 데이터로 occupancy label 무제한 생성 가능.

```
BEV feature (200×200×256)
        ↓
Occupancy head (vehicle/ped/free) ← 합성 데이터 supervision
        ↓
Detection head
```

---

## 5. 우선순위 요약

```
즉시 (이번 주)
  □ A. Official weights fine-tuning 실험 추가     ← 가장 임팩트 큼
  □ B. Temperature scaling / Focal loss 튜닝      ← FP 문제 해결

단기 (이번 달)
  □ C. 데이터 효율성 곡선 (15개 실험)             ← 논문 핵심 Figure
  □ D. Sim-to-Real Curriculum                     ← 연구 차별점

중기
  □ E. Doppler-Aware Attention                    ← 아키텍처 contribution
  □ F. Scene Completion                           ← 새로운 모듈

장기
  □ G. Occupancy Intermediate Task                ← 멀티태스크 학습
```

---

## 6. Compute 계획

| 실험 | 소요 시간 (A100) | 비용 (RunPod) |
|---|---|---|
| Exp A/B/C 재실험 (3회) | ~6시간 | ~$3 |
| Fine-tuning A'/C' (2회) | ~4시간 | ~$2 |
| 데이터 효율성 곡선 (15회) | ~30시간 | ~$15 |
| Curriculum 실험 | ~5시간 | ~$2.5 |
| **전체 TIER 1+2** | **~45시간** | **~$22** |

→ **GPU 살 필요 없이 RunPod $25 이내로 전체 실험 완주 가능.**
→ 5070 구매는 Tier 3 아키텍처 실험 시작할 때 고려.
