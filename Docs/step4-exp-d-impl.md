# Exp D 구현 — Real Camera + Radar Augmentation

> 작성: 2026-05-18
> 목적: 3DGS 카메라를 제거하고 레이더 augmentation 효과만 순수하게 측정

---

## 배경 (왜 Exp D가 필요한가)

density_fix 실험(2026-05-16) 결과 분석:

| | 효과 |
|---|---|
| 합성 카메라 (3DGS novel view) | FP +32% — frozen backbone domain gap |
| 합성 레이더 (Doppler 수정) | IDSW −40% — 실제 개선 |

레이더 augmentation 효과는 실제로 있지만, 3DGS 카메라가 domain gap으로 상쇄함.
→ **카메라는 real 원본 유지, 레이더만 augmentation** 하면 순수한 레이더 효과 측정 가능.

---

## 구현된 파일

### 1. `NeuralSensorSim/scripts/augment_radar_real.py` (신규)

**역할:** nuScenes train 프레임에 cross-scene copy-paste radar augmentation 적용.

**파이프라인:**
```
train scene 0~5 각 프레임 (target)
  → 다른 scene에서 무작위 2개 donor 프레임 선택
  → donor의 GT bbox 기반으로 동적 객체 레이더 포인트 샘플링
  → target ego frame에 배치 (상대 좌표 그대로 유지)
  → real 카메라 이미지 원본 + augmented radar → .npz 저장
```

**핵심 설계 결정:**

| 결정 | 이유 |
|------|------|
| GT bbox에서 직접 샘플링 | real nuScenes radar가 희소(206pts/frame) → bbox 필터로 object pts 거의 0 |
| 클래스별 표준 크기 사용 | 차량(2×4.5×1.6), 보행자(0.6×0.6×1.8) |
| 상대 좌표 유지 (world 변환 없음) | cross-scene이면 world 변환 시 objects가 수백m 밖에 배치됨 |
| velocity만 frame 변환 | world frame velocity → target ego heading으로 회전 |

**클래스별 샘플링 수:**
- 차량 (cls=0): 4 pts
- 보행자 (cls=1): 1 pt
- 자전거 (cls=2): 2 pts

**결과:**
```
242 samples (train 전체)
real 206.0 pts/frame → aug 364.3 pts/frame (+158.3, +77%)
저장: NeuralSensorSim/outputs/radar_aug/
```

**버그 수정 이력 (구현 중 발견):**

| 버그 | 원인 | 수정 |
|------|------|------|
| `train_scenes` KeyError | NeuralSensorSim config에 해당 키 없음 | `--train-scenes` CLI 인수로 변경 |
| +0.0 pts (1차) | dynamic_compositor bbox 필터가 sparse radar에서 작동 안 함 | GT bbox 직접 샘플링으로 교체 |
| +0.0 pts (2차) | cross-scene world 변환 시 objects 50m 초과 | world 변환 제거, 상대 좌표 유지 |

---

### 2. `BEVFormerRadar/src/data/radar_aug_dataset.py` (신규)

**역할:** augment_radar_real.py 출력 로딩 + 인라인 augmentation.

**SyntheticDataset과의 차이:**
- 카메라: uint8 real 이미지 (3DGS 렌더링 아님) → domain gap 없음
- 레이더: real + copy-pasted dynamic objects

**인라인 augmentation (B단계):**
```
match_real_density  → 포인트 수를 실제 분포로 서브샘플링 (μ=113.6, σ=32.0)
xyz noise jitter    → σ=0.05m (range error 모델)
수평 flip           → 카메라 6개 FLIP_ORDER + vx 부호 반전
```

---

### 3. `BEVFormerRadar/train_mixed.py` (수정)

**추가된 인수:**
```python
parser.add_argument("--radar-aug-dir", default=None,
                    help="radar_aug .npz 디렉토리 (real cam + aug radar, Exp D)")
```

**우선순위 로직:**
```python
if args.radar_aug_dir and os.path.isdir(args.radar_aug_dir):
    # Exp D: RadarAugDataset 사용
elif args.synth_dir and args.ratio > 0 and os.path.isdir(args.synth_dir):
    # Exp B/C: SyntheticDataset 사용 (기존)
else:
    # Exp A: real only
```

**exp_label:** `real+radar_aug_r{ratio}` (checkpoint 디렉토리명)

---

### 4. `BEVFormerRadar/run_radar_aug.ps1` (신규)

**실행 순서:**
```
Step 0: augment_radar_real.py (이미 완료)
Exp A:  real_only baseline         → checkpoints/radar_aug/real_only/
Exp D1: real cam + radar aug 1:1   → checkpoints/radar_aug/real+radar_aug_r1/
Exp D3: real cam + radar aug 1:3   → checkpoints/radar_aug/real+radar_aug_r3/
eval:   evaluate_all.py --ckpt-root checkpoints/radar_aug
```

---

## 기대 결과

| Method | MOTA | FP/f | IDSW | 의미 |
|--------|------|------|------|------|
| real_only (Exp A) | -9.504 | 30.4 | 456 | baseline |
| real+synth_r1 (3DGS cam 포함) | -10.232 | 40.2 | 273 | camera domain gap |
| **real+radar_aug_r1 (Exp D)** | **TBD** | **≈30?** | **<273?** | domain gap 제거 |

가설: FP ≈ real_only 수준 유지 + IDSW < r1 → "camera domain gap이 FP 증가 원인" 확인.

---

## 실행 방법

```powershell
cd BEVFormerRadar
powershell -ExecutionPolicy Bypass -File run_radar_aug.ps1
```

또는 개별 실행:
```powershell
# Exp D1
python train_mixed.py `
  --radar-aug-dir ..\NeuralSensorSim\outputs\radar_aug `
  --ratio 1 `
  --ckpt-root checkpoints/radar_aug `
  2>&1 | Tee-Object logs/radar_aug_exp_d1.log
```
