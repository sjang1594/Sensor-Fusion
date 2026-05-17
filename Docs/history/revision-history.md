# Revision History

---

## v0.3 — 2026-05-13

### BEVFormerRadar — Exp A/B/C 최종 평가 완료

#### 실험 결과

| Method | MOTA | MOTP | FP/frame | IDSW | best_ep |
|---|---|---|---|---|---|
| PMBM (radar GT) | -0.177 | 1.20m | 0.8 | 3 | — |
| Real only | -13.23 | 1.26m | 36.4 | 514 | 30 |
| Real + Synth 1:1 | -11.69 | 1.23m | 35.5 | 422 | 20 |
| **Real + Synth 1:3** | **-3.75** | **1.24m** | **30.9** | **80** | **15** |

#### 핵심 발견: val_loss vs MOTA Decoupling
- val_loss 기반 best.pt 선택이 실제 MOTA peak 체크포인트를 놓침
- Exp C: val_loss 최소 = ep5 (-9.05) vs 실제 MOTA peak = ep15 (-3.75)
- nuScenes mini 규모에서 ep15 이후 val_loss ↓ but MOTA ↓ (overfitting)

#### Threshold Sweep (Exp C / ep15)

| thr | MOTA | FP/frame | IDSW | 비고 |
|---|---|---|---|---|
| 1e-4 | -3.75 | 30.9 | 80 | 운영 기준점 |
| 5e-4 | -1.11 | 63.5 | 2 | — |
| 1e-3 | **-0.51** | 56.3 | **1** | threshold 최적화 시 |

- thr=1e-3에서 MOTA -0.51, IDSW 1 → PMBM(-0.177 / IDSW 3)에 0.33 차이
- 잔여 문제: score calibration (모든 score < 2e-3)

#### 신규 스크립트
- `evaluate_epochs.py` — epoch별 MOTA curve (--score-thr 옵션)
- `evaluate_all.py` — `--score-thr` 파라미터 추가
- `sweep_threshold.py` — forward pass 1회로 threshold sweep

#### 문서
- `REPORT.md` 작성 (영문, SensorFusion/ 루트)
- `Docs/experiments.md` 최종 결과로 업데이트
- `Docs/results-analysis.md` 신규 (epoch / threshold 분석)
- Revision History v0.3 추가

---

## v0.2 — 2026-05-12

### NeuralSensorSim

#### Step 1: gsplat 1.x 마이그레이션
- `gsplat==0.1.11` → `gsplat>=1.0.0`
- `project_gaussians` + `rasterize_gaussians` → `rasterization()` 단일 API
- `DefaultStrategy` densification 활성화 (`initialize_state()` 필수)
- Per-parameter optimizer 구조로 변경
- PSNR: 24.5dB → **30.94dB** / Gaussians: 83K → 7.17M

#### Step 2: Doppler Velocity 수정
- 기존: 독립 xy noise (물리적으로 부정확)
- 수정: radial 방향 noise만 적용 (`vx = noise_r × r_hat_x`)
- `data_extractor.py`에 `ego_velocity` 추가 (연속 ego_pose 차분)

#### Step 3: 동적 객체 Copy-Paste Augmentation
- `DynamicCompositor` 신규 구현
- nuScenes GT bbox 내 실제 레이더 포인트 추출
- `orig_ego → world → new_ego` 좌표 변환
- GT velocity (world frame) → new_ego compensated velocity

#### 인프라
- Windows MSVC + 한국어 로케일 빌드 문제 3가지 해결
  - cp949 UnicodeDecodeError 수정
  - cl.exe PATH 자동 추가
  - `-Wno-attributes` GCC 플래그 MSVC 호환 패치
- 학습 결과: 15K iters, refine stop @ 10K

### BEVFormerRadar

#### 실험 설계
- Exp A (real only) / Exp B (1:1) / Exp C (1:3) 구조 확립
- `train_mixed.py` 통합 스크립트
- `run_bc.ps1` 자동 실험 스크립트 (경로 버그 수정 포함)

#### 현재 결과 (구버전 데이터 기준)

| Method | MOTA | MOTP | FP/frame | IDSW |
|---|---|---|---|---|
| PMBM (radar GT) | -0.177 | 1.20m | 0.8 | 3 |
| Real only (v0) | -7.664 | 1.23m | 49.6 | 143 |
| Real+Synth 1:1 (old) | -5.928 | 1.23m | 87.4 | 48 |

> 신버전 데이터(Doppler 수정 + 동적 객체) 재실험 진행 중

### 문서
- `Docs/` 폴더 구조 정립
- 시스템 아키텍처 다이어그램 작성
- Step별 기술 문서 작성 (step1~3, windows-build)
- GitBook 업로드용 zip 생성

### 파일 정리
- 중간 PLY 체크포인트 삭제 (`gaussians_01000~14000.ply`)
- epoch_*.pt 체크포인트 삭제 (best.pt만 보존)
- `zip/` 폴더 구조 정립
- 총 ~7GB 절약

---

## v0.1 — 2026-05-07 (초기)

### NeuralSensorSim
- Phase 1~5 완료 (Exp B 1:1만)
- gsplat 0.1.11 기반 (densification 비활성화)
- PSNR 24.5dB, 83K Gaussians
- 합성 레이더: 395pts/sample (Doppler 미수정)

### BEVFormerRadar
- Exp B 1:1 학습 완료
- FP -58% (49.6 → 20.7) ✅
- MOTA -7.664 → -9.666 ❌ (IDSW 폭증)

### 분석
- IDSW 폭증 근본 원인: 합성 데이터 Doppler(vx≈0) 오류
- 동적 객체 레이더 포인트 전무
- v0.2에서 수정 예정으로 기록
