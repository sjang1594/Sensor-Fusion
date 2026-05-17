# Research Roadmap

> Current status: Phase 1–5 complete (3DGS fitting, novel view synthesis, synthetic radar, Exp A/B/C training, evaluation)

---

## Near-Term Experiments

### Exp A / B / C Full Re-run

After Steps 1–3 (gsplat 1.x, Doppler correction, DynamicCompositor), run fair comparison:

| Method | MOTA | MOTP | FP/frame | IDSW |
|---|---|---|---|---|
| PMBM Phase29 (radar GT) | -0.177 | 1.20m | 0.8 | 3 |
| Real only (Exp A) | -13.23 | — | — | 514 |
| Real + Synth 1:1 (Exp B) | -11.69 | — | — | 422 |
| Real + Synth 1:3 (Exp C) | **-3.75** | — | 30.9 | **80** |

Threshold=1e-3 post-processing reduces Exp C to MOTA=-0.506, IDSW=1 (near PMBM level).

---

### Data Efficiency Curve

Core portfolio result: how much does synthetic data substitute for real data?

```
Experiment design:
  real_fraction: [10%, 25%, 50%, 75%, 100%] × Exp A/B/C
  Measure MOTA at each combination
  Result: "X synthetic samples ≡ Y real samples"
```

```python
# python train_mixed.py --real-fraction 0.25 --ratio 1
# → train on 60 real + 60 synthetic samples
```

Output: learning curve plot (x=real data fraction, y=MOTA, lines=Exp A/B/C)

---

## Generalization Experiments

### Sim-to-Real Curriculum

Synthetic pre-training → real fine-tuning.

```
Stage 1: synthetic only (720 samples) — 50 epochs
Stage 2: real only (242 samples) — fine-tune 20 epochs
Compare: scratch real-only vs curriculum
```

Question: "Is synthetic pre-training as effective as real data?"

---

### Radar Scene Completion via 3DGS Prior

Real radar is sparse and occluded. Use 3DGS geometry to predict missing radar returns.

```
3DGS depth map → radar visibility estimate
Occluded regions → depth-guided synthetic points added
BEVFormer input: real radar + 3DGS-completed points
```

New module: `NeuralSensorSim/src/synthesis/scene_completer.py`

---

## Architecture Extensions

### Doppler-Aware BEV Attention

Inject Doppler velocity directly into BEVFormer's radar cross-attention.

```
Current:  radar [x,y,z,vx,vy] → linear projection → BEV bias
Proposed: velocity magnitude + direction → SCA cross-attention conditioning
```

New file: `BEVFormerRadar/src/model/radar_bev_encoder.py`

---

### Radar Occupancy Intermediate Task

Add occupancy head above BEVFormer encoder. Synthetic data enables unlimited occupancy labels.

```
BEV feature (B, 200×200, 256)
    ↓
Occupancy head (vehicle/ped/free)
    ↓
Detection head
```

New file: `BEVFormerRadar/src/model/occupancy_head.py`