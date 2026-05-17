# Step 1: gsplat 1.x 마이그레이션

## 배경

기존에 사용하던 `gsplat==0.1.11`은 두 가지 문제가 있었다:

1. **Backward shape 버그** — densification 비활성화 필요
2. **구식 API** — `project_gaussians` + `rasterize_gaussians` 두 번 호출

gsplat 1.x는 두 문제를 모두 해결하고 `rasterization()` 단일 호출 API와 `DefaultStrategy` 기반 densification을 제공한다.

---

## API 변경 사항

### 구버전 (gsplat 0.1.x)

```python
from gsplat import project_gaussians, rasterize_gaussians

xys, depths, radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(
    means, scales, 1, quats,
    viewmat, fx, fy, cx, cy, H, W, 16
)
result = rasterize_gaussians(
    xys, depths, radii, conics, num_tiles_hit,
    colors, opacities, H, W, 16, return_alpha=True,
)
render = result[0]  # (H, W, 3)
```

### 신버전 (gsplat 1.x)

```python
from gsplat import rasterization

renders, alphas, info = rasterization(
    means=means,
    quats=quats,
    scales=scales,
    opacities=opacities.squeeze(-1),
    colors=colors,
    viewmats=viewmat.unsqueeze(0),   # (1, 4, 4)
    Ks=K.unsqueeze(0),               # (1, 3, 3)
    width=W,
    height=H,
    packed=False,                    # DefaultStrategy 호환 필요
)
render = renders[0]  # (H, W, 3)
```

---

## Densification: DefaultStrategy

### 핵심 사항

```python
from gsplat.strategy import DefaultStrategy

strategy = DefaultStrategy(
    prune_opa=0.005,
    grow_grad2d=2e-4,
    refine_start_iter=500,
    refine_stop_iter=10000,   # 전체 iter의 2/3 지점에서 중단
    refine_every=100,
    reset_every=3000,
)

# 학습 루프
strategy_state = strategy.initialize_state()   # ← 반드시 호출 ({}가 아님)

for it in range(1, iters + 1):
    for opt in optimizers.values():
        opt.zero_grad()

    render, info = rasterization(...)
    info["means2d"].retain_grad()   # DefaultStrategy 필요

    strategy.step_pre_backward(params, optimizers, strategy_state, it, info)
    loss.backward()
    strategy.step_post_backward(params, optimizers, strategy_state, it, info, packed=False)

    for opt in optimizers.values():
        opt.step()
```

### 주의: 반드시 `initialize_state()` 호출

```python
# ❌ 잘못된 방법 — KeyError: 'grad2d' 발생
strategy_state = {}

# ✅ 올바른 방법
strategy_state = strategy.initialize_state()
# 반환값: {"grad2d": None, "count": None, "scene_scale": 1.0}
```

### Per-parameter Optimizer 구조

DefaultStrategy는 Gaussian 추가/삭제 시 optimizer state를 직접 조작한다. 따라서 단일 optimizer가 아닌 **param별 개별 optimizer** 필요:

```python
params = {
    "means":     model.means,
    "scales":    model.scales,
    "quats":     model.quats,
    "opacities": model.opacities,
    "colors":    model.colors,
}
optimizers = {
    name: torch.optim.Adam([{"params": param, "lr": lr_map[name]}], eps=1e-15)
    for name, param in params.items()
}
```

---

## 학습 결과

| 설정 | PSNR | Gaussian 수 |
|---|---|---|
| gsplat 0.1.11 (densification 비활성화) | 24.5 dB | 83K |
| gsplat 1.x (5K iters) | 28.80 dB | 4.8M |
| gsplat 1.x (15K iters, stop@10K) | **30.94 dB** | **7.17M** |

> Densification이 iter 10K에서 중단된 후 마지막 5K iters는 순수 최적화. Gaussian 수는 10K부터 고정.

---

## config.yaml 설정

```yaml
train:
  iterations: 15000
  densify_start: 500
  densify_interval: 100
  densify_grad_threshold: 2.0e-4
  opacity_reset_interval: 3000

# DefaultStrategy refine_stop_iter = min(iterations * 2/3, 10000) = 10000
```
