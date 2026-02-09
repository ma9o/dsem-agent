# Inference Method Results

Empirical recovery benchmarks for each inference backend on a standardized 4-latent CT-SEM.

## Test Problem

**Ground truth**: 4 latent processes (Stress -> Fatigue -> Focus -> Performance) with lower-triangular drift, diagonal diffusion, 6 Gaussian indicators (4 identity + 2 cross-loading), and free continuous intercepts.

```
Drift A (continuous-time):
           Stress  Fatigue  Focus   Perf
Stress     -0.80    0.00    0.00   0.00
Fatigue     0.30   -0.50    0.00   0.00
Focus      -0.20   -0.30   -0.60   0.00
Perf        0.00    0.00    0.40  -0.70

Diffusion SD:  [0.40, 0.30, 0.30, 0.25]
Cint:          [0.50, 0.00, 0.30, 0.00]
Obs noise SD:  [0.10, 0.10, 0.10, 0.10, 0.15, 0.15]
```

52 free parameters total (drift 4x4, diffusion 4, cint 4, lambda 8 free loadings, manifest means 6, manifest var 6, t0 means 4, t0 var 4). All observation noise parameters fixed at ground truth. Measurement model (lambda, manifest means, manifest var) estimated jointly.

Metrics:
- **RMSE**: root mean squared error of posterior means vs ground truth (over drift + diffusion + cint)
- **Corr**: Pearson correlation between posterior means and ground truth
- **Coverage**: fraction of parameters where 90% CI contains the true value

## PGAS (Particle Gibbs with Ancestor Sampling)

**Algorithm**: Gibbs-alternate between (1) CSMC sweep with ancestor sampling for latent trajectories and (2) MALA MH steps for parameters. Bootstrap proposals (no gradient shift). Systematic resampling via blackjax.

**Reference**: Lindsten, Jordan & Schon (2014), "Particle Gibbs with Ancestor Sampling."

### Results

| Setting | T | Iters | Warmup | CSMC particles | MH steps | Time | RMSE | Corr | Coverage |
|---------|---|-------|--------|----------------|----------|------|------|------|----------|
| Local CPU | 80 | 200 | 100 | 30 | 10 | 24s | 0.262 | 0.87 | 12/24 |
| A100 GPU | 200 | 500 | 250 | 50 | 15 | 521s | 0.258 | 0.89 | 14/24 |

### Parameter-level coverage (A100, T=200)

| Category | Coverage | Notes |
|----------|----------|-------|
| Drift diagonal | 2/4 | Fatigue and Perf biased negative (overestimated decay) |
| Drift off-diagonal | 9/12 | Best-recovered category |
| Diffusion SD | 1/4 | Hardest category; confounded with drift diagonal |
| Continuous intercept | 2/4 | Moderate recovery |

### Diagnostics

- MALA acceptance: ~37% mean (healthy range 25-50%)
- Step size adapted from 0.05 -> 0.021 via continuous adaptation
- Post-warmup adaptation prevents frozen chains (a failure mode when adaptation stops at warmup)

### Known limitations

1. **Drift-diffusion confounding**: In continuous-time SSMs, faster decay (larger |A_ii|) can be compensated by larger process noise (Q_ii). With bootstrap proposals and finite T, these parameters are partially non-identifiable. This explains the systematic negative bias in drift diagonal + poor diffusion coverage.

2. **Gradient proposal feedback loop**: Gradient-informed CSMC proposals (langevin_step_size > 0) create a positive feedback loop when observation noise is jointly estimated. The gradient nabla_x log g(y|x) = Lambda^T R^{-1}(y - Lambda*x - mu) grows inversely with R, pushing particles too close to observations -> MALA overestimates diffusion/decay -> R drops -> loop amplifies. Fix: default to bootstrap proposals; gradient proposals are opt-in with shift clipping.

3. **Scaling**: 521s on A100 for T=200, 500 iters. The bottleneck is the sequential CSMC sweep (T steps, not parallelizable). GPU helps with the vmap over particles but doesn't help with the time axis.

## Hess-MC2 (SMC-squared with Hessian proposals)

*Results pending.*

## PMMH (Particle Marginal Metropolis-Hastings)

*Results pending.*
