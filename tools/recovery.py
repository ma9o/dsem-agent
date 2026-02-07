"""Parameter recovery on a realistic 4D DSEM model — runs on Modal GPU.

Ground truth: 4 latent processes (Stress -> Fatigue -> Focus -> Performance)
with off-diagonal drift, a 6-indicator measurement model, and continuous
intercepts. Fits with SVI, NUTS, and PMMH; prints recovery tables.

Usage:
    modal run tools/recovery.py            # remote GPU
    uv run python tools/recovery.py        # local CPU (smoke test)
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # project root

# ---------------------------------------------------------------------------
# Modal setup (only used when invoked via `modal run`)
# ---------------------------------------------------------------------------
try:
    import modal

    GPU = "L4"

    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git")
        .pip_install("uv")
        .uv_sync(uv_project_dir=str(ROOT), groups=["dev"], frozen=True)
        .uv_pip_install("jax[cuda12]", gpu=GPU)
        .env({"PYTHONPATH": "/root/src"})
        .add_local_dir(ROOT / "src" / "dsem_agent", remote_path="/root/src/dsem_agent")
    )
    app = modal.App("dsem-recovery", image=image)
    HAS_MODAL = True
except Exception:
    HAS_MODAL = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def header(title: str):
    w = 70
    print("=" * w)
    print(f" {title}")
    print("=" * w)


def print_recovery(name: str, true_val, samples_arr) -> bool:
    """Print one row of recovery stats. Returns True if 90% CI covers truth."""
    import jax.numpy as jnp

    mean = float(jnp.mean(samples_arr))
    std = float(jnp.std(samples_arr))
    q5 = float(jnp.percentile(samples_arr, 5))
    q95 = float(jnp.percentile(samples_arr, 95))
    true = float(true_val)
    covered = q5 <= true <= q95
    bias = mean - true
    tag = "OK" if covered else "MISS"
    print(
        f"  {name:<20s}  true={true:+.3f}  post={mean:+.3f}+-{std:.3f}"
        f"  90%CI=[{q5:+.3f},{q95:+.3f}]  {tag}  bias={bias:+.3f}"
    )
    return covered


def print_matrix(m, row_labels, col_labels):
    """Print a matrix with row/col labels."""
    import numpy as np

    a = np.array(m)
    w = 8
    print(f"{'':>{w}s}", "".join(f"{c:>{w}s}" for c in col_labels))
    for i, rl in enumerate(row_labels):
        vals = "".join(f"{a[i, j]:>{w}.3f}" for j in range(a.shape[1]))
        print(f"{rl:>{w}s}{vals}")


def extract_drift(samples, n_latent):
    """Extract full drift matrix samples, handling both SVI and MCMC formats.

    SVI Predictive returns deterministic sites: samples["drift"] is (S, n, n).
    PMMH returns raw sites: samples["drift_diag_pop"] and ["drift_offdiag_pop"].
    NUTS returns both.
    """
    import jax.numpy as jnp

    if "drift" in samples and samples["drift"].ndim == 3:
        return samples["drift"]

    # Reconstruct from raw sites
    drift_diag = samples["drift_diag_pop"]      # (S, n)
    drift_offdiag = samples["drift_offdiag_pop"]  # (S, n_offdiag)

    def assemble_one(diag, offdiag):
        d = jnp.diag(-jnp.abs(diag))
        idx = 0
        for i in range(n_latent):
            for j in range(n_latent):
                if i != j:
                    d = d.at[i, j].set(offdiag[idx])
                    idx += 1
        return d

    from jax import vmap
    return vmap(assemble_one)(drift_diag, drift_offdiag)


def extract_diffusion_diag(samples, n_latent):
    """Extract diffusion diagonal samples (Cholesky SD)."""
    import jax.numpy as jnp

    if "diffusion" in samples and samples["diffusion"].ndim == 3:
        return jnp.array([samples["diffusion"][:, i, i] for i in range(n_latent)]).T

    return samples["diffusion_diag_pop"]


def extract_cint(samples):
    """Extract continuous intercept samples."""
    if "cint" in samples and samples["cint"].ndim == 2:
        return samples["cint"]
    return samples["cint_pop"]


def extract_lambda(samples):
    """Extract lambda matrix samples (or None if not available)."""
    if "lambda" in samples and samples["lambda"].ndim == 3:
        return samples["lambda"]
    return None


# ---------------------------------------------------------------------------
# Recovery report for one backend
# ---------------------------------------------------------------------------


def report_recovery(
    method_name, samples, true_drift, true_diff_diag, true_cint,
    true_lambda, true_manifest_means, names, n_latent, n_manifest,  # noqa: ARG001
):
    import jax.numpy as jnp
    import numpy as np

    header(f"RESULTS: {method_name}")

    offdiag_pairs = [(i, j) for i in range(n_latent) for j in range(n_latent) if i != j]

    drift_s = extract_drift(samples, n_latent)
    diff_diag_s = extract_diffusion_diag(samples, n_latent)
    cint_s = extract_cint(samples)
    lam_s = extract_lambda(samples)

    # --- Drift diagonal ---
    print("  DRIFT DIAGONAL (auto-effects)")
    drift_diag_ok = 0
    for i in range(n_latent):
        drift_diag_ok += print_recovery(names[i], true_drift[i, i], drift_s[:, i, i])
    print(f"  Coverage: {drift_diag_ok}/{n_latent}")
    print()

    # --- Drift off-diagonal ---
    print("  DRIFT OFF-DIAGONAL (cross-effects)")
    drift_off_ok = 0
    for i, j in offdiag_pairs:
        drift_off_ok += print_recovery(
            f"{names[j]}->{names[i]}", true_drift[i, j], drift_s[:, i, j]
        )
    print(f"  Coverage: {drift_off_ok}/{len(offdiag_pairs)}")
    print()

    # --- Diffusion ---
    print("  DIFFUSION (process noise SD)")
    diff_ok = 0
    for i in range(n_latent):
        diff_ok += print_recovery(names[i], true_diff_diag[i], diff_diag_s[:, i])
    print(f"  Coverage: {diff_ok}/{n_latent}")
    print()

    # --- CINT ---
    print("  CONTINUOUS INTERCEPT")
    cint_ok = 0
    for i in range(n_latent):
        cint_ok += print_recovery(names[i], true_cint[i], cint_s[:, i])
    print(f"  Coverage: {cint_ok}/{n_latent}")
    print()

    # --- Lambda ---
    if lam_s is not None:
        print("  LAMBDA (cross-loadings)")
        lam_ok, lam_n = 0, 0
        for i in range(n_latent, n_manifest):
            for j in range(n_latent):
                lam_ok += print_recovery(
                    f"M{i}<-{names[j]}", true_lambda[i, j], lam_s[:, i, j]
                )
                lam_n += 1
        print(f"  Coverage: {lam_ok}/{lam_n}")
        print()

    # --- Drift matrix comparison ---
    drift_post = jnp.mean(drift_s, axis=0)
    print("  True A:")
    print_matrix(true_drift, names, names)
    print()
    print("  Posterior mean A:")
    print_matrix(drift_post, names, names)
    print()
    print("  Bias (post - true):")
    print_matrix(drift_post - true_drift, names, names)
    print()

    # --- Summary stats ---
    pairs = []
    for i in range(n_latent):
        pairs.append((float(true_drift[i, i]), float(jnp.mean(drift_s[:, i, i]))))
    for i, j in offdiag_pairs:
        pairs.append((float(true_drift[i, j]), float(jnp.mean(drift_s[:, i, j]))))
    for i in range(n_latent):
        pairs.append((float(true_diff_diag[i]), float(jnp.mean(diff_diag_s[:, i]))))
    for i in range(n_latent):
        pairs.append((float(true_cint[i]), float(jnp.mean(cint_s[:, i]))))

    trues = np.array([p[0] for p in pairs])
    posts = np.array([p[1] for p in pairs])
    rmse = float(np.sqrt(np.mean((posts - trues) ** 2)))
    corr = float(np.corrcoef(trues, posts)[0, 1]) if len(trues) > 2 else 0.0

    print(f"  RMSE={rmse:.4f}  Corr={corr:.4f}")
    print()
    return rmse, corr


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def run(local: bool = False):
    import time

    import jax
    import jax.numpy as jnp
    import jax.random as random
    import jax.scipy.linalg as jla
    import numpy as np

    from dsem_agent.models.ssm import SSMModel, SSMPriors, SSMSpec, discretize_system, fit

    # NOTE: do NOT enable x64 — cuthbert's PF uses jax.lax.cond with
    # int32 outputs that break under x64 promotion.

    # -- Tuning knobs --
    # SVI uses PF (tolerates noisy gradients; Kalman gives NaN via Lyapunov solver).
    # NUTS uses Kalman (exact + smooth; handles occasional NaN via trajectory rejection).
    # PMMH uses PF (gradient-free sampler).
    if local:
        T = 80
        SVI_STEPS = 1000
        SVI_SAMPLES = 200
        SVI_PARTICLES = 200
        NUTS_WARMUP = 200
        NUTS_SAMPLES = 200
        PMMH_WARMUP = 50
        PMMH_SAMPLES = 50
        PMMH_PARTICLES = 100
    else:
        T = 200
        SVI_STEPS = 5000
        SVI_SAMPLES = 1000
        SVI_PARTICLES = 500
        NUTS_WARMUP = 1000
        NUTS_SAMPLES = 1000
        PMMH_WARMUP = 500
        PMMH_SAMPLES = 500
        PMMH_PARTICLES = 2000

    header("ENVIRONMENT")
    print(f"JAX {jax.__version__}  backend={jax.default_backend()}  devices={jax.devices()}")
    print(f"T={T}")
    print(f"SVI:  {SVI_STEPS} steps, {SVI_SAMPLES} samples ({SVI_PARTICLES} particles)")
    print(f"NUTS: {NUTS_WARMUP} warmup, {NUTS_SAMPLES} samples (Kalman likelihood)")
    print(f"PMMH: {PMMH_WARMUP} warmup, {PMMH_SAMPLES} samples ({PMMH_PARTICLES} particles)")
    print()

    # ==================================================================
    # 1. Ground truth
    # ==================================================================
    header("GROUND TRUTH")

    n_latent, n_manifest, dt = 4, 6, 0.5
    names = ["Stress", "Fatigue", "Focus", "Perf"]

    true_drift = jnp.array([
        [-0.8,  0.0,   0.0,  0.0],
        [ 0.3, -0.5,   0.0,  0.0],
        [-0.2, -0.3,  -0.6,  0.0],
        [ 0.0,  0.0,   0.4, -0.7],
    ])
    true_diff_diag = jnp.array([0.4, 0.3, 0.3, 0.25])
    true_cint = jnp.array([0.5, 0.0, 0.3, 0.0])

    true_lambda = jnp.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.6, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.4, 0.7],
    ])
    true_manifest_means = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 2.0])
    true_mvar_diag = jnp.array([0.1, 0.1, 0.1, 0.1, 0.15, 0.15])

    eigs = np.linalg.eigvals(np.array(true_drift))
    print("Drift matrix A (continuous-time):")
    print_matrix(true_drift, names, names)
    print()
    print("Eigenvalues:", "  ".join(f"{e.real:+.3f}" for e in eigs))
    assert all(e.real < 0 for e in eigs), "Drift is not stable!"
    print("All eigenvalues negative — stable.")
    print()
    print(f"Lambda ({n_manifest}x{n_latent}), diffusion diag={[round(float(x), 2) for x in true_diff_diag]}")
    print(f"CINT={[round(float(x), 2) for x in true_cint]}")
    print()

    # ==================================================================
    # 2. Simulate
    # ==================================================================
    header("SIMULATE")

    diff_cov = jnp.diag(true_diff_diag) @ jnp.diag(true_diff_diag).T
    Ad, Qd, cd = discretize_system(true_drift, diff_cov, true_cint, dt)
    Qd_chol = jla.cholesky(Qd + jnp.eye(n_latent) * 1e-8, lower=True)
    R_chol = jla.cholesky(jnp.diag(true_mvar_diag), lower=True)

    key = random.PRNGKey(42)
    key, init_key = random.split(key)
    x0 = jla.cholesky(jnp.eye(n_latent) * 0.5, lower=True) @ random.normal(init_key, (n_latent,))

    states = [x0]
    for _ in range(T - 1):
        key, nk = random.split(key)
        states.append(Ad @ states[-1] + cd.flatten() + Qd_chol @ random.normal(nk, (n_latent,)))
    latent = jnp.stack(states)

    key, ok = random.split(key)
    obs = jax.vmap(lambda x: true_lambda @ x + true_manifest_means)(latent) + \
        random.normal(ok, (T, n_manifest)) @ R_chol.T
    times = jnp.arange(T, dtype=float) * dt

    print(f"Latent  shape={latent.shape}  mean={np.array(latent.mean(0)).round(2)}")
    print(f"Obs     shape={obs.shape}  mean={np.array(obs.mean(0)).round(2)}")
    print()

    # ==================================================================
    # 3. Model spec (shared across all backends)
    # ==================================================================
    spec = SSMSpec(
        n_latent=n_latent,
        n_manifest=n_manifest,
        drift="free",
        diffusion="diag",
        cint="free",
        lambda_mat="free",
        manifest_means="free",
        manifest_var="diag",
        t0_means="free",
        t0_var="diag",
        latent_names=names,
    )
    priors = SSMPriors(
        drift_diag={"mu": -0.5, "sigma": 0.5},
        drift_offdiag={"mu": 0.0, "sigma": 0.5},
        diffusion_diag={"sigma": 0.5},
        cint={"mu": 0.0, "sigma": 1.0},
        lambda_free={"mu": 0.5, "sigma": 0.5},
        manifest_means={"mu": 0.0, "sigma": 2.0},
        manifest_var_diag={"sigma": 0.5},
    )

    report_args = {
        "true_drift": true_drift, "true_diff_diag": true_diff_diag,
        "true_cint": true_cint, "true_lambda": true_lambda,
        "true_manifest_means": true_manifest_means,
        "names": names, "n_latent": n_latent, "n_manifest": n_manifest,
    }
    summary = {}

    # ==================================================================
    # 4a. SVI (PF — tolerates noisy gradients from differentiable PF)
    # ==================================================================
    header("FIT: SVI")
    model = SSMModel(spec, priors=priors, n_particles=SVI_PARTICLES, pf_seed=42)

    t0 = time.perf_counter()
    result_svi = fit(
        model, observations=obs, times=times,
        method="svi",
        num_steps=SVI_STEPS, num_samples=SVI_SAMPLES,
        learning_rate=0.005, seed=0,
    )
    elapsed = time.perf_counter() - t0
    losses = result_svi.diagnostics["losses"]
    n = len(losses)
    ckpts = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    print(f"Done in {elapsed:.1f}s")
    print("ELBO: ", "  ".join(f"{i}={float(losses[i]):.0f}" for i in ckpts))
    print()

    rmse, corr = report_recovery("SVI", result_svi.get_samples(), **report_args)
    summary["SVI"] = (elapsed, rmse, corr)

    # ==================================================================
    # 4b. NUTS (Kalman — exact likelihood, smooth gradients)
    # ==================================================================
    header("FIT: NUTS")
    model = SSMModel(spec, priors=priors, likelihood="kalman")

    t0 = time.perf_counter()
    result_nuts = fit(
        model, observations=obs, times=times,
        method="nuts",
        num_warmup=NUTS_WARMUP, num_samples=NUTS_SAMPLES,
        num_chains=1, seed=0,
    )
    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s ({elapsed / (NUTS_WARMUP + NUTS_SAMPLES):.2f}s/step)")
    print()

    rmse, corr = report_recovery("NUTS", result_nuts.get_samples(), **report_args)
    summary["NUTS"] = (elapsed, rmse, corr)

    # ==================================================================
    # 4c. PMMH (still uses PF — gradient-free sampler)
    # ==================================================================
    header("FIT: PMMH")
    model = SSMModel(spec, priors=priors, n_particles=PMMH_PARTICLES, pf_seed=42)

    t0 = time.perf_counter()
    result_pmmh = fit(
        model, observations=obs, times=times,
        method="pmmh",
        num_warmup=PMMH_WARMUP, num_samples=PMMH_SAMPLES,
        seed=0,
    )
    elapsed = time.perf_counter() - t0
    diag = result_pmmh.diagnostics
    print(f"Done in {elapsed:.1f}s ({elapsed / (PMMH_WARMUP + PMMH_SAMPLES):.2f}s/step)")
    print(f"Accept rate: {diag.get('acceptance_rate', 0):.3f}")
    print(f"Final proposal scale: {diag.get('final_proposal_scale', 0):.4f}")
    print()

    rmse, corr = report_recovery("PMMH", result_pmmh.get_samples(), **report_args)
    summary["PMMH"] = (elapsed, rmse, corr)

    # ==================================================================
    # 5. Comparison
    # ==================================================================
    header("COMPARISON")
    print(f"{'Method':<10s}  {'Time(s)':>8s}  {'RMSE':>8s}  {'Corr':>8s}")
    print("-" * 40)
    for method, (t, r, c) in summary.items():
        print(f"{method:<10s}  {t:>8.1f}  {r:>8.4f}  {c:>8.4f}")


# ---------------------------------------------------------------------------
# Entrypoints
# ---------------------------------------------------------------------------

if HAS_MODAL:

    @app.function(gpu=GPU, timeout=3600)
    def recovery_remote():
        run(local=False)

    @app.local_entrypoint()
    def main():
        recovery_remote.remote()


if __name__ == "__main__":
    # Direct invocation: local CPU smoke test
    run(local=True)
