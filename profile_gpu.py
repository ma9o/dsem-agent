"""Profile PF+NUTS on Modal GPU to diagnose performance and convergence.

Usage:
    modal run profile_gpu.py
"""

from pathlib import Path

import modal

ROOT = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("uv")
    .uv_sync(uv_project_dir=str(ROOT), groups=["dev"], frozen=True)
    .uv_pip_install("jax[cuda12]", gpu="A100")
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir(ROOT / "src" / "dsem_agent", remote_path="/root/src/dsem_agent")
)

app = modal.App("dsem-gpu-profile", image=image)


@app.function(gpu="A100", timeout=300)
def profile():
    import subprocess
    import time

    import jax
    import jax.numpy as jnp
    import jax.random as random
    import numpyro

    from dsem_agent.models.likelihoods.base import CTParams, InitialStateParams, MeasurementParams
    from dsem_agent.models.likelihoods.particle import ParticleLikelihood
    from dsem_agent.models.ssm import SSMModel, SSMSpec

    print("=" * 60)
    print("ENVIRONMENT")
    print("=" * 60)
    print(f"JAX {jax.__version__}  backend={jax.default_backend()}  devices={jax.devices()}")
    subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=False
    )
    print()

    # ── Simulate data (FIXED: diag(exp(d*dt)) not exp(diag(d)*dt)) ──────
    true_drift_diag = jnp.array([-0.6, -0.9])
    key = random.PRNGKey(42)
    T, n_latent, dt = 60, 2, 0.5
    discrete_coef = jnp.diag(jnp.exp(true_drift_diag * dt))

    states = [jnp.zeros(n_latent)]
    for _ in range(T - 1):
        key, subkey = random.split(key)
        states.append(discrete_coef @ states[-1] + random.normal(subkey, (n_latent,)) * 0.3)

    key, subkey = random.split(key)
    observations = jnp.stack(states) + random.normal(subkey, (T, n_latent)) * 0.1
    times = jnp.arange(T, dtype=float) * dt
    time_intervals = jnp.diff(times, prepend=times[0])
    time_intervals = time_intervals.at[0].set(1e-6)

    print(f"obs range: [{float(jnp.min(observations)):.2f}, {float(jnp.max(observations)):.2f}]")
    print()

    # ── Gradient check ──────────────────────────────────────────────────
    print("=" * 60)
    print("GRADIENT CHECK (true params)")
    print("=" * 60)

    meas_params = MeasurementParams(
        lambda_mat=jnp.eye(2), manifest_means=jnp.zeros(2), manifest_cov=jnp.eye(2) * 0.01
    )
    init = InitialStateParams(mean=jnp.zeros(2), cov=jnp.eye(2))

    backend = ParticleLikelihood(n_latent=2, n_manifest=2, n_particles=200)

    def ll_fn(drift_diag):
        ct = CTParams(drift=jnp.diag(drift_diag), diffusion_cov=jnp.eye(2) * 0.09, cint=None)
        return backend.compute_log_likelihood(ct, meas_params, init, observations, time_intervals)

    # Compile
    t0 = time.perf_counter()
    ll_val = ll_fn(true_drift_diag)
    jax.block_until_ready(ll_val)
    print(f"LL (compile+run):  {float(ll_val):.2f}  ({time.perf_counter()-t0:.2f}s)")

    t0 = time.perf_counter()
    grad_val = jax.grad(ll_fn)(true_drift_diag)
    jax.block_until_ready(grad_val)
    print(f"Grad (compile+run): {grad_val}  ({time.perf_counter()-t0:.2f}s)")
    print(f"Grad finite: {bool(jnp.all(jnp.isfinite(grad_val)))}")

    # Cached
    t0 = time.perf_counter()
    for _ in range(10):
        jax.block_until_ready(ll_fn(true_drift_diag))
    t_cached = (time.perf_counter() - t0) / 10
    print(f"LL cached avg:  {t_cached*1000:.1f}ms")

    t0 = time.perf_counter()
    for _ in range(10):
        jax.block_until_ready(jax.grad(ll_fn)(true_drift_diag))
    t_grad_cached = (time.perf_counter() - t0) / 10
    print(f"Grad cached avg: {t_grad_cached*1000:.1f}ms")
    print()

    # ── MCMC run ────────────────────────────────────────────────────────
    print("=" * 60)
    print("MCMC RUN (50 warmup + 50 samples)")
    print("=" * 60)

    spec = SSMSpec(n_latent=2, n_manifest=2, lambda_mat=jnp.eye(2), diffusion="diag")
    model = SSMModel(spec, n_particles=200)
    numpyro.set_host_device_count(1)

    t0 = time.perf_counter()
    mcmc = model.fit(
        observations=observations,
        times=times,
        num_warmup=50,
        num_samples=50,
        num_chains=1,
    )
    t_mcmc = time.perf_counter() - t0
    print(f"\nTotal: {t_mcmc:.1f}s  ({t_mcmc/100:.2f}s/step)")

    samples = mcmc.get_samples()
    extra = mcmc.get_extra_fields()

    print(f"Divergences: {int(jnp.sum(extra.get('diverging', jnp.array(0))))}")
    if "accept_prob" in extra:
        print(f"Accept prob: {float(jnp.mean(extra['accept_prob'])):.3f}")
    if "num_steps" in extra:
        print(f"Avg tree steps: {float(jnp.mean(extra['num_steps'])):.1f}")

    print("\nPosterior vs True:")
    drift_samples = samples.get("drift_diag_pop")
    if drift_samples is not None:
        print(f"  drift_diag_pop: mean={jnp.mean(drift_samples, axis=0)} (true: {true_drift_diag})")
    for name in ["diffusion_diag_pop", "manifest_var_diag"]:
        if name in samples:
            print(f"  {name}: mean={jnp.mean(samples[name], axis=0)}")

    print()
    subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"],
        check=False,
    )


@app.local_entrypoint()
def main():
    profile.remote()
