"""Hess-MC² inference: SMC with gradient-based change-of-variables L-kernels.

Implements the SMC sampler from the Hess-MC² paper with momentum-based
proposals and change-of-variables (CoV) L-kernels:
- Random Walk (RW) proposals
- First-Order Langevin (MALA) proposals using gradient of log-posterior
- Second-Order (Hessian) proposals using diagonal curvature information

All proposals are accepted — quality is controlled through importance weight
correction via the CoV L-kernel, not MH accept/reject. Gradients and Hessians
target the log-posterior (paper Eq 9, 11).

Reference: Murphy et al., "Hess-MC²: Sequential Monte Carlo Squared using
Hessian Information and Second Order Proposals", 2025.
"""

from __future__ import annotations

from typing import Any, Literal

import jax
import jax.numpy as jnp
import jax.random as random
import numpyro.distributions as dist
from blackjax.smc.resampling import systematic as _systematic_resample
from jax.flatten_util import ravel_pytree
from numpyro import handlers

from dsem_agent.models.ssm.inference import InferenceResult, _eval_model

# ---------------------------------------------------------------------------
# Model tracing and differentiable evaluators
# ---------------------------------------------------------------------------


def _discover_sites(model, observations, times, subject_ids, rng_key):
    """Trace model once to discover sample sites (names, shapes, transforms)."""
    with handlers.seed(rng_seed=int(rng_key[0])):
        trace = handlers.trace(model.model).get_trace(observations, times, subject_ids)

    site_info = {}
    for name, site in trace.items():
        if (
            site["type"] == "sample"
            and not site.get("is_observed", False)
            and name != "log_likelihood"
        ):
            d = site["fn"]
            site_info[name] = {
                "shape": site["value"].shape,
                "distribution": d,
                "transform": dist.transforms.biject_to(d.support),
                "value": site["value"],
            }
    return site_info


def _build_eval_fns(model, observations, times, subject_ids, site_info, unravel_fn):
    """Build differentiable functions for log-likelihood and log-prior.

    Returns:
        log_lik_fn(z, pf_key) -> scalar log p(y|theta)
        log_prior_unc_fn(z) -> scalar log p_unc(z) = log p(T(z)) + log|J|
    """
    transforms = {name: info["transform"] for name, info in site_info.items()}
    distributions = {name: info["distribution"] for name, info in site_info.items()}

    def _constrain(z):
        unc = unravel_fn(z)
        return {name: transforms[name](unc[name]) for name in unc}, unc

    def log_lik_fn(z, pf_key):
        """Log-likelihood p(y|theta) via PF or Kalman."""
        con, _ = _constrain(z)
        model.pf_key = pf_key
        log_lik, _ = _eval_model(model.model, con, observations, times, subject_ids)
        return log_lik

    def log_prior_unc_fn(z):
        """Log-prior in unconstrained space: log p(T(z)) + log|J(z)|."""
        con, unc = _constrain(z)
        lp = sum(jnp.sum(distributions[name].log_prob(con[name])) for name in unc)
        lj = sum(
            jnp.sum(transforms[name].log_abs_det_jacobian(unc[name], con[name])) for name in unc
        )
        return lp + lj

    return log_lik_fn, log_prior_unc_fn


# ---------------------------------------------------------------------------
# Diagonal Hessian via forward-over-reverse
# ---------------------------------------------------------------------------


def _diag_hessian(f, x, *args):
    """Compute diagonal of Hessian of f at x via forward-over-reverse."""
    grad_fn = jax.grad(f)

    def hvp_diag_element(basis_vec):
        _, jvp_val = jax.jvp(grad_fn, (x, *args), (basis_vec, *[jnp.zeros_like(a) for a in args]))
        return jnp.dot(basis_vec, jvp_val)

    eye = jnp.eye(x.shape[0])
    return jax.vmap(hvp_diag_element)(eye)


# ---------------------------------------------------------------------------
# CoV L-kernel density
# ---------------------------------------------------------------------------


def _log_cov_density(v, m_diag, step_size, D):
    """Log proposal density in v-space with Jacobian correction.

    Computes log N(v; 0, diag(m)) + log|det(J)|^{-1} where J = eps * M^{-1},
    dropping the -D/2*log(2*pi) constant which cancels in L - q.

    For FO/RW pass m_diag = ones(D): Jacobian is step_size^D which
    cancels between forward and reverse if step sizes match.
    """
    return 0.5 * jnp.sum(jnp.log(m_diag) - v**2 / m_diag) - D * jnp.log(step_size)


# ---------------------------------------------------------------------------
# Pure-JAX proposal functions (vmappable)
# ---------------------------------------------------------------------------


def _propose_rw(x, grad, hess_diag, z, eps, eps_fb):  # noqa: ARG001
    """RW proposal (Eq 28): x_new = x + eps * z."""
    v = z
    v_half = v
    x_new = x + eps * v
    return x_new, v, v_half, jnp.ones_like(x), eps


def _propose_fo(x, grad, hess_diag, z, eps, eps_fb):  # noqa: ARG001
    """First-order / MALA proposal with leapfrog structure (Eq 30-33)."""
    v = z
    v_half = 0.5 * eps * grad + v
    x_new = x + eps * v_half
    return x_new, v, v_half, jnp.ones_like(x), eps


def _propose_so(x, grad, hess_diag, z, eps, eps_fb):
    """Second-order proposal with FO fallback when not PSD (Eq 39-41)."""
    neg_hd = -hess_diag
    is_psd = jnp.all(neg_hd > 1e-8)
    m = jnp.maximum(neg_hd, 1e-8)
    ones = jnp.ones_like(x)

    # SO path
    v_so = z * jnp.sqrt(m)
    v_half_so = 0.5 * eps * grad + v_so
    x_new_so = x + eps * (v_half_so / m)

    # FO fallback path
    v_fo = z
    v_half_fo = 0.5 * eps_fb * grad + v_fo
    x_new_fo = x + eps_fb * v_half_fo

    return (
        jnp.where(is_psd, x_new_so, x_new_fo),
        jnp.where(is_psd, v_so, v_fo),
        jnp.where(is_psd, v_half_so, v_half_fo),
        jnp.where(is_psd, m, ones),
        jnp.where(is_psd, eps, eps_fb),
    )


# ---------------------------------------------------------------------------
# Pure-JAX reverse momentum functions (vmappable)
# ---------------------------------------------------------------------------


def _reverse_rw(v_half, grad_new, hess_diag_new, eps, eps_fb):  # noqa: ARG001
    """RW reverse: symmetric, v_new = v_half (Eq 29)."""
    return v_half, jnp.ones_like(v_half), eps


def _reverse_fo(v_half, grad_new, hess_diag_new, eps, eps_fb):  # noqa: ARG001
    """FO reverse momentum kick (Eq 34)."""
    v_new = 0.5 * eps * grad_new + v_half
    return v_new, jnp.ones_like(v_half), eps


def _reverse_so(v_half, grad_new, hess_diag_new, eps, eps_fb):
    """SO reverse with FO fallback (Eq 42, 44)."""
    neg_hd = -hess_diag_new
    is_psd = jnp.all(neg_hd > 1e-8)
    m = jnp.maximum(neg_hd, 1e-8)
    ones = jnp.ones_like(v_half)

    v_new_so = 0.5 * eps * grad_new + v_half
    v_new_fo = 0.5 * eps_fb * grad_new + v_half

    return (
        jnp.where(is_psd, v_new_so, v_new_fo),
        jnp.where(is_psd, m, ones),
        jnp.where(is_psd, eps, eps_fb),
    )


# ---------------------------------------------------------------------------
# Pure-JAX weight update (vmappable)
# ---------------------------------------------------------------------------


def _compute_weight(
    logw_old, log_post_new, log_post_old, v, v_new, fwd_m, rev_m, fwd_ss, rev_ss, D
):
    """Importance weight update with CoV L-kernel correction (Eq 25)."""
    log_L = _log_cov_density(-v_new, rev_m, rev_ss, D)
    log_q = _log_cov_density(v, fwd_m, fwd_ss, D)
    lw = logw_old + log_post_new - log_post_old + log_L - log_q
    lw = jnp.where(jnp.isfinite(log_post_new), lw, -jnp.inf)
    lw = jnp.where(jnp.isfinite(logw_old), lw, -jnp.inf)
    return lw


# ---------------------------------------------------------------------------
# Hess-MC² main sampler
# ---------------------------------------------------------------------------


def fit_hessmc2(
    model,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    subject_ids: jnp.ndarray | None = None,
    n_smc_particles: int = 64,
    n_iterations: int = 20,
    proposal: Literal["rw", "mala", "hessian"] = "mala",
    step_size: float = 0.1,
    fallback_step_size: float = 0.01,
    seed: int = 0,
    **kwargs: Any,  # noqa: ARG001
) -> InferenceResult:
    """Fit SSM parameters via Hess-MC² (SMC with CoV L-kernels).

    Uses importance-weighted SMC where all proposals are accepted and
    quality is controlled through change-of-variables L-kernel weight
    correction. Gradients/Hessians target the log-posterior (Eq 9, 11).

    Args:
        model: SSMModel instance
        observations: (T, n_manifest) observed data
        times: (T,) observation times
        subject_ids: optional subject indices for hierarchical models
        n_smc_particles: N -- number of parameter particles
        n_iterations: K -- number of SMC iterations
        proposal: "rw", "mala", or "hessian"
        step_size: epsilon -- proposal step size
        fallback_step_size: step size when Hessian is not PSD (SO only)
        seed: random seed

    Returns:
        InferenceResult with posterior samples
    """
    rng_key = random.PRNGKey(seed)
    N = n_smc_particles
    K = n_iterations

    # 1. Discover model sites
    rng_key, trace_key = random.split(rng_key)
    site_info = _discover_sites(model, observations, times, subject_ids, trace_key)
    example_unc = {name: info["transform"].inv(info["value"]) for name, info in site_info.items()}
    flat_example, unravel_fn = ravel_pytree(example_unc)
    D = flat_example.shape[0]

    # 2. Build differentiable functions
    log_lik_fn, log_prior_unc_fn = _build_eval_fns(
        model, observations, times, subject_ids, site_info, unravel_fn
    )
    pf_key = model.pf_key

    # Gradient and Hessian target the log-POSTERIOR (paper Eq 9, 11)
    def log_post_fn(z, pf_key):
        return log_lik_fn(z, pf_key) + log_prior_unc_fn(z)

    grad_post_fn = jax.grad(log_post_fn, argnums=0)

    # Select proposal/reverse functions and build vmapped batches
    if proposal == "rw":
        _prop, _rev = _propose_rw, _reverse_rw
    elif proposal == "mala":
        _prop, _rev = _propose_fo, _reverse_fo
    else:
        _prop, _rev = _propose_so, _reverse_so

    propose_batch = jax.jit(jax.vmap(_prop, in_axes=(0, 0, 0, 0, None, None)))
    reverse_batch = jax.jit(jax.vmap(_rev, in_axes=(0, 0, 0, None, None)))
    weight_batch = jax.jit(jax.vmap(_compute_weight, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, None)))

    # 3. Initialize N particles from prior
    particles = jnp.zeros((N, D))
    log_posts = jnp.zeros(N)
    grads = jnp.zeros((N, D))
    hess_diags = jnp.zeros((N, D))

    print(f"Hess-MC²: N={N}, K={K}, D={D}, proposal={proposal}, eps={step_size}")
    print(f"  Initializing {N} particles from prior...")

    for i in range(N):
        rng_key, init_key = random.split(rng_key)
        with handlers.seed(rng_seed=int(init_key[0])):
            trace = handlers.trace(model.model).get_trace(observations, times, subject_ids)
        init_unc = {}
        for name, info in site_info.items():
            init_unc[name] = info["transform"].inv(trace[name]["value"])
        particles = particles.at[i].set(ravel_pytree(init_unc)[0])

        ll = log_lik_fn(particles[i], pf_key)
        lp = log_prior_unc_fn(particles[i])
        log_posts = log_posts.at[i].set(
            jnp.where(jnp.isfinite(ll) & jnp.isfinite(lp), lp + ll, -1e30)
        )

        if proposal in ("mala", "hessian"):
            g = grad_post_fn(particles[i], pf_key)
            grads = grads.at[i].set(jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0))
        if proposal == "hessian":
            hd = _diag_hessian(log_post_fn, particles[i], pf_key)
            hess_diags = hess_diags.at[i].set(jnp.nan_to_num(hd, nan=0.0, posinf=0.0, neginf=0.0))

    # Initial weights: log w = log [pi(theta)/q(theta)] = log_post - log_prior = log_lik
    init_log_priors = jnp.array([log_prior_unc_fn(particles[i]) for i in range(N)])
    logw = log_posts - init_log_priors

    # Diagnostics and recycling storage (Eq 26, [18])
    ess_history = []
    resample_points = []
    recycled_particles = [particles]
    recycled_logw = [logw]

    # 4. Main SMC loop
    for k in range(K):
        # --- Normalize weights, ESS ---
        log_wn = logw - jax.nn.logsumexp(logw)
        wn = jnp.exp(log_wn)
        ess = float(1.0 / jnp.sum(wn**2))
        ess_history.append(ess)

        # --- Resample if ESS < N/2 (carry grads and hessians) ---
        did_resample = False
        if ess < N / 2:
            resample_points.append(k)
            rng_key, resample_key = random.split(rng_key)
            idx = _systematic_resample(resample_key, jnp.exp(logw - jax.nn.logsumexp(logw)), N)
            particles = particles[idx]
            log_posts = log_posts[idx]
            grads = grads[idx]
            hess_diags = hess_diags[idx]
            logw = jnp.full(N, -jnp.log(float(N)))
            did_resample = True

        # --- Draw noise for all particles at once ---
        rng_key, noise_key = random.split(rng_key)
        z_all = random.normal(noise_key, (N, D))

        # --- Vectorized proposal (pure JAX, no handlers) ---
        particles_new, v_all, v_half_all, fwd_m_all, fwd_ss_all = propose_batch(
            particles, grads, hess_diags, z_all, step_size, fallback_step_size
        )

        # --- Evaluate model at new particles (sequential; handlers not vmappable) ---
        log_posts_new = jnp.zeros(N)
        grads_new = jnp.zeros((N, D))
        hess_diags_new = jnp.zeros((N, D))

        for i in range(N):
            ll = log_lik_fn(particles_new[i], pf_key)
            lp = log_prior_unc_fn(particles_new[i])
            log_posts_new = log_posts_new.at[i].set(
                jnp.where(jnp.isfinite(ll) & jnp.isfinite(lp), lp + ll, -1e30)
            )

            if proposal in ("mala", "hessian"):
                g = grad_post_fn(particles_new[i], pf_key)
                grads_new = grads_new.at[i].set(jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0))
            if proposal == "hessian":
                hd = _diag_hessian(log_post_fn, particles_new[i], pf_key)
                hess_diags_new = hess_diags_new.at[i].set(
                    jnp.nan_to_num(hd, nan=0.0, posinf=0.0, neginf=0.0)
                )

        # --- Vectorized reverse momentum + weight update ---
        v_new_all, rev_m_all, rev_ss_all = reverse_batch(
            v_half_all, grads_new, hess_diags_new, step_size, fallback_step_size
        )

        logw = weight_batch(
            logw,
            log_posts_new,
            log_posts,
            v_all,
            v_new_all,
            fwd_m_all,
            rev_m_all,
            fwd_ss_all,
            rev_ss_all,
            D,
        )

        # Update state
        particles = particles_new
        log_posts = log_posts_new
        grads = grads_new
        hess_diags = hess_diags_new

        # Store for recycling (Eq 26, [18])
        recycled_particles.append(particles)
        recycled_logw.append(logw)

        resamp_tag = " [resampled]" if did_resample else ""
        print(f"  step {k + 1}/{K}  ESS={ess:.1f}/{N}{resamp_tag}")

    # 5. Final resampling from recycled pool
    rng_key, final_key = random.split(rng_key)
    all_particles = jnp.concatenate(recycled_particles, axis=0)  # ((K+1)*N, D)
    all_logw = jnp.concatenate(recycled_logw, axis=0)  # ((K+1)*N,)
    idx = _systematic_resample(final_key, jnp.exp(all_logw - jax.nn.logsumexp(all_logw)), N)
    final_particles = all_particles[idx]

    # Extract final samples in constrained space
    transforms = {name: info["transform"] for name, info in site_info.items()}
    samples = {}
    for name in transforms:
        vals = []
        for i in range(N):
            unc = unravel_fn(final_particles[i])
            vals.append(transforms[name](unc[name]))
        samples[name] = jnp.stack(vals)

    det_samples = _extract_deterministic_sites(
        model,
        observations,
        times,
        subject_ids,
        site_info,
        unravel_fn,
        final_particles,
    )
    samples.update(det_samples)

    return InferenceResult(
        _samples=samples,
        method="hessmc2",
        diagnostics={
            "ess_history": ess_history,
            "resample_points": resample_points,
            "n_smc_particles": N,
            "n_iterations": K,
            "proposal": proposal,
            "step_size": step_size,
        },
    )


def _extract_deterministic_sites(
    model,
    observations,
    times,
    subject_ids,
    site_info,
    unravel_fn,
    particles,
):
    """Run model with each particle to extract deterministic sites."""
    transforms = {name: info["transform"] for name, info in site_info.items()}
    N = particles.shape[0]
    det_samples: dict[str, list] = {}

    for i in range(N):
        unc = unravel_fn(particles[i])
        con = {name: transforms[name](unc[name]) for name in unc}

        with handlers.seed(rng_seed=0), handlers.substitute(data=con):
            trace = handlers.trace(model.model).get_trace(observations, times, subject_ids)

        for name, site in trace.items():
            if site["type"] == "deterministic":
                if name not in det_samples:
                    det_samples[name] = []
                det_samples[name].append(site["value"])

    return {name: jnp.stack(vals) for name, vals in det_samples.items()}
