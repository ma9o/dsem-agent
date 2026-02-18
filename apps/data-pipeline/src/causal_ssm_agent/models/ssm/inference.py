"""Inference backends for SSM models.

Separates inference from model definition. SSMModel defines the probabilistic
model; this module provides fit() to run inference with different backends:

- SVI (default): Fast approximate posterior via ELBO optimization.
  Tolerates PF gradient noise because SGD is designed for noisy gradients.
- NUTS: HMC-based sampling. Works well with Kalman likelihood but struggles
  with PF resampling discontinuities.
- NUTS-DA: Data augmentation MCMC — jointly samples parameters and latent
  states with NUTS. No filter needed. Non-centered parameterization (default).
- Hess-MC²: SMC with gradient-based change-of-variables L-kernels.
- PGAS: Particle Gibbs with ancestor sampling + gradient-informed proposals.
- Tempered SMC: Adaptive tempering with preconditioned HMC/MALA mutations.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import jax.numpy as jnp
import jax.random as random
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO, init_to_median
from numpyro.infer.autoguide import AutoDelta, AutoMultivariateNormal, AutoNormal
from numpyro.optim import ClippedAdam

if TYPE_CHECKING:
    from causal_ssm_agent.models.ssm.model import SSMModel


@dataclass
class InferenceResult:
    """Container for inference results across all backends.

    Provides a uniform interface regardless of which backend was used.
    """

    _samples: dict[str, jnp.ndarray]  # name -> (n_draws, *shape)
    method: Literal[
        "nuts",
        "nuts_da",
        "svi",
        "hessmc2",
        "pgas",
        "tempered_smc",
        "laplace_em",
        "structured_vi",
        "dpf",
    ]
    diagnostics: dict = field(default_factory=dict)

    def get_samples(self) -> dict[str, jnp.ndarray]:
        """Return posterior samples dict."""
        return self._samples

    def get_mcmc_diagnostics(self) -> dict[str, Any] | None:
        """Extract JSON-serializable MCMC diagnostics.

        Returns per-parameter R-hat, ESS, and sampler-level divergence/tree stats.
        Returns None for non-MCMC methods (SVI, etc.).
        """
        if self.method in ("svi", "structured_vi", "laplace_em"):
            return None

        mcmc = self.diagnostics.get("mcmc")
        if mcmc is None:
            return None

        from numpyro.diagnostics import summary as numpyro_summary

        result: dict[str, Any] = {}

        # Per-parameter convergence diagnostics via numpyro.diagnostics.summary
        try:
            chain_samples = mcmc.get_samples(group_by_chain=True)
            summ = numpyro_summary(chain_samples)
            per_param: list[dict[str, Any]] = []
            for name, stats in summ.items():
                entry: dict[str, Any] = {"parameter": name}
                if "r_hat" in stats:
                    val = stats["r_hat"]
                    entry["r_hat"] = float(val) if val.ndim == 0 else [float(v) for v in val.flat]
                if "n_eff" in stats:
                    val = stats["n_eff"]
                    entry["ess_bulk"] = float(val) if val.ndim == 0 else [float(v) for v in val.flat]
                per_param.append(entry)
            result["per_parameter"] = per_param
        except Exception:
            result["per_parameter"] = []

        # Sampler-level diagnostics from extra fields
        try:
            extra = mcmc.get_extra_fields()
            if "diverging" in extra:
                div = extra["diverging"]
                result["num_divergences"] = int(jnp.sum(div))
                result["divergence_rate"] = float(jnp.mean(div))
            if "num_steps" in extra:
                steps = extra["num_steps"]
                result["tree_depth_mean"] = float(jnp.mean(steps))
                result["tree_depth_max"] = int(jnp.max(steps))
            if "accept_prob" in extra:
                ap = extra["accept_prob"]
                result["accept_prob_mean"] = float(jnp.mean(ap))
        except Exception:
            pass

        result["num_chains"] = int(mcmc.num_chains) if hasattr(mcmc, "num_chains") else None
        result["num_samples"] = int(mcmc._num_samples) if hasattr(mcmc, "_num_samples") else None

        return result

    def get_svi_diagnostics(self) -> dict[str, Any] | None:
        """Extract JSON-serializable SVI diagnostics (ELBO loss curve).

        Returns None for non-SVI methods.
        """
        if self.method != "svi":
            return None

        losses = self.diagnostics.get("losses")
        if losses is None:
            return None

        loss_list = [float(v) for v in losses]
        # Thin to at most 500 points for the frontend
        if len(loss_list) > 500:
            step = len(loss_list) / 500
            loss_list = [loss_list[int(i * step)] for i in range(500)]

        return {"elbo_losses": loss_list}

    def print_summary(self) -> None:
        """Print summary statistics for posterior samples."""
        print(f"\nInference method: {self.method}")
        print(f"{'Parameter':<30} {'Mean':>10} {'Std':>10} {'5%':>10} {'95%':>10}")
        print("-" * 72)
        for name, values in self._samples.items():
            if values.ndim == 1:
                mean = float(jnp.mean(values))
                std = float(jnp.std(values))
                q5 = float(jnp.percentile(values, 5))
                q95 = float(jnp.percentile(values, 95))
                print(f"{name:<30} {mean:>10.4f} {std:>10.4f} {q5:>10.4f} {q95:>10.4f}")
            elif values.ndim >= 2:
                # Flatten parameter dimensions for summary
                flat = values.reshape(values.shape[0], -1)
                for i in range(flat.shape[1]):
                    label = f"{name}[{i}]"
                    mean = float(jnp.mean(flat[:, i]))
                    std = float(jnp.std(flat[:, i]))
                    q5 = float(jnp.percentile(flat[:, i], 5))
                    q95 = float(jnp.percentile(flat[:, i], 95))
                    print(f"{label:<30} {mean:>10.4f} {std:>10.4f} {q5:>10.4f} {q95:>10.4f}")


def fit(
    model: SSMModel,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    method: Literal[
        "svi",
        "nuts",
        "nuts_da",
        "hessmc2",
        "pgas",
        "tempered_smc",
        "laplace_em",
        "structured_vi",
        "dpf",
    ] = "svi",
    **kwargs: Any,
) -> InferenceResult:
    """Fit an SSM using the specified inference method.

    Args:
        model: SSMModel instance defining the probabilistic model
        observations: (N, n_manifest) observed data
        times: (N,) observation times
        method: Inference method - "svi" (default), "nuts", "hessmc2", "pgas", "tempered_smc"
        **kwargs: Method-specific arguments

    Returns:
        InferenceResult with posterior samples and diagnostics
    """
    if method == "nuts":
        return _fit_nuts(model, observations, times, **kwargs)
    elif method == "nuts_da":
        from causal_ssm_agent.models.ssm.nuts_da import fit_nuts_da

        return fit_nuts_da(model, observations, times, **kwargs)
    elif method == "svi":
        return _fit_svi(model, observations, times, **kwargs)
    elif method == "hessmc2":
        from causal_ssm_agent.models.ssm.hessmc2 import fit_hessmc2

        return fit_hessmc2(model, observations, times, **kwargs)
    elif method == "pgas":
        from causal_ssm_agent.models.ssm.pgas import fit_pgas

        return fit_pgas(model, observations, times, **kwargs)
    elif method == "tempered_smc":
        from causal_ssm_agent.models.ssm.tempered_smc import fit_tempered_smc

        return fit_tempered_smc(model, observations, times, **kwargs)
    elif method == "laplace_em":
        from causal_ssm_agent.models.ssm.laplace_em import fit_laplace_em

        return fit_laplace_em(model, observations, times, **kwargs)
    elif method == "structured_vi":
        from causal_ssm_agent.models.ssm.structured_vi import fit_structured_vi

        return fit_structured_vi(model, observations, times, **kwargs)
    elif method == "dpf":
        from causal_ssm_agent.models.ssm.dpf import fit_dpf

        return fit_dpf(model, observations, times, **kwargs)
    else:
        raise ValueError(
            f"Unknown inference method: {method!r}. "
            "Use 'svi', 'nuts', 'nuts_da', 'hessmc2', 'pgas', 'tempered_smc', "
            "'laplace_em', 'structured_vi', or 'dpf'."
        )


def prior_predictive(
    model: SSMModel,
    times: jnp.ndarray,
    num_samples: int = 100,
    seed: int = 0,
) -> dict[str, jnp.ndarray]:
    """Sample from the prior predictive distribution.

    Uses handlers.block to skip the PF likelihood computation,
    which is unnecessary and expensive for prior sampling.

    Args:
        model: SSMModel instance
        times: (T,) time points
        num_samples: Number of prior samples
        seed: Random seed

    Returns:
        Dict of prior predictive samples
    """
    rng_key = random.PRNGKey(seed)
    model_fn = functools.partial(model.model, likelihood_backend=model.make_likelihood_backend())
    blocked_model = handlers.block(model_fn, hide=["log_likelihood"])
    predictive = Predictive(blocked_model, num_samples=num_samples)
    dummy_obs = jnp.zeros((len(times), model.spec.n_manifest))
    return predictive(rng_key, dummy_obs, times)


def _fit_nuts(
    model: SSMModel,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 4,
    seed: int = 0,
    dense_mass: bool = False,
    target_accept_prob: float = 0.85,
    max_tree_depth: int = 8,
    **kwargs: Any,
) -> InferenceResult:
    """Fit using NUTS (HMC).

    Args:
        model: SSMModel instance
        observations: (N, n_manifest) observed data
        times: (N,) observation times
        num_warmup: Number of warmup samples
        num_samples: Number of posterior samples
        num_chains: Number of MCMC chains
        seed: Random seed
        dense_mass: Use dense mass matrix
        target_accept_prob: Target acceptance probability
        max_tree_depth: Max tree depth
        **kwargs: Additional MCMC arguments

    Returns:
        InferenceResult with NUTS samples
    """
    model_fn = functools.partial(model.model, likelihood_backend=model.make_likelihood_backend())
    kernel = NUTS(
        model_fn,
        init_strategy=init_to_median(num_samples=15),
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
        dense_mass=dense_mass,
        regularize_mass_matrix=True,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        **kwargs,
    )

    rng_key = random.PRNGKey(seed)
    mcmc.run(rng_key, observations, times, extra_fields=("diverging", "num_steps", "accept_prob"))

    return InferenceResult(
        _samples=mcmc.get_samples(),
        method="nuts",
        diagnostics={"mcmc": mcmc},
    )


def _fit_svi(
    model: SSMModel,
    observations: jnp.ndarray,
    times: jnp.ndarray,
    guide_type: str = "mvn",
    num_steps: int = 5000,
    num_samples: int = 1000,
    learning_rate: float = 0.01,
    seed: int = 0,
    **kwargs: Any,  # noqa: ARG001
) -> InferenceResult:
    """Fit using Stochastic Variational Inference.

    Uses AutoGuide to learn an approximate posterior. numpyro.factor() sites
    are handled automatically - the guide only models latent sample sites.

    Args:
        model: SSMModel instance
        observations: (N, n_manifest) observed data
        times: (N,) observation times
        guide_type: Guide family - "normal", "mvn", or "delta"
        num_steps: Number of SVI optimization steps
        num_samples: Number of posterior samples to draw after fitting
        learning_rate: Adam learning rate
        seed: Random seed
        **kwargs: Ignored

    Returns:
        InferenceResult with approximate posterior samples
    """
    model_fn = functools.partial(model.model, likelihood_backend=model.make_likelihood_backend())
    guide_cls = {
        "normal": AutoNormal,
        "mvn": AutoMultivariateNormal,
        "delta": AutoDelta,
    }[guide_type]
    guide = guide_cls(model_fn)

    optimizer = ClippedAdam(step_size=learning_rate)
    svi = SVI(model_fn, guide, optimizer, Trace_ELBO())

    rng_key = random.PRNGKey(seed)
    svi_result = svi.run(rng_key, num_steps, observations, times)

    # Draw posterior samples from the fitted guide
    sample_key = random.PRNGKey(seed + 1)
    predictive = Predictive(
        model_fn,
        guide=guide,
        params=svi_result.params,
        num_samples=num_samples,
    )
    raw_samples = predictive(sample_key, observations, times)

    # Filter out the log_likelihood factor site (observed)
    samples = {name: values for name, values in raw_samples.items() if name != "log_likelihood"}

    return InferenceResult(
        _samples=samples,
        method="svi",
        diagnostics={"losses": svi_result.losses, "params": svi_result.params},
    )


def _eval_model(
    model_fn,
    params_dict: dict[str, jnp.ndarray],
    observations: jnp.ndarray,
    times: jnp.ndarray,
) -> tuple[float, float]:
    """Evaluate model with substituted params. Returns (log_likelihood, log_prior).

    Uses numpyro.handlers to substitute parameter values and trace the model,
    computing log_prior + log_likelihood without any code duplication.

    Args:
        model_fn: NumPyro model function
        params_dict: Parameter values to substitute
        observations: Observed data
        times: Time points

    Returns:
        Tuple of (log_likelihood, log_prior)
    """
    with handlers.seed(rng_seed=0), handlers.substitute(data=params_dict):
        trace = handlers.trace(model_fn).get_trace(observations, times)

    log_lik = 0.0
    log_prior = 0.0
    for name, site in trace.items():
        if site["type"] == "sample":
            if name == "log_likelihood":
                # Factor site: fn is Unit with log_factor attribute
                log_lik = site["fn"].log_factor
            elif not site.get("is_observed", False):
                log_prior = log_prior + jnp.sum(site["fn"].log_prob(site["value"]))

    return log_lik, log_prior
