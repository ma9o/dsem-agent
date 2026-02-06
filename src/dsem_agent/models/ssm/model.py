"""NumPyro State-Space Model.

Hierarchical Bayesian State-Space Model using NumPyro for inference.

Supports:
- Single-subject time series
- Multi-subject panel data with shared parameters
- Hierarchical models with individual variation in parameters
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from jax import lax, vmap
from numpyro.infer import MCMC, NUTS

from dsem_agent.models.likelihoods.base import CTParams, InitialStateParams, MeasurementParams
from dsem_agent.models.strategy_selector import InferenceStrategy, get_likelihood_backend

if TYPE_CHECKING:
    from dsem_agent.models.pmmh import PMMHResult


class NoiseFamily(StrEnum):
    """Supported noise distribution families for state-space models.

    Used to specify process noise (diffusion) and observation noise (manifest)
    distributions for strategy selection.
    """

    GAUSSIAN = "gaussian"  # Standard Gaussian - enables Kalman/UKF
    STUDENT_T = "student_t"  # Heavy-tailed - requires particle filter
    POISSON = "poisson"  # Count data - requires particle filter
    GAMMA = "gamma"  # Positive continuous - requires particle filter


@dataclass
class SSMSpec:
    """Specification for a state-space model.

    Matrix naming convention:
    - DRIFT: n_latent x n_latent continuous-time auto/cross effects
    - DIFFUSION: n_latent x n_latent process noise (Cholesky)
    - CINT: n_latent x 1 continuous intercept
    - LAMBDA: n_manifest x n_latent factor loadings
    - MANIFESTMEANS: n_manifest x 1 manifest intercepts
    - MANIFESTVAR: n_manifest x n_manifest measurement error (Cholesky)
    - T0MEANS: n_latent x 1 initial state means
    - T0VAR: n_latent x n_latent initial state variance (Cholesky)
    """

    n_latent: int
    n_manifest: int

    # Fixed or "free" specification for each matrix
    # If a matrix, use those fixed values; if "free", estimate
    drift: jnp.ndarray | Literal["free"] = "free"
    diffusion: jnp.ndarray | Literal["free", "diag"] = "free"
    cint: jnp.ndarray | Literal["free"] | None = None
    lambda_mat: jnp.ndarray | Literal["free"] = "free"
    manifest_means: jnp.ndarray | Literal["free"] | None = None
    manifest_var: jnp.ndarray | Literal["free", "diag"] = "diag"
    t0_means: jnp.ndarray | Literal["free"] = "free"
    t0_var: jnp.ndarray | Literal["free", "diag"] = "free"

    # Distribution families for strategy selection
    # Used by select_strategy() to choose Kalman vs UKF vs Particle filter
    diffusion_dist: NoiseFamily = NoiseFamily.GAUSSIAN  # Process noise family
    manifest_dist: NoiseFamily = NoiseFamily.GAUSSIAN  # Observation noise family

    # Hierarchical structure
    hierarchical: bool = False
    n_subjects: int = 1

    # Which parameters vary across individuals (only used if hierarchical=True)
    # Options: "drift_diag", "drift_offdiag", "diffusion", "cint", "t0_means"
    indvarying: list[str] = field(default_factory=lambda: ["t0_means"])

    # Parameter names for interpretability
    latent_names: list[str] | None = None
    manifest_names: list[str] | None = None


@dataclass
class SSMPriors:
    """Prior specifications for state-space model parameters.

    Each prior is specified as a dict with distribution parameters.
    """

    # Drift diagonal (auto-effects, typically negative for stability)
    drift_diag: dict = field(
        default_factory=lambda: {"mu": -0.5, "sigma": 1.0}
    )
    # Drift off-diagonal (cross-effects)
    drift_offdiag: dict = field(
        default_factory=lambda: {"mu": 0.0, "sigma": 0.5}
    )

    # Diffusion (log scale for positivity)
    diffusion_diag: dict = field(
        default_factory=lambda: {"sigma": 1.0}
    )
    diffusion_offdiag: dict = field(
        default_factory=lambda: {"mu": 0.0, "sigma": 0.5}
    )

    # Continuous intercept
    cint: dict = field(
        default_factory=lambda: {"mu": 0.0, "sigma": 1.0}
    )

    # Factor loadings
    lambda_free: dict = field(
        default_factory=lambda: {"mu": 0.5, "sigma": 0.5}
    )

    # Manifest means
    manifest_means: dict = field(
        default_factory=lambda: {"mu": 0.0, "sigma": 2.0}
    )

    # Manifest variance (measurement error)
    manifest_var_diag: dict = field(
        default_factory=lambda: {"sigma": 1.0}
    )

    # Initial state
    t0_means: dict = field(
        default_factory=lambda: {"mu": 0.0, "sigma": 2.0}
    )
    t0_var_diag: dict = field(
        default_factory=lambda: {"sigma": 2.0}
    )

    # Hierarchical (population-level SD for random effects)
    pop_sd: dict = field(
        default_factory=lambda: {"sigma": 1.0}
    )


class SSMModel:
    """NumPyro state-space model.

    Implements hierarchical Bayesian state-space model with:
    - Continuous-time dynamics via stochastic differential equations
    - Kalman filter likelihood computation
    - Optional hierarchical structure for multiple subjects with individual variation
    """

    def __init__(
        self,
        spec: SSMSpec,
        priors: SSMPriors | None = None,
    ):
        """Initialize state-space model.

        Args:
            spec: Model specification
            priors: Prior distributions (uses defaults if None)
        """
        self.spec = spec
        self.priors = priors or SSMPriors()
        self._strategy = None  # Cached strategy selection

    def get_inference_strategy(self):
        """Get the inference strategy for this model.

        Returns the appropriate marginalization strategy (Kalman/UKF/Particle)
        based on the model specification's dynamics and distribution families.

        Returns:
            InferenceStrategy enum value

        Example:
            >>> model = SSMModel(SSMSpec(n_latent=2, n_manifest=2))
            >>> model.get_inference_strategy()
            <InferenceStrategy.KALMAN: 'kalman'>
        """
        if self._strategy is None:
            from dsem_agent.models.strategy_selector import select_strategy
            self._strategy = select_strategy(self.spec)
        return self._strategy

    def _sample_drift(
        self, spec: SSMSpec, n_subjects: int = 1, hierarchical: bool = False
    ) -> jnp.ndarray:
        """Sample drift matrix with stability constraints.

        Args:
            spec: Model specification
            n_subjects: Number of subjects (for hierarchical)
            hierarchical: Whether to use hierarchical sampling

        Returns:
            drift: (n_latent, n_latent) or (n_subjects, n_latent, n_latent)
        """
        n = spec.n_latent

        if isinstance(spec.drift, jnp.ndarray):
            if hierarchical and n_subjects > 1:
                return jnp.broadcast_to(spec.drift, (n_subjects, n, n))
            return spec.drift

        # Population-level diagonal (auto-effects)
        drift_diag_pop = numpyro.sample(
            "drift_diag_pop",
            dist.Normal(
                self.priors.drift_diag["mu"], self.priors.drift_diag["sigma"]
            ).expand([n]),
        )

        # Population-level off-diagonal (cross-effects)
        n_offdiag = n * n - n
        if n_offdiag > 0:
            drift_offdiag_pop = numpyro.sample(
                "drift_offdiag_pop",
                dist.Normal(
                    self.priors.drift_offdiag["mu"], self.priors.drift_offdiag["sigma"]
                ).expand([n_offdiag]),
            )
        else:
            drift_offdiag_pop = jnp.array([])

        if hierarchical and n_subjects > 1:
            # Individual variation in drift
            if "drift_diag" in spec.indvarying:
                drift_diag_sd = numpyro.sample(
                    "drift_diag_sd",
                    dist.HalfNormal(self.priors.pop_sd["sigma"]).expand([n]),
                )
                drift_diag_raw = numpyro.sample(
                    "drift_diag_raw",
                    dist.Normal(0, 1).expand([n_subjects, n]),
                )
                drift_diag = drift_diag_pop + drift_diag_sd * drift_diag_raw
            else:
                drift_diag = jnp.broadcast_to(drift_diag_pop, (n_subjects, n))

            if "drift_offdiag" in spec.indvarying and n_offdiag > 0:
                drift_offdiag_sd = numpyro.sample(
                    "drift_offdiag_sd",
                    dist.HalfNormal(self.priors.pop_sd["sigma"]).expand([n_offdiag]),
                )
                drift_offdiag_raw = numpyro.sample(
                    "drift_offdiag_raw",
                    dist.Normal(0, 1).expand([n_subjects, n_offdiag]),
                )
                drift_offdiag = drift_offdiag_pop + drift_offdiag_sd * drift_offdiag_raw
            else:
                drift_offdiag = jnp.broadcast_to(drift_offdiag_pop, (n_subjects, n_offdiag))

            # Assemble drift matrices for each subject
            def assemble_drift(diag, offdiag):
                # Constrain diagonal to be negative for stability
                diag_neg = -jnp.abs(diag)
                drift = jnp.diag(diag_neg)
                offdiag_idx = 0
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            drift = drift.at[i, j].set(offdiag[offdiag_idx])
                            offdiag_idx += 1
                return drift

            drift = vmap(assemble_drift)(drift_diag, drift_offdiag)
        else:
            # Single drift matrix
            drift_diag = -jnp.abs(drift_diag_pop)
            drift = jnp.diag(drift_diag)
            offdiag_idx = 0
            for i in range(n):
                for j in range(n):
                    if i != j:
                        drift = drift.at[i, j].set(drift_offdiag_pop[offdiag_idx])
                        offdiag_idx += 1

        numpyro.deterministic("drift", drift)
        return drift

    def _sample_diffusion(
        self, spec: SSMSpec, n_subjects: int = 1, hierarchical: bool = False
    ) -> jnp.ndarray:
        """Sample diffusion matrix (lower Cholesky)."""
        n = spec.n_latent

        if isinstance(spec.diffusion, jnp.ndarray):
            if hierarchical and n_subjects > 1:
                return jnp.broadcast_to(spec.diffusion, (n_subjects, n, n))
            return spec.diffusion

        # Population-level diagonal
        diff_diag_pop = numpyro.sample(
            "diffusion_diag_pop",
            dist.HalfNormal(self.priors.diffusion_diag["sigma"]).expand([n]),
        )

        if spec.diffusion == "diag":
            if hierarchical and n_subjects > 1 and "diffusion" in spec.indvarying:
                diff_diag_sd = numpyro.sample(
                    "diffusion_diag_sd",
                    dist.HalfNormal(self.priors.pop_sd["sigma"]).expand([n]),
                )
                diff_diag_raw = numpyro.sample(
                    "diffusion_diag_raw",
                    dist.Normal(0, 1).expand([n_subjects, n]),
                )
                diff_diag = jnp.abs(diff_diag_pop + diff_diag_sd * diff_diag_raw)
                diffusion = vmap(jnp.diag)(diff_diag)
            elif hierarchical and n_subjects > 1:
                diffusion = jnp.broadcast_to(jnp.diag(diff_diag_pop), (n_subjects, n, n))
            else:
                diffusion = jnp.diag(diff_diag_pop)
        else:
            # Full lower triangular (non-hierarchical for simplicity)
            n_lower = n * (n - 1) // 2
            if n_lower > 0:
                diff_lower = numpyro.sample(
                    "diffusion_lower",
                    dist.Normal(
                        self.priors.diffusion_offdiag["mu"],
                        self.priors.diffusion_offdiag["sigma"],
                    ).expand([n_lower]),
                )
            else:
                diff_lower = jnp.array([])

            diffusion = jnp.diag(diff_diag_pop)
            lower_idx = 0
            for i in range(n):
                for j in range(i):
                    diffusion = diffusion.at[i, j].set(diff_lower[lower_idx])
                    lower_idx += 1

            if hierarchical and n_subjects > 1:
                diffusion = jnp.broadcast_to(diffusion, (n_subjects, n, n))

        numpyro.deterministic("diffusion", diffusion)
        return diffusion

    def _sample_cint(
        self, spec: SSMSpec, n_subjects: int = 1, hierarchical: bool = False
    ) -> jnp.ndarray | None:
        """Sample continuous intercept."""
        if spec.cint is None:
            return None

        n = spec.n_latent

        if isinstance(spec.cint, jnp.ndarray):
            if hierarchical and n_subjects > 1:
                return jnp.broadcast_to(spec.cint, (n_subjects, n))
            return spec.cint

        cint_pop = numpyro.sample(
            "cint_pop",
            dist.Normal(self.priors.cint["mu"], self.priors.cint["sigma"]).expand([n]),
        )

        if hierarchical and n_subjects > 1 and "cint" in spec.indvarying:
            cint_sd = numpyro.sample(
                "cint_sd",
                dist.HalfNormal(self.priors.pop_sd["sigma"]).expand([n]),
            )
            cint_raw = numpyro.sample(
                "cint_raw",
                dist.Normal(0, 1).expand([n_subjects, n]),
            )
            cint = cint_pop + cint_sd * cint_raw
        elif hierarchical and n_subjects > 1:
            cint = jnp.broadcast_to(cint_pop, (n_subjects, n))
        else:
            cint = cint_pop

        numpyro.deterministic("cint", cint)
        return cint

    def _sample_lambda(self, spec: SSMSpec) -> jnp.ndarray:
        """Sample factor loading matrix (shared across subjects)."""
        if isinstance(spec.lambda_mat, jnp.ndarray):
            return spec.lambda_mat

        n_m, n_l = spec.n_manifest, spec.n_latent

        # Start with identity mapping for first n_latent manifests
        lambda_mat = jnp.eye(n_m, n_l)

        # Sample additional loadings if needed
        if n_m > n_l:
            n_free = (n_m - n_l) * n_l
            free_loadings = numpyro.sample(
                "lambda_free",
                dist.Normal(
                    self.priors.lambda_free["mu"], self.priors.lambda_free["sigma"]
                ).expand([n_free]),
            )

            idx = 0
            for i in range(n_l, n_m):
                for j in range(n_l):
                    lambda_mat = lambda_mat.at[i, j].set(free_loadings[idx])
                    idx += 1

        numpyro.deterministic("lambda", lambda_mat)
        return lambda_mat

    def _sample_manifest_params(self, spec: SSMSpec) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Sample manifest means and variance (shared across subjects)."""
        n_m = spec.n_manifest

        # Means
        if spec.manifest_means is None:
            manifest_means = jnp.zeros(n_m)
        elif isinstance(spec.manifest_means, jnp.ndarray):
            manifest_means = spec.manifest_means
        else:
            manifest_means = numpyro.sample(
                "manifest_means",
                dist.Normal(
                    self.priors.manifest_means["mu"], self.priors.manifest_means["sigma"]
                ).expand([n_m]),
            )

        # Variance (Cholesky)
        if isinstance(spec.manifest_var, jnp.ndarray):
            manifest_chol = spec.manifest_var
        else:
            var_diag = numpyro.sample(
                "manifest_var_diag",
                dist.HalfNormal(self.priors.manifest_var_diag["sigma"]).expand([n_m]),
            )
            manifest_chol = jnp.diag(var_diag)

        numpyro.deterministic("manifest_cov", manifest_chol @ manifest_chol.T)
        return manifest_means, manifest_chol

    def _sample_t0_params(
        self, spec: SSMSpec, n_subjects: int = 1, hierarchical: bool = False
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Sample initial state parameters."""
        n_l = spec.n_latent

        # Means (can vary by subject)
        if isinstance(spec.t0_means, jnp.ndarray):
            t0_means = spec.t0_means
            if hierarchical and n_subjects > 1:
                t0_means = jnp.broadcast_to(t0_means, (n_subjects, n_l))
        else:
            t0_means_pop = numpyro.sample(
                "t0_means_pop",
                dist.Normal(
                    self.priors.t0_means["mu"], self.priors.t0_means["sigma"]
                ).expand([n_l]),
            )

            if hierarchical and n_subjects > 1 and "t0_means" in spec.indvarying:
                t0_means_sd = numpyro.sample(
                    "t0_means_sd",
                    dist.HalfNormal(self.priors.pop_sd["sigma"]).expand([n_l]),
                )
                t0_means_raw = numpyro.sample(
                    "t0_means_raw",
                    dist.Normal(0, 1).expand([n_subjects, n_l]),
                )
                t0_means = t0_means_pop + t0_means_sd * t0_means_raw
            elif hierarchical and n_subjects > 1:
                t0_means = jnp.broadcast_to(t0_means_pop, (n_subjects, n_l))
            else:
                t0_means = t0_means_pop

        # Variance (Cholesky) - shared across subjects
        if isinstance(spec.t0_var, jnp.ndarray):
            t0_chol = spec.t0_var
        else:
            var_diag = numpyro.sample(
                "t0_var_diag",
                dist.HalfNormal(self.priors.t0_var_diag["sigma"]).expand([n_l]),
            )
            t0_chol = jnp.diag(var_diag)

        numpyro.deterministic("t0_means", t0_means)
        numpyro.deterministic("t0_cov", t0_chol @ t0_chol.T)
        return t0_means, t0_chol

    def model(
        self,
        observations: jnp.ndarray,
        times: jnp.ndarray,
        subject_ids: jnp.ndarray | None = None,
    ) -> None:
        """NumPyro model function.

        Args:
            observations: (N, n_manifest) observed data
            times: (N,) observation times
            subject_ids: (N,) subject indices (0-indexed, for hierarchical models)
        """
        spec = self.spec
        hierarchical = spec.hierarchical and subject_ids is not None

        if hierarchical:
            n_subjects = int(jnp.max(subject_ids)) + 1
        else:
            n_subjects = 1

        # Sample parameters
        drift = self._sample_drift(spec, n_subjects, hierarchical)
        diffusion_chol = self._sample_diffusion(spec, n_subjects, hierarchical)
        cint = self._sample_cint(spec, n_subjects, hierarchical)
        lambda_mat = self._sample_lambda(spec)
        manifest_means, manifest_chol = self._sample_manifest_params(spec)
        t0_means, t0_chol = self._sample_t0_params(spec, n_subjects, hierarchical)

        # Convert to covariances
        if hierarchical and n_subjects > 1:
            diffusion_cov = vmap(lambda x: x @ x.T)(diffusion_chol)
        else:
            diffusion_cov = diffusion_chol @ diffusion_chol.T
        manifest_cov = manifest_chol @ manifest_chol.T
        t0_cov = t0_chol @ t0_chol.T

        # Compute log-likelihood via selected backend (Kalman or UKF)
        # PARTICLE strategy uses PMMH (separate inference path, not NumPyro).
        strategy = self.get_inference_strategy()
        if strategy == InferenceStrategy.PARTICLE:
            raise ValueError(
                "PARTICLE strategy requires PMMH inference, not NumPyro model(). "
                "Use fit() which auto-dispatches, or call fit_pmmh() directly."
            )
        backend = get_likelihood_backend(strategy)

        ct_params = CTParams(drift=drift, diffusion_cov=diffusion_cov, cint=cint)
        meas_params = MeasurementParams(
            lambda_mat=lambda_mat,
            manifest_means=manifest_means,
            manifest_cov=manifest_cov,
        )

        if not hierarchical or n_subjects == 1:
            # Single subject
            time_intervals = jnp.diff(times, prepend=times[0])
            time_intervals = time_intervals.at[0].set(1e-6)

            init = InitialStateParams(mean=t0_means, cov=t0_cov)
            ll = backend.compute_log_likelihood(
                ct_params, meas_params, init, observations, time_intervals
            )
        else:
            # Multiple subjects with hierarchical structure
            ll = self._hierarchical_likelihood(
                backend,
                ct_params,
                meas_params,
                observations,
                times,
                subject_ids,
                n_subjects,
                t0_means,
                t0_cov,
            )

        numpyro.factor("log_likelihood", ll)

    def _hierarchical_likelihood(
        self,
        backend,
        ct_params: CTParams,
        meas_params: MeasurementParams,
        observations: jnp.ndarray,
        times: jnp.ndarray,
        subject_ids: jnp.ndarray,
        n_subjects: int,
        t0_means: jnp.ndarray,
        t0_cov: jnp.ndarray,
    ) -> float:
        """Compute log-likelihood for hierarchical model with subject-varying params."""

        def subject_ll(subj_idx):
            # Get subject-specific CT params
            subj_drift = ct_params.drift[subj_idx] if ct_params.drift.ndim == 3 else ct_params.drift
            subj_diff_cov = (
                ct_params.diffusion_cov[subj_idx]
                if ct_params.diffusion_cov.ndim == 3
                else ct_params.diffusion_cov
            )
            subj_cint = (
                ct_params.cint[subj_idx]
                if ct_params.cint is not None and ct_params.cint.ndim == 2
                else ct_params.cint
            )
            subj_t0_mean = t0_means[subj_idx] if t0_means.ndim == 2 else t0_means

            subj_ct = CTParams(drift=subj_drift, diffusion_cov=subj_diff_cov, cint=subj_cint)
            subj_init = InitialStateParams(mean=subj_t0_mean, cov=t0_cov)

            # Build subject-specific observation mask
            mask = subject_ids == subj_idx
            obs_mask = mask[:, None] & ~jnp.isnan(observations)

            # Count valid observations
            n_valid = jnp.sum(mask)

            # Compute subject-specific time intervals without inf padding
            def scan_fn(carry, inputs):
                last_time, seen = carry
                t, is_obs = inputs
                dt = jnp.where(is_obs, jnp.where(seen, t - last_time, 1e-6), 0.0)
                new_last = jnp.where(is_obs, t, last_time)
                new_seen = seen | is_obs
                return (new_last, new_seen), dt

            (_, _), time_intervals = lax.scan(
                scan_fn, (0.0, False), (times, mask)
            )

            # Compute likelihood via backend (masked)
            ll = backend.compute_log_likelihood(
                subj_ct,
                meas_params,
                subj_init,
                observations,
                time_intervals,
                obs_mask=obs_mask,
            )

            # Only count if subject has observations
            return jnp.where(n_valid > 0, ll, 0.0)

        # Sum over all subjects
        subject_indices = jnp.arange(n_subjects)
        return jnp.sum(vmap(subject_ll)(subject_indices))

    def fit(
        self,
        observations: jnp.ndarray,
        times: jnp.ndarray,
        subject_ids: jnp.ndarray | None = None,
        num_warmup: int = 1000,
        num_samples: int = 1000,
        num_chains: int = 4,
        seed: int = 0,
        **mcmc_kwargs: Any,
    ) -> MCMC | PMMHResult:
        """Fit the model using MCMC.

        Automatically dispatches to the appropriate inference engine:
        - KALMAN/UKF: NumPyro NUTS (gradient-based HMC) → returns MCMC
        - PARTICLE: PMMH (particle marginal Metropolis-Hastings) → returns PMMHResult

        For PARTICLE strategy, use fit_pmmh() directly for full control
        over particle filter and proposal settings.

        Args:
            observations: (N, n_manifest) observed data
            times: (N,) observation times
            subject_ids: (N,) subject indices, 0-indexed (for hierarchical)
            num_warmup: Number of warmup samples
            num_samples: Number of posterior samples
            num_chains: Number of MCMC chains
            seed: Random seed
            **mcmc_kwargs: Additional MCMC arguments

        Returns:
            MCMC object (for KALMAN/UKF) or PMMHResult (for PARTICLE)
        """
        strategy = self.get_inference_strategy()

        if strategy == InferenceStrategy.PARTICLE:
            return self.fit_pmmh(
                observations=observations,
                times=times,
                n_samples=num_samples,
                n_warmup=num_warmup,
                seed=seed,
            )

        kernel = NUTS(self.model)
        mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            **mcmc_kwargs,
        )

        rng_key = random.PRNGKey(seed)
        mcmc.run(rng_key, observations, times, subject_ids)

        return mcmc

    def fit_pmmh(
        self,
        observations: jnp.ndarray,
        times: jnp.ndarray,
        n_samples: int = 1000,
        n_warmup: int = 500,
        n_particles: int = 1000,
        proposal_cov: jnp.ndarray | None = None,
        seed: int = 0,
    ) -> PMMHResult:
        """Fit the model using Particle Marginal Metropolis-Hastings (PMMH).

        Builds parameter layout, priors, and unpack function from SSMSpec,
        then delegates to run_pmmh(). Use this for non-Gaussian observation
        models (Poisson, Student-t, Gamma) where Kalman/UKF are not applicable.

        Args:
            observations: (T, n_manifest) observed data
            times: (T,) observation times
            n_samples: Number of posterior samples (after warmup)
            n_warmup: Number of warmup samples (discarded)
            n_particles: Number of particles for bootstrap filter
            proposal_cov: (d, d) proposal covariance (auto-scaled if None)
            seed: Random seed

        Returns:
            PMMHResult with posterior samples and diagnostics

        Raises:
            NotImplementedError: If hierarchical model or full diffusion matrix
        """
        from dsem_agent.models.pmmh import SSMAdapter, run_pmmh

        spec = self.spec
        priors = self.priors

        if spec.hierarchical and spec.n_subjects > 1:
            raise NotImplementedError(
                "PMMH does not yet support hierarchical models. "
                "Use single-subject data or Kalman/UKF for hierarchical."
            )

        if spec.diffusion == "free":
            raise NotImplementedError(
                "PMMH only supports diagonal diffusion (diffusion='diag'). "
                "Full lower-triangular diffusion is not yet supported."
            )

        n = spec.n_latent
        n_m = spec.n_manifest

        # --- Build parameter layout ---
        # Each segment: (name, size, transform_to_model, prior_log_prob)
        segments: list[tuple[str, int]] = []

        # drift_diag: (n,) — negative abs for stability
        if not isinstance(spec.drift, jnp.ndarray):
            segments.append(("drift_diag", n))
            n_offdiag = n * n - n
            if n_offdiag > 0:
                segments.append(("drift_offdiag", n_offdiag))

        # diffusion_diag: (n,) — abs, squared for cov
        if not isinstance(spec.diffusion, jnp.ndarray):
            segments.append(("diffusion_diag", n))

        # cint: (n,) — unconstrained
        if spec.cint == "free":
            segments.append(("cint", n))

        # t0_means: (n,) — unconstrained
        if not isinstance(spec.t0_means, jnp.ndarray) and spec.t0_means == "free":
            segments.append(("t0_means", n))

        # manifest_var_diag: (n_m,) — abs, squared for cov
        if not isinstance(spec.manifest_var, jnp.ndarray):
            segments.append(("manifest_var_diag", n_m))

        # lambda_free: free loadings beyond identity block
        if not isinstance(spec.lambda_mat, jnp.ndarray) and spec.lambda_mat == "free" and n_m > n:
            n_lambda_free = (n_m - n) * n
            segments.append(("lambda_free", n_lambda_free))

        # manifest_means
        if isinstance(spec.manifest_means, str) and spec.manifest_means == "free":
            segments.append(("manifest_means", n_m))

        # Compute offsets
        offsets: dict[str, tuple[int, int]] = {}
        pos = 0
        for name, size in segments:
            offsets[name] = (pos, pos + size)
            pos += size

        # --- Build unpack_fn ---
        def unpack_fn(theta: jnp.ndarray) -> dict:
            params: dict[str, Any] = {}

            # Drift
            if isinstance(spec.drift, jnp.ndarray):
                params["drift"] = spec.drift
            else:
                s, e = offsets["drift_diag"]
                drift_diag = -jnp.abs(theta[s:e])
                drift = jnp.diag(drift_diag)
                if "drift_offdiag" in offsets:
                    s, e = offsets["drift_offdiag"]
                    offdiag = theta[s:e]
                    offdiag_idx = 0
                    for i in range(n):
                        for j in range(n):
                            if i != j:
                                drift = drift.at[i, j].set(offdiag[offdiag_idx])
                                offdiag_idx += 1
                params["drift"] = drift

            # Diffusion covariance
            if isinstance(spec.diffusion, jnp.ndarray):
                params["diffusion_cov"] = spec.diffusion @ spec.diffusion.T
            else:
                s, e = offsets["diffusion_diag"]
                diff_chol_diag = jnp.abs(theta[s:e])
                params["diffusion_cov"] = jnp.diag(diff_chol_diag ** 2)

            # CINT
            if spec.cint is None:
                params["cint"] = None
            elif isinstance(spec.cint, jnp.ndarray):
                params["cint"] = spec.cint
            else:
                s, e = offsets["cint"]
                params["cint"] = theta[s:e]

            # Lambda matrix
            if isinstance(spec.lambda_mat, jnp.ndarray):
                params["lambda_mat"] = spec.lambda_mat
            else:
                lambda_mat = jnp.eye(n_m, n)
                if "lambda_free" in offsets and n_m > n:
                    s, e = offsets["lambda_free"]
                    free_loadings = theta[s:e]
                    idx = 0
                    for i in range(n, n_m):
                        for j in range(n):
                            lambda_mat = lambda_mat.at[i, j].set(free_loadings[idx])
                            idx += 1
                params["lambda_mat"] = lambda_mat

            # Manifest means
            if spec.manifest_means is None:
                params["manifest_means"] = jnp.zeros(n_m)
            elif isinstance(spec.manifest_means, jnp.ndarray):
                params["manifest_means"] = spec.manifest_means
            elif "manifest_means" in offsets:
                s, e = offsets["manifest_means"]
                params["manifest_means"] = theta[s:e]
            else:
                params["manifest_means"] = jnp.zeros(n_m)

            # Manifest covariance
            if isinstance(spec.manifest_var, jnp.ndarray):
                params["manifest_cov"] = spec.manifest_var @ spec.manifest_var.T
            else:
                s, e = offsets["manifest_var_diag"]
                var_chol_diag = jnp.abs(theta[s:e])
                params["manifest_cov"] = jnp.diag(var_chol_diag ** 2)

            # T0 params (mean estimated, cov fixed at identity)
            if isinstance(spec.t0_means, jnp.ndarray):
                params["t0_mean"] = spec.t0_means
            elif "t0_means" in offsets:
                s, e = offsets["t0_means"]
                params["t0_mean"] = theta[s:e]
            else:
                params["t0_mean"] = jnp.zeros(n)
            params["t0_cov"] = jnp.eye(n)

            return params

        # --- Build log_prior_fn ---
        def log_prior_fn(theta: jnp.ndarray) -> float:
            lp = 0.0

            if "drift_diag" in offsets:
                s, e = offsets["drift_diag"]
                vals = theta[s:e]
                mu = priors.drift_diag["mu"]
                sigma = priors.drift_diag["sigma"]
                lp = lp + jnp.sum(-0.5 * ((vals - mu) / sigma) ** 2 - jnp.log(sigma))

            if "drift_offdiag" in offsets:
                s, e = offsets["drift_offdiag"]
                vals = theta[s:e]
                mu = priors.drift_offdiag["mu"]
                sigma = priors.drift_offdiag["sigma"]
                lp = lp + jnp.sum(-0.5 * ((vals - mu) / sigma) ** 2 - jnp.log(sigma))

            if "diffusion_diag" in offsets:
                s, e = offsets["diffusion_diag"]
                vals = jnp.abs(theta[s:e])
                sigma = priors.diffusion_diag["sigma"]
                # HalfNormal: log p(x) = -x^2/(2*sigma^2) + const (for x>0)
                lp = lp + jnp.sum(-0.5 * (vals / sigma) ** 2 - jnp.log(sigma))

            if "cint" in offsets:
                s, e = offsets["cint"]
                vals = theta[s:e]
                mu = priors.cint["mu"]
                sigma = priors.cint["sigma"]
                lp = lp + jnp.sum(-0.5 * ((vals - mu) / sigma) ** 2 - jnp.log(sigma))

            if "t0_means" in offsets:
                s, e = offsets["t0_means"]
                vals = theta[s:e]
                mu = priors.t0_means["mu"]
                sigma = priors.t0_means["sigma"]
                lp = lp + jnp.sum(-0.5 * ((vals - mu) / sigma) ** 2 - jnp.log(sigma))

            if "manifest_var_diag" in offsets:
                s, e = offsets["manifest_var_diag"]
                vals = jnp.abs(theta[s:e])
                sigma = priors.manifest_var_diag["sigma"]
                lp = lp + jnp.sum(-0.5 * (vals / sigma) ** 2 - jnp.log(sigma))

            if "lambda_free" in offsets:
                s, e = offsets["lambda_free"]
                vals = theta[s:e]
                mu = priors.lambda_free["mu"]
                sigma = priors.lambda_free["sigma"]
                lp = lp + jnp.sum(-0.5 * ((vals - mu) / sigma) ** 2 - jnp.log(sigma))

            if "manifest_means" in offsets:
                s, e = offsets["manifest_means"]
                vals = theta[s:e]
                mu = priors.manifest_means["mu"]
                sigma = priors.manifest_means["sigma"]
                lp = lp + jnp.sum(-0.5 * ((vals - mu) / sigma) ** 2 - jnp.log(sigma))

            return lp

        # --- Build init_theta from prior means ---
        init_parts = []
        for name, size in segments:
            if name == "drift_diag":
                init_parts.append(jnp.full(size, priors.drift_diag["mu"]))
            elif name == "drift_offdiag":
                init_parts.append(jnp.full(size, priors.drift_offdiag["mu"]))
            elif name == "diffusion_diag":
                init_parts.append(jnp.full(size, priors.diffusion_diag["sigma"] * 0.5))
            elif name == "cint":
                init_parts.append(jnp.full(size, priors.cint["mu"]))
            elif name == "t0_means":
                init_parts.append(jnp.full(size, priors.t0_means["mu"]))
            elif name == "manifest_var_diag":
                init_parts.append(jnp.full(size, priors.manifest_var_diag["sigma"] * 0.5))
            elif name == "lambda_free":
                init_parts.append(jnp.full(size, priors.lambda_free["mu"]))
            elif name == "manifest_means":
                init_parts.append(jnp.full(size, priors.manifest_means["mu"]))
        init_theta = jnp.concatenate(init_parts) if init_parts else jnp.array([])

        # --- Compute time intervals ---
        time_intervals = jnp.diff(times, prepend=times[0])
        time_intervals = time_intervals.at[0].set(1e-6)

        # --- Observation mask ---
        obs_mask = ~jnp.isnan(observations)
        clean_obs = jnp.nan_to_num(observations, nan=0.0)

        # --- Create SSMAdapter ---
        adapter = SSMAdapter(
            n_latent=n,
            n_manifest=n_m,
            manifest_dist=spec.manifest_dist.value,
            diffusion_dist=spec.diffusion_dist.value,
        )

        return run_pmmh(
            model=adapter,
            observations=clean_obs,
            time_intervals=time_intervals,
            obs_mask=obs_mask,
            log_prior_fn=log_prior_fn,
            unpack_fn=unpack_fn,
            init_theta=init_theta,
            n_samples=n_samples,
            n_warmup=n_warmup,
            n_particles=n_particles,
            proposal_cov=proposal_cov,
            seed=seed,
        )

    def prior_predictive(
        self,
        times: jnp.ndarray,
        num_samples: int = 100,
        seed: int = 0,
    ) -> dict[str, jnp.ndarray]:
        """Sample from the prior predictive distribution.

        Args:
            times: (T,) time points
            num_samples: Number of prior samples
            seed: Random seed

        Returns:
            Dict of prior predictive samples
        """
        from numpyro.infer import Predictive

        rng_key = random.PRNGKey(seed)
        predictive = Predictive(self.model, num_samples=num_samples)

        # Create dummy observations
        dummy_obs = jnp.zeros((len(times), self.spec.n_manifest))

        return predictive(rng_key, dummy_obs, times)


def build_ssm_model(
    n_latent: int,
    n_manifest: int,
    lambda_mat: jnp.ndarray | None = None,
    hierarchical: bool = False,
    n_subjects: int = 1,
    indvarying: list[str] | None = None,
) -> SSMModel:
    """Convenience function to build a state-space model.

    Args:
        n_latent: Number of latent processes
        n_manifest: Number of manifest indicators
        lambda_mat: Fixed factor loadings (optional)
        hierarchical: Whether to use hierarchical structure
        n_subjects: Number of subjects (for hierarchical)
        indvarying: Which parameters vary across individuals

    Returns:
        SSMModel instance
    """
    spec = SSMSpec(
        n_latent=n_latent,
        n_manifest=n_manifest,
        lambda_mat=lambda_mat if lambda_mat is not None else "free",
        hierarchical=hierarchical,
        n_subjects=n_subjects,
        indvarying=indvarying or ["t0_means"],
    )
    return SSMModel(spec)
