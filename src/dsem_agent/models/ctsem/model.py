"""NumPyro CT-SEM model (stub).

Implementation will be merged from numpyro-ctsem branch.
"""

from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
from numpyro.infer import MCMC


@dataclass
class CTSEMSpec:
    """Specification for a CT-SEM model."""

    n_latent: int
    n_manifest: int
    lambda_mat: jnp.ndarray  # Factor loadings (n_manifest, n_latent)

    # Parameter constraints
    drift: str = "free"  # "free", "diag", or fixed matrix
    diffusion: str = "diag"  # "diag", "full", or fixed matrix
    cint: str | None = None  # "free" or None (zero)
    manifest_means: jnp.ndarray | None = None  # Fixed or None (zero)
    manifest_var: str = "diag"  # "diag", "full", or fixed matrix

    # Initial state
    t0_means: str = "free"  # "free" or fixed
    t0_var: str = "diag"  # "diag", "full", or fixed

    # Hierarchical structure
    hierarchical: bool = False
    n_subjects: int = 1
    indvarying: list[str] = field(default_factory=list)

    # Manifest variable names
    manifest_names: list[str] | None = None


@dataclass
class CTSEMPriors:
    """Prior specifications for CT-SEM parameters."""

    # Drift matrix priors
    drift_diag: dict = field(default_factory=lambda: {"mu": -0.5, "sigma": 1.0})
    drift_offdiag: dict = field(default_factory=lambda: {"mu": 0.0, "sigma": 0.5})

    # Diffusion priors
    diffusion_diag: dict = field(default_factory=lambda: {"sigma": 1.0})

    # CINT priors
    cint: dict = field(default_factory=lambda: {"mu": 0.0, "sigma": 1.0})

    # Manifest variance priors
    manifest_var: dict = field(default_factory=lambda: {"sigma": 1.0})

    # Initial state priors
    t0_mean: dict = field(default_factory=lambda: {"mu": 0.0, "sigma": 2.0})
    t0_var: dict = field(default_factory=lambda: {"sigma": 2.0})

    # Hierarchical priors
    indvarying_sd: dict = field(default_factory=lambda: {"sigma": 1.0})


class CTSEMModel:
    """Hierarchical Bayesian CT-SEM model using NumPyro."""

    def __init__(self, spec: CTSEMSpec, priors: CTSEMPriors | None = None):
        """Initialize CT-SEM model.

        Args:
            spec: Model specification
            priors: Prior distributions (uses defaults if None)
        """
        self.spec = spec
        self.priors = priors or CTSEMPriors()
        self._mcmc: MCMC | None = None

    def model(
        self,
        observations: jnp.ndarray,
        times: jnp.ndarray,
        subject_ids: jnp.ndarray | None = None,
    ) -> None:
        """NumPyro model function."""
        raise NotImplementedError("Will be merged from numpyro-ctsem")

    def fit(
        self,
        observations: jnp.ndarray,
        times: jnp.ndarray,
        subject_ids: jnp.ndarray | None = None,
        num_warmup: int = 1000,
        num_samples: int = 1000,
        num_chains: int = 4,
        seed: int = 0,
        **kwargs: Any,
    ) -> MCMC:
        """Fit the model using MCMC."""
        raise NotImplementedError("Will be merged from numpyro-ctsem")

    def prior_predictive(
        self,
        times: jnp.ndarray,
        num_samples: int = 500,
        seed: int = 0,
    ) -> dict[str, jnp.ndarray]:
        """Sample from the prior predictive distribution."""
        raise NotImplementedError("Will be merged from numpyro-ctsem")

    def get_samples(self) -> dict[str, jnp.ndarray]:
        """Get posterior samples."""
        if self._mcmc is None:
            raise ValueError("Model must be fit before getting samples")
        return self._mcmc.get_samples()
