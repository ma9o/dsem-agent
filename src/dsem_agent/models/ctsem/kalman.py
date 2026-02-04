"""Kalman filter for CT-SEM (stub).

Implementation will be merged from numpyro-ctsem branch.
"""

from typing import NamedTuple

import jax.numpy as jnp


class KalmanState(NamedTuple):
    """State of the Kalman filter at a single time point."""

    mean: jnp.ndarray  # Filtered state mean (n_latent,)
    cov: jnp.ndarray  # Filtered state covariance (n_latent, n_latent)
    log_lik: float  # Log-likelihood contribution


class KalmanFilterResult(NamedTuple):
    """Full result of Kalman filtering."""

    filtered_means: jnp.ndarray  # (T, n_latent)
    filtered_covs: jnp.ndarray  # (T, n_latent, n_latent)
    predicted_means: jnp.ndarray  # (T, n_latent)
    predicted_covs: jnp.ndarray  # (T, n_latent, n_latent)
    log_likelihood: float  # Total log-likelihood


def kalman_filter(
    observations: jnp.ndarray,
    time_intervals: jnp.ndarray,
    drift: jnp.ndarray,
    diffusion_cov: jnp.ndarray,
    cint: jnp.ndarray | None,
    lambda_mat: jnp.ndarray,
    manifest_means: jnp.ndarray,
    manifest_cov: jnp.ndarray,
    t0_mean: jnp.ndarray,
    t0_cov: jnp.ndarray,
    obs_mask: jnp.ndarray | None = None,
) -> KalmanFilterResult:
    """Run Kalman filter over all observations."""
    raise NotImplementedError("Will be merged from numpyro-ctsem")


def kalman_log_likelihood(
    observations: jnp.ndarray,
    time_intervals: jnp.ndarray,
    drift: jnp.ndarray,
    diffusion_cov: jnp.ndarray,
    cint: jnp.ndarray | None,
    lambda_mat: jnp.ndarray,
    manifest_means: jnp.ndarray,
    manifest_cov: jnp.ndarray,
    t0_mean: jnp.ndarray,
    t0_cov: jnp.ndarray,
    obs_mask: jnp.ndarray | None = None,
) -> float:
    """Compute log-likelihood via Kalman filter (optimized for MCMC)."""
    raise NotImplementedError("Will be merged from numpyro-ctsem")
