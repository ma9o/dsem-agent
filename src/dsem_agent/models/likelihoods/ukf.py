"""Moments filter likelihood backend wrapping cuthbert gaussian.moments.

Approximate log-likelihood for mildly nonlinear Gaussian models via
Jacobian-based linearization (EKF-like). For linear models this is exact.

Use when:
- Dynamics are nonlinear but smooth (no discontinuities)
- Process noise is Gaussian
- Observation noise is Gaussian
- Measurement model may be nonlinear

Convention: same as KalmanLikelihood — time_intervals[t] is the gap
BEFORE observation t. cuthbert handles time indexing directly.

Note: This replaces the previous UKF (sigma-point) approach. For linear
models the result is identical. For mildly nonlinear models (our use case)
the accuracy is equivalent.
"""

from collections.abc import Callable

import jax.numpy as jnp
from cuthbert.filtering import filter as cuthbert_filter
from cuthbert.gaussian.moments import build_filter

from dsem_agent.models.likelihoods.base import (
    CTParams,
    InitialStateParams,
    MeasurementParams,
    preprocess_missing_data,
)
from dsem_agent.models.ssm.discretization import discretize_system_batched


def _safe_cholesky(M: jnp.ndarray) -> jnp.ndarray:
    """Compute Cholesky factor with jitter for numerical stability.

    Uses 1e-6 jitter to ensure cuthbert's internal QR decomposition
    produces well-conditioned matrices for stable gradient propagation.
    """
    n = M.shape[-1]
    return jnp.linalg.cholesky(M + jnp.eye(n) * 1e-6)


class UKFLikelihood:
    """Moments filter likelihood backend via cuthbert.

    Computes approximate log-likelihood using Jacobian-based linearization
    around the current filter mean. For linear models this is exact.

    Supports optional custom dynamics and measurement functions for
    nonlinear state-space models.

    Example:
        backend = UKFLikelihood()
        ll = backend.compute_log_likelihood(ct_params, meas_params, init, obs, dt)
        numpyro.factor("ssm", ll)
    """

    def __init__(
        self,
        dynamics_fn: Callable[[jnp.ndarray, jnp.ndarray, float], jnp.ndarray] | None = None,
        measurement_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = None,
    ):
        """Initialize moments filter with optional custom functions.

        Args:
            dynamics_fn: Custom dynamics x_{t+1} = f(x_t, params, dt). If None,
                uses linear dynamics from ct_params.
            measurement_fn: Custom measurement y = h(x, params). If None,
                uses linear measurement from measurement_params.
        """
        self.custom_dynamics_fn = dynamics_fn
        self.custom_measurement_fn = measurement_fn

    def compute_log_likelihood(
        self,
        ct_params: CTParams,
        measurement_params: MeasurementParams,
        initial_state: InitialStateParams,
        observations: jnp.ndarray,
        time_intervals: jnp.ndarray,
        obs_mask: jnp.ndarray | None = None,
    ) -> float:
        """Compute log-likelihood via cuthbert moments filter.

        Args:
            ct_params: Continuous-time dynamics (drift, diffusion_cov, cint)
            measurement_params: Observation model (lambda_mat, manifest_means, manifest_cov)
            initial_state: Initial state distribution (mean, cov)
            observations: (T, n_manifest) observed data
            time_intervals: (T,) time intervals BEFORE each observation
            obs_mask: (T, n_manifest) boolean mask for observed values

        Returns:
            Total log-likelihood (scalar)
        """
        T = observations.shape[0]
        n_latent = initial_state.mean.shape[0]

        # 1. Discretize dynamics for each time step
        Ad, Qd, cd = discretize_system_batched(
            ct_params.drift, ct_params.diffusion_cov, ct_params.cint, time_intervals
        )

        # 2. Preprocess missing data
        clean_obs, R_adjusted, _mask = preprocess_missing_data(
            observations, measurement_params.manifest_cov, obs_mask
        )

        # 3. Compute Cholesky factors (cuthbert operates in square-root form)
        chol_Qd = jnp.vectorize(_safe_cholesky, signature="(n,n)->(n,n)")(Qd)
        chol_R = jnp.vectorize(_safe_cholesky, signature="(n,n)->(n,n)")(R_adjusted)
        chol_P0 = _safe_cholesky(initial_state.cov)

        # 4. Build cd
        if cd is not None:
            dynamics_bias = cd  # (T, n_latent)
        else:
            dynamics_bias = jnp.zeros((T, n_latent))

        # 5. Broadcast static params to temporal dimension
        H = jnp.broadcast_to(
            measurement_params.lambda_mat, (T, *measurement_params.lambda_mat.shape)
        )
        d = jnp.broadcast_to(
            measurement_params.manifest_means, (T, *measurement_params.manifest_means.shape)
        )

        model_inputs = {
            "Ad": Ad,              # (T, n_latent, n_latent)
            "cd": dynamics_bias,   # (T, n_latent)
            "chol_Qd": chol_Qd,   # (T, n_latent, n_latent)
            "H": H,               # (T, n_manifest, n_latent)
            "d": d,                # (T, n_manifest)
            "chol_R": chol_R,      # (T, n_manifest, n_manifest)
            "y": clean_obs,        # (T, n_manifest)
            "m0": jnp.broadcast_to(initial_state.mean, (T, n_latent)),
            "chol_P0": jnp.broadcast_to(chol_P0, (T, n_latent, n_latent)),
            "dt": time_intervals,  # (T,) needed for custom dynamics
        }

        # 6. Define closures
        custom_dynamics = self.custom_dynamics_fn
        custom_measurement = self.custom_measurement_fn
        _ct_params = ct_params
        _meas_params = measurement_params

        def get_init_params(mi):
            return mi["m0"], mi["chol_P0"]

        def get_dynamics_moments(state, mi):
            Ad_t = mi["Ad"]
            cd_t = mi["cd"]
            chol_Qd_t = mi["chol_Qd"]

            if custom_dynamics is not None:
                dt_t = mi["dt"]

                def mean_and_chol_cov(x):
                    mean = custom_dynamics(x, _ct_params, dt_t)
                    return mean, chol_Qd_t

            else:

                def mean_and_chol_cov(x):
                    mean = Ad_t @ x + cd_t
                    return mean, chol_Qd_t

            linearization_point = state.mean
            return mean_and_chol_cov, linearization_point

        def get_observation_moments(state, mi):
            H_t = mi["H"]
            d_t = mi["d"]
            chol_R_t = mi["chol_R"]
            y_t = mi["y"]

            if custom_measurement is not None:

                def mean_and_chol_cov(x):
                    mean = custom_measurement(x, _meas_params)
                    return mean, chol_R_t

            else:

                def mean_and_chol_cov(x):
                    mean = H_t @ x + d_t
                    return mean, chol_R_t

            linearization_point = state.mean
            return mean_and_chol_cov, linearization_point, y_t

        # 7. Build and run filter (non-associative: linearization depends on state)
        filter_obj = build_filter(
            get_init_params=get_init_params,
            get_dynamics_params=get_dynamics_moments,
            get_observation_params=get_observation_moments,
            associative=False,
        )

        states = cuthbert_filter(filter_obj, model_inputs)

        # 8. Return cumulative log marginal likelihood
        return states.log_normalizing_constant[-1]
