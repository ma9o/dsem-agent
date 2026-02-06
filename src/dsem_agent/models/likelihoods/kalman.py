"""Kalman filter likelihood backend wrapping cuthbert gaussian.moments.

Exact log-likelihood for linear-Gaussian state-space models.
Uses cuthbert's moments filter with linear closures — since the dynamics
and observation are linear, the Jacobian-based linearization is exact.

Use when:
- Dynamics are linear (drift matrix is constant)
- Process noise is Gaussian
- Measurement model is linear
- Observation noise is Gaussian

Convention mapping:
    Our code: time_intervals[t] = gap BEFORE observation t
    cuthbert: model_inputs[0] = init + first observation
              model_inputs[k] (k>=1) = dynamics from k-1 to k + observation k
              dynamics[k] uses time_intervals[k] directly (no shifting needed)
"""

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


class KalmanLikelihood:
    """Kalman filter likelihood backend via cuthbert moments filter.

    Computes exact log-likelihood for linear-Gaussian SSMs using cuthbert's
    moments filter with linear closures (exact for linear models).

    Example usage in NumPyro:
        backend = KalmanLikelihood()
        ll = backend.compute_log_likelihood(ct_params, meas_params, init, obs, dt)
        numpyro.factor("ssm", ll)
    """

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

        # 4. Build model_inputs with leading temporal dimension T
        H = jnp.broadcast_to(
            measurement_params.lambda_mat, (T, *measurement_params.lambda_mat.shape)
        )
        d = jnp.broadcast_to(
            measurement_params.manifest_means, (T, *measurement_params.manifest_means.shape)
        )

        if cd is not None:
            dynamics_bias = cd  # (T, n_latent)
        else:
            dynamics_bias = jnp.zeros((T, n_latent))

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
        }

        # 5. Define closures — linear dynamics and observation functions
        def get_init_params(mi):
            return mi["m0"], mi["chol_P0"]

        def get_dynamics_moments(state, mi):
            Ad_t = mi["Ad"]
            cd_t = mi["cd"]
            chol_Qd_t = mi["chol_Qd"]

            def mean_and_chol_cov(x):
                return Ad_t @ x + cd_t, chol_Qd_t

            return mean_and_chol_cov, state.mean

        def get_observation_moments(state, mi):
            H_t = mi["H"]
            d_t = mi["d"]
            chol_R_t = mi["chol_R"]
            y_t = mi["y"]

            def mean_and_chol_cov(x):
                return H_t @ x + d_t, chol_R_t

            return mean_and_chol_cov, state.mean, y_t

        # 6. Build and run filter (non-associative for stable gradients)
        filter_obj = build_filter(
            get_init_params=get_init_params,
            get_dynamics_params=get_dynamics_moments,
            get_observation_params=get_observation_moments,
            associative=False,
        )

        states = cuthbert_filter(filter_obj, model_inputs)

        # 7. Return cumulative log marginal likelihood
        return states.log_normalizing_constant[-1]
