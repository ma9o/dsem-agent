"""Kalman filter likelihood backend wrapping dynamax lgssm_filter.

Exact log-likelihood for linear-Gaussian state-space models.
Delegates all numerics (Cholesky, symmetry, stability) to dynamax.

Use when:
- Dynamics are linear (drift matrix is constant)
- Process noise is Gaussian
- Measurement model is linear
- Observation noise is Gaussian

Convention mapping:
    Our code: time_intervals[t] = gap BEFORE observation t
              (time_intervals[0] is typically 1e-6, a dummy)
    dynamax:  F[t], Q[t], b[t] = dynamics applied AFTER observation t
              to predict state at t+1

    So we shift: dynamax_dynamics[t] = discretize(time_intervals[t+1])
    For the first observation, dynamax uses (m0, P0) as the predicted state
    directly (no dynamics applied before t=0).
"""

import jax.numpy as jnp
from dynamax.linear_gaussian_ssm.inference import (
    ParamsLGSSM,
    ParamsLGSSMDynamics,
    ParamsLGSSMEmissions,
    ParamsLGSSMInitial,
    lgssm_filter,
)

from dsem_agent.models.likelihoods.base import (
    CTParams,
    InitialStateParams,
    MeasurementParams,
    preprocess_missing_data,
)
from dsem_agent.models.ssm.discretization import discretize_system_batched


class KalmanLikelihood:
    """Kalman filter likelihood backend via dynamax.

    Computes exact log-likelihood by running dynamax's lgssm_filter,
    integrating out the latent states analytically.

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
        """Compute log-likelihood via dynamax Kalman filter.

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
        n_latent = initial_state.mean.shape[0]

        # 1. Build shifted dynamics: dynamax needs transitions AFTER each obs
        #    dynamics[t] transitions from obs t to obs t+1
        #    So we use time_intervals[1:] for t=0..T-2, and a dummy for t=T-1
        shifted_dt = jnp.concatenate([time_intervals[1:], time_intervals[-1:]])
        Ad, Qd, cd = discretize_system_batched(
            ct_params.drift, ct_params.diffusion_cov, ct_params.cint, shifted_dt
        )

        # 2. Preprocess missing data
        clean_obs, R_adjusted, _mask = preprocess_missing_data(
            observations, measurement_params.manifest_cov, obs_mask
        )

        # 3. Pack into dynamax ParamsLGSSM
        if cd is not None:
            dynamics_bias = cd  # (T, n_latent)
        else:
            dynamics_bias = jnp.zeros(n_latent)  # static zero bias

        params = ParamsLGSSM(
            initial=ParamsLGSSMInitial(
                mean=initial_state.mean,
                cov=initial_state.cov,
            ),
            dynamics=ParamsLGSSMDynamics(
                weights=Ad,  # (T, n, n) time-varying
                bias=dynamics_bias,
                input_weights=jnp.zeros((n_latent, 0)),
                cov=Qd,  # (T, n, n) time-varying
            ),
            emissions=ParamsLGSSMEmissions(
                weights=measurement_params.lambda_mat,  # static
                bias=measurement_params.manifest_means,  # static
                input_weights=jnp.zeros((measurement_params.lambda_mat.shape[0], 0)),
                cov=R_adjusted,  # (T, m, m) time-varying (missing data)
            ),
        )

        # 4. Run dynamax filter
        posterior = lgssm_filter(params, clean_obs)
        return posterior.marginal_loglik
