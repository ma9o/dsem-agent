"""Unscented Kalman Filter likelihood backend using dynamax UKF building blocks.

Approximate log-likelihood for mildly nonlinear Gaussian models.
Uses dynamax's _predict and _condition_on for sigma point math,
but runs its own scan loop to support time-varying Q (from varying dt).

Use when:
- Dynamics are nonlinear but smooth (no discontinuities)
- Process noise is Gaussian
- Observation noise is Gaussian
- Measurement model may be nonlinear

Convention: same as KalmanLikelihood — time_intervals[t] is the gap
BEFORE observation t. We shift to dynamax's convention where dynamics[t]
transitions AFTER observation t.
"""

from collections.abc import Callable

import jax.numpy as jnp
from dynamax.nonlinear_gaussian_ssm.inference_ukf import (
    UKFHyperParams,
    _compute_lambda,
    _compute_weights,
    _condition_on,
    _predict,
)
from jax import lax

from dsem_agent.models.likelihoods.base import (
    CTParams,
    InitialStateParams,
    MeasurementParams,
    preprocess_missing_data,
)
from dsem_agent.models.ssm.discretization import discretize_system_batched


class UKFLikelihood:
    """Unscented Kalman Filter likelihood backend via dynamax building blocks.

    Uses dynamax's _predict/_condition_on for sigma point propagation,
    with a custom scan loop that supports time-varying Q from irregular dt.

    For linear models, this is equivalent to the standard Kalman filter
    but with slightly higher computational cost.

    Example:
        backend = UKFLikelihood(alpha=1e-3, beta=2.0)
        ll = backend.compute_log_likelihood(ct_params, meas_params, init, obs, dt)
        numpyro.factor("ssm", ll)
    """

    def __init__(
        self,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
        dynamics_fn: Callable[[jnp.ndarray, jnp.ndarray, float], jnp.ndarray] | None = None,
        measurement_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = None,
    ):
        """Initialize UKF with tuning parameters and optional custom functions.

        Args:
            alpha: Spread of sigma points (small positive, e.g., 1e-3)
            beta: Distribution parameter (2.0 optimal for Gaussian)
            kappa: Secondary scaling (typically 0 or 3-n)
            dynamics_fn: Custom dynamics x_{t+1} = f(x_t, params, dt). If None,
                uses linear dynamics from ct_params.
            measurement_fn: Custom measurement y = h(x, params). If None,
                uses linear measurement from measurement_params.
        """
        self.hyperparams = UKFHyperParams(alpha=alpha, beta=beta, kappa=kappa)
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
        """Compute log-likelihood via UKF with dynamax building blocks.

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

        # 1. Build shifted dynamics (same convention as KalmanLikelihood)
        shifted_dt = jnp.concatenate([time_intervals[1:], time_intervals[-1:]])
        Ad, Qd, cd = discretize_system_batched(
            ct_params.drift, ct_params.diffusion_cov, ct_params.cint, shifted_dt
        )

        # 2. Preprocess missing data
        clean_obs, R_adjusted, _mask = preprocess_missing_data(
            observations, measurement_params.manifest_cov, obs_mask
        )

        # 3. Compute UKF weights (static, depends only on state dim)
        alpha, beta, kappa = self.hyperparams
        lamb = _compute_lambda(alpha, kappa, n_latent)
        w_mean, w_cov = _compute_weights(n_latent, alpha, beta, lamb)

        # 4. Build dynamics function: f(x, u) -> x_next
        #    u is unused (dummy input); timestep handled via scan index
        if self.custom_dynamics_fn is not None:
            custom_fn = self.custom_dynamics_fn

            def make_dynamics(_Ad_t, _cd_t, dt_t):
                def f(x, _u):
                    return custom_fn(x, ct_params, dt_t)

                return f
        else:

            def make_dynamics(Ad_t, cd_t, _dt_t):
                def f(x, _u):
                    result = Ad_t @ x
                    if cd_t is not None:
                        result = result + cd_t
                    return result

                return f

        # 5. Build emission function: h(x, u) -> y_predicted
        #    u is required by dynamax's vmap signature but unused here
        if self.custom_measurement_fn is not None:
            custom_meas = self.custom_measurement_fn

            def emission_fn(x, _u):
                return custom_meas(x, measurement_params)
        else:
            lambda_mat = measurement_params.lambda_mat
            manifest_means = measurement_params.manifest_means

            def emission_fn(x, _u):
                return lambda_mat @ x + manifest_means

        # 6. Scan: condition on obs, then predict next state
        dummy_input = jnp.zeros(0)

        def scan_fn(carry, t):
            ll, pred_mean, pred_cov = carry

            Q_t = Qd[t]
            R_t = R_adjusted[t]
            y_t = clean_obs[t]

            # Condition on this emission
            log_lik, filt_mean, filt_cov = _condition_on(
                pred_mean, pred_cov, emission_fn, R_t, lamb, w_mean, w_cov, dummy_input, y_t
            )
            ll = ll + log_lik

            # Build per-step dynamics function
            Ad_t = Ad[t]
            cd_t = cd[t] if cd is not None else None
            dt_t = shifted_dt[t]
            dynamics_fn = make_dynamics(Ad_t, cd_t, dt_t)

            # Predict next state
            next_mean, next_cov, _ = _predict(
                filt_mean, filt_cov, dynamics_fn, Q_t, lamb, w_mean, w_cov, dummy_input
            )

            return (ll, next_mean, next_cov), None

        init_carry = (0.0, initial_state.mean, initial_state.cov)
        (total_ll, _, _), _ = lax.scan(scan_fn, init_carry, jnp.arange(observations.shape[0]))

        return total_ll
