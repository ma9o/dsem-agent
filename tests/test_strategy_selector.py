"""Tests for inference strategy selection."""

import jax.numpy as jnp


class TestStrategySelection:
    """Test select_strategy() routing logic."""

    def test_linear_gaussian_returns_kalman(self):
        """Linear-Gaussian model should use Kalman filter."""
        from dsem_agent.models.ssm import NoiseFamily, SSMSpec
        from dsem_agent.models.strategy_selector import InferenceStrategy, select_strategy

        spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            diffusion_dist=NoiseFamily.GAUSSIAN,
            manifest_dist=NoiseFamily.GAUSSIAN,
        )
        assert select_strategy(spec) == InferenceStrategy.KALMAN

    def test_nongaussian_observation_returns_particle(self):
        """Non-Gaussian observation noise should use particle filter."""
        from dsem_agent.models.ssm import NoiseFamily, SSMSpec
        from dsem_agent.models.strategy_selector import InferenceStrategy, select_strategy

        spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            diffusion_dist=NoiseFamily.GAUSSIAN,
            manifest_dist=NoiseFamily.POISSON,
        )
        assert select_strategy(spec) == InferenceStrategy.PARTICLE

    def test_student_t_observation_returns_particle(self):
        """Student-t observation noise should use particle filter."""
        from dsem_agent.models.ssm import NoiseFamily, SSMSpec
        from dsem_agent.models.strategy_selector import InferenceStrategy, select_strategy

        spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            diffusion_dist=NoiseFamily.GAUSSIAN,
            manifest_dist=NoiseFamily.STUDENT_T,
        )
        assert select_strategy(spec) == InferenceStrategy.PARTICLE

    def test_gamma_observation_returns_particle(self):
        """Gamma observation noise should use particle filter."""
        from dsem_agent.models.ssm import NoiseFamily, SSMSpec
        from dsem_agent.models.strategy_selector import InferenceStrategy, select_strategy

        spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            diffusion_dist=NoiseFamily.GAUSSIAN,
            manifest_dist=NoiseFamily.GAMMA,
        )
        assert select_strategy(spec) == InferenceStrategy.PARTICLE

    def test_nongaussian_process_returns_particle(self):
        """Non-Gaussian process noise should use particle filter."""
        from dsem_agent.models.ssm import NoiseFamily, SSMSpec
        from dsem_agent.models.strategy_selector import InferenceStrategy, select_strategy

        spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            diffusion_dist=NoiseFamily.STUDENT_T,
            manifest_dist=NoiseFamily.GAUSSIAN,
        )
        assert select_strategy(spec) == InferenceStrategy.PARTICLE

    def test_fixed_matrices_returns_kalman(self):
        """Fixed matrices imply linear dynamics → Kalman filter."""
        from dsem_agent.models.ssm import NoiseFamily, SSMSpec
        from dsem_agent.models.strategy_selector import InferenceStrategy, select_strategy

        spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            drift=jnp.array([[-0.5, 0.1], [0.2, -0.8]]),
            diffusion=jnp.eye(2) * 0.3,
            lambda_mat=jnp.eye(2),
            diffusion_dist=NoiseFamily.GAUSSIAN,
            manifest_dist=NoiseFamily.GAUSSIAN,
        )
        assert select_strategy(spec) == InferenceStrategy.KALMAN

    def test_default_spec_is_kalman(self):
        """Default SSMSpec (no explicit distributions) should use Kalman."""
        from dsem_agent.models.ssm import SSMSpec
        from dsem_agent.models.strategy_selector import InferenceStrategy, select_strategy

        spec = SSMSpec(n_latent=2, n_manifest=2)
        assert select_strategy(spec) == InferenceStrategy.KALMAN


class TestHasStateDependentTerms:
    """Test _has_state_dependent_terms() linearity detection."""

    def test_ndarray_is_linear(self):
        """Constant ndarray should be linear."""
        from dsem_agent.models.strategy_selector import _has_state_dependent_terms

        assert _has_state_dependent_terms(jnp.eye(2)) is False
        assert _has_state_dependent_terms(jnp.zeros((2, 3))) is False

    def test_free_is_linear(self):
        """'free' string should be linear."""
        from dsem_agent.models.strategy_selector import _has_state_dependent_terms

        assert _has_state_dependent_terms("free") is False

    def test_diag_is_linear(self):
        """'diag' string should be linear."""
        from dsem_agent.models.strategy_selector import _has_state_dependent_terms

        assert _has_state_dependent_terms("diag") is False

    def test_none_is_linear(self):
        """None should be linear."""
        from dsem_agent.models.strategy_selector import _has_state_dependent_terms

        assert _has_state_dependent_terms(None) is False

    def test_state_reference_is_nonlinear(self):
        """Expressions with state# references are nonlinear."""
        from dsem_agent.models.strategy_selector import _has_state_dependent_terms

        assert _has_state_dependent_terms("drift_eta1 * state#eta1") is True
        assert _has_state_dependent_terms("state#latent_1 + param") is True
        assert _has_state_dependent_terms("state[0] * param") is True

    def test_ss_reference_is_nonlinear(self):
        """Expressions with ss_ (state-space) references are nonlinear."""
        from dsem_agent.models.strategy_selector import _has_state_dependent_terms

        assert _has_state_dependent_terms("param * ss_level") is True
        assert _has_state_dependent_terms("ss_slope + constant") is True
        assert _has_state_dependent_terms("ss[0] + ss[1]") is True

    def test_nonlinear_function_is_nonlinear(self):
        """Expressions with nonlinear functions are nonlinear."""
        from dsem_agent.models.strategy_selector import _has_state_dependent_terms

        assert _has_state_dependent_terms("exp(param)") is True
        assert _has_state_dependent_terms("log(x)") is True
        assert _has_state_dependent_terms("tanh(drift)") is True
        assert _has_state_dependent_terms("sqrt(variance)") is True

    def test_simple_param_is_linear(self):
        """Simple parameter names without state refs are linear."""
        from dsem_agent.models.strategy_selector import _has_state_dependent_terms

        assert _has_state_dependent_terms("drift_eta1") is False
        assert _has_state_dependent_terms("param_1 + param_2") is False

    def test_numeric_constant_is_linear(self):
        """Numeric constants (int, float) are linear."""
        from dsem_agent.models.strategy_selector import _has_state_dependent_terms

        assert _has_state_dependent_terms(0.5) is False
        assert _has_state_dependent_terms(1) is False


class TestGetLikelihoodBackend:
    """Test get_likelihood_backend() factory."""

    def test_kalman_backend(self):
        """Should return KalmanLikelihood for KALMAN strategy."""
        from dsem_agent.models.likelihoods.kalman import KalmanLikelihood
        from dsem_agent.models.strategy_selector import (
            InferenceStrategy,
            get_likelihood_backend,
        )

        backend = get_likelihood_backend(InferenceStrategy.KALMAN)
        assert isinstance(backend, KalmanLikelihood)

    def test_ukf_backend_created(self):
        """UKF backend should be created and work."""
        from dsem_agent.models.likelihoods.ukf import UKFLikelihood
        from dsem_agent.models.strategy_selector import (
            InferenceStrategy,
            get_likelihood_backend,
        )

        backend = get_likelihood_backend(InferenceStrategy.UKF)
        assert isinstance(backend, UKFLikelihood)

    def test_particle_backend_raises(self):
        """PARTICLE strategy should raise ValueError (uses PMMH instead)."""
        import pytest

        from dsem_agent.models.strategy_selector import (
            InferenceStrategy,
            get_likelihood_backend,
        )

        with pytest.raises(ValueError, match="PMMH"):
            get_likelihood_backend(InferenceStrategy.PARTICLE)


class TestCuthbertBootstrapFilter:
    """Test cuthbert-backed bootstrap particle filter."""

    def test_cuthbert_filter_finite_likelihood(self):
        """Cuthbert filter should produce finite log-likelihood."""
        import jax.random as random

        from dsem_agent.models.pmmh import SSMAdapter, cuthbert_bootstrap_filter

        n_latent, n_manifest, T = 2, 2, 5

        model = SSMAdapter(n_latent, n_manifest)
        params = {
            "drift": jnp.array([[-1.0, 0.0], [0.0, -1.0]]),
            "diffusion_cov": 0.1 * jnp.eye(n_latent),
            "lambda_mat": jnp.eye(n_manifest, n_latent),
            "manifest_means": jnp.zeros(n_manifest),
            "manifest_cov": 0.5 * jnp.eye(n_manifest),
            "t0_mean": jnp.zeros(n_latent),
            "t0_cov": jnp.eye(n_latent),
        }

        observations = jnp.ones((T, n_manifest)) * 0.5
        time_intervals = jnp.ones(T)
        obs_mask = ~jnp.isnan(observations)

        result = cuthbert_bootstrap_filter(
            model,
            params,
            observations,
            time_intervals,
            obs_mask,
            n_particles=500,
            key=random.PRNGKey(42),
        )

        assert jnp.isfinite(result.log_likelihood)
        assert result.final_particles.shape == (500, n_latent)
        assert result.final_log_weights.shape == (500,)

    def test_cuthbert_filter_consistency_across_seeds(self):
        """Cuthbert filter should give similar results across seeds."""
        import jax.random as random

        from dsem_agent.models.pmmh import SSMAdapter, cuthbert_bootstrap_filter

        n_latent, n_manifest, T = 2, 2, 5

        model = SSMAdapter(n_latent, n_manifest)
        params = {
            "drift": jnp.array([[-0.5, 0.0], [0.0, -0.5]]),
            "diffusion_cov": 0.1 * jnp.eye(n_latent),
            "lambda_mat": jnp.eye(n_manifest, n_latent),
            "manifest_means": jnp.zeros(n_manifest),
            "manifest_cov": 0.5 * jnp.eye(n_manifest),
            "t0_mean": jnp.zeros(n_latent),
            "t0_cov": jnp.eye(n_latent),
        }

        observations = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.2, 0.1], [0.4, 0.3], [0.5, 0.5]])
        time_intervals = jnp.ones(T)
        obs_mask = ~jnp.isnan(observations)

        lls = []
        for seed in [0, 1, 2]:
            result = cuthbert_bootstrap_filter(
                model,
                params,
                observations,
                time_intervals,
                obs_mask,
                n_particles=1000,
                key=random.PRNGKey(seed),
            )
            lls.append(float(result.log_likelihood))

        # Check variance is reasonable
        mean_ll = sum(lls) / len(lls)
        max_deviation = max(abs(ll - mean_ll) for ll in lls)
        assert max_deviation / abs(mean_ll) < 0.2

    def test_pmmh_kernel_uses_cuthbert_by_default(self):
        """PMMH kernel should use cuthbert filter by default."""
        import jax.random as random

        from dsem_agent.models.pmmh import SSMAdapter, pmmh_kernel

        n_latent, n_manifest, T = 2, 2, 5

        model = SSMAdapter(n_latent, n_manifest)
        observations = jnp.ones((T, n_manifest)) * 0.5
        time_intervals = jnp.ones(T)
        obs_mask = ~jnp.isnan(observations)

        def unpack(theta):
            return {
                "drift": jnp.array([[-theta[0], 0.0], [0.0, -theta[0]]]),
                "diffusion_cov": 0.1 * jnp.eye(n_latent),
                "lambda_mat": jnp.eye(n_manifest, n_latent),
                "manifest_means": jnp.zeros(n_manifest),
                "manifest_cov": 0.5 * jnp.eye(n_manifest),
                "t0_mean": jnp.zeros(n_latent),
                "t0_cov": jnp.eye(n_latent),
            }

        def log_prior(theta):
            return -0.5 * jnp.sum(theta**2)

        init_fn, step_fn = pmmh_kernel(
            model, observations, time_intervals, obs_mask,
            log_prior, unpack, n_particles=100,
        )

        # Should initialize without error (uses cuthbert internally)
        state = init_fn(jnp.array([1.0]), random.PRNGKey(0))
        assert jnp.isfinite(state.log_likelihood)

        # Should step without error
        new_state = step_fn(state, random.PRNGKey(1))
        assert jnp.isfinite(new_state.log_likelihood)


class TestNoiseFamily:
    """Test NoiseFamily enum."""

    def test_noise_family_values(self):
        """NoiseFamily should have expected values."""
        from dsem_agent.models.ssm.model import NoiseFamily

        assert NoiseFamily.GAUSSIAN == "gaussian"
        assert NoiseFamily.STUDENT_T == "student_t"
        assert NoiseFamily.POISSON == "poisson"
        assert NoiseFamily.GAMMA == "gamma"

    def test_noise_family_in_spec(self):
        """SSMSpec should accept NoiseFamily enum values."""
        from dsem_agent.models.ssm import NoiseFamily, SSMSpec

        spec = SSMSpec(
            n_latent=2,
            n_manifest=2,
            diffusion_dist=NoiseFamily.GAUSSIAN,
            manifest_dist=NoiseFamily.POISSON,
        )
        assert spec.diffusion_dist == NoiseFamily.GAUSSIAN
        assert spec.manifest_dist == NoiseFamily.POISSON
