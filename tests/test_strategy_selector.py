"""Tests for inference strategy selection."""

import jax.numpy as jnp
import pytest


class TestStrategySelection:
    """Test select_strategy() routing logic."""

    def test_linear_gaussian_returns_kalman(self):
        """Linear-Gaussian model should use Kalman filter."""
        from dsem_agent.models.ctsem import CTSEMSpec, NoiseFamily
        from dsem_agent.models.strategy_selector import InferenceStrategy, select_strategy

        spec = CTSEMSpec(
            n_latent=2,
            n_manifest=2,
            diffusion_dist=NoiseFamily.GAUSSIAN,
            manifest_dist=NoiseFamily.GAUSSIAN,
        )
        assert select_strategy(spec) == InferenceStrategy.KALMAN

    def test_nongaussian_observation_returns_particle(self):
        """Non-Gaussian observation noise should use particle filter."""
        from dsem_agent.models.ctsem import CTSEMSpec, NoiseFamily
        from dsem_agent.models.strategy_selector import InferenceStrategy, select_strategy

        spec = CTSEMSpec(
            n_latent=2,
            n_manifest=2,
            diffusion_dist=NoiseFamily.GAUSSIAN,
            manifest_dist=NoiseFamily.POISSON,
        )
        assert select_strategy(spec) == InferenceStrategy.PARTICLE

    def test_nongaussian_process_returns_particle(self):
        """Non-Gaussian process noise should use particle filter."""
        from dsem_agent.models.ctsem import CTSEMSpec, NoiseFamily
        from dsem_agent.models.strategy_selector import InferenceStrategy, select_strategy

        spec = CTSEMSpec(
            n_latent=2,
            n_manifest=2,
            diffusion_dist=NoiseFamily.STUDENT_T,
            manifest_dist=NoiseFamily.GAUSSIAN,
        )
        assert select_strategy(spec) == InferenceStrategy.PARTICLE

    def test_default_spec_is_kalman(self):
        """Default CTSEMSpec (no explicit distributions) should use Kalman."""
        from dsem_agent.models.ctsem import CTSEMSpec
        from dsem_agent.models.strategy_selector import InferenceStrategy, select_strategy

        spec = CTSEMSpec(n_latent=2, n_manifest=2)
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

    def test_ss_reference_is_nonlinear(self):
        """Expressions with ss_ (state-space) references are nonlinear."""
        from dsem_agent.models.strategy_selector import _has_state_dependent_terms

        assert _has_state_dependent_terms("param * ss_level") is True
        assert _has_state_dependent_terms("ss_slope + constant") is True

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
        """UKF backend should be created (raises on compute_log_likelihood)."""
        from dsem_agent.models.likelihoods.ukf import UKFLikelihood
        from dsem_agent.models.strategy_selector import (
            InferenceStrategy,
            get_likelihood_backend,
        )

        backend = get_likelihood_backend(InferenceStrategy.UKF)
        assert isinstance(backend, UKFLikelihood)

        # NotImplementedError raised when trying to compute
        with pytest.raises(NotImplementedError, match="dynamax"):
            backend.compute_log_likelihood(None, None, None, None, None)

    def test_particle_backend_created(self):
        """Particle backend should be created (raises on compute_log_likelihood)."""
        from dsem_agent.models.likelihoods.particle import ParticleLikelihood
        from dsem_agent.models.strategy_selector import (
            InferenceStrategy,
            get_likelihood_backend,
        )

        backend = get_likelihood_backend(InferenceStrategy.PARTICLE)
        assert isinstance(backend, ParticleLikelihood)

        # NotImplementedError raised when trying to compute
        with pytest.raises(NotImplementedError, match="cuthbert"):
            backend.compute_log_likelihood(None, None, None, None, None)


class TestNoiseFamily:
    """Test NoiseFamily enum."""

    def test_noise_family_values(self):
        """NoiseFamily should have expected values."""
        from dsem_agent.models.ctsem.model import NoiseFamily

        assert NoiseFamily.GAUSSIAN == "gaussian"
        assert NoiseFamily.STUDENT_T == "student_t"
        assert NoiseFamily.POISSON == "poisson"
        assert NoiseFamily.GAMMA == "gamma"

    def test_noise_family_in_spec(self):
        """CTSEMSpec should accept NoiseFamily enum values."""
        from dsem_agent.models.ctsem import CTSEMSpec, NoiseFamily

        spec = CTSEMSpec(
            n_latent=2,
            n_manifest=2,
            diffusion_dist=NoiseFamily.GAUSSIAN,
            manifest_dist=NoiseFamily.POISSON,
        )
        assert spec.diffusion_dist == NoiseFamily.GAUSSIAN
        assert spec.manifest_dist == NoiseFamily.POISSON
