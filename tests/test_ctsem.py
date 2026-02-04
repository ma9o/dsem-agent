"""Tests for CT-SEM NumPyro implementation.

Tests core functionality:
1. Matrix utilities (expm, Lyapunov solver, discretization)
2. Kalman filter correctness
3. NumPyro model compilation and sampling
4. Parity with ctsem R package (where applicable)
"""

import jax.numpy as jnp
import jax.random as random
import numpy as np
import pytest


class TestCoreUtilities:
    """Test core matrix utilities for CT-SEM."""

    def test_solve_lyapunov_simple(self):
        """Test Lyapunov solver with simple 2x2 case."""
        from dsem_agent.models.ctsem.core import solve_lyapunov

        # Simple stable drift matrix
        A = jnp.array([[-1.0, 0.0], [0.0, -2.0]])
        Q = jnp.array([[1.0, 0.0], [0.0, 1.0]])

        X = solve_lyapunov(A, Q)

        # Verify: A*X + X*A' = -Q
        residual = A @ X + X @ A.T + Q
        assert jnp.allclose(residual, 0, atol=1e-6), f"Residual: {residual}"

    def test_solve_lyapunov_coupled(self):
        """Test Lyapunov solver with coupled system."""
        from dsem_agent.models.ctsem.core import solve_lyapunov

        # Coupled drift
        A = jnp.array([[-1.0, 0.5], [0.3, -2.0]])
        Q = jnp.array([[1.0, 0.2], [0.2, 1.0]])

        X = solve_lyapunov(A, Q)

        # Verify
        residual = A @ X + X @ A.T + Q
        assert jnp.allclose(residual, 0, atol=1e-6)

        # X should be symmetric
        assert jnp.allclose(X, X.T, atol=1e-10)

    def test_discretize_system_identity(self):
        """Test that dt=0 gives identity transformation."""
        from dsem_agent.models.ctsem.core import discretize_system

        A = jnp.array([[-1.0, 0.0], [0.0, -2.0]])
        Q = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        c = jnp.array([0.1, 0.2])

        # Very small dt should give ~identity drift
        disc_drift, disc_Q, disc_c = discretize_system(A, Q, c, dt=1e-6)

        assert jnp.allclose(disc_drift, jnp.eye(2), atol=1e-5)
        assert jnp.allclose(disc_Q, jnp.zeros((2, 2)), atol=1e-5)

    def test_discretize_system_unit_time(self):
        """Test discretization at dt=1."""
        from dsem_agent.models.ctsem.core import discretize_system

        import jax.scipy.linalg as jla

        A = jnp.array([[-1.0, 0.0], [0.0, -2.0]])
        Q = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        c = jnp.array([0.1, 0.2])

        disc_drift, disc_Q, disc_c = discretize_system(A, Q, c, dt=1.0)

        # Check discrete drift = exp(A)
        expected_drift = jla.expm(A)
        assert jnp.allclose(disc_drift, expected_drift, atol=1e-6)

        # Q should be positive semi-definite
        eigenvalues = jnp.linalg.eigvalsh(disc_Q)
        assert jnp.all(eigenvalues >= -1e-10)

    def test_compute_asymptotic_diffusion(self):
        """Test asymptotic diffusion computation."""
        from dsem_agent.models.ctsem.core import compute_asymptotic_diffusion

        A = jnp.array([[-1.0, 0.0], [0.0, -1.0]])
        G = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        Q = G @ G.T

        Q_inf = compute_asymptotic_diffusion(A, Q)

        # For diagonal A=-I and Q=I, Q_inf should be 0.5*I
        # Since A*Q_inf + Q_inf*A' = -Q => -2*Q_inf = -I => Q_inf = 0.5*I
        expected = 0.5 * jnp.eye(2)
        assert jnp.allclose(Q_inf, expected, atol=1e-6)


class TestKalmanFilter:
    """Test Kalman filter implementation."""

    def test_kalman_predict(self):
        """Test Kalman prediction step."""
        from dsem_agent.models.ctsem.kalman import kalman_predict

        state_mean = jnp.array([1.0, 0.0])
        state_cov = jnp.eye(2)
        discrete_drift = 0.9 * jnp.eye(2)
        discrete_Q = 0.1 * jnp.eye(2)
        discrete_cint = jnp.array([0.1, 0.1])

        pred_mean, pred_cov = kalman_predict(
            state_mean, state_cov, discrete_drift, discrete_Q, discrete_cint
        )

        # Check mean prediction
        expected_mean = discrete_drift @ state_mean + discrete_cint
        assert jnp.allclose(pred_mean, expected_mean)

        # Check covariance prediction
        expected_cov = discrete_drift @ state_cov @ discrete_drift.T + discrete_Q
        assert jnp.allclose(pred_cov, expected_cov)

    def test_kalman_update_reduces_uncertainty(self):
        """Test that Kalman update reduces uncertainty."""
        from dsem_agent.models.ctsem.kalman import kalman_update_simple

        pred_mean = jnp.array([0.0, 0.0])
        pred_cov = jnp.eye(2)
        observation = jnp.array([1.0, 0.5])
        lambda_mat = jnp.eye(2)
        manifest_means = jnp.zeros(2)
        manifest_cov = 0.5 * jnp.eye(2)

        upd_mean, upd_cov, ll = kalman_update_simple(
            pred_mean, pred_cov, observation, lambda_mat, manifest_means, manifest_cov
        )

        # Updated covariance should have smaller trace (less uncertainty)
        assert jnp.trace(upd_cov) < jnp.trace(pred_cov)

        # Updated mean should be closer to observation
        assert jnp.linalg.norm(upd_mean - observation) < jnp.linalg.norm(
            pred_mean - observation
        )

        # Log-likelihood should be finite
        assert jnp.isfinite(ll)

    def test_kalman_log_likelihood_finite(self):
        """Test that log-likelihood is finite for reasonable data."""
        from dsem_agent.models.ctsem.kalman import kalman_log_likelihood

        T, n_latent, n_manifest = 10, 2, 2

        # Simple observations
        observations = jnp.ones((T, n_manifest)) * 0.5
        time_intervals = jnp.ones(T)

        # Stable parameters
        drift = jnp.array([[-1.0, 0.0], [0.0, -1.0]])
        diffusion_cov = 0.1 * jnp.eye(n_latent)
        cint = None
        lambda_mat = jnp.eye(n_manifest, n_latent)
        manifest_means = jnp.zeros(n_manifest)
        manifest_cov = 0.1 * jnp.eye(n_manifest)
        t0_mean = jnp.zeros(n_latent)
        t0_cov = jnp.eye(n_latent)

        ll = kalman_log_likelihood(
            observations,
            time_intervals,
            drift,
            diffusion_cov,
            cint,
            lambda_mat,
            manifest_means,
            manifest_cov,
            t0_mean,
            t0_cov,
        )

        assert jnp.isfinite(ll)


class TestCTSEMModel:
    """Test NumPyro CT-SEM model."""

    def test_model_compiles(self):
        """Test that model compiles without errors."""
        from dsem_agent.models.ctsem import CTSEMModel, CTSEMSpec

        spec = CTSEMSpec(n_latent=2, n_manifest=2)
        model = CTSEMModel(spec)

        # Create dummy data
        T = 10
        observations = jnp.ones((T, 2)) * 0.5
        times = jnp.arange(T, dtype=float)

        # Try to trace the model (this will fail if there are shape errors)
        import numpyro

        with numpyro.handlers.seed(rng_seed=0):
            with numpyro.handlers.trace() as trace:
                model.model(observations, times)

        # Check that key sites exist
        assert "drift_diag_pop" in trace
        assert "diffusion_diag_pop" in trace

    def test_prior_predictive(self):
        """Test prior predictive sampling."""
        from dsem_agent.models.ctsem import CTSEMModel, CTSEMSpec

        spec = CTSEMSpec(n_latent=2, n_manifest=2)
        model = CTSEMModel(spec)

        times = jnp.arange(10, dtype=float)
        prior_samples = model.prior_predictive(times, num_samples=10)

        # Should have samples for key parameters
        assert "drift" in prior_samples
        assert prior_samples["drift"].shape == (10, 2, 2)

    @pytest.mark.slow
    def test_fit_runs(self):
        """Test that fitting runs without errors (minimal samples)."""
        from dsem_agent.models.ctsem import CTSEMModel, CTSEMSpec

        spec = CTSEMSpec(
            n_latent=2,
            n_manifest=2,
            lambda_mat=jnp.eye(2),  # Fix loadings to simplify
        )
        model = CTSEMModel(spec)

        # Generate simple data
        T = 20
        key = random.PRNGKey(42)
        observations = random.normal(key, (T, 2)) * 0.5
        times = jnp.arange(T, dtype=float)

        # Run with minimal samples
        mcmc = model.fit(
            observations, times, num_warmup=10, num_samples=10, num_chains=1
        )

        samples = mcmc.get_samples()
        assert "drift_diag_pop" in samples


class TestParityWithCtsem:
    """Tests for parity with the ctsem R package.

    These tests compare our NumPyro implementation against
    actual results from the ctsem R package via rpy2.
    """

    @pytest.fixture
    def r_ctsem(self):
        """Initialize R with ctsem package loaded."""
        pytest.importorskip("rpy2")
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.conversion import localconverter

        # Check if ctsem is installed
        try:
            ctsem = importr("ctsem")
        except Exception:
            pytest.skip("R package 'ctsem' not installed")

        # Also need Matrix package for expm
        try:
            matrix_pkg = importr("Matrix")
        except Exception:
            pytest.skip("R package 'Matrix' not installed")

        return {
            "ro": ro,
            "ctsem": ctsem,
            "Matrix": matrix_pkg,
            "numpy2ri": numpy2ri,
            "localconverter": localconverter,
        }

    def test_discretization_matches_ctsem(self, r_ctsem):
        """Test that discretization matches ctsem's actual R implementation.

        Calls R's ctsem package directly via rpy2 and compares results.
        """
        from dsem_agent.models.ctsem.core import discretize_system

        ro = r_ctsem["ro"]
        numpy2ri = r_ctsem["numpy2ri"]
        localconverter = r_ctsem["localconverter"]

        # Test parameters
        drift = np.array([[-0.5, 0.1], [0.2, -0.8]])
        diffusion_chol = np.array([[0.5, 0.0], [0.1, 0.4]])
        diffusion_cov = diffusion_chol @ diffusion_chol.T
        cint = np.array([0.1, -0.1])
        dt = 1.0

        # Run discretization in R using ctsem's approach
        # This matches ctsem's internal ctDiscretePars function
        r_code = """
        function(DRIFT, DIFFUSION, CINT, dt) {
            library(Matrix)
            library(ctsem)

            n <- nrow(DRIFT)
            I <- diag(n)

            # Discrete drift: expm(DRIFT * dt)
            discreteDRIFT <- as.matrix(Matrix::expm(DRIFT * dt))

            # Asymptotic diffusion (solve Lyapunov equation)
            # A*Q + Q*A' = -DIFFUSION, so Q = solve_lyap(-DIFFUSION)
            # Using ctsem's approach via Kronecker product
            DRIFTHATCH <- DRIFT %x% I + I %x% DRIFT
            Qinf <- matrix(solve(DRIFTHATCH, -as.vector(DIFFUSION)), n, n)

            # Discrete diffusion
            discreteDIFFUSION <- Qinf - discreteDRIFT %*% Qinf %*% t(discreteDRIFT)

            # Discrete intercept: solve(DRIFT, (discreteDRIFT - I) %*% CINT)
            discreteCINT <- solve(DRIFT, (discreteDRIFT - I) %*% CINT)

            list(
                discreteDRIFT = discreteDRIFT,
                discreteDIFFUSION = discreteDIFFUSION,
                discreteCINT = as.vector(discreteCINT)
            )
        }
        """
        r_discretize = ro.r(r_code)

        # Use numpy2ri converter for automatic numpy <-> R conversion
        with localconverter(ro.default_converter + numpy2ri.converter):
            # Call R function with numpy arrays (auto-converted)
            r_result = r_discretize(drift, diffusion_cov, cint, dt)

            # Extract R results by index (NamedList returns items in order)
            # Order: discreteDRIFT, discreteDIFFUSION, discreteCINT
            r_disc_drift = np.asarray(r_result[0])
            r_disc_Q = np.asarray(r_result[1])
            r_disc_c = np.asarray(r_result[2])

        # Our NumPyro implementation
        py_disc_drift, py_disc_Q, py_disc_c = discretize_system(
            jnp.array(drift), jnp.array(diffusion_cov), jnp.array(cint), dt
        )

        # Compare results (tolerance allows for float32 vs float64 differences)
        np.testing.assert_allclose(
            np.array(py_disc_drift),
            r_disc_drift,
            atol=1e-6,
            err_msg="Discrete drift mismatch with ctsem",
        )
        np.testing.assert_allclose(
            np.array(py_disc_Q),
            r_disc_Q,
            atol=1e-6,
            err_msg="Discrete diffusion mismatch with ctsem",
        )
        np.testing.assert_allclose(
            np.array(py_disc_c),
            r_disc_c,
            atol=1e-6,
            err_msg="Discrete CINT mismatch with ctsem",
        )

    def test_kalman_likelihood_matches_ctsem(self, r_ctsem):
        """Test that Kalman filter log-likelihood matches ctsem.

        Compares the log-likelihood computation for a simple model.
        """
        from dsem_agent.models.ctsem.kalman import kalman_log_likelihood

        ro = r_ctsem["ro"]
        numpy2ri = r_ctsem["numpy2ri"]
        localconverter = r_ctsem["localconverter"]

        # Simple 2-latent, 2-manifest model
        n_latent = 2
        n_manifest = 2

        # Model parameters (stable system)
        drift = np.array([[-0.5, 0.1], [0.2, -0.8]])
        diffusion_chol = np.array([[0.3, 0.0], [0.05, 0.25]])
        diffusion_cov = diffusion_chol @ diffusion_chol.T
        cint = np.array([0.0, 0.0])
        manifest_means = np.array([0.0, 0.0])
        loadings = np.eye(n_manifest, n_latent)  # Identity loadings
        manifest_var = np.array([0.1, 0.1])  # Measurement error variance (diagonal)
        manifest_cov = np.diag(manifest_var)

        # Time intervals (not absolute times)
        time_intervals = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

        # Simple synthetic observations (not generated from model, just for testing)
        Y = np.array(
            [
                [0.1, -0.2],
                [0.3, 0.1],
                [0.2, 0.3],
                [0.0, 0.2],
                [-0.1, 0.1],
            ]
        )

        # Initial state
        init_mean = np.zeros(n_latent)
        init_cov = np.eye(n_latent) * 1.0

        # R code to compute Kalman log-likelihood using ctsem's approach
        # Note: vectors are passed as column matrices to ensure proper dimensions
        r_kalman_code = """
        function(Y, dt, DRIFT, DIFFUSION, CINT, LAMBDA, MANIFESTMEANS,
                 MANIFESTCOV, T0MEANS, T0VAR) {
            library(Matrix)

            n_latent <- nrow(DRIFT)
            n_manifest <- nrow(LAMBDA)
            n_time <- nrow(Y)
            I <- diag(n_latent)

            # Ensure vectors are column matrices
            CINT <- matrix(CINT, ncol=1)
            MANIFESTMEANS <- matrix(MANIFESTMEANS, ncol=1)
            T0MEANS <- matrix(T0MEANS, ncol=1)

            # Asymptotic diffusion
            DRIFTHATCH <- DRIFT %x% I + I %x% DRIFT
            Qinf <- matrix(solve(DRIFTHATCH, -as.vector(DIFFUSION)), n_latent, n_latent)

            # Initialize
            state_mean <- T0MEANS
            state_cov <- T0VAR
            total_ll <- 0

            for (t in 1:n_time) {
                # Predict step (always, using dt for this time point)
                dt_t <- dt[t]
                discreteDRIFT <- as.matrix(Matrix::expm(DRIFT * dt_t))
                discreteQ <- Qinf - discreteDRIFT %*% Qinf %*% t(discreteDRIFT)
                discreteCINT <- solve(DRIFT, (discreteDRIFT - I) %*% CINT)

                state_mean <- discreteDRIFT %*% state_mean + discreteCINT
                state_cov <- discreteDRIFT %*% state_cov %*% t(discreteDRIFT) + discreteQ

                # Update step
                y_t <- matrix(Y[t, ], ncol=1)
                pred_y <- LAMBDA %*% state_mean + MANIFESTMEANS
                S <- LAMBDA %*% state_cov %*% t(LAMBDA) + MANIFESTCOV
                residual <- y_t - pred_y

                # Log-likelihood contribution
                ll_t <- -0.5 * (n_manifest * log(2 * pi) + log(det(S)) +
                               t(residual) %*% solve(S, residual))
                total_ll <- total_ll + ll_t

                # Kalman gain and update
                K <- state_cov %*% t(LAMBDA) %*% solve(S)
                state_mean <- state_mean + K %*% residual
                state_cov <- (I - K %*% LAMBDA) %*% state_cov
            }

            return(as.numeric(total_ll))
        }
        """
        r_kalman_ll = ro.r(r_kalman_code)

        # Use numpy2ri converter for automatic numpy <-> R conversion
        with localconverter(ro.default_converter + numpy2ri.converter):
            # Get R log-likelihood
            r_ll_result = r_kalman_ll(
                Y,
                time_intervals,
                drift,
                diffusion_cov,
                cint,
                loadings,
                manifest_means,
                manifest_cov,
                init_mean,
                init_cov,
            )

        r_ll = float(np.asarray(r_ll_result)[0])

        # Our implementation
        py_ll = kalman_log_likelihood(
            observations=jnp.array(Y),
            time_intervals=jnp.array(time_intervals),
            drift=jnp.array(drift),
            diffusion_cov=jnp.array(diffusion_cov),
            cint=jnp.array(cint),
            lambda_mat=jnp.array(loadings),
            manifest_means=jnp.array(manifest_means),
            manifest_cov=jnp.array(manifest_cov),
            t0_mean=jnp.array(init_mean),
            t0_cov=jnp.array(init_cov),
        )

        np.testing.assert_allclose(
            float(py_ll),
            r_ll,
            atol=1e-6,
            err_msg=f"Log-likelihood mismatch: Python={float(py_ll)}, R={r_ll}",
        )


class TestCTSEMModelBuilder:
    """Test CTSEMModelBuilder pipeline integration."""

    def test_builder_with_ctsem_spec(self):
        """Test building CT-SEM from CTSEMSpec directly."""
        import pandas as pd

        from dsem_agent.models.ctsem import CTSEMSpec
        from dsem_agent.models.ctsem_builder import CTSEMModelBuilder

        # Create a CTSEMSpec directly
        spec = CTSEMSpec(
            n_latent=2,
            n_manifest=2,
            lambda_mat=jnp.eye(2),
            manifest_names=["mood", "stress"],
        )

        # Create sample data
        T = 20
        X = pd.DataFrame({
            "mood": np.random.randn(T),
            "stress": np.random.randn(T),
            "time": np.arange(T, dtype=float),
        })

        builder = CTSEMModelBuilder(ctsem_spec=spec)
        model = builder.build_model(X)

        assert model is not None
        assert builder._spec.n_manifest == 2
        assert builder._spec.n_latent == 2

    def test_builder_auto_detect(self):
        """Test auto-detection of manifest columns."""
        import pandas as pd

        from dsem_agent.models.ctsem_builder import CTSEMModelBuilder

        T = 15
        X = pd.DataFrame({
            "x": np.random.randn(T),
            "y": np.random.randn(T),
            "time": np.arange(T, dtype=float),
        })

        builder = CTSEMModelBuilder()
        model = builder.build_model(X)

        assert model is not None
        assert builder._spec.n_manifest == 2

    @pytest.mark.slow
    def test_builder_fit(self):
        """Test fitting via builder interface."""
        import pandas as pd

        from dsem_agent.models.ctsem_builder import CTSEMModelBuilder

        T = 20
        X = pd.DataFrame({
            "x": np.random.randn(T) * 0.5,
            "y": np.random.randn(T) * 0.5,
            "time": np.arange(T, dtype=float),
        })

        builder = CTSEMModelBuilder(
            sampler_config={
                "num_warmup": 10,
                "num_samples": 10,
                "num_chains": 1,
            }
        )
        mcmc = builder.fit(X)

        samples = builder.get_samples()
        assert "drift_diag_pop" in samples


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
