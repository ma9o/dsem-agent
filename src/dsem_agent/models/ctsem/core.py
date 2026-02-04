"""CT-SEM core matrix utilities (stub).

Implementation will be merged from numpyro-ctsem branch.
"""

import jax.numpy as jnp


def solve_lyapunov(A: jnp.ndarray, Q: jnp.ndarray) -> jnp.ndarray:
    """Solve the continuous Lyapunov equation A @ X + X @ A.T + Q = 0."""
    raise NotImplementedError("Will be merged from numpyro-ctsem")


def compute_asymptotic_diffusion(
    drift: jnp.ndarray, diffusion_cov: jnp.ndarray
) -> jnp.ndarray:
    """Compute asymptotic diffusion covariance."""
    raise NotImplementedError("Will be merged from numpyro-ctsem")


def compute_discrete_diffusion(
    drift: jnp.ndarray, diffusion_cov: jnp.ndarray, dt: float
) -> jnp.ndarray:
    """Compute discrete-time diffusion covariance Q_d."""
    raise NotImplementedError("Will be merged from numpyro-ctsem")


def compute_discrete_cint(
    drift: jnp.ndarray, cint: jnp.ndarray, dt: float
) -> jnp.ndarray:
    """Compute discrete-time intercept c_d."""
    raise NotImplementedError("Will be merged from numpyro-ctsem")


def discretize_system(
    drift: jnp.ndarray,
    diffusion_cov: jnp.ndarray,
    cint: jnp.ndarray | None,
    dt: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
    """Discretize continuous-time system for a given time interval."""
    raise NotImplementedError("Will be merged from numpyro-ctsem")


def matrix_fraction_decomposition(
    drift: jnp.ndarray, diffusion_cov: jnp.ndarray, dt: float
) -> jnp.ndarray:
    """Compute discrete diffusion using matrix fraction decomposition."""
    raise NotImplementedError("Will be merged from numpyro-ctsem")
