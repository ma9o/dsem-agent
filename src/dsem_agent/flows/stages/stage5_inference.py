"""Stage 5: Intervention Analysis.

Fits PyMC model and runs interventions/counterfactuals.
"""

from typing import Any

from prefect import task


@task(timeout_seconds=3600, retries=1)
def fit_model(model_spec: dict, priors: dict, data: list[str]) -> Any:
    """Fit PyMC model.

    TODO: Implement PyMC model fitting.
    """
    pass


@task
def run_interventions(fitted_model: Any, interventions: list[str]) -> list[dict]:
    """Run interventions and counterfactuals, rank by effect size.

    TODO: Implement intervention analysis and ranking.
    """
    pass
