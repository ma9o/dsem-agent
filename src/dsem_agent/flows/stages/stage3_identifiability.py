"""Stage 3: Identifiability & Sensitivity Analysis.

Uses DoWhy to check if target causal effects are identifiable.
Runs Cinelli-Hazlett sensitivity analysis if unidentified.
"""

from typing import Any

from prefect import task
from prefect.cache_policies import INPUTS


@task(cache_policy=INPUTS)
def check_identifiability(dag: dict, target_effects: list[str]) -> bool:
    """Run DoWhy identifiability checks.

    TODO: Implement using DoWhy's identify_effect().
    """
    pass


@task
def run_sensitivity_analysis(dag: dict, naive_model: Any) -> dict:
    """Cinelli-Hazlett sensitivity analysis if unidentifiable.

    TODO: Implement sensitivity analysis for unobserved confounders.
    """
    pass
