"""Stage 4: Model Specification & Prior Elicitation.

The orchestrator specifies GLMs in PyMC, then workers provide priors.
"""

from prefect import task
from prefect.cache_policies import INPUTS


@task(retries=1, cache_policy=INPUTS)
def specify_model(dag: dict, schema: dict) -> dict:
    """Orchestrator specifies GLM in PyMC.

    TODO: Implement model specification based on DAG structure and dtypes.
    """
    pass


@task(retries=2, retry_delay_seconds=10, task_run_name="elicit-priors")
def elicit_priors(model_spec: dict) -> dict:
    """Workers provide priors for model parameters.

    TODO: Implement prior elicitation using stage4 model from config.
    """
    pass
