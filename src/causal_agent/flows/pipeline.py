from prefect import flow, task
from prefect.cache_policies import INPUTS
from typing import Any


@task(retries=2, retry_delay_seconds=30, cache_policy=INPUTS)
def propose_structure(data_sample: list[Any]) -> dict:
    """Stage 1: Orchestrator proposes dimensions, autocorrelations, time granularities, DAG."""
    pass


@task(
    retries=2,
    retry_delay_seconds=10,
    task_run_name="populate-chunk-{chunk_id}",
)
def populate_dimensions(chunk: Any, chunk_id: int, schema: dict) -> dict:
    """Stage 2: Workers populate candidate dimensions for each chunk."""
    pass


@task(retries=1, cache_policy=INPUTS)
def merge_suggestions(base_schema: dict, worker_suggestions: list[dict]) -> dict:
    """Stage 2b: Orchestrator performs 3-way merge and backfill."""
    pass


@task(cache_policy=INPUTS)
def check_identifiability(dag: dict, target_effects: list[str]) -> bool:
    """Stage 3: Run DoWhy identifiability checks."""
    pass


@task
def run_sensitivity_analysis(dag: dict, naive_model: Any) -> dict:
    """Stage 3b: Cinelli-Hazlett sensitivity analysis if unidentifiable."""
    pass


@task(retries=1, cache_policy=INPUTS)
def specify_model(dag: dict, schema: dict) -> dict:
    """Stage 4: Orchestrator specifies GLM in PyMC."""
    pass


@task(retries=2, retry_delay_seconds=10, task_run_name="elicit-priors")
def elicit_priors(model_spec: dict) -> dict:
    """Stage 4b: Workers provide priors."""
    pass


@task(timeout_seconds=3600, retries=1)
def fit_model(model_spec: dict, priors: dict, data: list[Any]) -> Any:
    """Stage 5: Fit PyMC model."""
    pass


@task
def run_interventions(fitted_model: Any, interventions: list[str]) -> list[dict]:
    """Stage 5b: Run interventions and counterfactuals, rank by effect size."""
    pass


@flow(log_prints=True)
def causal_inference_pipeline(data_chunks: list[Any], target_effects: list[str]):
    """Main 5-stage causal inference pipeline."""
    # Stage 1: Propose structure from sample
    schema = propose_structure(data_chunks[:3])

    # Stage 2: Parallel dimension population
    worker_results = populate_dimensions.map(
        data_chunks,
        chunk_id=list(range(len(data_chunks))),
        schema=schema,
    )
    schema = merge_suggestions(schema, worker_results)

    # Stage 3: Identifiability
    identifiable = check_identifiability(schema["dag"], target_effects)
    # TODO: conditional logic for sensitivity analysis

    # Stage 4: Model specification
    model_spec = specify_model(schema["dag"], schema)
    priors = elicit_priors(model_spec)

    # Stage 5: Fit and intervene
    fitted = fit_model(model_spec, priors, data_chunks)
    results = run_interventions(fitted, target_effects)

    return results
