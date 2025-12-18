"""Main causal inference pipeline.

Orchestrates all stages from structure proposal to intervention analysis.
"""

from prefect import flow
from prefect.utilities.annotations import unmapped

from causal_agent.utils.data import (
    resolve_input_path,
    load_query,
    SAMPLE_CHUNKS,
)
from .stages import (
    # Stage 1
    load_orchestrator_chunks,
    propose_structure,
    # Stage 2
    load_worker_chunks,
    populate_dimensions,
    aggregate_measurements,
    # Stage 3
    check_identifiability,
    # Stage 4
    specify_model,
    elicit_priors,
    # Stage 5
    fit_model,
    run_interventions,
)


@flow(log_prints=True)
def causal_inference_pipeline(
    query_file: str,
    target_effects: list[str],
    input_file: str | None = None,
):
    """
    Main causal inference pipeline.

    Args:
        query_file: Filename in data/queries/ (e.g., 'smoking-cancer')
        target_effects: Causal effects to estimate
        input_file: Filename in data/processed/ (default: latest file)
    """
    # Stage 0: Load question and resolve input path
    question = load_query(query_file)
    print(f"Query: {query_file}")
    print(f"Question: {question[:100]}..." if len(question) > 100 else f"Question: {question}")

    input_path = resolve_input_path(input_file)
    print(f"Using input file: {input_path.name}")

    # Stage 1: Propose structure from sample (orchestrator chunk size)
    orchestrator_chunks = load_orchestrator_chunks(input_path)
    print(f"Loaded {len(orchestrator_chunks)} orchestrator chunks")
    schema = propose_structure(question, orchestrator_chunks[:SAMPLE_CHUNKS])

    # Stage 2: Parallel dimension population (worker chunk size)
    # Each worker returns a WorkerResult with extractions as a Polars dataframe
    worker_chunks = load_worker_chunks(input_path)
    print(f"Loaded {len(worker_chunks)} worker chunks")
    worker_results = populate_dimensions.map(
        worker_chunks,
        question=unmapped(question),
        schema=unmapped(schema),
    )

    # Stage 2b: Aggregate measurements into time-series by causal_granularity
    measurements = aggregate_measurements(worker_results, schema)
    for granularity, df in measurements.items():
        n_dims = len([c for c in df.columns if c != "time_bucket"])
        if granularity == "time_invariant":
            print(f"  {granularity}: {n_dims} dimensions")
        else:
            print(f"  {granularity}: {df.height} time points × {n_dims} dimensions")

    # TODO: Stage 2c - Merge proposed dimensions from workers

    # Stage 3: Identifiability
    identifiable = check_identifiability(schema["dag"], target_effects)
    # TODO: conditional logic for sensitivity analysis

    # Stage 4: Model specification
    model_spec = specify_model(schema["dag"], schema)
    priors = elicit_priors(model_spec)

    # Stage 5: Fit and intervene
    fitted = fit_model(model_spec, priors, worker_chunks)
    results = run_interventions(fitted, target_effects)

    return results


if __name__ == "__main__":
    # Serve the flow for UI access
    causal_inference_pipeline.serve(
        name="causal-inference",
        tags=["causal", "llm"],
    )
