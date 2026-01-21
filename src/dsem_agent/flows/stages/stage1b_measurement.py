"""Stage 1b: Measurement Model Proposal (Orchestrator).

The orchestrator proposes indicators to operationalize the theoretical
constructs from Stage 1a, using sample data to inform operationalization.

This follows the Anderson & Gerbing (1988) two-step approach where the
measurement model is specified after the latent model.
"""

from pathlib import Path

from prefect import task
from prefect.cache_policies import INPUTS

from dsem_agent.orchestrator.agents import (
    build_dsem_model as build_dsem_model_agent,
    propose_measurement_model as propose_measurement_model_agent,
)
from dsem_agent.utils.data import (
    get_orchestrator_chunk_size,
    load_text_chunks as load_text_chunks_util,
)


@task(cache_policy=INPUTS)
def load_orchestrator_chunks(input_path: Path) -> list[str]:
    """Load chunks sized for orchestrator (stage 1b)."""
    return load_text_chunks_util(input_path, chunk_size=get_orchestrator_chunk_size())


@task(retries=2, retry_delay_seconds=30, cache_policy=INPUTS)
def propose_measurement_model(
    question: str,
    latent_model: dict,
    data_sample: list[str],
    dataset_summary: str = "",
) -> dict:
    """Orchestrator proposes indicators to operationalize constructs.

    This is Stage 1b - seeing data to inform operationalization.

    Args:
        question: The causal research question
        latent_model: The latent model dict from Stage 1a
        data_sample: Sample chunks from the dataset
        dataset_summary: Brief overview of the full dataset

    Returns:
        MeasurementModel as a dictionary with 'indicators'
    """
    return propose_measurement_model_agent(question, latent_model, data_sample, dataset_summary)


@task(cache_policy=INPUTS)
def build_dsem_model(latent_model: dict, measurement_model: dict) -> dict:
    """Combine latent and measurement models into full DSEMModel.

    Args:
        latent_model: The latent model dict from Stage 1a
        measurement_model: The measurement model dict from Stage 1b

    Returns:
        DSEMModel as a dictionary with 'latent' and 'measurement'
    """
    return build_dsem_model_agent(latent_model, measurement_model)
