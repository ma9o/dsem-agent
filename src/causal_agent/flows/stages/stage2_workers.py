"""Stage 2: Dimension Population (Workers).

Workers process chunks in parallel to extract dimension values.
Each worker returns a validated Polars dataframe of extractions.
"""

from pathlib import Path

import polars as pl
from prefect import task
from prefect.cache_policies import INPUTS

from causal_agent.utils.aggregations import aggregate_worker_measurements
from causal_agent.utils.data import (
    load_text_chunks as load_text_chunks_util,
    get_worker_chunk_size,
)
from causal_agent.workers.agents import process_chunk, WorkerResult


@task(cache_policy=INPUTS)
def load_worker_chunks(input_path: Path) -> list[str]:
    """Load chunks sized for workers (stage 2)."""
    return load_text_chunks_util(input_path, chunk_size=get_worker_chunk_size())


@task(
    retries=2,
    retry_delay_seconds=10,
)
def populate_dimensions(chunk: str, question: str, schema: dict) -> WorkerResult:
    """Worker extracts dimension values from a chunk.

    Returns:
        WorkerResult containing:
        - output: Validated WorkerOutput with extractions and proposed dimensions
        - dataframe: Polars DataFrame with columns (dimension, value, timestamp)
    """
    return process_chunk(chunk, question, schema)


@task
def aggregate_measurements(
    worker_results: list[WorkerResult],
    schema: dict,
) -> dict[str, pl.DataFrame]:
    """Aggregate worker measurements into time-series DataFrames by granularity.

    Combines all worker extractions and aggregates to causal_granularity:
    1. Concatenates worker DataFrames (dimension, value, timestamp)
    2. Groups dimensions by their causal_granularity
    3. Parses timestamps and buckets to each granularity
    4. Applies dimension-specific aggregation (mean, sum, max, etc.)
    5. Returns one DataFrame per granularity

    Args:
        worker_results: List of WorkerResults from parallel workers
        schema: DSEM schema dict with dimension definitions

    Returns:
        Dict mapping granularity -> DataFrame. Each DataFrame has 'time_bucket'
        column and dimension columns. Time-invariant dimensions are in
        'time_invariant' key as a single-row DataFrame.
    """
    dataframes = [wr.dataframe for wr in worker_results]
    return aggregate_worker_measurements(dataframes, schema)
