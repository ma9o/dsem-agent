"""Stage 2: Indicator Extraction (Workers).

Workers process chunks in parallel to extract raw indicator values.
Each worker returns a Polars DataFrame with (indicator, value, timestamp) tuples.

This is the "E" (Extract) in ETL. Transformation (aggregation) happens in Stage 3.
"""

import asyncio
import logging

from prefect import task
from prefect.cache_policies import INPUTS

from causal_ssm_agent.utils.config import get_config
from causal_ssm_agent.utils.data import chunk_lines, get_worker_chunk_size
from causal_ssm_agent.workers.agents import WorkerResult, process_chunk_async

logger = logging.getLogger(__name__)


@task(cache_policy=INPUTS, result_serializer="json")
def load_worker_chunks(lines: list[str]) -> list[str]:
    """Group preprocessed lines into worker-sized chunks."""
    return chunk_lines(lines, chunk_size=get_worker_chunk_size())


async def populate_all_indicators(
    chunks: list[str], question: str, causal_spec: dict
) -> list[WorkerResult | None]:
    """Process all worker chunks using asyncio.gather with concurrency control.

    This is a plain async function (not a Prefect task) to avoid the overhead of
    Prefect serializing 1234 chunks for task run metadata. Concurrency is managed
    via asyncio.Semaphore.

    Returns:
        List parallel to ``chunks`` — WorkerResult on success, None on failure.
    """
    import sys
    print(f"[stage2] Starting populate_all_indicators with {len(chunks)} chunks", flush=True)
    sys.stdout.flush()
    config = get_config()
    sem = asyncio.Semaphore(config.stage2_workers.max_concurrent)
    n = len(chunks)
    completed = 0

    async def process_one(i: int, chunk: str) -> WorkerResult | None:
        nonlocal completed
        async with sem:
            try:
                if i < 3:
                    print(f"[stage2] Worker {i} starting", flush=True)
                result = await process_chunk_async(chunk, question, causal_spec)
                completed += 1
                if completed % 10 == 0 or completed == n:
                    print(f"[stage2] Workers: {completed}/{n} completed", flush=True)
                return result
            except Exception as exc:
                completed += 1
                print(f"[stage2] Chunk {i}/{n} failed: {exc}", flush=True)
                return None

    print(f"[stage2] Creating {n} coroutines", flush=True)
    tasks = [process_one(i, chunk) for i, chunk in enumerate(chunks)]
    print(f"[stage2] Starting asyncio.gather", flush=True)
    result = await asyncio.gather(*tasks)
    print(f"[stage2] asyncio.gather complete", flush=True)
    return result
