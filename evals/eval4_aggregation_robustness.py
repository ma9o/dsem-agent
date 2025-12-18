"""Inspect AI evaluation for aggregation function robustness.

Tests the aggregate_worker_measurements function on N different sets of M chunks
processed by workers from gemini-3-flash. Scores 1 if aggregation completes
successfully, 0 if it raises an exception.

This eval verifies that the aggregation pipeline can handle the diverse outputs
produced by worker LLMs without breaking.

Usage:
    inspect eval evals/eval4_aggregation_robustness.py --model google/vertex/gemini-3-flash-preview
    inspect eval evals/eval4_aggregation_robustness.py -T n_sets=10 -T chunks_per_set=5
"""

import sys
from pathlib import Path

# Add project root to path for evals.common import
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import json
import traceback

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model
from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import Generate, TaskState, solver

from causal_agent.utils.aggregations import aggregate_worker_measurements
from causal_agent.utils.llm import get_generate_config, make_worker_tools, multi_turn_generate, parse_json_response
from causal_agent.workers.prompts import WORKER_SYSTEM, WORKER_USER
from causal_agent.workers.agents import (
    _format_dimensions,
    _get_outcome_description,
)
from causal_agent.workers.schemas import WorkerOutput

from evals.common import (
    get_sample_chunks_worker,
    load_example_dag,
)


# Default question (matches example_dag.json)
DEFAULT_QUESTION = "I want to sleep better"


def create_eval_dataset(
    n_sets: int = 10,
    chunks_per_set: int = 5,
    seed: int = 42,
    input_file: str | None = None,
    question: str = DEFAULT_QUESTION,
) -> MemoryDataset:
    """Create evaluation dataset with chunk sets for aggregation testing.

    Each sample is a set of chunks that will be processed by workers and then
    aggregated together. The eval tests whether aggregation succeeds or fails.

    Args:
        n_sets: Number of chunk sets (each becomes one aggregation test)
        chunks_per_set: Number of chunks per set (M workers processing M chunks)
        seed: Random seed for reproducible chunk sampling
        input_file: Specific input file name, or None for latest
        question: The causal question to use

    Returns:
        MemoryDataset with one sample per chunk set
    """
    schema = load_example_dag()
    dimensions_text = _format_dimensions(schema)
    outcome_description = _get_outcome_description(schema)

    # Get all chunks needed
    total_chunks = n_sets * chunks_per_set
    all_chunks = get_sample_chunks_worker(total_chunks, seed, input_file)

    samples = []
    for set_idx in range(n_sets):
        # Get chunks for this set
        start_idx = set_idx * chunks_per_set
        end_idx = start_idx + chunks_per_set
        chunk_set = all_chunks[start_idx:end_idx]

        # Store metadata for each chunk's prompt
        chunk_prompts = []
        for i, chunk in enumerate(chunk_set):
            user_prompt = WORKER_USER.format(
                question=question,
                outcome_description=outcome_description,
                dimensions=dimensions_text,
                chunk=chunk,
            )
            chunk_prompts.append(user_prompt)

        samples.append(
            Sample(
                input=f"Aggregation test set {set_idx + 1} with {len(chunk_set)} chunks",
                id=f"set_{set_idx:04d}",
                metadata={
                    "set_index": set_idx,
                    "chunk_prompts": chunk_prompts,
                    "chunks": chunk_set,
                    "question": question,
                },
            )
        )

    return MemoryDataset(samples)


async def generate_worker_output(
    model_id: str,
    chunk: str,
    question: str,
    schema: dict,
) -> str:
    """Generate worker output for a single chunk.

    Returns the raw completion text (including JSON).
    """
    model = get_model(model_id)

    dimensions_text = _format_dimensions(schema)
    outcome_description = _get_outcome_description(schema)

    messages = [
        ChatMessageSystem(content=WORKER_SYSTEM),
        ChatMessageUser(
            content=WORKER_USER.format(
                question=question,
                outcome_description=outcome_description,
                dimensions=dimensions_text,
                chunk=chunk,
            )
        ),
    ]

    config = get_generate_config()

    completion = await multi_turn_generate(
        messages=messages,
        model=model,
        tools=make_worker_tools(schema),
        config=config,
    )

    return completion


def aggregation_solver(worker_timeout: float = 300):
    """Solver that generates worker outputs for all chunks and runs aggregation.

    Args:
        worker_timeout: Timeout in seconds for each worker.
    """

    @solver
    def _solver():
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            schema = load_example_dag()
            question = state.metadata["question"]
            chunks = state.metadata["chunks"]

            # Get the model being evaluated (passed via --model flag)
            model = get_model()
            model_id = str(model)

            # Generate outputs from workers in parallel
            async def safe_generate(chunk: str, chunk_idx: int) -> tuple[int, str | None, str | None]:
                """Generate with error handling, returns (chunk_idx, result, error)."""
                try:
                    result = await asyncio.wait_for(
                        generate_worker_output(model_id, chunk, question, schema),
                        timeout=worker_timeout,
                    )
                    return chunk_idx, result, None
                except asyncio.TimeoutError:
                    return chunk_idx, None, f"TIMEOUT after {worker_timeout}s"
                except Exception as e:
                    return chunk_idx, None, str(e)

            results = await asyncio.gather(*[
                safe_generate(chunk, i) for i, chunk in enumerate(chunks)
            ])

            # Collect worker outputs
            worker_outputs = []
            worker_errors = []
            for chunk_idx, result, error in results:
                if error:
                    worker_errors.append(f"chunk_{chunk_idx}: {error}")
                elif result:
                    # Parse JSON and convert to DataFrame
                    try:
                        data = parse_json_response(result)
                        output = WorkerOutput.model_validate(data)
                        df = output.to_dataframe()
                        worker_outputs.append(df)
                    except Exception as e:
                        worker_errors.append(f"chunk_{chunk_idx}: parse error - {e}")

            # Store worker info in metadata
            state.metadata["n_successful_workers"] = len(worker_outputs)
            state.metadata["n_failed_workers"] = len(worker_errors)
            state.metadata["worker_errors"] = worker_errors

            # Now try aggregation
            agg_result = None
            agg_error = None

            if worker_outputs:
                try:
                    agg_result = aggregate_worker_measurements(worker_outputs, schema)
                    state.metadata["agg_keys"] = list(agg_result.keys())
                    state.metadata["agg_shapes"] = {
                        k: (df.height, df.width) for k, df in agg_result.items()
                    }
                except Exception as e:
                    agg_error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

            # Build completion message summarizing results
            completion_parts = [
                f"Workers: {len(worker_outputs)} successful, {len(worker_errors)} failed",
            ]

            if worker_errors:
                completion_parts.append(f"Worker errors: {'; '.join(worker_errors[:3])}")

            if agg_result is not None:
                completion_parts.append(f"Aggregation: SUCCESS - keys={list(agg_result.keys())}")
            elif agg_error:
                completion_parts.append(f"Aggregation: FAILED - {agg_error}")
            else:
                completion_parts.append("Aggregation: SKIPPED (no worker outputs)")

            state.output.completion = "\n".join(completion_parts)
            state.metadata["agg_error"] = agg_error
            state.metadata["agg_success"] = agg_result is not None

            return state

        return solve

    return _solver()


@scorer(metrics=[accuracy(), stderr()])
def aggregation_scorer():
    """Score aggregation robustness.

    Returns:
        - "C" (correct) if aggregation completed successfully
        - "I" (incorrect) if aggregation raised an exception
    """

    async def score(state: TaskState, target: Target) -> Score:
        agg_success = state.metadata.get("agg_success", False)
        agg_error = state.metadata.get("agg_error")
        n_successful = state.metadata.get("n_successful_workers", 0)
        n_failed = state.metadata.get("n_failed_workers", 0)
        agg_keys = state.metadata.get("agg_keys", [])
        agg_shapes = state.metadata.get("agg_shapes", {})

        if agg_success:
            # Build success explanation
            shape_info = ", ".join(f"{k}={s}" for k, s in agg_shapes.items())
            explanation = (
                f"Aggregation succeeded with {n_successful} worker outputs. "
                f"Result keys: {agg_keys}. Shapes: {shape_info}"
            )
            return Score(
                value="C",
                answer="SUCCESS",
                explanation=explanation,
                metadata={
                    "n_successful_workers": n_successful,
                    "n_failed_workers": n_failed,
                    "agg_keys": agg_keys,
                    "agg_shapes": agg_shapes,
                },
            )
        else:
            # Build failure explanation
            explanation = f"Aggregation failed with {n_successful} worker outputs."
            if agg_error:
                explanation += f"\nError: {agg_error}"
            if n_successful == 0:
                explanation += "\nNote: No successful worker outputs to aggregate."

            return Score(
                value="I",
                answer="FAILED",
                explanation=explanation,
                metadata={
                    "n_successful_workers": n_successful,
                    "n_failed_workers": n_failed,
                    "error": agg_error,
                },
            )

    return score


@task
def aggregation_robustness_eval(
    n_sets: int = 10,
    chunks_per_set: int = 5,
    seed: int = 42,
    input_file: str | None = None,
    question: str = DEFAULT_QUESTION,
    worker_timeout: int = 300,
):
    """Evaluate aggregation function robustness on worker outputs.

    Generates N sets of M chunks, processes each set through workers, and tests
    whether aggregate_worker_measurements handles the outputs without crashing.

    Args:
        n_sets: Number of aggregation tests (N)
        chunks_per_set: Chunks per test (M workers per test)
        seed: Random seed for chunk sampling
        input_file: Specific preprocessed file name, or None for latest
        question: The causal question to use
        worker_timeout: Timeout in seconds for each worker (default: 300s)
    """
    return Task(
        dataset=create_eval_dataset(
            n_sets=n_sets,
            chunks_per_set=chunks_per_set,
            seed=seed,
            input_file=input_file,
            question=question,
        ),
        solver=[
            aggregation_solver(worker_timeout=worker_timeout),
        ],
        scorer=aggregation_scorer(),
    )
