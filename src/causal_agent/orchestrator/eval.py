"""Inspect AI evaluation for the orchestrator DSEM structure proposals.

Evaluates multiple LLMs on their ability to propose valid DSEM structures
given a causal question and sample data chunks.

Usage:
    inspect eval src/causal_agent/orchestrator/eval.py --model openrouter/google/gemini-2.0-flash-001
    inspect eval src/causal_agent/orchestrator/eval.py --model openrouter/anthropic/claude-sonnet-4

Run multiple models:
    inspect eval src/causal_agent/orchestrator/eval.py --model openrouter/google/gemini-2.5-flash
    inspect eval src/causal_agent/orchestrator/eval.py --model openrouter/anthropic/claude-sonnet-4
    inspect eval src/causal_agent/orchestrator/eval.py --model openrouter/openai/gpt-4o
"""

import json
import re
from dataclasses import dataclass

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState, generate, system_message

from causal_agent.orchestrator.prompts import (
    STRUCTURE_PROPOSER_SYSTEM,
    STRUCTURE_PROPOSER_USER,
)
from causal_agent.orchestrator.scoring import _count_rule_points
from causal_agent.orchestrator.schemas import DSEMStructure
from causal_agent.utils.data import (
    PROCESSED_DIR,
    load_text_chunks,
)

# Files to exclude when finding the latest data file (script outputs)
EXCLUDE_FILES = {"orchestrator-samples-manual.txt"}


def get_latest_data_file() -> str:
    """Find the most recent preprocessed data file, excluding script outputs."""
    txt_files = [
        f for f in PROCESSED_DIR.glob("*.txt")
        if f.name not in EXCLUDE_FILES
    ]
    if not txt_files:
        raise FileNotFoundError(f"No data files found in {PROCESSED_DIR}")
    return max(txt_files, key=lambda p: p.stat().st_mtime)


# Default models to compare (via OpenRouter)
DEFAULT_MODELS = [
    "openrouter/google/gemini-2.5-pro-preview-06-05",
    "openrouter/anthropic/claude-sonnet-4",
    "openrouter/openai/gpt-4o",
]


@dataclass
class EvalQuestion:
    """An evaluation question with metadata."""

    id: int
    question: str
    difficulty: float
    domain: str
    primary_challenge: str


def load_eval_questions() -> list[EvalQuestion]:
    """Load evaluation questions from the JSON file."""
    eval_file = PROCESSED_DIR.parent / "eval" / "orchestrator_eval_questions.json"

    with open(eval_file) as f:
        data = json.load(f)

    return [
        EvalQuestion(
            id=q["id"],
            question=q["question"],
            difficulty=q["total_difficulty"],
            domain=q["domain"],
            primary_challenge=q["primary_challenge"],
        )
        for q in data["evaluation_questions"]
    ]


def sample_chunks_evenly(input_file, n: int, seed: int) -> list[str]:
    """Sample n chunks evenly spaced across the input file with jitter."""
    import random

    chunks = load_text_chunks(input_file)
    random.seed(seed)
    n = min(n, len(chunks))

    if n >= len(chunks):
        return chunks

    segment_size = len(chunks) / n
    sampled = []
    for i in range(n):
        segment_start = int(i * segment_size)
        segment_end = int((i + 1) * segment_size)
        idx = random.randint(segment_start, segment_end - 1)
        sampled.append(chunks[idx])

    return sampled


def format_chunks(chunks: list[str]) -> str:
    """Format chunks for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks):
        parts.append(f"--- CHUNK {i + 1} ---\n{chunk}")
    return "\n\n".join(parts)


def create_eval_dataset(
    n_chunks: int = 5,
    seed: int = 42,
    input_file: str | None = None,
) -> MemoryDataset:
    """Create evaluation dataset by combining questions with sampled chunks.

    Args:
        n_chunks: Number of chunks to sample per question
        seed: Random seed for reproducible chunk sampling
        input_file: Specific input file name, or None for latest

    Returns:
        MemoryDataset with samples for each question
    """
    questions = load_eval_questions()

    # Resolve input file
    if input_file:
        data_file = PROCESSED_DIR / input_file
        if not data_file.exists():
            raise FileNotFoundError(f"File not found: {data_file}")
    else:
        data_file = get_latest_data_file()

    # Sample chunks (same for all questions for fair comparison)
    chunks = sample_chunks_evenly(data_file, n_chunks, seed)
    formatted_chunks = format_chunks(chunks)

    samples = []
    for q in questions:
        # Build the user prompt
        user_prompt = STRUCTURE_PROPOSER_USER.format(
            question=q.question,
            dataset_summary=f"Source: {data_file.name}",
            chunks=formatted_chunks,
        )

        samples.append(
            Sample(
                input=user_prompt,
                id=f"q{q.id}",
                metadata={
                    "question": q.question,
                    "difficulty": q.difficulty,
                    "domain": q.domain,
                    "primary_challenge": q.primary_challenge,
                    "n_chunks": n_chunks,
                    "seed": seed,
                },
            )
        )

    return MemoryDataset(samples)


def extract_json_from_response(text: str) -> str | None:
    """Extract JSON from model response, handling markdown code blocks."""
    # Try to find JSON in code blocks first
    code_block_pattern = r"```(?:json)?\s*\n?([\s\S]*?)\n?```"
    matches = re.findall(code_block_pattern, text)

    for match in matches:
        try:
            json.loads(match.strip())
            return match.strip()
        except json.JSONDecodeError:
            continue

    # Try to find raw JSON object
    brace_pattern = r"\{[\s\S]*\}"
    matches = re.findall(brace_pattern, text)

    for match in matches:
        try:
            json.loads(match)
            return match
        except json.JSONDecodeError:
            continue

    return None


@scorer(metrics=[accuracy(), stderr()])
def dsem_structure_scorer():
    """Score DSEM structure proposals using the validation and point system.

    Returns:
        - "C" (correct) if structure is valid (parses and validates)
        - "I" (incorrect) if structure is invalid
        - Numeric score (points) stored in metadata for detailed analysis
    """

    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion

        # Extract JSON from response
        json_str = extract_json_from_response(completion)
        if json_str is None:
            return Score(
                value="I",
                answer="[No valid JSON found]",
                explanation="Could not extract JSON from model response",
                metadata={"points": 0.0, "error": "no_json"},
            )

        # Parse JSON
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            return Score(
                value="I",
                answer=json_str[:200] + "..." if len(json_str) > 200 else json_str,
                explanation=f"JSON parse error: {e}",
                metadata={"points": 0.0, "error": "json_parse"},
            )

        # Validate against schema
        try:
            structure = DSEMStructure(**data)
        except Exception as e:
            return Score(
                value="I",
                answer=json_str[:200] + "..." if len(json_str) > 200 else json_str,
                explanation=f"Schema validation error: {e}",
                metadata={"points": 0.0, "error": "schema_validation"},
            )

        # Count points for valid structure
        points = _count_rule_points(structure)

        return Score(
            value="C",
            answer=json_str[:500] + "..." if len(json_str) > 500 else json_str,
            explanation=f"Valid structure with {len(structure.dimensions)} dimensions, {len(structure.edges)} edges",
            metadata={
                "points": points,
                "n_dimensions": len(structure.dimensions),
                "n_edges": len(structure.edges),
            },
        )

    return score


@task
def orchestrator_eval(
    n_chunks: int = 5,
    seed: int = 42,
    input_file: str | None = None,
):
    """Evaluate LLM ability to propose DSEM structures.

    Args:
        n_chunks: Number of data chunks to include in each sample
        seed: Random seed for chunk sampling (reproducibility)
        input_file: Specific preprocessed file name, or None for latest
    """
    return Task(
        dataset=create_eval_dataset(n_chunks=n_chunks, seed=seed, input_file=input_file),
        solver=[
            system_message(STRUCTURE_PROPOSER_SYSTEM),
            generate(),
        ],
        scorer=dsem_structure_scorer(),
    )
