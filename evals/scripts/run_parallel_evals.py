#!/usr/bin/env python
"""Run evals across all models in parallel.

Usage:
    # Orchestrator eval (default)
    uv run python evals/scripts/run_parallel_evals.py
    uv run python evals/scripts/run_parallel_evals.py --models claude gemini

    # Worker eval
    uv run python evals/scripts/run_parallel_evals.py --eval worker
    uv run python evals/scripts/run_parallel_evals.py --eval worker --models gemini-flash claude-sonnet

    # Common options
    uv run python evals/scripts/run_parallel_evals.py -n 10 --seed 123
    uv run python evals/scripts/run_parallel_evals.py -v  # Verbose output
"""

import argparse
import asyncio
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Track running processes for cleanup on interrupt
_running_procs: list[asyncio.subprocess.Process] = []

# Import model registries from eval modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from orchestrator_structure import MODELS as ORCHESTRATOR_MODELS
from worker_extraction import MODELS as WORKER_MODELS

# Eval configurations
EVAL_CONFIGS = {
    "orchestrator": {
        "file": "evals/orchestrator_structure.py",
        "models": ORCHESTRATOR_MODELS,
        "task_params": ["n_chunks", "seed", "input_file"],
    },
    "worker": {
        "file": "evals/worker_extraction.py",
        "models": WORKER_MODELS,
        "task_params": ["n_chunks", "seed", "input_file", "question"],
    },
}


@dataclass
class EvalResult:
    model: str
    success: bool
    duration: float
    output: str
    mean_score: float | None = None
    stderr: float | None = None
    log_path: Path | None = None


def parse_results(output: str) -> tuple[float | None, float | None, Path | None]:
    """Extract mean, stderr, and log path from inspect output."""
    mean_score = None
    stderr = None
    log_path = None

    for line in output.split("\n"):
        line = line.strip()
        if line.startswith("mean"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    mean_score = float(parts[1])
                except ValueError:
                    pass
        elif line.startswith("stderr"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    stderr = float(parts[1])
                except ValueError:
                    pass
        elif line.startswith("logs/") and line.endswith(".eval"):
            log_path = Path(line)

    return mean_score, stderr, log_path


def short_model_name(model: str, models_dict: dict) -> str:
    """Get short display name for model."""
    return models_dict.get(model, model.split("/")[-1])


async def run_eval(
    model: str,
    eval_file: str,
    models_dict: dict,
    task_params: dict,
) -> EvalResult:
    """Run a single eval and return results."""
    short_name = short_model_name(model, models_dict)
    print(f"[{short_name}] Starting...", file=sys.stderr)

    cmd = ["uv", "run", "inspect", "eval", eval_file, "--model", model]
    for key, value in task_params.items():
        if value is not None:
            cmd.extend(["-T", f"{key}={value}"])

    start = time.time()
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    _running_procs.append(proc)

    try:
        stdout, _ = await proc.communicate()
    finally:
        if proc in _running_procs:
            _running_procs.remove(proc)
    duration = time.time() - start
    output = stdout.decode()

    success = proc.returncode == 0
    mean_score, stderr, log_path = parse_results(output) if success else (None, None, None)

    status = "done" if success else "FAILED"
    score_str = f" (mean: {mean_score:.1f})" if mean_score is not None else ""
    print(f"[{short_name}] {status} in {duration:.0f}s{score_str}", file=sys.stderr)

    return EvalResult(
        model=model,
        success=success,
        duration=duration,
        output=output,
        mean_score=mean_score,
        stderr=stderr,
        log_path=log_path,
    )


async def run_all_evals(
    models: list[str],
    eval_file: str,
    models_dict: dict,
    task_params: dict,
) -> list[EvalResult]:
    """Run all evals in parallel."""
    tasks = [
        run_eval(model, eval_file, models_dict, task_params)
        for model in models
    ]
    return await asyncio.gather(*tasks)


def print_summary(results: list[EvalResult], models_dict: dict) -> list[EvalResult]:
    """Print summary table of results. Returns sorted results."""
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    # Sort by mean score descending (failures at bottom)
    sorted_results = sorted(
        results,
        key=lambda r: (r.mean_score is not None, r.mean_score or 0),
        reverse=True,
    )

    print(f"{'Model':<15} {'Score':>10} {'Stderr':>10} {'Time':>10} {'Status':<10}")
    print("-" * 60)

    for r in sorted_results:
        name = short_model_name(r.model, models_dict)
        score = f"{r.mean_score:.1f}" if r.mean_score is not None else "-"
        stderr = f"{r.stderr:.2f}" if r.stderr is not None else "-"
        time_str = f"{r.duration:.0f}s"
        status = "OK" if r.success else "FAILED"
        print(f"{name:<15} {score:>10} {stderr:>10} {time_str:>10} {status:<10}")

    print("=" * 60)
    return sorted_results


def cleanup_subprocesses():
    """Kill all running subprocess evals."""
    for proc in _running_procs:
        if proc.returncode is None:  # Still running
            print(f"\nKilling subprocess {proc.pid}...", file=sys.stderr)
            proc.terminate()
    # Give them a moment to terminate gracefully
    time.sleep(0.5)
    for proc in _running_procs:
        if proc.returncode is None:
            proc.kill()


def main():
    # Set up signal handler to kill subprocesses on Ctrl+C
    def signal_handler(sig, frame):
        print("\nInterrupted! Cleaning up...", file=sys.stderr)
        cleanup_subprocesses()
        sys.exit(130)  # 128 + SIGINT

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(description="Run evals in parallel")
    parser.add_argument(
        "--eval",
        choices=["orchestrator", "worker"],
        default="orchestrator",
        help="Which eval to run (default: orchestrator)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Models to eval (use aliases or full model IDs)",
    )
    parser.add_argument("-n", "--n-chunks", type=int, default=5, help="Chunks per sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("-i", "--input-file", help="Specific input file name")
    parser.add_argument("-q", "--question", help="Causal question (worker eval only)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show full output")
    args = parser.parse_args()

    # Get eval config
    config = EVAL_CONFIGS[args.eval]
    models_dict = config["models"]
    alias_to_model = {alias: model_id for model_id, alias in models_dict.items()}

    # Resolve model names
    if args.models:
        models = []
        for m in args.models:
            if m in alias_to_model:
                models.append(alias_to_model[m])
            else:
                models.append(m)
    else:
        models = list(models_dict.keys())

    # Build task params
    task_params = {
        "n_chunks": args.n_chunks,
        "seed": args.seed,
        "input_file": args.input_file,
    }
    if args.eval == "worker" and args.question:
        task_params["question"] = args.question

    print(f"Running {args.eval} eval for {len(models)} models in parallel...", file=sys.stderr)
    print(f"Config: n_chunks={args.n_chunks}, seed={args.seed}", file=sys.stderr)
    print(f"Aliases: {', '.join(models_dict.values())}", file=sys.stderr)
    print(file=sys.stderr)

    results = asyncio.run(run_all_evals(
        models,
        config["file"],
        models_dict,
        task_params,
    ))

    if args.verbose:
        for r in results:
            print(f"\n{'=' * 60}")
            print(f"MODEL: {r.model}")
            print("=" * 60)
            print(r.output)

    print_summary(results, models_dict)

    # Exit with error if any failed
    if not all(r.success for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
