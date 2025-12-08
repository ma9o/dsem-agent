#!/usr/bin/env python
"""Sample data chunks for manual LLM testing.

Usage:
    uv run python scripts/sample_data_chunks.py
    uv run python scripts/sample_data_chunks.py -n 20
    uv run python scripts/sample_data_chunks.py -i google_activity_20251208.txt -n 30
"""

import argparse
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data/preprocessed"
OUTPUT_FILE = Path(__file__).parent.parent / "data/orchestrator-samples-manual.txt"


def get_latest_file() -> Path:
    """Get the most recently modified preprocessed file."""
    files = list(DATA_DIR.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {DATA_DIR}")
    return max(files, key=lambda f: f.stat().st_mtime)


def sample_chunks(input_file: Path, n: int, seed: int | None = None) -> list[str]:
    """Sample n contiguous chunks from input file."""
    with open(input_file) as f:
        lines = [line.strip() for line in f if line.strip()]

    if seed is not None:
        random.seed(seed)

    n = min(n, len(lines))
    # Pick a random start position and take contiguous block
    max_start = len(lines) - n
    start = random.randint(0, max_start) if max_start > 0 else 0
    return lines[start : start + n]


def main():
    parser = argparse.ArgumentParser(description="Sample data chunks for manual testing")
    parser.add_argument("-n", type=int, default=15, help="Number of chunks to sample")
    parser.add_argument("-i", "--input", type=str, help="Input file name (in data/preprocessed/)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Get input file
    if args.input:
        input_file = DATA_DIR / args.input
    else:
        input_file = get_latest_file()

    print(f"Sampling from: {input_file.name}")

    # Sample chunks
    chunks = sample_chunks(input_file, args.n, args.seed)

    # Write output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        f.write(f"# Sampled {len(chunks)} chunks from {input_file.name}\n")
        f.write(f"# Use these with an LLM to test causal graph construction\n\n")
        for chunk in chunks:
            f.write(chunk + "\n")

    print(f"Wrote {len(chunks)} chunks to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
