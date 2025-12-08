#!/usr/bin/env python
"""Sample data chunks for manual LLM testing.

Usage:
    uv run python scripts/sample_data_chunks.py
    uv run python scripts/sample_data_chunks.py -n 3
    uv run python scripts/sample_data_chunks.py -i google_activity_20251208.txt -n 5
"""

import argparse
import random
from pathlib import Path

from causal_agent.utils.data import (
    CHUNK_SIZE,
    SAMPLE_CHUNKS,
    get_latest_preprocessed_file,
    load_text_chunks,
    PREPROCESSED_DIR,
)

OUTPUT_FILE = Path(__file__).parent.parent / "data/orchestrator-samples-manual.txt"


def sample_chunks(input_file: Path, n: int, seed: int | None = None) -> list[str]:
    """Sample n contiguous chunks from input file."""
    chunks = load_text_chunks(input_file)

    if seed is not None:
        random.seed(seed)

    n = min(n, len(chunks))
    # Pick a random start position and take contiguous chunks
    max_start = len(chunks) - n
    start = random.randint(0, max_start) if max_start > 0 else 0
    return chunks[start : start + n]


def main():
    parser = argparse.ArgumentParser(description="Sample data chunks for manual testing")
    parser.add_argument("-n", type=int, default=SAMPLE_CHUNKS, help="Number of chunks to sample")
    parser.add_argument("-i", "--input", type=str, help="Input file name (in data/preprocessed/)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Get input file
    if args.input:
        input_file = PREPROCESSED_DIR / args.input
    else:
        input_file = get_latest_preprocessed_file()
        if not input_file:
            raise FileNotFoundError(f"No .txt files found in {PREPROCESSED_DIR}")

    print(f"Sampling from: {input_file.name}")
    print(f"Chunk size: {CHUNK_SIZE} lines per chunk")

    # Sample chunks
    chunks = sample_chunks(input_file, args.n, args.seed)

    # Write output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        f.write(f"# Sampled {len(chunks)} chunks from {input_file.name}\n")
        f.write(f"# Each chunk = {CHUNK_SIZE} contiguous lines\n")
        f.write(f"# Use these with an LLM to test causal graph construction\n\n")
        for i, chunk in enumerate(chunks):
            f.write(f"--- CHUNK {i + 1} ---\n")
            f.write(chunk + "\n\n")

    print(f"Wrote {len(chunks)} chunks ({CHUNK_SIZE} lines each) to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
