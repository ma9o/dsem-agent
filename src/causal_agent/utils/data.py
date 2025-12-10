import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
QUERIES_DIR = DATA_DIR / "queries"
TRAINING_DIR = DATA_DIR / "training"

# Backwards compatibility alias
PREPROCESSED_DIR = PROCESSED_DIR

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 50))
SAMPLE_CHUNKS = int(os.environ.get("SAMPLE_CHUNKS", 10))


def load_lines(path: Path) -> list[str]:
    """Load individual lines from a preprocessed file."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_text_chunks(path: Path, chunk_size: int | None = None) -> list[str]:
    """
    Load text chunks from a preprocessed file.

    Each chunk is a group of contiguous lines joined by newlines.

    Args:
        path: Path to preprocessed file (one record per line)
        chunk_size: Lines per chunk (default: CHUNK_SIZE from .env)

    Returns:
        List of chunks, where each chunk is multiple lines joined together
    """
    chunk_size = chunk_size or CHUNK_SIZE
    lines = load_lines(path)

    chunks = []
    for i in range(0, len(lines), chunk_size):
        chunk_lines = lines[i : i + chunk_size]
        chunks.append("\n".join(chunk_lines))

    return chunks


def get_latest_preprocessed_file(directory: Path | None = None) -> Path | None:
    """
    Find the most recently modified .txt file in the processed directory.

    Args:
        directory: Directory to search (default: data/processed/)

    Returns:
        Path to latest file, or None if no files found
    """
    search_dir = directory or PREPROCESSED_DIR
    txt_files = list(search_dir.glob("*.txt"))

    if not txt_files:
        return None

    # Sort by modification time, newest first
    return max(txt_files, key=lambda p: p.stat().st_mtime)


def resolve_input_path(filename: str | None = None) -> Path:
    """
    Resolve input path from filename or find latest preprocessed file.

    Args:
        filename: Optional filename (just name, not full path) in preprocessed dir.
                  If None, returns the latest preprocessed file.

    Returns:
        Full path to the input file

    Raises:
        FileNotFoundError: If no matching file found
    """
    if filename:
        path = PREPROCESSED_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return path

    latest = get_latest_preprocessed_file()
    if not latest:
        raise FileNotFoundError(f"No preprocessed files found in {PREPROCESSED_DIR}")

    return latest


def resolve_query_path(filename: str) -> Path:
    """
    Resolve query file path from filename.

    Args:
        filename: Filename (mnemonic name like 'smoking-cancer.txt') in test-queries dir.
                  Extension is optional - will try .txt and .md if not provided.

    Returns:
        Full path to the query file

    Raises:
        FileNotFoundError: If no matching file found
    """
    # Try exact match first
    path = QUERIES_DIR / filename
    if path.exists():
        return path

    # Try with extensions if not provided
    if not path.suffix:
        for ext in [".txt", ".md"]:
            path_with_ext = QUERIES_DIR / f"{filename}{ext}"
            if path_with_ext.exists():
                return path_with_ext

    raise FileNotFoundError(f"Query file not found: {filename} in {QUERIES_DIR}")


def load_query(filename: str) -> str:
    """
    Load query content from file.

    Args:
        filename: Filename in test-queries dir (e.g., 'smoking-cancer')

    Returns:
        Query text content
    """
    path = resolve_query_path(filename)
    return path.read_text().strip()


def list_queries() -> list[str]:
    """List available query files in test-queries directory."""
    return [p.name for p in QUERIES_DIR.glob("*") if p.is_file() and p.name != ".gitkeep"]
