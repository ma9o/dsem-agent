from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
QUERIES_DIR = DATA_DIR / "test-queries"


def load_text_chunks(path: Path, separator: str = "\n\n---\n\n") -> list[str]:
    """Load text chunks from a preprocessed file."""
    content = path.read_text()
    return [chunk.strip() for chunk in content.split(separator) if chunk.strip()]


def get_latest_preprocessed_file(directory: Path | None = None) -> Path | None:
    """
    Find the most recently modified .txt file in the preprocessed directory.

    Args:
        directory: Directory to search (default: data/preprocessed/)

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
