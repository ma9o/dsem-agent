from pathlib import Path


def load_text_chunks(path: Path, separator: str = "\n\n---\n\n") -> list[str]:
    """Load text chunks from a preprocessed file."""
    content = path.read_text()
    return [chunk.strip() for chunk in content.split(separator) if chunk.strip()]
