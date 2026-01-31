"""Prompts for worker LLM agents."""

from . import extraction

# Re-export with legacy names for backwards compatibility
WORKER_W_PROPOSALS_SYSTEM = extraction.SYSTEM_WITH_PROPOSALS
WORKER_WO_PROPOSALS_SYSTEM = extraction.SYSTEM_WITHOUT_PROPOSALS
WORKER_USER = extraction.USER

__all__ = [
    # Modules
    "extraction",
    # Legacy exports
    "WORKER_W_PROPOSALS_SYSTEM",
    "WORKER_WO_PROPOSALS_SYSTEM",
    "WORKER_USER",
]
