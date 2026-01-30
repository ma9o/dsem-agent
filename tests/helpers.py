"""Shared test helpers (non-fixtures).

These are utilities that can be imported directly into test modules.
For fixtures, see conftest.py.
"""

from dataclasses import dataclass


@dataclass
class MockPrediction:
    """Mock DSPy prediction object for scoring tests."""

    structure: str


def make_mock_generate(responses: list[str]):
    """Create a mock generate function that returns predefined responses.

    Args:
        responses: List of JSON strings to return in order

    Returns:
        Async function matching OrchestratorGenerateFn signature
    """
    call_count = [0]  # Use list to allow mutation in closure

    async def mock_generate(
        messages: list, tools: list | None, follow_ups: list[str] | None
    ) -> str:
        idx = min(call_count[0], len(responses) - 1)
        call_count[0] += 1
        return responses[idx]

    return mock_generate
