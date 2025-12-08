import json

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model

from .prompts import STRUCTURE_PROPOSER_SYSTEM, STRUCTURE_PROPOSER_USER
from .schemas import ProposedStructure


async def propose_structure_async(question: str, data_sample: list[str]) -> dict:
    """
    Use the orchestrator LLM to propose a causal model structure.

    Args:
        question: The causal research question
        data_sample: Sample chunks from the dataset

    Returns:
        ProposedStructure as a dictionary
    """
    model = get_model()

    # Format the chunks for the prompt
    chunks_text = "\n\n---\n\n".join(
        f"**Chunk {i+1}:**\n{chunk}" for i, chunk in enumerate(data_sample)
    )

    messages = [
        ChatMessageSystem(content=STRUCTURE_PROPOSER_SYSTEM),
        ChatMessageUser(
            content=STRUCTURE_PROPOSER_USER.format(
                question=question,
                chunks=chunks_text,
            )
        ),
    ]

    response = await model.generate(messages)
    content = response.completion

    # Parse and validate the response
    # Try to extract JSON from the response (handle markdown code blocks)
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]

    data = json.loads(content)
    structure = ProposedStructure.model_validate(data)

    return structure.model_dump()


def propose_structure(question: str, data_sample: list[str]) -> dict:
    """
    Synchronous wrapper for propose_structure_async.

    Args:
        question: The causal research question
        data_sample: Sample chunks from the dataset

    Returns:
        ProposedStructure as a dictionary
    """
    import asyncio

    return asyncio.run(propose_structure_async(question, data_sample))
