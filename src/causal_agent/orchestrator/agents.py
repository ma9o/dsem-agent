"""Orchestrator agents using Inspect AI with OpenRouter.

Two-stage approach following Anderson & Gerbing (1988):
1. Structural Model (Stage 1a) - theoretical constructs + causal edges, NO DATA
2. Measurement Model (Stage 1b) - operationalize constructs into indicators, WITH DATA
"""

import json
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    get_model,
)

from causal_agent.utils.config import get_config
from causal_agent.utils.llm import (
    make_validate_measurement_model_tool,
    multi_turn_generate,
    parse_json_response,
    validate_structural_model_tool,
)
from .prompts import (
    STRUCTURAL_MODEL_SYSTEM,
    STRUCTURAL_MODEL_USER,
    STRUCTURAL_MODEL_REVIEW,
    MEASUREMENT_MODEL_SYSTEM,
    MEASUREMENT_MODEL_USER,
    MEASUREMENT_MODEL_REVIEW,
)
from .schemas import DSEMModel, MeasurementModel, StructuralModel

# Load environment variables from .env file (for API keys)
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1a: STRUCTURAL MODEL (theory-driven, no data)
# ══════════════════════════════════════════════════════════════════════════════


async def propose_structural_model_async(question: str) -> dict:
    """
    Use the orchestrator LLM to propose a theoretical causal structure.

    This is Stage 1a - the LLM reasons from domain knowledge only, without seeing data.

    Two-step process:
    1. Initial proposal: Generate structure from question
    2. Self-review: Check theoretical coherence

    Args:
        question: The causal research question (natural language)

    Returns:
        StructuralModel as a dictionary
    """
    model_name = get_config().stage1_structure_proposal.model
    model = get_model(model_name)

    messages = [
        ChatMessageSystem(content=STRUCTURAL_MODEL_SYSTEM),
        ChatMessageUser(content=STRUCTURAL_MODEL_USER.format(question=question)),
    ]

    # Run multi-turn: initial proposal + self-review, with validation tool available
    completion = await multi_turn_generate(
        messages=messages,
        model=model,
        follow_ups=[STRUCTURAL_MODEL_REVIEW],
        tools=[validate_structural_model_tool()],
    )

    # Parse and validate final result
    data = parse_json_response(completion)
    structural_model = StructuralModel.model_validate(data)

    return structural_model.model_dump()


def propose_structural_model(question: str) -> dict:
    """
    Synchronous wrapper for propose_structural_model_async.

    Args:
        question: The causal research question

    Returns:
        StructuralModel as a dictionary
    """
    import asyncio

    return asyncio.run(propose_structural_model_async(question))


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1b: MEASUREMENT MODEL (data-driven operationalization)
# ══════════════════════════════════════════════════════════════════════════════


async def propose_measurement_model_async(
    question: str,
    structural_model: dict,
    data_sample: list[str],
    dataset_summary: str = "",
) -> dict:
    """
    Use the orchestrator LLM to propose a measurement model for the structural model.

    This is Stage 1b - the LLM sees data and operationalizes constructs into indicators.

    Two-step process:
    1. Initial proposal: Generate indicators from structural model + data
    2. Self-review: Check operationalization coherence

    Args:
        question: The causal research question (natural language)
        structural_model: The structural model dict from Stage 1a
        data_sample: Sample chunks from the dataset
        dataset_summary: Brief overview of the full dataset (size, timespan, etc.)

    Returns:
        MeasurementModel as a dictionary
    """
    model_name = get_config().stage1_structure_proposal.model
    model = get_model(model_name)

    # Parse structural model for validation tool
    structural = StructuralModel.model_validate(structural_model)

    # Format the chunks for the prompt
    chunks_text = "\n".join(data_sample)

    messages = [
        ChatMessageSystem(content=MEASUREMENT_MODEL_SYSTEM),
        ChatMessageUser(
            content=MEASUREMENT_MODEL_USER.format(
                question=question,
                structural_model_json=json.dumps(structural_model, indent=2),
                dataset_summary=dataset_summary or "Not provided",
                chunks=chunks_text,
            )
        ),
    ]

    # Run multi-turn: initial proposal + self-review, with validation tool available
    completion = await multi_turn_generate(
        messages=messages,
        model=model,
        follow_ups=[MEASUREMENT_MODEL_REVIEW],
        tools=[make_validate_measurement_model_tool(structural)],
    )

    # Parse and validate final result
    data = parse_json_response(completion)
    measurement_model = MeasurementModel.model_validate(data)

    return measurement_model.model_dump()


def propose_measurement_model(
    question: str,
    structural_model: dict,
    data_sample: list[str],
    dataset_summary: str = "",
) -> dict:
    """
    Synchronous wrapper for propose_measurement_model_async.

    Args:
        question: The causal research question
        structural_model: The structural model dict from Stage 1a
        data_sample: Sample chunks from the dataset
        dataset_summary: Brief overview of the full dataset

    Returns:
        MeasurementModel as a dictionary
    """
    import asyncio

    return asyncio.run(
        propose_measurement_model_async(question, structural_model, data_sample, dataset_summary)
    )


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED: FULL DSEM MODEL
# ══════════════════════════════════════════════════════════════════════════════


def build_dsem_model(structural_model: dict, measurement_model: dict) -> dict:
    """
    Combine structural and measurement models into a full DSEMModel.

    Args:
        structural_model: The structural model dict from Stage 1a
        measurement_model: The measurement model dict from Stage 1b

    Returns:
        DSEMModel as a dictionary
    """
    dsem = DSEMModel(
        structural=StructuralModel.model_validate(structural_model),
        measurement=MeasurementModel.model_validate(measurement_model),
    )
    return dsem.model_dump()
