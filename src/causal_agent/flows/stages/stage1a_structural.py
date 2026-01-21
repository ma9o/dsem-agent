"""Stage 1a: Structural Model Proposal (Orchestrator).

The orchestrator proposes theoretical constructs and causal edges based on
domain knowledge alone, WITHOUT seeing any data.

This follows the Anderson & Gerbing (1988) two-step approach where the
structural model is specified first from theory.
"""

from prefect import task
from prefect.cache_policies import INPUTS

from causal_agent.orchestrator.agents import propose_structural_model as propose_structural_model_agent


@task(retries=2, retry_delay_seconds=30, cache_policy=INPUTS)
def propose_structural_model(question: str) -> dict:
    """Orchestrator proposes theoretical constructs and causal edges.

    This is Stage 1a - reasoning from domain knowledge only, no data.

    Args:
        question: The causal research question

    Returns:
        StructuralModel as a dictionary with 'constructs' and 'edges'
    """
    return propose_structural_model_agent(question)
