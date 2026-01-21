"""Stage 1a: Latent Model Proposal (Orchestrator).

The orchestrator proposes theoretical constructs and causal edges based on
domain knowledge alone, WITHOUT seeing any data.

This follows the Anderson & Gerbing (1988) two-step approach where the
latent model is specified first from theory.
"""

from prefect import task
from prefect.cache_policies import INPUTS

from causal_agent.orchestrator.agents import propose_latent_model as propose_latent_model_agent


@task(retries=2, retry_delay_seconds=30, cache_policy=INPUTS)
def propose_latent_model(question: str) -> dict:
    """Orchestrator proposes theoretical constructs and causal edges (latent model).

    This is Stage 1a - reasoning from domain knowledge only, no data.

    Args:
        question: The causal research question

    Returns:
        LatentModel as a dictionary with 'constructs' and 'edges'
    """
    return propose_latent_model_agent(question)
