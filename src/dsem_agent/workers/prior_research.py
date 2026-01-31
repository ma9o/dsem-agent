"""Stage 4 Worker: Per-Parameter Prior Research.

Each worker researches a single parameter using:
1. Targeted Exa literature search based on search_context
2. LLM prior elicitation based on evidence
"""

import os

from dsem_agent.orchestrator.schemas_glmm import ParameterSpec
from dsem_agent.utils.llm import WorkerGenerateFn, parse_json_response
from dsem_agent.workers.prompts.prior_research import (
    SYSTEM as PRIOR_RESEARCH_SYSTEM,
    USER as PRIOR_RESEARCH_USER,
    format_literature_for_parameter,
)
from dsem_agent.workers.schemas_prior import (
    PriorProposal,
    PriorResearchResult,
    PriorSource,
)


async def _search_parameter_literature(
    parameter: ParameterSpec,
    timeout_ms: int = 60000,
) -> list[dict]:
    """Search Exa for literature relevant to this specific parameter.

    Args:
        parameter: The parameter spec with search_context
        timeout_ms: Timeout for Exa research

    Returns:
        List of source dicts with title, url, snippet, effect_size
    """
    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        return []

    try:
        from exa_py import AsyncExa

        exa = AsyncExa(api_key=api_key)

        # Build search instructions from parameter context
        instructions = f"""\
Search for empirical effect sizes related to:

{parameter.search_context}

Focus on:
1. Meta-analyses and systematic reviews (highest priority)
2. Large-scale longitudinal studies
3. Standardized effect sizes (Cohen's d, correlation r, standardized beta)

Report specific numerical values when available.
"""

        output_schema = {
            "type": "object",
            "properties": {
                "sources": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "url": {"type": "string"},
                            "snippet": {"type": "string"},
                            "effect_size": {"type": "string"},
                        },
                    },
                },
            },
        }

        # Use faster model for per-parameter search
        research = await exa.research.create(
            instructions=instructions,
            output_schema=output_schema,
            model="exa-research-fast",
        )

        result = await exa.research.poll_until_finished(
            research.id,
            timeout_ms=timeout_ms,
        )

        if result.status != "completed" or not result.data:
            return []

        return result.data.get("sources", [])

    except Exception:
        # Don't fail the pipeline if Exa search fails
        return []


async def research_single_prior(
    parameter: ParameterSpec,
    question: str,
    generate: WorkerGenerateFn,
    enable_literature: bool = True,
) -> PriorResearchResult:
    """Research and propose a prior for a single parameter.

    Args:
        parameter: The parameter spec from GLMMSpec
        question: The research question for context
        generate: Async generate function (messages, tools) -> str
        enable_literature: Whether to search Exa for literature

    Returns:
        PriorResearchResult with proposed prior
    """
    # Search literature if enabled
    literature_sources: list[dict] = []
    if enable_literature:
        literature_sources = await _search_parameter_literature(parameter)

    # Format literature for prompt
    literature_context = format_literature_for_parameter(literature_sources)

    # Build messages
    messages = [
        {"role": "system", "content": PRIOR_RESEARCH_SYSTEM},
        {"role": "user", "content": PRIOR_RESEARCH_USER.format(
            parameter_name=parameter.name,
            parameter_role=parameter.role.value,
            parameter_constraint=parameter.constraint.value,
            parameter_description=parameter.description,
            question=question,
            literature_context=literature_context,
        )},
    ]

    # Generate prior proposal
    completion = await generate(messages, None)

    # Parse JSON response
    prior_data = parse_json_response(completion)

    # Build sources from literature + LLM response
    sources = []
    for src in prior_data.get("sources", []):
        sources.append(PriorSource(
            title=src.get("title", "Unknown"),
            url=src.get("url"),
            snippet=src.get("snippet", ""),
            effect_size=src.get("effect_size"),
        ))

    # Build prior proposal
    proposal = PriorProposal(
        parameter=parameter.name,
        distribution=prior_data.get("distribution", "Normal"),
        params=prior_data.get("params", {"mu": 0.0, "sigma": 1.0}),
        sources=sources,
        confidence=prior_data.get("confidence", 0.5),
        reasoning=prior_data.get("reasoning", ""),
    )

    return PriorResearchResult(
        parameter=parameter.name,
        proposal=proposal,
        literature_found=len(literature_sources) > 0,
        raw_response=completion,
    )


def get_default_prior(parameter: ParameterSpec) -> PriorProposal:
    """Get a default prior when research fails.

    Args:
        parameter: The parameter spec

    Returns:
        Default PriorProposal based on parameter role/constraint
    """
    from dsem_agent.orchestrator.schemas_glmm import ParameterConstraint, ParameterRole

    # Choose distribution based on constraint
    if parameter.constraint == ParameterConstraint.POSITIVE:
        distribution = "HalfNormal"
        params = {"sigma": 1.0}
    elif parameter.constraint == ParameterConstraint.UNIT_INTERVAL:
        distribution = "Beta"
        params = {"alpha": 2.0, "beta": 2.0}
    elif parameter.constraint == ParameterConstraint.CORRELATION:
        distribution = "Uniform"
        params = {"lower": -1.0, "upper": 1.0}
    else:
        distribution = "Normal"
        params = {"mu": 0.0, "sigma": 0.5}

    # Adjust based on role
    if parameter.role == ParameterRole.AR_COEFFICIENT:
        distribution = "Beta"
        params = {"alpha": 2.0, "beta": 2.0}
    elif parameter.role == ParameterRole.RESIDUAL_SD:
        distribution = "HalfNormal"
        params = {"sigma": 1.0}
    elif parameter.role == ParameterRole.RANDOM_INTERCEPT_SD:
        distribution = "HalfNormal"
        params = {"sigma": 0.5}
    elif parameter.role == ParameterRole.RANDOM_SLOPE_SD:
        distribution = "HalfNormal"
        params = {"sigma": 0.25}

    return PriorProposal(
        parameter=parameter.name,
        distribution=distribution,
        params=params,
        sources=[],
        confidence=0.3,  # Low confidence for default
        reasoning=f"Default prior for {parameter.role.value} parameter",
    )
