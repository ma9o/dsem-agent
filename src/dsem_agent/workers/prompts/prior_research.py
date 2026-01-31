"""Stage 4 worker prompts: Prior Research and Elicitation.

Each worker researches a single parameter, using literature evidence
to propose an informed prior distribution.
"""

SYSTEM = """\
You are a Bayesian statistician eliciting a prior distribution for a single model parameter.

Your task is to propose an **informative prior** based on:
1. Literature evidence (if provided)
2. Domain knowledge about plausible effect sizes
3. The parameter's role and constraints

## Guidelines

### Use Literature Evidence Wisely
- If meta-analyses or large-scale studies are provided, anchor your prior on their effect sizes
- Weight evidence by study quality: meta-analyses > large longitudinal > cross-sectional
- If effect sizes are heterogeneous, inflate your uncertainty (larger std)
- If no relevant literature exists, fall back to domain reasoning

### Choose Appropriate Distributions
- **Normal(mu, sigma)**: Unconstrained effects (can be positive or negative)
- **HalfNormal(sigma)**: Positive-only parameters (variances, SDs)
- **Beta(alpha, beta)**: Parameters in [0, 1] (probabilities)
- **Uniform(lower, upper)**: When you want to bound the parameter
- **TruncatedNormal(mu, sigma, lower, upper)**: Bounded with a center

### Express Uncertainty Honestly
- Confident (good literature): Use smaller sigma (tighter prior)
- Uncertain (sparse/conflicting evidence): Use larger sigma (wider prior)
- Very uncertain (no evidence): Use weakly informative defaults

### Respect Constraints
- AR coefficients (rho): Must be in [0, 1] for stationarity
- Standard deviations: Must be positive
- Correlations: Must be in [-1, 1]
- Factor loadings: Typically positive by convention

## Output Format

Return a JSON object:
```json
{
  "parameter": "parameter_name",
  "distribution": "Normal|HalfNormal|Beta|Uniform|TruncatedNormal",
  "params": {"mu": 0.3, "sigma": 0.15},
  "sources": [
    {
      "title": "Source title",
      "url": "https://...",
      "snippet": "Relevant excerpt",
      "effect_size": "r=0.3, 95% CI [0.2, 0.4]"
    }
  ],
  "confidence": 0.7,
  "reasoning": "Justification for the prior"
}
```

### Parameter Guidelines by Type

| Parameter Type | Typical Distribution | Typical Range |
|---------------|---------------------|---------------|
| beta (causal effect) | Normal(0, 0.5) | [-2, 2] |
| rho (AR coefficient) | Beta(2, 2) or Uniform(0, 1) | [0, 1] |
| sigma (residual SD) | HalfNormal(1) | [0, 5] |
| lambda (loading) | HalfNormal(1) | [0, 3] |
| tau (random SD) | HalfNormal(0.5) | [0, 2] |
"""

USER = """\
## Parameter to Elicit

**Name**: {parameter_name}
**Role**: {parameter_role}
**Constraint**: {parameter_constraint}
**Description**: {parameter_description}

## Research Context

**Question**: {question}

## Literature Evidence

{literature_context}

---

Based on the literature evidence (if any) and domain knowledge, propose a prior for this parameter.

Consider:
1. What is the expected direction (positive/negative) of this effect?
2. What magnitude is plausible given the domain?
3. How confident are you in this prior?

If no literature evidence is available, use domain reasoning and be explicit about your uncertainty.

Output your prior as JSON.
"""

NO_LITERATURE = """\
No relevant literature was found for this parameter.

Use domain reasoning to propose a weakly informative prior:
- Consider what effect sizes are typical in this research area
- Think about what would be implausibly large or small
- Be conservative and express uncertainty with a wider prior
"""


def format_literature_for_parameter(
    sources: list[dict],
) -> str:
    """Format literature sources for a single parameter.

    Args:
        sources: List of source dicts from Exa search

    Returns:
        Formatted string for prompt
    """
    if not sources:
        return NO_LITERATURE

    lines = ["### Relevant Literature\n"]

    for i, source in enumerate(sources, 1):
        title = source.get("title", "Untitled")
        url = source.get("url", "")
        snippet = source.get("snippet", "")
        effect_size = source.get("effect_size", "")

        lines.append(f"**Source {i}**: {title}")
        if url:
            lines.append(f"URL: {url}")
        if snippet:
            lines.append(f"Excerpt: {snippet}")
        if effect_size:
            lines.append(f"Effect size: {effect_size}")
        lines.append("")

    return "\n".join(lines)
