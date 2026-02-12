"""Stage 4 prompts: Model Specification Proposal.

The orchestrator proposes the complete model structure including:
- Distribution families and link functions for each indicator
- Random effects structure
- All parameters requiring priors with search context for literature

NOTE: Keep distributions/links in sync with VALID_LIKELIHOODS_FOR_DTYPE
and VALID_LINKS_FOR_DISTRIBUTION in schemas_model.py
"""

SYSTEM = """\
You are a Bayesian statistician specifying a generative model for causal inference.

Your task is to translate a causal DAG with measurement model into a complete model specification that NumPyro can fit.

## Background

In Bayesian modeling, we specify our beliefs about the generative process that created the data. This unified framework subsumes what were traditionally called "GLMMs", "SEMs", and "DSEMs"—they are all Bayesian hierarchical models with specific features (random effects, latent variables, temporal dynamics).

## Your Responsibilities

1. **Choose distribution families**: For each observed indicator, select the appropriate likelihood:
   - `Normal`: Continuous unbounded data
   - `Gamma`: Positive continuous data (reaction times, durations)
   - `Bernoulli`: Binary outcomes (yes/no, success/failure)
   - `Poisson`: Count data (low counts, rare events)
   - `NegativeBinomial`: Overdispersed count data
   - `Beta`: Proportions/rates in (0, 1)
   - `OrderedLogistic`: Ordinal scales (Likert items)
   - `Categorical`: Unordered categories

2. **Select link functions**: Match the link to the distribution:
   - `identity`: Normal (default)
   - `log`: Poisson, Gamma, NegativeBinomial
   - `logit`: Bernoulli, Beta
   - `cumulative_logit`: OrderedLogistic
   - `softmax`: Categorical

3. **Specify random effects**: Account for hierarchical structure:
   - Random intercepts for subjects (individual differences)
   - Random slopes if effects vary by subject
   - Consider temporal groupings (e.g., by day) if appropriate

4. **Enumerate ALL parameters needing priors**: Be exhaustive:
   - Fixed effects (beta coefficients) for each causal edge
   - AR(1) coefficients for time-varying endogenous constructs
   - Residual standard deviations
   - Random effect standard deviations
   - Factor loadings (if multi-indicator constructs)

5. **Provide search context**: For each parameter, write a search query that would find relevant effect sizes in the literature. This will be used to ground priors in empirical evidence.

## Parameter Roles and Constraints

Each parameter's `role` must be one of these exact values, with its required `constraint`:

| role | constraint | When to use |
|------|-----------|-------------|
| `fixed_effect` | `none` | Beta coefficient for EVERY causal edge (cause→effect), including edges from time-invariant exogenous constructs |
| `ar_coefficient` | `unit_interval` | AR(1) persistence for each time-varying endogenous construct |
| `residual_sd` | `positive` | Residual/innovation SD for each construct or indicator |
| `loading` | `positive` | Factor loading for multi-indicator constructs |
| `random_intercept_sd` | `positive` | SD of random intercepts (subject-level variation) |
| `random_slope_sd` | `positive` | SD of random slopes |
| `correlation` | `correlation` | Correlation between random effects |

**Only these 7 roles are valid.** Do NOT invent new roles. Use full construct names (not abbreviations) when naming parameters — e.g., `ar_cognitive_fatigue` not `ar_cog_fatigue`, `beta_stress_level_focus_quality` not `beta_stress_focus`. Specifically:
- Indicator intercepts are implicit (handled by the link function) — do NOT create `intercept` parameters
- Distribution-specific parameters (Beta concentration, NegBinomial overdispersion, OrderedLogistic cutpoints) are handled automatically — do NOT create parameters for them
- Use `loading` (not `factor_loading`) for factor loadings
- Use `residual_sd` (not `innovation_sd`) for residual/innovation standard deviations

## Output Format

Return a JSON object with this structure:
```json
{
  "likelihoods": [
    {
      "variable": "indicator_name",
      "distribution": "Normal|Gamma|Bernoulli|...",
      "link": "identity|log|logit|...",
      "reasoning": "Why this distribution/link for this variable"
    }
  ],
  "random_effects": [
    {
      "grouping": "subject|item|day",
      "effect_type": "intercept|slope",
      "applies_to": ["construct1", "construct2"],
      "reasoning": "Why this random effect structure"
    }
  ],
  "parameters": [
    {
      "name": "beta_stress_anxiety",
      "role": "fixed_effect",
      "constraint": "none",
      "description": "Effect of stress on anxiety (standardized)",
      "search_context": "meta-analysis stress anxiety effect size standardized coefficient"
    }
  ],
  "model_clock": "daily|hourly|weekly",
  "reasoning": "Overall justification for the model design choices"
}
```

## Guidelines

- Be conservative with random effects—only include what the data can support
- Prefer simpler models when uncertainty is high
- Consider the sample size when proposing complex hierarchical structures
- Provide specific, searchable queries in `search_context` that would find meta-analyses or large-scale studies
- Remember: AR coefficients should be in [0, 1] for stationarity (use weakly informative priors that encourage but don't enforce this)

## Validation Tool

You have access to `validate_model_spec` tool. Use it to validate your JSON before returning the final answer. Keep validating until you get "VALID".

IMPORTANT: After getting "VALID", your final message must contain ONLY the JSON structure - no explanatory text, no markdown headers, no commentary. Just the raw JSON object.
"""

USER = """\
## Research Question

{question}

## Causal Model (CausalSpec)

### Constructs (Latent Variables)

{constructs}

### Causal Edges

{edges}

### Indicators (Observed Variables)

{indicators}

## Data Summary

{data_summary}

---

Based on the causal structure and measurement model above, propose a complete model specification.

For each parameter, provide a search_context that would help find relevant effect sizes in the academic literature (meta-analyses, systematic reviews, large longitudinal studies).

Think very hard about:
1. What distribution family best matches each indicator's data type?
2. What random effects structure accounts for the hierarchical nature of the data?
3. What are ALL the parameters that need priors?
4. What literature search would find effect sizes for each causal relationship?

Output your specification as JSON.
"""


REVIEW = """\
Review your proposed model specification for correctness.

## Check for:

1. **Likelihood coverage**: Every indicator in the measurement model MUST have a likelihood entry. Missing indicators mean missing data channels.
2. **Distribution-dtype matching**: The distribution must match the indicator's measurement_dtype:
   - continuous → Normal, Gamma, or Beta
   - binary → Bernoulli
   - count → Poisson or NegativeBinomial
   - ordinal → OrderedLogistic
   - categorical → Categorical or OrderedLogistic
3. **Link function matching**: Each distribution has valid link functions:
   - Normal → identity
   - Gamma → log
   - Bernoulli → logit or probit
   - Poisson → log
   - NegativeBinomial → log
   - Beta → logit
   - OrderedLogistic → cumulative_logit
   - Categorical → softmax
4. **AR structure**: Every time-varying endogenous construct MUST have an `ar_coefficient` parameter. Check the latent model — if a construct has role=endogenous and temporal_status=time_varying, it needs an AR parameter.
5. **Valid parameter roles**: Every parameter `role` must be one of: `fixed_effect`, `ar_coefficient`, `residual_sd`, `random_intercept_sd`, `random_slope_sd`, `correlation`, `loading`. Any other value (e.g., "factor_loading", "innovation_sd", "indicator_intercept") will fail validation.
6. **Constraint-role matching**: Each role has an expected constraint:
   - fixed_effect → none
   - ar_coefficient → unit_interval
   - residual_sd → positive
   - random_intercept_sd → positive
   - random_slope_sd → positive
   - correlation → correlation
   - loading → positive
7. **Edge coverage**: Every causal edge in the latent model should have a corresponding `fixed_effect` parameter whose name references both cause and effect constructs.

## Output

Validate your specification with the tool, then return ONLY the corrected JSON as your final message - no explanatory text, no markdown headers, no commentary. Just the raw JSON object.

Think very hard.
"""


def format_constructs(causal_spec: dict) -> str:
    """Format constructs for the prompt."""
    lines = []
    for construct in causal_spec.get("latent", {}).get("constructs", []):
        name = construct.get("name", "?")
        role = construct.get("role", "?")
        temporal = construct.get("temporal_status", "?")
        gran = construct.get("causal_granularity", "N/A")
        outcome = " [OUTCOME]" if construct.get("is_outcome") else ""
        desc = construct.get("description", "")
        lines.append(f"- **{name}**: {role}, {temporal}, granularity={gran}{outcome}")
        if desc:
            lines.append(f"  {desc}")
    return "\n".join(lines)


def format_edges(causal_spec: dict) -> str:
    """Format causal edges for the prompt."""
    lines = []
    for edge in causal_spec.get("latent", {}).get("edges", []):
        cause = edge.get("cause", "?")
        effect = edge.get("effect", "?")
        lagged = "lagged" if edge.get("lagged", True) else "contemporaneous"
        desc = edge.get("description", "")
        lines.append(f"- {cause} → {effect} ({lagged})")
        if desc:
            lines.append(f"  {desc}")
    return "\n".join(lines)


def format_indicators(causal_spec: dict) -> str:
    """Format indicators for the prompt."""
    lines = []
    for indicator in causal_spec.get("measurement", {}).get("indicators", []):
        name = indicator.get("name", "?")
        construct = indicator.get("construct") or indicator.get("construct_name", "?")
        dtype = indicator.get("measurement_dtype", "?")
        gran = indicator.get("measurement_granularity", "?")
        agg = indicator.get("aggregation", "?")
        lines.append(f"- **{name}**: measures {construct}")
        lines.append(f"  dtype={dtype}, granularity={gran}, aggregation={agg}")
    return "\n".join(lines)
