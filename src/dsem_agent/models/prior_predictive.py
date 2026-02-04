"""Prior Predictive Validation for Stage 4.

Validates proposed priors by sampling from the prior predictive distribution
and checking for domain violations (NaN/Inf, wrong sign for constrained params).
"""

import numpy as np
import polars as pl

from dsem_agent.models.ctsem_builder import CTSEMModelBuilder
from dsem_agent.orchestrator.schemas_glmm import GLMMSpec
from dsem_agent.workers.schemas_prior import PriorProposal, PriorValidationResult


def validate_prior_predictive(
    glmm_spec: GLMMSpec | dict,
    priors: dict[str, PriorProposal] | dict[str, dict],
    measurements_data: dict[str, pl.DataFrame],
    n_samples: int = 500,
) -> tuple[bool, list[PriorValidationResult]]:
    """Validate priors via prior predictive sampling.

    Checks for:
    1. Model builds successfully
    2. No NaN/Inf in samples
    3. Domain violations (negative for positive-constrained, outside [0,1] for unit_interval)

    Args:
        glmm_spec: GLMM specification
        priors: Prior proposals for each parameter
        measurements_data: Dict of granularity -> polars DataFrame
        n_samples: Number of prior predictive samples

    Returns:
        Tuple of (is_valid, list of validation results)
    """
    results: list[PriorValidationResult] = []
    all_valid = True

    # Convert inputs to dicts if needed
    if isinstance(glmm_spec, GLMMSpec):
        glmm_dict = glmm_spec.model_dump()
    else:
        glmm_dict = glmm_spec

    priors_dict = {}
    for name, prior in priors.items():
        if isinstance(prior, PriorProposal):
            priors_dict[name] = prior.model_dump()
        else:
            priors_dict[name] = prior

    # Build a minimal DataFrame for model construction
    X = _build_minimal_dataframe(measurements_data)

    try:
        # Build the CT-SEM model
        builder = CTSEMModelBuilder(glmm_spec=glmm_dict, priors=priors_dict)
        builder.build_model(X)

        # Sample from prior predictive
        prior_samples = builder.sample_prior_predictive(samples=n_samples)

        # Check each sampled parameter
        for param_name in prior_samples.keys():
            result = _validate_parameter_samples(param_name, prior_samples[param_name])
            results.append(result)
            if not result.is_valid:
                all_valid = False

    except Exception as e:
        # Model building failed - report as validation failure
        results.append(PriorValidationResult(
            parameter="model_build",
            is_valid=False,
            issue=f"Model building failed: {e}",
            suggested_adjustment=None,
        ))
        all_valid = False

    return all_valid, results


def _build_minimal_dataframe(
    measurements_data: dict[str, pl.DataFrame],
) -> "pd.DataFrame":
    """Build a minimal pandas DataFrame for model construction."""
    import pandas as pd

    for granularity, df in measurements_data.items():
        if granularity != "time_invariant" and df.height > 0:
            return df.to_pandas()

    return pd.DataFrame({"x": [0.0]})


def _validate_parameter_samples(
    param_name: str,
    samples: np.ndarray,
) -> PriorValidationResult:
    """Validate samples from prior predictive."""
    samples = np.asarray(samples).flatten()

    # Check for NaN/Inf
    n_invalid = np.sum(~np.isfinite(samples))
    if n_invalid > 0:
        pct = 100 * n_invalid / len(samples)
        return PriorValidationResult(
            parameter=param_name,
            is_valid=False,
            issue=f"{pct:.1f}% of samples are NaN/Inf",
            suggested_adjustment=None,
        )

    # Check domain violations based on parameter name
    constraint = _get_constraint_from_param_name(param_name)

    if constraint == "positive":
        n_negative = np.sum(samples < 0)
        if n_negative > 0:
            pct = 100 * n_negative / len(samples)
            return PriorValidationResult(
                parameter=param_name,
                is_valid=False,
                issue=f"{pct:.1f}% of samples are negative (should be positive)",
                suggested_adjustment=None,
            )

    elif constraint == "unit_interval":
        n_outside = np.sum((samples < 0) | (samples > 1))
        if n_outside > 0:
            pct = 100 * n_outside / len(samples)
            return PriorValidationResult(
                parameter=param_name,
                is_valid=False,
                issue=f"{pct:.1f}% of samples outside [0, 1]",
                suggested_adjustment=None,
            )

    return PriorValidationResult(
        parameter=param_name,
        is_valid=True,
        issue=None,
        suggested_adjustment=None,
    )


def _get_constraint_from_param_name(param_name: str) -> str:
    """Infer constraint from parameter name."""
    name_lower = param_name.lower()

    # Positive-constrained parameters
    if any(s in name_lower for s in ["diffusion", "sigma", "sd", "var", "scale"]):
        return "positive"

    # Unit interval parameters (loadings, correlations)
    if any(s in name_lower for s in ["corr", "loading"]):
        return "unit_interval"

    return "none"


def format_validation_report(
    is_valid: bool,
    results: list[PriorValidationResult],
) -> str:
    """Format validation results as a human-readable report."""
    lines = []

    if is_valid:
        lines.append("Prior predictive validation PASSED")
    else:
        lines.append("Prior predictive validation FAILED")

    failed = [r for r in results if not r.is_valid]
    if failed:
        lines.append("")
        for r in failed:
            lines.append(f"- {r.parameter}: {r.issue}")

    return "\n".join(lines)
