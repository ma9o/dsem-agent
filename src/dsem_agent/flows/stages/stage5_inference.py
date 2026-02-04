"""Stage 5: Bayesian inference and intervention analysis.

Fits the CT-SEM model using NumPyro and extracts treatment effects
from the continuous-time drift matrix.
"""

from typing import Any

import polars as pl
from prefect import flow, task


@task(task_run_name="fit-ctsem-model")
def fit_model_task(
    stage4_result: dict,
    measurements_data: dict[str, pl.DataFrame],
    sampler_config: dict | None = None,
) -> dict:
    """Fit the CT-SEM model to data.

    Args:
        stage4_result: Result from stage4_orchestrated_flow containing
            glmm_spec, priors, and model_info
        measurements_data: Measurement data by granularity
        sampler_config: Override sampler configuration

    Returns:
        Dict with fitted model samples and diagnostics
    """
    import pandas as pd

    from dsem_agent.models.ctsem_builder import CTSEMModelBuilder

    glmm_spec = stage4_result.get("glmm_spec", {})
    priors = stage4_result.get("priors", {})

    # Prepare data
    dfs = []
    for granularity, df in measurements_data.items():
        if granularity != "time_invariant" and df.height > 0:
            dfs.append(df.to_pandas())

    X = dfs[0] if dfs else pd.DataFrame({"x": [0.0]})

    default_config = {
        "num_warmup": 500,
        "num_samples": 1000,
        "num_chains": 2,
        "seed": 42,
    }
    config = {**default_config, **(sampler_config or {})}

    builder = CTSEMModelBuilder(glmm_spec=glmm_spec, priors=priors, sampler_config=config)

    try:
        mcmc = builder.fit(X)

        # Extract samples and convert to serializable format
        samples = mcmc.get_samples()
        samples_dict = {k: v.tolist() for k, v in samples.items()}

        # Get summary statistics
        summary_df = builder.summary()

        return {
            "model_type": "CT-SEM",
            "samples": samples_dict,
            "summary": summary_df.to_dict(orient="records"),
            "n_samples": config["num_samples"],
            "n_chains": config["num_chains"],
            "fit_successful": True,
        }

    except Exception as e:
        return {
            "model_type": "CT-SEM",
            "fit_successful": False,
            "error": str(e),
        }


@task(task_run_name="extract-effects")
def extract_effects_task(
    fit_result: dict,
    treatments: list[str],
    dsem_model: dict | None = None,
) -> list[dict]:
    """Extract treatment effects from fitted CT-SEM model.

    For CT-SEM, effects are extracted from the drift matrix elements which
    represent continuous-time causal effects.

    Args:
        fit_result: Result from fit_model_task
        treatments: List of treatment construct names
        dsem_model: Optional DSEM model with identifiability status

    Returns:
        List of treatment effect estimates, sorted by effect size (descending)
    """
    import numpy as np

    if not fit_result.get("fit_successful", False):
        return [
            {
                "treatment": t,
                "effect_size": None,
                "error": fit_result.get("error", "Fit failed"),
            }
            for t in treatments
        ]

    # Get identifiability status
    id_status = dsem_model.get("identifiability") if dsem_model else None
    non_identifiable: set[str] = set()
    blocker_details: dict[str, list[str]] = {}
    if id_status:
        non_identifiable_map = id_status.get("non_identifiable_treatments", {})
        non_identifiable = set(non_identifiable_map.keys())
        blocker_details = {
            treatment: details.get("confounders", [])
            for treatment, details in non_identifiable_map.items()
            if isinstance(details, dict)
        }

    results = []
    samples = fit_result.get("samples", {})

    for treatment in treatments:
        result: dict[str, Any] = {
            "treatment": treatment,
            "identifiable": treatment not in non_identifiable,
        }

        if treatment in non_identifiable:
            blockers = blocker_details.get(treatment, [])
            if blockers:
                result["warning"] = f"Effect not identifiable (blocked by: {', '.join(blockers)})"
            else:
                result["warning"] = "Effect not identifiable (missing proxies)"

        # Look for drift off-diagonal elements for this treatment
        effect_key = None
        for key in samples:
            if treatment.lower() in key.lower() and "drift" in key.lower():
                effect_key = key
                break

        if effect_key and effect_key in samples:
            effect_samples = np.array(samples[effect_key])
            result["effect_size"] = float(np.mean(effect_samples))
            result["std"] = float(np.std(effect_samples))
            result["credible_interval"] = [
                float(np.percentile(effect_samples, 5)),
                float(np.percentile(effect_samples, 95)),
            ]
        else:
            result["effect_size"] = None
            result["note"] = "Effect parameter not found in samples"

        results.append(result)

    # Sort by absolute effect size (descending)
    results.sort(
        key=lambda x: abs(x.get("effect_size") or 0),
        reverse=True,
    )

    return results


@flow(name="stage5-inference", log_prints=True)
def stage5_inference_flow(
    stage4_result: dict,
    measurements_data: dict[str, pl.DataFrame],
    treatments: list[str],
    dsem_model: dict | None = None,
    sampler_config: dict | None = None,
) -> dict:
    """Stage 5: Fit CT-SEM model and extract treatment effects.

    Args:
        stage4_result: Result from stage4_orchestrated_flow
        measurements_data: Measurement data by granularity
        treatments: List of treatment construct names to analyze
        dsem_model: Optional DSEM model with identifiability status
        sampler_config: Override sampler configuration

    Returns:
        Dict with fit_result and treatment_effects
    """
    # 1. Fit the model
    fit_result = fit_model_task(stage4_result, measurements_data, sampler_config)
    fit_result_val = fit_result.result() if hasattr(fit_result, "result") else fit_result

    # 2. Extract treatment effects
    effects = extract_effects_task(fit_result_val, treatments, dsem_model)
    effects_val = effects.result() if hasattr(effects, "result") else effects

    return {
        "fit_result": fit_result_val,
        "treatment_effects": effects_val,
    }
