"""Scoring function for DSEM structure proposals.

Scoring strategy:
- Award points for each INSTANCE of a hard rule being respected
- Complex valid structures score higher (more dimensions/edges = more rule instances)
- Return 0 immediately if ANY validation rule is violated

Can be used for manual evaluation or with DSPy optimization.
"""

import json

from pydantic import ValidationError

from causal_agent.orchestrator.schemas import (
    GRANULARITY_HOURS,
    DSEMStructure,
    Observability,
    Role,
    TemporalStatus,
)
from causal_agent.utils.aggregations import AGGREGATION_REGISTRY


def score_structure_proposal(example, pred, trace=None) -> float:
    """Score a DSEM structure proposal.

    Compatible with DSPy metric interface but can be used standalone.

    Args:
        example: Context/reference (unused, for DSPy compatibility)
        pred: Object with 'structure' field containing JSON string
        trace: Optional trace (unused, for DSPy compatibility)

    Returns:
        Float score: 0 if any rule violated, otherwise sum of rule instance points
    """
    try:
        structure_json = pred.structure
    except AttributeError:
        return 0.0

    # Parse JSON
    try:
        data = json.loads(structure_json)
    except json.JSONDecodeError:
        return 0.0

    # Validate against schema (returns 0 if any hard rule violated)
    try:
        structure = DSEMStructure(**data)
    except (ValidationError, ValueError, TypeError):
        return 0.0

    # Count points for each rule instance respected
    return _count_rule_points(structure)


def _count_rule_points_detailed(structure: DSEMStructure) -> dict:
    """Count points with detailed breakdown per category.

    Returns dict with:
        - total: float total score
        - dimensions: dict of dimension-level scoring
        - edges: dict of edge-level scoring
        - breakdown: list of human-readable scoring explanations
    """
    breakdown = []
    dim_points = {}
    edge_points = {}
    total = 0.0
    dim_map = {d.name: d for d in structure.dimensions}

    # Points for dimensions
    for dim in structure.dimensions:
        pts = 0
        details = []

        # Valid role
        pts += 1
        details.append(f"+1 valid role ({dim.role.value})")

        # Valid observability
        pts += 1
        details.append(f"+1 valid observability ({dim.observability.value})")

        # Valid temporal_status
        pts += 1
        details.append(f"+1 valid temporal_status ({dim.temporal_status.value})")

        is_time_varying = dim.temporal_status == TemporalStatus.TIME_VARYING

        # Correct causal_granularity constraint
        if is_time_varying:
            if dim.causal_granularity is not None:
                pts += 1
                details.append(f"+1 has causal_granularity ({dim.causal_granularity})")
                if dim.causal_granularity in GRANULARITY_HOURS:
                    pts += 1
                    details.append("+1 valid granularity value")
        else:
            if dim.causal_granularity is None:
                pts += 1
                details.append("+1 correctly omits causal_granularity (time_invariant)")

        # Correct aggregation constraint
        if is_time_varying:
            if dim.aggregation is not None:
                pts += 1
                details.append(f"+1 has aggregation ({dim.aggregation})")
                if dim.aggregation in AGGREGATION_REGISTRY:
                    pts += 1
                    details.append("+1 valid aggregation name")
        else:
            if dim.aggregation is None:
                pts += 1
                details.append("+1 correctly omits aggregation (time_invariant)")

        # Correct measurement_granularity constraint
        is_observed = dim.observability == Observability.OBSERVED
        valid_measurement_granularities = {"finest"} | set(GRANULARITY_HOURS.keys())
        if is_time_varying and is_observed:
            if dim.measurement_granularity is not None:
                pts += 1
                details.append(f"+1 has measurement_granularity ({dim.measurement_granularity})")
                if dim.measurement_granularity in valid_measurement_granularities:
                    pts += 1
                    details.append("+1 valid measurement_granularity value")
        else:
            if dim.measurement_granularity is None:
                pts += 1
                details.append("+1 correctly omits measurement_granularity")

        # Valid measurement_dtype
        if dim.measurement_dtype in ("continuous", "binary", "count", "ordinal", "categorical"):
            pts += 1
            details.append(f"+1 valid measurement_dtype ({dim.measurement_dtype})")

        # Bonus for latent variables
        if dim.observability == Observability.LATENT:
            pts += 1
            details.append("+1 latent variable bonus")

        dim_points[dim.name] = {"points": pts, "details": details}
        total += pts

    # Points for edges
    for edge in structure.edges:
        pts = 0
        details = []

        if edge.cause in dim_map:
            pts += 1
            details.append("+1 cause exists")

        if edge.effect in dim_map:
            pts += 1
            details.append("+1 effect exists")

        cause_dim = dim_map.get(edge.cause)
        effect_dim = dim_map.get(edge.effect)

        if cause_dim and effect_dim:
            if effect_dim.role == Role.ENDOGENOUS:
                pts += 1
                details.append("+1 effect is endogenous")

            cause_gran = cause_dim.causal_granularity
            effect_gran = effect_dim.causal_granularity

            if cause_gran == effect_gran:
                pts += 1
                details.append(f"+1 same timescale ({cause_gran or 'invariant'})")
            else:
                pts += 2
                details.append(f"+2 cross-timescale bonus ({cause_gran} → {effect_gran})")

        edge_key = f"{edge.cause} → {edge.effect}"
        edge_points[edge_key] = {"points": pts, "details": details}
        total += pts

    # Build breakdown summary
    breakdown.append(f"DIMENSIONS ({len(structure.dimensions)}):")
    for name, info in dim_points.items():
        breakdown.append(f"  {name}: {info['points']} pts")
        for d in info["details"]:
            breakdown.append(f"    {d}")

    breakdown.append(f"\nEDGES ({len(structure.edges)}):")
    for edge_key, info in edge_points.items():
        breakdown.append(f"  {edge_key}: {info['points']} pts")
        for d in info["details"]:
            breakdown.append(f"    {d}")

    breakdown.append(f"\nTOTAL: {total} points")

    return {
        "total": total,
        "dimensions": dim_points,
        "edges": edge_points,
        "breakdown": "\n".join(breakdown),
    }


def _count_rule_points(structure: DSEMStructure) -> float:
    """Count points for each rule instance correctly applied.

    Points per dimension:
    - +1 valid role
    - +1 valid observability
    - +1 valid temporal_status
    - +1 correct causal_granularity constraint (required for time_varying, forbidden for time_invariant)
    - +1 correct aggregation constraint (required for time_varying, forbidden for time_invariant)
    - +1 correct measurement_granularity constraint (required for observed time_varying, forbidden otherwise)
    - +1 valid measurement_dtype
    - +1 valid aggregation name (if specified)
    - +1 valid causal_granularity value (if specified)
    - +1 valid measurement_granularity value (if specified)

    Points per edge:
    - +1 cause exists in dimensions
    - +1 effect exists in dimensions
    - +1 effect is endogenous
    - +1 correct timescale handling (same-scale or cross-scale rules)

    Bonus points:
    - +2 per cross-timescale edge (more complex modeling)
    - +1 per latent variable (captures unobserved heterogeneity)
    """
    points = 0.0
    dim_map = {d.name: d for d in structure.dimensions}

    # Points for dimensions
    for dim in structure.dimensions:
        # Valid role (already validated by schema, but count it)
        points += 1

        # Valid observability
        points += 1

        # Valid temporal_status
        points += 1

        is_time_varying = dim.temporal_status == TemporalStatus.TIME_VARYING

        # Correct causal_granularity constraint
        if is_time_varying:
            if dim.causal_granularity is not None:
                points += 1
                # Valid granularity value
                if dim.causal_granularity in GRANULARITY_HOURS:
                    points += 1
        else:  # time_invariant
            if dim.causal_granularity is None:
                points += 1

        # Correct aggregation constraint
        if is_time_varying:
            if dim.aggregation is not None:
                points += 1
                # Valid aggregation name
                if dim.aggregation in AGGREGATION_REGISTRY:
                    points += 1
        else:  # time_invariant
            if dim.aggregation is None:
                points += 1

        # Correct measurement_granularity constraint
        is_observed = dim.observability == Observability.OBSERVED
        valid_measurement_granularities = {"finest"} | set(GRANULARITY_HOURS.keys())
        if is_time_varying and is_observed:
            if dim.measurement_granularity is not None:
                points += 1
                if dim.measurement_granularity in valid_measurement_granularities:
                    points += 1
        else:
            if dim.measurement_granularity is None:
                points += 1

        # Valid measurement_dtype
        if dim.measurement_dtype in ("continuous", "binary", "count", "ordinal", "categorical"):
            points += 1

        # Bonus for latent variables (modeling unobserved heterogeneity)
        if dim.observability == Observability.LATENT:
            points += 1

    # Points for edges
    for edge in structure.edges:
        # Cause exists
        if edge.cause in dim_map:
            points += 1

        # Effect exists
        if edge.effect in dim_map:
            points += 1

        cause_dim = dim_map.get(edge.cause)
        effect_dim = dim_map.get(edge.effect)

        if cause_dim and effect_dim:
            # Effect is endogenous
            if effect_dim.role == Role.ENDOGENOUS:
                points += 1

            cause_gran = cause_dim.causal_granularity
            effect_gran = effect_dim.causal_granularity

            # Timescale handling
            if cause_gran == effect_gran:
                # Same timescale: valid lag handling
                points += 1
            else:
                # Cross-timescale: bonus for complexity
                points += 2

    return points


def score_structure_proposal_normalized(example, pred, trace=None) -> float:
    """Normalized version of score_structure_proposal (0-1 range).

    Useful when comparing across very different structure complexities.
    Divides by theoretical maximum for the given structure size.
    """
    raw_score = score_structure_proposal(example, pred, trace)
    if raw_score == 0:
        return 0.0

    # Parse to get structure size for normalization
    try:
        data = json.loads(pred.structure)
        n_dims = len(data.get("dimensions", []))
        n_edges = len(data.get("edges", []))
    except (json.JSONDecodeError, AttributeError):
        return 0.0

    # Theoretical max per dimension: ~10-11 points (3 classification + 2 causal_granularity + 2 aggregation + 2 measurement_granularity + 1 dtype + 1 latent bonus)
    # Theoretical max per edge: ~4 points (cause + effect + endogenous + timescale/cross-timescale bonus)
    max_dim_points = n_dims * 11
    max_edge_points = n_edges * 4
    max_points = max_dim_points + max_edge_points

    if max_points == 0:
        return 0.0

    return min(1.0, raw_score / max_points)
