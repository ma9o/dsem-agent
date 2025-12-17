from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

from causal_agent.utils.aggregations import AGGREGATION_REGISTRY


class Role(str, Enum):
    """Whether a variable is modeled (endogenous) or given (exogenous)."""

    ENDOGENOUS = "endogenous"  # Has inbound edges, is modeled
    EXOGENOUS = "exogenous"  # No inbound edges, given/external


class Observability(str, Enum):
    """Whether a variable is directly measurable."""

    OBSERVED = "observed"  # Directly measured in data
    LATENT = "latent"  # Not directly measured, inferred


class TemporalStatus(str, Enum):
    """Whether a variable changes over time."""

    TIME_VARYING = "time_varying"  # Changes within person over time
    TIME_INVARIANT = "time_invariant"  # Fixed for each person


# Hours per granularity unit
GRANULARITY_HOURS = {
    "hourly": 1,
    "daily": 24,
    "weekly": 168,
    "monthly": 720,  # 30 days
    "yearly": 8760,
}


class Dimension(BaseModel):
    """A variable in the causal model."""

    model_config = {"populate_by_name": True}

    name: str = Field(description="Variable name (e.g., 'sleep_quality')")
    description: str = Field(description="What this variable represents")
    role: Role = Field(description="'endogenous' (modeled) or 'exogenous' (given)")
    is_outcome: bool = Field(
        default=False,
        description="True if this is the primary outcome variable Y implied by the question",
    )
    observability: Observability = Field(description="'observed' (measured) or 'latent' (inferred)")
    how_to_measure: str | None = Field(
        default=None,
        description=(
            "Instructions for workers on how to extract this variable from data. "
            "Required for observed variables, must be null for latent variables."
        ),
    )
    temporal_status: TemporalStatus = Field(
        description="'time_varying' (changes over time) or 'time_invariant' (fixed)"
    )
    causal_granularity: str | None = Field(
        default=None,
        description=(
            "'hourly', 'daily', 'weekly', 'monthly', 'yearly'. Required for time-varying variables. "
            "The granularity at which causal relationships make sense."
        ),
    )
    measurement_granularity: str | None = Field(
        default=None,
        description=(
            "'finest' (one datapoint per raw entry) or 'hourly', 'daily', 'weekly', 'monthly', 'yearly'. "
            "Required for observed time-varying variables. "
            "The resolution at which workers should extract measurements from data."
        ),
    )
    measurement_dtype: str = Field(
        description="'continuous', 'binary', 'count', 'ordinal', 'categorical'"
    )
    aggregation: str | None = Field(
        default=None,
        description=f"Aggregation function from registry. Available: {', '.join(sorted(AGGREGATION_REGISTRY.keys()))}",
    )
    # Review metadata - tracks what changed during self-review stage
    changed: str | None = Field(
        default=None,
        alias="_changed",
        description="Review notes on what was modified (e.g., 'measurement_dtype: continuous→count')",
    )

    @field_validator("aggregation")
    @classmethod
    def validate_aggregation(cls, v: str | None) -> str | None:
        if v is not None and v not in AGGREGATION_REGISTRY:
            available = ", ".join(sorted(AGGREGATION_REGISTRY.keys()))
            raise ValueError(f"Unknown aggregation '{v}'. Available: {available}")
        return v

    @model_validator(mode="after")
    def validate_field_consistency(self):
        """Validate that causal_granularity, measurement_granularity, and aggregation are consistent."""
        is_time_varying = self.temporal_status == TemporalStatus.TIME_VARYING
        is_observed = self.observability == Observability.OBSERVED

        if is_time_varying:
            if self.causal_granularity is None:
                raise ValueError(
                    f"Time-varying variable '{self.name}' requires causal_granularity"
                )
            if self.aggregation is None:
                raise ValueError(
                    f"Time-varying variable '{self.name}' requires aggregation (how to collapse raw data)"
                )
            # measurement_granularity required for observed time-varying variables
            if is_observed and self.measurement_granularity is None:
                raise ValueError(
                    f"Observed time-varying variable '{self.name}' requires measurement_granularity"
                )
        else:
            if self.causal_granularity is not None:
                raise ValueError(
                    f"Time-invariant variable '{self.name}' must not have causal_granularity"
                )
            if self.aggregation is not None:
                raise ValueError(
                    f"Time-invariant variable '{self.name}' must not have aggregation"
                )

        # measurement_granularity only for observed time-varying
        if self.measurement_granularity is not None:
            if not is_observed:
                raise ValueError(
                    f"Latent variable '{self.name}' must not have measurement_granularity"
                )
            if not is_time_varying:
                raise ValueError(
                    f"Time-invariant variable '{self.name}' must not have measurement_granularity"
                )
            # Validate measurement_granularity value
            valid_measurement_granularities = {"finest"} | set(GRANULARITY_HOURS.keys())
            if self.measurement_granularity not in valid_measurement_granularities:
                raise ValueError(
                    f"Invalid measurement_granularity '{self.measurement_granularity}' for '{self.name}'. "
                    f"Must be 'finest' or one of: {', '.join(sorted(GRANULARITY_HOURS.keys()))}"
                )

        # Outcomes must be endogenous
        if self.is_outcome and self.role != Role.ENDOGENOUS:
            raise ValueError(
                f"Outcome variable '{self.name}' must be endogenous, got {self.role.value}"
            )

        # how_to_measure required for observed, must be null for latent
        if is_observed and not self.how_to_measure:
            raise ValueError(
                f"Observed variable '{self.name}' requires how_to_measure instructions"
            )
        if not is_observed and self.how_to_measure:
            raise ValueError(
                f"Latent variable '{self.name}' must not have how_to_measure (it's not directly measurable)"
            )

        return self


class CausalEdge(BaseModel):
    """A directed causal edge between variables."""

    cause: str = Field(description="Name of cause variable")
    effect: str = Field(description="Name of effect variable")
    description: str = Field(description="Why this causal relationship exists")
    lagged: bool = Field(
        default=True,
        description=(
            "If True, effect at t is caused by cause at t-1. "
            "If False (contemporaneous), effect at t is caused by cause at t. "
            "Cross-timescale edges are always lagged."
        ),
    )
    # Computed field - set by DSEMStructure validator
    lag_hours: int | None = Field(
        default=None,
        description="Lag in hours. Computed from granularities - do not set manually.",
    )


def compute_lag_hours(
    cause_granularity: str | None,
    effect_granularity: str | None,
    lagged: bool,
) -> int:
    """Compute lag in hours based on granularities and lagged flag.

    Rules (Markov property):
    - Same timescale, contemporaneous: lag = 0
    - Same timescale, lagged: lag = 1 granularity unit
    - Cross timescale: lag = coarser granularity (always lagged)
    """
    cause_hours = GRANULARITY_HOURS.get(cause_granularity, 0) if cause_granularity else 0
    effect_hours = GRANULARITY_HOURS.get(effect_granularity, 0) if effect_granularity else 0

    # Cross-timescale: always use coarser granularity
    if cause_granularity != effect_granularity:
        return max(cause_hours, effect_hours)

    # Same timescale: depends on lagged flag
    if lagged:
        return cause_hours  # 1 unit of the shared granularity
    return 0  # contemporaneous


class DSEMStructure(BaseModel):
    """Complete DSEM specification."""

    dimensions: list[Dimension] = Field(description="Variables in the model")
    edges: list[CausalEdge] = Field(description="Causal edges including cross-lags")

    @model_validator(mode="after")
    def validate_and_compute_lags(self):
        """Validate structure and compute lag_hours for each edge."""
        # Exactly one outcome required
        outcomes = [d for d in self.dimensions if d.is_outcome]
        if len(outcomes) == 0:
            raise ValueError("Exactly one dimension must have is_outcome=true")
        if len(outcomes) > 1:
            names = [d.name for d in outcomes]
            raise ValueError(f"Only one outcome allowed, got {len(outcomes)}: {names}")

        dim_map = {d.name: d for d in self.dimensions}

        for edge in self.edges:
            # Check variables exist
            if edge.cause not in dim_map:
                raise ValueError(f"Edge cause '{edge.cause}' not in dimensions")
            if edge.effect not in dim_map:
                raise ValueError(f"Edge effect '{edge.effect}' not in dimensions")

            cause_dim = dim_map[edge.cause]
            effect_dim = dim_map[edge.effect]

            # No inbound edges to exogenous
            if effect_dim.role == Role.EXOGENOUS:
                raise ValueError(f"Exogenous variable '{edge.effect}' cannot be an effect")

            cause_gran = cause_dim.causal_granularity
            effect_gran = effect_dim.causal_granularity

            # Contemporaneous (lagged=False) requires same timescale
            # Exception: time-invariant causes (granularity=None) can affect any timescale
            both_time_varying = cause_gran is not None and effect_gran is not None
            if not edge.lagged and both_time_varying and cause_gran != effect_gran:
                raise ValueError(
                    f"Contemporaneous edge (lagged=false) requires same timescale: "
                    f"{edge.cause} ({cause_gran}) -> {edge.effect} ({effect_gran})"
                )

            # Compute and set lag_hours
            edge.lag_hours = compute_lag_hours(cause_gran, effect_gran, edge.lagged)

        # Check acyclicity within time slice (contemporaneous edges only)
        # Lagged edges can form cycles across time - that's the point of DSEMs
        contemporaneous_edges = [(e.cause, e.effect) for e in self.edges if not e.lagged]
        if contemporaneous_edges:
            import networkx as nx

            G = nx.DiGraph(contemporaneous_edges)
            if not nx.is_directed_acyclic_graph(G):
                cycles = list(nx.simple_cycles(G))
                raise ValueError(
                    f"Contemporaneous edges form cycle(s) within time slice: {cycles}. "
                    "DSEMs must be acyclic within time slice. Use lagged=true for feedback loops."
                )

        return self

    def to_networkx(self):
        """Convert to NetworkX DiGraph (compatible with DoWhy)."""
        import networkx as nx

        G = nx.DiGraph()
        for dim in self.dimensions:
            G.add_node(dim.name, **dim.model_dump())
        for edge in self.edges:
            G.add_edge(
                edge.cause,
                edge.effect,
                description=edge.description,
                lag_hours=edge.lag_hours,
                lagged=edge.lagged,
            )
        return G

    def to_edge_list(self) -> list[tuple[str, str]]:
        """Convert to edge list format."""
        return [(e.cause, e.effect) for e in self.edges]


def validate_structure(data: dict) -> tuple[DSEMStructure | None, list[str]]:
    """Validate a structure dict, collecting ALL errors instead of failing on first.

    Args:
        data: Dictionary to validate as DSEMStructure

    Returns:
        Tuple of (validated structure or None, list of error messages)
    """
    errors = []

    # First check basic structure
    if not isinstance(data, dict):
        return None, ["Input must be a dictionary"]

    dimensions = data.get("dimensions", [])
    edges = data.get("edges", [])

    if not isinstance(dimensions, list):
        errors.append("'dimensions' must be a list")
        dimensions = []
    if not isinstance(edges, list):
        errors.append("'edges' must be a list")
        edges = []

    # Validate each dimension individually to collect all errors
    valid_dimensions = []
    dim_names = set()

    for i, dim_data in enumerate(dimensions):
        if not isinstance(dim_data, dict):
            errors.append(f"dimensions[{i}]: must be a dictionary")
            continue

        name = dim_data.get("name", f"<unnamed_{i}>")

        # Check for duplicate names
        if name in dim_names:
            errors.append(f"Duplicate dimension name: '{name}'")
        dim_names.add(name)

        try:
            dim = Dimension.model_validate(dim_data)
            valid_dimensions.append(dim)
        except Exception as e:
            # Extract error messages from Pydantic validation
            error_msg = str(e)
            # Clean up Pydantic error formatting
            if "validation error" in error_msg.lower():
                for line in error_msg.split("\n")[1:]:
                    line = line.strip()
                    if line and not line.startswith("For further"):
                        errors.append(f"dimensions[{i}] ({name}): {line}")
            else:
                errors.append(f"dimensions[{i}] ({name}): {error_msg}")

    # Build dim map for edge validation
    dim_map = {d.name: d for d in valid_dimensions}

    # Validate each edge individually
    valid_edges = []
    for i, edge_data in enumerate(edges):
        if not isinstance(edge_data, dict):
            errors.append(f"edges[{i}]: must be a dictionary")
            continue

        cause = edge_data.get("cause", "<missing>")
        effect = edge_data.get("effect", "<missing>")
        edge_label = f"edges[{i}] ({cause} -> {effect})"

        # Check basic edge structure
        try:
            edge = CausalEdge.model_validate(edge_data)
        except Exception as e:
            errors.append(f"{edge_label}: {e}")
            continue

        # Check references
        if edge.cause not in dim_map:
            errors.append(f"{edge_label}: cause '{edge.cause}' not in dimensions")
            continue
        if edge.effect not in dim_map:
            errors.append(f"{edge_label}: effect '{edge.effect}' not in dimensions")
            continue

        cause_dim = dim_map[edge.cause]
        effect_dim = dim_map[edge.effect]

        # Check exogenous constraint
        if effect_dim.role == Role.EXOGENOUS:
            errors.append(f"{edge_label}: exogenous variable '{edge.effect}' cannot be an effect")
            continue

        # Check contemporaneous timescale constraint
        cause_gran = cause_dim.causal_granularity
        effect_gran = effect_dim.causal_granularity
        both_time_varying = cause_gran is not None and effect_gran is not None
        if not edge.lagged and both_time_varying and cause_gran != effect_gran:
            errors.append(
                f"{edge_label}: contemporaneous edge requires same timescale, "
                f"got {cause_gran} -> {effect_gran}"
            )
            continue

        # Compute lag_hours
        edge.lag_hours = compute_lag_hours(cause_gran, effect_gran, edge.lagged)
        valid_edges.append(edge)

    # Check outcome constraints
    outcomes = [d for d in valid_dimensions if d.is_outcome]
    if len(outcomes) == 0:
        errors.append("Exactly one dimension must have is_outcome=true (none found)")
    elif len(outcomes) > 1:
        names = [d.name for d in outcomes]
        errors.append(f"Only one outcome allowed, got {len(outcomes)}: {names}")

    # Check acyclicity of contemporaneous edges
    contemporaneous_edges = [(e.cause, e.effect) for e in valid_edges if not e.lagged]
    if contemporaneous_edges:
        import networkx as nx

        G = nx.DiGraph(contemporaneous_edges)
        if not nx.is_directed_acyclic_graph(G):
            cycles = list(nx.simple_cycles(G))
            errors.append(
                f"Contemporaneous edges form cycle(s): {cycles}. "
                "Use lagged=true for feedback loops."
            )

    # If no errors, build and return the structure
    if not errors:
        try:
            structure = DSEMStructure(dimensions=valid_dimensions, edges=valid_edges)
            return structure, []
        except Exception as e:
            errors.append(f"Final validation failed: {e}")

    return None, errors
