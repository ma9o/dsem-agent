"""Pipeline stages."""

from .stage1_structure import (
    load_orchestrator_chunks,
    propose_structure,
)
from .stage2_workers import (
    load_worker_chunks,
    populate_dimensions,
    aggregate_measurements,
)
from .stage3_identifiability import (
    check_identifiability,
    run_sensitivity_analysis,
)
from .stage4_model import (
    specify_model,
    elicit_priors,
)
from .stage5_inference import (
    fit_model,
    run_interventions,
)

__all__ = [
    # Stage 1
    "load_orchestrator_chunks",
    "propose_structure",
    # Stage 2
    "load_worker_chunks",
    "populate_dimensions",
    "aggregate_measurements",
    # Stage 3
    "check_identifiability",
    "run_sensitivity_analysis",
    # Stage 4
    "specify_model",
    "elicit_priors",
    # Stage 5
    "fit_model",
    "run_interventions",
]
