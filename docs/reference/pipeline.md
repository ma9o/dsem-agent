# Pipeline

## Overview

[1] Global Hypothesis (Orchestrator): We feed a small data sample to the Orchestrator LLM to generate a structural "prior." It establishes the Global Vibe: proposing a candidate structural-only DAG, defining the dimensions to extract, and setting the time granularity based on the user's question.

[2] Distributed Discovery (Workers): Worker LLMs process the full dataset in parallel chunks, extracting data for the proposed dimensions. (Future: workers may also critique the global graph based on local evidence, suggesting new confounders found only in specific chunks. The Orchestrator would then reconcile these structural suggestions into a unified model. Currently disabled—measurement extraction is the priority.)

[3] The orchestrator then runs DoWhy to check if our target causal effects are identifiable. In case of unobserved confounders that make effects unidentifiable, we run a sensitivity analysis (Cinelli-Hazlett) on a naive linear model and continue if the bias is bounded (and return control to the user if not).

[4] The orchestrator then specifies the statistical model (GLMs) in PyMC and queries the workers for priors. (see: Zhu et al. 2024).

[5] Finally we fit the model with PyMC and then the orchestrator runs the proposed interventions and counterfactual simulations and then returns them to the user, ranked by effect size.

---

## Stage Files

Stages are in `src/causal_agent/flows/stages/` with naming convention `stage{N}_{name}.py`.

---

## Implementation Notes

### Cross-Timescale Edge Aggregation (TODO: Functional Layer)

When implementing the functional layer, cross-timescale edges require aggregation:

- **Finer → Coarser** (e.g., hourly → daily): Aggregate the cause variable using its dimension's `aggregation` field. For example, if hourly `steps` (aggregation: "sum") affects daily `mood`, sum the 24 hourly step counts before modeling the relationship.

- **Coarser → Finer** (e.g., weekly → daily): No aggregation needed. The coarser variable's value is broadcast to all finer time points within its period.

The aggregation function is defined once per dimension (not per edge) because it captures the semantic meaning of how that variable should be rolled up (e.g., steps are summed, temperature is averaged).
