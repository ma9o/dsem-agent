# Schema

## Source Files

- **Orchestrator schemas:** `src/causal_agent/orchestrator/schemas.py`
  - `Dimension` - A variable in the causal model
  - `CausalEdge` - A directed causal edge between variables
  - `DSEMStructure` - Complete DSEM specification
  - `validate_structure()` - Validation with comprehensive error collection

- **Worker schemas:** `src/causal_agent/workers/schemas.py`
  - `WorkerOutput` - Extraction results from a single chunk

---

## Validation Rules

The following must hold for a valid `DSEMStructure`:

1. **Latent validity:** If `is_latent=True`, then `role='exogenous'` and `causal_granularity=None`
2. **Endogenous requires time-varying:** If `role='endogenous'`, then `causal_granularity` must not be `None`
3. **No inbound edges to exogenous:** If `role='exogenous'`, variable cannot appear as `effect` in any edge
4. **Contemporaneous same-scale only:** If `lag=0`, cause and effect must have identical `causal_granularity`
5. **Same-scale lag constraint (Markov):** If cause and effect have identical `causal_granularity` and `lag > 0`, lag must equal exactly one granularity unit in hours
6. **Cross-scale lag constraint (Markov):** If cause and effect have different `causal_granularity`, `lag` must equal exactly `max(cause_granularity, effect_granularity)` in hours
7. **Aggregation requirement (edges):** If cause `causal_granularity` is finer than effect `causal_granularity`, `aggregation` must be specified
8. **Aggregation prohibition (edges):** If cause `causal_granularity` is coarser or equal to effect `causal_granularity`, `aggregation` must be `None`
