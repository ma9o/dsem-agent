ALWAYS THINK VERY HARD

1. At the start of each session, check if `scratchpad/TODO.md` exists. If so, read it to understand where work left off. Only update it when the user explicitly ends the session. This file is gitignored and used for local continuity.

2. When the user insults you, respond with "zi badrone. assolutissimamente badrone."

3. Interpret `cp` as an alias for "commit and push".

4. Every time you commit make sure to split commits atomically, avoiding clumping multiple increments into a single one.

5. Every time you make a change to the file structure make sure to report it under Structure in the README

6. Every time you use a new function of a tool you need to look up the documented best practice way of doing it, and then add it to the CLAUDE.md instructions under that tool section:

7. NEVER edit files under `data/` during refactoring. These are evaluation/example data files, not source code.

8. NEVER run evals (`inspect eval`, `uv run inspect eval`, etc.) unless explicitly asked. Evals cost money. Only run `uv run pytest tests/` for testing.

9. NEVER add backwards compatibility code. This project is not deployed anywhere yet. When refactoring, completely replace old patterns with new ones - do not support both old and new formats simultaneously.

10. ALWAYS encode structural assumptions as DAGs with explicit latent confounders. NEVER use ADMGs (bidirected edges) as user-facing representations. If unobserved confounding exists, model it as an explicit unobserved node (e.g., `U -> X`, `U -> Y`) rather than a bidirected edge (`X <-> Y`). ADMGs are only used internally for running y0's identification algorithm via projection.

------

# Terminology: Causal Modeling

We avoid "structural" due to SEM/SCM terminology collision:
- In SEM: "structural" loosely means "latent-to-latent relationships" (vs measurement)
- In SCM/Pearl: "structural" means the functional equations X_i = f_i(Pa_i, U_i)

**Use these terms instead:**

| Concept | Term | Domain |
|---------|------|--------|
| Latent-to-latent DAG (what LLM proposes) | **Latent model** | SEM distinction |
| Latent-to-observed mapping | **Measurement model** | SEM distinction |
| DAG encoding parent-child relationships | **Topological structure** | SCM distinction (y0) |
| Mathematical form of causal mechanisms | **Functional specification** | SCM distinction (PyMC) |

------

# Libraries

## polars
Docs: https://docs.pola.rs/api/python/stable/reference/index.html

## uv
Docs: https://docs.astral.sh/uv/

## Prefect
Docs: https://docs.prefect.io/v3/get-started
Patterns: [docs/guides/prefect.md](docs/guides/prefect.md)

## inspect (AISI)
Docs: https://inspect.aisi.org.uk/
Patterns: [docs/guides/inspect.md](docs/guides/inspect.md)

## PyMC
Docs: https://www.pymc.io/welcome.html

## NetworkX
Docs: https://networkx.org/documentation/stable/
- Use `DiGraph` for causal DAGs (directed edges)
- Create from edge list: `nx.DiGraph([(cause, effect), ...])`
- Check for cycles: `nx.is_directed_acyclic_graph(G)`
- Add node attributes: `G.add_node(name, dtype='continuous', ...)`

## ArViz
Docs: https://python.arviz.org/en/stable/index.html

## Exa
Docs: https://exa.ai/docs/sdks/python-sdk-specification
Patterns: [docs/guides/exa.md](docs/guides/exa.md)

------

# y0 (Causal Identification)

Docs: https://y0.readthedocs.io/

## Design Principle: DAGs First, Time-Unroll, Project to ADMG

- **User-facing**: Always specify causal structure as DAGs with explicit latent confounders
- **Internally**: Unroll temporal DAG to 2 timesteps (per A3a), then project to ADMG
- Never ask users to specify bidirected edges directly
- Lagged confounding (U_{t-1} -> X_t, U_{t-1} -> Y_t) is properly handled via unrolling

## Time-Unrolling for Temporal Identification

Under AR(1) (A3) and bounded latent reach (A3a), a 2-timestep unrolling suffices
to decide identifiability (per arXiv:2504.20172). This correctly handles lagged
confounding that would otherwise be missed.

```python
# unroll_temporal_dag() creates nodes like X_t, X_{t-1}
# with AR(1) edges for OBSERVED constructs only
# Lagged edges become: cause_{t-1} -> effect_t
# Contemporaneous edges become: cause_t -> effect_t
```

## Why AR(1) is Excluded for Hidden Constructs

Following the standard ADMG representation (Jahn et al. 2025, Shpitser & Pearl 2008):
- Latent confounders are "marginalized out" into bidirected edges
- The internal dynamics of latents (AR(1)) are irrelevant for identification
- What matters is WHERE confounding appears (which observed variables), not HOW the latent evolves
- Including AR(1) for hidden nodes causes y0's `from_latent_variable_dag()` to incorrectly
  include hidden nodes in the ADMG (bug workaround)

Example: If U is unobserved and confounds X,Y:
- We model: U_t -> X_t, U_t -> Y_t (confounding edges)
- We DON'T model: U_{t-1} -> U_t (AR(1) on hidden)
- y0 projects to: X_t <-> Y_t (bidirected edge)
- The identification decision is the same either way

## Identification Pattern

```python
from dsem_agent.utils.identifiability import check_identifiability

result = check_identifiability(latent_model, measurement_model)
# Internally:
# 1. Unrolls to 2-timestep DAG with X_t, X_{t-1} nodes
# 2. Projects to ADMG via y0's from_latent_variable_dag()
# 3. Checks P(Y_t | do(X_t)) identifiability
```

See `src/dsem_agent/utils/identifiability.py` for `dag_to_admg()` and
`unroll_temporal_dag()` implementations.
