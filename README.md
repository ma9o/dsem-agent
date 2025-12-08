# causal-agent

This project explores an end-to-end, LLM-orchestrated framework for causal inference over long-context, multi-source data (e.g. large document collections or aggregated web search). An “orchestrator” LLM proposes candidate variables, time granularities, and a causal DAG; “worker” LLMs then populate those dimensions at scale, after which we use DoWhy for identifiability checks and sensitivity analysis, and PyMC for full Bayesian GLM estimation with LLM-elicited priors. The goal is to build a system that not only estimates causal effects and counterfactuals from messy, high-dimensional evidence, but also knows when to trust those numeric estimates and when to fall back to purely structural, qualitative reasoning.

## Tech stack

- polars dataframes
- uv
- Prefect for pipeline orchestration
- AISI's inspect agent framework
- PyMC for functional specification and inference
- DoWhy for structural specification and identificability checks
- ArViz for posterior diagnostics

## Pipeline

[1] We sample a few data chunks and we have the (expensive) orchestrator LLM suggests dimensionality, autocorrelations, time granularities and a structural DAG.

[2] For each chunk of data, (cheaper) worker LLMs populate the candidate dimensions, potentially suggesting new ones which the data chunk has elicited. This process is parallelized, scaling to millions of tokens. At the end, the orchestrator incorporates suggestions (3-way merge) and we backfill.

[3] The orchestrator then runs DoWhy to check if our target causal effects are identifiable. In case of unobserved confounders that make effects unidentifiable, we run a sensitivity analysis (Cinelli-Hazlett) on a naive linear model and continue if the bias is bounded (and return control to the user if not).

[4] The orchestrator then specifies the statistical model (GLMs) in PyMC and queries the workers for priors. (see: Zhu et al. 2024).

[5] Finally we fit the model with PyMC and then the orchestrator runs the proposed interventions and counterfactual simulations and then returns them to the user, ranked by effect size.

## Data Workflow

The pipeline is **input-agnostic** and operates on preprocessed text chunks. Raw data sources are converted to a standardized format before running the causal inference pipeline.

### Directory Structure

```
data/
├── google-takeout/    # Raw zip exports (gitignored)
└── preprocessed/      # Converted text files (gitignored)
```

### Preprocessing

1. Place raw data exports in the appropriate `data/` subdirectory
2. Run the preprocessing script to convert to text chunks:

```bash
# Process all zips in data/google-takeout/
uv run python scripts/preprocess_google_takeout.py

# Process a specific file
uv run python scripts/preprocess_google_takeout.py -i data/google-takeout/export.zip
```

This outputs `data/preprocessed/<filename>_<timestamp>.txt` with text chunks separated by `---`.

### Running the Pipeline

**Option 1: Direct execution**
```bash
uv run python -c "
from causal_agent.flows.pipeline import causal_inference_pipeline

# Uses latest preprocessed file automatically
causal_inference_pipeline(target_effects=['effect_of_X_on_Y'])

# Or specify a file
causal_inference_pipeline(
    target_effects=['effect_of_X_on_Y'],
    input_file='export_20241208_153022.txt',
)
"
```

**Option 2: Serve with UI**
```bash
# Terminal 1: Start Prefect server
uv run prefect server start

# Terminal 2: Serve the flow
uv run python -m causal_agent.flows.pipeline
```

Then open http://localhost:4200 to trigger runs with custom parameters.

## Structure