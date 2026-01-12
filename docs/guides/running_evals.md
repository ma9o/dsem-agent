# Running Evaluations

Evaluate LLM performance on structure proposal tasks using Inspect AI. Only top-tier models with max thinking budget are used.

## Run all models in parallel

```bash
# Run all 5 models concurrently (recommended)
uv run python evals/scripts/run_parallel_evals.py

# Run specific models using aliases
uv run python evals/scripts/run_parallel_evals.py --models claude gemini gpt

# Customize parameters
uv run python evals/scripts/run_parallel_evals.py -n 10 --seed 123

# Available aliases: claude, gemini, gpt, deepseek, kimi
```

## Run individual models

```bash
uv run inspect eval evals/eval1_orchestrator_structure.py \
    --model openrouter/anthropic/claude-opus-4.5

# View detailed results
uv run inspect view
```

Logs are saved to `logs/` directory. The eval scores models on cumulative points for valid DSEM structures.
