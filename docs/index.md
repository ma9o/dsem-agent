# Documentation Index

This index helps coding agents navigate the documentation structure.

## Structure

```
docs/
├── index.md           # This file
├── modeling/          # Theoretical foundations (DSEM, assumptions, scope)
├── reference/         # Technical specs (schemas, pipeline)
└── guides/            # Practical usage (quickstart, data, evals)
```

## Quick Links by Task

**Understanding the modeling approach:**
- Start with `modeling/dsem_overview.md` for what DSEMs are
- Read `modeling/assumptions.md` for constraints on what can be modeled
- Check `modeling/scope.md` for supported/excluded variable types

**Implementing or modifying:**
- `reference/schema.md` for Pydantic models and validation
- `reference/pipeline.md` for stage-by-stage breakdown

**Running the system:**
- `guides/quickstart.md` for installation and first run
- `guides/data_workflow.md` for data preprocessing
- `guides/running_evals.md` for evaluation
