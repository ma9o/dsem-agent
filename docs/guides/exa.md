# Exa Literature Search Best Practices

Docs: https://exa.ai/docs/sdks/python-sdk-specification

## Setup

- Set `EXA_API_KEY` environment variable (in `.env`)
- Install: `uv add exa-py`

## Async Client

```python
from exa_py import AsyncExa
exa = AsyncExa()  # Reads EXA_API_KEY from env
```

## Research API (Preferred for Literature)

Use the Research API for in-depth literature searches with structured output:

```python
# Create research task with structured output
research = await exa.research.create(
    instructions="Find effect sizes for stress -> mood relationship",
    output_schema={
        "type": "object",
        "properties": {
            "effect_sizes": {"type": "array", "items": {"type": "string"}},
            "sources": {"type": "array", "items": {"type": "object"}}
        }
    },
    model="exa-research",  # or exa-research-fast, exa-research-pro
)

# Poll until complete
result = await exa.research.poll_until_finished(
    research.id,
    timeout_ms=120000,
)

if result.status == "completed":
    data = result.data  # Structured JSON per output_schema
```

## Search API (Quick Lookups)

For simpler searches without structured output:

```python
results = await exa.search(
    query="meta-analysis stress anxiety effect size",
    num_results=10,
    contents={"text": True},
    start_published_date="2020-01-01",
)
```

## Models

- `exa-research-fast`: Quick, lower quality
- `exa-research`: Balanced (default)
- `exa-research-pro`: Highest quality, slower

## Error Handling

- Always wrap Exa calls in try/except to avoid pipeline failures
- Return `None` or empty results on failure
- Literature search is optional enhancement, not critical path
