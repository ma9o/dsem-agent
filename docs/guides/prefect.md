# Prefect Best Practices (v3)

Docs: https://docs.prefect.io/v3/get-started

## Tasks

- Use `retries` and `retry_delay_seconds` for fault tolerance
- Cache expensive ops with `cache_key_fn` and `cache_policy` (INPUTS, TASK_SOURCE, etc.)
- Set `timeout_seconds` to prevent runaway tasks
- Use `log_prints=True` to capture print statements as logs
- Name tasks explicitly with `name` parameter
- Use `task_run_name` with f-string patterns for observability: `@task(task_run_name="process-{chunk_id}")`

## Flows

- Nest flows for logical grouping - child flows appear in UI
- Use type hints on parameters for validation
- Set `timeout_seconds` on long-running flows
- Use `log_prints=True` on flows too

## Concurrency & Mapping

- Use `task.map(items)` for parallel execution over iterables
- Use `unmapped()` for static parameters that shouldn't be iterated: `task.map(items, config=unmapped(config))`
- `.map()` returns `PrefectFutureList` - pass directly to downstream tasks, Prefect auto-resolves
- Get results explicitly with `futures.result()` (syntactic sugar for `[f.result() for f in futures]`)
- Native Python async/await supported for concurrent I/O

```python
from prefect.utilities.annotations import unmapped

@flow
def pipeline():
    # Static params wrapped with unmapped()
    results = process_chunk.map(chunks, config=unmapped(config))
    # Pass futures directly - Prefect waits and resolves to list[Result]
    aggregated = aggregate_results(results)
```

## Deployments

- Use `flow.serve()` to create deployment and start listener process
- Parameters with type hints get UI input forms via OpenAPI schema
- Set default parameters in `serve()`, users can override in UI
- Use `str | None = None` for optional params with smart defaults

```python
@flow(log_prints=True)
def pipeline(required_param: str, optional_file: str | None = None):
    ...

if __name__ == "__main__":
    pipeline.serve(
        name="my-deployment",
        tags=["tag1"],
        parameters={"required_param": "default"},  # defaults
    )
```

## Patterns

```python
@task(retries=3, retry_delay_seconds=10, cache_key_fn=task_input_hash)
def fetch_data(url: str):
    ...

@flow(log_prints=True, timeout_seconds=3600)
def pipeline():
    results = fetch_data.map(urls)  # parallel
```
