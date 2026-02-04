# AISI inspect Framework Best Practices

Docs: https://inspect.aisi.org.uk/

## OpenRouter Provider

- Set `OPENROUTER_API_KEY` environment variable
- Model format: `openrouter/<provider>/<model>` e.g. `openrouter/google/gemini-2.5-pro-preview-06-05`
- Requires `openai` package: `uv add openai`

## Model Interaction

- Use `get_model("openrouter/...")` to get specific model
- Use `model.generate(messages)` for single-turn generation
- Use `model.generate(messages, config=GenerateConfig(...))` for custom settings
- Use `model.generate_loop(messages, tools=...)` for multi-turn with tool calling

## Messages

- `ChatMessageSystem(content=...)` for system prompts
- `ChatMessageUser(content=...)` for user messages
- Messages are lists passed to generate()

## Structured Output

- Use Pydantic models for response schemas
- Parse JSON from response, handle markdown code blocks
- Validate with `Model.model_validate(data)`

## Patterns

```python
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, get_model

async def my_agent(prompt: str) -> dict:
    model = get_model("openrouter/google/gemini-2.5-pro-preview-06-05")
    messages = [
        ChatMessageSystem(content="You are a helpful assistant."),
        ChatMessageUser(content=prompt),
    ]
    config = GenerateConfig(temperature=0.7, max_tokens=4096)
    response = await model.generate(messages, config=config)
    return parse_json(response.completion)
```

## Sync Wrapper

```python
def sync_wrapper(prompt: str) -> dict:
    import asyncio
    return asyncio.run(my_agent(prompt))
```

## Evals with Tasks

- Define tasks with `@task` decorator returning `Task(dataset, solver, scorer)`
- Use `MemoryDataset([Sample(...)])` for programmatic datasets
- Sample fields: `input` (required), `target`, `id`, `metadata`
- Solvers: `system_message(prompt)`, `generate()` (calls model)
- Custom scorers with `@scorer(metrics=[accuracy(), stderr()])`

```python
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState, generate, system_message

@scorer(metrics=[accuracy(), stderr()])
def my_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion
        is_correct = validate(completion)
        return Score(
            value="C" if is_correct else "I",
            answer=completion[:500],
            explanation="reason",
            metadata={"extra": "data"},
        )
    return score

@task
def my_eval(param: int = 5):
    return Task(
        dataset=MemoryDataset([
            Sample(input="prompt", id="s1", metadata={"key": "val"})
        ]),
        solver=[system_message("system prompt"), generate()],
        scorer=my_scorer(),
    )
```

## Running Evals

```bash
inspect eval path/to/eval.py --model openrouter/google/gemini-2.0-flash-001
inspect eval path/to/eval.py --model openrouter/anthropic/claude-sonnet-4 -T param=10 --limit 5
```

## Reading Eval Logs

See [running_evals.md](running_evals.md) for detailed instructions on reading and debugging eval logs.
