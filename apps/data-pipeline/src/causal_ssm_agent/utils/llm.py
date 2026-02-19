"""Shared LLM utilities for multi-turn generation."""

import dataclasses
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
    Model,
    ModelOutput,
)
from inspect_ai.tool import Tool, tool

if TYPE_CHECKING:
    from inspect_ai.model import ChatMessage

    from causal_ssm_agent.orchestrator.schemas import LatentModel


# ---------------------------------------------------------------------------
# Trace dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TraceMessage:
    """A single message in an LLM trace."""

    role: str
    content: str
    reasoning: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_name: str | None = None
    tool_result: str | None = None
    tool_is_error: bool = False


@dataclass
class TraceUsage:
    """Token usage for an LLM trace."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int | None = None


@dataclass
class LLMTrace:
    """Full trace of an LLM multi-turn conversation."""

    messages: list[TraceMessage] = field(default_factory=list)
    model: str = ""
    total_time_seconds: float = 0.0
    usage: TraceUsage = field(default_factory=TraceUsage)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _chat_message_to_trace(msg: "ChatMessage") -> TraceMessage:
    """Convert an inspect_ai ChatMessage to a TraceMessage."""
    from inspect_ai._util.content import ContentReasoning, ContentText

    role = msg.role
    content_text = ""
    reasoning_text = None
    tool_calls_list = None
    tool_name = None
    tool_result = None
    tool_is_error = False

    # Extract text and reasoning from content
    if isinstance(msg.content, str):
        content_text = msg.content
    elif isinstance(msg.content, list):
        text_parts = []
        reasoning_parts = []
        for part in msg.content:
            if isinstance(part, ContentText):
                text_parts.append(part.text)
            elif isinstance(part, ContentReasoning):
                reasoning_parts.append(part.reasoning)
        content_text = "\n".join(text_parts)
        if reasoning_parts:
            reasoning_text = "\n".join(reasoning_parts)

    # Extract tool calls from assistant messages
    if isinstance(msg, ChatMessageAssistant) and msg.tool_calls:
        tool_calls_list = [
            {"name": tc.function, "arguments": tc.arguments} for tc in msg.tool_calls
        ]

    # Extract tool results from tool messages
    if isinstance(msg, ChatMessageTool):
        tool_name = msg.function
        tool_result = content_text
        tool_is_error = msg.error is not None

    return TraceMessage(
        role=role,
        content=content_text,
        reasoning=reasoning_text,
        tool_calls=tool_calls_list,
        tool_name=tool_name,
        tool_result=tool_result,
        tool_is_error=tool_is_error,
    )


def _build_trace(all_messages: list["ChatMessage"], output: ModelOutput) -> LLMTrace:
    """Build an LLMTrace from a final message list and ModelOutput."""
    messages = [_chat_message_to_trace(m) for m in all_messages]
    usage = TraceUsage()
    if output.usage:
        usage = TraceUsage(
            input_tokens=output.usage.input_tokens,
            output_tokens=output.usage.output_tokens,
            total_tokens=output.usage.total_tokens,
            reasoning_tokens=output.usage.reasoning_tokens,
        )
    return LLMTrace(
        messages=messages,
        model=output.model or "",
        total_time_seconds=output.time or 0.0,
        usage=usage,
    )


# ---------------------------------------------------------------------------
# Type aliases for generate functions (unified)
# ---------------------------------------------------------------------------

GenerateFn = Callable[..., Awaitable[str]]

# Backward-compatible aliases
OrchestratorGenerateFn = GenerateFn
WorkerGenerateFn = GenerateFn


def get_generate_config() -> GenerateConfig:
    """Get standard GenerateConfig for all model calls.

    Reads settings from config.yaml llm section.
    """
    from causal_ssm_agent.utils.config import get_config

    llm = get_config().llm
    return GenerateConfig(
        max_tokens=llm.max_tokens,
        timeout=llm.timeout,
        reasoning_effort=llm.reasoning_effort,
        reasoning_history="all",  # Preserve reasoning across tool calls (required by Gemini)
    )


def dict_messages_to_chat(messages: list[dict]) -> list["ChatMessage"]:
    """Convert dict messages to ChatMessage objects.

    Args:
        messages: List of dicts with 'role' and 'content' keys

    Returns:
        List of ChatMessage objects (ChatMessageSystem or ChatMessageUser)
    """
    chat_messages = []
    for msg in messages:
        if msg["role"] == "system":
            chat_messages.append(ChatMessageSystem(content=msg["content"]))
        elif msg["role"] == "user":
            chat_messages.append(ChatMessageUser(content=msg["content"]))
    return chat_messages


# ---------------------------------------------------------------------------
# Generate function factory (unified for orchestrator and worker)
# ---------------------------------------------------------------------------


def make_generate_fn(
    model: Model,
    config: GenerateConfig | None = None,
    trace_capture: dict | None = None,
) -> GenerateFn:
    """Create a generate function for LLM calls.

    The returned function has signature: (messages, tools=None, follow_ups=None) -> str
    Works for both orchestrator stages (with follow_ups) and worker stages (without).

    Args:
        model: The model to use for generation
        config: Optional generation config (uses get_generate_config() if None)
        trace_capture: Optional dict for capturing the LLM trace

    Returns:
        An async function that handles multi-turn generation with tools and follow-ups
    """
    if config is None:
        config = get_generate_config()

    async def generate(
        messages: list,
        tools: list | None = None,
        follow_ups: list[str] | None = None,
    ) -> str:
        chat_messages = dict_messages_to_chat(messages)

        if follow_ups or tools:
            return await multi_turn_generate(
                messages=chat_messages,
                model=model,
                follow_ups=follow_ups,
                tools=tools or [],
                config=config,
                trace_capture=trace_capture,
            )
        else:
            response = await model.generate(chat_messages, config=config)
            return response.completion

    return generate


# Backward-compatible aliases
make_orchestrator_generate_fn = make_generate_fn
make_worker_generate_fn = make_generate_fn


def parse_json_response(content: str) -> dict:
    """Parse JSON from model response, handling markdown code blocks."""
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]

    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Content length: {len(content)}")
        print(f"Content preview: {content[:500]}...")
        raise ValueError(f"Failed to parse model response as JSON: {e}") from e


# ---------------------------------------------------------------------------
# Shared validation logic for all validation tools
# ---------------------------------------------------------------------------


def _validate_json_and_format(
    json_str: str,
    validate_fn: Callable[[dict], tuple[Any, list[str]]],
    capture: dict | None = None,
    capture_key: str | None = None,
    capture_result: bool = False,
) -> str:
    """Parse JSON, validate, and format errors.

    Args:
        json_str: Raw JSON string to parse
        validate_fn: (data_dict) -> (validated_result_or_None, error_list)
        capture: Optional dict to store successful results in
        capture_key: Key under which to store in capture dict
        capture_result: If True, store the validated result; if False, store raw data
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return f"JSON parse error: {e}"

    result, errors = validate_fn(data)

    if not errors:
        if capture is not None and capture_key:
            capture[capture_key] = result if capture_result else data
        return "VALID"

    return "VALIDATION ERRORS:\n" + "\n".join(f"- {e}" for e in errors)


# ---------------------------------------------------------------------------
# Validation tool factories
# ---------------------------------------------------------------------------


def make_validate_latent_model_tool() -> tuple[Tool, dict]:
    """Create a validation tool for latent model JSON that captures the last valid result.

    Returns:
        Tuple of (tool, capture_dict). After generate_loop, check
        capture["latent"] for the last validated LatentModel dict (or None).
    """
    capture: dict = {}

    @tool
    def validate_latent_model_tool():
        """Tool for validating latent model JSON (Stage 1a)."""

        async def execute(structure_json: str) -> str:
            """
            Validate a latent model and return all validation errors.

            Args:
                structure_json: The JSON string containing the latent model to validate.

            Returns:
                "VALID" if the structure passes validation, otherwise a list of all errors found.
            """
            from causal_ssm_agent.orchestrator.schemas import validate_latent_model

            return _validate_json_and_format(
                structure_json,
                validate_latent_model,
                capture=capture,
                capture_key="latent",
            )

        return execute

    return validate_latent_model_tool(), capture


def make_validate_measurement_model_tool(latent_model: "LatentModel") -> tuple[Tool, dict]:
    """Create a validation tool for measurement model, bound to a latent model.

    Args:
        latent_model: The latent model to validate against

    Returns:
        Tuple of (tool, capture_dict). After generate_loop, check
        capture["measurement"] for the last validated measurement dict (or None).
    """
    capture: dict = {}

    @tool
    def validate_measurement_model_tool():
        """Tool for validating measurement model JSON (Stage 1b)."""

        async def execute(measurement_json: str) -> str:
            """
            Validate a measurement model and return all validation errors.

            Args:
                measurement_json: The JSON string containing the measurement model to validate.

            Returns:
                "VALID" if the model passes validation, otherwise a list of all errors found.
            """
            from causal_ssm_agent.orchestrator.schemas import validate_measurement_model

            return _validate_json_and_format(
                measurement_json,
                lambda data: validate_measurement_model(data, latent_model),
                capture=capture,
                capture_key="measurement",
            )

        return execute

    return validate_measurement_model_tool(), capture


def make_validate_model_spec_tool(
    causal_spec: dict,
    *,
    resolved_likelihoods: list[dict] | None = None,
    ambiguous_indicators: list[dict] | None = None,
    parameters: list[dict] | None = None,
    loading_params: list[dict] | None = None,  # noqa: ARG001
) -> tuple[Tool, dict]:
    """Create a validation tool for model spec, bound to a causal spec.

    When skeleton parts (resolved_likelihoods, parameters, etc.) are provided,
    validates ModelSpecDecisions and merges with the skeleton. Otherwise falls
    back to validating a full ModelSpec dict (backward-compatible).

    Args:
        causal_spec: The full CausalSpec dict (to extract indicators for dtype checking)
        resolved_likelihoods: Pre-computed deterministic likelihoods
        ambiguous_indicators: Indicators needing LLM decisions
        parameters: Pre-computed parameters
        loading_params: Loading parameters needing constraint decisions

    Returns:
        Tuple of (tool, capture_dict). After generate_loop, check
        capture["spec"] for the last validated ModelSpec (or None).
    """
    from causal_ssm_agent.utils.causal_spec import get_indicators

    indicators = get_indicators(causal_spec)
    use_decisions_mode = resolved_likelihoods is not None and parameters is not None
    capture: dict = {}

    @tool
    def validate_model_spec_tool():
        """Tool for validating model specification JSON (Stage 4)."""

        async def execute(model_spec_json: str) -> str:
            """
            Validate a model specification and return all validation errors.

            Args:
                model_spec_json: The JSON string containing the model spec to validate.

            Returns:
                "VALID" if the spec passes validation, otherwise a list of all errors found.
            """
            if use_decisions_mode:
                from causal_ssm_agent.orchestrator.schemas_model import (
                    validate_model_spec_decisions_dict,
                )

                return _validate_json_and_format(
                    model_spec_json,
                    lambda data: validate_model_spec_decisions_dict(
                        data,
                        resolved_likelihoods=resolved_likelihoods,  # type: ignore[arg-type]
                        ambiguous_indicators=ambiguous_indicators or [],
                        parameters=parameters,  # type: ignore[arg-type]
                    ),
                    capture=capture,
                    capture_key="spec",
                    capture_result=True,
                )
            else:
                from causal_ssm_agent.orchestrator.schemas_model import validate_model_spec_dict

                return _validate_json_and_format(
                    model_spec_json,
                    lambda data: validate_model_spec_dict(data, indicators=indicators or None),
                    capture=capture,
                    capture_key="spec",
                    capture_result=True,
                )

        return execute

    return validate_model_spec_tool(), capture


def make_worker_tools(schema: dict) -> tuple[list[Tool], dict]:
    """Create the standard toolset for worker agents.

    This is the single source of truth for worker tools.
    Used by both production workers and evals.

    Args:
        schema: The model schema dict to validate extractions against

    Returns:
        Tuple of (tools_list, capture_dict). After generate_loop, check
        capture["output"] for the last validated worker output dict (or None).
    """
    tool, capture = make_validate_worker_output_tool(schema)
    return [tool, parse_date(), calculate()], capture


def make_validate_worker_output_tool(schema: dict) -> tuple[Tool, dict]:
    """Create a validation tool for worker output, bound to a specific schema.

    Args:
        schema: The model schema dict to validate extractions against

    Returns:
        Tuple of (tool, capture_dict). After generate_loop, check
        capture["output"] for the last validated worker output dict (or None).
    """
    capture: dict = {}

    @tool
    def validate_extractions():
        """Tool for validating worker extraction output JSON."""

        async def execute(output_json: str) -> str:
            """
            Validate worker extractions and return all validation errors.

            Args:
                output_json: The JSON string containing the worker output to validate.

            Returns:
                "VALID" if the output passes validation, otherwise a list of all errors found.
            """
            from causal_ssm_agent.workers.schemas import validate_worker_output

            return _validate_json_and_format(
                output_json,
                lambda data: validate_worker_output(data, schema),
                capture=capture,
                capture_key="output",
            )

        return execute

    return validate_extractions(), capture


@tool
def calculate():
    """Tool for evaluating simple arithmetic calculations."""

    async def execute(expression: str) -> str:
        """
        Evaluate a simple arithmetic expression.

        Args:
            expression: A Python arithmetic expression (e.g., "2 + 3 * 4", "100 / 5", "(10 + 5) * 2", "10 % 3", "2 ** 8")

        Returns:
            The result of the calculation, or an error message if evaluation fails.
        """
        # Whitelist of allowed characters for safe evaluation
        allowed_chars = set("0123456789+-*/%()._ ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Expression contains invalid characters. Only numbers and +-*/%()._ are allowed."

        try:
            result = eval(expression)
            return str(result)
        except (SyntaxError, ZeroDivisionError, TypeError, NameError) as e:
            return f"Error evaluating expression: {e}"

    return execute


@tool
def parse_date():
    """Tool for parsing dates into a human-readable spelled out format."""

    async def execute(date_string: str) -> str:
        """
        Parse a date or timestamp into spelled out format.

        Args:
            date_string: A date or timestamp string (e.g., "2024-03-15", "2024-03-15T10:30:00")

        Returns:
            Spelled out date (e.g., "Friday, March 15, 2024") or an error message if parsing fails.
        """
        from datetime import datetime

        # Common formats to try
        formats = [
            "%Y-%m-%d",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%m-%d-%Y",
            "%m/%d/%Y",
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(date_string.strip(), fmt)
                return dt.strftime("%A, %B %d, %Y")  # e.g., "Friday, March 15, 2024"
            except ValueError:
                continue

        return f"Could not parse date: {date_string}"

    return execute


# ---------------------------------------------------------------------------
# Multi-turn generation
# ---------------------------------------------------------------------------


async def multi_turn_generate(
    messages: list["ChatMessage"],
    model: Model,
    follow_ups: list[str] | None = None,
    tools: list[Tool] | None = None,
    follow_up_tools: list[Tool] | None = None,
    config: GenerateConfig | None = None,
    trace_capture: dict | None = None,
) -> str:
    """
    Run a multi-turn conversation with optional tool use.

    Args:
        messages: Initial messages (typically system + user prompt)
        model: The model to use for generation
        follow_ups: List of follow-up user prompts to send after each response (default: none)
        tools: Optional list of tools the model can use on the first turn (enables generate_loop)
        follow_up_tools: Optional list of tools for follow-up (self-review) turns.
            Defaults to None, meaning follow-up turns use no tools (plain generation).
            This prevents the LLM from re-invoking validation tools during self-review
            and potentially overwriting a previously captured valid model.
        config: Optional generation config
        trace_capture: Optional dict; when provided, the full LLMTrace is stored
            under ``trace_capture["trace"]`` before returning.

    Returns:
        The final completion string
    """
    messages = list(messages)  # Don't mutate original
    follow_ups = follow_ups or []

    if tools:
        # Use generate_loop for tool-enabled generation
        final_messages, output = await model.generate_loop(
            messages,
            tools=tools,
            config=config,
        )
        messages = list(final_messages)
        last_nonempty = output.completion

        # Follow-up turns: use follow_up_tools if provided, otherwise no tools
        for prompt in follow_ups:
            messages.append(ChatMessageUser(content=prompt))
            if follow_up_tools:
                final_messages, output = await model.generate_loop(
                    messages,
                    tools=follow_up_tools,
                    config=config,
                )
                messages = list(final_messages)
            else:
                response = await model.generate(messages, config=config)
                messages.append(ChatMessageAssistant(content=response.completion))
                output = response
            if output.completion and output.completion.strip():
                last_nonempty = output.completion

        if trace_capture is not None:
            trace_capture["trace"] = _build_trace(messages, output)

        return last_nonempty
    else:
        # Simple generation without tools
        response = await model.generate(messages, config=config)
        messages.append(ChatMessageAssistant(content=response.completion))
        last_nonempty = response.completion

        for prompt in follow_ups:
            messages.append(ChatMessageUser(content=prompt))
            response = await model.generate(messages, config=config)
            messages.append(ChatMessageAssistant(content=response.completion))
            if response.completion and response.completion.strip():
                last_nonempty = response.completion

        if trace_capture is not None:
            trace_capture["trace"] = _build_trace(messages, response)

        return last_nonempty


# ---------------------------------------------------------------------------
# Trace capture helper
# ---------------------------------------------------------------------------


def attach_trace(output: dict, trace_capture: dict) -> None:
    """Attach LLM trace to output dict if available.

    Replaces the repeated boilerplate:
        trace = trace_capture.get("trace")
        if trace is not None:
            out["llm_trace"] = trace.to_dict()
    """
    trace = trace_capture.get("trace")
    if trace is not None:
        output["llm_trace"] = trace.to_dict()
