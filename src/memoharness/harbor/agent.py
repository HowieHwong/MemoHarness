"""Thin Harbor agent wrapper for MemoHarness.

Loads a generated Python harness (HarnessImpl class) from MEMOHARNESS_HARNESS_PATH
and delegates all execution to it.  Each iteration of the training loop the
controller writes a new harness .py file implementing all six MemoHarness
dimensions (D1-D6) as plain Python code, with full freedom over prompts,
control flow, memory management, and output processing.

Environment variables
---------------------
MEMOHARNESS_HARNESS_PATH  path to the generated harness .py file
MEMOHARNESS_CONFIG        path to MemoHarnessRuntimeConfig JSON (API keys / model)
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional
from urllib.parse import urlparse

if TYPE_CHECKING:
    from harbor.agents.base import BaseAgent as _BaseAgent
    from harbor.environments.base import BaseEnvironment
    from harbor.models.agent.context import AgentContext

logger = logging.getLogger(__name__)

_BASH_RE = re.compile(r"<bash>(.*?)</bash>", re.DOTALL | re.IGNORECASE)
_COMMAND_TOOL_NAME = "run_command"
_TOOL_PROTOCOL_ENV = "MEMOHARNESS_TOOL_PROTOCOL"
_VALID_TOOL_PROTOCOLS = frozenset({"native", "bash_tags"})
_VALID_OPENAI_API_MODES = frozenset({"responses", "chat_completions"})
_MAX_TURNS = sys.maxsize  # legacy compatibility for older generated harnesses


# ---------------------------------------------------------------------------
# Utilities importable by generated harnesses
# ---------------------------------------------------------------------------

def build_openai_client(api_config=None):
    """Build an AsyncOpenAI client from a runtime config block or environment."""
    import openai
    if api_config is None:
        return openai.AsyncOpenAI()
    key_value = api_config.api_key_env
    if key_value.startswith(("sk-", "sk_")):
        api_key = key_value
    else:
        api_key = os.environ.get(key_value)
    kwargs: dict = {"api_key": api_key}
    if getattr(api_config, "api_base", None):
        kwargs["base_url"] = api_config.api_base
    return openai.AsyncOpenAI(**kwargs)


def extract_bash_blocks(text: str) -> list[str]:
    """Return all <bash>...</bash> command strings from model output."""
    return [b.strip() for b in _BASH_RE.findall(text) if b.strip()]


def strip_bash_blocks(text: str) -> str:
    """Remove <bash>...</bash> tags, keeping surrounding prose."""
    return _BASH_RE.sub("", text).strip()


def build_command_tool_spec() -> list[dict[str, object]]:
    """Return the native tool schema for executing one shell command."""
    return [
        {
            "type": "function",
            "function": {
                "name": _COMMAND_TOOL_NAME,
                "description": (
                    "Execute a shell command inside the live Linux terminal environment. "
                    "Use it to inspect files, edit files, install dependencies, start "
                    "services, and run checks."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "A shell command to execute in the environment.",
                        }
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def should_use_responses_api(api_config: object | None = None) -> bool:
    """Return True when the runtime points at the official OpenAI Responses API."""
    api_base = str(getattr(api_config, "api_base", "") or "").strip()
    if not api_base:
        return True
    try:
        host = (urlparse(api_base).hostname or "").strip().lower()
    except Exception:
        host = ""
    if not host:
        return False
    return host == "api.openai.com" or host.endswith(".openai.com")


def normalize_openai_api_mode(
    value: object | None,
    *,
    default: str = "chat_completions",
) -> str:
    """Normalize the configured OpenAI API surface."""
    candidate = str(value or "").strip().lower()
    if not candidate:
        candidate = default
    if candidate not in _VALID_OPENAI_API_MODES:
        allowed = ", ".join(sorted(_VALID_OPENAI_API_MODES))
        raise ValueError(
            f"Invalid OpenAI API mode {candidate!r}; expected one of: {allowed}"
        )
    return candidate


def preferred_openai_api_mode(
    api_config: object | None = None,
    *,
    native_tool_calling: bool = True,
) -> str:
    """Choose the initial OpenAI API mode for the current runtime."""
    if native_tool_calling and should_use_responses_api(api_config):
        return "responses"
    return "chat_completions"


def alternate_openai_api_mode(api_mode: object | None) -> str:
    """Return the other OpenAI API mode."""
    normalized = normalize_openai_api_mode(api_mode)
    if normalized == "responses":
        return "chat_completions"
    return "responses"


def normalize_tool_protocol(value: object | None, *, default: str = "native") -> str:
    """Normalize the configured shell-tool protocol."""
    candidate = str(value or "").strip().lower()
    if not candidate:
        candidate = default
    if candidate not in _VALID_TOOL_PROTOCOLS:
        allowed = ", ".join(sorted(_VALID_TOOL_PROTOCOLS))
        raise ValueError(
            f"Invalid tool protocol {candidate!r}; expected one of: {allowed}"
        )
    return candidate


def resolve_tool_protocol(config: object | None = None, *, default: str = "native") -> str:
    """Resolve the active shell-tool protocol from env override or harness config."""
    env_value = os.environ.get(_TOOL_PROTOCOL_ENV)
    if env_value is not None and str(env_value).strip():
        return normalize_tool_protocol(env_value, default=default)
    if isinstance(config, dict):
        d2 = config.get("D2", {})
        if isinstance(d2, dict):
            return normalize_tool_protocol(d2.get("tool_protocol"), default=default)
    return normalize_tool_protocol(None, default=default)


def _message_attr(message: object, name: str, default: Any = None) -> Any:
    if isinstance(message, dict):
        return message.get(name, default)
    return getattr(message, name, default)


def extract_message_text(message: object) -> str:
    """Best-effort text extraction from chat-completions or responses payloads."""
    output_text = _message_attr(message, "output_text")
    if isinstance(output_text, str) and output_text:
        return output_text
    output_items = _message_attr(message, "output")
    if isinstance(output_items, list):
        parts = [
            extract_message_text(item)
            for item in output_items
            if str(_message_attr(item, "type", "") or "").lower() == "message"
        ]
        return "\n".join(part for part in parts if part).strip()
    content = _message_attr(message, "content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                if item:
                    parts.append(item)
                continue
            item_type = str(_message_attr(item, "type", "") or "").lower()
            text_value = _message_attr(item, "text")
            if isinstance(text_value, dict):
                text_value = text_value.get("value") or text_value.get("text")
            if isinstance(text_value, str) and text_value:
                parts.append(text_value)
                continue
            if item_type in {"text", "output_text", "input_text"}:
                value = _message_attr(item, "value")
                if isinstance(value, str) and value:
                    parts.append(value)
        return "\n".join(part for part in parts if part).strip()
    return str(content or "")


def _normalize_tool_arguments(name: str, raw_arguments: object) -> tuple[dict[str, object], str]:
    if isinstance(raw_arguments, dict):
        arguments = json.loads(json.dumps(raw_arguments))
        arguments_json = json.dumps(arguments, ensure_ascii=True, sort_keys=True)
        return arguments, arguments_json
    arguments_json = str(raw_arguments or "{}")
    try:
        parsed = json.loads(arguments_json)
    except json.JSONDecodeError:
        if name == _COMMAND_TOOL_NAME and arguments_json.strip():
            parsed = {"command": arguments_json.strip()}
        else:
            parsed = {"raw_arguments": arguments_json.strip()}
    arguments = parsed if isinstance(parsed, dict) else {"value": parsed}
    return arguments, arguments_json


def extract_tool_calls(message: object) -> list[dict[str, object]]:
    """Normalize tool calls from chat-completions or responses payloads."""
    raw_tool_calls = _message_attr(message, "tool_calls", []) or []
    normalized: list[dict[str, object]] = []
    if raw_tool_calls:
        for index, tool_call in enumerate(raw_tool_calls, start=1):
            function = _message_attr(tool_call, "function", {}) or {}
            name = str(_message_attr(function, "name", "") or "").strip()
            call_id = str(_message_attr(tool_call, "id", "") or f"tool_call_{index}")
            arguments, arguments_json = _normalize_tool_arguments(
                name,
                _message_attr(function, "arguments", "{}"),
            )
            normalized.append(
                {
                    "id": call_id,
                    "name": name,
                    "arguments": arguments,
                    "arguments_json": arguments_json,
                }
            )
        return normalized

    output_items = _message_attr(message, "output", [])
    if isinstance(message, list):
        output_items = message
    for index, item in enumerate(output_items or [], start=1):
        if str(_message_attr(item, "type", "") or "").lower() != "function_call":
            continue
        name = str(_message_attr(item, "name", "") or "").strip()
        call_id = str(
            _message_attr(item, "call_id", "")
            or _message_attr(item, "id", "")
            or f"tool_call_{index}"
        )
        arguments, arguments_json = _normalize_tool_arguments(
            name,
            _message_attr(item, "arguments", "{}"),
        )
        normalized.append(
            {
                "id": call_id,
                "name": name,
                "arguments": arguments,
                "arguments_json": arguments_json,
            }
        )
    return normalized


def build_assistant_tool_message(
    message: object | None = None,
    *,
    text: str | None = None,
    tool_calls: Optional[list[dict[str, object]]] = None,
) -> dict[str, object]:
    """Serialize an assistant message with tool calls for the next API turn."""
    if message is not None:
        if text is None:
            text = extract_message_text(message)
        if tool_calls is None:
            tool_calls = extract_tool_calls(message)
    payload: dict[str, object] = {"role": "assistant", "content": text or ""}
    if tool_calls:
        payload["tool_calls"] = [
            {
                "id": str(call.get("id") or f"tool_call_{index}"),
                "type": "function",
                "function": {
                    "name": str(call.get("name") or ""),
                    "arguments": str(call.get("arguments_json") or "{}"),
                },
            }
            for index, call in enumerate(tool_calls, start=1)
        ]
    return payload


def build_tool_result_message(tool_call_id: str, content: object) -> dict[str, str]:
    """Serialize a tool result message for the next API turn."""
    return {
        "role": "tool",
        "tool_call_id": str(tool_call_id),
        "content": str(content or ""),
    }


def build_response_input_message(role: str, content: object) -> dict[str, object]:
    """Serialize one Responses API message item."""
    return {
        "type": "message",
        "role": str(role or "user"),
        "content": str(content or ""),
    }


def build_function_call_item(
    tool_call_id: str,
    name: str,
    arguments: object,
) -> dict[str, str]:
    """Serialize one Responses API function_call item."""
    _, arguments_json = _normalize_tool_arguments(name, arguments)
    return {
        "type": "function_call",
        "call_id": str(tool_call_id),
        "name": str(name or ""),
        "arguments": arguments_json,
    }


def build_function_call_output_item(tool_call_id: str, content: object) -> dict[str, str]:
    """Serialize one Responses API function_call_output item."""
    return {
        "type": "function_call_output",
        "call_id": str(tool_call_id),
        "output": str(content or ""),
    }


def build_response_input_from_messages(
    messages: list[object] | tuple[object, ...],
) -> list[dict[str, object]]:
    """Convert chat-completions style messages into Responses API input items."""
    items: list[dict[str, object]] = []
    for message in messages or []:
        role = str(_message_attr(message, "role", "user") or "user").strip().lower()
        if role == "system":
            continue
        if role == "tool":
            items.append(
                build_function_call_output_item(
                    str(_message_attr(message, "tool_call_id", "") or ""),
                    _message_attr(message, "content", ""),
                )
            )
            continue
        if role == "assistant":
            text = extract_message_text(message)
            if text:
                items.append(build_response_input_message("assistant", text))
            for tool_call in extract_tool_calls(message):
                items.append(
                    build_function_call_item(
                        str(tool_call.get("id") or ""),
                        str(tool_call.get("name") or ""),
                        tool_call.get("arguments_json") or tool_call.get("arguments") or "{}",
                    )
                )
            continue
        items.append(
            build_response_input_message(
                role,
                _message_attr(message, "content", ""),
            )
        )
    return items


async def call_openai_model_with_fallback(
    client: object,
    *,
    api_mode: str,
    model: str,
    messages: list[dict[str, object]],
    response_input: Optional[list[dict[str, object]]] = None,
    previous_response_id: Optional[str] = None,
    system_prompt: str = "",
    temperature: float = 0.0,
    max_completion_tokens: int = 2048,
    native_tool_calling: bool = False,
    call_responses: Optional[Callable[..., Awaitable[object]]] = None,
    call_chat_completions: Optional[Callable[..., Awaitable[object]]] = None,
    on_fallback: Optional[Callable[[str, str, Exception], None]] = None,
) -> tuple[dict[str, object], object, str]:
    """Call the preferred OpenAI API surface and fall back to the alternate once."""

    primary_mode = normalize_openai_api_mode(api_mode)
    attempt_modes = [primary_mode, alternate_openai_api_mode(primary_mode)]
    errors: list[tuple[str, Exception]] = []

    async def _invoke_responses(
        *,
        input_items: list[dict[str, object]],
        response_id: Optional[str],
    ) -> tuple[dict[str, object], object]:
        request: dict[str, object] = {
            "model": model,
            "instructions": system_prompt,
            "input": input_items,
            "temperature": temperature,
            "max_output_tokens": max_completion_tokens,
        }
        if native_tool_calling:
            request["tools"] = build_command_tool_spec()
            request["tool_choice"] = "auto"
        if response_id:
            request["previous_response_id"] = response_id

        caller = call_responses
        if caller is None:
            caller = getattr(getattr(client, "responses", None), "create", None)
        if caller is None:
            raise AttributeError("client.responses.create is not available")

        response = await caller(**request)
        text = extract_message_text(response)
        tool_calls = extract_tool_calls(response) if native_tool_calling else []
        return {
            "text": text,
            "tool_calls": tool_calls,
            "assistant_message": build_assistant_tool_message(
                text=text,
                tool_calls=tool_calls,
            ),
            "response_id": str(getattr(response, "id", "") or ""),
            "api_mode": "responses",
        }, response

    async def _invoke_chat() -> tuple[dict[str, object], object]:
        request: dict[str, object] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
        }
        if native_tool_calling:
            request["tools"] = build_command_tool_spec()
            request["tool_choice"] = "auto"

        caller = call_chat_completions
        if caller is None:
            caller = getattr(getattr(getattr(client, "chat", None), "completions", None), "create", None)
        if caller is None:
            raise AttributeError("client.chat.completions.create is not available")

        response = await caller(**request)
        message = response.choices[0].message
        text = extract_message_text(message)
        tool_calls = extract_tool_calls(message) if native_tool_calling else []
        return {
            "text": text,
            "tool_calls": tool_calls,
            "assistant_message": build_assistant_tool_message(
                message,
                text=text,
                tool_calls=tool_calls,
            ),
            "response_id": "",
            "api_mode": "chat_completions",
        }, response

    for index, attempt_mode in enumerate(attempt_modes):
        try:
            if attempt_mode == "responses":
                input_items = (
                    response_input or []
                    if index == 0 and primary_mode == "responses"
                    else build_response_input_from_messages(messages)
                )
                response_id = (
                    previous_response_id
                    if index == 0 and primary_mode == "responses"
                    else None
                )
                model_turn, response = await _invoke_responses(
                    input_items=input_items,
                    response_id=response_id,
                )
            else:
                model_turn, response = await _invoke_chat()
            return model_turn, response, attempt_mode
        except Exception as exc:
            errors.append((attempt_mode, exc))
            if index == 0 and on_fallback is not None:
                try:
                    on_fallback(attempt_mode, attempt_modes[1], exc)
                except Exception:
                    logger.debug("OpenAI fallback callback failed.", exc_info=True)
            if index == len(attempt_modes) - 1:
                break

    final_exc = errors[-1][1]
    if native_tool_calling and any(
        is_tool_calling_unsupported_error(exc) for _, exc in errors
    ):
        details = "; ".join(
            f"{mode}: {type(exc).__name__}: {exc}"
            for mode, exc in errors
        )
        raise RuntimeError(
            "Native tool calling is required, but both OpenAI API modes failed: "
            f"{details}"
        ) from final_exc
    raise final_exc


def is_tool_calling_unsupported_error(exc: Exception) -> bool:
    """Return True when the provider rejects native tool-calling parameters."""
    lowered = str(exc or "").lower()
    if not lowered:
        return False
    signals = (
        "unknown parameter",
        "unsupported parameter",
        "tool_calls",
        "tools",
        "function calling",
        "functions",
        "extra inputs are not permitted",
        "additional properties are not allowed",
        "does not support tools",
        "not support tool",
    )
    return any(signal in lowered for signal in signals)


def load_runtime_config():
    """Load MemoHarnessRuntimeConfig from MEMOHARNESS_CONFIG env var, or None."""
    path = os.environ.get("MEMOHARNESS_CONFIG")
    if path and Path(path).exists():
        from memoharness.config.runtime import MemoHarnessRuntimeConfig
        return MemoHarnessRuntimeConfig.from_json_file(Path(path))
    return None


def _normalize_trace_list(values: Optional[list[str]]) -> list[str]:
    normalized: list[str] = []
    for value in values or []:
        if value is None:
            continue
        text = str(value)
        if not text:
            continue
        normalized.append(text)
    return normalized


def _normalize_status_events(values: Optional[list[object]]) -> list[object]:
    normalized: list[object] = []
    for value in values or []:
        if value is None:
            continue
        if isinstance(value, dict):
            normalized.append(json.loads(json.dumps(value)))
            continue
        text = str(value)
        if not text:
            continue
        normalized.append(text)
    return normalized


def _normalize_status_value(value: object | None) -> object | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return json.loads(json.dumps(value))
    text = str(value)
    return text or None


def _merge_context_metadata(context, meta: dict[str, object]) -> None:
    for attr in ("metadata", "trajectory", "trace"):
        try:
            existing = getattr(context, attr)
        except AttributeError:
            existing = None
        if isinstance(existing, dict):
            existing.update(meta)
            return
        if existing is None:
            try:
                setattr(context, attr, dict(meta))
                return
            except AttributeError:
                continue
    try:
        setattr(context, "metadata", dict(meta))
    except AttributeError:
        pass


def _build_harness_selection(
    *,
    mode: str,
    harness_path: str | None,
    impl,
    fallback_reason: str = "",
) -> dict[str, object]:
    return {
        "memoharness_harness_mode": mode,
        "memoharness_harness_path": str(harness_path or ""),
        "memoharness_harness_impl_class": type(impl).__name__,
        "memoharness_harness_fallback_reason": str(fallback_reason or ""),
        "w0_fallback": mode != "live",
    }


def _emit_harness_selection(selection: dict[str, object], *, stage: str) -> None:
    event = {
        "event": "harness_selection",
        "stage": stage,
        "mode": selection["memoharness_harness_mode"],
        "harness_path": selection["memoharness_harness_path"],
        "impl_class": selection["memoharness_harness_impl_class"],
        "w0_fallback": bool(selection["w0_fallback"]),
    }
    fallback_reason = str(selection.get("memoharness_harness_fallback_reason") or "").strip()
    if fallback_reason:
        event["fallback_reason"] = fallback_reason
    payload = json.dumps(event, ensure_ascii=True, sort_keys=True)
    print(f"[memoharness-agent] {payload}", file=sys.stderr, flush=True)
    logger.info("Harness selection: %s", payload)


def populate_context(
    context,
    output: str,
    num_calls: int,
    total_tokens: int,
    *,
    latency_ms: Optional[int] = None,
    tools_invoked: Optional[list[str]] = None,
    intermediate_outputs: Optional[list[str]] = None,
    status_events: Optional[list[object]] = None,
    current_status: object | None = None,
    final_output: Optional[str] = None,
) -> None:
    """Write result and metadata into Harbor's AgentContext."""
    resolved_output = final_output if final_output is not None else output
    for attr in ("result", "output", "answer", "final_output"):
        if hasattr(context, attr):
            try:
                setattr(context, attr, resolved_output)
            except AttributeError:
                pass

    meta = {"num_llm_calls": num_calls, "total_tokens": total_tokens}
    if latency_ms is not None:
        meta["latency_ms"] = int(latency_ms)
    if tools_invoked is not None:
        meta["tools_invoked"] = _normalize_trace_list(tools_invoked)
    if intermediate_outputs is not None:
        normalized_intermediates = _normalize_trace_list(intermediate_outputs)
        meta["intermediate_outputs"] = normalized_intermediates
        # Legacy alias kept for older Harbor parsing code.
        meta["intermediates"] = list(normalized_intermediates)
    if status_events is not None:
        meta["status_events"] = _normalize_status_events(status_events)
    normalized_current_status = _normalize_status_value(current_status)
    if normalized_current_status is not None:
        meta["current_status"] = normalized_current_status
    if resolved_output:
        meta["final_output"] = resolved_output
    for attr in ("metadata", "trajectory", "trace"):
        if hasattr(context, attr):
            try:
                existing = getattr(context, attr)
                if isinstance(existing, dict):
                    existing.update(meta)
                else:
                    setattr(context, attr, meta)
                break
            except AttributeError:
                pass


# ---------------------------------------------------------------------------
# Harbor base-class resolution
# ---------------------------------------------------------------------------

def _get_base_agent_class():
    try:
        from harbor.agents.base import BaseAgent
        return BaseAgent
    except ImportError:
        class _StubBase:
            pass
        return _StubBase


_BaseAgentClass = _get_base_agent_class()


# ---------------------------------------------------------------------------
# Harness loader
# ---------------------------------------------------------------------------

def _load_harness_impl(path: str) -> object:
    """Dynamically load HarnessImpl from a generated Python harness file."""
    spec = importlib.util.spec_from_file_location("memoharness_generated_harness", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load harness spec from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, "HarnessImpl", None)
    if cls is None:
        raise AttributeError(f"No 'HarnessImpl' class found in {path}")
    return cls()


# ---------------------------------------------------------------------------
# MemoHarnessAgent — stable Harbor entry point, never modified between runs
# ---------------------------------------------------------------------------

class MemoHarnessAgent(_BaseAgentClass):
    """Harbor agent that delegates entirely to a generated Python harness.

    This class contains no task-solving logic.  It is a stable entry point
    for Harbor that loads the generated harness at setup time and forwards
    every call to it.
    """

    @staticmethod
    def name() -> str:
        return "memoharness"

    def version(self) -> Optional[str]:
        try:
            from importlib.metadata import version
            return version("memoharness")
        except Exception:
            return None

    async def setup(self, environment: "BaseEnvironment") -> None:
        harness_path = os.environ.get("MEMOHARNESS_HARNESS_PATH")
        fallback_reason = ""
        if harness_path and Path(harness_path).exists():
            logger.info("Loading generated harness from %s", harness_path)
            try:
                self._impl = _load_harness_impl(harness_path)
                mode = "live"
            except Exception as exc:
                fallback_reason = "load error: {0}: {1}".format(type(exc).__name__, exc)
                logger.warning(
                    "Failed to load generated harness (%s) — falling back to W0 baseline.", exc
                )
                self._impl = _MinimalHarnessImpl()
                self._impl._is_fallback = True
                mode = "minimal_fallback"
        else:
            fallback_reason = (
                "MEMOHARNESS_HARNESS_PATH not set or file not found: "
                f"{harness_path or '(unset)'}"
            )
            logger.warning(
                "MEMOHARNESS_HARNESS_PATH not set or file not found — using W0 baseline."
            )
            self._impl = _MinimalHarnessImpl()
            self._impl._is_fallback = True
            mode = "minimal_fallback"
        self._harness_selection = _build_harness_selection(
            mode=mode,
            harness_path=harness_path,
            impl=self._impl,
            fallback_reason=fallback_reason,
        )
        _emit_harness_selection(self._harness_selection, stage="setup")
        await self._impl.setup(environment)

    async def run(
        self,
        instruction: str,
        environment: "BaseEnvironment",
        context: "AgentContext",
    ) -> None:
        selection = getattr(
            self,
            "_harness_selection",
            _build_harness_selection(
                mode="minimal_fallback",
                harness_path=os.environ.get("MEMOHARNESS_HARNESS_PATH"),
                impl=getattr(self, "_impl", _MinimalHarnessImpl()),
                fallback_reason="run called before harness setup metadata was initialized",
            ),
        )
        _emit_harness_selection(selection, stage="run")
        _merge_context_metadata(context, selection)
        await self._impl.run(instruction, environment, context)


# ---------------------------------------------------------------------------
# W0 baseline — minimal fallback used before any generated harness exists
# ---------------------------------------------------------------------------

class _MinimalHarnessImpl:
    """Minimal W0 harness: plain agentic loop with no optimisations.

    Serves as the starting point that the controller tries to improve.
    All six dimensions are at their minimum:
      D1 - plain system prompt, no examples, no structured instruction
      D2 - bash tools via <bash> tags, no retrieval
      D3 - temperature=0.0, max_tokens=2048
      D4 - single agentic loop with no fixed turn ceiling
      D5 - full history kept, no compression
      D6 - raw output passthrough, no validation
    """

    async def setup(self, environment: "BaseEnvironment") -> None:
        runtime = load_runtime_config()
        if runtime:
            self._api_config = runtime.llm
            self._client = build_openai_client(runtime.llm)
            self._model: str = runtime.llm.model
        else:
            self._api_config = None
            self._client = build_openai_client()
            self._model = os.environ.get("MEMOHARNESS_MODEL", "gpt-4.1-mini")
        self._tool_protocol = resolve_tool_protocol()
        self._native_tool_calling: bool = self._tool_protocol == "native"
        self._openai_api_mode = preferred_openai_api_mode(
            self._api_config,
            native_tool_calling=self._native_tool_calling,
        )

    async def run(
        self,
        instruction: str,
        environment: "BaseEnvironment",
        context: "AgentContext",
    ) -> None:
        if self._native_tool_calling:
            system = (
                "You are an AI agent solving a task inside a live Linux terminal.\n"
                "Use the provided `run_command` tool for every shell command you need to execute.\n"
                "The command output will be returned as a tool result in the next message.\n"
                "Only respond without tool calls when the task is fully complete."
            )
        else:
            system = (
                "You are an AI agent solving a task inside a live Linux terminal.\n"
                "To run a shell command, wrap it in <bash> tags:\n"
                "  <bash>ls -la</bash>\n"
                "The command output will be returned in the next message.\n"
                "Only respond without <bash> tags when the task is fully complete."
            )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": instruction},
        ]
        use_responses_api = self._openai_api_mode == "responses"
        response_input = (
            [build_response_input_message("user", instruction)]
            if use_responses_api
            else []
        )
        previous_response_id: str | None = None
        started_at = time.monotonic()
        output = ""
        total_tokens = 0
        num_calls = 0
        executed_rounds = 0  # rounds where bash commands were actually executed
        tools_invoked: list[str] = []
        intermediate_outputs: list[str] = []

        def _append_intermediate(label: str, text: str) -> None:
            if text is None:
                return
            rendered = str(text)
            if not rendered:
                return
            intermediate_outputs.append(f"[{label}] {rendered}")

        while True:
            model_turn, resp, api_mode = await call_openai_model_with_fallback(
                self._client,
                api_mode=self._openai_api_mode,
                model=self._model,
                messages=messages,
                response_input=response_input if use_responses_api else None,
                previous_response_id=previous_response_id if use_responses_api else None,
                system_prompt=system,
                temperature=0.0,
                max_completion_tokens=2048,
                native_tool_calling=self._native_tool_calling,
                on_fallback=lambda from_mode, to_mode, exc: logger.warning(
                    "OpenAI API %s failed; falling back to %s: %s",
                    from_mode,
                    to_mode,
                    exc,
                ),
            )
            self._openai_api_mode = api_mode
            use_responses_api = api_mode == "responses"
            text = str(model_turn.get("text", "") or "")
            tool_calls = list(model_turn.get("tool_calls") or [])
            assistant_message = model_turn.get("assistant_message") or {
                "role": "assistant",
                "content": text,
            }
            if use_responses_api:
                previous_response_id = str(
                    model_turn.get("response_id") or previous_response_id or ""
                )
            else:
                previous_response_id = None
            usage = getattr(resp, "usage", None)
            if usage:
                total_tokens += int(
                    getattr(usage, "total_tokens", None)
                    or (
                        (getattr(usage, "prompt_tokens", 0) or 0)
                        + (getattr(usage, "completion_tokens", 0) or 0)
                    )
                )
            num_calls += 1

            if tool_calls:
                command_outputs: list[str] = []
                messages.append(assistant_message)
                response_input = []
                for tool_call in tool_calls:
                    args = tool_call.get("arguments", {})
                    cmd = str(args.get("command", "") if isinstance(args, dict) else "").strip()
                    if tool_call.get("name") != _COMMAND_TOOL_NAME:
                        out = f"[tool error] unknown tool: {tool_call.get('name')}"
                    elif not cmd:
                        out = "[tool error] missing required `command` argument."
                    else:
                        tools_invoked.append(cmd)
                        try:
                            result = await environment.exec(cmd)
                            out = result if isinstance(result, str) else str(result)
                        except Exception as exc:
                            out = f"[bash error] {exc}"
                        command_outputs.append(f"$ {cmd}\n{out}")
                        _append_intermediate("command", cmd)
                        _append_intermediate("observation", out)
                    if use_responses_api:
                        response_input.append(
                            build_function_call_output_item(str(tool_call.get("id") or ""), out)
                        )
                    messages.append(
                        build_tool_result_message(str(tool_call.get("id") or ""), out)
                    )
                if command_outputs:
                    executed_rounds += 1
                    output = "\n\n".join(command_outputs)
                continue

            bash_blocks = extract_bash_blocks(text)
            if not bash_blocks:
                # Nudge: if the agent hasn't executed any bash yet, push it to act
                if executed_rounds == 0:
                    _append_intermediate("assistant", text or "[empty response]")
                    retry_prompt = (
                        "Do not conclude yet. "
                        + (
                            "Use the native `run_command` tool to inspect, edit files, or run checks until the verifier state is satisfied."
                            if self._native_tool_calling
                            else "Use <bash> commands to inspect, edit files, or run checks until the verifier state is satisfied."
                        )
                    )
                    if use_responses_api:
                        messages.append({"role": "assistant", "content": text or "[empty response]"})
                        messages.append(
                            {
                                "role": "user",
                                "content": retry_prompt,
                            }
                        )
                        response_input = [build_response_input_message("user", retry_prompt)]
                    else:
                        messages.append({"role": "assistant", "content": text or "[empty response]"})
                        messages.append(
                            {
                                "role": "user",
                                "content": retry_prompt,
                            }
                        )
                    continue
                output = strip_bash_blocks(text) or text
                _append_intermediate("final", output)
                break

            bash_outputs: list[str] = []
            for cmd in bash_blocks:
                tools_invoked.append(cmd)
                try:
                    result = await environment.exec(cmd)
                    out = result if isinstance(result, str) else str(result)
                except Exception as exc:
                    out = f"[bash error] {exc}"
                bash_outputs.append(f"$ {cmd}\n{out}")
                _append_intermediate("command", cmd)
                _append_intermediate("observation", out)

            executed_rounds += 1
            if use_responses_api:
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": "\n".join(bash_outputs)})
                response_input = [build_response_input_message("user", "\n".join(bash_outputs))]
            else:
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": "\n".join(bash_outputs)})
            output = strip_bash_blocks(text)

        # Propagate W0 fallback flag if set
        if getattr(self, "_is_fallback", False):
            meta = getattr(context, "metadata", None)
            if isinstance(meta, dict):
                meta["w0_fallback"] = True

        populate_context(
            context,
            output,
            num_calls,
            total_tokens,
            latency_ms=int((time.monotonic() - started_at) * 1000),
            tools_invoked=tools_invoked,
            intermediate_outputs=intermediate_outputs,
            final_output=output,
        )
        logger.info("W0 baseline finished: calls=%d tokens=%d", num_calls, total_tokens)
