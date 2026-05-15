from __future__ import annotations

import json
import logging
import re
from collections import Counter
from textwrap import dedent

from ..bank.experience import ExperienceBank
from ..config.runtime import ApiModelConfig
from ..core.models import (
    DIMENSIONS,
    BenchmarkCase,
    ControllerDecision,
    HarnessConfig,
    RetrievalRequest,
    make_minimal_config,
)
from ..llm.retry import call_with_retries

logger = logging.getLogger(__name__)


def _get_validation_error(code: str) -> str | None:
    """Return a human-readable description of why *code* fails validation, or None if valid.

    Shared by LLMController and ClaudeCodeController.
    """
    required_markers = (
        "class HarnessImpl",
        "async def setup",
        "async def run",
    )
    for marker in required_markers:
        if marker not in code:
            return f"Missing required marker: `{marker}`"

    try:
        compile(code, "<memoharness_generated_harness>", "exec")
    except SyntaxError as exc:
        return f"SyntaxError at line {exc.lineno}: {exc.msg}"

    if ".acreate(" in code:
        return "Uses removed async API `.acreate(` — use `.create(` with `await` instead."

    return None


_HARNESS_INTERFACE_DOC = """
== Interface contract ==
Your class MUST be named HarnessImpl and implement exactly:

    async def setup(self, environment) -> None
    async def run(self, instruction: str, environment, context) -> None

Utilities (import from memoharness.harbor.agent):
    build_openai_client(api_config=None)  -> AsyncOpenAI
    build_command_tool_spec()             -> list[dict]  # native shell tool schema
    build_response_input_message(...)     -> dict
    build_function_call_output_item(...)  -> dict
    extract_message_text(message)         -> str
    extract_tool_calls(message)           -> list[dict]
    build_assistant_tool_message(...)     -> dict
    build_tool_result_message(...)        -> dict
    is_tool_calling_unsupported_error(exc)-> bool
    resolve_tool_protocol(config=None)    -> str         # "native" | "bash_tags"
    should_use_responses_api(api_config)  -> bool
    extract_bash_blocks(text)             -> list[str]   # legacy <bash> fallback
    strip_bash_blocks(text)               -> str
    load_runtime_config()                 -> MemoHarnessRuntimeConfig | None
    populate_context(context, output, num_calls, total_tokens)
    Agent loops should not rely on a fixed turn ceiling.

Tasks run inside a live Linux terminal. Verifiers check files, process state, or
shell command output - not prose. Inspect the detected workspace root first.
Harbor's official verifier is a post-agent phase, so do not rely on /tests
during bootstrap or repair turns.
"""

_DIMENSIONS_DOC = """
== Six optimization dimensions ==
D1 (Context Assembly)  – system prompt quality, few-shot examples, instruction framing
D2 (Tool Access)       – shell command strategy, inspection approach, tool selection/protocol
D3 (Generation)        – temperature, max_completion_tokens, sampling strategy
D4 (Orchestration)     – loop structure, turn limit, retry logic, multi-stage plans
D5 (Memory)            – history sliding-window size, context compression
D6 (Post-processing)   – stopping criteria, output validation, fallback behavior
"""

_DIMENSION_ACTION_PLAYBOOK_DOC = """
== Dimension-to-action playbook ==
- D1: improve verifier-target extraction and instruction scaffolding.
- D2: harden bootstrap/tool setup and offline-safe command paths.
- D3: tune generation controls only when outputs are unstable/truncated.
- D4: tighten orchestration, retries, and stagnation break logic.
- D5: adjust history window/compression when context-loss is evidenced.
- D6: harden pass/fail validation and false-positive guards.
Prefer 1-3 targeted interventions with explicit expected verifier evidence.
"""

_PYTHON_OUTPUT_DOC = """
Respond with the complete HarnessImpl inside <python> tags, plus a concise D1-D6
summary inside <config> tags:

<python>
import asyncio
from memoharness.harbor.agent import (
    build_assistant_tool_message, build_command_tool_spec, build_function_call_output_item,
    build_response_input_message, build_tool_result_message,
    build_openai_client, extract_bash_blocks, extract_message_text, extract_tool_calls,
    is_tool_calling_unsupported_error, resolve_tool_protocol, should_use_responses_api,
    strip_bash_blocks, load_runtime_config,
    populate_context,
)

class HarnessImpl:
    async def setup(self, environment) -> None:
        ...

    async def run(self, instruction: str, environment, context) -> None:
        ...
</python>
<config>
{"D1": {"strategy": "..."}, "D2": {"strategy": "..."}, "D3": {"temperature": 0.0, "max_completion_tokens": 2048}, "D4": {"strategy": "..."}, "D5": {"history_keep": 12}, "D6": {"strategy": "..."}}
</config>
"""

_CONFIG_ONLY_DOC = """
Respond with exactly one XML-tagged block and nothing else:

<config>
{"D1": {"strategy": "..."}, "D2": {"strategy": "..."}, "D3": {"temperature": 0.0, "max_tokens": 2048}, "D4": {"strategy": "..."}, "D5": {"history_keep": 12}, "D6": {"strategy": "..."}}
</config>

Return only concise JSON updates for dimensions that should change.
"""


class LLMController:
    """Uses an LLM to choose harness configs while code comes from a stable template."""

    def __init__(
        self,
        client,
        model: str = "gpt-4.1-mini",
        api_config: ApiModelConfig | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.api_config = api_config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_initial_harness(self, dataset: str = "") -> tuple[str, HarnessConfig]:
        """Ask the LLM to write an initial HarnessImpl in Python."""
        if not hasattr(self.client, "chat"):
            config = self.stabilize_config(make_minimal_config())
            return self._render_harness_code(config), config

        prompt = (
            "You are writing the initial Python harness for a Linux terminal benchmark.\n"
            f"Dataset: {dataset or 'terminal-bench'}\n\n"
            "Tasks require installing packages, editing files, running services, and fixing code.\n"
            "Verifiers check files, process state, or shell output — not prose.\n\n"
            + _HARNESS_INTERFACE_DOC
            + _DIMENSIONS_DOC
            + "\nWrite a HarnessImpl that:\n"
            "1. Loads the client via load_runtime_config() + build_openai_client().\n"
            "2. Bootstraps with a workspace inspection command (pwd, ls -la ., then best-effort probes for /app and /apps when relevant, but not /tests).\n"
            "3. Sets D2.tool_protocol to either `native` or `bash_tags`; default to `native` and use `run_command` for every shell command in that mode.\n"
            "4. Keeps a sliding-window message history (last 12 turns).\n"
            "5. Continues iterating until verifier state is satisfied.\n"
            "6. Calls populate_context(context, output, num_calls, total_tokens) before returning.\n\n"
            + _PYTHON_OUTPUT_DOC
        )
        try:
            code, config_dict = self._call_llm_for_harness(prompt)
            if self.should_normalize_harness(code):
                raise ValueError("Generated code failed structure validation.")
            config = self._build_harness_config(config_dict)
            config = self.stabilize_config(config)
            logger.info("LLM generated initial HarnessImpl for dataset '%s'.", dataset)
        except Exception as exc:
            logger.warning("LLM initial harness failed (%s) — using stable template.", exc)
            config = self.stabilize_config(make_minimal_config())
            code = self._render_harness_code(config)

        return code, config

    def decide_next_harness(
        self,
        bank: ExperienceBank,
        current_code: str,
        current_config: HarnessConfig,
        iteration: int,
        min_consecutive_failures: int = 3,
    ) -> tuple[str, HarnessConfig]:
        """Ask the LLM to rewrite HarnessImpl based on bank feedback.

        Falls back to *current_code* (not the stable template) so progress is
        never lost when the LLM produces invalid output.
        """
        base_config = self.stabilize_config(current_config)
        if not hasattr(self.client, "chat"):
            return current_code, base_config

        bank_summary = self._build_bank_summary(bank, iteration, min_consecutive_failures)
        signal_summary = self._build_change_outcome_signals(bank, iteration)
        # Trim current_code if it is very large: keep only the first 200 lines so the
        # prompt stays well within the context window and leaves room for the LLM to
        # generate a full new harness (target output budget: 8192 tokens).
        code_lines = current_code.splitlines()
        if len(code_lines) > 200:
            trimmed_code = "\n".join(code_lines[:200]) + "\n# ... (truncated for brevity)"
            logger.debug("Current harness trimmed from %d to 200 lines for prompt.", len(code_lines))
        else:
            trimmed_code = current_code
        prompt = (
            "You are optimizing a Python harness for a Linux terminal benchmark.\n"
            f"Current iteration: {iteration}\n\n"
            + _HARNESS_INTERFACE_DOC
            + _DIMENSIONS_DOC
            + _DIMENSION_ACTION_PLAYBOOK_DOC
            + "\n## Experience Bank Summary\n"
            + bank_summary
            + "\n\n## Change-Outcome Signals\n"
            + signal_summary
            + "\n\n## Current Harness\n"
            "```python\n"
            + trimmed_code
            + "```\n\n"
            "Rewrite the harness to address the failures shown above.\n"
            "You may freely change: system prompt, loop logic, memory strategy, "
            "stopping criteria, retry patterns.\n"
            "Before coding, choose 1-3 highest-priority interventions and link each to "
            "expected verifier evidence.\n"
            "Keep the class named HarnessImpl with setup() and run().\n\n"
            + _PYTHON_OUTPUT_DOC
        )
        try:
            new_code, config_dict = self._call_llm_for_harness(prompt)
            if self.should_normalize_harness(new_code):
                raise ValueError("Generated code failed structure validation.")
            next_config = self._build_harness_config(config_dict, base_config)
            next_config = self.stabilize_config(next_config)
            logger.info("LLM produced updated HarnessImpl at iteration %d.", iteration)
        except Exception as exc:
            logger.warning(
                "LLM harness update failed at iteration %d (%s) — keeping current harness.",
                iteration, exc,
            )
            return current_code, base_config

        return new_code, next_config

    def decide_next_config(
        self,
        bank: ExperienceBank,
        current_config: HarnessConfig,
        iteration: int,
    ) -> ControllerDecision:
        """Compatibility wrapper for the engine/controller interface."""
        _, next_config = self.decide_next_harness(
            bank=bank,
            current_code=self._render_harness_code(current_config),
            current_config=current_config,
            iteration=iteration,
        )
        return ControllerDecision(
            config=next_config,
            rationale="LLM controller proposed the next stable harness configuration.",
            variants=self._build_variants(next_config),
        )

    def adapt_for_case(
        self,
        bank: ExperienceBank,
        case: BenchmarkCase,
        base_config: HarnessConfig,
    ) -> ControllerDecision:
        """Adapt a base harness config to a specific case using similar-history context."""
        config = self.stabilize_config(base_config)
        if not hasattr(self.client, "chat"):
            return ControllerDecision(
                config=config,
                rationale="LLM controller unavailable; using the provided base configuration.",
            )

        case_summary = self._build_case_summary(bank, case)
        test_evidence = self._build_test_time_evidence_summary(bank, case)
        prompt = (
            "You are adapting a MemoHarness configuration for one evaluation case.\n\n"
            "## Target Case\n"
            f"case_id={case.case_id}\n"
            f"prompt={case.prompt}\n"
            f"features={json.dumps(vars(case.features), indent=2)}\n\n"
            "## Base Dimension Summary\n"
            f"{json.dumps(config.as_dict(), indent=2)}\n\n"
            "## Test-Time Evidence\n"
            f"{test_evidence}\n\n"
            "## Similar Cases Summary\n"
            f"{case_summary}\n\n"
            "Only adjust dimensions when the structured retrieved slice or similar-case evidence justifies it.\n\n"
            + _CONFIG_ONLY_DOC
        )
        config_updates = self._call_llm_for_config(prompt)
        next_config = self.stabilize_config(self._build_harness_config(config_updates, config))
        return ControllerDecision(
            config=next_config,
            rationale="LLM controller adapted the harness configuration for this case.",
        )

    def stabilize_config(self, config: HarnessConfig | None = None) -> HarnessConfig:
        """Align summary config with the stable harness template's actual behavior."""
        base = config.clone() if config else make_minimal_config()

        base.D1.setdefault(
            "strategy",
            "inspect verifier requirements before declaring completion",
        )

        base.D2["tool_access"] = "bash"
        base.D2.setdefault("tool_protocol", "native")
        base.D2.setdefault("retrieval_mode", "none")
        base.D2.setdefault("top_k", 0)
        base.D2.setdefault(
            "strategy",
            "inspect the workspace root first, probe /app or /apps only if relevant, and use lightweight local probes between edits",
        )

        base.D3.setdefault("temperature", 0.0)
        base.D3.setdefault("max_tokens", 2048)
        base.D3.setdefault("top_p", 1.0)
        base.D3.setdefault("candidate_count", 1)

        base.D4["workflow"] = "agentic_loop"
        base.D4.setdefault("stop_rule", "task_complete")
        base.D4.setdefault(
            "strategy",
            "bootstrap workspace inspection, then iterate until verifier state is satisfied",
        )

        base.D5.setdefault("memory_policy", "sliding_window")
        base.D5.setdefault("history_keep", 12)
        base.D5.setdefault(
            "strategy",
            "keep recent shell observations while trimming older turns",
        )

        base.D6.setdefault("postprocess", "raw_passthrough")
        base.D6.setdefault("validator", "workspace_checks")
        base.D6.setdefault("fallback", "last_observation")
        base.D6.setdefault(
            "strategy",
            "do not stop before required files, services, or shell checks look correct",
        )
        return base

    def should_normalize_harness(self, code: str) -> bool:
        """Return True when the code is structurally broken and must be replaced.

        Only flags genuinely unusable code:
        - Missing required class/method markers.
        - Python syntax errors.
        - Use of the removed async API (.acreate).
        """
        required_markers = (
            "class HarnessImpl",
            "async def setup",
            "async def run",
        )
        if any(marker not in code for marker in required_markers):
            return True

        try:
            compile(code, "<memoharness_generated_harness>", "exec")
        except SyntaxError:
            return True

        # Only flag the removed async API — everything else is valid Python.
        return ".acreate(" in code

    def normalize_harness_code(self, code: str, config: HarnessConfig) -> str:
        """Replace unstable legacy code with the stable rendered harness template."""
        stable_config = self.stabilize_config(config)
        if not self.should_normalize_harness(code):
            return code
        return self._render_harness_code(stable_config)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_llm_for_config(self, prompt: str) -> dict:
        response = call_with_retries(
            lambda: self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise JSON config editor. "
                            "Always respond with a valid JSON object inside <config> tags."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_completion_tokens=1024,
            ),
            api_config=self.api_config,
            logger=logger,
            operation="LLM controller config API call",
        )
        text = response.choices[0].message.content or ""
        return self._parse_config_response(text)

    def _call_llm_for_harness(self, prompt: str) -> tuple[str, dict]:
        """Backwards-compatible parser retained for tests and older callers."""
        response = call_with_retries(
            lambda: self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise Python code generator. "
                        "Always respond with valid Python inside <python> tags "
                        "and a valid JSON object inside <config> tags."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
                max_completion_tokens=8192,
            ),
            api_config=self.api_config,
            logger=logger,
            operation="LLM controller harness API call",
        )
        text = response.choices[0].message.content or ""
        return self._parse_harness_response(text)

    def _parse_config_response(self, text: str) -> dict:
        config_match = re.search(r"<config>(.*?)</config>", text, re.DOTALL)
        raw = config_match.group(1).strip() if config_match else text.strip()
        if not raw:
            logger.warning("LLM config response was empty — no dimension updates applied.")
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("LLM config response JSON parse failed (%s) — no updates applied.", exc)
            return {}
        if not isinstance(parsed, dict):
            logger.warning("LLM config response was not a dict — no updates applied.")
            return {}
        return parsed

    def _parse_harness_response(self, text: str) -> tuple[str, dict]:
        python_match = re.search(r"<python>(.*?)</python>", text, re.DOTALL)
        config_match = re.search(r"<config>(.*?)</config>", text, re.DOTALL)

        if python_match:
            code = python_match.group(1).strip()
        else:
            code = text.strip()

        config_dict: dict = {}
        if config_match:
            try:
                config_dict = json.loads(config_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        if not code:
            raise ValueError(f"Could not extract Python code from LLM response:\n{text[:500]}")

        return code, config_dict

    def _build_harness_config(
        self, config_dict: dict, fallback: HarnessConfig | None = None
    ) -> HarnessConfig:
        """Merge LLM-provided dimension updates into a fallback config."""
        base = fallback.clone() if fallback else HarnessConfig()
        for dim in DIMENSIONS:
            llm_dim = config_dict.get(dim)
            if isinstance(llm_dim, dict) and llm_dim:
                getattr(base, dim).update(llm_dim)
        return base

    def _render_harness_code(self, config: HarnessConfig) -> str:
        config = self.stabilize_config(config)
        temperature = float(config.D3.get("temperature", 0.0) or 0.0)
        max_tokens = int(config.D3.get("max_tokens", 2048) or 2048)
        history_keep = int(config.D5.get("history_keep", 12) or 12)
        config_json = json.dumps(config.as_dict(), sort_keys=True)
        strategy_notes = [
            f"D1: {config.D1.get('strategy', '')}",
            f"D2: {config.D2.get('strategy', '')}",
            f"D4: {config.D4.get('strategy', '')}",
            f"D6: {config.D6.get('strategy', '')}",
        ]
        strategy_block = "\n".join(f"- {note}" for note in strategy_notes if note.strip())
        bootstrap_command = (
            "echo '=== pwd ===' && pwd && "
            "echo '=== workspace ===' && (ls -la . 2>/dev/null || true) && "
            "echo '=== /app ===' && (ls -la /app 2>/dev/null || true) && "
            "echo '=== /apps ===' && (ls -la /apps 2>/dev/null || true)"
        )
        return dedent(
            f"""
import json
import time
import asyncio
from memoharness.harbor.agent import (
    build_assistant_tool_message,
    build_command_tool_spec,
    build_response_input_from_messages,
    build_function_call_output_item,
    build_response_input_message,
    build_tool_result_message,
    build_openai_client,
    call_openai_model_with_fallback,
    preferred_openai_api_mode,
    extract_bash_blocks,
    extract_message_text,
    extract_tool_calls,
    resolve_tool_protocol,
    strip_bash_blocks,
    load_runtime_config,
    populate_context,
)

_HARNESS_CONFIG = json.loads({config_json!r})
_TEMPERATURE = {temperature}
_MAX_COMPLETION_TOKENS = {max_tokens}
_HISTORY_KEEP = {history_keep}
_COMMAND_TIMEOUT_SECONDS = 120.0
_STRATEGY_NOTES = {json.dumps(strategy_block)}
_BOOTSTRAP_COMMAND = {json.dumps(bootstrap_command)}


def _attach_config(context) -> None:
    metadata = getattr(context, "metadata", None)
    if isinstance(metadata, dict):
        metadata.setdefault("memoharness_config", _HARNESS_CONFIG)


def _record_command_status(
    context,
    *,
    command: str,
    status: str,
    observation: str,
    timeout_sec: float | None = None,
) -> None:
    metadata = getattr(context, "metadata", None)
    if not isinstance(metadata, dict):
        return
    metadata["last_command"] = command
    metadata["last_command_status"] = status
    if timeout_sec is not None:
        metadata["command_timeout_sec"] = timeout_sec
    metadata["last_observation_preview"] = observation[:500]


def _append_trace(trace_items, label: str, text: str) -> None:
    if text is None:
        return
    rendered = str(text)
    if not rendered:
        return
    trace_items.append(f"[{{label}}] {{rendered}}")


def _publish_context(
    context,
    output: str,
    num_llm_calls: int,
    total_tokens: int,
    started_at: float,
    tools_invoked,
    intermediate_outputs,
    *,
    final_output: str | None = None,
) -> None:
    populate_context(
        context,
        output,
        num_llm_calls,
        total_tokens,
        latency_ms=int((time.time() - started_at) * 1000),
        tools_invoked=list(tools_invoked),
        intermediate_outputs=list(intermediate_outputs),
        final_output=final_output or output,
    )


class HarnessImpl:
    def __init__(self):
        self._client = None
        self._model = None
        self._api_config = None
        self._tool_protocol = resolve_tool_protocol(_HARNESS_CONFIG)
        self._native_tool_calling = self._tool_protocol == "native"
        self._openai_api_mode = "chat_completions"

    async def setup(self, environment) -> None:
        runtime = load_runtime_config()
        if runtime:
            self._api_config = runtime.llm
            self._client = build_openai_client(runtime.llm)
            self._model = runtime.llm.model
        else:
            import openai
            import os

            self._api_config = None
            self._client = openai.AsyncOpenAI()
            self._model = os.environ.get("MEMOHARNESS_MODEL", "gpt-4.1-mini")
        self._openai_api_mode = preferred_openai_api_mode(
            self._api_config,
            native_tool_calling=self._native_tool_calling,
        )

    def _usage_tokens(self, response) -> int:
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0
        total = getattr(usage, "total_tokens", None)
        if total is not None:
            return int(total)
        prompt = getattr(usage, "prompt_tokens", 0) or 0
        completion = getattr(usage, "completion_tokens", 0) or 0
        return int(prompt + completion)

    def _trim_messages(self, messages):
        if len(messages) <= 2 or _HISTORY_KEEP <= 0:
            return list(messages)
        keep = max(2, _HISTORY_KEEP * 2)
        return [messages[0]] + list(messages[-keep:])

    async def _call_model(
        self,
        messages,
        *,
        response_input=None,
        previous_response_id=None,
        system_prompt="",
    ):
        model_turn, response, api_mode = await call_openai_model_with_fallback(
            self._client,
            api_mode=self._openai_api_mode,
            model=self._model,
            messages=messages,
            response_input=response_input if self._openai_api_mode == "responses" else None,
            previous_response_id=previous_response_id if self._openai_api_mode == "responses" else None,
            system_prompt=system_prompt,
            temperature=_TEMPERATURE,
            max_completion_tokens=_MAX_COMPLETION_TOKENS,
            native_tool_calling=self._native_tool_calling,
        )
        self._openai_api_mode = api_mode
        return model_turn, self._usage_tokens(response)

    async def _run_commands(self, environment, commands):
        for cmd in commands:
            try:
                result = await asyncio.wait_for(
                    environment.exec(cmd),
                    timeout=_COMMAND_TIMEOUT_SECONDS,
                )
                rendered = result if isinstance(result, str) else str(result)
                status = "completed"
                timeout_sec = None
            except asyncio.TimeoutError:
                rendered = f"[bash timeout after {{_COMMAND_TIMEOUT_SECONDS}}s] {{cmd}}"
                status = "timed_out"
                timeout_sec = _COMMAND_TIMEOUT_SECONDS
            except Exception as exc:
                rendered = f"[bash error] {{exc}}"
                status = "error"
                timeout_sec = None
            yield cmd, status, rendered, timeout_sec

    async def run(self, instruction: str, environment, context) -> None:
        system_prompt = (
            "You are an AI agent solving a task inside a live Linux terminal.\\n"
            + (
                "Use the native `run_command` tool for every shell command you want to run.\\n"
                if self._native_tool_calling
                else "Use <bash>...</bash> tags for every shell command you want to run.\\n"
            )
            "Most tasks are verified by files, services, or shell checks rather than prose.\\n"
            "Inspect the detected workspace root first, then probe /app or /apps opportunistically before concluding the task is complete.\\n"
            "Keep iterating until the required files, processes, or command outputs look correct.\\n"
            + (
                "When you believe the task is done, return a short final answer with no tool calls.\\n"
                if self._native_tool_calling
                else "When you believe the task is done, return a short final answer with no <bash> tags.\\n"
            )
            "Harness strategy notes:\\n"
            f"{{_STRATEGY_NOTES}}"
        )
        messages = [
            {{"role": "system", "content": system_prompt}},
            {{"role": "user", "content": instruction}},
        ]
        use_responses_api = self._openai_api_mode == "responses"
        response_input = (
            [build_response_input_message("user", instruction)]
            if use_responses_api
            else []
        )
        previous_response_id = None

        def _set_next_response_input(user_content, prefix_items=None):
            nonlocal response_input
            response_input = list(prefix_items or [])
            response_input.append(build_response_input_message("user", user_content))

        started_at = time.time()
        num_llm_calls = 0
        total_tokens = 0
        executed_rounds = 0
        last_observation = ""
        last_command_batch = []
        tools_invoked = []
        intermediate_outputs = []

        bootstrap_outputs = []
        async for cmd, status, rendered, timeout_sec in self._run_commands(
            environment,
            [_BOOTSTRAP_COMMAND],
        ):
            del cmd, status, timeout_sec
            bootstrap_outputs.append(f"$ {{_BOOTSTRAP_COMMAND}}\\n{{rendered}}")
        bootstrap_output = "\\n\\n".join(bootstrap_outputs)
        if bootstrap_output:
            last_observation = bootstrap_output.strip()
            _append_trace(intermediate_outputs, "bootstrap", bootstrap_output)
            messages.append(
                {{
                    "role": "user",
                    "content": "Initial workspace inspection:\\n" + bootstrap_output,
                }}
            )
            if use_responses_api:
                response_input.append(
                    build_response_input_message(
                        "user",
                        "Initial workspace inspection:\\n" + bootstrap_output,
                    )
                )

        while True:
            if not use_responses_api:
                messages = self._trim_messages(messages)
            if use_responses_api:
                model_turn, used_tokens = await self._call_model(
                    messages,
                    response_input=response_input,
                    previous_response_id=previous_response_id,
                    system_prompt=system_prompt,
                )
            else:
                model_turn, used_tokens = await self._call_model(messages)
            num_llm_calls += 1
            total_tokens += used_tokens

            text = str(model_turn.get("text", "") or "")
            self._openai_api_mode = str(
                model_turn.get("api_mode") or self._openai_api_mode or "chat_completions"
            )
            use_responses_api = self._openai_api_mode == "responses"
            if use_responses_api:
                previous_response_id = str(
                    model_turn.get("response_id") or previous_response_id or ""
                )
            else:
                previous_response_id = None
            tool_calls = list(model_turn.get("tool_calls") or [])
            assistant_message = model_turn.get("assistant_message") or {{"role": "assistant", "content": text}}
            bash_commands = [cmd for cmd in extract_bash_blocks(text) if cmd.strip()]
            final_message = strip_bash_blocks(text).strip()
            if final_message:
                _append_trace(intermediate_outputs, "assistant", final_message)

            command_entries = []
            if tool_calls:
                for tool_call in tool_calls:
                    arguments = tool_call.get("arguments", {{}})
                    command_entries.append(
                        {{
                            "command": str(arguments.get("command", "") if isinstance(arguments, dict) else "").strip(),
                            "tool_call_id": str(tool_call.get("id") or ""),
                            "tool_name": str(tool_call.get("name") or ""),
                        }}
                    )
            else:
                command_entries = [
                    {{"command": cmd, "tool_call_id": "", "tool_name": "run_command"}}
                    for cmd in bash_commands
                ]

            if not command_entries:
                if executed_rounds == 0:
                    prompt_content = (
                        "Do not conclude yet. "
                        + (
                            "Use the native `run_command` tool to inspect, edit files, or run checks until the verifier state is satisfied."
                            if self._native_tool_calling
                            else "Use <bash> commands to inspect, edit files, or run checks until the verifier state is satisfied."
                        )
                    )
                    if use_responses_api:
                        messages.append({{"role": "assistant", "content": text or "[empty] response"}})
                        messages.append(
                            {{
                                "role": "user",
                                "content": prompt_content,
                            }}
                        )
                        _set_next_response_input(prompt_content)
                    else:
                        messages.append({{"role": "assistant", "content": text or "[empty] response"}})
                        messages.append(
                            {{
                                "role": "user",
                                "content": prompt_content,
                            }}
                        )
                    continue

                output = final_message or last_observation or "Task finished without a final message."
                _attach_config(context)
                _publish_context(
                    context,
                    output,
                    num_llm_calls,
                    total_tokens,
                    started_at,
                    tools_invoked,
                    intermediate_outputs,
                    final_output=output,
                )
                return

            command_batch = [item["command"] or f"[tool:{{item['tool_name']}}]" for item in command_entries]
            if command_batch == last_command_batch:
                prompt_content = (
                    "You repeated the exact same command batch. "
                    "Choose a different diagnostic or fix step."
                )
                if use_responses_api:
                    messages.append(
                        assistant_message if tool_calls else {{"role": "assistant", "content": text}}
                    )
                    messages.append(
                        {{
                            "role": "user",
                            "content": prompt_content,
                        }}
                    )
                    _set_next_response_input(prompt_content)
                else:
                    messages.append(
                        assistant_message if tool_calls else {{"role": "assistant", "content": text}}
                    )
                    messages.append(
                        {{
                            "role": "user",
                            "content": prompt_content,
                        }}
                    )
                continue

            last_command_batch = list(command_batch)
            tools_invoked.extend(command_batch)
            _append_trace(intermediate_outputs, "command_batch", " | ".join(command_batch))
            command_outputs = []
            response_tool_outputs = []
            if tool_calls:
                messages.append(assistant_message)
            for index, item in enumerate(command_entries, start=1):
                cmd = str(item.get("command") or "")
                tool_name = str(item.get("tool_name") or "")
                tool_call_id = str(item.get("tool_call_id") or f"tool_call_{{index}}")
                if tool_calls and tool_name != "run_command":
                    status = "error"
                    rendered = f"[tool error] unknown tool: {{tool_name or '(empty)'}}"
                    timeout_sec = None
                elif not cmd:
                    status = "error"
                    rendered = "[tool error] missing required `command` argument."
                    timeout_sec = None
                else:
                    async for cmd_result, status, rendered, timeout_sec in self._run_commands(
                        environment,
                        [cmd],
                    ):
                        del cmd_result
                        break
                line = f"$ {{cmd or '[missing command]'}}\\n{{rendered}}"
                command_outputs.append(line)
                _append_trace(intermediate_outputs, "observation", rendered)
                _record_command_status(
                    context,
                    command=cmd or f"[tool:{{tool_name}}]",
                    status=status,
                    observation=rendered,
                    timeout_sec=timeout_sec,
                )
                if tool_calls:
                    if use_responses_api:
                        response_tool_outputs.append(
                            build_function_call_output_item(tool_call_id, line)
                        )
                    messages.append(build_tool_result_message(tool_call_id, line))
            command_output = "\\n\\n".join(command_outputs)
            executed_rounds += 1
            if command_output.strip():
                last_observation = command_output.strip()

            if use_responses_api:
                if not tool_calls:
                    messages.append({{"role": "assistant", "content": text}})
                messages.append({{"role": "user", "content": command_output}})
                _set_next_response_input(
                    command_output,
                    prefix_items=response_tool_outputs if tool_calls else None,
                )
            else:
                if not tool_calls:
                    messages.append({{"role": "assistant", "content": text}})
                messages.append({{"role": "user", "content": command_output}})

        output = last_observation or "No output generated within max turns."
        _attach_config(context)
        _publish_context(
            context,
            output + "\\n[Warning] Max turns reached without an explicit completion signal.",
            num_llm_calls,
            total_tokens,
            started_at,
            tools_invoked,
            intermediate_outputs,
            final_output=output + "\\n[Warning] Max turns reached without an explicit completion signal.",
        )
"""
        ).rstrip() + "\n"

    def _build_variants(self, base_config: HarnessConfig) -> list[HarnessConfig]:
        """Produce lightweight exploration variants for training-time search."""
        variants: list[HarnessConfig] = []

        v1 = base_config.clone()
        v1.D3["temperature"] = max(float(v1.D3.get("temperature", 0.0) or 0.0), 0.2)
        v1.D3["top_p"] = max(float(v1.D3.get("top_p", 1.0) or 1.0), 0.9)
        variants.append(self.stabilize_config(v1))

        v2 = base_config.clone()
        v2.D3["max_tokens"] = max(int(v2.D3.get("max_tokens", 2048) or 2048), 3072)
        v2.D3["candidate_count"] = max(int(v2.D3.get("candidate_count", 1) or 1), 2)
        variants.append(self.stabilize_config(v2))

        return variants

    def _build_case_summary(self, bank: ExperienceBank, case: BenchmarkCase) -> str:
        """Summarize nearest successful and failed cases for one target case."""
        try:
            similar = bank.retrieve_similar_cases_for_case(case)
        except Exception:
            return "No similar-case summary available."

        parts: list[str] = []

        if similar.successful:
            parts.append("Successful similar cases:")
            for entry in similar.successful[:5]:
                parts.append(
                    "  {case_id} domain={domain} complexity={complexity:.2f} "
                    "requires_external_knowledge={needs_knowledge} instruction={instruction}".format(
                        case_id=entry.case_id,
                        domain=entry.case_features.domain,
                        complexity=entry.case_features.complexity_estimate,
                        needs_knowledge=entry.case_features.requires_external_knowledge,
                        instruction=entry.case_features.instruction or entry.case_id,
                    )
                )

        if similar.failed:
            parts.append("Failed similar cases:")
            for entry in similar.failed[:3]:
                parts.append(
                    "  {case_id} primary_dim={primary_dim} analysis={analysis}".format(
                        case_id=entry.case_id,
                        primary_dim=entry.diagnosis.diagnostic_signal.primary_dim,
                        analysis=entry.diagnosis.analysis[:120],
                    )
                )

        if not parts:
            parts.append("No similar-case summary available.")

        return "\n".join(parts)

    def _render_retrieval_request(self, request: RetrievalRequest) -> str:
        payload = {
            "feature_filters": [
                {
                    "field": feature_filter.field,
                    "operator": feature_filter.operator,
                    "value": feature_filter.value,
                }
                for feature_filter in request.feature_filters
            ],
            "min_consecutive_failures": request.min_consecutive_failures,
            "reward_trend": request.reward_trend,
            "primary_dim": request.primary_dim,
            "iteration_range": list(request.iteration_range) if request.iteration_range else None,
            "case_ids": request.case_ids,
            "sample_k": request.sample_k,
            "sample_by_cluster_k": request.sample_by_cluster_k,
            "max_entries": request.max_entries,
            "max_global_patterns": request.max_global_patterns,
        }
        compact = {key: value for key, value in payload.items() if value not in (None, [], {})}
        return json.dumps(compact, ensure_ascii=False, indent=2)

    def _render_retrieved_slice(self, retrieved_slice) -> str:
        parts: list[str] = []
        if retrieved_slice.global_patterns:
            parts.append("Global patterns:")
            for pattern in retrieved_slice.global_patterns:
                parts.append(
                    "  [{pattern_id}] primary_dim={primary_dim} effect={effect}".format(
                        pattern_id=pattern.pattern_id,
                        primary_dim=pattern.primary_dim,
                        effect=pattern.effect[:160],
                    )
                )

        if retrieved_slice.entries:
            parts.append("Retrieved per-case entries:")
            for entry in retrieved_slice.entries[:6]:
                stats = retrieved_slice.case_stats.get(entry.case_id)
                parts.append(
                    "  case={case_id} iter={iteration} reward={reward:.2f} domain={domain} "
                    "primary_dim={primary_dim} consecutive_failures={consecutive_failures} "
                    "trend={trend} analysis={analysis}".format(
                        case_id=entry.case_id,
                        iteration=entry.iteration,
                        reward=entry.primary_reward,
                        domain=entry.case_features.domain,
                        primary_dim=entry.diagnosis.diagnostic_signal.primary_dim,
                        consecutive_failures=(
                            stats.consecutive_failures if stats is not None else "n/a"
                        ),
                        trend=stats.reward_trend if stats is not None else "n/a",
                        analysis=entry.diagnosis.analysis[:140],
                    )
                )
        else:
            parts.append("Retrieved per-case entries: none")

        return "\n".join(parts)

    def _build_test_time_evidence_summary(self, bank: ExperienceBank, case: BenchmarkCase) -> str:
        try:
            request, retrieved_slice = bank.retrieve_feature_matched_slice_for_case(case)
        except Exception as exc:
            request = RetrievalRequest(max_entries=8, max_global_patterns=3)
            retrieved_slice = bank.retrieve(request)
            logger.warning(
                "Falling back to unconditioned test-time retrieval for case %s: %s",
                case.case_id,
                exc,
            )

        parts = [
            "Structured query:",
            self._render_retrieval_request(request),
            "",
            "Retrieved slice:",
            self._render_retrieved_slice(retrieved_slice),
        ]

        case_summary = self._build_case_summary(bank, case)
        if case_summary:
            parts.extend(["", "Nearest successful/failed neighborhoods:", case_summary])
        return "\n".join(parts)

    def _build_iteration_retrieval_evidence(
        self,
        bank: ExperienceBank,
        iteration: int,
        *,
        min_consecutive_failures: int = 3,
    ) -> str:
        window_start = max(0, iteration - 4)
        recent_failures = [
            entry
            for entry in bank.entries
            if window_start <= entry.iteration <= iteration and not entry.diagnosis.success
        ]
        primary_dim = None
        case_ids = None
        reward_trend = None
        min_failure_threshold = None

        if recent_failures:
            primary_dim = Counter(
                entry.diagnosis.diagnostic_signal.primary_dim for entry in recent_failures
            ).most_common(1)[0][0]
            case_ids = list(dict.fromkeys(entry.case_id for entry in recent_failures[-6:]))
            recent_stats = [
                bank.case_stats[entry.case_id]
                for entry in recent_failures
                if entry.case_id in bank.case_stats
            ]
            if any(
                stats.consecutive_failures >= min_consecutive_failures
                for stats in recent_stats
            ):
                min_failure_threshold = min_consecutive_failures
            if any(
                stats.reward_trend == "degrading"
                for stats in recent_stats
            ):
                reward_trend = "degrading"

        request = RetrievalRequest(
            primary_dim=primary_dim,
            min_consecutive_failures=min_failure_threshold,
            reward_trend=reward_trend,
            iteration_range=(window_start, iteration),
            case_ids=case_ids,
            sample_by_cluster_k=1,
            max_entries=8,
            max_global_patterns=3,
        )
        retrieved_slice = bank.retrieve(request)
        return "\n".join(
            [
                "Structured query:",
                self._render_retrieval_request(request),
                "",
                "Retrieved slice:",
                self._render_retrieved_slice(retrieved_slice),
            ]
        )

    def _build_bank_summary(
        self,
        bank: ExperienceBank,
        iteration: int,
        min_consecutive_failures: int = 3,
    ) -> str:
        """Build a concise text summary of the experience bank for the controller prompt.

        Three sections:
          1. Overall + recent success/failure rates (accurate counts, no mislabelling).
          2. Persistent failures — cases whose consecutive_failures >= min_consecutive_failures,
             taken directly from case_stats so the threshold matches the distillation trigger.
          3. Global patterns distilled by the LLM distiller.
        """
        summary_slice = bank.retrieve(
            RetrievalRequest(
                max_entries=8,
                max_global_patterns=3,
            )
        )
        total = len(bank.entries)
        if total == 0:
            return "No experience data available yet."

        parts: list[str] = []

        # --- Section 1: overall + recent stats ---
        successes = sum(1 for e in bank.entries if e.diagnosis.success)
        pct = 100 * successes // total
        parts.append(
            f"Overall: {total} entries — {successes} successes, "
            f"{total - successes} failures ({pct}% success rate)"
        )

        recent = [e for e in bank.entries if e.iteration > iteration - 5]
        if recent:
            recent_ok = sum(1 for e in recent if e.diagnosis.success)
            parts.append(
                f"Last 5 iterations: {len(recent)} cases — "
                f"{recent_ok} succeeded, {len(recent) - recent_ok} failed"
            )

        # --- Section 2: persistent failures (consecutive_failures >= threshold) ---
        persistent = sorted(
            (
                (case_id, stats)
                for case_id, stats in bank.case_stats.items()
                if stats.consecutive_failures >= min_consecutive_failures
            ),
            key=lambda x: x[1].consecutive_failures,
            reverse=True,
        )

        if persistent:
            dim_counts: dict[str, int] = {}
            for case_id, stats in persistent:
                for dim, cnt in stats.dim_failure_counts.items():
                    if cnt > 0:
                        dim_counts[dim] = dim_counts.get(dim, 0) + cnt

            parts.append(
                f"Persistently failing cases "
                f"({min_consecutive_failures}+ consecutive failures): {len(persistent)}"
            )
            parts.append(f"Failure dimension distribution: {json.dumps(dim_counts)}")
            for case_id, stats in persistent[:4]:
                last_entry = bank.latest_entry_for_case(case_id)
                if last_entry is None:
                    continue
                parts.append(
                    f"  case={case_id} "
                    f"consec_failures={stats.consecutive_failures} "
                    f"primary_dim={last_entry.diagnosis.diagnostic_signal.primary_dim} "
                    f"analysis={last_entry.diagnosis.analysis[:100]}"
                )
        else:
            parts.append(
                f"No persistent failures yet "
                f"(threshold: {min_consecutive_failures} consecutive)."
            )

        # --- Section 3: global patterns ---
        if summary_slice.global_patterns:
            parts.append("Global patterns:")
            for pattern in summary_slice.global_patterns:
                parts.append(f"  [{pattern.pattern_id}] {pattern.description}")

        return "\n".join(parts)

    def _build_change_outcome_signals(
        self,
        bank: ExperienceBank,
        iteration: int,
        max_entries: int = 8,
    ) -> str:
        entries = [entry for entry in bank.entries if entry.iteration == iteration]
        if not entries:
            return f"No per-case signal records available for iteration {iteration}."

        lines: list[str] = []
        for entry in entries[-max_entries:]:
            change_parts: list[str] = []
            for dim, updates in sorted(entry.delta_from_prev.items()):
                if not isinstance(updates, dict) or not updates:
                    continue
                keys = [key for key in updates.keys() if key != "_removed"]
                if "_removed" in updates:
                    keys.append("_removed")
                if keys:
                    change_parts.append(f"{dim}:{','.join(sorted(keys))}")
            change_text = "; ".join(change_parts) if change_parts else "(no config delta)"
            lines.append(
                "case={case_id} reward={reward:.2f} primary_dim={dim} changes={changes} "
                "calls={calls} tokens={tokens} tools={tools} analysis={analysis}".format(
                    case_id=entry.case_id,
                    reward=entry.primary_reward,
                    dim=entry.diagnosis.diagnostic_signal.primary_dim,
                    changes=change_text,
                    calls=entry.trajectory.num_llm_calls,
                    tokens=entry.trajectory.total_tokens,
                    tools=len(entry.trajectory.tools_invoked),
                    analysis=entry.diagnosis.analysis[:140],
                )
            )
        return "\n".join(lines)
