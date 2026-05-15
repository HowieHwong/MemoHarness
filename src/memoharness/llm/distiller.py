"""LLM-based global pattern distillation for the Experience Bank.

Replaces the heuristic distill_global_patterns() with an LLM call that reads
the per-case failure histories and generates actionable GlobalPattern entries.

The distiller is intentionally decoupled from ExperienceBank so it can be
swapped or tested independently.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from ..config.runtime import ApiModelConfig
from ..core.models import DimensionName, GlobalPattern
from .retry import call_with_retries

if TYPE_CHECKING:
    from ..bank.experience import ExperienceBank

logger = logging.getLogger(__name__)

_VALID_DIMS: frozenset[str] = frozenset({"D1", "D2", "D3", "D4", "D5", "D6"})
_VALID_API_MODES: frozenset[str] = frozenset({"responses", "chat_completions"})

_SYSTEM_PROMPT = (
    "You are an expert analyzer of LLM harness optimization experiments. "
    "You identify systemic failure patterns across benchmark cases and suggest "
    "actionable configuration changes. Output only valid JSON."
)


class LLMDistiller:
    """Calls an LLM to extract global patterns from the Experience Bank.

    Args:
        client: An OpenAI-compatible client (sync).
        model: Model to use for distillation calls.
        min_consecutive_failures: Minimum consecutive failures for a case to be
            included in the distillation input.
        max_patterns: Maximum number of GlobalPattern objects to produce per
            distillation call.
        max_cases_in_prompt: Token-budget cap on how many failing cases to
            include in the prompt (most-failing cases are prioritised).
    """

    def __init__(
        self,
        client,
        model: str = "gpt-4.1-mini",
        min_consecutive_failures: int = 3,
        max_patterns: int = 3,
        max_cases_in_prompt: int = 10,
        api_config: ApiModelConfig | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.min_consecutive_failures = min_consecutive_failures
        self.max_patterns = max_patterns
        self.max_cases_in_prompt = max_cases_in_prompt
        self.api_config = api_config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def distill(
        self,
        bank: "ExperienceBank",
        current_iteration: int,
    ) -> list[GlobalPattern]:
        """Analyse the bank and return freshly distilled GlobalPattern objects.

        The patterns are NOT added to ``bank.global_patterns`` here — that is
        the caller's responsibility (see ExperienceBank.distill_global_patterns_llm).

        Returns an empty list when there is nothing worth distilling.
        """
        failing_cases = self._collect_failing_cases(bank)
        if not failing_cases:
            logger.info("LLMDistiller: no persistently failing cases — skipping distillation.")
            return []

        prompt = self._build_prompt(failing_cases, bank, current_iteration)
        raw = self._call_llm(prompt)
        patterns = self._parse_patterns(raw, current_iteration)

        logger.info(
            "LLMDistiller: distilled %d global pattern(s) at iteration %d.",
            len(patterns), current_iteration,
        )
        return patterns

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_failing_cases(self, bank: "ExperienceBank") -> list[str]:
        """Return case IDs sorted by consecutive failures (descending)."""
        failing = [
            (case_id, stats.consecutive_failures)
            for case_id, stats in bank.case_stats.items()
            if stats.consecutive_failures >= self.min_consecutive_failures
        ]
        failing.sort(key=lambda x: x[1], reverse=True)
        return [case_id for case_id, _ in failing[: self.max_cases_in_prompt]]

    def _build_prompt(
        self,
        failing_case_ids: list[str],
        bank: "ExperienceBank",
        current_iteration: int,
    ) -> str:
        lines: list[str] = [
            f"The harness optimizer has completed {current_iteration} iteration(s).",
            f"The following {len(failing_case_ids)} case(s) have persistently failed "
            f"({self.min_consecutive_failures}+ consecutive failures):",
            "",
        ]

        for case_id in failing_case_ids:
            stats = bank.case_stats[case_id]
            entries = bank.case_index.get(case_id, [])

            # Last 3 entries for this case
            recent = sorted(entries, key=lambda e: e.iteration)[-3:]

            lines.append(f"### Case: {case_id}")
            lines.append(f"- Domain: {entries[-1].case_features.domain if entries else 'unknown'}")
            lines.append(f"- Consecutive failures: {stats.consecutive_failures}")
            lines.append(f"- Reward trend: {stats.reward_trend}")
            lines.append(
                f"- Dimension failure counts: "
                + ", ".join(
                    f"{dim}={cnt}"
                    for dim, cnt in sorted(
                        stats.dim_failure_counts.items(), key=lambda x: x[1], reverse=True
                    )
                    if cnt > 0
                )
            )

            lines.append("- Recent history:")
            for entry in recent:
                lines.append(
                    f"  iter {entry.iteration}: reward={entry.primary_reward:.2f}, "
                    f"primary_dim={entry.diagnosis.diagnostic_signal.primary_dim}, "
                    f"analysis=\"{entry.diagnosis.analysis[:120]}\""
                )
                lines.append(
                    f"  config snapshot: {json.dumps(_trim_config(entry.config), separators=(',', ':'))}"
                )

            lines.append("")

        lines += [
            "---",
            f"Identify up to {self.max_patterns} global pattern(s) that explain these failures.",
            "Each pattern must:",
            "  1. Span at least 2 cases.",
            "  2. Point to a specific harness dimension (D1–D6).",
            "  3. Include a concrete, actionable recommendation.",
            "",
            "Respond with ONLY a JSON array (no markdown, no extra text):",
            '[',
            '  {',
            '    "description": "<what is failing and why, ≤80 words>",',
            '    "evidence": ["<case_id>", ...],',
            '    "effect": "<quantified impact and recommended config change, ≤60 words>",',
            '    "primary_dim": "<D1|D2|D3|D4|D5|D6>"',
            '  }',
            ']',
        ]
        return "\n".join(lines)

    def _call_llm(self, prompt: str) -> str:
        primary_mode = _preferred_openai_api_mode(self.api_config)
        attempt_modes = [primary_mode, _alternate_openai_api_mode(primary_mode)]
        errors: list[tuple[str, Exception]] = []

        for index, api_mode in enumerate(attempt_modes):
            try:
                if api_mode == "responses":
                    response = call_with_retries(
                        lambda: self.client.responses.create(
                            model=self.model,
                            instructions=_SYSTEM_PROMPT,
                            input=prompt,
                        ),
                        api_config=self.api_config,
                        logger=logger,
                        operation="LLM distiller Responses API call",
                    )
                    return _extract_message_text(response)

                response = call_with_retries(
                    lambda: self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": _SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.0,
                    ),
                    api_config=self.api_config,
                    logger=logger,
                    operation="LLM distiller Chat Completions API call",
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                errors.append((api_mode, exc))
                if index == 0:
                    logger.warning(
                        "LLMDistiller: %s endpoint failed for model %s; retrying with %s: %s",
                        api_mode,
                        self.model,
                        attempt_modes[1],
                        exc,
                    )
                    continue

                first_mode, first_error = errors[0]
                raise RuntimeError(
                    "LLM distiller API call failed across endpoints "
                    f"({first_mode} -> {api_mode}). First error: {first_error}; "
                    f"second error: {exc}"
                ) from exc

        return ""

    def _parse_patterns(
        self,
        text: str,
        current_iteration: int,
    ) -> list[GlobalPattern]:
        # Strip markdown fences if present
        text = text.strip()
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

        try:
            raw_list = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract the first JSON array
            match = re.search(r"\[[\s\S]*\]", text)
            if not match:
                logger.warning("LLMDistiller: could not parse LLM output as JSON array.")
                return []
            try:
                raw_list = json.loads(match.group())
            except json.JSONDecodeError:
                logger.warning("LLMDistiller: JSON extraction also failed.")
                return []

        if not isinstance(raw_list, list):
            logger.warning("LLMDistiller: expected a JSON array, got %s.", type(raw_list).__name__)
            return []

        patterns: list[GlobalPattern] = []
        for idx, item in enumerate(raw_list):
            if not isinstance(item, dict):
                continue
            primary_dim = item.get("primary_dim", "D4")
            if primary_dim not in _VALID_DIMS:
                primary_dim = "D4"
            pattern = GlobalPattern(
                pattern_id=f"llm-pattern-iter{current_iteration}-{idx + 1}",
                iteration_range=(max(0, current_iteration - 4), current_iteration),
                description=str(item.get("description", "")).strip(),
                evidence=list(item.get("evidence", [])),
                effect=str(item.get("effect", "")).strip(),
                primary_dim=primary_dim,
            )
            if pattern.description:
                patterns.append(pattern)

        return patterns[: self.max_patterns]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trim_config(config) -> dict:
    """Return a compact config snapshot omitting default/empty values."""
    result: dict = {}
    for dim in ("D1", "D2", "D3", "D4", "D5", "D6"):
        dim_cfg = dict(getattr(config, dim))
        # Drop keys whose values are obviously empty/default
        trimmed = {
            k: v for k, v in dim_cfg.items()
            if v not in (None, False, 0, "", "none", "disabled", "raw_passthrough")
        }
        if trimmed:
            result[dim] = trimmed
    return result


def _should_use_responses_api(api_config: object | None = None) -> bool:
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


def _normalize_openai_api_mode(value: object | None, *, default: str = "chat_completions") -> str:
    candidate = str(value or "").strip().lower()
    if not candidate:
        candidate = default
    if candidate not in _VALID_API_MODES:
        allowed = ", ".join(sorted(_VALID_API_MODES))
        raise ValueError(
            f"Invalid OpenAI API mode {candidate!r}; expected one of: {allowed}"
        )
    return candidate


def _preferred_openai_api_mode(api_config: object | None = None) -> str:
    if _should_use_responses_api(api_config):
        return "responses"
    return "chat_completions"


def _alternate_openai_api_mode(api_mode: object | None) -> str:
    normalized = _normalize_openai_api_mode(api_mode)
    if normalized == "responses":
        return "chat_completions"
    return "responses"


def _message_attr(message: object, name: str, default=None):
    if isinstance(message, dict):
        return message.get(name, default)
    return getattr(message, name, default)


def _extract_message_text(message: object) -> str:
    output_text = _message_attr(message, "output_text")
    if isinstance(output_text, str) and output_text:
        return output_text

    output_items = _message_attr(message, "output")
    if isinstance(output_items, list):
        parts = [
            _extract_message_text(item)
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
            text_value = _message_attr(item, "text")
            if isinstance(text_value, dict):
                text_value = text_value.get("value") or text_value.get("text")
            if isinstance(text_value, str) and text_value:
                parts.append(text_value)
        return "\n".join(parts).strip()
    return str(content or "")
