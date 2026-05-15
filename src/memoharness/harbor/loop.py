"""Iterative Harbor training loop with ExperienceBank feedback.

Each iteration:
  1. Runs ``harbor run`` with the current HarnessImpl Python file.
  2. Parses ``{jobs_dir}/{run_id}/{job-name}/result.json`` → adds PerCaseEntry records.
  3. Triggers distillation when either condition is met:
       (a) Any case reaches ``min_consecutive_failures`` (default 3) consecutive failures.
       (b) ``last_distill_entry_count >= distill_every`` new entries have been added.
  4. Calls CodexBundleController to update the Harbor Codex bundle and config summary.
  5. Saves ExperienceBank to disk (pickle).

All run directories are namespaced by ``<run_id> = <dataset>__{timestamp>`` so that
repeated experiments never overwrite each other.

CLI usage::

    python -m memoharness.harbor.loop \\
        --config configs/experiment.json
    # bank saved to: artifacts/<run_id>/bank.pkl

The loop passes ``MEMOHARNESS_HARNESS_CONFIG`` (pointing to the .py file) to each
Harbor subprocess so the live HarnessImpl is always up to date.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests

from harbor.models.registry import Registry
from memoharness.config.runtime import DaytonaConfig
from memoharness.harbor.daytona_scheduler import (
    DaytonaRelocationRequest,
    build_daytona_shard_plan,
    choose_relocation_target_key,
)
from memoharness.runtime.codex_bundle import (
    ensure_codex_bundle,
    load_codex_bundle,
    refresh_codex_bundle_support_docs,
    resolve_codex_bundle_paths,
    restore_codex_bundle,
    snapshot_codex_bundle,
)

logger = logging.getLogger(__name__)

_RUN_CONSOLE_LOG_NAME = "console.log"
_HARBOR_COMBINED_LOG_NAME = "harbor.console.log"
_HARBOR_LAUNCHER_META_NAME = "launcher.json"
_HARBOR_STATUS_NAME = "status.json"
_HARBOR_STATUS_PREFIX = "[harness-status] "
_HARBOR_RUNTIME_PATCH_ENV = "MEMOHARNESS_ENABLE_HARBOR_RUNTIME_PATCH"
_HARBOR_AGENT_TIMEOUT_ENV = "MEMOHARNESS_HARBOR_AGENT_TIMEOUT_SEC"
_HARBOR_DISABLE_VERIFIER_RETRY_ENV = "MEMOHARNESS_DISABLE_HARBOR_VERIFIER_RETRY"
_HARBOR_VERIFIER_TIMEOUT_ENV = "MEMOHARNESS_HARBOR_VERIFIER_TIMEOUT_SEC"
_HARBOR_VERIFIER_ENV_OVERRIDES_ENV = "MEMOHARNESS_VERIFIER_ENV_OVERRIDES"
_HARBOR_CODEX_BUNDLE_ENV = "MEMOHARNESS_CODEX_BUNDLE_PATH"
_HARBOR_CODEX_PROMPT_TEMPLATE_ENV = "MEMOHARNESS_CODEX_PROMPT_TEMPLATE"
_HARBOR_CODEX_HOME_ENV = "CODEX_HOME"
_HARBOR_CODEX_HOME_OVERRIDE_ENV = "MEMOHARNESS_HARBOR_CODEX_HOME"
_HARBOR_CODEX_EXPORT_ROOT_ENV = "MEMOHARNESS_HARBOR_CODEX_EXPORT_ROOT"
_HARBOR_CODEX_BASE_URL_ENV = "MEMOHARNESS_CODEX_BASE_URL"
_HARBOR_CODEX_PROVIDER_NAME_ENV = "MEMOHARNESS_CODEX_PROVIDER_NAME"
_HARBOR_CODEX_WIRE_API_ENV = "MEMOHARNESS_CODEX_WIRE_API"
_DEFAULT_HARBOR_CODEX_HOME = "/tmp/codex-home"
_DEFAULT_HARBOR_CODEX_EXPORT_ROOT = "/logs/agent"
_DEFAULT_HARBOR_CODEX_PROVIDER_NAME = "memoharness_custom"
_DEFAULT_HARBOR_CODEX_WIRE_API = "responses"
_BEST_HARNESS_MODE_MEAN_REWARD = "mean_reward"
_BEST_HARNESS_MODE_PERFECT_SUCCESS_COUNT = "perfect_success_count"
_DEFAULT_MEMOHARNESS_AGENT_IMPORT = "memoharness.harbor.agent:MemoHarnessAgent"
_DEFAULT_MEMOHARNESS_CODEX_AGENT_IMPORT = (
    "memoharness.harbor.codex_agent:MemoHarnessCodexAgent"
)


class _TeeTextStream:
    def __init__(self, primary, mirror, *, mirror_path: Path) -> None:
        self._primary = primary
        self._mirror = mirror
        self._mirror_path = Path(mirror_path)
        self._lock = threading.Lock()
        self._live_renderer = None

    @property
    def mirror_path(self) -> Path:
        return self._mirror_path

    def write(self, data):
        renderer = self._live_renderer
        if renderer is not None and data:
            renderer.before_external_write()
        with self._lock:
            written = self._write_locked(data)
        if renderer is not None and data:
            renderer.after_external_write()
        return written

    def write_from_renderer(self, data):
        with self._lock:
            return self._write_locked(data)

    def flush(self) -> None:
        with self._lock:
            self._primary.flush()
            self._mirror.flush()

    def attach_live_renderer(self, renderer) -> None:
        self._live_renderer = renderer

    def detach_live_renderer(self, renderer) -> None:
        if self._live_renderer is renderer:
            self._live_renderer = None

    def isatty(self) -> bool:
        return bool(getattr(self._primary, "isatty", lambda: False)())

    def writable(self) -> bool:
        return True

    def fileno(self):
        return self._primary.fileno()

    def __getattr__(self, name):
        return getattr(self._primary, name)

    def _write_locked(self, data):
        written = self._primary.write(data)
        self._primary.flush()
        self._mirror.write(data)
        self._mirror.flush()
        return written


def _atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Write *content* to *path* atomically via a sibling temp file + os.replace."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fh:
            fh.write(content)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    """Write *content* to *path* atomically via a sibling temp file + os.replace."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(content)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _format_elapsed_seconds(seconds: float) -> str:
    if seconds < 10:
        return f"{seconds:.1f}s"
    return f"{int(round(seconds))}s"


def _summarize_task_names(task_names: list[str], *, max_items: int = 2) -> str:
    if not task_names:
        return "(no tasks)"
    if len(task_names) <= max_items:
        return ", ".join(task_names)
    head = ", ".join(task_names[:max_items])
    return f"{head} +{len(task_names) - max_items} more"


def _mask_secret(value: str | None) -> str | None:
    if not value:
        return None
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def _resolve_api_key_value(
    raw_value: str | None,
    *,
    environ: dict[str, str] | None = None,
) -> str:
    value = str(raw_value or "").strip()
    if not value:
        return ""
    if value.startswith(("sk-", "sk_")):
        return value
    env_map = os.environ if environ is None else environ
    return str(env_map.get(value, "") or "").strip()


def _read_text_tail(path: Path, *, max_bytes: int = 32768) -> str:
    if not path.exists():
        return ""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(size - max_bytes, 0))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _task_name_from_trial_dir(trial_dir: Path) -> str:
    name = str(trial_dir.name)
    if "__" in name:
        return name.split("__", 1)[0]
    return name


_TIMEOUT_TRIAL_LOG_RE = re.compile(r"Trial\s+([^\s]+)\s+failed:", flags=re.IGNORECASE)


def _timeout_exception_type_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    lowered = text.lower()
    if "verifiertimeouterror" in lowered or "verifier execution timed out" in lowered:
        return "VerifierTimeoutError"
    if "agenttimeouterror" in lowered or "agent execution timed out" in lowered:
        return "AgentTimeoutError"
    return None


def _find_timeout_trial(
    job_dir: Path,
    *,
    allowed_exception_types: set[str],
) -> Optional[tuple[str, str]]:
    for exception_path in sorted(job_dir.glob("*/exception.txt")):
        exception_text = _read_text_tail(exception_path, max_bytes=8192)
        exception_type = _timeout_exception_type_from_text(exception_text)
        if exception_type not in allowed_exception_types:
            continue

        result_path = exception_path.with_name("result.json")
        if result_path.exists():
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            exception_info = payload.get("exception_info") or {}
            payload_exception_type = str(exception_info.get("exception_type") or "")
            if payload_exception_type and payload_exception_type not in allowed_exception_types:
                continue
            if payload_exception_type:
                exception_type = payload_exception_type
            task_name = str(payload.get("task_name") or "").strip()
            if task_name:
                return task_name, exception_type

        return _task_name_from_trial_dir(exception_path.parent), exception_type

    for result_path in sorted(job_dir.glob("*/result.json")):
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        exception_info = payload.get("exception_info") or {}
        exception_type = str(exception_info.get("exception_type") or "")
        if exception_type not in allowed_exception_types:
            continue
        task_name = str(payload.get("task_name") or "").strip()
        if task_name:
            return task_name, exception_type
        return _task_name_from_trial_dir(result_path.parent), exception_type
    return None


def _find_timeout_trial_from_log(
    log_path: Path,
    *,
    allowed_exception_types: set[str],
) -> Optional[tuple[str, str]]:
    tail = _read_text_tail(log_path, max_bytes=131072)
    if not tail:
        return None

    for raw_line in reversed(tail.splitlines()):
        line = raw_line.strip()
        exception_type = _timeout_exception_type_from_text(line)
        if exception_type not in allowed_exception_types:
            continue

        match = _TIMEOUT_TRIAL_LOG_RE.search(line)
        if match:
            task_name = _task_name_from_trial_dir(Path(match.group(1)))
            return task_name, exception_type
        return "", exception_type
    return None


def _find_agent_timeout_trial(job_dir: Path) -> Optional[str]:
    found = _find_timeout_trial(
        job_dir,
        allowed_exception_types={"AgentTimeoutError"},
    )
    if found is None:
        return None
    return found[0]


def _trial_log_reports_failure(trial_log_path: Path) -> bool:
    tail = _read_text_tail(trial_log_path, max_bytes=8192)
    if not tail:
        return False

    for raw_line in reversed(tail.splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        if _TIMEOUT_TRIAL_LOG_RE.search(line):
            return True
        if _timeout_exception_type_from_text(line):
            return True
    return False


def _trial_artifact_dirs(job_dir: Path) -> dict[str, Path]:
    trial_dirs: dict[str, Path] = {}
    for pattern in ("*/result.json", "*/exception.txt", "*/trial.log"):
        for path in sorted(job_dir.glob(pattern)):
            if path.parent.is_dir():
                trial_dirs.setdefault(path.parent.name, path.parent)
    return trial_dirs


def _task_name_from_trial_artifacts(trial_dir: Path) -> str:
    result_path = trial_dir / "result.json"
    if result_path.exists():
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if isinstance(payload, dict):
            task_name = _task_name_from_payload(payload, trial_dir=trial_dir)
            if task_name:
                return task_name
    return _task_name_from_trial_dir(trial_dir)


def _exception_type_from_termination_reason(reason: str) -> Optional[str]:
    lowered = (reason or "").lower()
    if "shard wall clock timeout" in lowered:
        return "ShardTimeoutError"
    if "verifier timeout" in lowered:
        return "VerifierTimeoutError"
    if "agent timeout" in lowered:
        return "AgentTimeoutError"
    return None


def _parse_harness_status_line(line: str) -> Optional[dict[str, Any]]:
    if not line.startswith(_HARBOR_STATUS_PREFIX):
        return None
    payload = line[len(_HARBOR_STATUS_PREFIX):].strip()
    if not payload:
        return None
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _latest_harness_status(log_path: Path) -> Optional[dict[str, Any]]:
    tail = _read_text_tail(log_path)
    if not tail:
        return None
    for line in reversed(tail.splitlines()):
        status = _parse_harness_status_line(line.strip())
        if status is not None:
            return status
    return None


def _summarize_harbor_result_payload(payload: dict[str, Any]) -> str:
    stats = _as_dict(payload.get("stats"))
    evals = _as_dict(stats.get("evals"))
    exception_types: list[str] = []
    for raw_eval in evals.values():
        exception_stats = _as_dict(_as_dict(raw_eval).get("exception_stats"))
        for exception_type in exception_stats:
            text = str(exception_type or "").strip()
            if text:
                exception_types.append(text)
    exception_types = sorted(set(exception_types))

    details: list[str] = []
    n_trials = stats.get("n_trials")
    n_total_trials = payload.get("n_total_trials")
    n_errors = stats.get("n_errors")
    if isinstance(n_trials, int) and isinstance(n_total_trials, int):
        details.append(f"trials reported {n_trials}/{n_total_trials}")
    elif isinstance(n_trials, int):
        details.append(f"trials reported {n_trials}")
    elif isinstance(n_total_trials, int):
        details.append(f"total trials {n_total_trials}")
    if isinstance(n_errors, int):
        details.append(f"errors {n_errors}")
    if exception_types:
        details.append(f"exception_types={', '.join(exception_types[:3])}")
    finished_at = str(payload.get("finished_at") or "").strip()
    if not details and finished_at:
        details.append(f"finished_at={finished_at}")
    return "; ".join(details)


def _latest_trial_exception_detail(job_dir: Path) -> str:
    exception_paths = sorted(
        job_dir.glob("*/exception.txt"),
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
        reverse=True,
    )
    for exception_path in exception_paths:
        task_name = _task_name_from_trial_dir(exception_path.parent)
        tail = _read_text_tail(exception_path, max_bytes=8192)
        if not tail:
            continue
        signal_lines = _extract_signal_lines(tail, max_lines=1)
        if signal_lines:
            return f"{task_name}: {signal_lines[0]}"
        for raw_line in reversed(tail.splitlines()):
            line = re.sub(r"\s+", " ", raw_line).strip()
            if line:
                return f"{task_name}: {line[:240]}"
    return ""


def _fallback_harbor_status(job_dir: Path, log_path: Path) -> Optional[dict[str, Any]]:
    top_level_result_path = job_dir / "result.json"
    if top_level_result_path.exists():
        try:
            payload = json.loads(top_level_result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        detail = _summarize_harbor_result_payload(payload)
        status = {"stage": "completed"}
        if detail:
            status["detail"] = detail
        return status

    latest_exception = _latest_trial_exception_detail(job_dir)
    if latest_exception:
        return {"stage": "running harbor", "detail": latest_exception}

    has_trial_artifacts = any(job_dir.glob("*/config.json")) or any(job_dir.glob("*/result.json"))
    if has_trial_artifacts:
        signal_lines = _extract_signal_lines(_read_text_tail(log_path, max_bytes=8192), max_lines=1)
        detail = signal_lines[0] if signal_lines else "trial artifacts detected"
        return {"stage": "running harbor", "detail": detail}

    log_tail = _read_text_tail(log_path, max_bytes=8192)
    if log_tail.strip():
        signal_lines = _extract_signal_lines(log_tail, max_lines=1)
        detail = signal_lines[0] if signal_lines else ""
        status = {"stage": "running harbor"}
        if detail:
            status["detail"] = detail
        return status
    return None


def _normalize_console_stage_label(value: str | None) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "unknown"
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "unknown"


def _inner_console_stage(latest_status: dict[str, Any]) -> str:
    stage = _normalize_console_stage_label(str(latest_status.get("stage") or "starting harness run"))
    detail = str(latest_status.get("detail") or "").strip().lower()
    if stage == "completed":
        if "full suite passed" in detail:
            return "completed_full_suite_passed"
        if "max runtime reached" in detail:
            return "completed_max_runtime"
        if "tests pass after auto-fix" in detail:
            return "completed_after_auto_fix"
        if "tests pass on first run" in detail:
            return "completed_first_run_pass"
    return stage


def _infer_outer_console_stage(
    *,
    latest_status: dict[str, Any],
    log_path: Path,
    process_running: bool,
    termination_reason: str | None,
) -> tuple[str, str]:
    reason = str(termination_reason or "").strip()
    if reason:
        reason_label = _normalize_console_stage_label(reason.split(":", 1)[0])
        if process_running:
            return f"terminating_{reason_label}", ""
        return f"terminated_{reason_label}", ""
    if not process_running:
        return "finished", ""
    if not latest_status:
        return "launching_harbor", ""

    latest_stage = _normalize_console_stage_label(str(latest_status.get("stage") or ""))
    if latest_stage == "completed":
        tail = _read_text_tail(log_path, max_bytes=4096).lower()
        if "running verifier" in tail:
            return "verifier_cleanup", ""
        if "cleanup interrupted" in tail or "environment stop is shielded" in tail:
            return "cleanup", ""
        return "waiting_harbor_exit", ""
    return "running_harbor", ""


@dataclass
class _OuterConsoleTaskState:
    display_name: str
    task_state: str
    wave_index: int
    total_waves: int
    inner_stage: str
    outer_stage: str
    elapsed_seconds: float
    agent_elapsed_seconds: Optional[float]


@dataclass(frozen=True)
class _OuterProgressTarget:
    line_id: str
    display_name: str
    wave_index: int = 1
    total_waves: int = 1


class _OuterConsoleProgressRenderer:
    def __init__(self, stream, *, enabled: bool) -> None:
        self._stream = stream
        self.enabled = bool(enabled and getattr(stream, "isatty", lambda: False)())
        self._lock = threading.RLock()
        self._tasks: dict[str, _OuterConsoleTaskState] = {}
        self._rendered_line_count = 0
        self._display_cleared = False

    def attach_stream(self, stream) -> None:
        if hasattr(stream, "attach_live_renderer"):
            stream.attach_live_renderer(self)

    def before_external_write(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._clear_locked()

    def after_external_write(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._render_locked()

    def update_task(
        self,
        job_name: str,
        display_name: str,
        *,
        inner_stage: str,
        outer_stage: str,
        elapsed_seconds: float,
        agent_elapsed_seconds: Optional[float],
        task_state: str = "running",
        wave_index: int = 1,
        total_waves: int = 1,
    ) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._tasks[job_name] = _OuterConsoleTaskState(
                display_name=display_name,
                task_state=str(task_state or "running"),
                wave_index=max(1, int(wave_index or 1)),
                total_waves=max(1, int(total_waves or 1)),
                inner_stage=inner_stage,
                outer_stage=outer_stage,
                elapsed_seconds=max(0.0, float(elapsed_seconds)),
                agent_elapsed_seconds=(
                    None if agent_elapsed_seconds is None else max(0.0, float(agent_elapsed_seconds))
                ),
            )
            self._render_locked()

    def remove_task(self, job_name: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._tasks.pop(job_name, None)
            self._render_locked()

    def _render_locked(self) -> None:
        lines = self._build_render_lines_locked()
        if not lines and self._rendered_line_count == 0:
            self._display_cleared = False
            return

        max_lines = max(self._rendered_line_count, len(lines))
        parts: list[str] = []
        for index in range(max_lines):
            parts.append("\r\x1b[2K")
            if index < len(lines):
                parts.append(lines[index])
            if index < max_lines - 1:
                parts.append("\n")
        if max_lines > 1:
            parts.append(f"\r\x1b[{max_lines - 1}A")
        else:
            parts.append("\r")
        rendered = "".join(parts)
        if rendered:
            self._write_raw(rendered)
        self._rendered_line_count = len(lines)
        self._display_cleared = len(lines) == 0

    def _build_render_lines_locked(self) -> list[str]:
        snapshots = list(self._tasks.values())
        if not snapshots:
            return []

        max_rows = self._terminal_rows()
        if len(snapshots) <= max_rows:
            return [self._format_line(snapshot) for snapshot in snapshots]

        visible_capacity = max(1, max_rows - 1)
        visible_indices = self._select_visible_snapshot_indices(
            snapshots,
            visible_capacity=visible_capacity,
        )
        visible_index_set = set(visible_indices)
        hidden_snapshots = [
            snapshot
            for index, snapshot in enumerate(snapshots)
            if index not in visible_index_set
        ]
        lines = [self._format_line(snapshots[index]) for index in visible_indices]
        lines.append(self._format_overflow_line(hidden_snapshots))
        return lines

    def _select_visible_snapshot_indices(
        self,
        snapshots: list[_OuterConsoleTaskState],
        *,
        visible_capacity: int,
    ) -> list[int]:
        selected: list[int] = []
        selected_set: set[int] = set()
        state_priority = (
            "running",
            "launching",
            "failed",
            "queued",
            "finished",
        )
        for state in state_priority:
            for index, snapshot in enumerate(snapshots):
                if index in selected_set or snapshot.task_state != state:
                    continue
                selected.append(index)
                selected_set.add(index)
                if len(selected) >= visible_capacity:
                    return selected
        for index in range(len(snapshots)):
            if index in selected_set:
                continue
            selected.append(index)
            if len(selected) >= visible_capacity:
                return selected
        return selected

    def _format_overflow_line(self, hidden_snapshots: list[_OuterConsoleTaskState]) -> str:
        counts: dict[str, int] = {}
        for snapshot in hidden_snapshots:
            counts[snapshot.task_state] = counts.get(snapshot.task_state, 0) + 1
        state_parts = [
            f"{state}={counts[state]}"
            for state in ("running", "launching", "failed", "queued", "finished")
            if counts.get(state)
        ]
        suffix = f" ({', '.join(state_parts)})" if state_parts else ""
        return self._truncate_line(
            f"... {len(hidden_snapshots)} more task(s) hidden to fit terminal{suffix}"
        )

    def _clear_locked(self) -> None:
        if self._rendered_line_count <= 0 or self._display_cleared:
            return
        parts: list[str] = []
        for index in range(self._rendered_line_count):
            parts.append("\r\x1b[2K")
            if index < self._rendered_line_count - 1:
                parts.append("\n")
        if self._rendered_line_count > 1:
            parts.append(f"\r\x1b[{self._rendered_line_count - 1}A")
        else:
            parts.append("\r")
        self._write_raw("".join(parts))
        self._display_cleared = True

    def _write_raw(self, data: str) -> None:
        if not data:
            return
        write_from_renderer = getattr(self._stream, "write_from_renderer", None)
        if callable(write_from_renderer):
            write_from_renderer(data)
            return
        self._stream.write(data)
        self._stream.flush()

    def _format_line(self, snapshot: _OuterConsoleTaskState) -> str:
        elapsed = (
            "-"
            if snapshot.task_state == "queued" and snapshot.elapsed_seconds <= 0
            else _format_elapsed_seconds(snapshot.elapsed_seconds)
        )
        if snapshot.agent_elapsed_seconds is None:
            agent_elapsed = "-"
        else:
            agent_elapsed = _format_elapsed_seconds(snapshot.agent_elapsed_seconds)
        display_name = snapshot.display_name[:36]
        slot = f"{snapshot.wave_index}/{snapshot.total_waves}"
        line = (
            f"{display_name:<36} "
            f"state={snapshot.task_state:<9} "
            f"slot={slot:<5} "
            f"inner={snapshot.inner_stage:<24} "
            f"outer={snapshot.outer_stage:<24} "
            f"elapsed={elapsed:<6} agent={agent_elapsed}"
        )
        return self._truncate_line(line)

    def _terminal_columns(self) -> int:
        fallback = shutil.get_terminal_size(fallback=(120, 40)).columns
        try:
            fileno = getattr(self._stream, "fileno", None)
            if callable(fileno):
                return max(20, os.get_terminal_size(fileno()).columns)
        except (OSError, ValueError):
            pass
        return max(20, int(fallback or 120))

    def _terminal_rows(self) -> int:
        fallback = shutil.get_terminal_size(fallback=(120, 40)).lines
        try:
            fileno = getattr(self._stream, "fileno", None)
            if callable(fileno):
                return max(3, os.get_terminal_size(fileno()).lines)
        except (OSError, ValueError):
            pass
        return max(3, int(fallback or 40))

    def _truncate_line(self, line: str) -> str:
        max_columns = self._terminal_columns()
        if len(line) <= max_columns:
            return line
        if max_columns <= 3:
            return line[:max_columns]
        return line[: max_columns - 3] + "..."


def _normalize_extra_harbor_args(args: list[str]) -> tuple[list[str], list[str]]:
    """Drop Harbor CLI flags that are no longer supported by current versions."""
    normalized: list[str] = []
    dropped: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--agent-override-timeout-sec":
            dropped.append(arg)
            if index + 1 < len(args) and not str(args[index + 1]).startswith("--"):
                index += 2
            else:
                index += 1
            continue
        if str(arg).startswith("--agent-override-timeout-sec="):
            dropped.append("--agent-override-timeout-sec")
            index += 1
            continue
        normalized.append(arg)
        index += 1
    return normalized, dropped


def _harbor_args_specify_model(args: list[str]) -> bool:
    index = 0
    while index < len(args):
        arg = str(args[index] or "")
        if arg in {"--model", "-m"}:
            return True
        if arg.startswith("--model="):
            return True
        index += 1
    return False


_TASK_FILTER_FLAG_PRIORITY = ("--task-name", "--include-task-name", "--task")
_TASK_FILTER_FLAG_FALLBACK = "--include-task-name"


def _probe_harbor_task_filter_flag() -> tuple[str, str]:
    """Detect the Harbor task-filter option supported by the current CLI."""
    try:
        completed = subprocess.run(
            ["harbor", "run", "-h"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:
        return _TASK_FILTER_FLAG_FALLBACK, f"probe failed ({exc}); fallback to {_TASK_FILTER_FLAG_FALLBACK}"

    output = f"{completed.stdout or ''}\n{completed.stderr or ''}".lower()
    options = set(re.findall(r"--[a-z0-9][a-z0-9-]*", output))
    for candidate in _TASK_FILTER_FLAG_PRIORITY:
        if candidate in options:
            return candidate, f"detected from `harbor run -h` (exit={completed.returncode})"
    return _TASK_FILTER_FLAG_FALLBACK, (
        f"flag not found in `harbor run -h` output (exit={completed.returncode}); "
        f"fallback to {_TASK_FILTER_FLAG_FALLBACK}"
    )


_DAYTONA_DISK_LIMIT_MARKER = "total disk limit exceeded"
_DAYTONA_MEMORY_LIMIT_MARKERS = (
    "total memory limit exceeded",
    "memory limit exceeded",
)
_DAYTONA_CONNECTIVITY_MARKERS = (
    "cannot connect to host proxy.app.daytona.io",
    "cannot connect to host app.daytona.io",
    "cannot connect to host api.daytona.io",
    "cannot connect to host",
)
_DAYTONA_RETRY_WAIT_SECONDS = 60
_DAYTONA_CONNECTIVITY_RETRY_LIMIT = 3
_DAYTONA_SHORT_RETRY_WAIT_SECONDS = 5
_TIMEOUT_EXCEPTION_TYPES = {"AgentTimeoutError", "VerifierTimeoutError"}
_RETRYABLE_TIMEOUT_EXCEPTION_TYPES: set[str] = set()
_TIMEOUT_RETRY_LIMIT = 3
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DAYTONA_SANDBOX_CLEANUP_SCRIPT = _REPO_ROOT / "scripts" / "delete_daytona_sandboxes.py"
_FINANCEAGENT_PROMPT_TEMPLATE_NAME = "financeagent_prompt.md"


def _contains_daytona_memory_limit(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(marker in lowered for marker in _DAYTONA_MEMORY_LIMIT_MARKERS)


@dataclass(frozen=True)
class DaytonaShardAssignment:
    job_name: str
    task_names: list[str]
    daytona_key: str


class DaytonaKeyPool:
    def __init__(self, config: DaytonaConfig) -> None:
        self._keys: list[str] = []
        for entry in config.api_keys:
            if entry.startswith("$"):
                resolved = os.environ.get(entry[1:])
                if resolved:
                    self._keys.append(resolved)
                else:
                    logger.warning("Daytona key env var %s is not set; skipping.", entry[1:])
            elif entry:
                self._keys.append(entry)
        self._cooldowns: dict[str, float] = {}
        self._cooldown_seconds = config.key_cooldown_seconds
        self._next_index = 0

    @property
    def size(self) -> int:
        return len(self._keys)

    @property
    def enabled(self) -> bool:
        return self.size > 0

    def lease_keys(self, limit: int) -> list[str]:
        if not self._keys or limit <= 0:
            return []

        now = time.time()
        selected: list[str] = []
        next_index = self._next_index
        for offset in range(len(self._keys)):
            idx = (self._next_index + offset) % len(self._keys)
            key = self._keys[idx]
            if now < self._cooldowns.get(key, 0.0):
                continue
            selected.append(key)
            next_index = (idx + 1) % len(self._keys)
            if len(selected) >= limit:
                break

        if selected:
            self._next_index = next_index
        return selected

    def available_keys(self) -> list[str]:
        if not self._keys:
            return []
        now = time.time()
        return [key for key in self._keys if now >= self._cooldowns.get(key, 0.0)]

    def cooldown_key(self, key: str) -> None:
        if key not in self._keys:
            return
        self._cooldowns[key] = time.time() + self._cooldown_seconds
        logger.info(
            "Placed Daytona key #%d on %ds cooldown.",
            self._keys.index(key) + 1,
            self._cooldown_seconds,
        )

    def any_available(self) -> bool:
        if not self._keys:
            return False
        now = time.time()
        return any(now >= self._cooldowns.get(key, 0.0) for key in self._keys)

    def soonest_available_in(self) -> float:
        if not self._keys:
            return 0.0
        now = time.time()
        return max(
            0.0,
            min(self._cooldowns.get(key, 0.0) for key in self._keys) - now,
        )


def _sanitize_dirname(name: str) -> str:
    """Return a filesystem-safe version of *name*.

    Replaces every run of characters that are illegal in a directory name
    (anything except alphanumerics, dash, underscore, and dot) with a single
    underscore, then strips leading/trailing underscores.

    Example: "terminal-bench-sample@2.0" → "terminal-bench-sample_2.0"
    """
    safe = re.sub(r"[^a-zA-Z0-9_\-.]", "_", name)
    return re.sub(r"_+", "_", safe).strip("_")


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

def _base_case_result(
    reward: float = 0.0,
    total_tokens: int = 0,
    reward_observed: bool = False,
) -> dict[str, Any]:
    success = reward >= 0.5
    return {
        "reward": reward,
        "reward_observed": reward_observed,
        "total_tokens": total_tokens,
        "num_llm_calls": 0,
        "latency_ms": 0,
        "tools_invoked": [],
        "intermediate_outputs": [],
        "final_output": "",
        "analysis": f"Harbor reward={reward:.2f}.",
        "primary_dim": "D1" if success else "D4",
        "external_blocker": False,
    }


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _coerce_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0


def _sum_tokens_from_codex_stream(trial_dir: Path) -> int:
    """Fallback: aggregate usage across every `turn.completed` in agent/codex.txt."""
    stream_path = trial_dir / "agent" / "codex.txt"
    if not stream_path.exists():
        return 0
    total = 0
    try:
        raw = stream_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return 0
    for line in raw.splitlines():
        line = line.strip()
        if not line or line[0] != "{":
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "turn.completed":
            continue
        usage = event.get("usage") or {}
        total += _coerce_int(usage.get("input_tokens")) + _coerce_int(usage.get("output_tokens"))
    return total


def _extract_trial_total_tokens(
    agent_result: dict[str, Any],
    metadata: dict[str, Any],
    trial_dir: Optional[Path] = None,
) -> int:
    """Best-effort token extraction across Harbor result.json schema variants."""
    metadata_total = _coerce_int(metadata.get("total_tokens"))
    if metadata_total > 0:
        return metadata_total

    direct_total = _coerce_int(agent_result.get("total_tokens"))
    if direct_total > 0:
        return direct_total

    usage = _as_dict(agent_result.get("usage"))
    usage_total = _coerce_int(usage.get("total_tokens"))
    if usage_total > 0:
        return usage_total

    usage_prompt = _coerce_int(usage.get("prompt_tokens"))
    usage_completion = _coerce_int(usage.get("completion_tokens"))
    if usage_prompt > 0 or usage_completion > 0:
        return usage_prompt + usage_completion

    n_input_tokens = _coerce_int(agent_result.get("n_input_tokens"))
    n_output_tokens = _coerce_int(agent_result.get("n_output_tokens"))
    if n_input_tokens > 0 or n_output_tokens > 0:
        # Harbor Codex trial results typically expose these fields directly.
        return n_input_tokens + n_output_tokens

    input_tokens = _coerce_int(agent_result.get("input_tokens"))
    output_tokens = _coerce_int(agent_result.get("output_tokens"))
    if input_tokens > 0 or output_tokens > 0:
        return input_tokens + output_tokens

    # Harbor's installed Codex agent sometimes returns None from run(), in which
    # case result.json carries an all-null AgentResult. Fall back to the raw
    # codex exec stream, summing every turn.completed.usage.
    if trial_dir is not None:
        stream_total = _sum_tokens_from_codex_stream(trial_dir)
        if stream_total > 0:
            return stream_total

    return 0


def _read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _extract_signal_lines(text: str, max_lines: int = 3) -> list[str]:
    markers = (
        "assertionerror",
        "filenotfounderror",
        "does not exist",
        "not found in path",
        "command not found",
        "no such file or directory",
        "http 000",
        "returned non-zero exit status",
        "permission denied",
        "failed",
        "timed out",
        "unauthorized",
        "incorrect api key",
        "sandbox not found",
        "cannot connect to host",
        "memory limit exceeded",
        "maximum allowed",
        "total disk limit exceeded",
        "no reward file found",
    )
    lines: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        lower = line.lower()
        if not any(marker in lower for marker in markers):
            continue
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
        if len(lines) >= max_lines:
            break
    return lines


def _classify_trial_failure(summary: str, exception_type: str = "") -> tuple[str, bool]:
    lower = summary.lower()
    exc = exception_type.lower()
    if exc == "daytonaerror" or "total disk limit exceeded" in lower or _contains_daytona_memory_limit(lower):
        return "D4", True
    if exc == "addtestsdirerror" or "failed to add tests directory" in lower:
        return "D4", True
    if exc == "rewardfilenotfounderror" or "no reward file found" in lower:
        return "D6", True
    if (
        "filenotfounderror" in lower
        or "does not exist" in lower
        or ("/app/" in summary and "no such file or directory" in lower)
    ):
        return "D6", False
    if (
        "not found in path" in lower
        or "command not found" in lower
        or "uv: command not found" in lower
        or ("sqlite3" in lower and "no such file or directory" in lower)
    ):
        return "D2", False
    if (
        "http 000" in lower
        or "returned non-zero exit status" in lower
        or "connection refused" in lower
    ):
        return "D4", False
    return "D4", False


def _format_timeout_seconds(value: Any) -> str:
    try:
        return str(float(value))
    except (TypeError, ValueError):
        return str(value)


def _build_timeout_diagnostics(metadata: dict[str, Any]) -> str:
    last_command = str(metadata.get("last_command") or "").strip()
    status = str(metadata.get("last_command_status") or "").strip()
    timeout_sec = metadata.get("command_timeout_sec")
    preview = str(metadata.get("last_observation_preview") or "").strip()

    parts: list[str] = []
    if last_command:
        parts.append(f"last_command={last_command}")
    if status:
        if timeout_sec is not None:
            parts.append(
                "command_status={0} after {1}s".format(
                    status,
                    _format_timeout_seconds(timeout_sec),
                )
            )
        else:
            parts.append(f"command_status={status}")
    elif timeout_sec is not None:
        parts.append(f"command_timeout_sec={_format_timeout_seconds(timeout_sec)}s")
    if preview:
        parts.append(f"observation={preview[:200]}")
    return "; ".join(parts)


def _is_retryable_timeout(detail: dict[str, Any]) -> bool:
    return (
        str(detail.get("exception_type") or "")
        in _RETRYABLE_TIMEOUT_EXCEPTION_TYPES
    )


def _has_observed_reward(detail: Optional[dict[str, Any]]) -> bool:
    return bool(detail and detail.get("reward_observed", False))


def _needs_completion_retry(detail: Optional[dict[str, Any]]) -> bool:
    exception_type = str((detail or {}).get("exception_type") or "")
    return (
        not _has_observed_reward(detail)
        and _retryable_daytona_kind(detail or {}) is None
        and exception_type not in _TIMEOUT_EXCEPTION_TYPES
    )


def _completion_retry_cleared_message(
    prior_detail: Optional[dict[str, Any]],
    retry_count: int,
    retry_limit: int,
) -> str:
    prior_exception_type = str((prior_detail or {}).get("exception_type") or "")
    if prior_exception_type in _TIMEOUT_EXCEPTION_TYPES:
        return "Timeout cleared after retry round {0}/{1}.".format(
            retry_count,
            retry_limit,
        )
    return "Reward observed after retry round {0}/{1}.".format(
        retry_count,
        retry_limit,
    )


def _completion_retry_exhausted_message(
    detail: Optional[dict[str, Any]],
    retry_limit: int,
) -> str:
    exception_type = str((detail or {}).get("exception_type") or "")
    if exception_type in _TIMEOUT_EXCEPTION_TYPES:
        return "Timed out after {0} retry rounds.".format(retry_limit)
    return "No reward observed after {0} retry rounds.".format(retry_limit)


def _append_analysis(detail: dict[str, Any], note: str) -> None:
    note = note.strip()
    if not note:
        return
    existing = str(detail.get("analysis") or "").strip()
    detail["analysis"] = f"{existing} {note}".strip() if existing else note


def _parse_trial_result(
    trial_result_path: Path,
    reward: float,
    *,
    reward_observed: bool = False,
) -> dict[str, Any]:
    detail = _base_case_result(reward=reward, reward_observed=reward_observed)
    trial_dir = trial_result_path.parent
    if not trial_result_path.exists():
        fallback_type, fallback_message = _fallback_trial_exception(trial_dir)
        detail["exception_type"] = fallback_type
        detail["exception_message"] = fallback_message
        summary = re.sub(r"\s+", " ", fallback_message).strip()
        if summary:
            primary_dim, external_blocker = _classify_trial_failure(summary, fallback_type)
            detail["primary_dim"] = primary_dim
            detail["external_blocker"] = external_blocker
            prefix = "External blocker" if external_blocker else "Verifier failure"
            detail["analysis"] = f"{prefix}: {summary}"
        return detail

    data = json.loads(trial_result_path.read_text())
    agent_result = _as_dict(data.get("agent_result"))
    metadata = _as_dict(agent_result.get("metadata"))
    detail["total_tokens"] = _extract_trial_total_tokens(agent_result, metadata, trial_dir)
    detail["num_llm_calls"] = metadata.get("num_llm_calls") or 0
    detail["latency_ms"] = metadata.get("latency_ms") or 0
    detail["tools_invoked"] = list(metadata.get("tools_invoked") or [])
    detail["intermediate_outputs"] = list(
        metadata.get("intermediate_outputs")
        or metadata.get("intermediates")
        or []
    )
    detail["status_events"] = list(metadata.get("status_events") or [])
    detail["current_status"] = metadata.get("current_status") or None
    detail["final_output"] = metadata.get("final_output") or ""

    exception_info = data.get("exception_info") or {}
    exception_type = str(exception_info.get("exception_type") or "")
    exception_message = str(exception_info.get("exception_message") or "")
    if not exception_type or not exception_message:
        fallback_type, fallback_message = _fallback_trial_exception(
            trial_result_path.parent,
            exception_type=exception_type,
            exception_message=exception_message,
        )
        if not exception_type:
            exception_type = fallback_type
        if not exception_message:
            exception_message = fallback_message
    detail["exception_type"] = exception_type
    detail["exception_message"] = exception_message

    verifier_dir = trial_result_path.parent / "verifier"
    verifier_stdout = _read_text_if_exists(verifier_dir / "test-stdout.txt")
    signal_lines = _extract_signal_lines(verifier_stdout)
    summary = " | ".join(signal_lines)
    if not summary and exception_message:
        summary = re.sub(r"\s+", " ", exception_message).strip()
    if not summary:
        intermediate_text = "\n".join(str(item) for item in detail.get("intermediate_outputs") or [])
        summary = " | ".join(_extract_signal_lines(intermediate_text))
    if not summary:
        summary = " | ".join(_extract_signal_lines(str(detail.get("final_output") or "")))
    if exception_type in _TIMEOUT_EXCEPTION_TYPES:
        timeout_diagnostics = _build_timeout_diagnostics(metadata)
        if timeout_diagnostics:
            summary = f"{summary} | {timeout_diagnostics}" if summary else timeout_diagnostics

    if reward >= 0.5 and not summary:
        detail["analysis"] = (
            f"Harbor reward={reward:.2f}; num_llm_calls={detail['num_llm_calls']}; "
            f"tools={len(detail['tools_invoked'])}."
        )
        return detail

    primary_dim, external_blocker = _classify_trial_failure(summary, exception_type)
    detail["primary_dim"] = primary_dim
    detail["external_blocker"] = external_blocker
    if summary:
        prefix = "External blocker" if external_blocker else "Verifier failure"
        detail["analysis"] = f"{prefix}: {summary}"
    return detail


def _task_name_from_payload(payload: dict[str, Any], *, trial_dir: Path) -> str:
    for candidate in (
        str(payload.get("task_name") or "").strip(),
        str(_as_dict(payload.get("task_id")).get("name") or "").strip(),
        str(_as_dict(payload.get("task_id")).get("path") or "").strip(),
    ):
        if candidate:
            return candidate.rsplit("/", 1)[-1]
    return _task_name_from_trial_dir(trial_dir)


def _trial_reward_from_payload(payload: dict[str, Any], *, trial_dir: Path) -> tuple[float, bool]:
    reward_value = _as_dict(_as_dict(payload.get("verifier_result")).get("rewards")).get("reward")
    if reward_value is not None:
        try:
            return float(reward_value), True
        except (TypeError, ValueError):
            pass

    reward_path = trial_dir / "verifier" / "reward.txt"
    if reward_path.exists():
        text = _read_text_if_exists(reward_path).strip()
        if text:
            try:
                return float(text), True
            except ValueError:
                return 0.0, True
        return 0.0, True
    return 0.0, False


def _fallback_trial_exception(
    trial_dir: Path,
    *,
    exception_type: str = "",
    exception_message: str = "",
) -> tuple[str, str]:
    if exception_type and exception_message:
        return exception_type, exception_message

    for candidate_path in (trial_dir / "exception.txt", trial_dir / "trial.log"):
        text = _read_text_tail(candidate_path, max_bytes=131072)
        if not text:
            continue

        inferred_type = exception_type or _timeout_exception_type_from_text(text) or ""
        lowered = text.lower()
        if not inferred_type:
            if _DAYTONA_DISK_LIMIT_MARKER in lowered or _contains_daytona_memory_limit(lowered):
                inferred_type = "DaytonaAuthorizationError"
            elif any(marker in lowered for marker in _DAYTONA_CONNECTIVITY_MARKERS):
                inferred_type = "DaytonaError"
            elif "environment start timed out" in lowered:
                inferred_type = "DaytonaError"

        inferred_message = exception_message
        if not inferred_message:
            for raw_line in reversed(text.splitlines()):
                line = re.sub(r"\s+", " ", raw_line).strip()
                if not line:
                    continue
                inferred_message = line
                break

        if inferred_type or inferred_message:
            return inferred_type, inferred_message

    return exception_type, exception_message


def _find_result_key_for_task(
    results: dict[str, dict[str, Any]],
    task_name: str,
) -> Optional[str]:
    if task_name in results:
        return task_name

    prefix = f"{task_name}__"
    matches = sorted(key for key in results if key.startswith(prefix))
    if not matches:
        return None
    if len(matches) > 1:
        logger.warning(
            "Multiple Harbor trial ids matched task %s: %s. Using %s.",
            task_name,
            matches,
            matches[0],
        )
    return matches[0]


def _normalize_results_for_tasks(
    results: dict[str, dict[str, Any]],
    task_names: list[str],
) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for task_name in task_names:
        result_key = _find_result_key_for_task(results, task_name)
        if result_key is None:
            continue
        normalized[task_name] = results[result_key]
    return normalized


def _filter_learning_results(
    results: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        case_id: detail
        for case_id, detail in results.items()
        if not detail.get("external_blocker", False)
    }


def _is_retryable_daytona_disk_limit(detail: dict[str, Any]) -> bool:
    analysis = str(detail.get("analysis") or "")
    return bool(detail.get("external_blocker")) and _DAYTONA_DISK_LIMIT_MARKER in analysis.lower()


def _is_daytona_memory_limit(detail: dict[str, Any]) -> bool:
    for candidate in (
        str(detail.get("exception_message") or ""),
        str(detail.get("analysis") or ""),
    ):
        if _contains_daytona_memory_limit(candidate):
            return True
    return False


def _is_retryable_daytona_connectivity(detail: dict[str, Any]) -> bool:
    if not detail.get("external_blocker"):
        return False
    analysis = str(detail.get("analysis") or "").lower()
    return any(marker in analysis for marker in _DAYTONA_CONNECTIVITY_MARKERS)
def _retryable_daytona_kind(detail: dict[str, Any]) -> Optional[str]:
    if _is_retryable_daytona_disk_limit(detail):
        return "disk_limit"
    if _is_retryable_daytona_connectivity(detail):
        return "connectivity"
    return None


def _daytona_relocation_error_kind(detail: dict[str, Any]) -> Optional[str]:
    if _is_daytona_memory_limit(detail):
        return "memory_limit"
    return _retryable_daytona_kind(detail)


def _should_retry_daytona_kind(
    kind: Optional[str],
    retry_count: int,
    *,
    disk_limit_retry_limit: int,
    connectivity_retry_limit: int,
) -> bool:
    if kind == "disk_limit":
        return disk_limit_retry_limit < 0 or retry_count < disk_limit_retry_limit
    if kind == "connectivity":
        return connectivity_retry_limit < 0 or retry_count < connectivity_retry_limit
    return False


def _retry_batch_size(n_concurrent: int) -> int:
    if n_concurrent <= 1:
        return 1
    return n_concurrent - 1


def _chunk_case_ids(case_ids: list[str], chunk_size: int) -> list[list[str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [
        case_ids[index:index + chunk_size]
        for index in range(0, len(case_ids), chunk_size)
    ]


def _build_daytona_case_batches(
    case_ids: list[str],
    chunk_size: int,
    exclusive_task_ids: set[str],
) -> list[list[str]]:
    shard_batches = _chunk_case_ids(case_ids, chunk_size)
    if not exclusive_task_ids:
        return shard_batches

    expanded_batches: list[list[str]] = []
    for batch in shard_batches:
        exclusive_batch_ids = [case_id for case_id in batch if case_id in exclusive_task_ids]
        if not exclusive_batch_ids:
            expanded_batches.append(batch)
            continue
        for case_id in exclusive_batch_ids:
            expanded_batches.append([case_id])
        remaining = [case_id for case_id in batch if case_id not in exclusive_task_ids]
        if remaining:
            expanded_batches.append(remaining)
    return expanded_batches


def _partition_case_ids(case_ids: list[str], group_count: int) -> list[list[str]]:
    if group_count <= 0:
        raise ValueError("group_count must be positive")
    if not case_ids:
        return []

    group_count = min(group_count, len(case_ids))
    base_size, remainder = divmod(len(case_ids), group_count)
    partitions: list[list[str]] = []
    start = 0
    for index in range(group_count):
        size = base_size + (1 if index < remainder else 0)
        partitions.append(case_ids[start:start + size])
        start += size
    return [partition for partition in partitions if partition]


def _parse_result_json(result_path: Path) -> dict[str, dict[str, Any]]:
    """Return per-case Harbor results enriched with per-trial diagnostics.

    Token counts are parsed on a best-effort basis from several locations
    Harbor may write them.  Missing counts fall back to 0.
    """
    data = json.loads(result_path.read_text())
    results: dict[str, dict[str, Any]] = {}
    job_dir = result_path.parent

    evals = _as_dict(_as_dict(data.get("stats")).get("evals"))
    for eval_key, eval_data in evals.items():
        eval_data = _as_dict(eval_data)
        reward_stats = _as_dict(_as_dict(eval_data.get("reward_stats")).get("reward"))
        for reward_str, case_ids in reward_stats.items():
            reward = float(reward_str)
            for cid in case_ids or []:
                existing = results.get(cid, _base_case_result())
                existing["reward"] = reward
                existing["reward_observed"] = True
                results[cid] = existing

        for exception_type, case_ids in _as_dict(eval_data.get("exception_stats")).items():
            del exception_type
            for cid in case_ids or []:
                results.setdefault(cid, _base_case_result())

        eval_tokens = (
            _as_dict(eval_data.get("usage")).get("total_tokens")
            or _as_dict(eval_data.get("tokens")).get("total")
            or eval_data.get("total_tokens")
            or 0
        )
        if eval_tokens:
            for cid in list(results):
                if not results[cid]["total_tokens"]:
                    results[cid]["total_tokens"] = int(eval_tokens)

        cases_detail = _as_dict(eval_data.get("cases"))
        for cid, case_info in cases_detail.items():
            if cid in results:
                case_info = _as_dict(case_info)
                total_tokens = (
                    _as_dict(case_info.get("usage")).get("total_tokens")
                    or case_info.get("total_tokens")
                    or results[cid]["total_tokens"]
                )
                results[cid]["total_tokens"] = int(total_tokens)

    top_tokens = (
        _as_dict(data.get("usage")).get("total_tokens")
        or data.get("total_tokens")
        or 0
    )
    if top_tokens and results:
        per_case = int(top_tokens) // len(results)
        for detail in results.values():
            if not detail["total_tokens"]:
                detail["total_tokens"] = per_case

    trial_dir_keys = {path.parent.name: {} for path in job_dir.glob("*/result.json")}

    for trial_result_path in sorted(job_dir.glob("*/result.json")):
        try:
            payload = json.loads(trial_result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue

        task_name = _task_name_from_payload(payload, trial_dir=trial_result_path.parent)
        if not task_name:
            continue

        result_key = _find_result_key_for_task(results, task_name) or task_name
        trial_reward, trial_reward_observed = _trial_reward_from_payload(
            payload,
            trial_dir=trial_result_path.parent,
        )
        existing = results.get(result_key, _base_case_result())
        if trial_reward_observed and not existing.get("reward_observed", False):
            existing["reward"] = trial_reward
            existing["reward_observed"] = True
        results[result_key] = existing

    for cid, detail in list(results.items()):
        result_key = _find_result_key_for_task(trial_dir_keys, cid) or cid
        trial_detail = _parse_trial_result(
            job_dir / result_key / "result.json",
            reward=detail["reward"],
            reward_observed=bool(detail.get("reward_observed", False)),
        )
        for key, value in trial_detail.items():
            if key == "reward":
                continue
            if key == "total_tokens" and detail["total_tokens"]:
                continue
            detail[key] = value

    return results


def _parse_partial_job_results(job_dir: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for trial_dir in sorted(_trial_artifact_dirs(job_dir).values()):
        result_path = trial_dir / "result.json"
        has_result = result_path.exists()
        has_exception = bool(_read_text_tail(trial_dir / "exception.txt", max_bytes=8192).strip())
        has_failed_log = _trial_log_reports_failure(trial_dir / "trial.log")
        if not has_result and not has_exception and not has_failed_log:
            continue

        payload: dict[str, Any] = {}
        if has_result:
            try:
                parsed_payload = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                parsed_payload = {}
            if isinstance(parsed_payload, dict):
                payload = parsed_payload

        task_name = (
            _task_name_from_payload(payload, trial_dir=trial_dir)
            if payload
            else _task_name_from_trial_artifacts(trial_dir)
        )
        if not task_name:
            continue

        reward, reward_observed = _trial_reward_from_payload(payload, trial_dir=trial_dir)
        results[task_name] = _parse_trial_result(
            result_path,
            reward=reward,
            reward_observed=reward_observed,
        )
    return results


def _observed_shard_terminal_tasks(
    job_dir: Path,
    task_names: list[str],
    *,
    extra_exception_tasks: Optional[set[str]] = None,
) -> tuple[set[str], set[str]]:
    partial_results = _parse_partial_job_results(job_dir)
    terminal_tasks: set[str] = set()
    exception_tasks: set[str] = set()

    for task_name in task_names:
        result_key = _find_result_key_for_task(partial_results, task_name)
        if result_key is None:
            continue
        terminal_tasks.add(task_name)
        detail = partial_results.get(result_key) or {}
        if str(detail.get("exception_type") or "").strip() or str(
            detail.get("exception_message") or ""
        ).strip():
            exception_tasks.add(task_name)

    for task_name in extra_exception_tasks or set():
        if task_name in task_names:
            terminal_tasks.add(task_name)
            exception_tasks.add(task_name)

    return terminal_tasks, exception_tasks


# ---------------------------------------------------------------------------
# PerCaseEntry construction
# ---------------------------------------------------------------------------

def _make_entry(
    case_id: str,
    reward: float,
    iteration: int,
    config,
    prev_config,
    primary_dim: str,
    analysis: str,
    total_tokens: int = 0,
    num_llm_calls: int = 0,
    latency_ms: int = 0,
    tools_invoked: Optional[list[str]] = None,
    intermediate_outputs: Optional[list[str]] = None,
    final_output: str = "",
):
    """Build a PerCaseEntry from Harbor result.json plus trial artifacts."""
    from memoharness.core.models import (
        CaseFeatures,
        Diagnosis,
        DiagnosticSignal,
        PerCaseEntry,
        Trajectory,
    )

    success = reward >= 0.5
    # Heuristic: blame D4 (orchestration) for failures — the LLM distiller
    # will refine this diagnosis across cases.

    diagnosis = Diagnosis(
        success=success,
        analysis=f"{analysis} (iteration {iteration})",
        diagnostic_signal=DiagnosticSignal(primary_dim=primary_dim),
    )

    trajectory = Trajectory(
        num_llm_calls=num_llm_calls,
        total_tokens=total_tokens,
        latency_ms=latency_ms,
        tools_invoked=list(tools_invoked or []),
        intermediate_outputs=list(intermediate_outputs or []),
        final_output=final_output,
    )

    features = CaseFeatures(
        input_length=0,
        complexity_estimate=0.5,
        domain="terminal",
        requires_external_knowledge=primary_dim == "D2",
        safety_sensitivity=0.0,
        ambiguity_score=0.0,
        instruction=case_id,
    )

    return PerCaseEntry(
        case_id=case_id,
        iteration=iteration,
        case_features=features,
        config=config.clone(),
        delta_from_prev=config.delta_from(prev_config),
        trajectory=trajectory,
        primary_reward=reward,
        cost_actual=0.0,
        diagnosis=diagnosis,
    )


# ---------------------------------------------------------------------------
# Main loop class
# ---------------------------------------------------------------------------

class HarborTrainingLoop:
    """Orchestrates multi-iteration Harbor runs with ExperienceBank feedback.

    The dataset is automatically split into train (80%) and test (20%) sets
    via ``_split_tasks()`` before training begins.  The split is persisted
    to ``{bank_dir}/{run_id}/bank.pkl.split.json`` so it is stable across restarts.

    Args:
        dataset: Harbor dataset spec, e.g. ``"terminal-bench-sample@2.0"``.
        agent_import_path: Harbor agent import path.
        config_path: Path to the unified MemoHarness runtime config JSON.
        harness_config_path: Path to the live HarnessImpl .py file written between
            iterations (companion .json with dimension summary is written alongside).
        bank_dir: Directory under which per-run bank.pkl files are stored.
        jobs_dir: Root folder for Harbor job outputs.
        distill_every: Trigger distillation after every N *new entries* are added.
        min_consecutive_failures: Also trigger distillation immediately when any case
            reaches this many consecutive failures.
        train_split: Fraction of tasks to use for training (default 0.8).
        seed: Random seed for the train/test split (default 42).
        extra_harbor_args: Additional arguments passed verbatim to ``harbor run``.
    """

    def __init__(
        self,
        dataset: str,
        agent_import_path: str = _DEFAULT_MEMOHARNESS_AGENT_IMPORT,
        run_id: Optional[str] = None,
        config_path: Optional[str] = None,
        harness_config_path: Optional[str] = None,
        bank_dir: str = "artifacts",
        jobs_dir: str = "jobs",
        distill_every: int = 5,
        min_consecutive_failures: int = 3,
        train_split: float = 0.8,
        train_task_limit: Optional[int] = None,
        seed: int = 42,
        n_concurrent: int = 3,
        console_mode: str = "normal",
        console_heartbeat_seconds: int = 30,
        harbor_agent_timeout_seconds: Optional[float] = None,
        verifier_timeout_seconds: Optional[float] = None,
        disable_harbor_verifier_retry: bool = False,
        extra_harbor_args: Optional[list[str]] = None,
        daytona_config: Optional[DaytonaConfig] = None,
    ) -> None:
        self.dataset = dataset
        self.agent_import_path = agent_import_path
        self._configured_run_id = run_id
        self.config_path = config_path or "configs/experiment.json"
        self.harness_config_path = harness_config_path or "configs/harness_terminal.py"
        self.bank_dir = Path(bank_dir)
        self.distill_every = distill_every
        self.min_consecutive_failures = min_consecutive_failures
        self.train_split = train_split
        if train_task_limit is not None and train_task_limit <= 0:
            raise ValueError("train_task_limit must be a positive integer")
        self.train_task_limit = train_task_limit
        self.seed = seed
        self.n_concurrent = n_concurrent
        self.console_mode = str(console_mode or "normal").lower()
        self.console_heartbeat_seconds = max(0, int(console_heartbeat_seconds))
        self.harbor_agent_timeout_seconds = (
            None
            if harbor_agent_timeout_seconds is None
            else float(harbor_agent_timeout_seconds)
        )
        if self.harbor_agent_timeout_seconds is not None and self.harbor_agent_timeout_seconds <= 0:
            raise ValueError("harbor_agent_timeout_seconds must be positive when set")
        self.verifier_timeout_seconds = (
            None if verifier_timeout_seconds is None else float(verifier_timeout_seconds)
        )
        if self.verifier_timeout_seconds is not None and self.verifier_timeout_seconds <= 0:
            raise ValueError("verifier_timeout_seconds must be positive when set")
        self.disable_harbor_verifier_retry = bool(disable_harbor_verifier_retry)
        self.extra_harbor_args, self._deprecated_harbor_args = _normalize_extra_harbor_args(
            list(extra_harbor_args or [])
        )
        self._warned_deprecated_harbor_args = False
        self._task_filter_flag: Optional[str] = None
        self._task_filter_flag_probe_note: Optional[str] = None
        self._task_filter_flag_lock = threading.Lock()
        self._daytona_cfg_explicit = daytona_config is not None
        self._daytona_cfg = daytona_config or DaytonaConfig()
        self._daytona_key_pool = DaytonaKeyPool(self._daytona_cfg)
        self.controller_canary_enabled = False
        self.controller_canary_task_count = 3
        self.controller_canary_min_reward_delta = -0.02
        self.controller_canary_max_blocker_increase = 0
        self.best_harness_selection_modes = [_BEST_HARNESS_MODE_MEAN_REWARD]
        self.test_time_case_adaptation = False

        # jobs_dir defaults to jobs/<sanitized-dataset-name>/ so that runs for
        # different datasets are kept in separate sub-trees and never clash.
        self.jobs_dir = (
            Path(jobs_dir).expanduser().resolve()
            if jobs_dir
            else (Path("jobs") / _sanitize_dirname(dataset))
        )

        # Resolved lazily in _setup()
        self._bank = None
        self._controller = None
        self._distiller = None
        self._current_config = None
        self._current_harness_code: str = ""  # Python source of current HarnessImpl
        self._openai_client = None
        self._runtime = None
        self._train_tasks: list[str] = []
        self._test_tasks: list[str] = []
        self._run_console_log_path: Optional[Path] = None
        self._run_console_stream = None
        self._outer_progress_renderer: Optional[_OuterConsoleProgressRenderer] = None
        self._status_write_warning_paths: set[Path] = set()
        self._status_write_warning_lock = threading.Lock()
        # Set in _setup() — e.g. "terminal-bench-sample_2.0__2026-04-04__19-41-53"
        self._run_id: str = ""
        # Resolved lazily in _setup() after _run_id is known:
        #   <bank_dir>/<run_id>/bank.pkl
        self._resolved_bank_path: Path = None
        self._harness_runtime_mode = "memoharness"
        self._codex_bundle_root: Path | None = None

    def _resolve_harness_paths(self) -> tuple[Path, Path]:
        """Return normalized live-harness (.py) and summary (.json) paths."""
        if self._is_harbor_codex_mode():
            bundle_paths = resolve_codex_bundle_paths(self._resolve_codex_bundle_root())
            return bundle_paths.agents_path, bundle_paths.policy_path
        configured = Path(self.harness_config_path)
        if configured.suffix == ".py":
            return configured, configured.with_suffix(".json")
        if configured.suffix == ".json":
            return configured.with_suffix(".py"), configured
        return configured.with_suffix(".py"), configured.with_suffix(".json")

    def _is_harbor_codex_mode(self) -> bool:
        return self._harness_runtime_mode == "harbor_codex"

    def _resolve_codex_bundle_root(self) -> Path:
        if self._codex_bundle_root is not None:
            return self._codex_bundle_root
        configured = Path(self.harness_config_path).expanduser()
        if configured.suffix:
            self._codex_bundle_root = configured.parent.resolve()
        else:
            self._codex_bundle_root = configured.resolve()
        return self._codex_bundle_root

    def _resolve_harbor_codex_model(self) -> str:
        runtime = getattr(self, "_runtime", None)
        llm = getattr(runtime, "llm", None)
        raw_model = str(getattr(llm, "model", "") or "").strip()
        if not raw_model:
            return ""

        # When a custom base_url is configured we wire it through config.toml
        # (see MemoHarnessCodexAgent._write_codex_config): Harbor should pass
        # the bare model name so Codex routes via model_provider in config.toml
        # rather than trying to re-resolve an openai/<model> pair.
        api_base = str(getattr(llm, "api_base", "") or "").strip()
        if api_base:
            return raw_model

        provider = str(getattr(llm, "provider", "") or "").strip()
        if provider and "/" not in raw_model:
            return f"{provider}/{raw_model}"
        return raw_model

    def _is_financeagent_dataset(self) -> bool:
        dataset_name, _, _ = self.dataset.partition("@")
        return dataset_name.strip().lower() == "financeagent"

    def _resolve_financeagent_prompt_template_path(self) -> Path | None:
        # Prefer the live Codex bundle root (same location Harbor syncs), then
        # fall back to the repository default.
        bundle_candidate = self._resolve_codex_bundle_root() / _FINANCEAGENT_PROMPT_TEMPLATE_NAME
        if bundle_candidate.is_file():
            return bundle_candidate.resolve()
        repo_candidate = _REPO_ROOT / "configs" / "harness_codex" / _FINANCEAGENT_PROMPT_TEMPLATE_NAME
        if repo_candidate.is_file():
            return repo_candidate.resolve()
        return None

    def _resolve_harbor_codex_runtime_env(self, base_env: dict[str, str]) -> dict[str, str]:
        runtime = getattr(self, "_runtime", None)
        llm = getattr(runtime, "llm", None)
        env_updates: dict[str, str] = {}
        codex_home = str(
            base_env.get(_HARBOR_CODEX_HOME_OVERRIDE_ENV, "") or _DEFAULT_HARBOR_CODEX_HOME
        ).strip() or _DEFAULT_HARBOR_CODEX_HOME
        export_root = str(
            base_env.get(_HARBOR_CODEX_EXPORT_ROOT_ENV, "") or _DEFAULT_HARBOR_CODEX_EXPORT_ROOT
        ).strip() or _DEFAULT_HARBOR_CODEX_EXPORT_ROOT
        env_updates[_HARBOR_CODEX_HOME_ENV] = codex_home
        env_updates[_HARBOR_CODEX_HOME_OVERRIDE_ENV] = codex_home
        env_updates[_HARBOR_CODEX_EXPORT_ROOT_ENV] = export_root

        if llm is None:
            return env_updates

        api_key = _resolve_api_key_value(
            getattr(llm, "api_key_env", None),
            environ=base_env,
        )
        if api_key:
            env_updates["OPENAI_API_KEY"] = api_key
            env_updates["CODEX_FORCE_API_KEY"] = "1"

        # Codex CLI ignores OPENAI_BASE_URL for custom providers (see
        # openai/codex#16719). Hand the base_url to MemoHarnessCodexAgent via a
        # dedicated env var so it can write a [model_providers.*] block into
        # $CODEX_HOME/config.toml — the only path Codex actually honors.
        # We still export OPENAI_BASE_URL as well: Harbor's built-in Codex
        # agent class reads it from os.environ and would otherwise default to
        # api.openai.com (confirmed by baseline/run_official_codex_daytona_wrapper.sh).
        api_base = str(getattr(llm, "api_base", "") or "").strip()
        if api_base:
            env_updates["OPENAI_BASE_URL"] = api_base
            env_updates[_HARBOR_CODEX_BASE_URL_ENV] = api_base
            provider_name = str(getattr(llm, "codex_provider_name", "") or "").strip()
            env_updates[_HARBOR_CODEX_PROVIDER_NAME_ENV] = (
                provider_name or _DEFAULT_HARBOR_CODEX_PROVIDER_NAME
            )
            wire_api = str(getattr(llm, "wire_api", "") or "").strip().lower()
            env_updates[_HARBOR_CODEX_WIRE_API_ENV] = (
                wire_api or _DEFAULT_HARBOR_CODEX_WIRE_API
            )

        verifier_overrides = self._build_verifier_env_overrides()
        if verifier_overrides:
            env_updates[_HARBOR_VERIFIER_ENV_OVERRIDES_ENV] = json.dumps(
                verifier_overrides, sort_keys=True
            )

        return env_updates

    def _build_verifier_env_overrides(self) -> dict[str, str]:
        """Overrides applied to Harbor's ``os.environ`` right before verify runs.

        Harbor 0.3.0 has no ``--verifier-env`` flag and ``Verifier.__init__``
        takes no ``override_env`` argument. The only path the task's
        ``[verifier.env]`` ``${OPENAI_API_KEY}`` template can reach is
        ``harbor.utils.env.resolve_env_vars``, which reads ``os.environ`` in
        the Harbor Python process at verify time. ``runtime_patch.py`` wraps
        ``Trial._verify_with_retry`` — we piggyback on that wrapper to
        temporarily rebind ``os.environ[...]`` to the values returned here,
        then restore them after verification.

        Behaviour: when ``experiment.json::providers.openai`` supplies a
        literal OpenAI key (``sk-proj-*`` / ``sk-*`` other than ``sk-or-*``),
        override ``OPENAI_API_KEY`` with that key so the financeagent judge
        hits ``api.openai.com`` successfully. The agent's environment is
        untouched because the swap is scoped to the verifier call.
        """
        runtime = getattr(self, "_runtime", None)
        providers = getattr(runtime, "providers", None) or {}
        openai_provider = providers.get("openai") if isinstance(providers, dict) else None
        if openai_provider is None:
            return {}

        raw = str(getattr(openai_provider, "api_key_env", "") or "").strip()
        if not raw:
            return {}

        api_key = _resolve_api_key_value(raw, environ=dict(os.environ))
        if not api_key:
            return {}
        if api_key.startswith("sk-or-"):
            return {}
        if not (api_key.startswith("sk-") or api_key.startswith("sk_")):
            return {}

        overrides: dict[str, str] = {"OPENAI_API_KEY": api_key}
        base_url = str(getattr(openai_provider, "api_base", "") or "").strip()
        if base_url:
            overrides["OPENAI_BASE_URL"] = base_url
        return overrides

    def _configure_run_console_capture(self) -> None:
        run_dir = self.jobs_dir / self._run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        self._run_console_log_path = run_dir / _RUN_CONSOLE_LOG_NAME
        if (
            isinstance(sys.stderr, _TeeTextStream)
            and sys.stderr.mirror_path == self._run_console_log_path
        ):
            self._run_console_stream = sys.stderr._mirror
            return

        self._run_console_stream = self._run_console_log_path.open(
            "a",
            encoding="utf-8",
            errors="replace",
            buffering=1,
        )
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = _TeeTextStream(old_stdout, self._run_console_stream, mirror_path=self._run_console_log_path)
        sys.stderr = _TeeTextStream(old_stderr, self._run_console_stream, mirror_path=self._run_console_log_path)

        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if not isinstance(handler, logging.StreamHandler):
                continue
            stream = getattr(handler, "stream", None)
            if stream in {old_stdout, sys.__stdout__}:
                handler.setStream(sys.stdout)
            elif stream in {old_stderr, sys.__stderr__}:
                handler.setStream(sys.stderr)

    def _console_heartbeat_enabled(self) -> bool:
        return self.console_mode in {"normal", "debug"} and self.console_heartbeat_seconds > 0

    def _get_outer_progress_renderer(self) -> _OuterConsoleProgressRenderer:
        if self._outer_progress_renderer is None:
            self._outer_progress_renderer = _OuterConsoleProgressRenderer(
                sys.stderr,
                enabled=self.console_mode in {"normal", "debug"},
            )
            self._outer_progress_renderer.attach_stream(sys.stderr)
            if sys.stdout is not sys.stderr:
                self._outer_progress_renderer.attach_stream(sys.stdout)
        return self._outer_progress_renderer

    def _update_outer_progress_targets(
        self,
        targets: list[_OuterProgressTarget],
        *,
        task_state: str,
        inner_stage: str,
        outer_stage: str,
        elapsed_seconds: float,
        agent_elapsed_seconds: Optional[float],
    ) -> None:
        progress_renderer = self._get_outer_progress_renderer()
        if not getattr(progress_renderer, "enabled", False):
            return
        for target in targets:
            progress_renderer.update_task(
                target.line_id,
                target.display_name,
                inner_stage=inner_stage,
                outer_stage=outer_stage,
                elapsed_seconds=elapsed_seconds,
                agent_elapsed_seconds=agent_elapsed_seconds,
                task_state=task_state,
                wave_index=target.wave_index,
                total_waves=target.total_waves,
            )

    def _remove_outer_progress_targets(self, targets: list[_OuterProgressTarget]) -> None:
        progress_renderer = self._get_outer_progress_renderer()
        if not getattr(progress_renderer, "enabled", False):
            return
        for target in targets:
            progress_renderer.remove_task(target.line_id)

    def _write_job_status(
        self,
        path: Path,
        *,
        iteration: int,
        local_name: str,
        task_names: list[str],
        log_path: Path,
        started_at: float,
        phase: str,
        detail: Optional[str] = None,
        returncode: Optional[int] = None,
    ) -> None:
        payload: dict[str, Any] = {
            "iteration": iteration,
            "job_name": local_name,
            "task_names": list(task_names),
            "log_file": log_path.name,
            "phase": phase,
            "started_at": datetime.fromtimestamp(started_at).isoformat(),
            "last_update_at": datetime.now().isoformat(),
            "elapsed_seconds": round(time.time() - started_at, 1),
            "returncode": returncode,
        }
        if detail:
            payload["detail"] = detail
        try:
            _atomic_write_text(path, json.dumps(payload, indent=2))
        except OSError as exc:
            should_warn = False
            with self._status_write_warning_lock:
                if path not in self._status_write_warning_paths:
                    self._status_write_warning_paths.add(path)
                    should_warn = True
            if should_warn:
                logger.warning(
                    "%s | failed to write job status %s: %s",
                    local_name,
                    path,
                    exc,
                )
        else:
            with self._status_write_warning_lock:
                self._status_write_warning_paths.discard(path)

    def _resolve_task_filter_flag(self) -> str:
        if self._task_filter_flag is not None:
            return self._task_filter_flag
        with self._task_filter_flag_lock:
            if self._task_filter_flag is not None:
                return self._task_filter_flag
            detected, note = _probe_harbor_task_filter_flag()
            self._task_filter_flag = detected
            self._task_filter_flag_probe_note = note
            if note.startswith("detected from"):
                logger.info("Using Harbor task filter flag %s (%s).", detected, note)
            else:
                logger.warning("Using Harbor task filter flag %s (%s).", detected, note)
            return detected

    def _config_from_summary(self, raw: dict):
        """Build HarnessConfig from a persisted D1-D6 JSON summary."""
        from memoharness.core.models import HarnessConfig

        if isinstance(raw.get("config"), dict):
            raw = raw["config"]

        return HarnessConfig(**{
            dim: raw.get(dim, {}) for dim in ("D1", "D2", "D3", "D4", "D5", "D6")
        })

    def _extract_config_from_code(self, code: str | None = None):
        """Return the embedded _HARNESS_CONFIG from a rendered harness, if present."""
        from memoharness.core.models import HarnessConfig

        source = code if code is not None else self._current_harness_code
        if not source:
            return None

        try:
            module = ast.parse(source)
        except SyntaxError:
            return None

        for node in module.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(
                isinstance(target, ast.Name) and target.id == "_HARNESS_CONFIG"
                for target in node.targets
            ):
                continue
            value = node.value
            if not (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and isinstance(value.func.value, ast.Name)
                and value.func.value.id == "json"
                and value.func.attr == "loads"
                and value.args
                and isinstance(value.args[0], ast.Constant)
                and isinstance(value.args[0].value, str)
            ):
                continue
            try:
                parsed = json.loads(value.args[0].value)
            except json.JSONDecodeError:
                return None
            if not isinstance(parsed, dict):
                return None
            return HarnessConfig(**{
                dim: parsed.get(dim, {}) for dim in ("D1", "D2", "D3", "D4", "D5", "D6")
            })

        return None

    def _infer_config_from_code(self, code: str | None = None):
        """Best-effort config for an existing HarnessImpl without a trustworthy summary JSON."""
        from memoharness.core.models import make_minimal_config

        parsed = self._extract_config_from_code(code)
        if parsed is not None:
            return parsed

        config = make_minimal_config()
        if hasattr(self._controller, "stabilize_config"):
            try:
                return self._controller.stabilize_config(config)
            except Exception:
                logger.debug("Could not stabilize inferred harness config from code.", exc_info=True)
        config.D2["tool_access"] = "bash"
        config.D4["workflow"] = "agentic_loop"
        return config

    def _write_harness_summary(self, path: Path, config) -> None:
        """Persist a concise D1-D6 JSON summary next to the live harness."""
        _atomic_write_text(path, json.dumps(config.as_dict(), indent=2))

    def _snapshot_live_harness_assets(self) -> dict[Path, str | None]:
        if self._is_harbor_codex_mode():
            return snapshot_codex_bundle(self._resolve_codex_bundle_root())
        harness_py, harness_json = self._resolve_harness_paths()
        snapshot: dict[Path, str | None] = {}
        for path in (harness_py, harness_json):
            snapshot[path] = path.read_text() if path.exists() else None
        return snapshot

    def _restore_live_harness_assets(self, snapshot: dict[Path, str | None]) -> None:
        if self._is_harbor_codex_mode():
            restore_codex_bundle(snapshot)
            return
        for path, content in snapshot.items():
            if content is None:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    logger.debug("Could not remove restored live harness file %s.", path, exc_info=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_text(path, content)

    def _archive_live_harness(
        self,
        *,
        iteration: int,
        preview: str,
        archive_meta_json: str,
    ) -> None:
        archive_dir = self._resolved_bank_path.parent / "harness"
        archive_dir.mkdir(parents=True, exist_ok=True)
        if self._is_harbor_codex_mode():
            bundle_root = self._resolve_codex_bundle_root()
            bundle_archive_dir = archive_dir / f"iter-{iteration:02d}.bundle"
            if bundle_archive_dir.exists():
                shutil.rmtree(bundle_archive_dir)
            shutil.copytree(bundle_root, bundle_archive_dir)
            _atomic_write_text(archive_dir / f"iter-{iteration:02d}.preview.txt", preview)
            _atomic_write_text(archive_dir / f"iter-{iteration:02d}.json", archive_meta_json)
            logger.info("Archived Harbor Codex bundle + stats -> %s/", archive_dir)
            return

        _atomic_write_text(archive_dir / f"iter-{iteration:02d}.py", preview)
        _atomic_write_text(archive_dir / f"iter-{iteration:02d}.json", archive_meta_json)
        logger.info("Archived HarnessImpl + stats -> %s/", archive_dir)

    def _restore_archived_harness(
        self,
        *,
        iteration: int,
        log_prefix: str,
        metric_name: str,
        metric_value: Any,
    ) -> bool:
        archive_dir = self._resolved_bank_path.parent / "harness"
        if self._is_harbor_codex_mode():
            bundle_archive_dir = archive_dir / f"iter-{iteration:02d}.bundle"
            if not bundle_archive_dir.exists():
                logger.warning(
                    "%s archive not found at %s; keeping current harness.",
                    log_prefix,
                    bundle_archive_dir,
                )
                return False
            live_root = self._resolve_codex_bundle_root()
            if live_root.exists():
                shutil.rmtree(live_root)
            shutil.copytree(bundle_archive_dir, live_root)
            preview, config = load_codex_bundle(live_root, fallback=self._current_config)
            self._current_harness_code = preview
            self._current_config = config
            logger.info(
                "%s: iter-%02d (%s=%s) -> restored Harbor Codex bundle at %s",
                log_prefix,
                iteration,
                metric_name,
                self._format_best_metric_value(metric_value),
                live_root,
            )
            return True

        best_archive_py = archive_dir / f"iter-{iteration:02d}.py"
        best_archive_json = archive_dir / f"iter-{iteration:02d}.json"
        if not best_archive_py.exists():
            logger.warning(
                "%s archive not found at %s; keeping current harness.",
                log_prefix,
                best_archive_py,
            )
            return False

        live_py, live_json = self._resolve_harness_paths()
        best_code = best_archive_py.read_text()
        self._current_harness_code = best_code
        _atomic_write_text(live_py, best_code)

        best_config = None
        if best_archive_json.exists():
            try:
                best_config = self._config_from_summary(json.loads(best_archive_json.read_text()))
            except json.JSONDecodeError:
                logger.warning(
                    "Best HarnessImpl archive summary %s was not valid JSON; inferring from code.",
                    best_archive_json,
                )
        if best_config is None:
            best_config = self._infer_config_from_code(best_code)

        self._current_config = best_config
        self._write_harness_summary(live_json, best_config)
        logger.info(
            "%s: iter-%02d (%s=%s) -> written to %s",
            log_prefix,
            iteration,
            metric_name,
            self._format_best_metric_value(metric_value),
            live_py,
        )
        return True

    def _load_daytona_sandbox_cleanup_module(self):
        if not _DAYTONA_SANDBOX_CLEANUP_SCRIPT.exists():
            raise RuntimeError(
                f"Daytona sandbox cleanup script not found: {_DAYTONA_SANDBOX_CLEANUP_SCRIPT}"
            )

        spec = importlib.util.spec_from_file_location(
            "memoharness_daytona_sandbox_cleanup",
            _DAYTONA_SANDBOX_CLEANUP_SCRIPT,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"Could not load Daytona sandbox cleanup script: {_DAYTONA_SANDBOX_CLEANUP_SCRIPT}"
            )

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _cleanup_daytona_sandboxes_before_run(self) -> None:
        """Delete all Daytona sandboxes for configured keys before a run starts."""
        module = self._load_daytona_sandbox_cleanup_module()

        logger.info(
            "Cleaning Daytona sandboxes before run (timeout=%ss) using %s.",
            self._daytona_cfg.cleanup_timeout_seconds,
            _DAYTONA_SANDBOX_CLEANUP_SCRIPT,
        )

        try:
            exit_code = module.delete_all_sandboxes(
                Path(self.config_path).resolve(),
                timeout=self._daytona_cfg.cleanup_timeout_seconds,
            )
        except (SystemExit, Exception) as exc:
            logger.warning("Daytona sandbox cleanup failed (non-fatal): %s", exc)
            return

        if exit_code != 0:
            logger.warning(
                "Daytona sandbox cleanup exited with status %d (non-fatal, continuing).",
                exit_code,
            )

    def _cleanup_daytona_sandboxes_for_key(self, daytona_key: str) -> None:
        """Delete all Daytona sandboxes for one key after its tasks finish."""
        module = self._load_daytona_sandbox_cleanup_module()
        mask_key = getattr(module, "_mask_key", lambda value: value)
        sandbox_label = getattr(module, "_sandbox_label", lambda sandbox: "<unknown>")

        logger.info(
            "Cleaning Daytona sandboxes for completed key %s (timeout=%ss).",
            mask_key(daytona_key),
            self._daytona_cfg.cleanup_timeout_seconds,
        )

        client = module._build_daytona_client(daytona_key)
        sandboxes = list(module._list_sandboxes(client, daytona_key))
        if not sandboxes:
            logger.info(
                "No Daytona sandboxes found for completed key %s.",
                mask_key(daytona_key),
            )
            return

        failed = 0
        for sandbox in sandboxes:
            try:
                module._delete_sandbox(
                    client,
                    daytona_key,
                    sandbox,
                    self._daytona_cfg.cleanup_timeout_seconds,
                )
            except Exception as exc:
                failed += 1
                logger.warning(
                    "Failed to delete Daytona sandbox for completed key %s: %s (%s)",
                    mask_key(daytona_key),
                    sandbox_label(sandbox),
                    exc,
                )

        if failed:
            raise RuntimeError(
                "Failed to delete {0} Daytona sandbox(es) for completed key {1}.".format(
                    failed,
                    mask_key(daytona_key),
                )
            )

    def _update_daytona_key_case_assignments(
        self,
        key_to_cases: dict[str, set[str]],
        case_to_key: dict[str, str],
        new_case_to_key: dict[str, str],
        dirty_keys: set[str],
    ) -> None:
        for case_id, new_key in new_case_to_key.items():
            old_key = case_to_key.get(case_id)
            if old_key is not None:
                key_to_cases.setdefault(old_key, set())
            if old_key and old_key != new_key:
                key_to_cases[old_key].discard(case_id)
            key_to_cases.setdefault(new_key, set()).add(case_id)
            case_to_key[case_id] = new_key
            dirty_keys.add(new_key)

    def _unfinished_daytona_cases_for_cleanup(
        self,
        task_names: list[str],
        results: dict[str, dict[str, Any]],
        *,
        disk_limit_retry_counts: dict[str, int],
        connectivity_retry_counts: dict[str, int],
        timeout_retry_counts: dict[str, int],
        completion_retry_counts: dict[str, int],
    ) -> set[str]:
        unfinished: set[str] = set()

        for case_id in task_names:
            detail = results.get(case_id)
            if detail is None:
                unfinished.add(case_id)
                continue

            kind = _retryable_daytona_kind(detail)
            if kind == "disk_limit" and _should_retry_daytona_kind(
                kind,
                disk_limit_retry_counts.get(case_id, 0),
                disk_limit_retry_limit=self._daytona_cfg.disk_limit_retry_limit,
                connectivity_retry_limit=self._daytona_cfg.connectivity_retry_limit,
            ):
                unfinished.add(case_id)
                continue
            if kind == "connectivity" and _should_retry_daytona_kind(
                kind,
                connectivity_retry_counts.get(case_id, 0),
                disk_limit_retry_limit=self._daytona_cfg.disk_limit_retry_limit,
                connectivity_retry_limit=self._daytona_cfg.connectivity_retry_limit,
            ):
                unfinished.add(case_id)
                continue
            if (
                _is_retryable_timeout(detail)
                and timeout_retry_counts.get(case_id, 0) < self._daytona_cfg.timeout_retry_limit
            ):
                unfinished.add(case_id)
                continue
            if (
                _needs_completion_retry(detail)
                and completion_retry_counts.get(case_id, 0) < self._daytona_cfg.timeout_retry_limit
            ):
                unfinished.add(case_id)

        return unfinished

    def _cleanup_completed_daytona_keys(
        self,
        key_to_cases: dict[str, set[str]],
        unfinished_cases: set[str],
        dirty_keys: set[str],
    ) -> None:
        for daytona_key in sorted(dirty_keys):
            assigned_cases = key_to_cases.get(daytona_key, set())
            if assigned_cases and not assigned_cases.isdisjoint(unfinished_cases):
                continue
            try:
                self._cleanup_daytona_sandboxes_for_key(daytona_key)
            except Exception as exc:
                logger.warning(
                    "Failed to cleanup Daytona sandboxes for completed key %s: %s",
                    daytona_key,
                    exc,
                )
            dirty_keys.discard(daytona_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, iterations: int, *, eval_after_train: bool = False) -> None:
        """Execute the full training loop for *iterations* rounds."""
        self._setup()
        self._split_tasks()

        # Track per-iteration metrics used by the configured best-harness selectors.
        iter_rewards: dict[int, float] = {}
        iter_perfect_success_counts: dict[int, int] = {}
        iter_total_tokens: dict[int, int] = {}

        for iteration in range(1, iterations + 1):
            logger.info("=== Iteration %d / %d (training on %d tasks) ===",
                        iteration, iterations, len(self._train_tasks))

            results = self._collect_results_with_daytona_retries(iteration, self._train_tasks)
            if results is None:
                continue

            learning_results = _filter_learning_results(results)
            mean_reward = (
                sum(detail["reward"] for detail in learning_results.values())
                / max(len(learning_results), 1)
            )
            total_tokens_iter = sum(
                int(detail["total_tokens"]) for detail in learning_results.values()
            )
            perfect_success_count = self._count_perfect_successes(learning_results)
            blocked_cases = len(results) - len(learning_results)
            logger.info(
                "Iteration %d: %d cases — mean reward %.3f — total tokens %d",
                iteration, len(learning_results), mean_reward, total_tokens_iter,
            )
            if blocked_cases:
                logger.info("Iteration %d skipped %d external blocker case(s).", iteration, blocked_cases)
            iter_rewards[iteration] = mean_reward
            iter_perfect_success_counts[iteration] = perfect_success_count
            iter_total_tokens[iteration] = int(total_tokens_iter)

            self._update_bank(learning_results, iteration)
            self._save_bank()

            # Trigger 2: every N new entries have been added since last distillation
            if self._bank.should_distill(self.distill_every):
                logger.info(
                    "Entry-count trigger: %d new entries (threshold=%d).",
                    self._bank.last_distill_entry_count,
                    self.distill_every,
                )
                self._distill(iteration)
                self._bank.mark_distill_done()
                self._save_bank()

            cases_succeeded = sum(
                1 for d in learning_results.values() if d["reward"] >= 0.5
            )
            self._update_config(
                iteration,
                mean_reward=mean_reward,
                total_tokens=total_tokens_iter,
                cases_run=len(learning_results),
                cases_succeeded=cases_succeeded,
                perfect_success_count=perfect_success_count,
                results_for_iteration=results,
            )
            self._cleanup_docker()

        logger.info("Training loop complete after %d iterations.", iterations)
        logger.info("Bank size: %d entries, %d global patterns.",
                    len(self._bank.entries), len(self._bank.global_patterns))

        selected_best_harnesses = self._select_best_harness(
            iter_rewards,
            iter_perfect_success_counts,
            iter_total_tokens,
        )
        if eval_after_train:
            if selected_best_harnesses:
                logger.info(
                    "Starting held-out test evaluation with %d selected best harness mode(s).",
                    len(selected_best_harnesses),
                )
                self._evaluate_selected_best_harnesses(
                    selected_best_harnesses,
                    label_prefix="Post-train eval",
                )
            else:
                logger.warning(
                    "Skipping post-train evaluation because no training iteration produced a reward summary."
                )

    def _run_test_evaluation(
        self,
        *,
        label: str,
        job_name: str = "eval-test",
    ) -> Optional[dict[str, dict[str, Any]]]:
        """Run a single evaluation pass on the held-out test split."""
        if not self._test_tasks:
            logger.warning("No test tasks found in split file - nothing to evaluate.")
            return None

        logger.info("=== %s: %d test tasks ===", label, len(self._test_tasks))
        if self.test_time_case_adaptation:
            results = self._run_test_evaluation_with_case_adaptation(
                label=label,
                job_name=job_name,
            )
        else:
            results = self._collect_results_with_daytona_retries(
                0,
                self._test_tasks,
                job_name=job_name,
            )
        if results is not None:
            learning_results = _filter_learning_results(results)
            mean_reward = (
                sum(detail["reward"] for detail in learning_results.values())
                / max(len(learning_results), 1)
            )
            total_tokens = sum(
                int(detail["total_tokens"]) for detail in learning_results.values()
            )
            perfect_success_count = self._count_perfect_successes(learning_results)
            logger.info(
                "Eval mean reward: %.3f on %d test tasks - perfect successes %d - total tokens %d.",
                mean_reward,
                len(learning_results),
                perfect_success_count,
                total_tokens,
            )
            if len(results) != len(learning_results):
                logger.info(
                    "Eval skipped %d external blocker case(s).",
                    len(results) - len(learning_results),
                )
            for case_id, detail in results.items():
                suffix = " [external blocker]" if detail.get("external_blocker") else ""
                logger.info(
                    "  %s: reward=%.3f  tokens=%d%s",
                    case_id,
                    detail["reward"],
                    int(detail["total_tokens"]),
                    suffix,
                )
        else:
            logger.warning("result.json not found for eval run.")

        self._cleanup_docker()
        return results

    def _run_test_evaluation_with_case_adaptation(
        self,
        *,
        label: str,
        job_name: str,
    ) -> Optional[dict[str, dict[str, Any]]]:
        if self._controller is None:
            logger.warning(
                "Test-time case adaptation is enabled, but controller is unavailable. "
                "Falling back to single-pass evaluation."
            )
            return self._collect_results_with_daytona_retries(
                0,
                self._test_tasks,
                job_name=job_name,
            )
        if not hasattr(self._controller, "adapt_for_case"):
            logger.warning(
                "Test-time case adaptation is enabled, but controller %s does not support adapt_for_case(). "
                "Falling back to single-pass evaluation.",
                type(self._controller).__name__,
            )
            return self._collect_results_with_daytona_retries(
                0,
                self._test_tasks,
                job_name=job_name,
            )
        if self._current_config is None:
            logger.warning(
                "Test-time case adaptation is enabled, but current harness config is missing. "
                "Falling back to single-pass evaluation."
            )
            return self._collect_results_with_daytona_retries(
                0,
                self._test_tasks,
                job_name=job_name,
            )

        logger.info(
            "=== %s: one-shot per-case adaptation enabled for %d test task(s) ===",
            label,
            len(self._test_tasks),
        )
        base_config = self._current_config.clone()
        base_snapshot = self._snapshot_live_harness_assets()
        aggregated: dict[str, dict[str, Any]] = {}
        try:
            for index, task_name in enumerate(self._test_tasks, start=1):
                adapted_config = self._adapt_eval_config_for_case(
                    task_name=task_name,
                    base_config=base_config,
                )
                self._apply_eval_config(adapted_config)
                per_case_job_name = (
                    f"{job_name}-case-{index:03d}-{self._selection_job_suffix(task_name)}"
                )
                logger.info(
                    "Eval case %d/%d: %s",
                    index,
                    len(self._test_tasks),
                    task_name,
                )
                per_case_results = self._collect_results_with_daytona_retries(
                    0,
                    [task_name],
                    job_name=per_case_job_name,
                )
                if per_case_results is None:
                    logger.warning(
                        "Eval case %s did not produce a result payload.",
                        task_name,
                    )
                    continue
                aggregated.update(per_case_results)
        finally:
            self._restore_live_harness_assets(base_snapshot)
            if self._is_harbor_codex_mode():
                self._current_harness_code, self._current_config = load_codex_bundle(
                    self._resolve_codex_bundle_root(),
                    fallback=base_config,
                )
            else:
                self._current_config = base_config

        if not aggregated:
            return None
        return aggregated

    def _adapt_eval_config_for_case(self, *, task_name: str, base_config):
        from memoharness.core.models import BenchmarkCase, CaseFeatures

        task_text = str(task_name or "").strip()
        if not task_text:
            return base_config.clone()

        features = CaseFeatures(
            input_length=len(task_text),
            complexity_estimate=min(1.0, max(0.1, len(task_text) / 80.0)),
            domain="terminal",
            requires_external_knowledge=any(
                token in task_text.lower()
                for token in (
                    "http",
                    "api",
                    "search",
                    "weather",
                    "finance",
                    "stock",
                    "news",
                )
            ),
            safety_sensitivity=0.0,
            ambiguity_score=0.5,
            instruction=task_text,
        )
        case = BenchmarkCase(
            case_id=task_text,
            prompt=task_text,
            expected_output="",
            features=features,
        )
        try:
            decision = self._controller.adapt_for_case(
                bank=self._bank,
                case=case,
                base_config=base_config,
            )
            adapted_config = getattr(decision, "config", None) or base_config
            if hasattr(self._controller, "stabilize_config"):
                adapted_config = self._controller.stabilize_config(adapted_config)
            return adapted_config.clone()
        except Exception as exc:
            logger.warning(
                "Test-time adaptation failed for case %s: %s. Using base harness config.",
                task_name,
                exc,
            )
            return base_config.clone()

    def _apply_eval_config(self, config) -> None:
        if self._is_harbor_codex_mode():
            refresh_codex_bundle_support_docs(
                self._resolve_codex_bundle_root(),
                config,
                distilled_patterns=self._bank.global_patterns,
            )
            self._current_harness_code, self._current_config = load_codex_bundle(
                self._resolve_codex_bundle_root(),
                fallback=config,
            )
            return
        self._current_config = config.clone()

    def run_eval_only(self) -> None:
        """Load bank + split, then run a single eval pass on the test set.

        Use this after ``run()`` to evaluate the trained harness on held-out tasks.
        Requires that ``{bank_dir}/{run_id}/bank.pkl.split.json`` already exists.
        """
        if not self._configured_run_id:
            raise SystemExit("eval-only requires an explicit run_id.")

        self._setup()
        self._split_tasks()   # loads the split saved during training
        selected_best_harnesses = self._load_best_harness_selections()
        if selected_best_harnesses:
            self._evaluate_selected_best_harnesses(
                selected_best_harnesses,
                label_prefix="Eval-only",
            )
            return
        self._run_test_evaluation(label="Eval-only")

    # ------------------------------------------------------------------
    # Docker cleanup (local execution only)
    # ------------------------------------------------------------------

    def _cleanup_docker(self) -> None:
        """Remove stopped containers and dangling images to reclaim disk space.

        Only runs when the environment is local Docker (i.e. ``--env daytona``
        or other cloud providers are NOT in ``extra_harbor_args``).  The cleanup
        is a best-effort operation — failures are logged but never propagate.
        """
        cloud_envs = {"daytona", "e2b", "modal", "runloop", "gke"}
        args_lower = [a.lower() for a in self.extra_harbor_args]
        if any(env in args_lower for env in cloud_envs):
            return

        for description, cmd in (
            ("stopped containers", ["docker", "container", "prune", "-f"]),
            ("dangling images", ["docker", "image", "prune", "-f"]),
            ("build cache", ["docker", "builder", "prune", "-f"]),
        ):
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if result.returncode == 0:
                    reclaimed = result.stdout.strip().splitlines()
                    space_line = [l for l in reclaimed if "reclaimed" in l.lower()]
                    if space_line:
                        logger.info("Docker cleanup (%s): %s", description, space_line[-1])
                else:
                    logger.debug("Docker cleanup (%s) exited %d.", description, result.returncode)
            except Exception as exc:
                logger.debug("Docker cleanup (%s) failed: %s", description, exc)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        from memoharness.config.runtime import MemoHarnessRuntimeConfig
        from memoharness.bank.experience import ExperienceBank
        from memoharness.core.models import make_minimal_config
        from memoharness.llm.client import build_openai_client

        # --- Runtime config & OpenAI client --------------------------------
        runtime = MemoHarnessRuntimeConfig.from_json_file(Path(self.config_path))
        self._runtime = runtime
        if self.harbor_agent_timeout_seconds is None:
            configured_agent_timeout = getattr(
                runtime.experiment,
                "harbor_agent_timeout_seconds",
                None,
            )
            if configured_agent_timeout is not None:
                self.harbor_agent_timeout_seconds = float(configured_agent_timeout)
        if self.verifier_timeout_seconds is None:
            configured_verifier_timeout = getattr(
                runtime.experiment,
                "verifier_timeout_seconds",
                None,
            )
            if configured_verifier_timeout is not None:
                self.verifier_timeout_seconds = float(configured_verifier_timeout)
        self._harness_runtime_mode = str(
            getattr(runtime.harness, "agent_runtime", "memoharness") or "memoharness"
        ).lower()
        if not self._is_harbor_codex_mode():
            raise ValueError(
                "configs/experiment.json chain only supports harness.agent_runtime='harbor_codex'."
            )
        if self._is_harbor_codex_mode():
            configured_bundle = str(
                getattr(runtime.harness, "codex_bundle_path", self.harness_config_path)
                or self.harness_config_path
            ).strip()
            if configured_bundle:
                self._codex_bundle_root = Path(configured_bundle).expanduser().resolve()
                self.harness_config_path = str(self._codex_bundle_root)
            if self.agent_import_path == _DEFAULT_MEMOHARNESS_AGENT_IMPORT:
                self.agent_import_path = _DEFAULT_MEMOHARNESS_CODEX_AGENT_IMPORT
        self.console_mode = str(
            getattr(runtime.experiment, "console_mode", self.console_mode) or self.console_mode
        ).lower()
        self.console_heartbeat_seconds = max(
            0,
            int(
                getattr(
                    runtime.experiment,
                    "console_heartbeat_seconds",
                    self.console_heartbeat_seconds,
                )
                or self.console_heartbeat_seconds
            ),
        )
        if not self._daytona_cfg_explicit:
            self._daytona_cfg = runtime.experiment.daytona
            self._daytona_key_pool = DaytonaKeyPool(self._daytona_cfg)
        self.controller_canary_enabled = bool(
            getattr(runtime.harness, "controller_canary_enabled", False)
        )
        self.controller_canary_task_count = max(
            0,
            int(getattr(runtime.harness, "controller_canary_task_count", 3) or 0),
        )
        self.controller_canary_min_reward_delta = float(
            getattr(runtime.harness, "controller_canary_min_reward_delta", -0.02)
            if getattr(runtime.harness, "controller_canary_min_reward_delta", None) is not None
            else -0.02
        )
        self.controller_canary_max_blocker_increase = max(
            0,
            int(getattr(runtime.harness, "controller_canary_max_blocker_increase", 0) or 0),
        )
        self.best_harness_selection_modes = list(
            getattr(
                runtime.harness,
                "best_harness_selection_modes",
                [_BEST_HARNESS_MODE_MEAN_REWARD],
            )
            or [_BEST_HARNESS_MODE_MEAN_REWARD]
        )
        self.test_time_case_adaptation = bool(
            getattr(runtime.harness, "test_time_case_adaptation", False)
        )
        self._openai_client = build_openai_client(runtime.llm)

        # Generate run_id first — all paths (bank, jobs, harness) depend on it.
        if self._configured_run_id:
            self._run_id = self._configured_run_id
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
            self._run_id = f"{_sanitize_dirname(self.dataset)}__{timestamp}"
        # Resolve the per-run bank path.
        # Each dataset run gets its own <bank_dir>/<run_id>/bank.pkl, so different
        # datasets never clash on the same pickle file.
        self._resolved_bank_path = self.bank_dir / self._run_id / "bank.pkl"

        # Ensure parent dirs exist before we try to load the bank
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._resolved_bank_path.parent.mkdir(parents=True, exist_ok=True)
        self._configure_run_console_capture()
        logger.info("Run ID: %s", self._run_id)
        logger.info("Run console log: %s", self._run_console_log_path)

        if self._daytona_cfg.cleanup_sandboxes_before_run and self._daytona_key_pool.enabled:
            self._cleanup_daytona_sandboxes_before_run()
        elif self._daytona_cfg.cleanup_sandboxes_before_run:
            logger.info(
                "Skipping Daytona sandbox cleanup before run because no Daytona API keys are configured."
            )

        # --- ExperienceBank (resume from disk or fresh) --------------------
        if self._resolved_bank_path.exists():
            self._bank = ExperienceBank.load(self._resolved_bank_path)
            # Update threshold in case it changed since the bank was saved
            self._bank.min_consecutive_failures = self.min_consecutive_failures
            logger.info("Resumed ExperienceBank from %s (%d entries).",
                        self._resolved_bank_path, len(self._bank.entries))
        else:
            self._bank = ExperienceBank(min_consecutive_failures=self.min_consecutive_failures)
            logger.info("Starting fresh ExperienceBank.")

        # --- Controller ----------------------------------------------------
        controller_kind = getattr(runtime.harness, "controller", "codex")
        if controller_kind != "codex":
            raise ValueError(
                "configs/experiment.json chain only supports harness.controller='codex'."
            )
        from memoharness.controllers.codex_bundle import CodexBundleController

        self._controller = CodexBundleController(
            bundle_root=self._resolve_codex_bundle_root(),
            command=runtime.harness.controller_command,
            args=runtime.harness.controller_args,
            use_stdin=runtime.harness.controller_use_stdin,
            timeout_seconds=runtime.harness.controller_timeout_seconds,
            recent_iterations=runtime.harness.controller_recent_iterations,
            workspace_root=_REPO_ROOT,
            jobs_dir=self.jobs_dir,
            run_id=self._run_id,
            bank_path=self._resolved_bank_path,
            artifact_dir=self._resolved_bank_path.parent,
            dataset=self.dataset,
            model=runtime.harness.controller_model,
            api_config=runtime.llm,
            env=runtime.harness.controller_env or None,
            heartbeat_seconds=self.console_heartbeat_seconds,
        )
        logger.info(
            "Using %s (command=%s bundle_root=%s).",
            type(self._controller).__name__,
            self._controller.command,
            self._resolve_codex_bundle_root(),
        )

        # --- LLM Distiller ------------------------------------------------
        try:
            from memoharness.llm.distiller import LLMDistiller
            self._distiller = LLMDistiller(
                client=self._openai_client,
                model=runtime.llm.model,
                api_config=runtime.llm,
            )
            logger.info("LLMDistiller enabled (model=%s).", runtime.llm.model)
        except Exception as exc:
            logger.warning("LLMDistiller unavailable: %s — using heuristic.", exc)
            self._distiller = None

        # --- Current HarnessImpl (code + config) ---------------------------
        # The configured path may point to the live .py or to the summary .json.
        harness_py, harness_json = self._resolve_harness_paths()
        if self._is_harbor_codex_mode():
            bundle_root = self._resolve_codex_bundle_root()
            bundle_paths = resolve_codex_bundle_paths(bundle_root)
            if any(path.exists() for path in bundle_paths.tracked_files()):
                ensure_codex_bundle(bundle_root, make_minimal_config())
                self._current_harness_code, self._current_config = load_codex_bundle(
                    bundle_root,
                    fallback=make_minimal_config(),
                )
                logger.info("Loaded Harbor Codex bundle from %s.", bundle_root)
            else:
                try:
                    code, config = self._controller.generate_initial_harness(dataset=self.dataset)
                    self._current_harness_code = code
                    self._current_config = config
                    logger.info(
                        "No Codex bundle found - controller generated initial bundle for '%s' -> %s.",
                        self.dataset,
                        bundle_root,
                    )
                except Exception as exc:
                    logger.warning(
                        "Controller could not generate Codex bundle (%s) - creating minimal bundle.",
                        exc,
                    )
                    ensure_codex_bundle(bundle_root, make_minimal_config())
                    self._current_harness_code, self._current_config = load_codex_bundle(
                        bundle_root,
                        fallback=make_minimal_config(),
                    )
        elif harness_py.exists():
            self._current_harness_code = harness_py.read_text()
            code_config = self._extract_config_from_code(self._current_harness_code)
            if harness_json.exists():
                try:
                    raw = json.loads(harness_json.read_text())
                except json.JSONDecodeError:
                    self._current_config = self._infer_config_from_code(self._current_harness_code)
                    self._write_harness_summary(harness_json, self._current_config)
                    logger.warning(
                        "Companion summary %s was not valid JSON — rewrote an inferred summary "
                        "for live harness %s.",
                        harness_json, harness_py,
                    )
                else:
                    summary_config = self._config_from_summary(raw)
                    if code_config is not None and summary_config.as_dict() != code_config.as_dict():
                        self._current_config = code_config
                        self._write_harness_summary(harness_json, self._current_config)
                        logger.warning(
                            "Companion summary %s disagreed with live harness %s — "
                            "rewrote the summary from the embedded harness config.",
                            harness_json, harness_py,
                        )
                    else:
                        self._current_config = code_config or summary_config
                        logger.info(
                            "Loaded HarnessImpl + companion config from %s / %s.",
                            harness_py, harness_json,
                        )
            else:
                self._current_config = self._infer_config_from_code(self._current_harness_code)
                self._write_harness_summary(harness_json, self._current_config)
                logger.warning(
                    "No companion .json for %s — wrote inferred config to %s. "
                    "The controller will overwrite it in the next iteration.",
                    harness_py, harness_json,
                )
        elif harness_json.exists():
            # Legacy JSON config or a misplaced Python harness living in a .json file.
            raw_text = harness_json.read_text()
            try:
                raw = json.loads(raw_text)
            except json.JSONDecodeError:
                self._current_harness_code = raw_text
                self._current_config = self._infer_config_from_code()
                _atomic_write_text(harness_py, self._current_harness_code)
                self._write_harness_summary(harness_json, self._current_config)
                logger.warning(
                    "Found Python harness code in %s — migrated live harness to %s and "
                    "rewrote %s as a JSON summary.",
                    harness_json, harness_py, harness_json,
                )
            else:
                self._current_config = self._config_from_summary(raw)
                logger.info(
                    "Loaded HarnessConfig JSON from %s — rendering HarnessImpl at %s.",
                    harness_json, harness_py,
                )
                try:
                    if hasattr(self._controller, "_render_harness_code"):
                        code = self._controller._render_harness_code(self._current_config)
                    else:
                        code = ""
                    if not code:
                        raise ValueError("Controller could not render a live harness from summary.")
                    self._current_harness_code = code
                    _atomic_write_text(harness_py, self._current_harness_code)
                    self._write_harness_summary(harness_json, self._current_config)
                    logger.info("Rendered HarnessImpl from summary JSON → %s.", harness_py)
                except Exception as exc:
                    logger.warning("Could not render HarnessImpl from summary (%s) — using minimal.", exc)
                    self._current_harness_code = ""
            self._current_config = make_minimal_config()
        else:
            # No existing harness — ask the LLM controller to generate one.
            try:
                code, config = self._controller.generate_initial_harness(dataset=self.dataset)
                self._current_harness_code = code
                self._current_config = config
                _atomic_write_text(harness_py, code)
                self._write_harness_summary(harness_json, self._current_config)
                logger.info(
                    "No harness found — controller generated initial HarnessImpl for '%s' → %s.",
                    self.dataset, harness_py,
                )
            except Exception as exc:
                logger.warning(
                    "Controller could not generate HarnessImpl (%s) — falling back to minimal.",
                    exc,
                )
                self._current_harness_code = ""
                self._current_config = make_minimal_config()

        if (
            not self._is_harbor_codex_mode()
            and
            self._current_harness_code
            and hasattr(self._controller, "normalize_harness_code")
            and hasattr(self._controller, "stabilize_config")
        ):
            normalized_config = self._controller.stabilize_config(self._current_config)
            normalized_code = self._controller.normalize_harness_code(
                self._current_harness_code,
                normalized_config,
            )
            if (
                normalized_code != self._current_harness_code
                or normalized_config.as_dict() != self._current_config.as_dict()
            ):
                self._current_harness_code = normalized_code
                self._current_config = normalized_config
                _atomic_write_text(harness_py, normalized_code)
                self._write_harness_summary(harness_json, normalized_config)
                logger.info(
                    "Normalized live HarnessImpl before iteration 1 at %s.",
                    harness_py,
                )

    # ------------------------------------------------------------------
    # Train / test split
    # ------------------------------------------------------------------

    def _get_all_tasks(self) -> list[str]:
        """Query Harbor for the complete task list in the dataset.

        Uses ``harbor.models.registry.Registry.from_url()`` to fetch the
        registry manifest, then searches for the matching dataset and
        returns the names of all its tasks.

        ``requests`` is used directly for network I/O so that it can opt out
        of the sandbox allowlist when the surrounding command already has
        network access (e.g. when running inside a Daytona environment).

        Raises:
            RuntimeError: if the dataset cannot be found in the registry or
                the network request fails.
        """
        # Hardcoded: matches the default registry used by the harbor CLI (v0.1.32).
        # Using requests directly so callers can control the network context
        # (e.g. opt out of sandbox allowlist when running inside Daytona).
        REGISTRY_URL = (
            "https://raw.githubusercontent.com/laude-institute/harbor/main/registry.json"
        )
        try:
            registry = Registry.from_url(REGISTRY_URL)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch Harbor registry from {REGISTRY_URL}: {exc}. "
                "Check your network connection."
            ) from exc

        target_name, _, target_version = self.dataset.partition("@")
        target_version = target_version or None

        for ds in registry.datasets:
            if ds.name != target_name:
                continue
            if target_version is not None and ds.version != target_version:
                continue
            task_names = [task.get_name() for task in ds.tasks]
            if task_names:
                return task_names

        raise RuntimeError(
            f"Dataset '{self.dataset}' not found in Harbor registry. "
            "Run 'harbor datasets list' to see available datasets."
        )

    def _random_split(self, tasks: list[str], split: float) -> tuple[list[str], list[str]]:
        """Shuffle *tasks* with the configured seed and split at *split*."""
        import random

        rng = random.Random(self.seed)
        shuffled = rng.sample(tasks, len(tasks))
        if split <= 0.0:
            return [], shuffled
        if split >= 1.0:
            return shuffled, []
        n_train = max(1, min(len(shuffled) - 1, round(len(shuffled) * split)))
        return shuffled[:n_train], shuffled[n_train:]

    def _apply_train_task_limit(self, train_tasks: list[str]) -> list[str]:
        """Return the active training subset after applying ``train_task_limit``."""
        if self.train_task_limit is None or self.train_task_limit >= len(train_tasks):
            return list(train_tasks)
        limited = list(train_tasks[: self.train_task_limit])
        logger.info(
            "Applying train task limit: using %d of %d training tasks.",
            len(limited), len(train_tasks),
        )
        return limited

    def _split_tasks(self) -> None:
        """Randomly split the dataset into train and test sets.

        Reads previously saved split from ``{bank_dir}/{run_id}/bank.pkl.split.json`` if it
        exists, so that training and evaluation use the *same* split.
        """
        split_file = Path(str(self._resolved_bank_path) + ".split.json")

        if split_file.exists():
            data = json.loads(split_file.read_text())
            full_train = data["train"]
            self._train_tasks = self._apply_train_task_limit(full_train)
            self._test_tasks = data["test"]
            logger.info(
                "Loaded existing train/test split from %s: %d train (%d active), %d test.",
                split_file, len(full_train), len(self._train_tasks), len(self._test_tasks),
            )
            return

        all_tasks = self._get_all_tasks()
        full_train, self._test_tasks = self._random_split(all_tasks, self.train_split)
        self._train_tasks = self._apply_train_task_limit(full_train)

        split_file.write_text(json.dumps({
            "train": full_train,
            "test": self._test_tasks,
            "seed": self.seed,
            "train_split": self.train_split,
            "train_task_limit": self.train_task_limit,
        }))
        logger.info(
            "Split %d tasks into %d train (%d active) / %d test (seed=%d). "
            "Split saved to %s.",
            len(all_tasks), len(full_train), len(self._train_tasks), len(self._test_tasks),
            self.seed, split_file,
        )

    # ------------------------------------------------------------------
    # Harbor subprocess
    # ------------------------------------------------------------------

    def _run_harbor(
        self,
        iteration: int,
        task_names: list[str],
        job_name: Optional[str] = None,
        daytona_key: Optional[str] = None,
        n_concurrent: Optional[int] = None,
        progress_targets: Optional[list[_OuterProgressTarget]] = None,
        remove_progress_on_exit: bool = True,
    ) -> Path:
        """Run ``harbor run`` for this iteration and return the job directory.

        All jobs for this run are placed under ``<jobs_dir>/<run_id>/`` so
        that repeated experiments are kept in separate sub-trees and never
        overwrite one another.
        """
        local_name = job_name or f"iter-{iteration:02d}"
        # Compose the full job name that Harbor will use as the folder name.
        # Harbor creates <cwd>/jobs/<full_job_name>/ when --job-name is given.
        full_job_name = f"{self._run_id}/{local_name}"
        job_dir = self.jobs_dir / self._run_id / local_name

        env = os.environ.copy()
        # Daytona exec sessions use aiohttp WebSockets which break through
        # local HTTP proxies.  Add Daytona domains to NO_PROXY so they
        # connect directly, while keeping the proxy for GitHub/OpenRouter.
        _daytona_no_proxy = "proxy.app.daytona.io,app.daytona.io,api.daytona.io"
        existing_no_proxy = env.get("NO_PROXY", env.get("no_proxy", ""))
        merged = f"{existing_no_proxy},{_daytona_no_proxy}" if existing_no_proxy else _daytona_no_proxy
        env["NO_PROXY"] = merged
        env["no_proxy"] = merged
        harness_py, _ = self._resolve_harness_paths()
        env["MEMOHARNESS_CONFIG"] = str(Path(self.config_path).resolve())
        harbor_codex_runtime_env: dict[str, str] = {}
        if self._is_harbor_codex_mode():
            env[_HARBOR_CODEX_BUNDLE_ENV] = str(self._resolve_codex_bundle_root().resolve())
            harbor_codex_runtime_env = self._resolve_harbor_codex_runtime_env(env)
            env.update(harbor_codex_runtime_env)
            prompt_template = str(env.get(_HARBOR_CODEX_PROMPT_TEMPLATE_ENV, "") or "").strip()
            if not prompt_template and self._is_financeagent_dataset():
                resolved_prompt_template = self._resolve_financeagent_prompt_template_path()
                if resolved_prompt_template is not None:
                    env[_HARBOR_CODEX_PROMPT_TEMPLATE_ENV] = str(resolved_prompt_template)
                else:
                    logger.warning(
                        "Dataset %s detected but %s could not be resolved; "
                        "Codex prompt template injection will be skipped.",
                        self.dataset,
                        _HARBOR_CODEX_PROMPT_TEMPLATE_ENV,
                    )
        else:
            env["MEMOHARNESS_HARNESS_PATH"] = str(harness_py.resolve())
        env["MEMOHARNESS_BANK_PATH"] = str(self._resolved_bank_path.resolve())
        runtime_harness = getattr(getattr(self, "_runtime", None), "harness", None)
        if runtime_harness is not None:
            env["MEMOHARNESS_TOOL_PROTOCOL"] = str(
                getattr(runtime_harness, "tool_protocol", "native") or "native"
            )
        patch_path = str((_REPO_ROOT / "src").resolve())
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{patch_path}{os.pathsep}{existing_pythonpath}"
            if existing_pythonpath
            else patch_path
        )
        env[_HARBOR_RUNTIME_PATCH_ENV] = "1"
        if self.harbor_agent_timeout_seconds is not None:
            env[_HARBOR_AGENT_TIMEOUT_ENV] = f"{self.harbor_agent_timeout_seconds:g}"
        env[_HARBOR_DISABLE_VERIFIER_RETRY_ENV] = (
            "1" if self.disable_harbor_verifier_retry else "0"
        )
        if self.verifier_timeout_seconds is not None:
            env[_HARBOR_VERIFIER_TIMEOUT_ENV] = f"{self.verifier_timeout_seconds:g}"
        if daytona_key is not None:
            env["DAYTONA_API_KEY"] = daytona_key

        if self._deprecated_harbor_args and not self._warned_deprecated_harbor_args:
            logger.warning(
                "Ignoring deprecated Harbor CLI flag(s) in extra_harbor_args: %s. "
                "Current Harbor no longer accepts them.",
                ", ".join(sorted(set(self._deprecated_harbor_args))),
            )
            self._warned_deprecated_harbor_args = True

        task_filter_flag = self._resolve_task_filter_flag()
        harbor_extra_args = list(self.extra_harbor_args)
        harbor_model = ""
        if self._is_harbor_codex_mode() and not _harbor_args_specify_model(harbor_extra_args):
            harbor_model = self._resolve_harbor_codex_model()
            if not harbor_model:
                raise ValueError(
                    "harbor_codex mode requires a Harbor model id. "
                    "Set the runtime llm model in the experiment config or pass --model "
                    "via experiment.extra_harbor_args."
                )
            harbor_extra_args += ["--model", harbor_model]

        cmd = [
            "harbor", "run",
            "-d", self.dataset,
            "--agent-import-path", self.agent_import_path,
            "--job-name", full_job_name,
            "--n-concurrent", str(n_concurrent if n_concurrent is not None else self.n_concurrent),
        ] + harbor_extra_args
        for task in task_names:
            cmd += [task_filter_flag, task]

        job_dir.mkdir(parents=True, exist_ok=True)
        combined_log_path = job_dir / _HARBOR_COMBINED_LOG_NAME
        launcher_meta_path = job_dir / _HARBOR_LAUNCHER_META_NAME
        status_path = job_dir / _HARBOR_STATUS_NAME
        started_at = time.time()
        task_summary = _summarize_task_names(task_names)
        progress_renderer = self._get_outer_progress_renderer()
        renderer_enabled = bool(getattr(progress_renderer, "enabled", False))
        progress_targets = list(progress_targets or [
            _OuterProgressTarget(
                line_id=local_name,
                display_name=task_summary,
            )
        ])
        logger.info(
            "Starting %s: %d task(s): %s. Log: %s",
            local_name,
            len(task_names),
            task_summary,
            combined_log_path,
        )
        if self.console_mode == "debug":
            logger.debug("Harbor command for %s: %s", local_name, " ".join(cmd))

        launcher_meta: dict[str, Any] = {
            "iteration": iteration,
            "job_name": local_name,
            "full_job_name": full_job_name,
            "dataset": self.dataset,
            "task_names": list(task_names),
            "task_filter_flag": task_filter_flag,
            "command": list(cmd),
            "cwd": str(Path.cwd()),
            "log_file": combined_log_path.name,
            "started_at": datetime.fromtimestamp(started_at).isoformat(),
            "n_concurrent": int(n_concurrent if n_concurrent is not None else self.n_concurrent),
            "daytona_key": _mask_secret(daytona_key),
            "max_tasks_per_shard": self._daytona_cfg.max_tasks_per_shard,
            "shard_timeout_seconds": self._daytona_cfg.shard_timeout_seconds,
            "disable_harbor_verifier_retry": self.disable_harbor_verifier_retry,
            "env": {
                "MEMOHARNESS_CONFIG": env["MEMOHARNESS_CONFIG"],
                "MEMOHARNESS_BANK_PATH": env["MEMOHARNESS_BANK_PATH"],
                "MEMOHARNESS_TOOL_PROTOCOL": env.get("MEMOHARNESS_TOOL_PROTOCOL", ""),
                "NO_PROXY": env["NO_PROXY"],
                _HARBOR_RUNTIME_PATCH_ENV: env[_HARBOR_RUNTIME_PATCH_ENV],
                _HARBOR_DISABLE_VERIFIER_RETRY_ENV: env[_HARBOR_DISABLE_VERIFIER_RETRY_ENV],
            },
            "returncode": None,
        }
        if self.harbor_agent_timeout_seconds is not None:
            launcher_meta["harbor_agent_timeout_seconds"] = self.harbor_agent_timeout_seconds
            launcher_meta["env"][_HARBOR_AGENT_TIMEOUT_ENV] = env[_HARBOR_AGENT_TIMEOUT_ENV]
        if self.verifier_timeout_seconds is not None:
            launcher_meta["verifier_timeout_seconds"] = self.verifier_timeout_seconds
            launcher_meta["env"][_HARBOR_VERIFIER_TIMEOUT_ENV] = env[_HARBOR_VERIFIER_TIMEOUT_ENV]
        if self._is_harbor_codex_mode():
            launcher_meta["env"][_HARBOR_CODEX_BUNDLE_ENV] = env.get(_HARBOR_CODEX_BUNDLE_ENV, "")
            launcher_meta["env"][_HARBOR_CODEX_HOME_ENV] = env.get(_HARBOR_CODEX_HOME_ENV, "")
            launcher_meta["env"][_HARBOR_CODEX_HOME_OVERRIDE_ENV] = env.get(
                _HARBOR_CODEX_HOME_OVERRIDE_ENV,
                "",
            )
            launcher_meta["env"][_HARBOR_CODEX_EXPORT_ROOT_ENV] = env.get(
                _HARBOR_CODEX_EXPORT_ROOT_ENV,
                "",
            )
            if env.get(_HARBOR_CODEX_PROMPT_TEMPLATE_ENV):
                launcher_meta["env"][_HARBOR_CODEX_PROMPT_TEMPLATE_ENV] = env[
                    _HARBOR_CODEX_PROMPT_TEMPLATE_ENV
                ]
            launcher_meta["harbor_model"] = harbor_model or self._resolve_harbor_codex_model()
            if harbor_codex_runtime_env.get("OPENAI_API_KEY"):
                launcher_meta["env"]["OPENAI_API_KEY"] = _mask_secret(
                    harbor_codex_runtime_env["OPENAI_API_KEY"]
                )
            if harbor_codex_runtime_env.get(_HARBOR_CODEX_BASE_URL_ENV):
                launcher_meta["env"][_HARBOR_CODEX_BASE_URL_ENV] = harbor_codex_runtime_env[
                    _HARBOR_CODEX_BASE_URL_ENV
                ]
            if harbor_codex_runtime_env.get("OPENAI_BASE_URL"):
                launcher_meta["env"]["OPENAI_BASE_URL"] = harbor_codex_runtime_env[
                    "OPENAI_BASE_URL"
                ]
            if harbor_codex_runtime_env.get(_HARBOR_CODEX_PROVIDER_NAME_ENV):
                launcher_meta["env"][_HARBOR_CODEX_PROVIDER_NAME_ENV] = harbor_codex_runtime_env[
                    _HARBOR_CODEX_PROVIDER_NAME_ENV
                ]
            if harbor_codex_runtime_env.get(_HARBOR_CODEX_WIRE_API_ENV):
                launcher_meta["env"][_HARBOR_CODEX_WIRE_API_ENV] = harbor_codex_runtime_env[
                    _HARBOR_CODEX_WIRE_API_ENV
                ]
            if harbor_codex_runtime_env.get("CODEX_FORCE_API_KEY"):
                launcher_meta["env"]["CODEX_FORCE_API_KEY"] = harbor_codex_runtime_env[
                    "CODEX_FORCE_API_KEY"
                ]
        else:
            launcher_meta["env"]["MEMOHARNESS_HARNESS_PATH"] = env["MEMOHARNESS_HARNESS_PATH"]
        _atomic_write_text(launcher_meta_path, json.dumps(launcher_meta, indent=2))
        self._write_job_status(
            status_path,
            iteration=iteration,
            local_name=local_name,
            task_names=task_names,
            log_path=combined_log_path,
            started_at=started_at,
            phase="starting harbor run",
        )
        if renderer_enabled:
            self._update_outer_progress_targets(
                progress_targets,
                task_state="launching",
                inner_stage="starting_harness_run",
                outer_stage="launching_harbor",
                elapsed_seconds=0.0,
                agent_elapsed_seconds=None,
            )

        with combined_log_path.open(
            "w",
            encoding="utf-8",
            errors="replace",
            buffering=1,
        ) as combined_log:
            process = subprocess.Popen(
                cmd,
                env=env,
                cwd=str(Path.cwd()),
                stdout=combined_log,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            last_phase = "starting harbor run"
            last_detail = ""
            next_heartbeat = started_at + self.console_heartbeat_seconds
            observed_exception_tasks: set[str] = set()
            termination_phase: Optional[str] = None
            termination_detail: Optional[str] = None
            termination_reason: Optional[str] = None
            shard_timeout_seconds = float(self._daytona_cfg.shard_timeout_seconds)
            shard_kill_grace_seconds = float(self._daytona_cfg.shard_kill_grace_seconds)

            def _terminate_running_harbor(
                *,
                phase: str,
                detail: str,
                reason: str,
                warning_message: str,
                kill_warning_message: str,
                wait_seconds: float,
            ) -> int:
                nonlocal termination_phase, termination_detail, termination_reason
                termination_phase = phase
                termination_detail = detail
                termination_reason = reason
                logger.warning("%s | %s | %s", local_name, task_summary, warning_message)
                process.terminate()
                try:
                    process.wait(timeout=wait_seconds)
                except subprocess.TimeoutExpired:
                    logger.warning("%s | %s | %s", local_name, task_summary, kill_warning_message)
                    process.kill()
                    process.wait(timeout=wait_seconds)
                return process.poll()

            while True:
                returncode = process.poll()
                elapsed_now = time.time() - started_at
                timed_out_task: Optional[str] = None
                timeout_exception_type: Optional[str] = None
                if returncode is None:
                    timeout_hit: Optional[tuple[str, str]] = None
                    if self._daytona_cfg.stop_on_timeout_per_task:
                        timeout_hit = _find_timeout_trial(
                            job_dir,
                            allowed_exception_types=set(_TIMEOUT_EXCEPTION_TYPES),
                        )
                        if timeout_hit is None:
                            timeout_hit = _find_timeout_trial_from_log(
                                combined_log_path,
                                allowed_exception_types=set(_TIMEOUT_EXCEPTION_TYPES),
                            )
                    else:
                        timed_out_task = _find_agent_timeout_trial(job_dir)
                        if timed_out_task:
                            timeout_hit = (timed_out_task, "AgentTimeoutError")

                    if timeout_hit is not None:
                        timed_out_task, timeout_exception_type = timeout_hit
                        if not timed_out_task and len(task_names) == 1:
                            timed_out_task = task_names[0]
                        if timed_out_task:
                            observed_exception_tasks.add(timed_out_task)

                        timeout_label = (
                            "verifier timeout"
                            if timeout_exception_type == "VerifierTimeoutError"
                            else "agent timeout"
                        )
                        if self._daytona_cfg.stop_on_timeout_per_task and len(task_names) > 1:
                            logger.warning(
                                "%s | %s | detected %s for %s, but this job has %d tasks; "
                                "skip early termination to avoid affecting unrelated tasks.",
                                local_name,
                                task_summary,
                                timeout_exception_type,
                                timed_out_task or "(unknown task)",
                                len(task_names),
                            )
                        else:
                            returncode = _terminate_running_harbor(
                                phase=f"terminated after {timeout_label}",
                                detail=timed_out_task or "(unknown task)",
                                reason=(
                                f"{timeout_label}: {timed_out_task}"
                                if timed_out_task
                                else timeout_label
                                ),
                                warning_message=(
                                    "detected {0} for {1}; terminating harbor run early.".format(
                                        timeout_exception_type,
                                        timed_out_task or "(unknown task)",
                                    )
                                ),
                                kill_warning_message=(
                                    "harbor run did not exit after terminate(); killing process."
                                ),
                                wait_seconds=5,
                            )

                    if returncode is None:
                        terminal_tasks, exception_tasks = _observed_shard_terminal_tasks(
                            job_dir,
                            task_names,
                            extra_exception_tasks=observed_exception_tasks,
                        )
                        observed_exception_tasks.update(exception_tasks)
                        if observed_exception_tasks and len(terminal_tasks) >= len(task_names):
                            exception_summary = _summarize_task_names(
                                sorted(observed_exception_tasks)
                            )
                            returncode = _terminate_running_harbor(
                                phase="terminated after shard exception drain",
                                detail=exception_summary,
                                reason=(
                                    "shard tasks finished after observed exceptions: "
                                    f"{exception_summary}"
                                ),
                                warning_message=(
                                    "observed exception task(s) {0} and all shard tasks reached "
                                    "terminal state; terminating harbor run early.".format(
                                        exception_summary
                                    )
                                ),
                                kill_warning_message=(
                                    "harbor run did not exit after shard exception drain terminate(); "
                                    "killing process."
                                ),
                                wait_seconds=5,
                            )

                    if (
                        returncode is None
                        and shard_timeout_seconds > 0
                        and elapsed_now >= shard_timeout_seconds
                    ):
                        returncode = _terminate_running_harbor(
                            phase="terminated after shard wall clock timeout",
                            detail=f"{int(round(shard_timeout_seconds))}s limit reached",
                            reason=f"shard wall clock timeout ({shard_timeout_seconds:g}s)",
                            warning_message=(
                                "shard wall clock timeout reached after {0}; "
                                "terminating harbor run.".format(
                                    _format_elapsed_seconds(elapsed_now)
                                )
                            ),
                            kill_warning_message=(
                                "harbor run did not exit after shard timeout terminate(); "
                                "killing process."
                            ),
                            wait_seconds=shard_kill_grace_seconds,
                        )

                latest_status = (
                    _latest_harness_status(combined_log_path)
                    or _fallback_harbor_status(job_dir, combined_log_path)
                    or {}
                )
                phase = termination_phase or str(latest_status.get("stage") or "starting harbor run")
                detail = termination_detail or str(latest_status.get("detail") or "").strip()
                phase_changed = (phase, detail) != (last_phase, last_detail)
                outer_stage, _ = _infer_outer_console_stage(
                    latest_status=latest_status,
                    log_path=combined_log_path,
                    process_running=returncode is None,
                    termination_reason=termination_reason,
                )
                agent_elapsed = latest_status.get("elapsed_s")
                if renderer_enabled:
                    self._update_outer_progress_targets(
                        progress_targets,
                        task_state="running" if returncode is None else "finished",
                        inner_stage=_inner_console_stage(latest_status),
                        outer_stage=outer_stage,
                        elapsed_seconds=elapsed_now,
                        agent_elapsed_seconds=(
                            float(agent_elapsed) if isinstance(agent_elapsed, (int, float)) else None
                        ),
                    )
                if phase_changed and self.console_mode in {"normal", "debug"} and not renderer_enabled:
                    logger.info(
                        "%s | %s | %s%s | %s elapsed",
                        local_name,
                        task_summary,
                        phase,
                        f" ({detail})" if detail else "",
                        _format_elapsed_seconds(elapsed_now),
                )
                if phase_changed:
                    last_phase, last_detail = phase, detail
                heartbeat_due = self._console_heartbeat_enabled() and time.time() >= next_heartbeat
                if phase_changed or heartbeat_due or returncode is not None:
                    self._write_job_status(
                        status_path,
                        iteration=iteration,
                        local_name=local_name,
                        task_names=task_names,
                        log_path=combined_log_path,
                        started_at=started_at,
                        phase=phase,
                        detail=detail or None,
                        returncode=returncode,
                    )
                if returncode is not None:
                    break
                if heartbeat_due:
                    if not phase_changed and not renderer_enabled:
                        logger.info(
                            "%s | %s | %s%s | %s elapsed",
                            local_name,
                            task_summary,
                            phase,
                            f" ({detail})" if detail else "",
                            _format_elapsed_seconds(time.time() - started_at),
                        )
                    next_heartbeat = time.time() + self.console_heartbeat_seconds
                time.sleep(1.0)

        elapsed = time.time() - started_at
        launcher_meta["completed_at"] = datetime.now().isoformat()
        launcher_meta["elapsed_seconds"] = round(elapsed, 1)
        launcher_meta["returncode"] = process.returncode
        if termination_reason:
            launcher_meta["terminated_early"] = True
            launcher_meta["termination_reason"] = termination_reason
        latest_status = (
            _latest_harness_status(combined_log_path)
            or _fallback_harbor_status(job_dir, combined_log_path)
            or {}
        )
        final_status: dict[str, Any] = dict(latest_status) if latest_status else {}
        if termination_phase:
            final_status["stage"] = termination_phase
            if termination_detail:
                final_status["detail"] = termination_detail
            elif "detail" in final_status:
                del final_status["detail"]
        if final_status:
            launcher_meta["latest_status"] = final_status
        _atomic_write_text(launcher_meta_path, json.dumps(launcher_meta, indent=2))
        self._write_job_status(
            status_path,
            iteration=iteration,
            local_name=local_name,
            task_names=task_names,
            log_path=combined_log_path,
            started_at=started_at,
            phase=str(final_status.get("stage") or "harbor run finished"),
            detail=str(final_status.get("detail") or "").strip() or None,
            returncode=process.returncode,
        )
        if renderer_enabled:
            self._update_outer_progress_targets(
                progress_targets,
                task_state="finished" if process.returncode == 0 else "failed",
                inner_stage=_inner_console_stage(final_status),
                outer_stage=str(final_status.get("stage") or "harbor_run_finished"),
                elapsed_seconds=elapsed,
                agent_elapsed_seconds=(
                    float(final_status.get("elapsed_s"))
                    if isinstance(final_status.get("elapsed_s"), (int, float))
                    else None
                ),
            )
            if remove_progress_on_exit:
                self._remove_outer_progress_targets(progress_targets)
        if process.returncode != 0:
            signal_lines = _extract_signal_lines(_read_text_tail(combined_log_path), max_lines=3)
            signal_summary = " | ".join(signal_lines)
            logger.warning(
                "%s failed with exit code %d after %s.%s Log: %s",
                local_name,
                process.returncode,
                _format_elapsed_seconds(elapsed),
                f" {signal_summary}" if signal_summary else "",
                combined_log_path,
            )
        else:
            logger.info(
                "Finished %s in %s. Log: %s",
                local_name,
                _format_elapsed_seconds(elapsed),
                combined_log_path,
            )

        return job_dir

    def _build_daytona_shard_assignments(
        self,
        task_names: list[str],
        job_name: str,
    ) -> list[DaytonaShardAssignment]:
        available_keys = self._daytona_key_pool.available_keys()
        if not available_keys:
            return []
        leased_keys = self._daytona_key_pool.lease_keys(
            min(len(task_names), len(available_keys))
        )
        if not leased_keys:
            return []

        plan = build_daytona_shard_plan(
            task_names,
            available_keys=leased_keys,
            max_tasks_per_shard=self._daytona_cfg.max_tasks_per_shard,
            exclusive_task_ids=set(self._daytona_cfg.exclusive_task_ids),
            assignment_strategy=self._daytona_cfg.assignment_strategy,
        )
        return [
            DaytonaShardAssignment(
                job_name=f"{job_name}-shard-{index:02d}",
                task_names=planned.task_names,
                daytona_key=planned.daytona_key,
            )
            for index, planned in enumerate(plan, start=1)
        ]

    def _run_daytona_sharded_jobs(
        self,
        iteration: int,
        task_names: list[str],
        job_name: str,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
        assignments = self._build_daytona_shard_assignments(task_names, job_name)
        if not assignments:
            return {}, {}

        shard_results: dict[str, dict[str, Any]] = {}
        case_to_key: dict[str, str] = {}
        key_queues: dict[str, deque[DaytonaShardAssignment]] = {}
        key_order: list[str] = []
        for assignment in assignments:
            if assignment.daytona_key not in key_queues:
                key_queues[assignment.daytona_key] = deque()
                key_order.append(assignment.daytona_key)
            key_queues[assignment.daytona_key].append(assignment)

        for daytona_key in self._daytona_key_pool.available_keys():
            if daytona_key in key_queues:
                continue
            key_queues[daytona_key] = deque()
            key_order.append(daytona_key)

        key_indices = {daytona_key: index for index, daytona_key in enumerate(key_order, start=1)}
        key_queue_depths = {
            daytona_key: len(queue)
            for daytona_key, queue in key_queues.items()
            if queue
        }
        base_total_waves = max(key_queue_depths.values(), default=1)
        max_tasks_per_shard = max(1, int(self._daytona_cfg.max_tasks_per_shard))

        task_targets: dict[str, _OuterProgressTarget] = {}
        for daytona_key, queue in key_queues.items():
            queue_snapshot = list(queue)
            total_slots = len(queue_snapshot)
            for slot_index, assignment in enumerate(queue_snapshot, start=1):
                for task_name in assignment.task_names:
                    task_targets[task_name] = _OuterProgressTarget(
                        line_id=task_name,
                        display_name=task_name,
                        wave_index=slot_index,
                        total_waves=max(1, total_slots),
                    )

        progress_renderer = self._get_outer_progress_renderer()
        renderer_enabled = bool(getattr(progress_renderer, "enabled", False))
        if renderer_enabled:
            self._update_outer_progress_targets(
                [task_targets[task_name] for task_name in task_names if task_name in task_targets],
                task_state="queued",
                inner_stage="queued",
                outer_stage="waiting_for_key",
                elapsed_seconds=0.0,
                agent_elapsed_seconds=None,
            )

        if base_total_waves > 1:
            logger.info(
                "%s | Daytona wave queue: %d Harbor shard run(s) over %d key(s) for %d task(s).",
                job_name,
                len(assignments),
                len(key_queue_depths),
                len(task_names),
            )
        else:
            logger.info(
                "%s | Daytona launching %d Harbor shard run(s) across %d key(s) for %d task(s).",
                job_name,
                len(assignments),
                len(key_queue_depths),
                len(task_names),
            )

        synthetic_assignment_count = 0
        wave_index = 0

        def _make_progress_targets(
            assignment: DaytonaShardAssignment,
            *,
            wave_index: int,
            total_waves: int,
        ) -> list[_OuterProgressTarget]:
            return [
                _OuterProgressTarget(
                    line_id=task_name,
                    display_name=task_name,
                    wave_index=wave_index,
                    total_waves=max(1, total_waves),
                )
                for task_name in assignment.task_names
            ]

        def _build_memory_only_assignment(
            *,
            daytona_key: str,
            wave_index: int,
        ) -> DaytonaShardAssignment:
            nonlocal synthetic_assignment_count
            synthetic_assignment_count += 1
            return DaytonaShardAssignment(
                job_name=(
                    f"{job_name}-wave-{wave_index:02d}-key-"
                    f"{key_indices.get(daytona_key, 0):02d}-retry-{synthetic_assignment_count:02d}"
                ),
                task_names=[],
                daytona_key=daytona_key,
            )

        relocation_queue: deque[DaytonaRelocationRequest] = deque()
        relocation_slot_loads: dict[str, int] = {}

        def _backfill_relocation_tasks(
            wave_assignments: list[DaytonaShardAssignment],
        ) -> int:
            inserted = 0
            deferred: deque[DaytonaRelocationRequest] = deque()
            while relocation_queue:
                request = relocation_queue.popleft()
                target_key = choose_relocation_target_key(
                    request,
                    candidates=[
                        (assignment.daytona_key, len(assignment.task_names))
                        for assignment in wave_assignments
                    ],
                    key_priority=key_indices,
                )
                if target_key is None:
                    deferred.append(request)
                    continue
                assignment = next(
                    item for item in wave_assignments if item.daytona_key == target_key
                )
                current_load = len(assignment.task_names)
                assignment.task_names.append(request.task_name)
                case_to_key[request.task_name] = assignment.daytona_key
                relocation_slot_loads[request.task_name] = current_load
                inserted += 1
            relocation_queue.extend(deferred)
            return inserted

        while any(key_queues[daytona_key] for daytona_key in key_order) or relocation_queue:
            wave_index += 1
            wave_assignments: list[DaytonaShardAssignment] = []
            for daytona_key in key_order:
                queue = key_queues.get(daytona_key)
                if queue and queue:
                    wave_assignments.append(queue.popleft())
                else:
                    wave_assignments.append(
                        _build_memory_only_assignment(
                            daytona_key=daytona_key,
                            wave_index=wave_index,
                        )
                    )

            relocation_tasks_inserted = _backfill_relocation_tasks(wave_assignments)
            launched_assignments = [
                assignment for assignment in wave_assignments if assignment.task_names
            ]
            if not launched_assignments:
                if relocation_queue:
                    for request in relocation_queue:
                        detail = shard_results.get(request.task_name)
                        if detail is not None:
                            _append_analysis(
                                detail,
                                "No alternate Daytona key was available for relocation.",
                            )
                break

            total_waves = max(
                base_total_waves,
                wave_index,
                wave_index + (
                    math.ceil(len(relocation_queue) / max(1, len(key_order) * max_tasks_per_shard))
                    if relocation_queue
                    else 0
                ),
            )
            if relocation_tasks_inserted:
                logger.info(
                    "%s | launching Daytona wave %d with %d task(s) across %d key(s); "
                    "relocation backfills=%d.",
                    job_name,
                    wave_index,
                    sum(len(assignment.task_names) for assignment in launched_assignments),
                    len(launched_assignments),
                    relocation_tasks_inserted,
                )
            else:
                logger.info(
                    "%s | launching Daytona wave %d with %d task(s) across %d key(s).",
                    job_name,
                    wave_index,
                    sum(len(assignment.task_names) for assignment in launched_assignments),
                    len(launched_assignments),
                )

            with ThreadPoolExecutor(max_workers=len(launched_assignments)) as executor:
                future_to_assignment: dict[Any, tuple[DaytonaShardAssignment, list[_OuterProgressTarget]]] = {}
                for assignment in launched_assignments:
                    case_to_key.update(
                        {case_id: assignment.daytona_key for case_id in assignment.task_names}
                    )
                    targets = _make_progress_targets(
                        assignment,
                        wave_index=wave_index,
                        total_waves=total_waves,
                    )
                    future = executor.submit(
                        self._run_harbor,
                        iteration,
                        assignment.task_names,
                        job_name=assignment.job_name,
                        daytona_key=assignment.daytona_key,
                        n_concurrent=self._daytona_cfg.per_key_concurrency,
                        progress_targets=targets,
                        remove_progress_on_exit=False,
                    )
                    future_to_assignment[future] = (assignment, targets)

                for future in as_completed(tuple(future_to_assignment)):
                    assignment, targets = future_to_assignment[future]
                    try:
                        job_dir = future.result()
                    except Exception as exc:
                        logger.warning(
                            "%s | %s | harbor run failed: %s",
                            assignment.job_name,
                            _summarize_task_names(assignment.task_names),
                            exc,
                        )
                        if renderer_enabled:
                            self._update_outer_progress_targets(
                                targets,
                                task_state="failed",
                                inner_stage="harbor_run_error",
                                outer_stage="harbor_run_failed",
                                elapsed_seconds=0.0,
                                agent_elapsed_seconds=None,
                            )
                        for task_name in assignment.task_names:
                            shard_results[task_name] = self._harbor_run_error_result(exc)
                        continue

                    raw_results = self._parse_job_results(job_dir) or {}
                    normalized_results = self._normalize_job_results_for_tasks(
                        job_dir,
                        raw_results,
                        assignment.task_names,
                    )
                    shard_results.update(normalized_results)

                    for task_name in assignment.task_names:
                        detail = normalized_results.get(task_name)
                        if detail is None:
                            continue
                        if not self._daytona_cfg.relocate_on_daytona_error:
                            continue

                        error_kind = _daytona_relocation_error_kind(detail)
                        if error_kind is None:
                            relocation_slot_loads.pop(task_name, None)
                            continue

                        previous_slot_load = relocation_slot_loads.get(task_name)
                        if previous_slot_load == 0:
                            _append_analysis(
                                detail,
                                "Daytona infrastructure error persisted after relocation onto an empty key.",
                            )
                            continue

                        allowed_target_loads = (0,) if previous_slot_load == 1 else (0, 1)
                        relocation_queue.append(
                            DaytonaRelocationRequest(
                                task_name=task_name,
                                failed_key=assignment.daytona_key,
                                error_kind=error_kind,
                                allowed_target_loads=allowed_target_loads,
                            )
                        )
                        if previous_slot_load == 1:
                            _append_analysis(
                                detail,
                                "Retry will wait for an empty Daytona key after this task hit another infrastructure error on a 1-task key.",
                            )
                        else:
                            _append_analysis(
                                detail,
                                "Rescheduled for a lighter Daytona key after an infrastructure error.",
                            )
                        if renderer_enabled:
                            self._update_outer_progress_targets(
                                [
                                    _OuterProgressTarget(
                                        line_id=task_name,
                                        display_name=task_name,
                                        wave_index=wave_index + 1,
                                        total_waves=max(total_waves, wave_index + 1),
                                    )
                                ],
                                task_state="queued",
                                inner_stage="queued",
                                outer_stage="waiting_for_wave",
                                elapsed_seconds=0.0,
                                agent_elapsed_seconds=None,
                            )

        logger.info(
            "%s | finished all Daytona shard runs.",
            job_name,
        )

        ordered_results = {
            task_name: shard_results[task_name]
            for task_name in task_names
            if task_name in shard_results
        }
        if renderer_enabled:
            self._remove_outer_progress_targets(
                [task_targets[task_name] for task_name in task_names if task_name in task_targets]
            )
        return ordered_results, case_to_key

    def _daytona_retry_wait_seconds(self) -> int:
        if self._daytona_key_pool.any_available():
            return _DAYTONA_SHORT_RETRY_WAIT_SECONDS
        return max(
            self._daytona_cfg.retry_wait_seconds,
            int(math.ceil(self._daytona_key_pool.soonest_available_in())),
        )

    def _parse_job_results(self, job_dir: Path) -> Optional[dict[str, dict[str, Any]]]:
        result_file = job_dir / "result.json"
        if not result_file.exists():
            partial_results = _parse_partial_job_results(job_dir)
            if partial_results:
                logger.warning(
                    "result.json not found in %s; salvaging %d partial trial result(s).",
                    job_dir,
                    len(partial_results),
                )
                return partial_results
            logger.warning("result.json not found in %s.", job_dir)
            return None
        return _parse_result_json(result_file)

    def _job_termination_reason(self, job_dir: Path) -> str:
        launcher_meta_path = job_dir / _HARBOR_LAUNCHER_META_NAME
        if not launcher_meta_path.exists():
            return ""
        try:
            payload = json.loads(launcher_meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ""
        if not bool(payload.get("terminated_early", False)):
            return ""
        return str(payload.get("termination_reason") or "").strip()

    def _harbor_run_error_result(self, exc: BaseException) -> dict[str, Any]:
        detail = _base_case_result(reward=0.0, reward_observed=False)
        message = re.sub(r"\s+", " ", str(exc)).strip() or exc.__class__.__name__
        detail["exception_type"] = "HarborRunError"
        detail["exception_message"] = str(exc)
        detail["analysis"] = f"Harbor failure: {message}"
        detail["primary_dim"] = "D4"
        detail["external_blocker"] = False
        return detail

    def _normalize_job_results_for_tasks(
        self,
        job_dir: Path,
        raw_results: dict[str, dict[str, Any]],
        task_names: list[str],
    ) -> dict[str, dict[str, Any]]:
        normalized = _normalize_results_for_tasks(raw_results, task_names)
        missing = [task_name for task_name in task_names if task_name not in normalized]
        if not missing:
            return normalized

        termination_reason = self._job_termination_reason(job_dir)
        for task_name in missing:
            detail = _base_case_result(reward=0.0, reward_observed=False)
            exception_type = _exception_type_from_termination_reason(termination_reason)
            detail["exception_type"] = exception_type or "TaskResultMissing"
            detail["exception_message"] = (
                termination_reason
                or "Harbor run finished without a task-level result artifact."
            )
            analysis_reason = termination_reason or "task result artifact missing"
            analysis_prefix = (
                "Verifier failure" if exception_type in _TIMEOUT_EXCEPTION_TYPES else "Harbor failure"
            )
            detail["analysis"] = f"{analysis_prefix}: {analysis_reason}"
            detail["primary_dim"] = "D4"
            detail["external_blocker"] = False
            normalized[task_name] = detail
        return normalized

    def _collect_results_without_retries(
        self,
        iteration: int,
        task_names: list[str],
        *,
        job_name: Optional[str] = None,
    ) -> Optional[dict[str, dict[str, Any]]]:
        local_name = job_name or f"iter-{iteration:02d}"
        if self._daytona_key_pool.enabled:
            results, case_to_key = self._run_daytona_sharded_jobs(iteration, task_names, local_name)
            if not results:
                return None
            if case_to_key:
                key_to_cases: dict[str, set[str]] = {}
                for case_id, daytona_key in case_to_key.items():
                    key_to_cases.setdefault(daytona_key, set()).add(case_id)
                self._cleanup_completed_daytona_keys(
                    key_to_cases,
                    set(),
                    set(case_to_key.values()),
                )
            return results

        job_dir = self._run_harbor(iteration, task_names, job_name=job_name)
        raw_results = self._parse_job_results(job_dir) or {}
        return self._normalize_job_results_for_tasks(job_dir, raw_results, task_names)

    def _collect_results_with_daytona_retries(
        self,
        iteration: int,
        task_names: list[str],
        job_name: Optional[str] = None,
    ) -> Optional[dict[str, dict[str, Any]]]:
        if not self._daytona_cfg.allow_retries:
            logger.info(
                "Retries disabled by config (experiment.daytona.allow_retries=false); "
                "running a single pass without retry rounds.",
            )
            return self._collect_results_without_retries(
                iteration,
                task_names,
                job_name=job_name,
            )

        if self._daytona_key_pool.enabled:
            return self._collect_results_with_multi_key_daytona_retries(
                iteration,
                task_names,
                job_name=job_name,
            )
        return self._collect_results_with_legacy_daytona_retries(
            iteration,
            task_names,
            job_name=job_name,
        )

    def _collect_results_with_legacy_daytona_retries(
        self,
        iteration: int,
        task_names: list[str],
        job_name: Optional[str] = None,
    ) -> Optional[dict[str, dict[str, Any]]]:
        local_name = job_name or f"iter-{iteration:02d}"
        job_dir = self._run_harbor(iteration, task_names, job_name=job_name)
        raw_results = self._parse_job_results(job_dir)
        if raw_results is None:
            return None
        results = _normalize_results_for_tasks(raw_results, task_names)

        pending = [
            case_id
            for case_id in task_names
            if _should_retry_daytona_kind(
                _retryable_daytona_kind(results.get(case_id, {})),
                0,
                disk_limit_retry_limit=self._daytona_cfg.disk_limit_retry_limit,
                connectivity_retry_limit=self._daytona_cfg.connectivity_retry_limit,
            )
        ]
        retry_round = 0
        batch_size = _retry_batch_size(self.n_concurrent)
        disk_limit_retry_counts: dict[str, int] = {}
        connectivity_retry_counts: dict[str, int] = {}
        retry_wait_seconds = self._daytona_cfg.retry_wait_seconds
        disk_limit_retry_limit = self._daytona_cfg.disk_limit_retry_limit
        connectivity_retry_limit = self._daytona_cfg.connectivity_retry_limit
        timeout_retry_limit = self._daytona_cfg.timeout_retry_limit

        while pending:
            retry_round += 1
            logger.info(
                "Detected %d retryable Daytona case(s); waiting %d seconds before retry round %d.",
                len(pending),
                retry_wait_seconds,
                retry_round,
            )
            if self.n_concurrent > 1:
                logger.info(
                    "Retrying Daytona cases in batches of %d (< n_concurrent=%d).",
                    batch_size,
                    self.n_concurrent,
                )
            time.sleep(retry_wait_seconds)

            next_pending: list[str] = []
            for batch_index, batch in enumerate(_chunk_case_ids(pending, batch_size), start=1):
                batch_retry_kinds = {
                    case_id: _retryable_daytona_kind(results.get(case_id, {}))
                    for case_id in batch
                }
                retry_job_name = f"{local_name}-daytona-retry-{retry_round:02d}-{batch_index:02d}"
                retry_dir = self._run_harbor(iteration, batch, job_name=retry_job_name)
                retry_raw_results = self._parse_job_results(retry_dir)
                if retry_raw_results:
                    results.update(_normalize_results_for_tasks(retry_raw_results, batch))
                for case_id in batch:
                    prior_kind = batch_retry_kinds.get(case_id)
                    if prior_kind == "disk_limit":
                        disk_limit_retry_counts[case_id] = (
                            disk_limit_retry_counts.get(case_id, 0) + 1
                        )
                    if prior_kind == "connectivity":
                        connectivity_retry_counts[case_id] = (
                            connectivity_retry_counts.get(case_id, 0) + 1
                        )

                    detail = results.get(case_id)
                    if detail is None:
                        continue

                    kind = _retryable_daytona_kind(detail)
                    if kind == "disk_limit":
                        retries = disk_limit_retry_counts.get(case_id, 0)
                        if _should_retry_daytona_kind(
                            kind,
                            retries,
                            disk_limit_retry_limit=disk_limit_retry_limit,
                            connectivity_retry_limit=connectivity_retry_limit,
                        ):
                            next_pending.append(case_id)
                        else:
                            _append_analysis(
                                detail,
                                "Daytona disk limit persisted after {0} retry rounds.".format(
                                    disk_limit_retry_limit,
                                ),
                            )
                        continue
                    if kind == "connectivity":
                        retries = connectivity_retry_counts.get(case_id, 0)
                        if _should_retry_daytona_kind(
                            kind,
                            retries,
                            disk_limit_retry_limit=disk_limit_retry_limit,
                            connectivity_retry_limit=connectivity_retry_limit,
                        ):
                            next_pending.append(case_id)
                        else:
                            _append_analysis(
                                detail,
                                "Daytona connectivity error persisted after {0} retry rounds.".format(
                                    connectivity_retry_limit,
                                ),
                            )
                        continue
                    if prior_kind == "connectivity":
                        _append_analysis(
                            detail,
                            "Daytona connectivity cleared after retry round {0}/{1}.".format(
                                connectivity_retry_counts.get(case_id, 0),
                                connectivity_retry_limit,
                            ),
                        )
            pending = next_pending

        timeout_pending = [
            case_id
            for case_id in task_names
            if _is_retryable_timeout(results.get(case_id, {}))
        ]
        timeout_retry_counts: dict[str, int] = {}

        for retry_round in range(1, timeout_retry_limit + 1):
            if not timeout_pending:
                break

            logger.info(
                "Detected %d timeout case(s); retry round %d/%d.",
                len(timeout_pending),
                retry_round,
                timeout_retry_limit,
            )
            if self.n_concurrent > 1:
                logger.info(
                    "Retrying timeout cases in batches of %d (< n_concurrent=%d).",
                    batch_size,
                    self.n_concurrent,
                )

            next_timeout_pending: list[str] = []
            for batch_index, batch in enumerate(
                _chunk_case_ids(timeout_pending, batch_size),
                start=1,
            ):
                retry_job_name = f"{local_name}-timeout-retry-{retry_round:02d}-{batch_index:02d}"
                retry_dir = self._run_harbor(iteration, batch, job_name=retry_job_name)
                retry_raw_results = self._parse_job_results(retry_dir)
                if retry_raw_results:
                    results.update(_normalize_results_for_tasks(retry_raw_results, batch))
                for case_id in batch:
                    timeout_retry_counts[case_id] = timeout_retry_counts.get(case_id, 0) + 1
                    detail = results.get(case_id)
                    if detail is None:
                        continue
                    detail["timeout_retry_count"] = timeout_retry_counts[case_id]
                    if _is_retryable_timeout(detail):
                        next_timeout_pending.append(case_id)
                    else:
                        _append_analysis(
                            detail,
                            "Timeout cleared after retry round {0}/{1}.".format(
                                timeout_retry_counts[case_id],
                                timeout_retry_limit,
                            ),
                        )

            timeout_pending = next_timeout_pending

        for case_id in timeout_pending:
            detail = results.get(case_id)
            if detail is None:
                continue
            detail["timeout_retry_count"] = timeout_retry_counts.get(case_id, timeout_retry_limit)
            _append_analysis(
                detail,
                "Timed out after {0} retry rounds.".format(timeout_retry_limit),
            )

        completion_pending = [
            case_id
            for case_id in task_names
            if _needs_completion_retry(results.get(case_id))
        ]
        completion_retry_counts: dict[str, int] = {}

        for retry_round in range(1, timeout_retry_limit + 1):
            if not completion_pending:
                break

            logger.info(
                "Detected %d incomplete case(s) without observed reward; retry round %d/%d.",
                len(completion_pending),
                retry_round,
                timeout_retry_limit,
            )
            if self.n_concurrent > 1:
                logger.info(
                    "Retrying incomplete cases in batches of %d (< n_concurrent=%d).",
                    batch_size,
                    self.n_concurrent,
                )

            next_completion_pending: list[str] = []
            for batch_index, batch in enumerate(
                _chunk_case_ids(completion_pending, batch_size),
                start=1,
            ):
                prior_details = {
                    case_id: dict(results.get(case_id) or {})
                    for case_id in batch
                }
                retry_job_name = f"{local_name}-completion-retry-{retry_round:02d}-{batch_index:02d}"
                retry_dir = self._run_harbor(iteration, batch, job_name=retry_job_name)
                retry_raw_results = self._parse_job_results(retry_dir)
                if retry_raw_results:
                    results.update(_normalize_results_for_tasks(retry_raw_results, batch))
                for case_id in batch:
                    completion_retry_counts[case_id] = (
                        completion_retry_counts.get(case_id, 0) + 1
                    )
                    detail = results.get(case_id)
                    if detail is None:
                        next_completion_pending.append(case_id)
                        continue
                    if str(detail.get("exception_type") or "") in _TIMEOUT_EXCEPTION_TYPES:
                        detail["timeout_retry_count"] = completion_retry_counts[case_id]
                    if _needs_completion_retry(detail):
                        next_completion_pending.append(case_id)
                        continue
                    _append_analysis(
                        detail,
                        _completion_retry_cleared_message(
                            prior_details.get(case_id),
                            completion_retry_counts[case_id],
                            timeout_retry_limit,
                        ),
                    )

            completion_pending = next_completion_pending

        for case_id in completion_pending:
            detail = results.setdefault(case_id, _base_case_result())
            if str(detail.get("exception_type") or "") in _TIMEOUT_EXCEPTION_TYPES:
                detail["timeout_retry_count"] = completion_retry_counts.get(
                    case_id,
                    timeout_retry_limit,
                )
            _append_analysis(
                detail,
                _completion_retry_exhausted_message(detail, timeout_retry_limit),
            )

        return results

    def _collect_results_with_multi_key_daytona_retries(
        self,
        iteration: int,
        task_names: list[str],
        job_name: Optional[str] = None,
    ) -> Optional[dict[str, dict[str, Any]]]:
        local_name = job_name or f"iter-{iteration:02d}"
        results, case_to_key = self._run_daytona_sharded_jobs(iteration, task_names, local_name)
        if not results:
            return None

        key_to_cases: dict[str, set[str]] = {}
        for case_id, daytona_key in case_to_key.items():
            key_to_cases.setdefault(daytona_key, set()).add(case_id)
        dirty_keys = set(case_to_key.values())

        pending = [
            case_id
            for case_id in task_names
            if _should_retry_daytona_kind(
                _retryable_daytona_kind(results.get(case_id, {})),
                0,
                disk_limit_retry_limit=self._daytona_cfg.disk_limit_retry_limit,
                connectivity_retry_limit=self._daytona_cfg.connectivity_retry_limit,
            )
        ]
        retry_round = 0
        disk_limit_retry_counts: dict[str, int] = {}
        connectivity_retry_counts: dict[str, int] = {}
        timeout_retry_counts: dict[str, int] = {}
        completion_retry_counts: dict[str, int] = {}
        cleanup_retry_activity_started = False

        unfinished_cases = self._unfinished_daytona_cases_for_cleanup(
            task_names,
            results,
            disk_limit_retry_counts=disk_limit_retry_counts,
            connectivity_retry_counts=connectivity_retry_counts,
            timeout_retry_counts=timeout_retry_counts,
            completion_retry_counts=completion_retry_counts,
        )
        if unfinished_cases:
            cleanup_retry_activity_started = True
            self._cleanup_completed_daytona_keys(
                key_to_cases,
                unfinished_cases,
                dirty_keys,
            )

        while pending:
            retry_round += 1
            prior_retry_kinds = {
                case_id: _retryable_daytona_kind(results.get(case_id, {}))
                for case_id in pending
            }
            for case_id, kind in prior_retry_kinds.items():
                if kind != "disk_limit":
                    continue
                daytona_key = case_to_key.get(case_id)
                if daytona_key:
                    self._daytona_key_pool.cooldown_key(daytona_key)

            wait_seconds = self._daytona_retry_wait_seconds()
            logger.info(
                "Detected %d retryable Daytona case(s); waiting %d seconds before retry round %d.",
                len(pending),
                wait_seconds,
                retry_round,
            )
            time.sleep(wait_seconds)

            retry_results, retry_case_to_key = self._run_daytona_sharded_jobs(
                iteration,
                pending,
                f"{local_name}-daytona-retry-{retry_round:02d}",
            )
            if retry_results:
                results.update(retry_results)
                results = {
                    task_name: results[task_name]
                    for task_name in task_names
                    if task_name in results
                }
            self._update_daytona_key_case_assignments(
                key_to_cases,
                case_to_key,
                retry_case_to_key,
                dirty_keys,
            )

            next_pending: list[str] = []
            for case_id in pending:
                prior_kind = prior_retry_kinds.get(case_id)
                if prior_kind == "disk_limit":
                    disk_limit_retry_counts[case_id] = (
                        disk_limit_retry_counts.get(case_id, 0) + 1
                    )
                if prior_kind == "connectivity":
                    connectivity_retry_counts[case_id] = (
                        connectivity_retry_counts.get(case_id, 0) + 1
                    )

                detail = results.get(case_id)
                if detail is None:
                    continue

                kind = _retryable_daytona_kind(detail)
                if kind == "disk_limit":
                    retries = disk_limit_retry_counts.get(case_id, 0)
                    if _should_retry_daytona_kind(
                        kind,
                        retries,
                        disk_limit_retry_limit=self._daytona_cfg.disk_limit_retry_limit,
                        connectivity_retry_limit=self._daytona_cfg.connectivity_retry_limit,
                    ):
                        next_pending.append(case_id)
                    else:
                        _append_analysis(
                            detail,
                            "Daytona disk limit persisted after {0} retry rounds.".format(
                                self._daytona_cfg.disk_limit_retry_limit,
                            ),
                        )
                    continue
                if kind == "connectivity":
                    retries = connectivity_retry_counts.get(case_id, 0)
                    if _should_retry_daytona_kind(
                        kind,
                        retries,
                        disk_limit_retry_limit=self._daytona_cfg.disk_limit_retry_limit,
                        connectivity_retry_limit=self._daytona_cfg.connectivity_retry_limit,
                    ):
                        next_pending.append(case_id)
                    else:
                        _append_analysis(
                            detail,
                            "Daytona connectivity error persisted after {0} retry rounds.".format(
                                self._daytona_cfg.connectivity_retry_limit,
                            ),
                        )
                    continue
                if prior_kind == "connectivity":
                    _append_analysis(
                        detail,
                        "Daytona connectivity cleared after retry round {0}/{1}.".format(
                            connectivity_retry_counts.get(case_id, 0),
                            self._daytona_cfg.connectivity_retry_limit,
                        ),
                    )

            unfinished_cases = self._unfinished_daytona_cases_for_cleanup(
                task_names,
                results,
                disk_limit_retry_counts=disk_limit_retry_counts,
                connectivity_retry_counts=connectivity_retry_counts,
                timeout_retry_counts=timeout_retry_counts,
                completion_retry_counts=completion_retry_counts,
            )
            if cleanup_retry_activity_started or unfinished_cases:
                cleanup_retry_activity_started = True
                self._cleanup_completed_daytona_keys(
                    key_to_cases,
                    unfinished_cases,
                    dirty_keys,
                )
            pending = next_pending

        timeout_pending = [
            case_id
            for case_id in task_names
            if _is_retryable_timeout(results.get(case_id, {}))
        ]

        for retry_round in range(1, self._daytona_cfg.timeout_retry_limit + 1):
            if not timeout_pending:
                break

            logger.info(
                "Detected %d timeout case(s); retry round %d/%d.",
                len(timeout_pending),
                retry_round,
                self._daytona_cfg.timeout_retry_limit,
            )

            retry_results, retry_case_to_key = self._run_daytona_sharded_jobs(
                iteration,
                timeout_pending,
                f"{local_name}-timeout-retry-{retry_round:02d}",
            )
            if retry_results:
                results.update(retry_results)
                results = {
                    task_name: results[task_name]
                    for task_name in task_names
                    if task_name in results
                }
            self._update_daytona_key_case_assignments(
                key_to_cases,
                case_to_key,
                retry_case_to_key,
                dirty_keys,
            )

            next_timeout_pending: list[str] = []
            for case_id in timeout_pending:
                timeout_retry_counts[case_id] = timeout_retry_counts.get(case_id, 0) + 1
                detail = results.get(case_id)
                if detail is None:
                    continue
                detail["timeout_retry_count"] = timeout_retry_counts[case_id]
                if _is_retryable_timeout(detail):
                    next_timeout_pending.append(case_id)
                else:
                    _append_analysis(
                        detail,
                        "Timeout cleared after retry round {0}/{1}.".format(
                            timeout_retry_counts[case_id],
                            self._daytona_cfg.timeout_retry_limit,
                        ),
                    )

            unfinished_cases = self._unfinished_daytona_cases_for_cleanup(
                task_names,
                results,
                disk_limit_retry_counts=disk_limit_retry_counts,
                connectivity_retry_counts=connectivity_retry_counts,
                timeout_retry_counts=timeout_retry_counts,
                completion_retry_counts=completion_retry_counts,
            )
            if cleanup_retry_activity_started or unfinished_cases:
                cleanup_retry_activity_started = True
                self._cleanup_completed_daytona_keys(
                    key_to_cases,
                    unfinished_cases,
                    dirty_keys,
                )
            timeout_pending = next_timeout_pending

        for case_id in timeout_pending:
            detail = results.get(case_id)
            if detail is None:
                continue
            detail["timeout_retry_count"] = timeout_retry_counts.get(
                case_id,
                self._daytona_cfg.timeout_retry_limit,
            )
            _append_analysis(
                detail,
                "Timed out after {0} retry rounds.".format(
                    self._daytona_cfg.timeout_retry_limit,
                ),
            )

        completion_pending = [
            case_id
            for case_id in task_names
            if _needs_completion_retry(results.get(case_id))
        ]

        for retry_round in range(1, self._daytona_cfg.timeout_retry_limit + 1):
            if not completion_pending:
                break

            logger.info(
                "Detected %d incomplete case(s) without observed reward; retry round %d/%d.",
                len(completion_pending),
                retry_round,
                self._daytona_cfg.timeout_retry_limit,
            )

            prior_details = {
                case_id: dict(results.get(case_id) or {})
                for case_id in completion_pending
            }
            retry_results, retry_case_to_key = self._run_daytona_sharded_jobs(
                iteration,
                completion_pending,
                f"{local_name}-completion-retry-{retry_round:02d}",
            )
            if retry_results:
                results.update(retry_results)
                results = {
                    task_name: results[task_name]
                    for task_name in task_names
                    if task_name in results
                }
            self._update_daytona_key_case_assignments(
                key_to_cases,
                case_to_key,
                retry_case_to_key,
                dirty_keys,
            )

            next_completion_pending: list[str] = []
            for case_id in completion_pending:
                completion_retry_counts[case_id] = (
                    completion_retry_counts.get(case_id, 0) + 1
                )
                detail = results.get(case_id)
                if detail is None:
                    next_completion_pending.append(case_id)
                    continue
                if str(detail.get("exception_type") or "") in _TIMEOUT_EXCEPTION_TYPES:
                    detail["timeout_retry_count"] = completion_retry_counts[case_id]
                if _needs_completion_retry(detail):
                    next_completion_pending.append(case_id)
                    continue
                _append_analysis(
                    detail,
                    _completion_retry_cleared_message(
                        prior_details.get(case_id),
                        completion_retry_counts[case_id],
                        self._daytona_cfg.timeout_retry_limit,
                    ),
                )

            unfinished_cases = self._unfinished_daytona_cases_for_cleanup(
                task_names,
                results,
                disk_limit_retry_counts=disk_limit_retry_counts,
                connectivity_retry_counts=connectivity_retry_counts,
                timeout_retry_counts=timeout_retry_counts,
                completion_retry_counts=completion_retry_counts,
            )
            if cleanup_retry_activity_started or unfinished_cases:
                cleanup_retry_activity_started = True
                self._cleanup_completed_daytona_keys(
                    key_to_cases,
                    unfinished_cases,
                    dirty_keys,
                )
            completion_pending = next_completion_pending

        for case_id in completion_pending:
            detail = results.setdefault(case_id, _base_case_result())
            if str(detail.get("exception_type") or "") in _TIMEOUT_EXCEPTION_TYPES:
                detail["timeout_retry_count"] = completion_retry_counts.get(
                    case_id,
                    self._daytona_cfg.timeout_retry_limit,
                )
            _append_analysis(
                detail,
                _completion_retry_exhausted_message(
                    detail,
                    self._daytona_cfg.timeout_retry_limit,
                ),
            )

        return results

    # ------------------------------------------------------------------
    # Bank update
    # ------------------------------------------------------------------

    def _update_bank(
        self, results: dict[str, dict[str, Any]], iteration: int
    ) -> None:
        """Add PerCaseEntry records for each case in the current iteration."""
        from memoharness.core.models import make_minimal_config

        # Determine previous config for delta computation
        prev_entries = self._bank.entries
        if prev_entries:
            last = max(prev_entries, key=lambda e: e.iteration)
            prev_config = last.config
        else:
            prev_config = make_minimal_config()

        for case_id, detail in results.items():
            entry = _make_entry(
                case_id=case_id,
                reward=float(detail["reward"]),
                iteration=iteration,
                config=self._current_config,
                prev_config=prev_config,
                primary_dim=str(detail.get("primary_dim") or "D4"),
                analysis=str(detail.get("analysis") or f"Harbor reward={detail['reward']:.2f}."),
                total_tokens=int(detail.get("total_tokens") or 0),
                num_llm_calls=int(detail.get("num_llm_calls") or 0),
                latency_ms=int(detail.get("latency_ms") or 0),
                tools_invoked=list(detail.get("tools_invoked") or []),
                intermediate_outputs=list(detail.get("intermediate_outputs") or []),
                final_output=str(detail.get("final_output") or ""),
            )
            distill_on_this_entry = self._bank.add_entry(entry)
            if distill_on_this_entry:
                logger.info(
                    "Consecutive-failure trigger: case '%s' hit %d consecutive failures.",
                    case_id,
                    self.min_consecutive_failures,
                )
                self._distill(iteration)
                self._bank.mark_distill_done()

        logger.info("Bank now has %d entries.", len(self._bank.entries))

    # ------------------------------------------------------------------
    # Distillation
    # ------------------------------------------------------------------

    def _distill(self, iteration: int) -> None:
        if self._distiller is not None:
            patterns = self._bank.distill_global_patterns_llm(
                distiller=self._distiller,
                current_iteration=iteration,
            )
        else:
            patterns = self._bank.distill_global_patterns(current_iteration=iteration)

        logger.info(
            "Distilled %d global pattern(s) at iteration %d.",
            len(patterns), iteration,
        )
        for p in patterns:
            logger.info("  [%s] %s", p.pattern_id, p.description[:80])
        if self._is_harbor_codex_mode() and self._current_config is not None:
            try:
                refresh_codex_bundle_support_docs(
                    self._resolve_codex_bundle_root(),
                    self._current_config,
                    distilled_patterns=self._bank.global_patterns,
                    updated_iteration=iteration,
                )
                self._current_harness_code, self._current_config = load_codex_bundle(
                    self._resolve_codex_bundle_root(),
                    fallback=self._current_config,
                )
                logger.info(
                    "Refreshed Harbor Codex bundle support docs from %d distilled pattern(s).",
                    len(self._bank.global_patterns),
                )
            except Exception:
                logger.warning(
                    "Could not refresh Harbor Codex bundle support docs after distillation.",
                    exc_info=True,
                )

    def _canary_task_subset(self) -> list[str]:
        if not self.controller_canary_enabled:
            return []
        if self.controller_canary_task_count <= 0:
            return []
        if not self._train_tasks:
            return []
        return list(self._train_tasks[: self.controller_canary_task_count])

    @staticmethod
    def _result_metrics(results: dict[str, dict[str, Any]]) -> dict[str, float]:
        learning = _filter_learning_results(results)
        cases_total = len(results)
        cases_learning = len(learning)
        cases_succeeded = sum(1 for detail in learning.values() if float(detail.get("reward", 0.0)) >= 0.5)
        mean_reward = (
            sum(float(detail.get("reward", 0.0)) for detail in learning.values())
            / max(cases_learning, 1)
        )
        success_rate = cases_succeeded / max(cases_learning, 1)
        blockers = max(0, cases_total - cases_learning)
        return {
            "cases_total": float(cases_total),
            "cases_learning": float(cases_learning),
            "cases_succeeded": float(cases_succeeded),
            "mean_reward": float(mean_reward),
            "success_rate": float(success_rate),
            "blockers": float(blockers),
        }

    def _baseline_canary_metrics(
        self,
        *,
        task_subset: list[str],
        results_for_iteration: dict[str, dict[str, Any]] | None,
        mean_reward: float,
        cases_run: int,
        cases_succeeded: int,
    ) -> dict[str, float]:
        if results_for_iteration:
            subset_results = {
                task_name: results_for_iteration[task_name]
                for task_name in task_subset
                if task_name in results_for_iteration
            }
            if subset_results:
                return self._result_metrics(subset_results)

        fallback_success_rate = (cases_succeeded / max(cases_run, 1)) if cases_run > 0 else 0.0
        return {
            "cases_total": float(cases_run),
            "cases_learning": float(cases_run),
            "cases_succeeded": float(cases_succeeded),
            "mean_reward": float(mean_reward),
            "success_rate": float(fallback_success_rate),
            "blockers": 0.0,
        }

    def _evaluate_candidate_with_canary(
        self,
        *,
        iteration: int,
        task_subset: list[str],
        baseline_metrics: dict[str, float],
    ) -> tuple[bool, dict[str, float], str]:
        job_name = f"iter-{iteration:02d}-canary"
        results = self._collect_results_with_daytona_retries(
            iteration,
            task_subset,
            job_name=job_name,
        )
        if results is None:
            return False, {}, "canary run did not produce results"

        metrics = self._result_metrics(results)
        if metrics["cases_learning"] <= 0:
            return False, metrics, "no learnable canary cases (all external blockers)"

        reward_delta = metrics["mean_reward"] - baseline_metrics["mean_reward"]
        blocker_increase = metrics["blockers"] - baseline_metrics["blockers"]
        success_delta = metrics["success_rate"] - baseline_metrics["success_rate"]
        if reward_delta < self.controller_canary_min_reward_delta:
            return (
                False,
                metrics,
                "reward delta {0:.3f} < threshold {1:.3f}".format(
                    reward_delta,
                    self.controller_canary_min_reward_delta,
                ),
            )
        if blocker_increase > self.controller_canary_max_blocker_increase:
            return (
                False,
                metrics,
                "blocker increase {0:.0f} > threshold {1}".format(
                    blocker_increase,
                    self.controller_canary_max_blocker_increase,
                ),
            )
        if success_delta < -0.2 and reward_delta <= 0.0:
            return (
                False,
                metrics,
                "success rate dropped too much ({0:.3f}) without reward gain".format(success_delta),
            )

        return True, metrics, "accepted by canary gate"

    @staticmethod
    def _count_perfect_successes(results: dict[str, dict[str, Any]]) -> int:
        return sum(
            1
            for detail in results.values()
            if math.isclose(float(detail.get("reward", 0.0)), 1.0, abs_tol=1e-9)
        )

    # ------------------------------------------------------------------
    # Config update
    # ------------------------------------------------------------------

    def _update_config(
        self,
        iteration: int,
        mean_reward: float = 0.0,
        total_tokens: int = 0,
        cases_run: int = 0,
        cases_succeeded: int = 0,
        perfect_success_count: int = 0,
        results_for_iteration: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Ask the controller for the next HarnessImpl, write it to disk, and archive it."""
        if not self._current_harness_code:
            logger.warning("No harness code available – skipping update.")
            return

        previous_code = self._current_harness_code
        previous_config = self._current_config
        harness_py, harness_json = self._resolve_harness_paths()
        live_snapshot = self._snapshot_live_harness_assets() if self._is_harbor_codex_mode() else None

        try:
            candidate_code, candidate_config = self._controller.decide_next_harness(
                bank=self._bank,
                current_code=previous_code,
                current_config=previous_config,
                iteration=iteration,
                min_consecutive_failures=self.min_consecutive_failures,
            )
        except Exception as exc:
            logger.warning("Controller update failed at iteration %d: %s", iteration, exc)
            return

        # Validate generated code before writing – normalize if legacy/broken
        if hasattr(self._controller, "should_normalize_harness"):
            if self._controller.should_normalize_harness(candidate_code):
                logger.warning(
                    "Generated controller update failed validation – normalizing via template."
                )
                candidate_code = self._controller.normalize_harness_code(
                    candidate_code,
                    candidate_config,
                )

        selected_code = candidate_code
        selected_config = candidate_config
        selected_note = "selected controller update (canary disabled)"

        canary_tasks = self._canary_task_subset()
        if canary_tasks:
            baseline_metrics = self._baseline_canary_metrics(
                task_subset=canary_tasks,
                results_for_iteration=results_for_iteration,
                mean_reward=mean_reward,
                cases_run=cases_run,
                cases_succeeded=cases_succeeded,
            )
            logger.info(
                "Canary gate enabled: tasks=%s baseline_mean=%.3f baseline_success=%.3f "
                "min_reward_delta=%.3f max_blocker_increase=%d",
                canary_tasks,
                baseline_metrics["mean_reward"],
                baseline_metrics["success_rate"],
                self.controller_canary_min_reward_delta,
                self.controller_canary_max_blocker_increase,
            )
            if not self._is_harbor_codex_mode():
                _atomic_write_text(harness_py, candidate_code)
                self._write_harness_summary(harness_json, candidate_config)
            accepted, metrics, reason = self._evaluate_candidate_with_canary(
                iteration=iteration,
                task_subset=canary_tasks,
                baseline_metrics=baseline_metrics,
            )
            logger.info(
                "Canary check: accepted=%s reason=%s mean_reward=%s success_rate=%s blockers=%s",
                accepted,
                reason,
                "{0:.3f}".format(metrics.get("mean_reward", float("nan"))) if metrics else "n/a",
                "{0:.3f}".format(metrics.get("success_rate", float("nan"))) if metrics else "n/a",
                "{0:.0f}".format(metrics.get("blockers", float("nan"))) if metrics else "n/a",
            )
            if accepted:
                selected_note = "selected controller update (mean_reward={0:.3f}, success_rate={1:.3f})".format(
                    metrics["mean_reward"],
                    metrics["success_rate"],
                )
            else:
                if self._is_harbor_codex_mode() and live_snapshot is not None:
                    self._restore_live_harness_assets(live_snapshot)
                selected_code = previous_code
                selected_config = previous_config
                selected_note = "controller update rejected by canary gate"
        elif self.controller_canary_enabled:
            logger.info("Canary gate enabled but no canary tasks available; skipping gate.")
            selected_note = "selected controller update (canary skipped)"

        self._current_harness_code = selected_code
        self._current_config = selected_config
        logger.info("HarnessImpl update decision: %s", selected_note)

        if self._is_harbor_codex_mode():
            if selected_code == previous_code and live_snapshot is not None:
                self._restore_live_harness_assets(live_snapshot)
            self._current_harness_code, self._current_config = load_codex_bundle(
                self._resolve_codex_bundle_root(),
                fallback=selected_config,
            )
            logger.info("Updated Harbor Codex bundle -> %s", self._resolve_codex_bundle_root())
        else:
            _atomic_write_text(harness_py, selected_code)
            logger.info("Wrote updated HarnessImpl -> %s", harness_py)
            self._write_harness_summary(harness_json, selected_config)
            logger.info("Wrote live harness summary -> %s", harness_json)

        archive_meta = {
            "iteration": iteration,
            "mean_reward": round(mean_reward, 4),
            "total_tokens": total_tokens,
            "cases_run": cases_run,
            "cases_succeeded": cases_succeeded,
            "perfect_success_count": int(perfect_success_count),
            "config": selected_config.as_dict(),
            "controller_canary_enabled": self.controller_canary_enabled,
            "controller_canary_task_count": len(canary_tasks),
            "controller_selection_note": selected_note,
        }
        archive_meta_json = json.dumps(archive_meta, indent=2)
        self._archive_live_harness(
            iteration=iteration,
            preview=self._current_harness_code,
            archive_meta_json=archive_meta_json,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _select_best_harness_legacy(self, iter_rewards: dict[int, float]) -> None:
        """Copy the best-performing iteration's archived HarnessImpl (.py) to the live path."""
        if not iter_rewards:
            return

        best_iter = max(iter_rewards, key=lambda k: iter_rewards[k])
        if not self._restore_archived_harness(
            iteration=best_iter,
            log_prefix="Best harness",
            metric_name="mean_reward",
            metric_value=iter_rewards[best_iter],
        ):
            return

        # Persist a summary alongside the bank for inspection
        archive_dir = self._resolved_bank_path.parent / "harness"
        summary_path = archive_dir / "best.json"
        _atomic_write_text(summary_path,
            json.dumps(
                {
                    "best_iteration": best_iter,
                    "mean_reward": iter_rewards[best_iter],
                    "all_iter_rewards": iter_rewards,
                },
                indent=2,
            )
        )
        logger.info("Reward summary written → %s", summary_path)

    @staticmethod
    def _format_best_metric_value(metric_value: Any) -> str:
        if isinstance(metric_value, float):
            return f"{metric_value:.3f}"
        return str(metric_value)

    @staticmethod
    def _selection_job_suffix(mode: str) -> str:
        suffix = re.sub(r"[^a-z0-9]+", "-", str(mode or "").lower()).strip("-")
        return suffix or "selection"

    def _build_best_harness_selections(
        self,
        *,
        iter_rewards: dict[int, float],
        iter_perfect_success_counts: dict[int, int],
        iter_total_tokens: Optional[dict[int, int]] = None,
    ) -> dict[str, dict[str, Any]]:
        metric_maps: dict[str, dict[int, Any]] = {
            _BEST_HARNESS_MODE_MEAN_REWARD: iter_rewards,
            _BEST_HARNESS_MODE_PERFECT_SUCCESS_COUNT: iter_perfect_success_counts,
        }
        selections: dict[str, dict[str, Any]] = {}
        for mode in self.best_harness_selection_modes:
            metric_values = metric_maps.get(mode) or {}
            if not metric_values:
                continue
            token_map = iter_total_tokens or {}
            if mode == _BEST_HARNESS_MODE_MEAN_REWARD:
                # Primary: mean reward. Tie-break: lower token usage wins.
                best_iter = max(
                    metric_values,
                    key=lambda k: (
                        float(metric_values[k]),
                        -int(token_map.get(k, 0) or 0),
                    ),
                )
            else:
                # Primary: perfect_success_count. Tie-break: lower token usage wins.
                best_iter = max(
                    metric_values,
                    key=lambda k: (
                        int(metric_values[k]),
                        -int(token_map.get(k, 0) or 0),
                    ),
                )
            selections[mode] = {
                "mode": mode,
                "iteration": int(best_iter),
                "metric_name": mode,
                "metric_value": metric_values[best_iter],
                "tokens": int((iter_total_tokens or {}).get(best_iter, 0) or 0),
            }
        return selections

    def _best_harness_summary_path(self) -> Path:
        return self._resolved_bank_path.parent / "harness" / "best.json"

    def _apply_best_harness_selection(
        self,
        selection: dict[str, Any],
        *,
        log_prefix: str,
    ) -> bool:
        iteration = int(selection.get("iteration") or 0)
        mode = str(selection.get("mode") or "best")
        metric_name = str(selection.get("metric_name") or mode)
        metric_value = selection.get("metric_value")
        return self._restore_archived_harness(
            iteration=iteration,
            log_prefix=f"{log_prefix} [{mode}]",
            metric_name=metric_name,
            metric_value=metric_value,
        )

    def _load_best_harness_selections(self) -> dict[str, dict[str, Any]]:
        summary_path = self._best_harness_summary_path()
        if not summary_path.exists():
            return {}
        try:
            payload = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            logger.warning("Best harness summary %s was not valid JSON.", summary_path)
            return {}

        selections: dict[str, dict[str, Any]] = {}
        raw_best_by_mode = payload.get("best_by_mode")
        if isinstance(raw_best_by_mode, dict):
            for mode in self.best_harness_selection_modes:
                raw_selection = raw_best_by_mode.get(mode)
                if not isinstance(raw_selection, dict):
                    continue
                iteration = raw_selection.get("iteration")
                if iteration is None:
                    continue
                selections[mode] = {
                    "mode": str(raw_selection.get("mode") or mode),
                    "iteration": int(iteration),
                    "metric_name": str(raw_selection.get("metric_name") or mode),
                    "metric_value": raw_selection.get("metric_value"),
                    "tokens": int(raw_selection.get("tokens") or 0),
                }
        if selections:
            return selections

        legacy_best_iter = payload.get("best_iteration")
        if (
            legacy_best_iter is not None
            and _BEST_HARNESS_MODE_MEAN_REWARD in self.best_harness_selection_modes
        ):
            return {
                _BEST_HARNESS_MODE_MEAN_REWARD: {
                    "mode": _BEST_HARNESS_MODE_MEAN_REWARD,
                    "iteration": int(legacy_best_iter),
                    "metric_name": _BEST_HARNESS_MODE_MEAN_REWARD,
                    "metric_value": payload.get("mean_reward"),
                    "tokens": int(payload.get("total_tokens") or 0),
                }
            }
        return {}

    def _evaluate_selected_best_harnesses(
        self,
        selections: dict[str, dict[str, Any]],
        *,
        label_prefix: str,
    ) -> None:
        ordered_modes = [mode for mode in self.best_harness_selection_modes if mode in selections]
        if not ordered_modes:
            return

        primary_mode = ordered_modes[0]
        for mode in ordered_modes:
            selection = selections[mode]
            if not self._apply_best_harness_selection(
                selection,
                log_prefix="Loaded held-out evaluation harness",
            ):
                logger.warning(
                    "Skipping held-out evaluation for best harness mode '%s'.",
                    mode,
                )
                continue
            self._run_test_evaluation(
                label=f"{label_prefix} [{mode}]",
                job_name=f"eval-test-{self._selection_job_suffix(mode)}",
            )

        self._apply_best_harness_selection(
            selections[primary_mode],
            log_prefix="Restored primary best harness",
        )

    def _select_best_harness(
        self,
        iter_rewards: dict[int, float],
        iter_perfect_success_counts: dict[int, int],
        iter_total_tokens: Optional[dict[int, int]] = None,
    ) -> dict[str, dict[str, Any]]:
        """Select configured best harnesses, restore the primary one, and persist summary."""
        selections = self._build_best_harness_selections(
            iter_rewards=iter_rewards,
            iter_perfect_success_counts=iter_perfect_success_counts,
            iter_total_tokens=iter_total_tokens,
        )
        if not selections:
            return {}

        primary_mode = next(
            (mode for mode in self.best_harness_selection_modes if mode in selections),
            None,
        )
        primary_selection = selections.get(primary_mode) if primary_mode is not None else None
        if primary_selection is not None:
            self._apply_best_harness_selection(
                primary_selection,
                log_prefix="Selected primary best harness",
            )

        legacy_mean_reward_selection = selections.get(_BEST_HARNESS_MODE_MEAN_REWARD)
        summary_path = self._best_harness_summary_path()
        _atomic_write_text(
            summary_path,
            json.dumps(
                {
                    "selected_modes": list(self.best_harness_selection_modes),
                    "primary_mode": primary_mode,
                    "best_iteration": (
                        legacy_mean_reward_selection["iteration"]
                        if legacy_mean_reward_selection is not None
                        else (primary_selection["iteration"] if primary_selection is not None else None)
                    ),
                    "mean_reward": (
                        legacy_mean_reward_selection["metric_value"]
                        if legacy_mean_reward_selection is not None
                        else None
                    ),
                    "best_by_mode": selections,
                    "all_iter_rewards": iter_rewards,
                    "all_iter_perfect_success_counts": iter_perfect_success_counts,
                    "all_iter_total_tokens": iter_total_tokens or {},
                },
                indent=2,
            ),
        )
        logger.info("Best harness summary written -> %s", summary_path)
        return selections

    def _save_bank(self) -> None:
        self._bank.save(self._resolved_bank_path)
        logger.debug("ExperienceBank saved → %s", self._resolved_bank_path)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        description="Run iterative Harbor training with MemoHarness ExperienceBank feedback.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset", "-d",
        default=None,
        help="Harbor dataset spec. Defaults to experiment.dataset in the config file.",
    )
    p.add_argument(
        "--agent-import-path",
        default=None,
        help="Harbor --agent-import-path value. Defaults to experiment.agent_import_path.",
    )
    p.add_argument(
        "--run-id",
        default=None,
        dest="run_id",
        help="Optional stable run identifier. Defaults to experiment.run_id; otherwise generated from dataset + timestamp.",
    )
    p.add_argument(
        "--iterations", "-n",
        type=int, default=None,
        help="Number of training iterations. Defaults to experiment.iterations.",
    )
    p.add_argument(
        "--config",
        default="configs/experiment.json",
        help="Path to the unified MemoHarness runtime config JSON.",
    )
    p.add_argument(
        "--harness-path",
        default=None,
        dest="harness_path",
        help="Path to the live HarnessImpl .py file (read + overwritten each iteration). "
             "Defaults to experiment.harness_path. Companion .json with dimension summary is written alongside.",
    )
    p.add_argument(
        "--bank-dir",
        default=None,
        help="Directory under which per-run bank.pkl files are stored. "
             "Defaults to experiment.bank_dir.",
    )
    p.add_argument(
        "--jobs-dir",
        default=None,
        help="Root directory for Harbor job outputs. "
             "Defaults to experiment.jobs_dir, or jobs/<dataset-name>/ if omitted there too.",
    )
    p.add_argument(
        "--distill-every",
        type=int, default=None,
        help="Trigger distillation after every N new entries are added to the bank. "
             "Defaults to experiment.distill_every.",
    )
    p.add_argument(
        "--min-consecutive-failures",
        type=int, default=None,
        dest="min_consecutive_failures",
        help="Trigger distillation immediately when any case reaches this many consecutive failures. "
             "Defaults to experiment.min_consecutive_failures.",
    )
    p.add_argument(
        "--train-split",
        type=float, default=None,
        dest="train_split",
        help="Fraction of dataset tasks to use for training. Defaults to experiment.train_split. "
             "1.0 = train only (skip eval), 0.0 = eval only",
    )
    p.add_argument(
        "--train-task-limit",
        type=int, default=None,
        dest="train_task_limit",
        help="Maximum number of training tasks to keep after the train/test split. "
             "Defaults to experiment.train_task_limit.",
    )
    p.add_argument(
        "--seed", "-s",
        type=int, default=None,
        help="Random seed for the train/test split. Defaults to experiment.seed.",
    )
    p.add_argument(
        "--n-concurrent",
        type=int, default=None,
        dest="n_concurrent",
        help="Number of Harbor trials to run in parallel. Defaults to experiment.n_concurrent.",
    )
    p.add_argument(
        "--eval-only",
        action="store_true",
        default=None,
        help="Skip training: load the existing bank + split, run eval on test set only, then exit. "
             "Defaults to experiment.eval_only.",
    )
    return p


def main(argv=None) -> None:
    parser = _build_arg_parser()
    args, extra = parser.parse_known_args(argv)
    from memoharness.config.runtime import MemoHarnessRuntimeConfig

    runtime = MemoHarnessRuntimeConfig.from_json_file(Path(args.config))
    log_level = logging.DEBUG if runtime.experiment.console_mode == "debug" else logging.INFO
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    else:
        root_logger.setLevel(log_level)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        for handler in root_logger.handlers:
            handler.setLevel(log_level)
            if handler.formatter is None:
                handler.setFormatter(formatter)
    experiment = runtime.experiment

    loop = HarborTrainingLoop(
        dataset=args.dataset or experiment.dataset,
        agent_import_path=args.agent_import_path or experiment.agent_import_path,
        run_id=args.run_id or experiment.run_id,
        config_path=args.config,
        harness_config_path=args.harness_path or experiment.harness_path,
        bank_dir=args.bank_dir or experiment.bank_dir,
        jobs_dir=args.jobs_dir if args.jobs_dir is not None else experiment.jobs_dir,
        distill_every=(
            args.distill_every
            if args.distill_every is not None
            else experiment.distill_every
        ),
        min_consecutive_failures=(
            args.min_consecutive_failures
            if args.min_consecutive_failures is not None
            else experiment.min_consecutive_failures
        ),
        train_split=args.train_split if args.train_split is not None else experiment.train_split,
        train_task_limit=(
            args.train_task_limit
            if args.train_task_limit is not None
            else experiment.train_task_limit
        ),
        seed=args.seed if args.seed is not None else experiment.seed,
        n_concurrent=(
            args.n_concurrent
            if args.n_concurrent is not None
            else experiment.n_concurrent
        ),
        console_mode=experiment.console_mode,
        console_heartbeat_seconds=experiment.console_heartbeat_seconds,
        harbor_agent_timeout_seconds=experiment.harbor_agent_timeout_seconds,
        verifier_timeout_seconds=experiment.verifier_timeout_seconds,
        disable_harbor_verifier_retry=experiment.disable_harbor_verifier_retry,
        extra_harbor_args=[*experiment.extra_harbor_args, *extra],
        daytona_config=experiment.daytona,
    )

    eval_only = args.eval_only if args.eval_only is not None else experiment.eval_only
    eval_after_train = experiment.eval_after_train
    iterations = args.iterations if args.iterations is not None else experiment.iterations

    if eval_only:
        loop.run_eval_only()
    else:
        loop.run(iterations, eval_after_train=eval_after_train)


if __name__ == "__main__":
    main(sys.argv[1:])
