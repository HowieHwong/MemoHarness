from __future__ import annotations

import inspect
import json
import logging
import os
from typing import Optional

_AGENT_TIMEOUT_ENV = "MEMOHARNESS_HARBOR_AGENT_TIMEOUT_SEC"
_DISABLE_VERIFIER_RETRY_ENV = "MEMOHARNESS_DISABLE_HARBOR_VERIFIER_RETRY"
_VERIFIER_TIMEOUT_ENV = "MEMOHARNESS_HARBOR_VERIFIER_TIMEOUT_SEC"
_VERIFIER_ENV_OVERRIDES_ENV = "MEMOHARNESS_VERIFIER_ENV_OVERRIDES"
_ENABLE_PATCH_ENV = "MEMOHARNESS_ENABLE_HARBOR_RUNTIME_PATCH"
_DEADLINE_NOTICE_ENV = "MEMOHARNESS_AGENT_DEADLINE_NOTICE"
_DEADLINE_RESERVE_ENV = "MEMOHARNESS_AGENT_DEADLINE_RESERVE_SEC"

_PATCH_APPLIED = False
_logger = logging.getLogger(__name__)


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str) -> Optional[float]:
    raw = os.getenv(name)
    if raw is None:
        return None
    value = float(raw.strip())
    if value <= 0:
        raise ValueError(f"{name} must be positive when set")
    return value


def _parse_verifier_env_overrides() -> dict[str, str]:
    """Read ``MEMOHARNESS_VERIFIER_ENV_OVERRIDES`` (JSON object) into a dict.

    Invalid JSON or non-mapping payloads are logged and ignored so a bad
    value can never brick the verifier step.
    """
    raw = os.getenv(_VERIFIER_ENV_OVERRIDES_ENV)
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        _logger.warning(
            "Ignoring %s: invalid JSON (%s).", _VERIFIER_ENV_OVERRIDES_ENV, exc
        )
        return {}
    if not isinstance(payload, dict):
        _logger.warning(
            "Ignoring %s: expected JSON object, got %s.",
            _VERIFIER_ENV_OVERRIDES_ENV,
            type(payload).__name__,
        )
        return {}
    return {str(k): str(v) for k, v in payload.items() if v is not None}



_DEADLINE_TEMPLATE = """

---
## Wall-clock budget (harness-enforced, read this)

Your agent session is killed automatically after {total:g} seconds of execution.
If you are killed mid-work you score **zero**, even when the required files are
already correct -- nothing is graded unless this session ends on its own.

Run `date +%s` right now and remember it as T0.
Re-check `date +%s` roughly every 10 tool calls.

At T0+{soft:g}s ({reserve:g}s before the kill) STOP exploring and STOP opening new
hypotheses, whatever state you are in, and spend the remaining time on:

1. every artifact the task asks for exists at the exact required path, non-empty;
2. anything required to stay running is started detached (`setsid` / `nohup ... &`)
   so it survives this session, then confirmed answering from a *fresh* shell;
3. build artifacts, binaries and temp files you created that the task did not ask
   for are deleted -- some tasks assert on the exact directory contents;
4. a short final summary.

A tidy, partially-complete final state scores far better than being killed in the
middle of an edit.
---
"""


def _deadline_reserve_sec(total_sec: float) -> float:
    configured = _env_float(_DEADLINE_RESERVE_ENV)
    if configured is not None:
        return min(configured, max(0.0, total_sec - 60.0))
    # ~12% of the budget, clamped to a sane band.
    return min(max(total_sec * 0.12, 120.0), 420.0, max(0.0, total_sec - 60.0))


def _with_deadline_notice(instruction: str, timeout_sec) -> str:
    try:
        total = float(timeout_sec)
    except (TypeError, ValueError):
        return instruction
    if total <= 60.0:
        return instruction
    reserve = _deadline_reserve_sec(total)
    if reserve <= 0:
        return instruction
    return str(instruction) + _DEADLINE_TEMPLATE.format(
        total=total, soft=total - reserve, reserve=reserve
    )


def _copy_callable_metadata(wrapper, target, *, default_name: str) -> None:
    wrapper.__name__ = getattr(target, "__name__", default_name)
    wrapper.__qualname__ = getattr(target, "__qualname__", wrapper.__qualname__)
    wrapper.__doc__ = getattr(target, "__doc__", None)


def _apply_timeout_override(
    trial,
    *,
    timeout_attr: str,
    timeout_sec: Optional[float],
    config_attr: str,
) -> None:
    if timeout_sec is None:
        return
    setattr(trial, timeout_attr, timeout_sec)
    trial_config = getattr(trial, "config", None)
    config_section = getattr(trial_config, config_attr, None)
    if config_section is None:
        return
    if hasattr(config_section, "override_timeout_sec"):
        setattr(config_section, "override_timeout_sec", timeout_sec)
    if hasattr(config_section, "max_timeout_sec"):
        setattr(config_section, "max_timeout_sec", timeout_sec)


def apply_runtime_patch() -> None:
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    if not _env_flag(_ENABLE_PATCH_ENV, default=True):
        _logger.info("Harbor runtime patch disabled via %s.", _ENABLE_PATCH_ENV)
        _PATCH_APPLIED = True
        return

    agent_timeout_sec = _env_float(_AGENT_TIMEOUT_ENV)
    disable_verifier_retry = _env_flag(_DISABLE_VERIFIER_RETRY_ENV, default=False)
    verifier_timeout_sec = _env_float(_VERIFIER_TIMEOUT_ENV)
    verifier_env_overrides = _parse_verifier_env_overrides()
    deadline_notice = _env_flag(_DEADLINE_NOTICE_ENV, default=True)
    if (
        agent_timeout_sec is None
        and verifier_timeout_sec is None
        and not disable_verifier_retry
        and not verifier_env_overrides
        and not deadline_notice
    ):
        _PATCH_APPLIED = True
        return

    import harbor.trial.trial as harbor_trial

    trial_cls = getattr(harbor_trial, "Trial", None)
    if trial_cls is None:
        _PATCH_APPLIED = True
        return

    original_execute_agent = getattr(trial_cls, "_execute_agent", None)
    if callable(original_execute_agent) and agent_timeout_sec is not None:
        target = inspect.unwrap(original_execute_agent)

        async def _patched_execute_agent(self, *args, _target=target, **kwargs):
            _apply_timeout_override(
                self,
                timeout_attr="_agent_timeout_sec",
                timeout_sec=agent_timeout_sec,
                config_attr="agent",
            )
            return await _target(self, *args, **kwargs)

        _copy_callable_metadata(
            _patched_execute_agent,
            target,
            default_name="_execute_agent",
        )
        setattr(trial_cls, "_execute_agent", _patched_execute_agent)

    # Tell the agent how much wall clock it has. Harbor only forwards
    # ``agent_timeout_sec`` to the ORACLE agent, so a normal agent explores as
    # if time were unbounded and gets killed mid-edit with nothing tidied up.
    original_agent_phase = getattr(trial_cls, "_run_agent_phase", None)
    if callable(original_agent_phase) and deadline_notice:
        target = inspect.unwrap(original_agent_phase)

        async def _patched_run_agent_phase(self, *args, _target=target, **kwargs):
            timeout_sec = kwargs.get("timeout_sec")
            instruction = kwargs.get("instruction")
            if timeout_sec is not None and isinstance(instruction, str):
                try:
                    kwargs["instruction"] = _with_deadline_notice(instruction, timeout_sec)
                except Exception:
                    _logger.exception(
                        "Failed to add the deadline notice; running with the raw instruction."
                    )
            return await _target(self, *args, **kwargs)

        _copy_callable_metadata(
            _patched_run_agent_phase,
            target,
            default_name="_run_agent_phase",
        )
        setattr(trial_cls, "_run_agent_phase", _patched_run_agent_phase)

    original_verify = getattr(trial_cls, "_verify_with_retry", None)
    if callable(original_verify) and (
        disable_verifier_retry
        or verifier_timeout_sec is not None
        or verifier_env_overrides
    ):
        target = inspect.unwrap(original_verify) if disable_verifier_retry else original_verify

        async def _patched_verify_with_retry(self, *args, _target=target, **kwargs):
            _apply_timeout_override(
                self,
                timeout_attr="_verifier_timeout_sec",
                timeout_sec=verifier_timeout_sec,
                config_attr="verifier",
            )

            # Harbor 0.3.0's Verifier has no override_env parameter. The
            # task's [verifier.env] ${VAR} templates are resolved via
            # harbor.utils.env.resolve_env_vars at verify time, reading
            # os.environ. We can't swap os.environ safely because concurrent
            # trials (n_concurrent > 1) share a single process and would see
            # each other's mutations during agent execution.
            #
            # Instead, resolve the templates right here (synchronously, no
            # yield point) using the current os.environ, overlay our literal
            # overrides on top, and stash the result back on
            # ``self._task.config.verifier.env`` — which is *per-trial*
            # because Trial._load_task() builds a fresh Task per trial. When
            # the real verify runs, resolve_env_vars sees only literal
            # values and passes them through unchanged, isolated from other
            # trials in the same process.
            saved_env = None
            verifier_cfg = None
            if verifier_env_overrides:
                task = getattr(self, "_task", None)
                task_config = getattr(task, "config", None)
                verifier_cfg = getattr(task_config, "verifier", None)
                if verifier_cfg is not None and hasattr(verifier_cfg, "env"):
                    try:
                        from harbor.utils.env import resolve_env_vars
                    except ImportError:
                        resolve_env_vars = None

                    original_env = getattr(verifier_cfg, "env", None) or {}
                    resolved: dict[str, str] = {}
                    if original_env and resolve_env_vars is not None:
                        try:
                            resolved = dict(resolve_env_vars(dict(original_env)))
                        except Exception:
                            _logger.exception(
                                "resolve_env_vars failed while preparing verifier env; "
                                "continuing with literal overrides only."
                            )
                            resolved = {}
                    elif original_env:
                        # Fallback: copy as-is; overrides below still apply.
                        resolved = dict(original_env)
                    resolved.update(verifier_env_overrides)
                    saved_env = original_env
                    try:
                        verifier_cfg.env = resolved
                    except Exception:
                        _logger.exception(
                            "Failed to apply verifier env overrides; judge may see "
                            "the agent's OPENAI_API_KEY."
                        )
                        saved_env = None

            try:
                return await _target(self, *args, **kwargs)
            finally:
                if saved_env is not None and verifier_cfg is not None:
                    try:
                        verifier_cfg.env = saved_env
                    except Exception:
                        pass

        _copy_callable_metadata(
            _patched_verify_with_retry,
            target,
            default_name="_verify_with_retry",
        )
        setattr(trial_cls, "_verify_with_retry", _patched_verify_with_retry)

    _logger.info(
        "Harbor runtime patch applied: agent_timeout=%s verifier_timeout=%s "
        "deadline_notice=%s disable_verifier_retry=%s verifier_env_overrides=%d",
        agent_timeout_sec,
        verifier_timeout_sec,
        deadline_notice,
        disable_verifier_retry,
        len(verifier_env_overrides),
    )
    _PATCH_APPLIED = True
