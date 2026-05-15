from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from delete_daytona_sandboxes import (  # noqa: E402
    ROOT,
    _build_daytona_client,
    _delete_sandbox,
    _list_sandboxes,
    _mask_key,
    _resolve_daytona_keys,
    _sandbox_label,
)


DEFAULT_MODELS = [
    "deepseek-v3.2",
    "qwen3.5-397b",
    "glm-5",
    "gemini-3.1-pro",
    "claude-sonnet-4.5",
]
DEFAULT_RUN_ID_TEMPLATE = "tb-{model_slug}"
DEFAULT_STATE_PATH = ROOT / "artifacts" / "tb_eval_scheduler_state.json"
DEFAULT_CLEANUP_INTERVAL_SECONDS = 3600
DEFAULT_CLEANUP_TIMEOUT_SECONDS = 60.0
TERMINAL_STATUSES = {"succeeded", "failed", "skipped"}
ACTIVE_STATUSES = TERMINAL_STATUSES | {"pending", "running"}


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected a JSON object in {path}")
    return payload


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _sanitize_token(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip())
    cleaned = cleaned.strip(".-_")
    return cleaned or "model"


def _dedupe_models(models: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for raw in models:
        model = str(raw or "").strip()
        if not model or model in seen:
            continue
        deduped.append(model)
        seen.add(model)
    return deduped


def _parse_model_run_id_overrides(values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise SystemExit(
                f"Invalid --model-run-id value '{raw}'. Expected MODEL=RUN_ID."
            )
        model, run_id = raw.split("=", 1)
        model = model.strip()
        run_id = run_id.strip()
        if not model or not run_id:
            raise SystemExit(
                f"Invalid --model-run-id value '{raw}'. Expected MODEL=RUN_ID."
            )
        overrides[model] = run_id
    return overrides


def _build_model_run_ids(
    models: list[str],
    run_id_template: str,
    overrides: dict[str, str],
) -> dict[str, str]:
    unknown_overrides = sorted(set(overrides) - set(models))
    if unknown_overrides:
        raise SystemExit(
            "Found --model-run-id override(s) for unknown model(s): "
            + ", ".join(unknown_overrides)
        )

    run_ids: dict[str, str] = {}
    for model in models:
        if model in overrides:
            run_ids[model] = overrides[model]
            continue
        try:
            run_id = run_id_template.format(
                model=model,
                model_slug=_sanitize_token(model),
            )
        except KeyError as exc:
            raise SystemExit(
                f"Unknown placeholder '{exc.args[0]}' in --run-id-template."
            ) from exc
        run_id = run_id.strip()
        if not run_id:
            raise SystemExit(f"Resolved empty run_id for model '{model}'.")
        run_ids[model] = run_id
    return run_ids


class ModelEvalScheduler:
    def __init__(
        self,
        *,
        config_path: Path,
        state_path: Path,
        models: list[str],
        model_run_ids: dict[str, str],
        interval_seconds: int,
        cleanup_interval_seconds: int,
        python_executable: str,
        root: Path = ROOT,
        reset_state: bool = False,
    ) -> None:
        self.root = Path(root).resolve()
        self.config_path = self._resolve_repo_path(config_path)
        self.state_path = self._resolve_repo_path(state_path)
        self.models = _dedupe_models(models)
        if not self.models:
            raise SystemExit("No models configured.")
        self.model_run_ids = {model: model_run_ids[model] for model in self.models}
        self.interval_seconds = interval_seconds
        self.cleanup_interval_seconds = int(cleanup_interval_seconds)
        self.python_executable = python_executable
        self.reset_state = reset_state

        if self.interval_seconds <= 0:
            raise SystemExit("--interval-seconds must be positive")
        if self.cleanup_interval_seconds < 0:
            raise SystemExit("--cleanup-interval-seconds must be >= 0")

        self._validate_models_exist()
        self.state = self._load_state()

    def _resolve_repo_path(self, raw_path: Path) -> Path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self.root / path
        return path.resolve()

    def _config_payload(self) -> dict[str, Any]:
        return _read_json(self.config_path)

    def _validate_models_exist(self) -> None:
        payload = self._config_payload()
        raw_models = payload.get("models")
        if not isinstance(raw_models, dict):
            raise SystemExit(
                f"Config {self.config_path} does not contain a top-level 'models' object."
            )
        missing = [model for model in self.models if model not in raw_models]
        if missing:
            raise SystemExit(
                "Model(s) missing from config 'models': " + ", ".join(missing)
            )

    def _new_state(self) -> dict[str, Any]:
        timestamp = _now()
        return {
            "created_at": timestamp,
            "updated_at": timestamp,
            "current_model": None,
            "last_cleanup_at": None,
            "last_cleanup_timestamp": None,
            "last_cleanup_summary": None,
            "models": {
                model: {
                    "run_id": self.model_run_ids[model],
                    "status": "pending",
                    "last_started_at": None,
                    "completed_at": None,
                    "last_error": None,
                    "attempts": [],
                }
                for model in self.models
            },
        }

    def _save_state(self) -> None:
        self.state["updated_at"] = _now()
        _atomic_write_json(self.state_path, self.state)

    def _load_state(self) -> dict[str, Any]:
        if self.reset_state or not self.state_path.exists():
            state = self._new_state()
            self.state = state
            self._save_state()
            return state

        try:
            raw = _read_json(self.state_path)
        except (OSError, json.JSONDecodeError):
            state = self._new_state()
            self.state = state
            self._save_state()
            return state

        state = self._new_state()
        state["created_at"] = str(raw.get("created_at") or state["created_at"])
        state["last_cleanup_at"] = raw.get("last_cleanup_at")
        raw_cleanup_timestamp = raw.get("last_cleanup_timestamp")
        if isinstance(raw_cleanup_timestamp, (int, float)):
            state["last_cleanup_timestamp"] = float(raw_cleanup_timestamp)
        state["last_cleanup_summary"] = raw.get("last_cleanup_summary")

        raw_models = raw.get("models")
        if isinstance(raw_models, dict):
            for model in self.models:
                entry = raw_models.get(model)
                if not isinstance(entry, dict):
                    continue
                attempts = entry.get("attempts")
                status = str(entry.get("status") or "pending")
                if status not in ACTIVE_STATUSES:
                    status = "pending"
                state["models"][model].update(
                    {
                        "status": status,
                        "last_started_at": entry.get("last_started_at"),
                        "completed_at": entry.get("completed_at"),
                        "last_error": entry.get("last_error"),
                        "attempts": attempts if isinstance(attempts, list) else [],
                    }
                )

        current_model = raw.get("current_model")
        if (
            current_model in self.models
            and state["models"][current_model]["status"] == "running"
        ):
            state["current_model"] = current_model

        self.state = state
        self._save_state()
        return state

    def next_model(self) -> str | None:
        current_model = self.state.get("current_model")
        if (
            current_model in self.models
            and self.state["models"][current_model]["status"] == "running"
        ):
            return current_model

        for model in self.models:
            status = self.state["models"][model]["status"]
            if status not in TERMINAL_STATUSES:
                return model
        return None

    def all_models_finished(self) -> bool:
        return self.next_model() is None

    def set_active_model(self, model: str) -> None:
        payload = self._config_payload()
        payload["active_model"] = model
        _atomic_write_json(self.config_path, payload)

    def _record_attempt(
        self,
        model: str,
        *,
        run_id: str,
        status: str,
        returncode: int | None,
        error: str | None,
        launched: bool,
        started_at: str,
        finished_at: str,
    ) -> None:
        entry = self.state["models"][model]
        entry["status"] = status
        entry["run_id"] = run_id
        entry["last_started_at"] = started_at
        entry["completed_at"] = finished_at if status in TERMINAL_STATUSES else None
        entry["last_error"] = error
        entry["attempts"].append(
            {
                "run_id": run_id,
                "status": status,
                "returncode": returncode,
                "error": error,
                "launched": launched,
                "started_at": started_at,
                "finished_at": finished_at,
            }
        )
        self.state["current_model"] = None if status in TERMINAL_STATUSES else model
        self._save_state()

    def mark_running(self, model: str, run_id: str) -> str:
        started_at = _now()
        entry = self.state["models"][model]
        entry["status"] = "running"
        entry["run_id"] = run_id
        entry["last_started_at"] = started_at
        entry["completed_at"] = None
        entry["last_error"] = None
        self.state["current_model"] = model
        self._save_state()
        return started_at

    def load_daytona_keys(self) -> list[str]:
        keys = _resolve_daytona_keys(self.config_path)
        if not keys:
            raise SystemExit(f"No Daytona API keys found in {self.config_path}")
        return keys

    def cleanup_all_keys(self) -> dict[str, int]:
        keys = self.load_daytona_keys()
        total_deleted = 0
        total_failed = 0
        total_seen = 0

        print(f"[{_now()}] Forced cleanup: scanning {len(keys)} key(s)...")
        for index, key in enumerate(keys, start=1):
            masked = _mask_key(key)
            try:
                client = _build_daytona_client(key)
                sandboxes = _list_sandboxes(client, key)
            except Exception as exc:
                total_failed += 1
                print(f"  [{index:02d}] key={masked} ERROR listing sandboxes: {exc}")
                continue

            total_seen += len(sandboxes)
            if not sandboxes:
                print(f"  [{index:02d}] key={masked} idle")
                continue

            print(f"  [{index:02d}] key={masked} deleting {len(sandboxes)} sandbox(es)...")
            for sandbox in sandboxes:
                label = _sandbox_label(sandbox)
                try:
                    _delete_sandbox(
                        client,
                        key,
                        sandbox,
                        timeout=DEFAULT_CLEANUP_TIMEOUT_SECONDS,
                    )
                    total_deleted += 1
                    print(f"    deleted: {label}")
                except Exception as exc:
                    total_failed += 1
                    print(f"    failed:  {label} -> {exc}")

        summary = {
            "deleted": total_deleted,
            "failed": total_failed,
            "seen": total_seen,
        }
        print(
            f"[{_now()}] Forced cleanup summary: deleted={total_deleted}, "
            f"failed={total_failed}, total={total_seen}"
        )
        return summary

    def _seconds_until_cleanup(self) -> float | None:
        if self.cleanup_interval_seconds <= 0:
            return None
        last_cleanup_timestamp = self.state.get("last_cleanup_timestamp")
        if not isinstance(last_cleanup_timestamp, (int, float)):
            return float(self.cleanup_interval_seconds)
        elapsed = time.time() - float(last_cleanup_timestamp)
        return max(0.0, float(self.cleanup_interval_seconds) - elapsed)

    def maybe_cleanup_due(self, *, force: bool = False) -> bool:
        due_in = self._seconds_until_cleanup()
        if not force:
            if due_in is None or due_in > 0:
                return False

        started_at = _now()
        summary = self.cleanup_all_keys()
        self.state["last_cleanup_at"] = started_at
        self.state["last_cleanup_timestamp"] = float(time.time())
        self.state["last_cleanup_summary"] = summary
        self._save_state()
        return True

    def run_startup_cleanup(self) -> None:
        print(f"[{_now()}] Startup cleanup: clearing all configured sandboxes before scheduling.")
        self.maybe_cleanup_due(force=True)

    def check_all_keys(self) -> tuple[bool, int, int, int]:
        keys = self.load_daytona_keys()

        total_sandboxes = 0
        errors = 0

        print(f"[{_now()}] Polling {len(keys)} key(s)...")
        for index, key in enumerate(keys, start=1):
            masked = _mask_key(key)
            try:
                client = _build_daytona_client(key)
                sandboxes = _list_sandboxes(client, key)
            except Exception as exc:
                errors += 1
                print(f"  [{index:02d}] key={masked} ERROR {exc}")
                continue

            count = len(sandboxes)
            total_sandboxes += count
            status = "idle" if count == 0 else f"BUSY count={count}"
            print(f"  [{index:02d}] key={masked} {status}")

        all_idle = total_sandboxes == 0 and errors == 0
        return all_idle, total_sandboxes, errors, len(keys)

    def build_command(self, run_id: str) -> list[str]:
        return [
            self.python_executable,
            "-m",
            "memoharness.harbor.loop",
            "--config",
            str(self.config_path),
            "--eval-only",
            "--run-id",
            run_id,
        ]

    def launch_command(self, model: str, run_id: str) -> int:
        command = self.build_command(run_id)
        print(
            f"[{_now()}] Launching {model} with run_id={run_id}: {' '.join(command)}"
        )
        result = subprocess.run(command, cwd=str(self.root))
        return result.returncode

    def run_once(self) -> str:
        model = self.next_model()
        if model is None:
            print(f"[{_now()}] All configured models have already been attempted.")
            return "complete"

        did_cleanup = self.maybe_cleanup_due()
        all_idle, total, errors, key_count = self.check_all_keys()
        if not all_idle:
            print(
                f"[{_now()}] Not idle for next model {model}: "
                f"total_sandboxes={total}, errors={errors}, key_count={key_count}"
            )
            if did_cleanup:
                print(f"[{_now()}] Cleanup just ran; rechecking immediately.")
                return "retry"
            return "waiting"

        run_id = self.model_run_ids[model]
        print(f"[{_now()}] All {key_count} keys idle. Preparing model {model}.")
        self.set_active_model(model)
        started_at = self.mark_running(model, run_id)
        try:
            returncode = self.launch_command(model, run_id)
        except KeyboardInterrupt:
            print(f"\n[{_now()}] Interrupted while {model} was running.")
            raise
        except Exception as exc:
            finished_at = _now()
            error = str(exc)
            print(f"[{finished_at}] Launch failed for {model}: {error}")
            self._record_attempt(
                model,
                run_id=run_id,
                status="failed",
                returncode=None,
                error=error,
                launched=False,
                started_at=started_at,
                finished_at=finished_at,
            )
            return "failed"

        finished_at = _now()
        status = "succeeded" if returncode == 0 else "failed"
        error = None if returncode == 0 else f"Command exited with {returncode}"
        print(
            f"[{finished_at}] Model {model} finished with status={status} "
            f"returncode={returncode}"
        )
        self._record_attempt(
            model,
            run_id=run_id,
            status=status,
            returncode=returncode,
            error=error,
            launched=True,
            started_at=started_at,
            finished_at=finished_at,
        )
        return status

    def run_forever(self) -> int:
        print(f"[{_now()}] Config: {self.config_path}")
        print(f"[{_now()}] State file: {self.state_path}")
        print(f"[{_now()}] Models: {', '.join(self.models)}")
        for model in self.models:
            print(
                f"[{_now()}] Planned run_id[{model}] = {self.model_run_ids[model]}"
            )
        print(f"[{_now()}] Poll interval: {self.interval_seconds}s")
        print(f"[{_now()}] Forced cleanup interval: {self.cleanup_interval_seconds}s")
        print()

        while True:
            try:
                outcome = self.run_once()
            except KeyboardInterrupt:
                print(f"\n[{_now()}] Interrupted by user.")
                return 130

            if outcome == "complete":
                return 0
            if outcome == "retry":
                print()
                continue
            if outcome == "waiting":
                cleanup_due_in = self._seconds_until_cleanup()
                sleep_seconds = float(self.interval_seconds)
                if cleanup_due_in is not None:
                    sleep_seconds = min(sleep_seconds, max(1.0, cleanup_due_in))
                print(f"[{_now()}] Sleeping {sleep_seconds:.0f}s...")
                print()
                try:
                    time.sleep(sleep_seconds)
                except KeyboardInterrupt:
                    print(f"\n[{_now()}] Interrupted during sleep.")
                    return 130
            else:
                print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Poll Daytona keys until all sandboxes are idle, then rotate "
            "active_model across a fixed model list and run eval-only once per model."
        )
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "experiment.json"),
        help="Path to MemoHarness experiment config JSON.",
    )
    parser.add_argument(
        "--state-file",
        default=str(DEFAULT_STATE_PATH),
        help="Path to the scheduler state JSON.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=1200,
        help="Seconds between idle checks. Default 1200 (20 minutes).",
    )
    parser.add_argument(
        "--cleanup-interval-seconds",
        type=int,
        default=DEFAULT_CLEANUP_INTERVAL_SECONDS,
        help="Force-delete all configured Daytona sandboxes every N seconds. Default 3600 (1 hour).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Ordered model list to attempt.",
    )
    parser.add_argument(
        "--run-id-template",
        default=DEFAULT_RUN_ID_TEMPLATE,
        help="Default run_id template. Supported placeholders: {model}, {model_slug}.",
    )
    parser.add_argument(
        "--model-run-id",
        action="append",
        default=[],
        help="Per-model run_id override in MODEL=RUN_ID form. Can be passed multiple times.",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used to launch memoharness.harbor.loop.",
    )
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Discard any existing scheduler state and start from the first model again.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single scheduling pass and exit.",
    )
    args = parser.parse_args(argv)

    models = _dedupe_models(args.models)
    overrides = _parse_model_run_id_overrides(args.model_run_id)
    model_run_ids = _build_model_run_ids(models, args.run_id_template, overrides)

    scheduler = ModelEvalScheduler(
        config_path=Path(args.config),
        state_path=Path(args.state_file),
        models=models,
        model_run_ids=model_run_ids,
        interval_seconds=args.interval_seconds,
        cleanup_interval_seconds=args.cleanup_interval_seconds,
        python_executable=args.python_executable,
        reset_state=args.reset_state,
    )

    scheduler.run_startup_cleanup()

    if args.once:
        try:
            outcome = scheduler.run_once()
        except KeyboardInterrupt:
            print(f"\n[{_now()}] Interrupted by user.")
            return 130
        if outcome in {"complete", "succeeded"}:
            return 0
        if outcome == "retry":
            return 3
        if outcome == "waiting":
            return 2
        return 1
    return scheduler.run_forever()


if __name__ == "__main__":
    raise SystemExit(main())
