from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from delete_daytona_sandboxes import (
    ROOT,
    _build_daytona_client,
    _list_sandboxes,
    _mask_key,
    _resolve_daytona_keys,
)


DEFAULT_COMMAND = [
    sys.executable,
    "-m",
    "memoharness.harbor.loop",
    "--config",
    "configs/experiment.json",
    "-d",
    "financeagent@public",
]

_FINANCE_PROMPT_TEMPLATE = (
    ROOT / "configs" / "harness_codex" / "financeagent_prompt.md"
)
_PROMPT_TEMPLATE_ENV = "MEMOHARNESS_CODEX_PROMPT_TEMPLATE"


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _check_all_keys(config_path: Path) -> tuple[bool, int, int, int]:
    keys = _resolve_daytona_keys(config_path)
    if not keys:
        raise SystemExit(f"No Daytona API keys found in {config_path}")

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Poll every Daytona key in the MemoHarness config; once all keys report "
            "zero sandboxes (and no API errors), launch the financeagent training run."
        )
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "experiment.json"),
        help="Path to MemoHarness experiment config JSON.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=1800,
        help="Seconds between polls. Default 1800 (30 minutes).",
    )
    parser.add_argument(
        "--command",
        default=None,
        help=(
            "Override the command to launch. Parsed with shlex.split. "
            "Default: python -m memoharness.harbor.loop --config configs/experiment.json -d financeagent@public"
        ),
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run a single check and exit (exit 0 if idle, 1 if not).",
    )
    args = parser.parse_args(argv)

    if args.interval_seconds <= 0:
        raise SystemExit("--interval-seconds must be positive")

    config_path = Path(args.config).resolve()
    command = shlex.split(args.command) if args.command else list(DEFAULT_COMMAND)

    print(f"[{_now()}] Config: {config_path}")
    print(f"[{_now()}] Launch when idle: {' '.join(command)}")
    print(f"[{_now()}] Poll interval: {args.interval_seconds}s")
    print()

    while True:
        try:
            all_idle, total, errors, key_count = _check_all_keys(config_path)
        except KeyboardInterrupt:
            print(f"\n[{_now()}] Interrupted by user.")
            return 130

        if args.check_only:
            return 0 if all_idle else 1

        if all_idle:
            print(f"[{_now()}] All {key_count} keys idle. Launching command.")
            print()
            launch_env = os.environ.copy()
            if _PROMPT_TEMPLATE_ENV not in launch_env:
                if _FINANCE_PROMPT_TEMPLATE.is_file():
                    launch_env[_PROMPT_TEMPLATE_ENV] = str(_FINANCE_PROMPT_TEMPLATE)
                    print(
                        f"[{_now()}] Injecting {_PROMPT_TEMPLATE_ENV}={_FINANCE_PROMPT_TEMPLATE}"
                    )
                else:
                    print(
                        f"[{_now()}] WARNING: {_FINANCE_PROMPT_TEMPLATE} missing; "
                        "launching without prompt template."
                    )
            try:
                result = subprocess.run(command, cwd=str(ROOT), env=launch_env)
            except KeyboardInterrupt:
                print(f"\n[{_now()}] Launched command interrupted by user.")
                return 130
            return result.returncode

        print(
            f"[{_now()}] Not idle: total_sandboxes={total}, errors={errors}. "
            f"Sleeping {args.interval_seconds}s..."
        )
        print()
        try:
            time.sleep(args.interval_seconds)
        except KeyboardInterrupt:
            print(f"\n[{_now()}] Interrupted during sleep.")
            return 130


if __name__ == "__main__":
    raise SystemExit(main())
