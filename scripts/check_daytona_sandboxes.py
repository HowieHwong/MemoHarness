from __future__ import annotations

import argparse
from pathlib import Path

from delete_daytona_sandboxes import (
    ROOT,
    _build_daytona_client,
    _mask_key,
    _list_sandboxes,
    _resolve_daytona_keys,
)


def _sandbox_id(sandbox) -> str:
    if isinstance(sandbox, dict):
        value = sandbox.get("id")
    else:
        value = getattr(sandbox, "id", None)
    return str(value or "")


def _sandbox_state(sandbox) -> str:
    if isinstance(sandbox, dict):
        value = sandbox.get("state")
    else:
        value = getattr(sandbox, "state", None)
    return str(value or "unknown")


def check_sandboxes(config_path: Path, expected_count: int) -> int:
    keys = _resolve_daytona_keys(config_path)
    if not keys:
        raise SystemExit(f"No Daytona API keys found in {config_path}")

    ok = 0
    mismatched = 0
    failed = 0

    print(f"Checking {len(keys)} Daytona key(s); expected sandbox count per key: {expected_count}")
    print()

    for index, key in enumerate(keys, start=1):
        masked_key = _mask_key(key)
        try:
            client = _build_daytona_client(key)
            sandboxes = _list_sandboxes(client, key)
        except Exception as exc:  # pragma: no cover - depends on remote API behavior
            failed += 1
            print(f"[{index:02d}] key={masked_key} status=ERROR error={exc}")
            continue

        sandbox_ids = [_sandbox_id(sandbox) for sandbox in sandboxes if _sandbox_id(sandbox)]
        states = [_sandbox_state(sandbox) for sandbox in sandboxes]
        matches = len(sandboxes) == expected_count
        if matches:
            ok += 1
            status = "OK"
        else:
            mismatched += 1
            status = "MISMATCH"

        ids_text = ",".join(sandbox_ids) if sandbox_ids else "-"
        states_text = ",".join(states) if states else "-"
        print(
            f"[{index:02d}] key={masked_key} status={status} "
            f"count={len(sandboxes)} sandbox_ids={ids_text} states={states_text}"
        )

    print()
    print(
        "Summary: "
        f"ok={ok} mismatched={mismatched} failed={failed} total={len(keys)}"
    )
    return 0 if mismatched == 0 and failed == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check how many Daytona sandboxes each configured key currently owns."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "experiment.json"),
        help="Path to MemoHarness experiment config JSON.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=1,
        help="Expected sandbox count per key. Defaults to 1.",
    )
    args = parser.parse_args(argv)

    if args.expected_count < 0:
        raise SystemExit("--expected-count must be >= 0")

    return check_sandboxes(Path(args.config).resolve(), expected_count=args.expected_count)


if __name__ == "__main__":
    raise SystemExit(main())
