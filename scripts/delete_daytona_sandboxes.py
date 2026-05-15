from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memoharness.config import MemoHarnessRuntimeConfig


def _mask_key(key: str) -> str:
    if len(key) <= 12:
        return key
    return f"{key[:8]}...{key[-4:]}"


def _resolve_daytona_keys(config_path: Path) -> list[str]:
    runtime = MemoHarnessRuntimeConfig.from_json_file(config_path)
    keys: list[str] = []
    for raw in runtime.experiment.daytona.api_keys:
        if not raw:
            continue
        if raw.startswith("$"):
            resolved = os.environ.get(raw[1:])
            if resolved:
                keys.append(resolved)
            else:
                print(f"Skipping unset Daytona env var: {raw[1:]}")
        else:
            keys.append(raw)
    deduped: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def _build_daytona_client(api_key: str):
    try:
        from daytona import Daytona, DaytonaConfig
    except ImportError as exc:
        return None

    return Daytona(DaytonaConfig(api_key=api_key))


def _http_request_json(api_key: str, method: str, url: str) -> object:
    request = urllib.request.Request(
        url,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code} {exc.reason}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error: {exc}") from exc

    if not raw:
        return None
    return json.loads(raw.decode("utf-8"))


def _list_sandboxes(client, api_key: str) -> list:
    if client is None:
        payload = _http_request_json(api_key, "GET", "https://app.daytona.io/api/sandbox")
        if isinstance(payload, dict):
            items = payload.get("items")
        elif isinstance(payload, list):
            items = payload
        else:
            items = None
        if items is None:
            raise RuntimeError("Unexpected Daytona API response: missing sandbox items.")
        return list(items)

    all_items: list = []
    page = 1
    limit = 200
    while True:
        result = client.list(page=page, limit=limit)
        items = getattr(result, "items", None)
        if items is None:
            raise RuntimeError("Unexpected Daytona list() response: missing 'items'.")
        batch = list(items)
        all_items.extend(batch)
        if len(batch) < limit:
            break
        page += 1
    return all_items


def _sandbox_label(sandbox) -> str:
    if isinstance(sandbox, dict):
        sandbox_id = sandbox.get("id") or "<unknown>"
        sandbox_name = sandbox.get("name")
        sandbox_state = sandbox.get("state") or "unknown"
    else:
        sandbox_id = getattr(sandbox, "id", None) or "<unknown>"
        sandbox_name = getattr(sandbox, "name", None)
        sandbox_state = getattr(sandbox, "state", None) or "unknown"
    if sandbox_name and sandbox_name != sandbox_id:
        return f"{sandbox_name} ({sandbox_id}, state={sandbox_state})"
    return f"{sandbox_id} (state={sandbox_state})"


def _sandbox_fingerprint(sandbox) -> str:
    """Stable identity for cross-key duplicate detection."""
    if isinstance(sandbox, dict):
        sandbox_id = sandbox.get("id")
        if sandbox_id:
            return f"id:{sandbox_id}"
        # Fallback when id is absent in payload.
        return "sig:" + json.dumps(sandbox, sort_keys=True, default=str, ensure_ascii=False)

    sandbox_id = getattr(sandbox, "id", None)
    if sandbox_id:
        return f"id:{sandbox_id}"

    # SDK object fallback: use the visible fields we can reliably read.
    fallback = {
        "name": getattr(sandbox, "name", None),
        "state": getattr(sandbox, "state", None),
    }
    return "sig:" + json.dumps(fallback, sort_keys=True, default=str, ensure_ascii=False)


def _report_cross_key_duplicate_sandboxes(
    key_rows: list[tuple[int, str, list]],
) -> int:
    """
    Print duplicate sandbox identities found across different keys.
    Returns the number of duplicate identities.
    """
    by_fingerprint: dict[str, list[str]] = {}
    for key_index, key, sandboxes in key_rows:
        masked_key = _mask_key(key)
        for sandbox in sandboxes:
            fp = _sandbox_fingerprint(sandbox)
            label = _sandbox_label(sandbox)
            by_fingerprint.setdefault(fp, []).append(
                f"key#{key_index}({masked_key}): {label}"
            )

    duplicates = {fp: refs for fp, refs in by_fingerprint.items() if len(refs) > 1}

    print("\n== Cross-key duplicate sandbox check ==")
    if not duplicates:
        print("No identical sandbox fingerprints found across keys.")
        return 0

    print(f"Found {len(duplicates)} duplicate sandbox fingerprint(s) across keys:")
    for fp in sorted(duplicates):
        print(f"- {fp}")
        for ref in duplicates[fp]:
            print(f"  {ref}")
    return len(duplicates)


def _delete_sandbox(client, api_key: str, sandbox, timeout: float) -> None:
    if client is None:
        if not isinstance(sandbox, dict) or not sandbox.get("id"):
            raise RuntimeError("Cannot delete sandbox without an id.")
        sandbox_id = urllib.parse.quote(str(sandbox["id"]), safe="")
        _http_request_json(api_key, "DELETE", f"https://app.daytona.io/api/sandbox/{sandbox_id}")
        return

    sandbox.delete(timeout=timeout)


def delete_all_sandboxes(
    config_path: Path,
    timeout: float,
    *,
    check_duplicates_only: bool = False,
) -> int:
    keys = _resolve_daytona_keys(config_path)
    if not keys:
        raise SystemExit(f"No Daytona API keys found in {config_path}")

    total_deleted = 0
    total_failed = 0
    total_seen = 0
    key_rows: list[tuple[int, str, object, list]] = []

    for index, key in enumerate(keys, start=1):
        client = _build_daytona_client(key)
        sandboxes = _list_sandboxes(client, key)
        total_seen += len(sandboxes)
        key_rows.append((index, key, client, sandboxes))

    # Report duplicates before any deletion happens.
    duplicate_count = _report_cross_key_duplicate_sandboxes(
        [(index, key, sandboxes) for index, key, _, sandboxes in key_rows]
    )

    if check_duplicates_only:
        print(
            "\nCheck-only summary: duplicate_fingerprints={0}, total_sandboxes={1}, keys={2}".format(
                duplicate_count, total_seen, len(keys)
            )
        )
        return 0

    for index, key, client, sandboxes in key_rows:
        print(f"\n== Daytona key #{index}: {_mask_key(key)} ==")
        if not sandboxes:
            print("No sandboxes found.")
            continue

        print(f"Found {len(sandboxes)} sandbox(es). Deleting...")
        deleted = 0
        failed = 0
        for sandbox in sandboxes:
            label = _sandbox_label(sandbox)
            try:
                _delete_sandbox(client, key, sandbox, timeout)
                deleted += 1
                print(f"Deleted: {label}")
            except Exception as exc:  # pragma: no cover - depends on remote API behavior
                failed += 1
                print(f"Failed:  {label} -> {exc}")

        total_deleted += deleted
        total_failed += failed
        print(
            f"Key summary: deleted={deleted}, failed={failed}, total={len(sandboxes)}"
        )

    print(
        f"\nOverall summary: deleted={total_deleted}, failed={total_failed}, total={total_seen}"
    )
    return 0 if total_failed == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Delete all Daytona sandboxes for all keys listed in experiment config."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "experiment.json"),
        help="Path to MemoHarness experiment config JSON.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-sandbox delete timeout in seconds.",
    )
    parser.add_argument(
        "--check-duplicates-only",
        action="store_true",
        help="Only check for identical sandboxes across keys, do not delete.",
    )
    args = parser.parse_args(argv)
    return delete_all_sandboxes(
        Path(args.config).resolve(),
        timeout=args.timeout,
        check_duplicates_only=args.check_duplicates_only,
    )


if __name__ == "__main__":
    raise SystemExit(main())
