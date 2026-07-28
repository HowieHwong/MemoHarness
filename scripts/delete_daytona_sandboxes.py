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


def _read_key_file(path: Path) -> list[str]:
    """Read one-key-per-line text file, ignoring blanks and # comments."""
    keys: list[str] = []
    if not path.exists():
        return keys
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        token = line.split()[0]
        if token and not token.startswith("#"):
            keys.append(token)
    return keys


def _resolve_daytona_keys(config_path: Path, keys_file: Path | None = None) -> list[str]:
    runtime = MemoHarnessRuntimeConfig.from_json_file(config_path)
    keys: list[str] = []

    # Optional explicit keys-file override (one key per line, # comments OK).
    if keys_file is not None:
        keys.extend(_read_key_file(keys_file))

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
    """List sandboxes for a key.

    Two paths:
    1. SDK client (preferred): iterate ``client.list()``. The installed
       ``daytona`` SDK returns an ``Iterator[Sandbox]``; older SDKs returned
       an object with ``.items``. We detect the shape at runtime so we keep
       working across SDK versions.
    2. HTTP fallback (when SDK is not importable): call the REST endpoint
       ``GET /api/sandbox`` directly and read ``items`` out of the JSON
       payload.
    """
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

    # Modern SDK (returns Iterator[Sandbox] from list()).
    import inspect as _inspect

    list_sig = _inspect.signature(client.list)
    accepts_kwargs = any(
        p.kind in (p.VAR_KEYWORD, p.KEYWORD_ONLY)
        for p in list_sig.parameters.values()
    ) or any(
        p.kind == p.VAR_KEYWORD for p in list_sig.parameters.values()
    )
    accepts_page = "page" in list_sig.parameters or accepts_kwargs

    # Legacy path: list(page=, limit=) returned an object with .items.
    if accepts_page:
        all_items: list = []
        page = 1
        limit = 200
        while True:
            result = client.list(page=page, limit=limit)
            items = getattr(result, "items", None)
            if items is None:
                raise RuntimeError(
                    "Unexpected Daytona list() response: missing 'items'."
                )
            batch = list(items)
            all_items.extend(batch)
            if len(batch) < limit:
                break
            page += 1
        return all_items

    # Modern path: list() returns an Iterator yielding Sandbox objects directly.
    # Iterating the generator runs the SDK code, which will raise
    # DaytonaError/TypeError up front if the kwargs are wrong; we don't
    # pre-check, just iterate.
    try:
        return list(client.list())
    except TypeError as exc:
        # Be defensive: if user passed kwargs even though we tried not to,
        # retry without any args. (No-arg list() returns *all* sandboxes
        # for the key — server-side pagination handles the rest.)
        raise RuntimeError(
            f"Daytona.list() signature not supported by installed SDK: {exc}"
        ) from exc


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
    """Delete one sandbox.

    Strategies, in order:
    1. Sandbox object with ``.delete(timeout=, wait=)`` (modern SDK).
    2. Top-level ``client.delete(sandbox)`` or ``client.delete(sandbox.id)``
       (legacy SDK API).
    3. HTTP fallback when no client is available.
    """
    if client is None:
        if not isinstance(sandbox, dict) or not sandbox.get("id"):
            raise RuntimeError("Cannot delete sandbox without an id.")
        sandbox_id = urllib.parse.quote(str(sandbox["id"]), safe="")
        _http_request_json(api_key, "DELETE", f"https://app.daytona.io/api/sandbox/{sandbox_id}")
        return

    last_exc: Exception | None = None

    # Modern SDK: Sandbox.delete(timeout=, wait=)
    delete_method = getattr(sandbox, "delete", None)
    if callable(delete_method):
        import inspect as _inspect

        try:
            sig = _inspect.signature(delete_method)
        except (ValueError, TypeError):
            sig = None
        try:
            if sig is not None and "timeout" in sig.parameters:
                if "wait" in sig.parameters:
                    delete_method(timeout=timeout, wait=False)
                else:
                    delete_method(timeout=timeout)
            else:
                delete_method()
            return
        except Exception as exc:
            # Already-gone sandboxes (DESTROYED/ERROR state) often raise
            # DaytonaNotFoundError; treat as success since the sandbox is
            # effectively deleted from our perspective.
            exc_name = type(exc).__name__
            if "NotFound" in exc_name or "404" in str(exc) or "not found" in str(exc).lower():
                return
            last_exc = exc
        else:
            return

    # Legacy SDK: client.delete(sandbox) — the new SDK also exposes
    # Daytona.delete(sandbox: 'Sandbox', timeout=, wait=), so passing the
    # whole sandbox object (NOT the string id) is the right call.
    client_delete = getattr(client, "delete", None)
    if callable(client_delete):
        import inspect as _inspect

        try:
            c_sig = _inspect.signature(client_delete)
        except (ValueError, TypeError):
            c_sig = None
        try:
            kwargs: dict = {}
            if c_sig is not None:
                if "timeout" in c_sig.parameters:
                    kwargs["timeout"] = timeout
                if "wait" in c_sig.parameters:
                    kwargs["wait"] = False
            client_delete(sandbox, **kwargs)
            return
        except Exception as exc:
            exc_name = type(exc).__name__
            if "NotFound" in exc_name or "404" in str(exc) or "not found" in str(exc).lower():
                return
            last_exc = exc

    # Last resort: HTTP fallback using the already-resolved sandbox id.
    sandbox_id = getattr(sandbox, "id", None)
    if sandbox_id is None and isinstance(sandbox, dict):
        sandbox_id = sandbox.get("id")
    if sandbox_id is None:
        raise RuntimeError("Cannot delete sandbox without an id.")
    sandbox_id_q = urllib.parse.quote(str(sandbox_id), safe="")
    try:
        _http_request_json(api_key, "DELETE", f"https://app.daytona.io/api/sandbox/{sandbox_id_q}")
        return
    except Exception as exc:
        # 404 from the REST endpoint usually means the sandbox is already
        # gone on the server side. Treat as success.
        if "404" in str(exc) or "Not Found" in str(exc) or "not found" in str(exc).lower():
            return
        raise RuntimeError(
            f"Failed to delete sandbox {sandbox_id!r}: {exc} (prior SDK error: {last_exc})"
        ) from exc


def delete_all_sandboxes(
    config_path: Path,
    timeout: float,
    *,
    check_duplicates_only: bool = False,
    keys_file: Path | None = None,
) -> int:
    keys = _resolve_daytona_keys(config_path, keys_file=keys_file)
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
    parser.add_argument(
        "--keys-file",
        type=Path,
        default=None,
        help=(
            "Optional text file with one Daytona API key per line "
            "(# comments allowed). Keys are appended to those in --config; "
            "use this to clean up keys stored in scripts/daytona_keys.txt."
        ),
    )
    args = parser.parse_args(argv)
    keys_file = args.keys_file.resolve() if args.keys_file is not None else None
    return delete_all_sandboxes(
        Path(args.config).resolve(),
        timeout=args.timeout,
        check_duplicates_only=args.check_duplicates_only,
        keys_file=keys_file,
    )


if __name__ == "__main__":
    raise SystemExit(main())
