from __future__ import annotations

import os
import sys


if os.getenv("MEMOHARNESS_ENABLE_HARBOR_RUNTIME_PATCH") == "1":
    try:
        from memoharness.harbor.runtime_patch import apply_runtime_patch

        apply_runtime_patch()
    except Exception as exc:
        print(
            f"[memoharness] failed to apply Harbor runtime patch: {exc}",
            file=sys.stderr,
            flush=True,
        )
