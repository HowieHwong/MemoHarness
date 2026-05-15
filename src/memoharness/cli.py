from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> None:
    """Run the only supported MemoHarness entrypoint."""
    from .harbor.loop import main as harbor_loop_main

    harbor_loop_main(argv if argv is not None else sys.argv[1:])


if __name__ == "__main__":
    main()
