from .agent import MemoHarnessAgent
from .codex_agent import MemoHarnessCodexAgent

__all__ = ["MemoHarnessAgent", "MemoHarnessCodexAgent", "HarborTrainingLoop"]


def __getattr__(name):
    if name == "HarborTrainingLoop":
        try:
            from .loop import HarborTrainingLoop
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith("harbor"):
                raise ModuleNotFoundError(
                    "HarborTrainingLoop requires the optional 'harbor' package."
                ) from exc
            raise
        return HarborTrainingLoop
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
