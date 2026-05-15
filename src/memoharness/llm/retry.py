from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar

from ..config.runtime import ApiModelConfig

T = TypeVar("T")

_INFINITE_RETRY_LIMIT = -1
_DEFAULT_CONNECTION_RETRY_LIMIT = -1
_RATE_LIMIT_RETRY_LIMIT = 1
_RATE_LIMIT_RETRY_DELAY_SEC = 5.0
_CONNECTION_RETRY_BASE_DELAY_SEC = 2.0
_CONNECTION_RETRY_MAX_DELAY_SEC = 30.0


def _iter_exception_chain(exc: BaseException):
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        yield current
        seen.add(id(current))
        current = current.__cause__ or current.__context__


def _exception_chain_matches(exc: BaseException, needle: str) -> bool:
    return any(needle in type(item).__name__ for item in _iter_exception_chain(exc))


def is_rate_limit_error(exc: BaseException) -> bool:
    return _exception_chain_matches(exc, "RateLimitError")


def is_connection_retryable(exc: BaseException) -> bool:
    return _exception_chain_matches(exc, "Timeout") or _exception_chain_matches(
        exc, "Connection"
    )


def _connection_retry_limit(api_config: ApiModelConfig | None) -> int:
    if api_config is None:
        return _DEFAULT_CONNECTION_RETRY_LIMIT
    return int(api_config.connection_retry_limit)


def _retry_limit_text(limit: int) -> str:
    return "inf" if limit < 0 else str(limit)


def _can_retry(retries_taken: int, retry_limit: int) -> bool:
    return retry_limit == _INFINITE_RETRY_LIMIT or retries_taken < retry_limit


def _connection_retry_delay(retry_number: int) -> float:
    exponent = min(max(retry_number - 1, 0), 4)
    delay = _CONNECTION_RETRY_BASE_DELAY_SEC * (2**exponent)
    return min(delay, _CONNECTION_RETRY_MAX_DELAY_SEC)


def call_with_retries(
    func: Callable[[], T],
    *,
    api_config: ApiModelConfig | None,
    logger: logging.Logger,
    operation: str,
) -> T:
    connection_retries_taken = 0
    rate_limit_retries_taken = 0
    connection_retry_limit = _connection_retry_limit(api_config)

    while True:
        try:
            return func()
        except Exception as exc:
            exc_name = type(exc).__name__

            if is_connection_retryable(exc):
                if not _can_retry(connection_retries_taken, connection_retry_limit):
                    raise RuntimeError(
                        "{0} failed after {1} connection/timeout retries ({2}): {3}".format(
                            operation,
                            connection_retries_taken,
                            exc_name,
                            exc,
                        )
                    ) from exc

                connection_retries_taken += 1
                retry_delay = _connection_retry_delay(connection_retries_taken)
                logger.warning(
                    "%s hit a transient connection/timeout error "
                    "(retry %d/%s); retrying in %.1fs: %s",
                    operation,
                    connection_retries_taken,
                    _retry_limit_text(connection_retry_limit),
                    retry_delay,
                    exc,
                )
                time.sleep(retry_delay)
                continue

            if is_rate_limit_error(exc):
                if not _can_retry(rate_limit_retries_taken, _RATE_LIMIT_RETRY_LIMIT):
                    raise RuntimeError(
                        "{0} failed after {1} rate-limit retries ({2}): {3}".format(
                            operation,
                            rate_limit_retries_taken,
                            exc_name,
                            exc,
                        )
                    ) from exc

                rate_limit_retries_taken += 1
                logger.warning(
                    "%s hit a rate limit (retry %d/%d); retrying in %.1fs: %s",
                    operation,
                    rate_limit_retries_taken,
                    _RATE_LIMIT_RETRY_LIMIT,
                    _RATE_LIMIT_RETRY_DELAY_SEC,
                    exc,
                )
                time.sleep(_RATE_LIMIT_RETRY_DELAY_SEC)
                continue

            raise RuntimeError("{0} failed ({1}): {2}".format(operation, exc_name, exc)) from exc
