from __future__ import annotations

import os
from typing import Any

from ..config.runtime import ApiModelConfig


def build_openai_client(api_config: ApiModelConfig | None = None):
    try:
        import openai
    except ImportError as exc:
        raise ImportError(
            "The 'openai' package is required. Install it with: pip install memoharness[openai]"
        ) from exc

    key_value = api_config.api_key_env if api_config else "OPENAI_API_KEY"
    if key_value.startswith(("sk-", "sk_")):
        api_key = key_value
    else:
        api_key = os.environ.get(key_value)
    if not api_key:
        raise RuntimeError(
            f"Could not resolve API key. Set the environment variable '{key_value}', "
            "or put the key directly in the config 'api_key_env' field."
        )

    kwargs: dict[str, Any] = {"api_key": api_key, "max_retries": 0}
    if api_config and api_config.api_base:
        kwargs["base_url"] = api_config.api_base
    if api_config and api_config.organization:
        kwargs["organization"] = api_config.organization
    return openai.OpenAI(**kwargs)


def _build_openai_client(api_config: ApiModelConfig | None = None):
    return build_openai_client(api_config)
