"""Harbor Codex adapter for TokenRouter's HTTPS Responses endpoint.

TokenRouter exposes the OpenAI Responses API over HTTPS but does not accept
Codex's Responses-over-WebSocket transport.  Harbor's built-in Codex agent
uses the stock OpenAI provider whenever ``OPENAI_BASE_URL`` is set, and that
provider advertises WebSocket support.  This adapter keeps the built-in Codex
agent unchanged apart from registering a custom provider with WebSockets
disabled.
"""

from __future__ import annotations

import json
import shlex

from harbor.agents.installed.codex import Codex


class TokenRouterCodexAgent(Codex):
    """Run Harbor's official Codex agent through TokenRouter over HTTPS."""

    @staticmethod
    def name() -> str:
        return "tokenrouter-codex"

    def _build_register_mcp_servers_command(self) -> str | None:
        commands: list[str] = []
        mcp_command = super()._build_register_mcp_servers_command()
        if mcp_command:
            commands.append(mcp_command)

        base_url = (
            self._get_env("TOKENROUTER_BASE_URL")
            or self._get_env("OPENAI_BASE_URL")
            or ""
        ).strip().rstrip("/")
        if not base_url:
            raise ValueError(
                "TOKENROUTER_BASE_URL or OPENAI_BASE_URL must be set for "
                "TokenRouterCodexAgent."
            )

        # JSON strings are valid TOML basic strings and safely escape any
        # unusual characters in a caller-provided base URL.
        config = (
            "\nmodel_provider = \"tokenrouter\"\n"
            "[model_providers.tokenrouter]\n"
            "name = \"TokenRouter\"\n"
            f"base_url = {json.dumps(base_url)}\n"
            "env_key = \"OPENAI_API_KEY\"\n"
            "wire_api = \"responses\"\n"
            "requires_openai_auth = false\n"
            "supports_websockets = false\n"
        )
        commands.append(
            f"printf %s {shlex.quote(config)} >> \"$CODEX_HOME/config.toml\""
        )
        return "\n".join(commands)
