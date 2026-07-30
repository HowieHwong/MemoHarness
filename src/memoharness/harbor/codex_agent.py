"""Thin wrapper around Harbor's official Codex agent with MemoHarness bundle sync."""

from __future__ import annotations

import importlib
import logging
import os
import shlex
from pathlib import Path
from typing import TYPE_CHECKING

from memoharness.harbor.runtime_patch import apply_runtime_patch
from memoharness.runtime.codex_bundle import sync_codex_bundle_to_environment

if TYPE_CHECKING:
    from harbor.environments.base import BaseEnvironment
    from harbor.models.agent.context import AgentContext

logger = logging.getLogger(__name__)

# Harbor loads this module through ``--agent-import-path`` inside the harbor
# subprocess, which is the earliest point where we can install the runtime
# patch. Without this call the patch module is dead code: the configured
# agent/verifier timeout overrides and the deadline notice never apply.
try:
    apply_runtime_patch()
except Exception:  # pragma: no cover - never let a patch failure kill a run
    logger.exception("Failed to apply the Harbor runtime patch; continuing unpatched.")

_BUNDLE_ENV = "MEMOHARNESS_CODEX_BUNDLE_PATH"
_PROMPT_TEMPLATE_ENV = "MEMOHARNESS_CODEX_PROMPT_TEMPLATE"
_CODEX_HOME_ENV = "CODEX_HOME"
_CODEX_HOME_OVERRIDE_ENV = "MEMOHARNESS_HARBOR_CODEX_HOME"
_CODEX_EXPORT_ROOT_ENV = "MEMOHARNESS_HARBOR_CODEX_EXPORT_ROOT"
_CODEX_BASE_URL_ENV = "MEMOHARNESS_CODEX_BASE_URL"
_CODEX_PROVIDER_NAME_ENV = "MEMOHARNESS_CODEX_PROVIDER_NAME"
_CODEX_WIRE_API_ENV = "MEMOHARNESS_CODEX_WIRE_API"
_CODEX_REQUIRES_OPENAI_AUTH_ENV = "MEMOHARNESS_CODEX_REQUIRES_OPENAI_AUTH"
_CODEX_SUPPORTS_WEBSOCKETS_ENV = "MEMOHARNESS_CODEX_SUPPORTS_WEBSOCKETS"
_CODEX_AUTH_JSON_PATH_ENV = "CODEX_AUTH_JSON_PATH"
_DEFAULT_CODEX_HOME = "/logs/agent"
_DEFAULT_CODEX_EXPORT_ROOT = "/logs/agent"
_DEFAULT_CODEX_PROVIDER_NAME = "memoharness_custom"
_DEFAULT_CODEX_WIRE_API = "responses"
_RESERVED_CODEX_PROVIDER_IDS = frozenset({"openai", "ollama", "lmstudio"})
_OPTIONAL_CODEX_DB_FILES = ("logs_2.sqlite", "state_5.sqlite")


def _get_codex_base_class():
    candidates = (
        ("harbor.agents.installed.codex", "Codex"),
        ("harbor.agents.installed.codex", "CodexCLI"),
        ("harbor.agents.installed.codex_cli", "Codex"),
        ("harbor.agents.installed.codex_cli", "CodexCLI"),
    )
    for module_name, attr_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        candidate = getattr(module, attr_name, None)
        if candidate is not None:
            return candidate

    class _StubCodex:
        @staticmethod
        def name() -> str:
            return "memoharness-codex"

        async def setup(self, environment):
            return None

        async def run(self, instruction, environment, context):
            raise RuntimeError(
                "Harbor Codex agent is unavailable. Install a Harbor build that provides "
                "the official Codex installed agent."
            )

    return _StubCodex


_CodexBaseClass = _get_codex_base_class()


class _PreservePrefixModelName(str):
    """``str`` subclass that leaves ``author/model`` IDs intact when split on '/'.

    Harbor's ``Codex.run()`` does ``model = self.model_name.split("/")[-1]`` to
    strip an ``openai/<model>`` prefix before invoking ``codex exec --model``.
    That stripping is fine for the built-in OpenAI provider but breaks custom
    providers like OpenRouter, which require the full ``z-ai/glm-5`` form —
    Codex would otherwise ship ``model: "glm-5"`` to OpenRouter's Responses
    endpoint and get a 404. We monkey the model name on a per-call basis with
    an instance of this class so ``split("/")[-1]`` returns the whole string,
    while every other ``str`` operation (hashing, f-string rendering, logging,
    equality) continues to behave like a plain string.
    """

    __slots__ = ()

    def split(self, sep=None, maxsplit=-1):  # type: ignore[override]
        if sep == "/":
            return [str.__str__(self)]
        return super().split(sep, maxsplit)


def _harbor_agent_codex_home() -> str:
    try:
        module = importlib.import_module("harbor.models.trial.paths")
    except ImportError:
        return _DEFAULT_CODEX_HOME

    paths = getattr(module, "EnvironmentPaths", None)
    agent_dir = getattr(paths, "agent_dir", None) if paths is not None else None
    if agent_dir is None:
        return _DEFAULT_CODEX_HOME

    rendered = agent_dir.as_posix() if hasattr(agent_dir, "as_posix") else str(agent_dir)
    rendered = str(rendered or "").strip()
    return rendered or _DEFAULT_CODEX_HOME


def _render_codex_config(
    *,
    base_url: str,
    provider_name: str,
    api_key_env: str,
    wire_api: str,
    requires_openai_auth: bool = True,
    supports_websockets: bool | None = None,
) -> str:
    def _escape(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')

    # Mirror the user-verified local config shape: responses wire, OpenAI-style
    # auth (which makes Codex read OPENAI_API_KEY from $CODEX_HOME/auth.json or
    # the env), and disable_response_storage so Codex does not rely on server-
    # side Responses-API state that non-OpenAI providers (OpenRouter, GLM, ...)
    # don't implement.
    provider_options = (
        f'model_provider = "{_escape(provider_name)}"\n'
        "disable_response_storage = true\n"
        "\n"
        f"[model_providers.{provider_name}]\n"
        'name = "MemoHarness Custom Provider"\n'
        f'wire_api = "{_escape(wire_api)}"\n'
        f"requires_openai_auth = {'true' if requires_openai_auth else 'false'}\n"
    )
    if supports_websockets is not None:
        provider_options += (
            f"supports_websockets = {'true' if supports_websockets else 'false'}\n"
        )
    return provider_options + (
        f'base_url = "{_escape(base_url)}"\n'
        f'env_key = "{_escape(api_key_env)}"\n'
    )


def _read_optional_bool_env(name: str, *, default: bool | None) -> bool | None:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(
        f"{name}='{raw}' is not a valid boolean "
        "(use true/false, 1/0, yes/no, or on/off)."
    )


class MemoHarnessCodexAgent(_CodexBaseClass):
    """Harbor Codex wrapper that syncs an iterated MemoHarness bundle into the sandbox."""

    def __init__(self, *args, **kwargs):
        if "prompt_template_path" not in kwargs:
            env_path = str(os.environ.get(_PROMPT_TEMPLATE_ENV, "") or "").strip()
            if env_path:
                resolved = Path(env_path).expanduser().resolve()
                if resolved.is_file():
                    kwargs["prompt_template_path"] = resolved
                else:
                    logger.warning(
                        "%s=%s does not point to an existing file; skipping prompt template injection.",
                        _PROMPT_TEMPLATE_ENV,
                        env_path,
                    )
        super().__init__(*args, **kwargs)

    @staticmethod
    def name() -> str:
        return "memoharness-codex"

    def _bundle_root(self) -> Path | None:
        raw = str(os.environ.get(_BUNDLE_ENV, "") or "").strip()
        if not raw:
            return None
        return Path(raw).expanduser().resolve()

    def _sandbox_codex_home(self) -> str:
        # Harbor's Codex.run() forces env["CODEX_HOME"] = EnvironmentPaths.agent_dir
        # (/logs/agent) inside the sandbox, overriding anything we set via
        # MEMOHARNESS_HARBOR_CODEX_HOME or CODEX_HOME. The config.toml we generate
        # for a custom model_provider MUST land at that exact path, otherwise
        # Codex never sees it and falls back to the built-in OpenAI provider.
        harbor_home = _harbor_agent_codex_home()
        if harbor_home:
            return harbor_home
        override = str(os.environ.get(_CODEX_HOME_OVERRIDE_ENV, "") or "").strip()
        if override:
            return override
        existing = str(os.environ.get(_CODEX_HOME_ENV, "") or "").strip()
        if existing:
            return existing
        return _DEFAULT_CODEX_HOME

    def _sandbox_export_root(self) -> str:
        raw = str(os.environ.get(_CODEX_EXPORT_ROOT_ENV, "") or _DEFAULT_CODEX_EXPORT_ROOT).strip()
        return raw or _DEFAULT_CODEX_EXPORT_ROOT

    async def _run_environment_command(
        self,
        environment: "BaseEnvironment",
        command: str,
        *,
        description: str,
    ) -> None:
        result = await environment.exec(command)
        rendered = result if isinstance(result, str) else str(result)
        if "[bash error]" in rendered.lower():
            raise RuntimeError(f"{description} failed: {rendered}")

    def _prepare_log_capture_command(self) -> str:
        export_root = self._sandbox_export_root()
        codehome = self._sandbox_codex_home()
        return f"""python - <<'PY'
from pathlib import Path
import shutil

export_root = Path({export_root!r})
codehome = Path({codehome!r})
export_root.mkdir(parents=True, exist_ok=True)
codehome.mkdir(parents=True, exist_ok=True)

for relative in ("skills", ".tmp", "setup", "sessions"):
    path = export_root / relative
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)

for relative in ("plugins.sha", "installation_id", "logs_2.sqlite", "state_5.sqlite"):
    path = export_root / relative
    if path.exists() or path.is_symlink():
        path.unlink(missing_ok=True)
PY"""

    def _export_selected_logs_command(self) -> str:
        export_root = self._sandbox_export_root()
        codehome = self._sandbox_codex_home()
        optional_dbs = tuple(_OPTIONAL_CODEX_DB_FILES)
        return f"""python - <<'PY'
from pathlib import Path
import shutil

export_root = Path({export_root!r})
codehome = Path({codehome!r})
export_root.mkdir(parents=True, exist_ok=True)

for relative in ("skills", ".tmp", "setup", "plugins.sha", "installation_id"):
    path = export_root / relative
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)

sessions_src = codehome / "sessions"
sessions_dst = export_root / "sessions"
same_tree = sessions_src.resolve() == sessions_dst.resolve()
if not same_tree:
    if sessions_dst.exists() or sessions_dst.is_symlink():
        if sessions_dst.is_symlink() or sessions_dst.is_file():
            sessions_dst.unlink(missing_ok=True)
        else:
            shutil.rmtree(sessions_dst)
    if sessions_src.exists():
        for src in sessions_src.rglob("rollout-*.jsonl"):
            dst = sessions_dst / src.relative_to(sessions_src)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

for name in {optional_dbs!r}:
    src = codehome / name
    dst = export_root / name
    if src.resolve() == dst.resolve():
        continue
    if dst.exists() or dst.is_symlink():
        dst.unlink(missing_ok=True)
    if src.exists():
        shutil.copy2(src, dst)
PY"""

    async def _prepare_log_capture(self, environment: "BaseEnvironment") -> None:
        await self._run_environment_command(
            environment,
            self._prepare_log_capture_command(),
            description="prepare minimal Codex log capture",
        )

    async def _export_selected_logs(self, environment: "BaseEnvironment") -> None:
        await self._run_environment_command(
            environment,
            self._export_selected_logs_command(),
            description="export selected Codex logs",
        )

    async def _sync_bundle(self, environment: "BaseEnvironment") -> None:
        bundle_root = self._bundle_root()
        if bundle_root is None:
            logger.info("%s not set; skipping Codex bundle sync.", _BUNDLE_ENV)
            return
        logger.info("Syncing Codex harness bundle from %s", bundle_root)
        await sync_codex_bundle_to_environment(environment, bundle_root)

    def _configured_base_url(self) -> str:
        return str(os.environ.get(_CODEX_BASE_URL_ENV, "") or "").strip()

    def _configured_provider_name(self) -> str:
        raw = str(os.environ.get(_CODEX_PROVIDER_NAME_ENV, "") or "").strip()
        name = raw or _DEFAULT_CODEX_PROVIDER_NAME
        if name in _RESERVED_CODEX_PROVIDER_IDS:
            raise RuntimeError(
                f"MEMOHARNESS_CODEX_PROVIDER_NAME='{name}' collides with a built-in Codex "
                "provider id; pick a different name."
            )
        return name

    def _configured_wire_api(self) -> str:
        raw = str(os.environ.get(_CODEX_WIRE_API_ENV, "") or "").strip().lower()
        if raw and raw not in {"chat", "responses"}:
            raise RuntimeError(
                f"MEMOHARNESS_CODEX_WIRE_API='{raw}' is not a valid Codex wire_api "
                "(must be 'chat' or 'responses')."
            )
        return raw or _DEFAULT_CODEX_WIRE_API

    def _configured_requires_openai_auth(self) -> bool:
        value = _read_optional_bool_env(
            _CODEX_REQUIRES_OPENAI_AUTH_ENV,
            default=True,
        )
        return bool(value)

    def _configured_supports_websockets(self) -> bool | None:
        return _read_optional_bool_env(
            _CODEX_SUPPORTS_WEBSOCKETS_ENV,
            default=None,
        )

    async def _write_codex_config(
        self,
        environment: "BaseEnvironment",
        *,
        base_url: str,
        provider_name: str,
        wire_api: str,
        requires_openai_auth: bool,
        supports_websockets: bool | None,
    ) -> None:
        codex_home = self._sandbox_codex_home()
        api_key_env = "OPENAI_API_KEY"
        config_text = _render_codex_config(
            base_url=base_url,
            provider_name=provider_name,
            api_key_env=api_key_env,
            wire_api=wire_api,
            requires_openai_auth=requires_openai_auth,
            supports_websockets=supports_websockets,
        )
        quoted_home = shlex.quote(codex_home)
        quoted_path = shlex.quote(f"{codex_home}/config.toml")
        command = (
            f"mkdir -p {quoted_home} && "
            f"cat > {quoted_path} <<'EOF'\n"
            f"{config_text}"
            "EOF"
        )
        await self._run_environment_command(
            environment,
            command,
            description="write Codex config.toml",
        )
        logger.info(
            "Wrote Codex config.toml to %s/config.toml "
            "(provider=%s, base_url=%s, wire_api=%s, requires_openai_auth=%s, "
            "supports_websockets=%s)",
            codex_home,
            provider_name,
            base_url,
            wire_api,
            requires_openai_auth,
            supports_websockets,
        )

    async def _prepare_codex_provider(self, environment: "BaseEnvironment") -> None:
        base_url = self._configured_base_url()
        if not base_url:
            return
        provider_name = self._configured_provider_name()
        wire_api = self._configured_wire_api()
        requires_openai_auth = self._configured_requires_openai_auth()
        supports_websockets = self._configured_supports_websockets()
        os.environ.pop(_CODEX_AUTH_JSON_PATH_ENV, None)
        os.environ[_CODEX_HOME_ENV] = self._sandbox_codex_home()
        await self._write_codex_config(
            environment,
            base_url=base_url,
            provider_name=provider_name,
            wire_api=wire_api,
            requires_openai_auth=requires_openai_auth,
            supports_websockets=supports_websockets,
        )

    async def setup(self, environment: "BaseEnvironment") -> None:
        await self._sync_bundle(environment)
        await self._prepare_log_capture(environment)
        await self._prepare_codex_provider(environment)
        parent_setup = getattr(super(), "setup", None)
        if callable(parent_setup):
            await parent_setup(environment)

    async def run(
        self,
        instruction: str,
        environment: "BaseEnvironment",
        context: "AgentContext",
    ):
        await self._sync_bundle(environment)
        await self._prepare_log_capture(environment)
        await self._prepare_codex_provider(environment)
        parent_run = getattr(super(), "run", None)
        if not callable(parent_run):
            raise RuntimeError("Harbor Codex base agent does not expose an async run() method.")

        # Harbor's Codex.run() strips an ``author/`` prefix from ``model_name``
        # before passing ``--model`` to the CLI. When a custom base_url is
        # wired up (OpenRouter, bigmodel, etc.), we need the full ``z-ai/glm-5``
        # style ID to survive. Temporarily rebind ``self.model_name`` to a
        # subclass whose ``split("/")`` is a no-op.
        original_model_name = getattr(self, "model_name", None)
        swapped_model_name = False
        if (
            self._configured_base_url()
            and original_model_name
            and "/" in str(original_model_name)
            and not isinstance(original_model_name, _PreservePrefixModelName)
        ):
            try:
                self.model_name = _PreservePrefixModelName(str(original_model_name))
                swapped_model_name = True
            except Exception:
                logger.exception(
                    "Failed to preserve full model name '%s'; Harbor may strip the provider prefix.",
                    original_model_name,
                )

        try:
            return await parent_run(instruction, environment, context)
        finally:
            if swapped_model_name:
                try:
                    self.model_name = original_model_name
                except Exception:
                    pass
            await self._export_selected_logs(environment)
