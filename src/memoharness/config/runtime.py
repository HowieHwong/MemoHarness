from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_BEST_HARNESS_SELECTION_MODES = frozenset({"mean_reward", "perfect_success_count"})


@dataclass
class ApiModelConfig:
    provider: str
    model: str
    api_key_env: str
    api_base: Optional[str] = None
    organization: Optional[str] = None
    connection_retry_limit: int = -1
    temperature: float = 0.0
    max_tokens: int = 2048
    top_p: float = 1.0
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)
    wire_api: Optional[str] = None
    codex_provider_name: Optional[str] = None


@dataclass
class ProviderConfig:
    client: str = "openai"
    api_key_env: str = "OPENAI_API_KEY"
    api_base: Optional[str] = None
    organization: Optional[str] = None
    connection_retry_limit: int = -1
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)
    wire_api: Optional[str] = None
    codex_provider_name: Optional[str] = None


@dataclass
class ModelSelectionConfig:
    provider: str = "default"
    name: str = "gpt-4.1-mini"
    temperature: float = 0.0
    max_tokens: int = 2048
    top_p: float = 1.0
    judge_model: Optional[str] = None
    diagnostics_model: Optional[str] = None
    controller_model: Optional[str] = None


@dataclass
class HarnessSettings:
    scorer: str = "llm_judge"
    embedder: str = "openai"
    diagnostics: str = "llm"
    tool_protocol: str = "native"
    agent_runtime: str = "harbor_codex"
    codex_bundle_path: str = "configs/harness_codex"
    controller: str = "codex"
    controller_command: str = "codex"
    controller_args: list[str] = field(
        default_factory=lambda: [
            "exec",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "--color",
            "never",
            "-",
        ]
    )
    controller_use_stdin: bool = True
    controller_timeout_seconds: float = 900.0
    controller_recent_iterations: int = 2
    controller_env: dict[str, str] = field(default_factory=dict)
    controller_canary_enabled: bool = True
    controller_canary_task_count: int = 6
    controller_canary_min_reward_delta: float = -0.34
    controller_canary_max_blocker_increase: int = 0
    controller_canary_tasks: list[str] = field(default_factory=list)
    best_harness_selection_modes: list[str] = field(default_factory=lambda: ["mean_reward"])
    best_harness_iteration_floor: int = 0
    test_time_case_adaptation: bool = False
    iterations: int = 8
    primary_metric: str = "accuracy"
    judge_model: str = "gpt-4.1-mini"
    diagnostics_model: str = "gpt-4.1-mini"
    controller_model: str = "gpt-4.1-mini"
    cases_path: str = "examples/demo_cases.json"

    def __post_init__(self) -> None:
        self.tool_protocol = str(self.tool_protocol or "native").strip().lower()
        if self.tool_protocol not in {"native", "bash_tags"}:
            raise ValueError("harness.tool_protocol must be one of: native, bash_tags")

        self.agent_runtime = str(self.agent_runtime or "harbor_codex").strip().lower()
        if self.agent_runtime != "harbor_codex":
            raise ValueError("harness.agent_runtime only supports 'harbor_codex'")

        self.controller = str(self.controller or "codex").strip().lower()
        if self.controller != "codex":
            raise ValueError("harness.controller only supports 'codex'")

        raw_modes = self.best_harness_selection_modes
        if raw_modes is None:
            normalized_modes = ["mean_reward"]
        elif isinstance(raw_modes, str):
            normalized_modes = [raw_modes]
        else:
            normalized_modes = list(raw_modes)

        deduped_modes: list[str] = []
        seen_modes: set[str] = set()
        for raw_mode in normalized_modes:
            mode = str(raw_mode or "").strip().lower()
            if not mode:
                continue
            if mode not in _BEST_HARNESS_SELECTION_MODES:
                allowed = ", ".join(sorted(_BEST_HARNESS_SELECTION_MODES))
                raise ValueError(
                    "harness.best_harness_selection_modes must contain only: "
                    f"{allowed}"
                )
            if mode in seen_modes:
                continue
            deduped_modes.append(mode)
            seen_modes.add(mode)

        if not deduped_modes:
            deduped_modes = ["mean_reward"]
        self.best_harness_selection_modes = deduped_modes


@dataclass
class DaytonaConfig:
    api_keys: list[str] = field(default_factory=list)
    per_key_concurrency: int = 3
    max_tasks_per_shard: int = 3
    assignment_strategy: str = "current"
    exclusive_task_ids: list[str] = field(default_factory=list)
    allow_retries: bool = True
    relocate_on_daytona_error: bool = True
    stop_on_timeout_per_task: bool = False
    shard_timeout_seconds: float = 1800.0
    shard_kill_grace_seconds: float = 5.0
    completed_exit_grace_seconds: float = 90.0
    agent_timeout_watchdog_grace_seconds: float = 600.0
    agent_timeout_verify_grace_seconds: float = 1800.0
    disk_limit_retry_limit: int = -1
    retry_wait_seconds: int = 60
    connectivity_retry_limit: int = 3
    timeout_retry_limit: int = 3
    key_cooldown_seconds: int = 120
    cleanup_sandboxes_before_run: bool = True
    cleanup_timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        self.max_tasks_per_shard = int(self.max_tasks_per_shard)
        if self.max_tasks_per_shard <= 0:
            raise ValueError("experiment.daytona.max_tasks_per_shard must be positive")
        self.assignment_strategy = str(self.assignment_strategy or "current").strip().lower()
        if self.assignment_strategy not in {"current", "strict_fill", "strict_balance"}:
            raise ValueError(
                "experiment.daytona.assignment_strategy must be one of: "
                "current, strict_fill, strict_balance"
            )
        raw_exclusive_task_ids = self.exclusive_task_ids
        if raw_exclusive_task_ids is None:
            raw_exclusive_task_ids = []
        self.exclusive_task_ids = list(
            dict.fromkeys(
                str(task_id).strip().split("__", 1)[0]
                for task_id in raw_exclusive_task_ids
                if str(task_id).strip()
            )
        )
        self.relocate_on_daytona_error = bool(self.relocate_on_daytona_error)
        self.shard_timeout_seconds = float(self.shard_timeout_seconds)
        if self.shard_timeout_seconds <= 0:
            raise ValueError("experiment.daytona.shard_timeout_seconds must be positive")
        self.shard_kill_grace_seconds = float(self.shard_kill_grace_seconds)
        if self.shard_kill_grace_seconds <= 0:
            raise ValueError("experiment.daytona.shard_kill_grace_seconds must be positive")
        self.completed_exit_grace_seconds = float(self.completed_exit_grace_seconds)
        if self.completed_exit_grace_seconds < 0:
            raise ValueError(
                "experiment.daytona.completed_exit_grace_seconds must be >= 0"
            )
        self.agent_timeout_watchdog_grace_seconds = float(
            self.agent_timeout_watchdog_grace_seconds
        )
        if self.agent_timeout_watchdog_grace_seconds < 0:
            raise ValueError(
                "experiment.daytona.agent_timeout_watchdog_grace_seconds must be >= 0"
            )
        self.agent_timeout_verify_grace_seconds = float(
            self.agent_timeout_verify_grace_seconds
        )
        if self.agent_timeout_verify_grace_seconds < 0:
            raise ValueError(
                "experiment.daytona.agent_timeout_verify_grace_seconds must be >= 0"
            )


@dataclass
class ExperimentConfig:
    dataset: str = "terminal-bench-sample@2.0"
    agent_import_path: str = "memoharness.harbor.codex_agent:MemoHarnessCodexAgent"
    run_id: Optional[str] = None
    iterations: int = 5
    harness_path: str = "configs/harness_terminal.py"
    bank_dir: str = "artifacts"
    jobs_dir: Optional[str] = None
    distill_every: int = 5
    min_consecutive_failures: int = 3
    train_split: float = 0.8
    train_task_limit: Optional[int] = None
    seed: int = 42
    n_concurrent: int = 3
    eval_only: bool = False
    eval_after_train: bool = False
    console_mode: str = "normal"
    console_heartbeat_seconds: int = 30
    # Global Harbor timeout overrides applied via MemoHarness runtime patching.
    harbor_agent_timeout_seconds: Optional[float] = None
    verifier_timeout_seconds: Optional[float] = None
    disable_harbor_verifier_retry: bool = False
    extra_harbor_args: list[str] = field(default_factory=list)
    daytona: DaytonaConfig = field(default_factory=DaytonaConfig)

    def __post_init__(self) -> None:
        if self.daytona is None:
            self.daytona = DaytonaConfig()
        elif isinstance(self.daytona, dict):
            self.daytona = DaytonaConfig(**self.daytona)
        if self.train_task_limit is not None and self.train_task_limit <= 0:
            raise ValueError("experiment.train_task_limit must be a positive integer")
        self.eval_only = bool(self.eval_only)
        self.eval_after_train = bool(self.eval_after_train)
        if self.harbor_agent_timeout_seconds is not None:
            self.harbor_agent_timeout_seconds = float(self.harbor_agent_timeout_seconds)
        if self.verifier_timeout_seconds is not None:
            self.verifier_timeout_seconds = float(self.verifier_timeout_seconds)
        self.disable_harbor_verifier_retry = bool(self.disable_harbor_verifier_retry)
        self.console_mode = str(self.console_mode or "normal").lower()
        if self.console_mode not in {"quiet", "normal", "debug"}:
            raise ValueError("experiment.console_mode must be one of: quiet, normal, debug")
        if int(self.console_heartbeat_seconds) < 0:
            raise ValueError("experiment.console_heartbeat_seconds must be >= 0")
        self.console_heartbeat_seconds = int(self.console_heartbeat_seconds)


@dataclass
class MemoHarnessRuntimeConfig:
    llm: ApiModelConfig
    model: Optional[ModelSelectionConfig] = None
    active_model: Optional[str] = None
    models: dict[str, ModelSelectionConfig] = field(default_factory=dict)
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    harness: HarnessSettings = field(default_factory=HarnessSettings)

    def __post_init__(self) -> None:
        if self.model is None:
            self.model = ModelSelectionConfig(
                provider=self.llm.provider,
                name=self.llm.model,
                temperature=self.llm.temperature,
                max_tokens=self.llm.max_tokens,
                top_p=self.llm.top_p,
            )
        if not self.active_model:
            self.active_model = self.model.name
        if self.active_model not in self.models:
            self.models[self.active_model] = self.model
        if self.model.provider not in self.providers:
            self.providers[self.model.provider] = ProviderConfig(
                client=self.llm.provider,
                api_key_env=self.llm.api_key_env,
                api_base=self.llm.api_base,
                organization=self.llm.organization,
                connection_retry_limit=self.llm.connection_retry_limit,
                extra_headers=dict(self.llm.extra_headers),
                extra_body=dict(self.llm.extra_body),
                wire_api=self.llm.wire_api,
                codex_provider_name=self.llm.codex_provider_name,
            )

    @classmethod
    def from_json_file(cls, path: Path) -> "MemoHarnessRuntimeConfig":
        payload = json.loads(path.read_text())
        raw_harness = dict(payload.get("harness", {}))
        raw_harness.pop("controller_candidates_per_iter", None)
        harness = HarnessSettings(**raw_harness)
        experiment = ExperimentConfig(**payload.get("experiment", {}))

        if "llm" in payload:
            llm = ApiModelConfig(**payload["llm"])
            model_payload = payload.get("model")
            model = (
                ModelSelectionConfig(**model_payload)
                if isinstance(model_payload, dict)
                else ModelSelectionConfig(
                    provider=llm.provider,
                    name=llm.model,
                    temperature=llm.temperature,
                    max_tokens=llm.max_tokens,
                    top_p=llm.top_p,
                )
            )
            providers = {
                name: ProviderConfig(**raw_cfg)
                for name, raw_cfg in payload.get("providers", {}).items()
            }
            return cls(
                llm=llm,
                model=model,
                active_model=payload.get("active_model"),
                models={
                    name: ModelSelectionConfig(**raw_cfg)
                    for name, raw_cfg in payload.get("models", {}).items()
                },
                providers=providers,
                experiment=experiment,
                harness=harness,
            )

        providers = {
            name: ProviderConfig(**raw_cfg)
            for name, raw_cfg in payload.get("providers", {}).items()
        }
        model_catalog = {
            name: ModelSelectionConfig(**raw_cfg)
            for name, raw_cfg in payload.get("models", {}).items()
        }
        active_model = payload.get("active_model")
        model_payload = payload.get("model")

        if model_catalog:
            if not active_model:
                if len(model_catalog) == 1:
                    active_model = next(iter(model_catalog))
                else:
                    raise ValueError(
                        "Runtime config with a 'models' block must also set 'active_model'."
                    )
            model = model_catalog.get(active_model)
            if model is None:
                raise ValueError(
                    f"Unknown active model '{active_model}'. "
                    f"Available models: {', '.join(sorted(model_catalog)) or '(none)'}"
                )
        elif isinstance(model_payload, dict):
            model = ModelSelectionConfig(**model_payload)
        else:
            raise ValueError(
                "Runtime config must include either an 'llm' block, a 'model' block, "
                "or a 'models' block with 'active_model'."
            )

        provider_cfg = providers.get(model.provider)
        if provider_cfg is None:
            raise ValueError(
                f"Unknown model provider '{model.provider}'. "
                f"Available providers: {', '.join(sorted(providers)) or '(none)'}"
            )
        if model.judge_model and "judge_model" not in raw_harness:
            harness.judge_model = model.judge_model
        if model.diagnostics_model and "diagnostics_model" not in raw_harness:
            harness.diagnostics_model = model.diagnostics_model
        if model.controller_model and "controller_model" not in raw_harness:
            harness.controller_model = model.controller_model
        llm = ApiModelConfig(
            provider=provider_cfg.client,
            model=model.name,
            api_key_env=provider_cfg.api_key_env,
            api_base=provider_cfg.api_base,
            organization=provider_cfg.organization,
            connection_retry_limit=provider_cfg.connection_retry_limit,
            temperature=model.temperature,
            max_tokens=model.max_tokens,
            top_p=model.top_p,
            extra_headers=dict(provider_cfg.extra_headers),
            extra_body=dict(provider_cfg.extra_body),
            wire_api=provider_cfg.wire_api,
            codex_provider_name=provider_cfg.codex_provider_name,
        )
        return cls(
            llm=llm,
            model=model,
            active_model=active_model,
            models=model_catalog,
            providers=providers,
            experiment=experiment,
            harness=harness,
        )
