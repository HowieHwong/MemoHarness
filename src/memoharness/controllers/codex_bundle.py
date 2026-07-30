from __future__ import annotations

import json
import logging
from pathlib import Path
from textwrap import dedent

from ..bank.experience import ExperienceBank
from ..core.models import HarnessConfig, make_minimal_config
from ..runtime.codex_bundle import (
    ensure_codex_bundle,
    load_codex_bundle,
    resolve_codex_bundle_paths,
    restore_codex_bundle,
    snapshot_codex_bundle,
)
from .local_cli import CodexController

# Characters allowed across AGENTS.override.md + policy.json. Sized just above
# the best-generalizing bundle observed so far (~8.8k) to stop rule accretion.
_BUNDLE_CHAR_BUDGET = 9000

logger = logging.getLogger(__name__)


class CodexBundleController(CodexController):
    """Direct-edit controller for a Harbor Codex harness bundle."""

    def __init__(
        self,
        *,
        bundle_root: Path | str,
        **kwargs,
    ) -> None:
        self.bundle_root = Path(bundle_root).expanduser().resolve()
        self.bundle_paths = resolve_codex_bundle_paths(self.bundle_root)
        ensure_codex_bundle(self.bundle_root, make_minimal_config())
        super().__init__(harness_path=self.bundle_paths.agents_path, **kwargs)

    def generate_initial_harness(self, dataset: str = "") -> tuple[str, HarnessConfig]:
        fallback = self.stabilize_config(make_minimal_config())
        ensure_codex_bundle(self.bundle_root, fallback)
        if not self._direct_edit_mode_enabled():
            raise RuntimeError(
                "Codex bundle controller requires direct-edit mode. "
                "Use bypass-permission Codex CLI args so the bundle files can be updated in place."
            )
        prompt = self._build_initial_bundle_prompt(dataset or self.dataset or "terminal-bench")
        return self._run_direct_edit_harness_with_retry(
            prompt,
            fallback=fallback,
            max_retries=3,
        )

    def decide_next_harness(
        self,
        bank: ExperienceBank,
        current_code: str,
        current_config: HarnessConfig,
        iteration: int,
        min_consecutive_failures: int = 3,
    ) -> tuple[str, HarnessConfig]:
        base_config = self.stabilize_config(current_config)
        ensure_codex_bundle(self.bundle_root, base_config)
        if not self._direct_edit_mode_enabled():
            raise RuntimeError(
                "Codex bundle controller requires direct-edit mode. "
                "Use bypass-permission Codex CLI args so the bundle files can be updated in place."
            )
        prompt = self._build_iteration_bundle_prompt(
            bank=bank,
            current_code=current_code,
            current_config=base_config,
            iteration=iteration,
            min_consecutive_failures=min_consecutive_failures,
        )
        return self._run_direct_edit_harness_with_retry(
            prompt,
            fallback=base_config,
            max_retries=3,
        )

    def should_normalize_harness(self, code: str) -> bool:
        return False

    def normalize_harness_code(self, code: str, config: HarnessConfig) -> str:
        return code

    def _summary_path(self) -> Path | None:
        return self.bundle_paths.policy_path

    def _project_brief_doc(self) -> str:
        return dedent(
            """
            ## Project Brief
            MemoHarness is optimizing a Harbor Codex harness bundle instead of a free-form
            Python HarnessImpl. Harbor still runs the official Codex agent, but this bundle
            controls the iterated MemoHarness policy through:
            - AGENTS.override.md: primary task-scaffolding and execution rules
            - .memoharness/playbook.md: stable repo-level playbook
            - .memoharness/memory.md: rolling distilled memory
            - policy.json: structured D1-D6 summary plus Codex runtime hints
            """
        ).strip()

    def _direct_edit_targets_doc(self) -> str:
        lines = [
            "## Live Harness Bundle Files",
            "Modify only these Codex harness bundle files:",
            f"- {self._display_path(self.bundle_paths.agents_path)}",
            f"- {self._display_path(self.bundle_paths.policy_path)}",
            f"- {self._display_path(self.bundle_paths.memory_path)}",
            f"- {self._display_path(self.bundle_paths.playbook_path)}",
            "",
            "Do not edit any other repository files.",
            "Keep policy.json synchronized with the prompt and memory files you leave behind.",
        ]
        return "\n".join(lines).strip()

    def _direct_edit_finish_doc(self) -> str:
        return dedent(
            """
            ## Required Finish State
            Run a publish gate before finishing. At minimum:
            - verify `policy.json` is valid JSON
            - verify it contains a top-level `config` object with D1-D6 entries
            - verify AGENTS.override.md exists and points Codex at the local playbook/memory files
            - verify `.memoharness/memory.md` and `.memoharness/playbook.md` exist
            - keep the bundle concise, specific, and grounded in the current iteration evidence

            Do not stop at analysis. If a check fails, repair the bundle files and rerun the checks.

            Print a short final status when you are done. Do not wrap your answer in <python> or
            <config> tags in this mode because the controller will reload the bundle from disk.
            """
        ).strip()

    def _build_workspace_context(self, iteration: int) -> str:
        del iteration
        lines = [
            f"- workspace_root: {self._display_path(self.workspace_root)}",
            f"- dataset: {self.dataset or '(unspecified)'}",
            f"- model_label: {self.model}",
            f"- codex_bundle_root: {self._display_path(self.bundle_root)}",
            f"- codex_agents_file: {self._display_path(self.bundle_paths.agents_path)}",
            f"- codex_policy_file: {self._display_path(self.bundle_paths.policy_path)}",
        ]
        return "\n".join(lines)

    def _snapshot_live_harness_files(self) -> dict[Path, str | None]:
        return snapshot_codex_bundle(self.bundle_root)

    def _restore_live_harness_snapshot(self, snapshot: dict[Path, str | None]) -> None:
        restore_codex_bundle(snapshot)

    def _load_live_harness_state(self, fallback: HarnessConfig) -> tuple[str, HarnessConfig]:
        ensure_codex_bundle(self.bundle_root, fallback)
        preview, config = load_codex_bundle(self.bundle_root, fallback=fallback)
        return preview, self.stabilize_config(config)

    def _bundle_inspection_doc(self, iteration: int) -> str:
        del iteration
        return dedent(
            f"""
            ## Inspect First
            Start with the smallest local file set:
            - the current Codex bundle files in the target section below
            - `src/memoharness/runtime/codex_bundle.py` only when you need to confirm bundle schema/sync behavior
            - expand to `src/memoharness/harbor/loop.py` only when a concrete runtime-wiring ambiguity remains

            Do not perform broad repository inventory reads unless one specific unresolved question
            requires it.
            """
        ).strip()

    def _build_initial_bundle_prompt(self, dataset: str) -> str:
        return dedent(
            f"""
            {self._controller_identity()}

            Harbor task execution happens on Daytona cloud machines, but the Codex harness bundle
            lives in this local workspace and will be synced into the sandbox before Harbor's
            official Codex agent runs. Update the bundle files directly.

            {self._project_brief_doc()}

            ## Local Context
            {self._build_workspace_context(iteration=0)}

            {self._direct_edit_job_doc(phase='initial Codex bundle generation')}

            {self._direct_edit_targets_doc()}

            {self._bundle_inspection_doc(iteration=0)}

            {self._controller_command_scope_doc()}

            ## Objective
            Create the initial publishable Codex harness bundle for dataset "{dataset}".
            Favor focused bundle/runtime inspection, verifier-aware execution rules,
            deterministic local checks, and concise memory scaffolding.

            {self._direct_edit_finish_doc()}
            """
        ).strip()

    def _build_iteration_bundle_prompt(
        self,
        *,
        bank: ExperienceBank,
        current_code: str,
        current_config: HarnessConfig,
        iteration: int,
        min_consecutive_failures: int,
    ) -> str:
        bank_summary = self._build_iteration_bank_summary(bank, iteration)
        recent_failures = self._build_recent_failure_excerpt(bank, iteration)
        signal_summary = self._build_iteration_signal_summary(bank, iteration)
        task_history = self._build_task_history_matrix(bank, iteration)
        bundle_budget = _BUNDLE_CHAR_BUDGET
        return dedent(
            f"""
            {self._controller_identity()}

            Harbor will still run the official Codex agent. Your job is to rewrite the local
            Codex harness bundle that Harbor syncs into the sandbox before execution.

            {self._project_brief_doc()}

            ## Local Context
            {self._build_workspace_context(iteration=iteration)}

            {self._direct_edit_job_doc(phase='iteration bundle repair and improvement')}

            {self._direct_edit_targets_doc()}

            {self._bundle_inspection_doc(iteration=iteration)}

            {self._controller_command_scope_doc()}

            ## Current Iteration
            iteration={iteration}
            dataset={self.dataset or '(unspecified)'}

            ## Current Config Summary
            {json.dumps(current_config.as_dict(), indent=2)}

            ## Per-Task History Across Iterations
            {task_history}

            ## Current Iteration Experience Summary
            {bank_summary}

            ## Recent Failure Excerpts
            {recent_failures}

            ## Change-Outcome Signals
            {signal_summary}

            {self._targeted_supporting_evidence_doc()}

            {self._on_disk_state_doc()}

            ## Hard Constraints On What You May Write
            These come from measured outcomes across the iterations above, not style preference.

            1. No self-certifying markers. Never instruct the agent to print or emit a completion
               token (`..._OK`, `GATE_PASSED`, and similar). Nothing in the harness reads them.
               Every failed run recorded so far still declared itself finished, so a claim is not
               evidence. Ask for a concrete check whose output the verifier would accept instead.

            2. Stay inside the bundle budget: AGENTS.override.md plus policy.json must total at
               most {bundle_budget} characters. In this lineage the bundle that generalized best to
               held-out tasks is also the smallest one; the rounds that only added text scored the
               same or worse. If you add a rule, delete or merge another.

            3. Prefer cutting a rule to adding one. Drop anything the evidence above does not
               directly support, and say in your final status which rules you removed.

            Rewrite the Codex bundle files in place. Start from the current bundle files and the
            summarized iteration evidence below. Keep the bundle grounded, concise,
            and centered on the files that actually shape Codex runtime behavior. Before editing,
            choose 1-3 highest-priority interventions and map each to expected verifier evidence
            (file, command output, or explicit pass marker).

            {self._direct_edit_finish_doc()}
            """
        ).strip()
