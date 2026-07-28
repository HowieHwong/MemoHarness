from __future__ import annotations

import ast
import json
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from textwrap import dedent

from ..bank.experience import ExperienceBank
from ..config.runtime import ApiModelConfig
from ..core.models import BenchmarkCase, ControllerDecision, HarnessConfig, make_minimal_config
from .llm_controller import (
    LLMController,
    _CONFIG_ONLY_DOC,
    _DIMENSIONS_DOC,
    _HARNESS_INTERFACE_DOC,
    _PYTHON_OUTPUT_DOC,
    _get_validation_error,
)

logger = logging.getLogger(__name__)

# The full set of per-iteration failure analyses is ~5k characters (~1.2k tokens)
# for a 33-task split, so there is no reason to clip it to 4 cases of 180 chars --
# that threw away almost all of the controller's evidence.
_PROMPT_FAILURE_ENTRY_LIMIT = 40
_PROMPT_SIGNAL_ENTRY_LIMIT = 40
_PROMPT_FAILURE_ANALYSIS_LIMIT = 2000
_PROMPT_SIGNAL_ANALYSIS_LIMIT = 2000
_PROMPT_CODE_PREVIEW_LIMIT = 12000
_PROMPT_RETRY_ERROR_LIMIT = 1200
_CONTROLLER_FAILURE_TAIL_BYTES = 131072
_CONTROLLER_FAILURE_TAIL_LINES = 80
_CONTROLLER_FAILURE_SIGNAL_LINES = 16
_CONTROLLER_FAILURE_SUMMARY_LIMIT = 6000


def _windows_path_to_wsl(path: str | Path) -> str:
    raw = str(path).replace("\\", "/")
    if len(raw) >= 2 and raw[1] == ":" and raw[0].isalpha():
        drive = raw[0].lower()
        tail = raw[2:].lstrip("/")
        return f"/mnt/{drive}/{tail}" if tail else f"/mnt/{drive}"
    return raw


def _should_prefer_wsl_for_codex() -> bool:
    return bool(os.name == "nt" and shutil.which("wsl.exe"))


class ClaudeCodeController:
    """Controller that shells out to a local CLI controller."""

    def __init__(
        self,
        *,
        command: str = "cc",
        args: list[str] | None = None,
        use_stdin: bool = False,
        timeout_seconds: float = 900.0,
        recent_iterations: int = 2,
        workspace_root: Path | str | None = None,
        jobs_dir: Path | str | None = None,
        run_id: str = "",
        harness_path: Path | str | None = None,
        bank_path: Path | str | None = None,
        artifact_dir: Path | str | None = None,
        dataset: str = "",
        model: str = "claude-code",
        api_config: ApiModelConfig | None = None,
        env: dict[str, str] | None = None,
        heartbeat_seconds: int = 30,
        controller_label: str = "Claude Code",
        prefer_wsl: bool = False,
    ) -> None:
        self.command = command
        self.args = list(args or ["-p", "{prompt}"])
        self.env = dict(env) if env else None
        self.use_stdin = use_stdin
        self.timeout_seconds = float(timeout_seconds)
        self.recent_iterations = max(1, int(recent_iterations))
        self.workspace_root = Path(workspace_root or Path.cwd()).resolve()
        self.jobs_dir = Path(jobs_dir).resolve() if jobs_dir else None
        self.run_id = run_id
        self.harness_path = Path(harness_path).resolve() if harness_path else None
        self.bank_path = Path(bank_path).resolve() if bank_path else None
        self.artifact_dir = Path(artifact_dir).resolve() if artifact_dir else None
        self.dataset = dataset
        self.model = model
        self.api_config = api_config
        self.heartbeat_seconds = max(0, int(heartbeat_seconds))
        self.controller_label = controller_label
        self._run_in_wsl = bool(prefer_wsl and os.name == "nt" and shutil.which("wsl.exe"))
        self._command_counter = 0
        self._helper = LLMController(client=None, model=model, api_config=api_config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_initial_harness(self, dataset: str = "") -> tuple[str, HarnessConfig]:
        if self._direct_edit_mode_enabled():
            prompt = self._build_initial_direct_edit_prompt(dataset or self.dataset or "terminal-bench")
            return self._run_direct_edit_harness_with_retry(
                prompt,
                fallback=make_minimal_config(),
                max_retries=3,
            )
        prompt = self._build_initial_prompt(dataset or self.dataset or "terminal-bench")
        code, config_dict = self._call_harness_with_retry(prompt, max_retries=3)
        config = self.stabilize_config(
            self._build_harness_config(config_dict, make_minimal_config())
        )
        return code, config

    def decide_next_harness(
        self,
        bank: ExperienceBank,
        current_code: str,
        current_config: HarnessConfig,
        iteration: int,
        min_consecutive_failures: int = 3,
    ) -> tuple[str, HarnessConfig]:
        base_config = self.stabilize_config(current_config)
        if self._direct_edit_mode_enabled():
            prompt = self._build_iteration_direct_edit_prompt(
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
        prompt = self._build_iteration_prompt(
            bank=bank,
            current_code=current_code,
            current_config=base_config,
            iteration=iteration,
            min_consecutive_failures=min_consecutive_failures,
        )
        new_code, config_dict = self._call_harness_with_retry(prompt, max_retries=3)
        next_config = self.stabilize_config(self._build_harness_config(config_dict, base_config))
        return new_code, next_config

    def decide_next_config(
        self,
        bank: ExperienceBank,
        current_config: HarnessConfig,
        iteration: int,
    ) -> ControllerDecision:
        _, next_config = self.decide_next_harness(
            bank=bank,
            current_code=self._render_harness_code(current_config),
            current_config=current_config,
            iteration=iteration,
        )
        return ControllerDecision(
            config=next_config,
            rationale=f"{self.controller_label} controller proposed the next stable harness configuration.",
            variants=self._helper._build_variants(next_config),
        )

    def adapt_for_case(
        self,
        bank: ExperienceBank,
        case: BenchmarkCase,
        base_config: HarnessConfig,
    ) -> ControllerDecision:
        config = self.stabilize_config(base_config)
        prompt = dedent(
            f"""
            You are adapting a MemoHarness configuration for one evaluation case.

            Workspace root: {self._display_path(self.workspace_root)}
            Dataset: {self.dataset or '(unspecified)'}
            Bank path: {self._display_path(self.bank_path)}

            You may inspect local files, but do not modify any files.
            Return only concise JSON updates inside <config> tags.

            ## Target Case
            case_id={case.case_id}
            prompt={case.prompt}
            features={json.dumps(vars(case.features), indent=2)}

            ## Base Dimension Summary
            {json.dumps(config.as_dict(), indent=2)}

            ## Similar Cases Summary
            {self._helper._build_case_summary(bank, case)}

            Only adjust dimensions when the similar-case evidence clearly justifies it.

            {_CONFIG_ONLY_DOC.strip()}
            """
        ).strip()
        config_updates = self._call_llm_for_config(prompt)
        next_config = self.stabilize_config(self._build_harness_config(config_updates, config))
        return ControllerDecision(
            config=next_config,
            rationale=f"{self.controller_label} controller adapted the harness configuration for this case.",
        )

    def stabilize_config(self, config: HarnessConfig | None = None) -> HarnessConfig:
        return self._helper.stabilize_config(config)

    def should_normalize_harness(self, code: str) -> bool:
        return self._helper.should_normalize_harness(code)

    def normalize_harness_code(self, code: str, config: HarnessConfig) -> str:
        return self._helper.normalize_harness_code(code, config)

    # ------------------------------------------------------------------
    # Compatibility helpers reused by harbor_loop publish gate
    # ------------------------------------------------------------------

    def _call_llm_for_harness(self, prompt: str) -> tuple[str, dict]:
        text = self._run_controller_command(prompt)
        return self._helper._parse_harness_response(text)

    def _call_llm_for_config(self, prompt: str) -> dict:
        text = self._run_controller_command(prompt)
        return self._helper._parse_config_response(text)

    def _build_harness_config(
        self, config_dict: dict, fallback: HarnessConfig | None = None
    ) -> HarnessConfig:
        return self._helper._build_harness_config(config_dict, fallback)

    def _render_harness_code(self, config: HarnessConfig) -> str:
        return self._helper._render_harness_code(config)

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    def _direct_edit_mode_enabled(self) -> bool:
        if self.harness_path is None:
            return False

        for index, raw_arg in enumerate(self.args):
            arg = str(raw_arg).strip()
            if arg in {
                "--dangerously-skip-permissions",
                "--dangerously-bypass-approvals-and-sandbox",
            }:
                return True
            if arg.startswith("--permission-mode="):
                return arg.partition("=")[2].strip() == "bypassPermissions"
            if arg == "--permission-mode" and index + 1 < len(self.args):
                return str(self.args[index + 1]).strip() == "bypassPermissions"
        return False

    def _display_path(self, path: Path | str | None) -> str:
        if path is None:
            return "(none)"
        raw = str(path)
        return _windows_path_to_wsl(raw) if self._run_in_wsl else raw

    def _controller_identity(self) -> str:
        return f"You are {self.controller_label} acting as the MemoHarness controller from the local workspace."

    def _summary_path(self) -> Path | None:
        if self.harness_path is None:
            return None
        return self.harness_path.with_suffix(".json")

    def _truncate_for_prompt(self, text: str, *, limit: int) -> str:
        content = str(text or "").strip()
        if not content:
            return "(empty)"
        if len(content) <= limit:
            return content
        reserve = min(96, max(24, limit // 8))
        clip_limit = max(1, limit - reserve)
        clipped = content[:clip_limit].rstrip()
        omitted = max(0, len(content) - len(clipped))
        return f"{clipped}\n...[truncated {omitted} chars]"

    def _extract_error_signal_lines(self, text: str, *, max_lines: int) -> list[str]:
        markers = (
            "error",
            "exception",
            "traceback",
            "failed",
            "fail",
            "timed out",
            "timeout",
            "request too large",
            "tokens per min",
            "tpm",
            "stream disconnected",
            "reconnecting",
            "rate limit",
            "invalid",
            "not found",
            "no such file",
            "can't read",
            "permission denied",
        )
        selected: list[str] = []
        seen: set[str] = set()
        for raw_line in str(text or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            lower = line.lower()
            if not any(marker in lower for marker in markers):
                continue
            compact = line if len(line) <= 320 else f"{line[:317]}..."
            if compact in seen:
                continue
            selected.append(compact)
            seen.add(compact)
            if len(selected) >= max_lines:
                break
        return selected

    def _read_text_tail(self, path: Path, *, max_bytes: int) -> str:
        try:
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                if size <= 0:
                    return ""
                read_bytes = min(size, max_bytes)
                offset = max(0, size - read_bytes)
                handle.seek(offset, os.SEEK_SET)
                data = handle.read(read_bytes)
        except OSError:
            return ""
        return data.decode("utf-8", errors="replace")

    def _summarize_controller_failure_output(self, output_log_path: Path) -> str:
        tail_text = self._read_text_tail(
            output_log_path,
            max_bytes=_CONTROLLER_FAILURE_TAIL_BYTES,
        )
        if not tail_text.strip():
            return "(empty output)"

        tail_lines = [line.rstrip() for line in tail_text.splitlines() if line.strip()]
        tail_excerpt = "\n".join(tail_lines[-_CONTROLLER_FAILURE_TAIL_LINES:])
        signal_lines = self._extract_error_signal_lines(
            tail_text,
            max_lines=_CONTROLLER_FAILURE_SIGNAL_LINES,
        )

        parts: list[str] = []
        if signal_lines:
            parts.append("Key error lines:\n" + "\n".join(signal_lines))
        if tail_excerpt:
            parts.append("Tail excerpt:\n" + tail_excerpt)
        combined = "\n\n".join(parts).strip() or "(empty output)"
        return self._truncate_for_prompt(combined, limit=_CONTROLLER_FAILURE_SUMMARY_LIMIT)

    def _summarize_error_for_prompt(
        self,
        exc: Exception,
        *,
        limit: int = _PROMPT_RETRY_ERROR_LIMIT,
    ) -> str:
        raw = str(exc or "").strip()
        if not raw:
            return "(empty)"
        signal_lines = self._extract_error_signal_lines(raw, max_lines=12)
        if signal_lines:
            header = raw.splitlines()[0].strip()
            ordered = [header] if header and header not in signal_lines else []
            ordered.extend(signal_lines)
            return self._truncate_for_prompt("\n".join(ordered), limit=limit)
        return self._truncate_for_prompt(raw, limit=limit)

    def _project_brief_doc(self) -> str:
        return dedent(
            """
            ## Project Brief
            MemoHarness is a Python training and evaluation framework that improves a live
            HarnessImpl for Harbor and Terminal-Bench style tasks. The live harness is the agent
            implementation Harbor executes on benchmark cases, and the companion JSON file is the
            persisted D1-D6 control summary that should stay aligned with that live harness.
            """
        ).strip()

    def _direct_edit_job_doc(self, *, phase: str) -> str:
        return dedent(
            f"""
            ## Your Job
            You are the harness maintainer for the {phase} phase of this run.
            Do the work directly in the workspace instead of returning a proposed patch.
            Understand how the current harness is loaded and executed, repair or improve the live
            harness, and leave the live harness files in a publishable state before you finish.
            """
        ).strip()

    def _direct_edit_inspection_doc(self, iteration: int) -> str:
        if iteration > 0:
            iteration_scope = (
                "- the summarized current-iteration evidence provided below\n"
                f"- one or two targeted synced outputs under jobs/<run_id>/iter-{iteration:02d}* "
                "or artifacts/<run_id>/ when a specific hypothesis needs confirmation"
            )
        else:
            iteration_scope = (
                "- any seed artifacts already present for this run, but only if the focused local "
                "reads leave a concrete ambiguity"
            )

        return dedent(
            f"""
            ## Inspect First
            Before editing, inspect a focused local file set first:
            - AGENTS.md for repository-specific working rules
            - README.md and configs/experiment.json for project purpose and run configuration
            - the current live harness .py/.json files
            - the code paths most likely to explain how the harness is loaded, run, validated, or archived, starting with:
              src/memoharness/harbor/loop.py
              src/memoharness/harbor/agent.py
              src/memoharness/controllers/local_cli.py
            {iteration_scope}

            Default to the smallest file set that can explain the required harness edit.
            Expand to other repository files only after the focused reads above leave a concrete gap.
            Prefer targeted file opens or path-scoped searches over repository-wide inventory
            commands. Prefer the current iteration's synced outputs over older iterations unless you
            have a concrete reason to widen the search.
            Base edits on the local repository state and synced artifacts, not on generic assumptions.
            """
        ).strip()

    def _direct_edit_targets_doc(self) -> str:
        harness_path = self._display_path(self.harness_path)
        summary_path = self._display_path(self._summary_path())
        return dedent(
            f"""
            ## Live Harness Files
            Modify only these live harness files:
            - {harness_path}
            - {summary_path}

            Do not edit any other repository files.
            Keep the companion JSON summary synchronized with the live Python harness.
            """
        ).strip()

    def _direct_edit_finish_doc(self) -> str:
        return dedent(
            """
            ## Required Finish State
            Run a publish gate before finishing. At minimum:
            - verify the live harness file parses/compiles as valid Python
            - verify imports/bootstrap assumptions that are required for HarnessImpl.setup/run
            - run a small targeted smoke test if the local workspace makes that practical
            - confirm the harness is not a no-op or actionless placeholder
            - confirm the companion JSON summary still matches the live harness behavior
            - if a check fails, repair the live harness files and rerun the checks

            Do not stop at analysis. If a check fails, keep repairing until the harness is
            publishable or you exhaust the retry budget enforced by the caller.

            ## Done Criteria
            The task is only complete when the live harness files are updated in place, the publish
            gate has been run, and you can briefly report what you checked and why the harness is
            ready.

            Print a short final status when you are done. Do not wrap your answer in <python> or
            <config> tags in this mode because the controller will reload the files from disk.
            """
        ).strip()

    def _on_disk_state_doc(self) -> str:
        return dedent(
            """
            ## Current Live State
            Treat the target files already on disk as the source of truth for this retry/edit cycle.
            Inspect them directly before editing. Inline full-file previews are intentionally omitted
            here to keep the controller prompt concise.
            """
        ).strip()

    def _controller_command_scope_doc(self) -> str:
        return dedent(
            """
            ## Per-Turn Command Scope (Hard Limits)
            Your backing model has a 40,000 tokens-per-minute cap. A single tool output that
            streams a large file overflows the next request and crashes this controller run.
            Treat these as hard constraints for every shell turn:
            - Cap any single `sed -n '1,Np'`, `head -n N`, `tail -n N`, or `cat` at N <= 200 lines.
              For larger regions, split across multiple turns using narrow line ranges.
            - Restrict `find`, `rg`, `grep -r` to one shard or one task directory per command.
              Never root a recursive search at the full `jobs/`, `artifacts/`, or repository tree
              in a single turn. Always pass `--max-count`, a file glob, or `| head -n 50` to bound
              the match volume.
            - Never run wildcard repository-wide searches such as `rg -n "<term>" -g '*'`.
              Always provide a concrete scoped path for `rg`/`grep` (single task, single shard,
              or one explicit file).
            - Do NOT `cat` session logs, `run.log`, `transcript.jsonl`, or any raw per-case output
              under `jobs/<run>/iter-*/` or `artifacts/<run>/`. Individual session files can exceed
              100k tokens per case on livecodebench and will exceed the TPM cap in one turn. If you
              must open one, pre-filter with `grep -n <keyword>` and then read a narrow line window
              with `sed -n 'A,Bp'`.
            - Do NOT read aggregated `console.log` files (for example `jobs/<run>/console.log`).
              If log evidence is required, inspect at most one specific task directory and only use
              bounded probes (`grep -n -m 20` plus one narrow `sed -n 'A,Bp'` window).
            - Do NOT open prior `controller/call-*.output.log` or `controller/call-*.prompt.txt`
              end-to-end. If needed, use `grep -n -m 20` and at most one narrow `sed -n 'A,Bp'`
              window.
            - Never chain unrelated reads with `&&` or `;` in the same command. One scoped read
              per turn so the model budget is predictable.
            - To aggregate across shards, iterate one shard per turn, or write a small summary
              to a workspace temp file and read only the summary back in a later turn.
            Target well under 5,000 tokens of tool output per turn; if a read returns more, stop
            and narrow the query instead of accepting the overflow.
            """
        ).strip()

    def _targeted_supporting_evidence_doc(self) -> str:
        return dedent(
            """
            ## Evidence Strategy
            Start from the current live files, the focused runtime files above, and the summarized
            iteration evidence below.
            Use synced artifacts or job outputs selectively when they sharpen a concrete
            verifier-facing hypothesis, confirm an exact detail, or explain a specific failure.
            Never use aggregated run-level `console.log` as primary evidence.
            If logs are necessary, inspect one task-specific artifact only.
            Prefer one narrow log or result file over broad sweeps of logs, session transcripts,
            whole run directories, or repository-wide inventories.
            """
        ).strip()

    def _build_direct_edit_retry_prompt(
        self,
        *,
        attempt: int,
        max_retries: int,
        failure: Exception,
    ) -> str:
        lines = [
            "## Retry Context",
            f"- attempt: {attempt}/{max_retries}",
            f"- workspace_root: {self._display_path(self.workspace_root)}",
            f"- dataset: {self.dataset or '(unspecified)'}",
        ]
        if self.bank_path:
            lines.append(f"- bank_path: {self._display_path(self.bank_path)}")
        if self.artifact_dir:
            lines.append(f"- artifact_run_dir: {self._display_path(self.artifact_dir)}")
        retry_context = "\n".join(lines)
        failure_summary = self._summarize_error_for_prompt(failure)
        return dedent(
            f"""
            {self._controller_identity()}

            {self._project_brief_doc()}

            {retry_context}

            ## Last Failure
            {failure_summary}

            Reload the current workspace state from disk before editing again. Do not start from
            scratch if the target files already contain useful partial fixes.

            {self._controller_command_scope_doc()}

            {self._on_disk_state_doc()}

            {self._direct_edit_targets_doc()}

            {self._direct_edit_finish_doc()}
            """
        ).strip()

    def _dimension_action_playbook_doc(self) -> str:
        return dedent(
            """
            ## Dimension-to-Action Playbook
            Use the dominant failure dimension to select concrete edits:
            - D1: tighten instruction scaffolding and target extraction from verifier assertions.
            - D2: harden environment/bootstrap/tooling commands and offline-safe install paths.
            - D3: adjust generation controls only when outputs are truncated or unstable.
            - D4: improve loop orchestration, retries, and stagnation handling.
            - D5: change memory window/compression only when context-loss is evidenced.
            - D6: strengthen success criteria and guard against false-positive "pass" states.

            Avoid broad rewrites. Prefer 1-3 targeted interventions linked to evidence.
            """
        ).strip()

    def _build_initial_direct_edit_prompt(self, dataset: str) -> str:
        return dedent(
            f"""
            {self._controller_identity()}

            Harbor task execution happens on Daytona cloud machines, but job outputs and the
            experience bank are synced back locally. Use focused local repository reads and synced
            artifacts to update the live harness files directly.

            {self._project_brief_doc()}

            ## Local Context
            {self._build_workspace_context(iteration=0)}

            {self._direct_edit_job_doc(phase='initial harness generation')}

            {self._direct_edit_inspection_doc(iteration=0)}

            {self._controller_command_scope_doc()}

            {self._direct_edit_targets_doc()}

            ## Objective
            Create the initial publishable HarnessImpl for dataset "{dataset}".
            Favor focused runtime inspection, verifier-aware loops, and deterministic stopping.

            {_HARNESS_INTERFACE_DOC.strip()}

            {_DIMENSIONS_DOC.strip()}

            {self._direct_edit_finish_doc()}
            """
        ).strip()

    def _build_iteration_direct_edit_prompt(
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
        return dedent(
            f"""
            {self._controller_identity()}

            Harbor task execution happens on Daytona cloud machines, but job outputs and the
            experience bank are synced back locally. Use focused local repository reads and synced
            artifacts before rewriting the live harness files directly.

            {self._project_brief_doc()}

            ## Local Context
            {self._build_workspace_context(iteration=iteration)}

            {self._direct_edit_job_doc(phase='iteration repair and improvement')}

            {self._direct_edit_inspection_doc(iteration=iteration)}

            {self._controller_command_scope_doc()}

            {self._direct_edit_targets_doc()}

            ## Current Iteration
            iteration={iteration}
            dataset={self.dataset or '(unspecified)'}

            ## Current Config Summary
            {json.dumps(current_config.as_dict(), indent=2)}

            ## Current Iteration Experience Summary
            {bank_summary}

            ## Recent Failure Excerpts
            {recent_failures}

            ## Change-Outcome Signals
            {signal_summary}

            {self._targeted_supporting_evidence_doc()}

            {self._on_disk_state_doc()}

            Rewrite the live harness to address the observed failures using the summary above and
            focused local evidence. Reach for synced artifacts or job outputs when they resolve a
            specific producer, verifier, or environment question. Favor concrete producer
            discovery, verifier-aware stopping, and deterministic offline-safe behavior when the
            local evidence points there.
            Before editing, choose 1-3 highest-priority interventions and map each to expected
            verifier evidence (file, command output, or explicit pass marker).

            {_HARNESS_INTERFACE_DOC.strip()}

            {_DIMENSIONS_DOC.strip()}
            
            {self._dimension_action_playbook_doc()}

            {self._direct_edit_finish_doc()}
            """
        ).strip()

    def _build_initial_prompt(self, dataset: str) -> str:
        return dedent(
            f"""
            {self._controller_identity()}

            Harbor task execution happens on Daytona cloud machines, but job outputs and the
            experience bank are synced back locally. Use focused local repository reads and synced
            artifacts to design a better harness. Do not modify any files. Print only the new
            harness and config using the required tags.

            ## Local Context
            {self._build_workspace_context(iteration=0)}

            ## Objective
            Create the initial HarnessImpl for dataset "{dataset}".
            Favor focused runtime inspection, verifier-aware loops, and deterministic stopping.

            {_HARNESS_INTERFACE_DOC.strip()}

            {_DIMENSIONS_DOC.strip()}

            {_PYTHON_OUTPUT_DOC.strip()}
            """
        ).strip()

    def _build_iteration_prompt(
        self,
        *,
        bank: ExperienceBank,
        current_code: str,
        current_config: HarnessConfig,
        iteration: int,
        min_consecutive_failures: int,
    ) -> str:
        bank_summary = self._helper._build_bank_summary(
            bank,
            iteration,
            min_consecutive_failures=min_consecutive_failures,
        )
        recent_failures = self._build_recent_failure_excerpt(bank, iteration)
        signal_summary = self._build_iteration_signal_summary(bank, iteration)
        current_code_preview = self._truncate_for_prompt(current_code, limit=_PROMPT_CODE_PREVIEW_LIMIT)
        return dedent(
            f"""
            {self._controller_identity()}

            Harbor task execution happens on Daytona cloud machines, but job outputs and the
            experience bank are synced back locally. Use focused local repository reads and synced
            artifacts before rewriting the harness. Do not modify any files. Print only the new
            harness and config using the required tags.

            ## Local Context
            {self._build_workspace_context(iteration=iteration)}

            ## Current Iteration
            iteration={iteration}
            dataset={self.dataset or '(unspecified)'}

            ## Current Config Summary
            {json.dumps(current_config.as_dict(), indent=2)}

            ## Experience Bank Summary
            {bank_summary}

            ## Recent Failure Excerpts
            {recent_failures}

            ## Change-Outcome Signals
            {signal_summary}

            {self._targeted_supporting_evidence_doc()}

            ## Current Harness
            ```python
            {current_code_preview}
            ```

            Rewrite the harness to address the observed failures using the summary above and
            focused local evidence. Reach for synced artifacts or job outputs when they resolve a
            specific producer, verifier, or environment question. Favor concrete producer
            discovery, verifier-aware stopping, and deterministic offline-safe behavior when the
            local evidence points there.
            Before generating code, choose 1-3 highest-priority interventions and map each to
            expected verifier evidence (file, command output, or explicit pass marker).

            {_HARNESS_INTERFACE_DOC.strip()}

            {_DIMENSIONS_DOC.strip()}

            {self._dimension_action_playbook_doc()}

            {_PYTHON_OUTPUT_DOC.strip()}
            """
        ).strip()

    def _build_workspace_context(self, iteration: int) -> str:
        lines = [
            f"- workspace_root: {self._display_path(self.workspace_root)}",
            f"- dataset: {self.dataset or '(unspecified)'}",
            f"- model_label: {self.model}",
        ]
        if self.harness_path:
            lines.append(f"- live_harness_path: {self._display_path(self.harness_path)}")
        if self.bank_path:
            lines.append(f"- bank_path: {self._display_path(self.bank_path)}")
        if self.artifact_dir:
            lines.append(f"- artifact_run_dir: {self._display_path(self.artifact_dir)}")

        supporting_context_parts: list[str] = []
        if self.artifact_dir:
            supporting_context_parts.append(self._display_path(self.artifact_dir))
        if self.jobs_dir and self.run_id and iteration > 0:
            supporting_context_parts.append(
                self._display_path(self.jobs_dir / self.run_id / f"iter-{iteration:02d}*")
            )
        if supporting_context_parts:
            lines.append(
                "- supporting_run_context: targeted follow-up reads are available under "
                + " and ".join(supporting_context_parts)
            )

        lines.append(
            "- note: only local synced artifacts are available; you cannot inspect live Daytona "
            "containers directly."
        )
        return "\n".join(lines)

    def _build_iteration_bank_summary(self, bank: ExperienceBank, iteration: int) -> str:
        entries = [entry for entry in bank.entries if entry.iteration == iteration]
        if not entries:
            return f"No experience data recorded for iteration {iteration}."

        successes = sum(1 for entry in entries if entry.diagnosis.success)
        failures = len(entries) - successes
        parts = [
            f"Iteration {iteration}: {len(entries)} entries — {successes} successes, "
            f"{failures} failures"
        ]

        dim_counts: dict[str, int] = {}
        for entry in entries:
            if entry.diagnosis.success:
                continue
            primary_dim = entry.diagnosis.diagnostic_signal.primary_dim
            dim_counts[primary_dim] = dim_counts.get(primary_dim, 0) + 1

        if dim_counts:
            parts.append(f"Failure dimension distribution: {json.dumps(dim_counts)}")
        else:
            parts.append("All recorded entries in this iteration succeeded.")

        return "\n".join(parts)

    def _build_task_history_matrix(
        self,
        bank: ExperienceBank,
        iteration: int,
        max_columns: int = 15,
    ) -> str:
        """Per-task solved/failed across every iteration recorded so far.

        Without this the controller only ever sees the current iteration and
        cannot tell "I just broke this task" from "this task has never passed".
        The summary line at the bottom is the important part: it says which
        tasks can actually respond to a harness change.
        """
        by_task: dict[str, dict[int, float]] = {}
        blocked: dict[tuple[str, int], bool] = {}
        for entry in bank.entries:
            by_task.setdefault(entry.case_id, {})[entry.iteration] = entry.primary_reward
            analysis = str(entry.diagnosis.analysis or "")
            blocked[(entry.case_id, entry.iteration)] = analysis.startswith("External blocker")
        if not by_task:
            return "No per-task history recorded yet."

        iterations = sorted({it for row in by_task.values() for it in row})
        if len(iterations) > max_columns:
            iterations = iterations[-max_columns:]

        header = "".join("%3d" % it for it in iterations)
        lines = ["%-38s%s   solved" % ("task", header)]
        always, never, flipping = [], [], []
        for task in sorted(by_task):
            row = by_task[task]
            cells = ""
            observed = 0
            solved = 0
            for it in iterations:
                value = row.get(it)
                if value is None:
                    cells += "  ."
                    continue
                if blocked.get((task, it)):
                    # Sandbox/credit/startup failure: the harness never got a
                    # fair shot, so it must not be read as a capability signal.
                    cells += "  x"
                    continue
                observed += 1
                if value >= 1:
                    solved += 1
                    cells += "  1"
                else:
                    cells += "  0"
            lines.append("%-38s%s   %d/%d" % (task, cells, solved, observed))
            if observed == 0:
                continue
            if solved == observed:
                always.append(task)
            elif solved == 0:
                never.append(task)
            else:
                flipping.append(task)

        lines.append("")
        lines.append(
            "Legend: 1=solved 0=failed x=external blocker (sandbox/credits/startup, "
            "not a harness signal) .=no result recorded."
        )
        lines.append(
            "%d task(s) always solved (insensitive to harness changes), "
            "%d never solved (likely capability or environment limits), "
            "%d flipping." % (len(always), len(never), len(flipping))
        )
        if flipping:
            lines.append(
                "Only the flipping tasks can respond to a harness edit, so weigh them "
                "highest: " + ", ".join(sorted(flipping))
            )
        if never:
            lines.append(
                "Never solved in any iteration - do not spend the bundle's budget on "
                "these unless the evidence shows a fixable harness cause: "
                + ", ".join(sorted(never))
            )
        return "\n".join(lines)

    def _build_recent_failure_excerpt(
        self,
        bank: ExperienceBank,
        iteration: int,
        max_entries: int = _PROMPT_FAILURE_ENTRY_LIMIT,
    ) -> str:
        recent_failures = [
            entry
            for entry in bank.entries
            if entry.iteration == iteration and not entry.diagnosis.success
        ]
        if not recent_failures:
            return f"No failing entries recorded for iteration {iteration}."

        lines: list[str] = []
        for entry in recent_failures[-max_entries:]:
            analysis = self._truncate_for_prompt(
                entry.diagnosis.analysis,
                limit=_PROMPT_FAILURE_ANALYSIS_LIMIT,
            )
            lines.append(
                f"case={entry.case_id} iter={entry.iteration} "
                f"primary_dim={entry.diagnosis.diagnostic_signal.primary_dim} "
                f"analysis={analysis}"
            )
        return "\n".join(lines)

    def _build_iteration_signal_summary(
        self,
        bank: ExperienceBank,
        iteration: int,
        max_entries: int = _PROMPT_SIGNAL_ENTRY_LIMIT,
    ) -> str:
        entries = [entry for entry in bank.entries if entry.iteration == iteration]
        if not entries:
            return f"No per-case signal records available for iteration {iteration}."

        selected = entries[-max_entries:]
        lines: list[str] = []
        for entry in selected:
            change_parts: list[str] = []
            for dim, updates in sorted(entry.delta_from_prev.items()):
                if not isinstance(updates, dict) or not updates:
                    continue
                keys = [key for key in updates.keys() if key != "_removed"]
                if "_removed" in updates:
                    keys.append("_removed")
                if keys:
                    change_parts.append(f"{dim}:{','.join(sorted(keys))}")
            change_text = "; ".join(change_parts) if change_parts else "(no config delta)"
            trajectory = entry.trajectory
            analysis = self._truncate_for_prompt(
                entry.diagnosis.analysis,
                limit=_PROMPT_SIGNAL_ANALYSIS_LIMIT,
            )
            behaviour = next(
                (
                    str(item)
                    for item in (trajectory.intermediate_outputs or [])
                    if str(item).startswith("behaviour:")
                ),
                "",
            )
            lines.append(
                f"case={entry.case_id} reward={entry.primary_reward:.2f} "
                f"primary_dim={entry.diagnosis.diagnostic_signal.primary_dim} "
                f"changes={change_text} calls={trajectory.num_llm_calls} "
                f"tokens={trajectory.total_tokens} tools={len(trajectory.tools_invoked)} "
                + (f"{behaviour} " if behaviour else "")
                + f"analysis={analysis}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------

    def _snapshot_live_harness_files(self) -> dict[Path, str | None]:
        snapshot: dict[Path, str | None] = {}
        for path in (self.harness_path, self._summary_path()):
            if path is None:
                continue
            if path.exists():
                snapshot[path] = path.read_text(encoding="utf-8")
            else:
                snapshot[path] = None
        return snapshot

    def _restore_live_harness_snapshot(self, snapshot: dict[Path, str | None]) -> None:
        for path, content in snapshot.items():
            if content is None:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    logger.debug("Could not remove restored direct-edit file %s.", path, exc_info=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

    def _extract_embedded_config_dict(self, code: str) -> dict | None:
        try:
            module = ast.parse(code)
        except SyntaxError:
            return None

        for node in module.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(
                isinstance(target, ast.Name) and target.id == "_HARNESS_CONFIG"
                for target in node.targets
            ):
                continue
            value = node.value
            if not (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and isinstance(value.func.value, ast.Name)
                and value.func.value.id == "json"
                and value.func.attr == "loads"
                and value.args
                and isinstance(value.args[0], ast.Constant)
                and isinstance(value.args[0].value, str)
            ):
                continue
            try:
                parsed = json.loads(value.args[0].value)
            except json.JSONDecodeError:
                return None
            return parsed if isinstance(parsed, dict) else None

        return None

    def _load_live_harness_state(self, fallback: HarnessConfig) -> tuple[str, HarnessConfig]:
        if self.harness_path is None:
            raise RuntimeError("Direct-edit mode requires a live harness path.")
        if not self.harness_path.exists():
            raise RuntimeError(
                f"{self.controller_label} direct-edit mode did not leave a live harness at {self.harness_path}."
            )

        code = self.harness_path.read_text(encoding="utf-8")
        validation_err = _get_validation_error(code)
        if validation_err is not None:
            raise ValueError(validation_err)

        config_dict: dict | None = None
        summary_path = self._summary_path()
        if summary_path is not None and summary_path.exists():
            try:
                raw = json.loads(summary_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning(
                    "Direct-edit summary %s was not valid JSON; falling back to embedded config.",
                    summary_path,
                )
            else:
                if isinstance(raw, dict):
                    config_dict = raw.get("config", raw) if isinstance(raw.get("config", raw), dict) else raw

        if config_dict is None:
            config_dict = self._extract_embedded_config_dict(code)

        config = self._build_harness_config(config_dict or {}, fallback)
        return code, self.stabilize_config(config)

    def _run_direct_edit_harness_with_retry(
        self,
        prompt: str,
        *,
        fallback: HarnessConfig,
        max_retries: int = 3,
    ) -> tuple[str, HarnessConfig]:
        last_exc: Exception | None = None
        retry_prompt = prompt
        snapshot = self._snapshot_live_harness_files()

        for attempt in range(1, max_retries + 1):
            try:
                self._run_controller_command(retry_prompt)
                return self._load_live_harness_state(fallback)
            except Exception as exc:
                last_exc = exc
                retry_prompt = self._build_direct_edit_retry_prompt(
                    attempt=attempt,
                    max_retries=max_retries,
                    failure=exc,
                )

        self._restore_live_harness_snapshot(snapshot)
        raise RuntimeError(
            f"{self.controller_label} direct-edit harness generation failed after "
            f"{max_retries} attempts. Last error: {last_exc}"
        )

    def _call_harness_with_retry(
        self,
        prompt: str,
        *,
        max_retries: int = 3,
    ) -> tuple[str, dict]:
        last_exc: Exception | None = None
        retry_prompt = prompt
        for attempt in range(1, max_retries + 1):
            try:
                code, config_dict = self._call_llm_for_harness(retry_prompt)
            except Exception as exc:
                last_exc = exc
                failure_summary = self._summarize_error_for_prompt(exc)
                retry_prompt = dedent(
                    f"""
                    {prompt}

                    Your previous response could not be used by the controller.
                    Attempt: {attempt}/{max_retries}
                    Failure: {failure_summary}

                    Return only:
                    1. the complete HarnessImpl inside <python> tags
                    2. the D1-D6 config summary inside <config> tags
                    """
                ).strip()
                continue

            validation_err = _get_validation_error(code)
            if validation_err is None:
                return code, config_dict

            last_exc = ValueError(validation_err)
            retry_prompt = dedent(
                f"""
                {prompt}

                Your previous harness failed controller-side validation.
                Attempt: {attempt}/{max_retries}
                Validation error: {validation_err}

                Fix the issue and return only:
                1. the complete HarnessImpl inside <python> tags
                2. the D1-D6 config summary inside <config> tags
                """
            ).strip()

        raise RuntimeError(
            f"{self.controller_label} harness generation failed after {max_retries} "
            f"attempts. Last error: {last_exc}"
        )

    def _controller_log_dir(self) -> Path:
        if self.artifact_dir is not None:
            base_dir = self.artifact_dir
        elif self.jobs_dir is not None and self.run_id:
            base_dir = self.jobs_dir / self.run_id
        else:
            base_dir = self.workspace_root / ".memoharness-controller"
        log_dir = base_dir / "controller"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    def _run_controller_command(self, prompt: str) -> str:
        placeholders = {
            "{prompt}": prompt,
            "{workspace_root}": self._display_path(self.workspace_root),
            "{jobs_dir}": self._display_path(self.jobs_dir) if self.jobs_dir else "",
            "{run_id}": self.run_id,
            "{dataset}": self.dataset,
        }

        prompt_file_path = ""
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".txt",
            delete=False,
        ) as handle:
            handle.write(prompt)
            prompt_file_path = handle.name
        placeholders["{prompt_file}"] = self._display_path(prompt_file_path)

        self._command_counter += 1
        call_id = f"call-{self._command_counter:03d}"
        log_dir = self._controller_log_dir()
        prompt_copy_path = log_dir / f"{call_id}.prompt.txt"
        output_log_path = log_dir / f"{call_id}.output.log"
        command_meta_path = log_dir / f"{call_id}.command.json"
        prompt_copy_path.write_text(prompt, encoding="utf-8")

        command = [self.command]
        for raw_arg in self.args:
            arg = raw_arg
            for needle, value in placeholders.items():
                arg = arg.replace(needle, value)
            command.append(arg)
        command = self._wrap_command_for_runtime(command)

        merged_env: dict[str, str] | None = None
        if self.env:
            merged_env = dict(os.environ)
            merged_env.update(self.env)

        started_at = time.time()
        command_meta = {
            "call_id": call_id,
            "command": command,
            "cwd": str(self.workspace_root),
            "timeout_seconds": self.timeout_seconds,
            "prompt_file": prompt_copy_path.name,
            "output_log": output_log_path.name,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started_at)),
            "returncode": None,
        }
        command_meta_path.write_text(json.dumps(command_meta, indent=2), encoding="utf-8")
        logger.info(
            "Controller %s started via %s. Log: %s",
            call_id,
            self.command,
            output_log_path,
        )

        try:
            with output_log_path.open(
                "w",
                encoding="utf-8",
                errors="replace",
                buffering=1,
            ) as output_handle:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE if self.use_stdin else None,
                    stdout=output_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=str(self.workspace_root),
                    env=merged_env,
                )
                if self.use_stdin and process.stdin is not None:
                    process.stdin.write(prompt)
                    process.stdin.close()

                next_heartbeat = started_at + self.heartbeat_seconds
                while True:
                    returncode = process.poll()
                    elapsed = time.time() - started_at
                    if returncode is not None:
                        break
                    if elapsed > self.timeout_seconds:
                        if hasattr(process, "kill"):
                            process.kill()
                        if hasattr(process, "wait"):
                            process.wait()
                        raise RuntimeError(
                            f"{self.controller_label} command timed out after "
                            f"{self.timeout_seconds:.0f}s. Log: {output_log_path}"
                        )
                    if self.heartbeat_seconds > 0 and time.time() >= next_heartbeat:
                        logger.info(
                            "Controller %s still running after %.0fs. Log: %s",
                            call_id,
                            elapsed,
                            output_log_path,
                        )
                        next_heartbeat = time.time() + self.heartbeat_seconds
                    time.sleep(1.0)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"{self.controller_label} command not found: {self.command}. "
                "Set harness.controller_command/controller_args to a valid local or WSL command."
            ) from exc
        finally:
            try:
                Path(prompt_file_path).unlink(missing_ok=True)
            except OSError:
                logger.debug("Could not delete temporary controller prompt file.", exc_info=True)

        elapsed = time.time() - started_at
        command_meta["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
        command_meta["elapsed_seconds"] = round(elapsed, 1)
        command_meta["returncode"] = process.returncode
        command_meta_path.write_text(json.dumps(command_meta, indent=2), encoding="utf-8")
        logger.info(
            "Controller %s finished in %.1fs. Log: %s",
            call_id,
            elapsed,
            output_log_path,
        )

        if process.returncode != 0:
            failure_summary = self._summarize_controller_failure_output(output_log_path)
            raise RuntimeError(
                f"{self.controller_label} command failed with exit code {process.returncode}. "
                f"Log: {output_log_path}\nSTDOUT/STDERR summary:\n{failure_summary}"
            )
        combined_output = output_log_path.read_text(encoding="utf-8", errors="replace").strip()
        if not combined_output:
            raise RuntimeError(f"{self.controller_label} command returned no output. Log: {output_log_path}")
        return combined_output

    def _wrap_command_for_runtime(self, command: list[str]) -> list[str]:
        if not self._run_in_wsl:
            return command
        shell_command = " ".join(shlex.quote(part) for part in command)
        workspace = self._display_path(self.workspace_root)
        return ["wsl.exe", "bash", "-lc", f"cd {shlex.quote(workspace)} && {shell_command}"]


class CodexController(ClaudeCodeController):
    _LOCAL_PROFILE_LABEL = "local Codex CLI profile"
    _WSL_PROFILE_LABEL = "WSL Codex CLI profile"

    def __init__(
        self,
        *,
        command: str = "codex",
        args: list[str] | None = None,
        use_stdin: bool = False,
        timeout_seconds: float = 900.0,
        recent_iterations: int = 2,
        workspace_root: Path | str | None = None,
        jobs_dir: Path | str | None = None,
        run_id: str = "",
        harness_path: Path | str | None = None,
        bank_path: Path | str | None = None,
        artifact_dir: Path | str | None = None,
        dataset: str = "",
        model: str = "codex",
        api_config: ApiModelConfig | None = None,
        env: dict[str, str] | None = None,
        heartbeat_seconds: int = 30,
    ) -> None:
        normalized_args, normalized_use_stdin = self._normalize_codex_invocation(args, use_stdin)
        prefer_wsl = _should_prefer_wsl_for_codex()
        profile_label = self._WSL_PROFILE_LABEL if prefer_wsl else self._LOCAL_PROFILE_LABEL
        super().__init__(
            command=command or "codex",
            args=normalized_args,
            use_stdin=normalized_use_stdin,
            timeout_seconds=timeout_seconds,
            recent_iterations=recent_iterations,
            workspace_root=workspace_root,
            jobs_dir=jobs_dir,
            run_id=run_id,
            harness_path=harness_path,
            bank_path=bank_path,
            artifact_dir=artifact_dir,
            dataset=dataset,
            # The Codex CLI should use its own local profile and auth inside the
            # selected runtime instead of inheriting MemoHarness model/provider settings.
            model=profile_label,
            api_config=None,
            env=None,
            heartbeat_seconds=heartbeat_seconds,
            controller_label="Codex",
            prefer_wsl=prefer_wsl,
        )

    @staticmethod
    def _normalize_codex_invocation(
        args: list[str] | None,
        use_stdin: bool,
    ) -> tuple[list[str], bool]:
        raw_args = [str(arg) for arg in (args or [])]
        if not raw_args or CodexController._looks_like_claude_code_invocation(raw_args):
            return [
                "exec",
                "--dangerously-bypass-approvals-and-sandbox",
                "--skip-git-repo-check",
                "--color",
                "never",
                "-",
            ], True

        filtered: list[str] = []
        index = 0
        force_stdin = bool(use_stdin)
        while index < len(raw_args):
            arg = raw_args[index]
            next_arg = raw_args[index + 1] if index + 1 < len(raw_args) else None
            if arg == "-p" and next_arg == "{prompt}":
                filtered.append("-")
                force_stdin = True
                index += 2
                continue
            if arg == "--output-format":
                index += 2 if next_arg is not None else 1
                continue
            if arg == "--dangerously-skip-permissions":
                filtered.append("--dangerously-bypass-approvals-and-sandbox")
                index += 1
                continue
            if arg.startswith("--permission-mode="):
                if arg.partition("=")[2].strip() == "bypassPermissions":
                    filtered.append("--dangerously-bypass-approvals-and-sandbox")
                index += 1
                continue
            if arg == "--permission-mode":
                if next_arg == "bypassPermissions":
                    filtered.append("--dangerously-bypass-approvals-and-sandbox")
                index += 2 if next_arg is not None else 1
                continue
            filtered.append(arg)
            index += 1

        if not filtered or filtered[0] != "exec":
            filtered = ["exec", *filtered]

        head = ["exec"]
        tail = filtered[1:]
        if "--dangerously-bypass-approvals-and-sandbox" not in tail:
            head.append("--dangerously-bypass-approvals-and-sandbox")
        if "--skip-git-repo-check" not in tail:
            head.append("--skip-git-repo-check")
        if "--color" not in tail and not any(arg.startswith("--color=") for arg in tail):
            head.extend(["--color", "never"])

        normalized = [*head, *tail]
        if "-" in normalized:
            force_stdin = True
        elif "{prompt}" not in normalized:
            if force_stdin:
                normalized.append("-")
            else:
                normalized.append("{prompt}")
        return normalized, force_stdin

    @staticmethod
    def _looks_like_claude_code_invocation(args: list[str]) -> bool:
        return any(
            arg in {"-p", "--dangerously-skip-permissions", "--output-format"}
            or arg.startswith("--permission-mode=")
            for arg in args
        )
