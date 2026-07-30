from __future__ import annotations

import json
import logging
import shlex
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..core.models import GlobalPattern, HarnessConfig, make_minimal_config

logger = logging.getLogger(__name__)

_AGENTS_FILENAME = "AGENTS.override.md"
_POLICY_FILENAME = "policy.json"
_INTERNAL_DIRNAME = ".memoharness"
_MEMORY_FILENAME = "memory.md"
_PLAYBOOK_FILENAME = "playbook.md"
_POLICY_READ_LINE = (
    "- Read `./policy.json` for the authoritative D1-D6 summary before using playbook or memory notes."
)
_POLICY_SUPPORT_LINE = (
    "- Treat `./.memoharness/playbook.md` as stable repo heuristics and "
    "`./.memoharness/memory.md` as rolling distilled notes."
)


@dataclass(frozen=True)
class CodexBundlePaths:
    root: Path
    agents_path: Path
    policy_path: Path
    memory_path: Path
    playbook_path: Path

    def tracked_files(self) -> tuple[Path, ...]:
        return (
            self.agents_path,
            self.policy_path,
            self.memory_path,
            self.playbook_path,
        )


def resolve_codex_bundle_paths(root: Path | str) -> CodexBundlePaths:
    bundle_root = Path(root).expanduser().resolve()
    internal_dir = bundle_root / _INTERNAL_DIRNAME
    return CodexBundlePaths(
        root=bundle_root,
        agents_path=bundle_root / _AGENTS_FILENAME,
        policy_path=bundle_root / _POLICY_FILENAME,
        memory_path=internal_dir / _MEMORY_FILENAME,
        playbook_path=internal_dir / _PLAYBOOK_FILENAME,
    )


def _pattern_payload(pattern: GlobalPattern) -> dict[str, object]:
    return {
        "pattern_id": pattern.pattern_id,
        "primary_dim": pattern.primary_dim,
        "description": pattern.description,
        "effect": pattern.effect,
        "evidence": list(pattern.evidence),
        "iteration_range": list(pattern.iteration_range),
    }


def _policy_payload(
    config: HarnessConfig,
    *,
    distilled_patterns: Iterable[GlobalPattern] = (),
    updated_iteration: int | None = None,
) -> dict[str, object]:
    d3 = dict(config.D3)
    pattern_payloads = [_pattern_payload(pattern) for pattern in distilled_patterns]
    payload: dict[str, object] = {
        "bundle_version": 1,
        "config": config.as_dict(),
        "runtime_files": {
            "agents": _AGENTS_FILENAME,
            "policy": _POLICY_FILENAME,
            "memory": f"{_INTERNAL_DIRNAME}/{_MEMORY_FILENAME}",
            "playbook": f"{_INTERNAL_DIRNAME}/{_PLAYBOOK_FILENAME}",
        },
        "codex": {
            "model": str(d3.get("codex_model") or ""),
            "reasoning_effort": str(d3.get("reasoning_effort") or "medium"),
            "reasoning_summary": str(d3.get("reasoning_summary") or "auto"),
        },
    }
    if updated_iteration is not None:
        payload["updated_iteration"] = int(updated_iteration)
    if pattern_payloads:
        payload["distilled_patterns"] = pattern_payloads
    return payload


def _default_agents_content(config: HarnessConfig) -> str:
    workflow = str(config.D4.get("workflow") or "agentic_loop")
    validator = str(config.D6.get("validator") or "repo_checks")
    memory_policy = str(config.D5.get("memory_policy") or "summary_buffer")
    return (
        "# MemoHarness Codex Overlay\n\n"
        "You are the coding agent running inside Harbor's Codex integration.\n"
        "Before acting, read `./policy.json` for the authoritative D1-D6 summary.\n"
        "Then read `./.memoharness/playbook.md` and `./.memoharness/memory.md`.\n\n"
        "Operating rules:\n"
        "- Inspect the repository and verifier clues before claiming completion.\n"
        "- Prefer deterministic repo-local checks over speculative answers.\n"
        "- Make concrete edits, run checks, and summarize the real result.\n"
        "- Stop only when the repo state appears consistent with the verifier contract.\n\n"
        "Current harness summary:\n"
        f"- workflow: {workflow}\n"
        f"- validator: {validator}\n"
        f"- memory_policy: {memory_policy}\n"
    )


def _default_memory_content() -> str:
    return (
        "# Rolling Memory\n\n"
        "- Read `./policy.json` first and use this file only for recent distilled lessons.\n"
        "- No iteration-specific memory has been distilled yet.\n"
        "- Record recurring verifier failures and stable repair heuristics here.\n"
    )


def _default_playbook_content(config: HarnessConfig) -> str:
    return (
        "# Repo Playbook\n\n"
        "Use `./policy.json` as the authoritative D1-D6 summary.\n"
        "Use this file as the stable MemoHarness playbook that Codex can consult during execution.\n\n"
        "Dimension hints:\n"
        f"- D1 strategy: {config.D1.get('strategy') or 'inspect verifier evidence first'}\n"
        f"- D2 strategy: {config.D2.get('strategy') or 'use local tooling and file inspection first'}\n"
        f"- D4 strategy: {config.D4.get('strategy') or 'iterate until checks are satisfied'}\n"
        f"- D6 strategy: {config.D6.get('strategy') or 'validate outputs before finishing'}\n"
    )


def _ensure_policy_reference_in_agents(path: Path) -> None:
    if not path.exists():
        return
    content = path.read_text(encoding="utf-8")
    if "policy.json" in content:
        return
    addition = (
        "\n\nBundle contract:\n"
        f"{_POLICY_READ_LINE}\n"
        f"{_POLICY_SUPPORT_LINE}\n"
    )
    path.write_text(content.rstrip() + addition + "\n", encoding="utf-8")


def _render_memory_content(
    distilled_patterns: Iterable[GlobalPattern],
    *,
    updated_iteration: int | None = None,
) -> str:
    selected_patterns = list(distilled_patterns)[-5:]
    lines = ["# Rolling Memory", ""]
    if updated_iteration is not None:
        lines.append(f"Updated after iteration {updated_iteration}.")
        lines.append("")
    lines.append("- Read `./policy.json` first and use this file only for recent distilled lessons.")
    if not selected_patterns:
        lines.append("- No stable distilled patterns are available yet.")
        lines.append("- Record recurring verifier failures and verified repair heuristics here.")
        return "\n".join(lines)

    lines.append("- Keep only recurring failures and verified repair heuristics here.")
    lines.append("")
    lines.append("Recent distilled patterns:")
    for pattern in reversed(selected_patterns):
        evidence = ", ".join(pattern.evidence[:3]) or "n/a"
        effect = pattern.effect or "No effect summary recorded."
        lines.append(f"- [{pattern.primary_dim}] {pattern.description}")
        lines.append(f"  Effect: {effect}")
        lines.append(f"  Evidence: {evidence}")
    return "\n".join(lines)


def _render_playbook_content(
    config: HarnessConfig,
    distilled_patterns: Iterable[GlobalPattern],
) -> str:
    selected_patterns = list(distilled_patterns)[-5:]
    dim_counts = Counter(pattern.primary_dim for pattern in selected_patterns if pattern.primary_dim)
    lines = [
        "# Repo Playbook",
        "",
        "Use `./policy.json` as the authoritative D1-D6 summary.",
        "Use this file for stable MemoHarness execution heuristics that should survive iteration-to-iteration.",
        "",
        "Current priorities:",
        "- Read `./policy.json` first, then use this playbook for stable execution heuristics.",
        f"- D1 strategy: {config.D1.get('strategy') or 'inspect verifier evidence first'}",
        f"- D2 strategy: {config.D2.get('strategy') or 'use local tooling and file inspection first'}",
        f"- D4 strategy: {config.D4.get('strategy') or 'iterate until checks are satisfied'}",
        f"- D5 strategy: {config.D5.get('strategy') or 'keep rolling memory compact and evidence-backed'}",
        f"- D6 strategy: {config.D6.get('strategy') or 'validate outputs before finishing'}",
    ]
    if dim_counts:
        lines.append("")
        lines.append("Distilled emphasis:")
        for dim, count in dim_counts.most_common(3):
            lines.append(
                f"- Recent distilled pressure is highest on {dim} ({count} pattern(s)); keep that dimension under active scrutiny."
            )
    return "\n".join(lines)


def ensure_codex_bundle(root: Path | str, config: HarnessConfig | None = None) -> CodexBundlePaths:
    bundle_paths = resolve_codex_bundle_paths(root)
    normalized = config.clone() if config is not None else make_minimal_config()
    bundle_paths.root.mkdir(parents=True, exist_ok=True)
    bundle_paths.memory_path.parent.mkdir(parents=True, exist_ok=True)
    if not bundle_paths.agents_path.exists():
        bundle_paths.agents_path.write_text(
            _default_agents_content(normalized),
            encoding="utf-8",
        )
    if not bundle_paths.memory_path.exists():
        bundle_paths.memory_path.write_text(_default_memory_content(), encoding="utf-8")
    if not bundle_paths.playbook_path.exists():
        bundle_paths.playbook_path.write_text(
            _default_playbook_content(normalized),
            encoding="utf-8",
        )
    if not bundle_paths.policy_path.exists():
        bundle_paths.policy_path.write_text(
            json.dumps(_policy_payload(normalized), indent=2),
            encoding="utf-8",
        )
    return bundle_paths


def write_codex_policy(root: Path | str, config: HarnessConfig) -> Path:
    bundle_paths = ensure_codex_bundle(root, config)
    bundle_paths.policy_path.write_text(
        json.dumps(_policy_payload(config), indent=2),
        encoding="utf-8",
    )
    return bundle_paths.policy_path


def _load_policy_dict(policy_path: Path) -> dict[str, object]:
    raw = json.loads(policy_path.read_text(encoding="utf-8"))
    if isinstance(raw.get("config"), dict):
        return dict(raw["config"])
    return raw if isinstance(raw, dict) else {}


def read_codex_bundle_config(
    root: Path | str,
    fallback: HarnessConfig | None = None,
) -> HarnessConfig:
    bundle_paths = ensure_codex_bundle(root, fallback)
    try:
        raw = _load_policy_dict(bundle_paths.policy_path)
    except json.JSONDecodeError:
        logger.warning(
            "Codex bundle policy %s was not valid JSON; falling back to provided config.",
            bundle_paths.policy_path,
        )
        return fallback.clone() if fallback is not None else make_minimal_config()
    config_dict = {
        dim: raw.get(dim, {})
        for dim in ("D1", "D2", "D3", "D4", "D5", "D6")
    }
    return HarnessConfig(**config_dict)


def render_codex_bundle_preview(root: Path | str) -> str:
    bundle_paths = resolve_codex_bundle_paths(root)
    sections: list[str] = []
    for label, path in (
        ("AGENTS.override.md", bundle_paths.agents_path),
        ("policy.json", bundle_paths.policy_path),
        ("memory.md", bundle_paths.memory_path),
        ("playbook.md", bundle_paths.playbook_path),
    ):
        if not path.exists():
            sections.append(f"## {label}\n(missing)")
            continue
        sections.append(f"## {label}\n{path.read_text(encoding='utf-8')}")
    return "\n\n".join(sections).strip()


def load_codex_bundle(
    root: Path | str,
    fallback: HarnessConfig | None = None,
) -> tuple[str, HarnessConfig]:
    config = read_codex_bundle_config(root, fallback)
    preview = render_codex_bundle_preview(root)
    return preview, config


def refresh_codex_bundle_support_docs(
    root: Path | str,
    config: HarnessConfig,
    *,
    distilled_patterns: Iterable[GlobalPattern] = (),
    updated_iteration: int | None = None,
) -> CodexBundlePaths:
    bundle_paths = ensure_codex_bundle(root, config)
    selected_patterns = list(distilled_patterns)[-5:]
    _ensure_policy_reference_in_agents(bundle_paths.agents_path)
    bundle_paths.memory_path.write_text(
        _render_memory_content(selected_patterns, updated_iteration=updated_iteration),
        encoding="utf-8",
    )
    bundle_paths.playbook_path.write_text(
        _render_playbook_content(config, selected_patterns),
        encoding="utf-8",
    )
    bundle_paths.policy_path.write_text(
        json.dumps(
            _policy_payload(
                config,
                distilled_patterns=selected_patterns,
                updated_iteration=updated_iteration,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    return bundle_paths


def snapshot_codex_bundle(root: Path | str) -> dict[Path, str | None]:
    bundle_paths = resolve_codex_bundle_paths(root)
    snapshot: dict[Path, str | None] = {}
    for path in bundle_paths.tracked_files():
        snapshot[path] = path.read_text(encoding="utf-8") if path.exists() else None
    return snapshot


def restore_codex_bundle(snapshot: dict[Path, str | None]) -> None:
    for path, content in snapshot.items():
        if content is None:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.debug("Could not remove restored Codex bundle file %s.", path, exc_info=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def bundle_file_payload(root: Path | str) -> list[tuple[str, str]]:
    bundle_paths = ensure_codex_bundle(root)
    payload: list[tuple[str, str]] = []
    for path in bundle_paths.tracked_files():
        rel_path = path.relative_to(bundle_paths.root).as_posix()
        payload.append((rel_path, path.read_text(encoding="utf-8")))
    return payload


def _write_file_command(relative_path: str, content: str) -> str:
    token = "__MEMOHARNESS_EOF__"
    mkdir = f"mkdir -p {shlex.quote(str(Path(relative_path).parent.as_posix() or '.'))}"
    return (
        f"{mkdir}\n"
        f"cat <<'{token}' > {shlex.quote(relative_path)}\n"
        f"{content}\n"
        f"{token}\n"
    )


async def sync_codex_bundle_to_environment(environment, root: Path | str) -> None:
    for relative_path, content in bundle_file_payload(root):
        result = await environment.exec(_write_file_command(relative_path, content))
        rendered = result if isinstance(result, str) else str(result)
        if "[bash error]" in rendered.lower():
            raise RuntimeError(
                f"Failed to sync Codex bundle file {relative_path}: {rendered}"
            )


def tracked_codex_bundle_files(root: Path | str) -> Iterable[Path]:
    return resolve_codex_bundle_paths(root).tracked_files()
