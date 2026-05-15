import asyncio
import contextlib
import json
import os
import re
import shlex
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from memoharness.harbor.agent import (
    build_assistant_tool_message,
    build_response_input_from_messages,
    build_function_call_output_item,
    build_response_input_message,
    build_tool_result_message,
    build_openai_client,
    call_openai_model_with_fallback,
    preferred_openai_api_mode,
    extract_bash_blocks,
    load_runtime_config,
    populate_context,
    resolve_tool_protocol,
    strip_bash_blocks,
)


def _log(msg: str) -> None:
    """Print to stderr with flush 鈥?always visible even without logging config."""
    print(f"[harness {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)

# ----------------------------
# Reliability-first tunables
# ----------------------------
_TEMPERATURE = 0.0
_MAX_COMPLETION_TOKENS = 8192

# Keep moderate chat history; structured state fills in the rest.
_HISTORY_KEEP = 14

# Timeouts and command budgets. Keep per-command caps, but do not impose a
# fixed overall agent runtime ceiling.
_TIMEOUT_PROBE = 30.0
_TIMEOUT_CMD = 150.0
_TIMEOUT_LONG_CMD = 300.0
_TIMEOUT_INSTALL = 360.0
_TIMEOUT_TEST = 420.0
_TIMEOUT_LLM = 180.0
_MAX_RUNTIME_SECONDS: Optional[float] = None
_LOCAL_SANITY_MAX_TURNS = 12
_LOW_SIGNAL_HANDOFF_MIN_TURNS = 6
_LOW_SIGNAL_HANDOFF_MIN_STAGNATION = 3
_TESTS_VISIBILITY_RETRY_ATTEMPTS = 3
_TESTS_VISIBILITY_RETRY_DELAY_SECONDS = 2.0
_MODEL_INSTALL_TIMEOUT_CAP = 180.0
_MODEL_COMMAND_MIN_TIMEOUT = 30.0
_MODEL_COMMAND_HARD_FLOOR = 5.0
_MODEL_COMMAND_LOW_TIME_CAP = 15.0
_MODEL_COMMAND_RUNTIME_RESERVE = 150.0

_ART_DIR = "/app/.artifacts"
_STATE_PATH = f"{_ART_DIR}/state.json"
_LAST_BOOT_PATH = f"{_ART_DIR}/bootstrap.txt"
_LAST_TEST_PATH = f"{_ART_DIR}/last_test.txt"
_LAST_OBS_PATH = f"{_ART_DIR}/last_obs.txt"

# Unique heredoc delimiter to avoid collision with content
_HEREDOC_DELIM = "MEMOHARNESS_EOF"

# Ensure common user-local bins are on PATH (many Terminal-Bench verifiers assume this).
_PATH_PREFIX = (
    "export PATH="
    "\"/root/.local/bin:$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH\""
)

_HARNESS_CONFIG = json.loads(
    r"""
{
  "D1": {
    "examples": false,
    "structured_instruction": true,
    "compression": "none",
    "strategy": "Verifier-first repair with read-only verifier boundaries: treat /tests plus discovered repo-local verifier files as read-only evidence, extract exact expectations from visible /tests or self-contained root-scoped bundled verifier files when present, and when /tests is hidden proactively preview sparse top-level or single-nested repo text/data/source files, split hyphenated instruction terms into searchable subterms, rank highest-overlap copy-first literals plus visible source/data paths while demoting generic dummy placeholders and target-file self-echoes, and surface producer heads/stderr so the model copies exact local values and fixes the real producer before it reaches for embeddings, OCR, or forensic search."
  },
  "D2": {
    "tool_access": "bash",
    "tool_protocol": "native",
    "retrieval_mode": "test_content_read",
    "top_k": 0,
    "strategy": "Deterministic bootstrap with nested-repo-root promotion, hidden-mode install gating, and producer-first diagnostics: ensure /root/.local/bin/env plus lightweight aliases, detect or alias the real workspace root when /app is absent, promote a single nested repo root under /app when the top-level only contains a wrapper directory, infer python/pip/uv/curl/git only from explicit task/verifier evidence plus high-signal free-text command mentions, harvest repo-documented smoke commands from README/build files, proactively preview sparse hidden-mode source/data files, broaden hidden producer discovery from exact visible matches plus source-like files under sparse or single-nested repos, persist newly observed runtime paths and sockets from model inspection back into later hidden probes, exclude harness-generated and cache trees from hidden searches, search both path matches and file contents, prioritize target/output artifacts over source heads in hidden probes, emit explicit text-target content and match markers plus producer-failure markers, auto-install explicit producer interpreters such as Rscript when a chosen producer requires them, block speculative embedding/OCR stacks when exact visible text/data candidates already exist, harvest .sock clues from instructions, source snippets, and command lines, expand basename socket probes through /tmp, /run, and /var/run, probe hidden service tasks with pgrep/ss plus /proc/<pid>/cmdline evidence before handoff, restrict bundled verifier detection to root-scoped verifier-style files (test_outputs.py, test.sh, verify.sh, ./tests, ./test), demote ones that shell into /tests to advisory-only, and wrap model commands with remaining-budget-aware timeouts."
  },
  "D3": {
    "temperature": 0.0,
    "max_tokens": 8192,
    "top_p": 1.0,
    "candidate_count": 1,
    "strategy": "Deterministic generation: temperature=0.0, max_completion_tokens=8192, single candidate. System prompt enforces short, command-focused responses."
  },
  "D4": {
    "workflow": "agentic_loop",
    "stop_rule": "return_code_0",
    "strategy": "Agentic loop with authoritative verifier promotion, fixed pytest nodeid reruns only for authoritative runners, compact model-error retries, and refreshed hidden-mode evidence: re-probe /tests throughout the run, avoid patching verifier-like files, summarize actual stdout/stderr instead of raw command bodies, suppress source-code and command-fragment noise in hidden summaries and literal candidates, prefer active instruction-driven or repo-documented producer smokes plus source inspection over passive target-only probes when /tests is hidden, normalize bare script-path smokes through interpreters, throttle expensive hidden source/discovery refreshes after the early turns unless evidence is still sparse, feed newly inspected runtime paths back into follow-up hidden probes, keep richer hidden advisory probes anchored on concrete target/output artifacts instead of bare file existence alone, promote only high-signal copy-first literals in hidden-mode follow-ups, extend hidden advisory loops up to the shared Harbor turn cap while producer/literal/service clues remain, treat file-only instruction smokes as advisory rather than rich evidence, reserve rich hidden loop extension for service tasks with concrete single-PID /proc plus socket/port proof, and only allow low-signal hidden handoff under genuine time pressure after clue-driven retries are exhausted."
  },
  "D5": {
    "memory_policy": "sliding_window",
    "history_keep": 14,
    "strategy": "Sliding window of recent turns (14) while pinning the original task instruction; persist compact state + assertion/task targets + condensed tails to artifacts, and rebuild compact retry prompts from state plus top literal candidates after model failures so hidden-verifier cases retain the contract without prompt bloat."
  },
  "D6": {
    "postprocess": "raw_passthrough",
    "validator": "return_code_check",
    "fallback": "none",
    "strategy": "Verifier-aware stopping: accept success only from official /tests runners or self-contained root-scoped bundled repo verifier entrypoints that do not depend on hidden /tests, reject stdout failure markers and shell-level rerun errors, and in hidden-mode keep exact source evidence plus rich producer stderr/source diagnostics in context while target-only advisory probes are upgraded with visible source/data previews and explicit text-target mismatch markers, never treat file-only local smokes as rich evidence, require a single PID plus readable /proc cmdline together with socket/process/port proof before treating hidden service checks as rich evidence, never use target-file self-echoes or generic dummy visible text as corroborating evidence, and never treat pure hidden service proof as a handoff signal unless a self-contained local verifier or explicit pass marker corroborates it."
  }
}
"""
)

_MAX_INTERMEDIATE_CHARS = 4000
_MAX_INTERMEDIATE_ITEMS = 120
_MAX_TOOL_TRACE_CHARS = 1200
_MAX_TOOL_TRACE_ITEMS = 120
_MAX_OBS_TEXT_CHARS = 16000
_MAX_TEST_TEXT_CHARS = 16000
_MAX_BOOT_TEXT_CHARS = 18000
_MAX_STATE_TEXT_CHARS = 12000
_MAX_INLINE_INSTRUCTION_CHARS = 5000
_MAX_DISCOVERY_COMMAND_ITEMS = 12
_MAX_COPY_FIRST_CANDIDATES = 8
_MAX_MODEL_ERROR_RETRIES = 2
_HIDDEN_PRODUCER_SEARCH_MAXDEPTH = 4
_HIDDEN_EVIDENCE_MAX_FILES = 6
_OBSERVED_PROBE_PATHS_MAX = 12

_DISCOURAGED_HIDDEN_STACK_HINTS = (
    "sentence-transformers",
    "sentence_transformers",
    "transformers",
    "faiss",
    "chromadb",
    "chroma",
    "langchain",
    "llama-index",
    "easyocr",
    "paddleocr",
    "pytesseract",
    "tesseract",
    "opencv",
    "cv2",
)

_SIGNAL_MARKERS = (
    "assert",
    "error",
    "failed",
    "failure",
    "traceback",
    "timeout",
    "missing",
    "no such file",
    "command not found",
    "not found",
    "exception",
    "rc=",
    "warning",
    "text_target",
    "producer_failure",
    "smoke_failure",
)

_SOURCEISH_LINE_PREFIXES = (
    "#",
    "//",
    "/*",
    "*",
    "assert ",
    "def ",
    "class ",
    "if ",
    "elif ",
    "else:",
    "for ",
    "while ",
    "return ",
    "import ",
    "from ",
    "with ",
    "try:",
    "except",
    "finally:",
    "echo ",
    "cat ",
    "mkdir ",
    "rm ",
    "mv ",
    "cp ",
    "cd ",
    "chmod ",
    "chown ",
    "<!doctype",
    "<html",
    "<head",
    "<body",
    "</html",
    "</body",
    "</head",
    "<script",
    "</script",
    "<style",
    "</style",
)

_COMMON_COMMAND_HINTS = (
    "curl",
    "wget",
    "git",
    "python",
    "python3",
    "pip",
    "pip3",
    "pytest",
    "uv",
    "uvx",
    "node",
    "npm",
    "npx",
    "bash",
    "sh",
    "make",
    "cmake",
    "gcc",
    "g++",
    "javac",
    "java",
    "go",
    "cargo",
    "rustc",
    "7z",
    "unzip",
    "file",
    "pkill",
    "rg",
)

_FREE_TEXT_COMMAND_HINTS = tuple(
    command
    for command in _COMMON_COMMAND_HINTS
    if command not in {"bash", "file", "make", "sh"}
)

_COMMON_FILE_SUFFIXES = (
    "txt",
    "json",
    "csv",
    "tsv",
    "dat",
    "xml",
    "yaml",
    "yml",
    "md",
    "log",
    "out",
    "cfg",
    "conf",
    "ini",
    "png",
    "jpg",
    "jpeg",
    "gif",
    "svg",
    "tga",
    "bmp",
    "pdf",
    "zip",
    "7z",
    "tar",
    "gz",
    "xz",
    "bz2",
    "js",
    "mjs",
    "cjs",
    "py",
    "r",
    "sh",
    "html",
    "css",
    "sql",
    "toml",
    "stan",
    "bin",
    "so",
    "dll",
    "exe",
    "jar",
    "wasm",
)

_TEXTISH_FILE_SUFFIXES = (
    "txt",
    "json",
    "csv",
    "tsv",
    "dat",
    "xml",
    "yaml",
    "yml",
    "md",
    "log",
    "cfg",
    "conf",
    "ini",
    "js",
    "mjs",
    "cjs",
    "py",
    "sh",
    "html",
    "css",
    "sql",
    "r",
    "toml",
    "stan",
)

_LOW_SIGNAL_LITERAL_HINTS = (
    "dummy",
    "placeholder",
    "example",
    "sample output",
    "template",
    "todo",
)

_TARGET_OUTPUT_NAME_HINTS = (
    "answer",
    "flag",
    "move",
    "output",
    "password",
    "report",
    "result",
    "solution",
)

_LOCAL_VERIFIER_FILENAMES = (
    "test_outputs.py",
    "test_output.py",
    "test.sh",
    "verify.sh",
)

_EXECUTABLE_FILE_SUFFIXES = (
    "sh",
    "py",
    "js",
    "mjs",
    "cjs",
    "r",
    "pl",
    "rb",
    "php",
    "bin",
    "exe",
)

_COMMON_PRODUCER_BASENAMES = (
    "benchmark.py",
    "benchmark.sh",
    "eval.py",
    "eval.sh",
    "evaluate.py",
    "evaluate.sh",
    "optimize.py",
    "optimize.sh",
    "run.sh",
    "start.sh",
    "start_vm.sh",
    "launch.sh",
    "serve.sh",
    "build.sh",
    "deploy.sh",
    "solve.sh",
    "solve.py",
    "solution.py",
    "answer.py",
    "main.py",
    "run.py",
    "app.py",
    "server.py",
)

_PRODUCER_NAME_HINTS = (
    "answer",
    "analysis",
    "analyze",
    "benchmark",
    "app",
    "build",
    "compile",
    "convert",
    "eval",
    "evaluate",
    "create",
    "deploy",
    "extract",
    "fit",
    "generate",
    "install",
    "launch",
    "main",
    "make",
    "optimize",
    "optimization",
    "recover",
    "recovery",
    "run",
    "serve",
    "server",
    "solution",
    "solve",
    "start",
    "train",
    "tune",
    "write",
)

_SEARCH_IGNORE_DIRS = (
    ".artifacts",
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
)

_VISIBLE_TEXT_CANDIDATE_MAX_ITEMS = 18
_REPO_MARKER_FILES = (
    "README.md",
    "README.rst",
    "README.txt",
    "README",
    "Makefile",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
)

_INSTRUCTION_SEARCH_STOPWORDS = {
    "about",
    "after",
    "agent",
    "also",
    "before",
    "between",
    "build",
    "create",
    "deliverable",
    "ensure",
    "exact",
    "exactly",
    "file",
    "files",
    "final",
    "first",
    "from",
    "hidden",
    "instruction",
    "into",
    "make",
    "named",
    "need",
    "output",
    "outputs",
    "path",
    "paths",
    "please",
    "process",
    "repo",
    "return",
    "root",
    "save",
    "should",
    "solve",
    "task",
    "tests",
    "that",
    "their",
    "then",
    "these",
    "this",
    "under",
    "using",
    "verifier",
    "when",
    "with",
    "without",
    "write",
    "your",
}

_OFFICIAL_AUTHORITATIVE_RUNNERS = {
    "test_sh",
    "verify_sh",
    "pytest_test_outputs",
    "pytest_tests_dir",
}

_BUNDLED_LOCAL_RUNNERS = {
    "app_test_sh",
    "app_verify_sh",
    "app_pytest_test_outputs",
    "app_pytest_tests_dir",
    "app_pytest_test_dir",
}

_LOW_SIGNAL_LOCAL_RUNNERS = {
    "local_target_probe",
    "local_command_probe",
    "local_probe",
}

_ADVISORY_LOCAL_RUNNERS = _BUNDLED_LOCAL_RUNNERS | _LOW_SIGNAL_LOCAL_RUNNERS | {
    "local_instruction_smoke",
    "local_visible_evidence_probe",
    "local_repo_pytest",
    "local_python_sanity",
    "local_shell_sanity",
    "local_node_sanity",
}

# Keep Harbor's official verifier as the only real pass/fail authority.
# Repo-local test runners may exist in task repos, but we do not execute them
# during agent repair turns because they are only approximate feedback and can
# waste budget or diverge from the official verifier.
_ENABLE_REPO_LOCAL_TEST_RUNNERS = False
_ENABLE_AGENT_LOCAL_VALIDATION = False
_POST_AGENT_VERIFIER_HANDOFF_RUNNER = "post_agent_verifier_handoff"
_POST_AGENT_VERIFIER_NOTE = (
    "Harbor's official verifier runs after agent execution; "
    "agent phase does not run local validation and hands off directly to Harbor's verifier"
)


def _elapsed(start: float) -> str:
    return f"{time.time() - start:.1f}s"


def _shq(value: str) -> str:
    return shlex.quote(str(value))


def _attach_config(context, cfg: dict) -> None:
    md = getattr(context, "metadata", None)
    if isinstance(md, dict):
        md.setdefault("memoharness_config", cfg)


def _usage_tokens(resp) -> int:
    usage = getattr(resp, "usage", None)
    if not usage:
        return 0
    v = getattr(usage, "total_tokens", None)
    if v is not None:
        return int(v)
    pt = getattr(usage, "prompt_tokens", 0) or 0
    ct = getattr(usage, "completion_tokens", 0) or 0
    return int(pt + ct)


def _trim_messages(messages: List[dict], keep: int) -> List[dict]:
    if keep <= 0 or len(messages) <= 3:
        return list(messages)
    anchor_count = 2 if len(messages) >= 2 else 1
    if len(messages) <= keep + anchor_count:
        return list(messages)
    return list(messages[:anchor_count]) + list(messages[-keep:])


def _json_dumps_safe(obj: Any) -> str:
    """JSON dump that handles non-serializable objects gracefully."""
    try:
        return json.dumps(obj, indent=2, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return json.dumps({"_error": "serialization failed"}, indent=2)


def _clip_inline(text: str, max_chars: int) -> str:
    rendered = str(text or "")
    if len(rendered) <= max_chars:
        return rendered
    keep = max(120, (max_chars - 32) // 2)
    return f"{rendered[:keep]}\n...\n{rendered[-keep:]}"


def _looks_like_sourceish_line(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    if lowered.startswith(_SOURCEISH_LINE_PREFIXES):
        return True
    if (
        re.search(
            r"\b(?:assert|def|class|return|import|from|with|echo|mkdir|chmod|python3?)\b",
            lowered,
        )
        and any(ch in stripped for ch in "(){}[]=<>")
    ):
        return True
    if stripped.endswith(("{", "}", ";")) and any(ch in stripped for ch in "()="):
        return True
    if stripped.count("(") >= 1 and stripped.count(")") >= 1 and "=" in stripped and stripped.count(" ") <= 6:
        return True
    return False


def _looks_like_commandish_fragment(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    if _looks_like_shell_command_line(stripped):
        return True
    if stripped.startswith("-"):
        return True
    if lowered.startswith(("unix:", "tcp:", "udp:")):
        return True
    if any(marker in lowered for marker in (" unix:", " tcp:", " udp:")):
        return True
    if re.search(r"(?:^|\s)--?[A-Za-z0-9][A-Za-z0-9-]*(?:[= ]|$)", stripped) and any(
        token in stripped for token in ("=", ",")
    ):
        return True
    return False


def _summarize_output(text: str, *, max_chars: int, max_lines: int = 120) -> str:
    rendered = str(text or "")
    if not rendered:
        return ""
    lines = rendered.splitlines()
    tail = "\n".join(lines[-max_lines:])

    signal_lines: List[str] = []
    seen: set[str] = set()
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        lowered = line.lower()
        if any(marker in lowered for marker in _SIGNAL_MARKERS):
            if _looks_like_sourceish_line(line) and not any(
                token in lowered
                for token in (
                    "assertionerror",
                    "failed",
                    "failure",
                    "traceback",
                    "warning",
                    "error:",
                    "missing",
                    "not found",
                    "timeout",
                    "text_target",
                    "producer_failure",
                    "smoke_failure",
                )
            ):
                continue
            if line not in seen:
                signal_lines.append(line)
                seen.add(line)
        if len(signal_lines) >= 20:
            break

    pieces: List[str] = []
    if len(rendered) > max_chars or len(lines) > max_lines:
        pieces.append(f"[truncated from {len(rendered)} chars / {len(lines)} lines]")
    if signal_lines:
        pieces.append("SIGNALS:")
        pieces.extend(signal_lines[:20])
    if tail:
        if signal_lines:
            pieces.append("TAIL:")
        pieces.append(tail)

    compact = "\n".join(piece for piece in pieces if piece)
    if not compact:
        compact = tail
    if len(compact) > max_chars:
        compact = _clip_inline(compact, max_chars)
    return compact


def _shell_split(text: str) -> List[str]:
    rendered = str(text or "").strip()
    if not rendered:
        return []
    try:
        return shlex.split(rendered, posix=True)
    except ValueError:
        return rendered.split()


def _command_head(command_line: str) -> str:
    words = _shell_split(command_line)
    if not words:
        return ""
    return os.path.basename(words[0]).strip()


def _is_workspace_repo_path(value: str) -> bool:
    rendered = str(value or "").strip()
    return (
        rendered == "/app"
        or rendered.startswith("/app/")
        or rendered == "/apps"
        or rendered.startswith("/apps/")
    )


def _looks_like_shell_command_line(text: str) -> bool:
    rendered = re.sub(r"\s+", " ", str(text or "").strip())
    if not rendered or len(rendered) > 400:
        return False
    if any(marker in rendered for marker in ("&&", "||", " | ", ";", " > ", " >> ", " < ", " 2>", " 1>")):
        return True

    words = _shell_split(rendered)
    if not words:
        return False

    if rendered.startswith("./") or _is_workspace_repo_path(words[0]):
        if len(words) > 1:
            return True
        if re.search(r"\.(?:sh|py|js|mjs|cjs|bin|exe)$", words[0], flags=re.IGNORECASE):
            return True
        return "." not in os.path.basename(words[0])

    head = _command_head(rendered).lower()
    if head in _COMMON_COMMAND_HINTS:
        return True
    if len(words) >= 2 and any(_is_workspace_repo_path(word) for word in words[1:]):
        return True
    return bool(
        len(words) >= 2
        and re.search(r"\.(?:py|sh|js|mjs|cjs|jar|exe|bin)$", words[0], flags=re.IGNORECASE)
    )


def _path_is_textish(path: str) -> bool:
    return bool(
        re.search(
            r"\.(?:"
            + "|".join(_TEXTISH_FILE_SUFFIXES)
            + r")$",
            str(path or "").strip(),
            flags=re.IGNORECASE,
        )
    )


def _path_suffix(path: str) -> str:
    return os.path.splitext(os.path.basename(str(path or "").strip()))[1].lstrip(".").lower()


def _extract_socket_targets(text: str) -> List[str]:
    targets: List[str] = []
    seen: set[str] = set()
    if not text:
        return targets

    patterns = (
        r"unix:((?:/|\.?/)?(?:tmp|run|var/run|app|apps)/[A-Za-z0-9_./-]+\.sock)\b",
        r"((?:/tmp|/run|/var/run|/app|/apps)/[A-Za-z0-9_./-]+\.sock)\b",
        r"\b([A-Za-z0-9_.-]+\.sock)\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, str(text), flags=re.IGNORECASE):
            candidate = str(match.group(1) or "").strip().strip(".,:;()[]{}")
            if not candidate or candidate in seen:
                continue
            targets.append(candidate)
            seen.add(candidate)
    return targets


def _should_ignore_repo_search_path(path: str) -> bool:
    rendered = str(path or "").strip().replace("\\", "/")
    if not rendered:
        return False
    if rendered.startswith("./"):
        rendered = rendered[2:]
    parts = [part for part in rendered.split("/") if part and part != "."]
    return any(part in _SEARCH_IGNORE_DIRS for part in parts)


def _search_hit_path(line: str) -> str:
    rendered = str(line or "").strip()
    if not rendered:
        return ""
    parts = rendered.split(":", 2)
    if parts and parts[0].isdigit() and len(parts) >= 2:
        return parts[1].strip()
    return parts[0].strip()


def _rg_search_excludes() -> str:
    return " ".join(f"-g '!**/{name}/**'" for name in _SEARCH_IGNORE_DIRS)


def _grep_search_excludes() -> str:
    return " ".join(f"--exclude-dir={name}" for name in _SEARCH_IGNORE_DIRS)


def _find_search_prune_clause() -> str:
    patterns: List[str] = []
    for name in _SEARCH_IGNORE_DIRS:
        patterns.append(f"-path './{name}'")
        patterns.append(f"-path './{name}/*'")
    return "\\( " + " -o ".join(patterns) + " \\) -prune -o"


def _command_line_targets(command_line: str) -> List[Tuple[str, str]]:
    targets: List[Tuple[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    words = _shell_split(command_line)
    if not words:
        return targets

    def _remember(tag: str, value: str) -> None:
        rendered = str(value or "").strip().strip(".,:;()[]{}")
        if not rendered:
            return
        key = (tag, rendered)
        if key in seen:
            return
        targets.append(key)
        seen.add(key)

    head = _command_head(command_line).lower()
    if head:
        _remember("command", head)

    for socket in _extract_socket_targets(command_line):
        _remember("socket", socket)

    redirect_tokens = {">", ">>", "1>", "1>>", "2>", "2>>"}
    expect_redirect_target = False
    for word in words[1:]:
        cleaned = str(word or "").strip().strip(".,:;()[]{}")
        if not cleaned:
            continue
        if expect_redirect_target:
            expect_redirect_target = False
            if _is_workspace_repo_path(cleaned):
                _remember("path", cleaned)
            elif _extract_socket_targets(cleaned):
                for socket in _extract_socket_targets(cleaned):
                    _remember("socket", socket)
            elif re.search(
                r"\.(" + "|".join(_COMMON_FILE_SUFFIXES) + r")$",
                cleaned,
                flags=re.IGNORECASE,
            ):
                _remember("artifact", cleaned)
            continue
        if cleaned in redirect_tokens:
            expect_redirect_target = True
            continue
        if _is_workspace_repo_path(cleaned):
            _remember("path", cleaned)
        elif _extract_socket_targets(cleaned):
            for socket in _extract_socket_targets(cleaned):
                _remember("socket", socket)
        elif re.search(
            r"\.(" + "|".join(_COMMON_FILE_SUFFIXES) + r")$",
            cleaned,
            flags=re.IGNORECASE,
        ) and not cleaned.startswith("-"):
            _remember("artifact", cleaned)
        elif cleaned.lower() in _COMMON_COMMAND_HINTS:
            _remember("command", cleaned.lower())

    for match in re.finditer(r"(?:^|\s)(?:1?>|1?>>|2>|2>>|>>)\s*([^\s|&;]+)", command_line):
        cleaned = match.group(1).strip().strip(".,:;()[]{}")
        if not cleaned:
            continue
        if _is_workspace_repo_path(cleaned):
            _remember("path", cleaned)
        elif _extract_socket_targets(cleaned):
            for socket in _extract_socket_targets(cleaned):
                _remember("socket", socket)
        elif re.search(
            r"\.(" + "|".join(_COMMON_FILE_SUFFIXES) + r")$",
            cleaned,
            flags=re.IGNORECASE,
        ):
            _remember("artifact", cleaned)

    return targets


def _looks_like_producer_filename(path: str) -> bool:
    base = os.path.basename(str(path or "").strip()).lower()
    if not base:
        return False
    if base in _LOCAL_VERIFIER_FILENAMES or base in {"test.sh", "verify.sh"}:
        return False
    suffix = os.path.splitext(base)[1].lstrip(".").lower()
    if suffix and suffix not in _EXECUTABLE_FILE_SUFFIXES:
        return False
    if base in _COMMON_PRODUCER_BASENAMES:
        return True
    stem = re.sub(
        r"\.(?:"
        + "|".join(_EXECUTABLE_FILE_SUFFIXES)
        + r")$",
        "",
        base,
        flags=re.IGNORECASE,
    )
    if stem in {"main", "app", "server"}:
        return True
    return bool(
        re.search(
            r"(?:^|[-_.])(?:"
            + "|".join(re.escape(part) for part in _PRODUCER_NAME_HINTS)
            + r")(?:[-_.]|$)",
            stem,
            flags=re.IGNORECASE,
        )
    )


def _extract_instruction_targets(text: str) -> List[str]:
    targets: List[str] = []
    seen: set[str] = set()

    def _add(tag: str, value: str) -> None:
        rendered = str(value or "").strip().strip(".,:;()[]{}")
        if not rendered:
            return
        key = f"{tag}: {rendered}"
        if key in seen:
            return
        targets.append(key)
        seen.add(key)

    if not text:
        return targets

    for match in re.finditer(r"(/(?:app|apps)/[^\s`\"'>)]+)", text):
        value = match.group(1).strip().strip(".,:;()[]{}")
        _add("path", value)
        if _looks_like_producer_filename(value):
            _add("producer_candidate", value)

    for match in re.finditer(r"((?:/tmp|/run|/var/run)/[^\s`\"'>)]+\.sock)\b", text):
        _add("socket", match.group(1))

    for socket in _extract_socket_targets(text):
        _add("socket", socket)

    for match in re.finditer(r"`([^`]+)`", text):
        token = re.sub(r"\s+", " ", match.group(1).strip())
        if not token:
            continue
        if _looks_like_shell_command_line(token):
            _add("command_line", token)
            for tag, value in _command_line_targets(token):
                _add(tag, value)
            continue
        if _is_workspace_repo_path(token):
            _add("path", token)
            if _looks_like_producer_filename(token):
                _add("producer_candidate", token)
        elif re.search(
            r"\.(" + "|".join(_COMMON_FILE_SUFFIXES) + r")$",
            token,
            flags=re.IGNORECASE,
        ):
            _add("artifact", token)
            if _looks_like_producer_filename(token):
                _add("producer_candidate", token)
        elif token.lower() in _COMMON_COMMAND_HINTS:
            _add("command", token.lower())

    for match in re.finditer(
        r"\b([A-Za-z0-9_.-]+\.(?:"
        + "|".join(_COMMON_FILE_SUFFIXES)
        + r"))\b",
        text,
        flags=re.IGNORECASE,
    ):
        value = match.group(1)
        _add("artifact", value)
        if _looks_like_producer_filename(value):
            _add("producer_candidate", value)

    for match in re.finditer(r"\b(?:127\.0\.0\.1|0\.0\.0\.0|localhost):(\d{2,5})\b", text):
        port = match.group(1)
        if 0 < int(port) <= 65535:
            _add("port", port)

    for match in re.finditer(r"\bport\s+(\d{2,5})\b", text, flags=re.IGNORECASE):
        port = match.group(1)
        if 0 < int(port) <= 65535:
            _add("port", port)

    lowered = text.lower()
    for command in _FREE_TEXT_COMMAND_HINTS:
        if re.search(rf"\b{re.escape(command)}\b", lowered):
            _add("command", command)

    return targets[:30]


def _extract_stdout(result) -> str:
    """Extract text output from environment.exec() result."""
    if isinstance(result, str):
        return result
    stdout = getattr(result, "stdout", None)
    stderr = getattr(result, "stderr", None)
    output = getattr(result, "output", None)
    if output is not None:
        return output or ""
    if stdout is not None or stderr is not None:
        merged = (stdout or "")
        if stderr:
            merged = (merged + "\n" + stderr) if merged else stderr
        return merged or ""
    return str(result)


def _extract_rc(result) -> int:
    """Extract return code from environment.exec() result."""
    rc = getattr(result, "return_code", None)
    if rc is not None:
        return int(rc)
    rc = getattr(result, "returncode", None)
    if rc is not None:
        return int(rc)
    return -1


def _extract_assertion_targets(test_text: str) -> List[str]:
    """Parse test file content for file paths and expected artifacts.

    Looks for:
    - Path("/app/...") patterns
    - os.path.exists() / os.path.isfile() / os.path.isdir() patterns
    - pathlib Path.exists() patterns
    - assert ... exists patterns
    - open("...") patterns
    - assert ... in ... patterns (command output expectations)
    - FileNotFoundError patterns
    - subprocess / command patterns
    - Directory patterns (mkdir, os.makedirs)
    """
    targets: List[str] = []
    seen = set()

    def _add(tag: str, value: str) -> None:
        key = f"{tag}: {value}"
        if key not in seen:
            targets.append(key)
            seen.add(key)

    # Path("/app/...") or Path("...") patterns
    for m in re.finditer(r'Path\(["\']([^"\']+)["\']\)', test_text):
        p = m.group(1)
        _add("path", p)

    # os.path.exists("...") / os.path.isfile("...") / os.path.isdir("...")
    for m in re.finditer(r"os\.path\.(?:exists|isfile|isdir)\([\"']([^\"']+)[\"']\)", test_text):
        _add("path", m.group(1))

    # pathlib: p = Path("...") then assert p.exists()
    for m in re.finditer(r'(?:\w+)\s*=\s*Path\(["\']([^"\']+)["\']\)', test_text):
        _add("path", m.group(1))

    # assert path.exists() or assert ... exists
    for m in re.finditer(r'assert\s+\w+\.exists\(\)', test_text):
        # Look for the path variable assignment nearby
        pass

    # open("...") patterns 鈥?both read and write
    for m in re.finditer(r'open\(["\']([^"\']+)["\']', test_text):
        p = m.group(1)
        if not p.startswith(("/proc", "/sys", "/dev")):
            _add("file", p)

    # /app/... or /apps/... paths in strings
    for m in re.finditer(r'["\'](/(?:app|apps)/[^"\']+)', test_text):
        p = m.group(1)
        # Skip test files themselves and Python source in /tests
        if p.endswith((".py", ".sh")) and p.startswith("/tests"):
            continue
        _add("path", p)

    # command expectations: assert "..." in result
    for m in re.finditer(r'assert\s+["\']([^"\']+)["\']\s+in\s+\w+', test_text):
        val = m.group(1)
        if len(val) > 2:
            _add("expected_output", val)

    # FileNotFoundError patterns
    for m in re.finditer(r'FileNotFoundError.*?["\']([^"\']+)["\']', test_text):
        _add("must_exist", m.group(1))

    # subprocess / command patterns 鈥?extract command binary
    for m in re.finditer(r'subprocess\.\w+\(["\']([^"\']+)', test_text):
        cmd = m.group(1).split()[0]  # get just the binary
        _add("command", cmd)

    # shutil.which("...") 鈥?command availability check
    for m in re.finditer(r'shutil\.which\(["\']([^"\']+)["\']\)', test_text):
        _add("command", m.group(1))

    # os.makedirs / os.mkdir 鈥?expected directories
    for m in re.finditer(r'os\.makedirs?\(["\']([^"\']+)["\']', test_text):
        _add("directory", m.group(1))

    # Directory existence checks in test: test -d /path or Path(...).is_dir()
    for m in re.finditer(r'["\'](/(?:app|apps)/[^"\']+)/?["\']', test_text):
        p = m.group(1).rstrip("/")
        # Heuristic: paths that look like directories (no file extension)
        if not re.search(r'\.\w{1,4}$', p) and p.count("/") > 1:
            _add("directory_candidate", p)

    # uvx / uv patterns (common in Terminal-Bench)
    for m in re.finditer(r'(?:uvx|uv run|uv pip)\s+(\S+)', test_text):
        _add("uv_command", m.group(1))

    # Check for test.sh pattern (common verifier format)
    if '/tests/test.sh' in test_text or '/tests/verify.sh' in test_text:
        _add("uses_test_sh", "true")

    return targets[:40]  # Cap at 40 for richer context


class HarnessImpl:
    def __init__(self) -> None:
        self._client = None
        self._model = None
        self._api_config = None
        self._openai_api_mode = "chat_completions"
        self._num_llm_calls = 0
        self._total_tokens = 0
        self._run_start = 0.0
        self._last_test_output = ""
        self._tools_invoked: List[str] = []
        self._intermediate_outputs: List[str] = []
        self._status_events: List[Dict[str, Any]] = []
        self._current_status: Optional[Dict[str, Any]] = None
        self._console_mode = "normal"
        self._heartbeat_seconds = 30
        self._assertion_targets: List[str] = []
        self._task_targets: List[str] = []
        self._task_instruction = ""
        self._test_content_cache: str = ""
        self._success_markers: List[str] = []
        self._needs: Dict[str, bool] = {}
        self._uv_installed: bool = False
        self._env_setup_done: bool = False
        self._pkg_manager: Optional[str] = None
        self._pkg_update_done: bool = False
        self._auto_fixes_done: set[str] = set()
        self._workspace_root = "/app"
        self._tool_protocol = resolve_tool_protocol(_HARNESS_CONFIG)
        self._native_tool_calling = self._tool_protocol == "native"
        self._hidden_exact_visible_matches: List[str] = []
        self._hidden_visible_text_candidates: List[str] = []
        self._hidden_evidence_paths: List[str] = []
        self._repo_command_candidates: List[str] = []
        self._observed_probe_paths: List[str] = []

        self._state: Dict[str, Any] = {
            "tools": {
                "has_pytest": None,
                "has_rg": None,
                "has_curl": None,
                "has_wget": None,
                "has_python3": None,
                "has_pip3": None,
                "has_verify_sh": None,
                "has_test_sh": None,
                "has_uv": None,
                "has_make": None,
                "has_git": None,
                "has_env_shim": None,
                "pkg_manager": None,
            },
            "repo": {
                "app_top": [],
                "tests_files": [],
                "local_verifier_files": [],
                "local_verifier_mentions_tests": False,
                "instruction_targets": [],
                "agent_can_read_tests": None,
                "readme": None,
                "makefile": None,
                "workspace_root": "/app",
                "workspace_boot_cwd": None,
                "workspace_alias_enabled": False,
                "command_candidates": [],
                "hidden_evidence_paths": [],
                "observed_probe_paths": [],
            },
            "test": {
                "cmd": None,
                "runner": None,
                "last_rc": None,
                "last_success": None,
                "last_nodeid": None,
                "last_tail": None,
            },
            "progress": {
                "turn": 0,
                "stagnation": 0,
                "last_patch_fingerprint": None,
                "strategy_phase": "initial",
                "forced_hidden_evidence_retry": False,
            },
        }

        self._cfg = json.loads(json.dumps(_HARNESS_CONFIG))

    async def setup(self, environment) -> None:
        _log("setup: loading runtime config...")
        runtime = load_runtime_config()
        if runtime:
            self._api_config = runtime.llm
            self._client = build_openai_client(runtime.llm)
            self._model = runtime.llm.model
            experiment = getattr(runtime, "experiment", None)
            if experiment is not None:
                self._console_mode = str(
                    getattr(experiment, "console_mode", self._console_mode) or self._console_mode
                ).lower()
                self._heartbeat_seconds = int(
                    getattr(experiment, "console_heartbeat_seconds", self._heartbeat_seconds)
                    or self._heartbeat_seconds
                )
            _log(f"setup: model={runtime.llm.model}, api_base={runtime.llm.api_base or '(default)'}")
        else:
            self._api_config = None
            import openai
            self._client = openai.AsyncOpenAI()
            self._model = os.environ.get("MEMOHARNESS_MODEL", "gpt-4.1-mini")
            _log(f"setup: model={self._model} (from env, no runtime config)")
        self._openai_api_mode = preferred_openai_api_mode(
            self._api_config,
            native_tool_calling=self._native_tool_calling,
        )
        _log(f"setup: env proxy vars: http_proxy={os.environ.get('http_proxy','(unset)')}, "
             f"https_proxy={os.environ.get('https_proxy','(unset)')}")

    def _status(self, stage: str, *, detail: Optional[str] = None, heartbeat: bool = False) -> None:
        event: Dict[str, Any] = {
            "stage": stage,
            "heartbeat": heartbeat,
            "elapsed_s": round(time.time() - self._run_start, 1) if self._run_start else 0.0,
        }
        if detail:
            event["detail"] = detail
        turn = self._state.get("progress", {}).get("turn")
        if turn:
            event["turn"] = int(turn)
        self._current_status = dict(event)
        self._status_events.append(dict(event))
        print(
            f"[harness-status] {json.dumps(event, ensure_ascii=True, sort_keys=True)}",
            file=sys.stderr,
            flush=True,
        )

    async def _run_with_heartbeat(
        self,
        stage: str,
        awaitable,
        *,
        detail: Optional[str] = None,
    ):
        self._status(stage, detail=detail, heartbeat=False)
        interval = max(0, int(self._heartbeat_seconds))
        if interval <= 0:
            return await awaitable

        stop_event = asyncio.Event()

        async def _ticker() -> None:
            while True:
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=interval)
                    return
                except asyncio.TimeoutError:
                    self._status(stage, detail=detail, heartbeat=True)

        ticker = asyncio.create_task(_ticker())
        try:
            return await awaitable
        finally:
            stop_event.set()
            with contextlib.suppress(Exception):
                await ticker

    async def _exec_raw(self, environment, cmd: str, timeout: float):
        """Execute command and return the raw ExecResult (or str)."""
        wrapped = f"{_PATH_PREFIX}\n{cmd}".strip()
        return await asyncio.wait_for(environment.exec(wrapped), timeout=timeout)

    async def _exec(self, environment, cmd: str, timeout: float) -> str:
        """Execute command and return stdout text."""
        result = await self._exec_raw(environment, cmd, timeout)
        return _extract_stdout(result)

    async def _safe_exec(self, environment, cmd: str, timeout: float) -> str:
        try:
            return await self._exec(environment, cmd, timeout)
        except asyncio.TimeoutError:
            _log(f"EXEC TIMEOUT ({timeout:.0f}s): {cmd[:120]}")
            return f"[timeout after {timeout}s] $ {cmd}"
        except Exception as e:
            _log(f"EXEC ERROR: {cmd[:120]} 鈥?{e}")
            return f"[error] {e} $ {cmd}"

    async def _exec_with_rc(self, environment, cmd: str, timeout: float) -> Tuple[int, str]:
        """Execute command, return (return_code, stdout) from ExecResult directly."""
        try:
            result = await self._exec_raw(environment, cmd, timeout)
            stdout = _extract_stdout(result)
            rc = _extract_rc(result)
            # Fallback: if rc is -1 (not found), try parsing from stdout
            if rc < 0:
                m = re.search(r"RC=(\d+)\s*$", stdout.strip())
                rc = int(m.group(1)) if m else 999
            return rc, stdout
        except asyncio.TimeoutError:
            _log(f"EXEC TIMEOUT ({timeout:.0f}s): {cmd[:120]}")
            return 999, f"[timeout after {timeout}s] $ {cmd}"
        except Exception as e:
            _log(f"EXEC ERROR: {cmd[:120]} 鈥?{e}")
            return 999, f"[error] {e} $ {cmd}"

    async def _write_file(self, environment, path: str, content: str) -> None:
        """Write a compact artifact file without huge inline base64 payloads."""
        rendered = str(content or "")
        limit = _MAX_OBS_TEXT_CHARS
        if path == _LAST_BOOT_PATH:
            limit = _MAX_BOOT_TEXT_CHARS
        elif path == _STATE_PATH:
            limit = _MAX_STATE_TEXT_CHARS
        elif path == _LAST_TEST_PATH:
            limit = _MAX_TEST_TEXT_CHARS
        if path == _STATE_PATH:
            if len(rendered) > limit:
                compact = json.dumps(
                    {
                        "_truncated": True,
                        "reason": "state exceeded artifact limit",
                        "tail": _clip_inline(rendered, max(800, limit - 200)),
                    },
                    indent=2,
                    sort_keys=True,
                )
            else:
                compact = rendered
        else:
            compact = _summarize_output(rendered, max_chars=limit, max_lines=180)
        delim = _HEREDOC_DELIM
        while delim in compact:
            delim += "_X"
        target_path = self._map_repo_path(path)
        cmd = (
            f"mkdir -p {_shq(os.path.dirname(target_path))}\n"
            f"cat > {_shq(target_path)} <<'{delim}'\n"
            f"{compact}\n"
            f"{delim}"
        )
        await self._safe_exec(environment, cmd, _TIMEOUT_CMD)

    async def _persist_state(self, environment) -> None:
        """Persist state to artifact file, handling serialization safely."""
        try:
            repo_state = dict(self._state.get("repo", {}))
            repo_state["app_top"] = list(repo_state.get("app_top", []))[:80]
            repo_state["tests_files"] = list(repo_state.get("tests_files", []))[:120]
            repo_state["instruction_targets"] = self._task_targets[:20]
            repo_state["observed_probe_paths"] = list(repo_state.get("observed_probe_paths", []))[:_OBSERVED_PROBE_PATHS_MAX]

            test_state = dict(self._state.get("test", {}))
            test_state["last_tail"] = _summarize_output(
                str(test_state.get("last_tail") or ""),
                max_chars=2000,
                max_lines=80,
            )

            state_with_targets = {
                "tools": dict(self._state.get("tools", {})),
                "repo": repo_state,
                "test": test_state,
                "progress": dict(self._state.get("progress", {})),
                "assertion_targets": self._assertion_targets[:40],
                "task_targets": self._task_targets[:20],
            }
            content = _json_dumps_safe(state_with_targets)
        except Exception:
            # If state serialization fails, write a minimal version
            content = _json_dumps_safe({"error": "state serialization failed", "turn": self._state.get("progress", {}).get("turn", 0)})
        await self._write_file(environment, _STATE_PATH, content)

    def _workspace_root_path(self) -> str:
        root = str(self._workspace_root or "/app").strip()
        return root if root.startswith("/") else "/app"

    def _workspace_cd(self) -> str:
        root = self._workspace_root_path()
        return f"cd {_shq(root)} 2>/dev/null || exit 1"

    def _map_repo_path(self, path: str) -> str:
        rendered = str(path or "").strip()
        if not rendered:
            return rendered
        root = self._workspace_root_path().rstrip("/") or "/"
        # Preserve literal /app paths when the real repo root is nested under /app.
        if root.startswith("/app/"):
            return rendered
        if rendered == "/app":
            return root
        if rendered.startswith("/app/"):
            tail = rendered[len("/app/"):].lstrip("/")
            if root == "/":
                return f"/{tail}" if tail else "/"
            return f"{root}/{tail}" if tail else root
        return rendered

    def _repo_path(self, value: str) -> str:
        rendered = str(value or "").strip()
        if not rendered:
            return self._workspace_root_path()
        if rendered.startswith("/"):
            return self._map_repo_path(rendered)
        root = self._workspace_root_path().rstrip("/") or "/"
        rel = rendered.lstrip("./")
        if root == "/":
            return f"/{rel}" if rel else "/"
        return f"{root}/{rel}" if rel else root

    def _rewrite_repo_paths_in_command(self, command_line: str) -> str:
        rendered = str(command_line or "")
        root = self._workspace_root_path().rstrip("/") or "/"
        if root == "/app":
            return rendered
        if root == "/":
            return rendered.replace("/app/", "/").replace(" /app", " /")
        return rendered.replace("/app/", f"{root}/").replace(" /app", f" {root}")

    def _filter_repo_search_lines(self, text: str, *, max_lines: int = 20) -> List[str]:
        lines: List[str] = []
        for raw in self._output_lines(text):
            line = raw.strip()
            if not line or line.startswith(("Binary file", "rg:")):
                continue
            path = _search_hit_path(line)
            if _should_ignore_repo_search_path(path):
                continue
            lines.append(line)
            if len(lines) >= max_lines:
                break
        return lines

    def _runnable_path_command(
        self,
        path: str,
        extra_args: Optional[List[str]] = None,
        *,
        wrap_timeout: bool = False,
        timeout_seconds: Optional[float] = None,
    ) -> str:
        mapped = self._map_repo_path(path)
        qpath = _shq(mapped)
        rendered_args = [str(arg or "").strip() for arg in (extra_args or []) if str(arg or "").strip()]
        suffix = ""
        if rendered_args:
            suffix = " " + " ".join(_shq(arg) for arg in rendered_args)
        base = os.path.basename(mapped).lower()

        if base.endswith(".sh"):
            raw_cmd = f"bash {qpath}{suffix}"
        elif base.endswith(".py"):
            raw_cmd = (
                f"if command -v python3 >/dev/null 2>&1; then python3 {qpath}{suffix}; "
                f"elif command -v python >/dev/null 2>&1; then python {qpath}{suffix}; "
                f"elif [ -x {qpath} ]; then {qpath}{suffix}; "
                "else echo 'MISSING python interpreter'; exit 127; fi"
            )
        elif base.endswith((".js", ".mjs", ".cjs")):
            raw_cmd = (
                f"if command -v node >/dev/null 2>&1; then node {qpath}{suffix}; "
                f"elif [ -x {qpath} ]; then {qpath}{suffix}; "
                "else echo 'MISSING node'; exit 127; fi"
            )
        elif base.endswith(".r"):
            raw_cmd = (
                f"if command -v Rscript >/dev/null 2>&1; then Rscript {qpath}{suffix}; "
                f"elif command -v R >/dev/null 2>&1; then R --vanilla -f {qpath}{suffix}; "
                f"elif [ -x {qpath} ]; then {qpath}{suffix}; "
                "else echo 'MISSING Rscript'; exit 127; fi"
            )
        elif base.endswith(".pl"):
            raw_cmd = (
                f"if command -v perl >/dev/null 2>&1; then perl {qpath}{suffix}; "
                f"elif [ -x {qpath} ]; then {qpath}{suffix}; "
                "else echo 'MISSING perl'; exit 127; fi"
            )
        elif base.endswith(".rb"):
            raw_cmd = (
                f"if command -v ruby >/dev/null 2>&1; then ruby {qpath}{suffix}; "
                f"elif [ -x {qpath} ]; then {qpath}{suffix}; "
                "else echo 'MISSING ruby'; exit 127; fi"
            )
        elif base.endswith(".php"):
            raw_cmd = (
                f"if command -v php >/dev/null 2>&1; then php {qpath}{suffix}; "
                f"elif [ -x {qpath} ]; then {qpath}{suffix}; "
                "else echo 'MISSING php'; exit 127; fi"
            )
        else:
            raw_cmd = (
                f"if [ -x {qpath} ]; then {qpath}{suffix}; "
                "else echo 'MISSING executable bit'; exit 126; fi"
            )

        if not wrap_timeout:
            return raw_cmd
        seconds = max(45, int(timeout_seconds or 45))
        return (
            "if command -v timeout >/dev/null 2>&1; then "
            f"timeout --preserve-status -k 5s {seconds}s bash -lc {_shq(raw_cmd)}; "
            f"else bash -lc {_shq(raw_cmd)}; fi"
        )

    def _normalize_probe_command_line(self, command_line: str) -> str:
        rendered = self._rewrite_repo_paths_in_command(command_line)
        words = _shell_split(rendered)
        if not words:
            return rendered

        head = words[0]
        if not head.startswith(("/", "./", "../")):
            return rendered

        base = os.path.basename(head).lower()
        if re.search(r"\.(?:sh|py|js|mjs|cjs|r|pl|rb|php)$", base, flags=re.IGNORECASE):
            return self._runnable_path_command(head, words[1:], wrap_timeout=False)
        if len(words) == 1:
            return self._runnable_path_command(head, wrap_timeout=False)
        return rendered

    def _hidden_candidate_source_path(self, item: str) -> str:
        rendered = str(item or "").strip()
        if not rendered:
            return ""
        if rendered.endswith(" [path match]"):
            return self._repo_path(rendered.rsplit(" [path match]", 1)[0].strip())
        prefix = rendered.split(": ", 1)[0].strip()
        if re.search(r":\d+$", prefix):
            prefix = prefix.rsplit(":", 1)[0]
        base = os.path.basename(prefix).lower()
        if not (
            prefix.startswith(("/", "./", "../"))
            or "/" in prefix
            or "." in base
            or base in {"readme", "makefile"}
            or base.startswith("dockerfile")
        ):
            return ""
        return self._repo_path(prefix)

    def _is_hidden_evidence_path(self, path: str) -> bool:
        mapped = self._map_repo_path(path)
        rel = self._repo_relpath(mapped).lstrip("./")
        if not rel or rel.startswith(("tests/", "test/")):
            return False
        if _should_ignore_repo_search_path(rel):
            return False
        base = os.path.basename(mapped).lower()
        if base in _LOCAL_VERIFIER_FILENAMES:
            return False
        if base in {"readme", "makefile"} or base.startswith("dockerfile"):
            return True
        return _path_is_textish(mapped)

    def _is_hidden_source_candidate_path(self, path: str) -> bool:
        mapped = self._map_repo_path(path)
        if not self._is_hidden_evidence_path(mapped):
            return False
        base = os.path.basename(mapped).lower()
        if base in {"__init__.py", "conftest.py"} or base.startswith("test_"):
            return False
        return _path_suffix(mapped) in _EXECUTABLE_FILE_SUFFIXES or _path_suffix(mapped) == "stan"

    def _hidden_evidence_path_score(self, path: str) -> int:
        mapped = self._map_repo_path(path)
        if not self._is_hidden_evidence_path(mapped):
            return -1
        rel = self._repo_relpath(mapped).lstrip("./")
        base = os.path.basename(mapped).lower()
        suffix = _path_suffix(mapped)
        score = 0
        if base in {"readme", "readme.md", "readme.rst", "readme.txt", "makefile"} or base.startswith("dockerfile"):
            score += 7
        if _looks_like_producer_filename(mapped):
            score += 6
        if suffix in _EXECUTABLE_FILE_SUFFIXES or suffix == "stan":
            score += 4
        elif suffix in {"txt", "json", "csv", "tsv", "dat", "md"}:
            score += 3
        elif suffix in {"yaml", "yml", "toml", "ini", "cfg", "conf"}:
            score += 2
        depth = rel.count("/")
        score += max(0, 3 - min(depth, 3))
        lowered_rel = rel.lower()
        for term in self._instruction_terms()[:18]:
            needle = term.lower().strip()
            if len(needle) < 4:
                continue
            if needle in lowered_rel:
                score += 4 if len(needle) >= 6 else 2
        return score

    def _remember_hidden_evidence_paths(self, paths: List[str]) -> None:
        remembered: List[str] = []
        seen: set[str] = set()
        for raw in paths:
            path = self._repo_path(raw)
            if not self._is_hidden_evidence_path(path) or path in seen:
                continue
            remembered.append(path)
            seen.add(path)
            if len(remembered) >= _HIDDEN_EVIDENCE_MAX_FILES:
                break
        self._hidden_evidence_paths = remembered[:_HIDDEN_EVIDENCE_MAX_FILES]
        self._state["repo"]["hidden_evidence_paths"] = self._hidden_evidence_paths[:]

    def _hidden_evidence_paths_block(self) -> str:
        if not self._hidden_evidence_paths:
            return ""
        return (
            "VISIBLE SOURCE / DATA FILES TO INSPECT FIRST "
            "(before embeddings, OCR, or disk/forensics search):\n"
            + "\n".join(f"  - {path}" for path in self._hidden_evidence_paths[:_HIDDEN_EVIDENCE_MAX_FILES])
        )

    def _hidden_evidence_probe_paths(self, *, limit: int = 4) -> List[str]:
        ranked: Dict[str, int] = {}

        def _remember(path: str, bonus: int) -> None:
            mapped = self._repo_path(path)
            if not self._is_hidden_evidence_path(mapped):
                return
            score = self._hidden_evidence_path_score(mapped) + bonus
            current = ranked.get(mapped)
            if current is None or score > current:
                ranked[mapped] = score

        for idx, path in enumerate(self._hidden_evidence_paths):
            _remember(path, max(1, 8 - idx))
        for idx, item in enumerate(self._hidden_exact_visible_matches):
            path = self._hidden_candidate_source_path(item)
            if path:
                _remember(path, max(1, 6 - idx))
        for idx, item in enumerate(self._hidden_visible_text_candidates):
            path = self._hidden_candidate_source_path(item)
            if path:
                _remember(path, max(1, 4 - idx))

        ordered = sorted(
            ranked.items(),
            key=lambda item: (-item[1], self._repo_relpath(item[0])),
        )
        return [path for path, _score in ordered[:limit]]

    def _looks_like_hidden_producer_path(self, path: str) -> bool:
        mapped = self._map_repo_path(path)
        rel = self._repo_relpath(mapped).lstrip("./")
        if rel.startswith("tests/") or rel.startswith("test/"):
            return False
        return self._is_hidden_source_candidate_path(mapped)

    def _producer_command_timeout_seconds(self, path: str) -> int:
        base = 120.0
        name = os.path.basename(str(path or "")).lower()
        if any(
            token in name
            for token in (
                "analysis",
                "analyze",
                "build",
                "convert",
                "fit",
                "install",
                "launch",
                "run",
                "serve",
                "solve",
                "start",
                "train",
                "tune",
            )
        ):
            base = min(_TIMEOUT_LONG_CMD, 180.0)

        return max(45, int(base))

    async def _collect_hidden_producer_paths(
        self,
        environment,
        *,
        limit: int = 6,
    ) -> List[str]:
        scored: Dict[str, int] = {}
        exact_match_paths = {
            self._hidden_candidate_source_path(item)
            for item in self._hidden_exact_visible_matches + self._hidden_visible_text_candidates
        }
        exact_match_paths.discard("")

        def _remember(value: str, *, bonus: int = 0) -> None:
            path = self._repo_path(value)
            if not self._is_hidden_source_candidate_path(path):
                return
            rel = self._repo_relpath(path).lstrip("./")
            lowered_rel = rel.lower()
            score = bonus
            evidence = False
            if _looks_like_producer_filename(path):
                score += 8
                evidence = True
            if path in exact_match_paths:
                score += 6
                evidence = True
            if path in self._hidden_evidence_paths:
                score += 3
                evidence = True
            for term in self._instruction_terms()[:18]:
                needle = term.lower().strip()
                if len(needle) < 4:
                    continue
                if needle in lowered_rel:
                    score += 4 if len(needle) >= 6 else 2
                    evidence = True
            suffix = _path_suffix(path)
            if suffix in _EXECUTABLE_FILE_SUFFIXES or suffix == "stan":
                score += 4
            depth = rel.count("/")
            score += max(0, 3 - min(depth, 3))
            if not evidence:
                return
            current = scored.get(path)
            if current is None or score > current:
                scored[path] = score

        for target in self._task_targets:
            tag, _, value = target.partition(": ")
            value = value.strip()
            if not value:
                continue
            if tag in {"producer_candidate", "path", "artifact", "file"}:
                _remember(value, bonus=5)
            elif tag == "command_line":
                for extra_tag, extra_value in _command_line_targets(value):
                    if extra_tag in {"path", "artifact", "file"}:
                        _remember(extra_value, bonus=5)

        for candidate in self._repo_command_candidates[:_MAX_DISCOVERY_COMMAND_ITEMS]:
            for extra_tag, extra_value in _command_line_targets(candidate):
                if extra_tag in {"path", "artifact", "file"}:
                    _remember(extra_value, bonus=5)

        for path in exact_match_paths:
            _remember(path, bonus=4)

        listing = await self._safe_exec(
            environment,
            f"{self._workspace_cd()}\n"
            f"find . -maxdepth {_HIDDEN_PRODUCER_SEARCH_MAXDEPTH} {_find_search_prune_clause()} -type f "
            "-print | sort | sed -n '1,400p'",
            _TIMEOUT_PROBE,
        )
        for raw in self._output_lines(listing):
            line = raw.strip()
            if not line or line.startswith("find:"):
                continue
            _remember(line)

        ranked = sorted(
            scored.items(),
            key=lambda item: (-item[1], self._repo_relpath(item[0])),
        )
        return [path for path, _score in ranked[:limit]]

    def _producer_command_for_path(self, path: str) -> Optional[str]:
        mapped = self._map_repo_path(path)
        if not self._looks_like_hidden_producer_path(mapped):
            return None
        return self._runnable_path_command(
            mapped,
            wrap_timeout=True,
            timeout_seconds=self._producer_command_timeout_seconds(mapped),
        )

    async def _ensure_workspace_root(self, environment) -> str:
        pwd_out = await self._safe_exec(environment, "pwd", _TIMEOUT_PROBE)
        pwd_lines = [line for line in self._output_lines(pwd_out) if line.startswith("/")]
        boot_cwd = pwd_lines[-1] if pwd_lines else "/"
        alias_enabled = False
        root = boot_cwd if boot_cwd.startswith("/") else "/app"

        self._workspace_root = root if root.startswith("/") else "/app"
        self._state["repo"]["workspace_root"] = self._workspace_root
        self._state["repo"]["workspace_boot_cwd"] = boot_cwd
        self._state["repo"]["workspace_alias_enabled"] = alias_enabled
        return self._workspace_root

    async def _maybe_promote_nested_workspace_root(self, environment) -> str:
        current = self._workspace_root_path()
        marker_checks = " ".join(f"{_shq(name)}" for name in _REPO_MARKER_FILES)
        marker_names = " -o ".join(f"-name {_shq(name)}" for name in _REPO_MARKER_FILES)
        for _ in range(4):
            probe = await self._safe_exec(
                environment,
                f"cd {_shq(current)} || exit 0\n"
                "root_has_marker=0\n"
                f"for name in {marker_checks}; do [ -e \"$name\" ] && root_has_marker=1; done\n"
                "[ -d .git ] && root_has_marker=1\n"
                "echo ROOT_HAS_MARKER=$root_has_marker\n"
                "find . -mindepth 1 -maxdepth 2 "
                "\\( -type d -name .git -o -type f \\( "
                f"{marker_names}"
                " \\) \\) -print 2>/dev/null | "
                "sed 's#^\\./##' | sed 's#/.git$##' | cut -d/ -f1 | sort -u | sed -n '1,8p'",
                _TIMEOUT_PROBE,
            )
            lines = [line.strip() for line in self._output_lines(probe) if line.strip()]
            root_has_marker = "ROOT_HAS_MARKER=1" in lines
            candidates = [
                line
                for line in lines
                if not line.startswith("ROOT_HAS_MARKER=") and "/" not in line and line != "."
            ]
            if root_has_marker or len(candidates) != 1:
                break
            promoted = self._repo_path(candidates[0])
            if promoted == current:
                break
            self._workspace_root = promoted
            self._state["repo"]["workspace_root"] = promoted
            current = promoted
        return current

    def _instruction_terms(self) -> List[str]:
        terms: List[str] = []
        seen: set[str] = set()

        def _add_candidate(candidate: str) -> None:
            cleaned = candidate.strip()
            if not cleaned:
                return
            key = cleaned.lower()
            if key in seen:
                return
            terms.append(cleaned)
            seen.add(key)

        def _add_candidate_fragments(candidate: str) -> None:
            _add_candidate(candidate)
            cleaned = candidate.strip().strip(".,:;()[]{}")
            if not cleaned or _is_workspace_repo_path(cleaned):
                return
            for piece in re.split(r"[-_/.:]+", cleaned):
                fragment = piece.strip().strip(".,:;()[]{}")
                lowered = fragment.lower()
                if (
                    not fragment
                    or lowered in _INSTRUCTION_SEARCH_STOPWORDS
                    or lowered in _COMMON_COMMAND_HINTS
                    or lowered in _COMMON_FILE_SUFFIXES
                ):
                    continue
                if len(fragment) < 4 and not fragment.isupper() and not any(ch.isdigit() for ch in fragment):
                    continue
                _add_candidate(fragment)

        for target in self._task_targets:
            tag, _, raw_value = target.partition(": ")
            value = raw_value.strip()
            if not value:
                continue
            if tag == "command_line":
                candidates = []
                words = _shell_split(value)
                head = _command_head(value)
                if head:
                    candidates.append(head)
                for word in words[1:]:
                    cleaned = str(word or "").strip().strip(".,:;()[]{}")
                    if not cleaned or cleaned in {">", ">>", "1>", "1>>", "2>", "2>>", "<", "<<", "|", "||", "&&", ";", "&"}:
                        continue
                    if _is_workspace_repo_path(cleaned):
                        candidates.extend([cleaned, os.path.basename(cleaned)])
                    elif re.search(
                        r"\.(" + "|".join(_COMMON_FILE_SUFFIXES) + r")$",
                        cleaned,
                        flags=re.IGNORECASE,
                    ):
                        candidates.append(cleaned)
            elif tag == "port":
                candidates = [f"port {value}", f":{value}", value]
            elif tag == "socket":
                candidates = [value, os.path.basename(value)]
            elif "/" in value:
                candidates = [value, os.path.basename(value)]
            else:
                candidates = [value]
            for candidate in candidates:
                _add_candidate_fragments(candidate)
            if len(terms) >= 12:
                break

        raw_instruction = str(self._task_instruction or "")
        for match in re.finditer(r'["\']([^"\']{4,80})["\']', raw_instruction):
            phrase = re.sub(r"\s+", " ", match.group(1).strip())
            if not phrase or _is_workspace_repo_path(phrase) or _looks_like_shell_command_line(phrase):
                continue
            if phrase.lower() in _INSTRUCTION_SEARCH_STOPWORDS:
                continue
            _add_candidate_fragments(phrase)
            if len(terms) >= 18:
                return terms[:18]

        for token in re.findall(r"\b[A-Za-z][A-Za-z0-9_./-]{3,}\b", raw_instruction):
            cleaned = token.strip().strip(".,:;()[]{}")
            lowered = cleaned.lower()
            if not cleaned or lowered in _INSTRUCTION_SEARCH_STOPWORDS:
                continue
            if _is_workspace_repo_path(cleaned) or lowered in _COMMON_COMMAND_HINTS:
                continue
            if not (
                any(ch.isupper() for ch in cleaned)
                or any(ch.isdigit() for ch in cleaned)
                or "/" in cleaned
                or "-" in cleaned
                or len(cleaned) >= 7
            ):
                continue
            _add_candidate_fragments(cleaned)
            if len(terms) >= 18:
                break
        return terms[:18]

    def _instruction_terms_block(self) -> str:
        terms = self._instruction_terms()
        if not terms:
            return ""
        return (
            "INSTRUCTION SEARCH TERMS / EXACT PHRASES:\n"
            + "\n".join(f"  - {term}" for term in terms[:18])
        )

    def _task_targets_block(self, *, include_instruction: bool = False) -> str:
        parts: List[str] = []
        workspace_root = self._workspace_root_path()
        if workspace_root != "/app":
            parts.append(f"DETECTED WORKSPACE ROOT:\n  - {workspace_root}")
        if include_instruction and self._task_instruction:
            parts.append(
                "TASK INSTRUCTION (anchor):\n"
                f"{_clip_inline(self._task_instruction, _MAX_INLINE_INSTRUCTION_CHARS)}"
            )
        if self._task_targets:
            parts.append(
                "TASK TARGETS EXTRACTED FROM INSTRUCTION:\n"
                + "\n".join(f"  - {item}" for item in self._task_targets[:20])
            )
        instruction_terms = self._instruction_terms_block()
        if instruction_terms:
            parts.append(instruction_terms)
        return "\n\n".join(parts).strip()

    def _extract_repo_command_candidates(self, text: str) -> List[str]:
        candidates: List[str] = []
        seen: set[str] = set()
        for raw in str(text or "").splitlines():
            stripped = raw.strip()
            if not stripped or stripped.startswith(("```", "# ", "//")):
                continue
            stripped = re.sub(r"^[*-]\s+", "", stripped)
            if stripped.startswith("$ "):
                stripped = stripped[2:].strip()
            if stripped.startswith("`") and stripped.endswith("`") and len(stripped) > 2:
                stripped = stripped[1:-1].strip()
            lowered = stripped.lower()
            if any(token in lowered for token in ("/tests/", "pytest", "test.sh", "verify.sh")):
                continue
            head = _command_head(stripped).lower()
            if head in {"apt", "apt-get", "pip", "pip3", "curl", "wget"}:
                continue
            if head == "uv" and " run " not in f" {lowered} ":
                continue
            if not (
                _looks_like_shell_command_line(stripped)
                or head in {"python", "python3", "bash", "sh", "make", "node", "npm", "npx", "cargo", "go", "uv"}
            ):
                continue
            if len(stripped) > 220:
                stripped = stripped[:217] + "..."
            if stripped in seen:
                continue
            candidates.append(stripped)
            seen.add(stripped)
            if len(candidates) >= _MAX_DISCOVERY_COMMAND_ITEMS:
                break
        return candidates

    def _remember_repo_command_candidates(self, text: str) -> None:
        merged = list(
            dict.fromkeys(
                self._repo_command_candidates + self._extract_repo_command_candidates(text)
            )
        )
        self._repo_command_candidates = merged[:_MAX_DISCOVERY_COMMAND_ITEMS]
        self._state["repo"]["command_candidates"] = self._repo_command_candidates[:]

    def _looks_like_observed_probe_path(self, path: str) -> bool:
        rendered = str(path or "").strip().rstrip("/")
        if not rendered or rendered in {"/", "/app", "/apps", "/tmp", "/run", "/var/run", "/var/www", "/git", "/home"}:
            return False
        if rendered.startswith("/tests/"):
            return False
        if _is_workspace_repo_path(rendered):
            rel = self._repo_relpath(rendered).lstrip("./")
            if _should_ignore_repo_search_path(rel):
                return False
        if not (
            _is_workspace_repo_path(rendered)
            or rendered.startswith(("/tmp/", "/run/", "/var/run/", "/var/www/", "/git/", "/home/"))
        ):
            return False
        base = os.path.basename(rendered).lower()
        if not base or base in _LOCAL_VERIFIER_FILENAMES:
            return False
        if rendered.endswith(".sock"):
            return True
        if "." in base:
            return True
        return rendered.count("/") >= 3

    def _extract_observed_probe_paths(self, text: str) -> List[str]:
        paths: List[str] = []
        seen: set[str] = set()
        for match in re.finditer(
            r"(/(?:app|apps|tmp|run|var/run|var/www|git|home)/[A-Za-z0-9_./:-]+)",
            str(text or ""),
        ):
            candidate = str(match.group(1) or "").strip().strip(".,:;()[]{}'\"<>")
            if not candidate:
                continue
            mapped = self._map_repo_path(candidate) if _is_workspace_repo_path(candidate) else candidate
            if not self._looks_like_observed_probe_path(mapped):
                continue
            if mapped in seen:
                continue
            paths.append(mapped)
            seen.add(mapped)
            if len(paths) >= _OBSERVED_PROBE_PATHS_MAX:
                break
        return paths

    def _remember_observed_probe_paths(self, text: str) -> None:
        merged = list(
            dict.fromkeys(self._observed_probe_paths + self._extract_observed_probe_paths(text))
        )
        self._observed_probe_paths = merged[:_OBSERVED_PROBE_PATHS_MAX]
        self._state["repo"]["observed_probe_paths"] = self._observed_probe_paths[:]

    def _hidden_is_target_output_path(self, source_path: str) -> bool:
        mapped = self._repo_path(source_path)
        base = os.path.basename(mapped).lower()
        for target in list(self._task_targets) + list(self._assertion_targets):
            tag, _, value = target.partition(": ")
            value = value.strip()
            if tag not in {"path", "artifact", "file", "must_exist"} or not value:
                continue
            target_path = self._repo_path(value)
            target_base = os.path.basename(target_path).lower()
            if mapped == target_path or (base and base == target_base):
                return True
        return False

    def _hidden_literal_payload_score(
        self,
        *,
        payload: str,
        source_path: str,
        lowered_item: str,
        terms: List[str],
    ) -> int:
        stripped_payload = payload.strip()
        lowered_payload = stripped_payload.lower()
        source_base = os.path.basename(source_path).lower() if source_path else ""
        score = 0

        for term in terms:
            if not term:
                continue
            if term in lowered_payload:
                score += 5 if len(term) >= 6 else 2
            elif term in lowered_item:
                score += 3 if len(term) >= 6 else 1
            if source_base and term in source_base:
                score += 4 if len(term) >= 6 else 2

        if source_path and _path_is_textish(source_path):
            score += 2
        if source_path and self._hidden_is_target_output_path(source_path):
            score -= 12
        if any(hint in source_base for hint in _TARGET_OUTPUT_NAME_HINTS):
            score -= 6
        if any(hint in lowered_payload for hint in _LOW_SIGNAL_LITERAL_HINTS):
            score -= 8
        if (
            re.search(r"\b20\d{2}[-/]\d{2}[-/]\d{2}\b", stripped_payload)
            and not any(term in lowered_payload for term in terms)
        ):
            score -= 5
        if len(stripped_payload.split()) > 16 and not any(term in lowered_payload for term in terms):
            score -= 4
        if re.fullmatch(r"[A-Z0-9]{8,}", stripped_payload):
            score += 6
        elif re.fullmatch(r"[a-h][1-8][a-h][1-8](?:\s+[a-h][1-8][a-h][1-8])+", stripped_payload):
            score += 5
        elif re.fullmatch(r"[a-h][1-8][a-h][1-8]", stripped_payload):
            score += 4
        elif any(ch.isalpha() for ch in stripped_payload) and any(ch.isdigit() for ch in stripped_payload):
            score += 1
        return score

    def _hidden_candidate_literal_values(self, *, limit: int = 4) -> List[str]:
        pool = self._hidden_copy_first_candidates(limit=max(limit * 2, limit))

        literals: List[str] = []
        seen: set[str] = set()
        for item in pool:
            lowered = str(item or "").lower()
            if lowered.endswith("[path match]"):
                continue
            payload = str(item or "")
            if ": " in payload:
                _prefix, _sep, payload = payload.partition(": ")
            payload = payload.strip()
            if (
                not payload
                or payload.startswith(("/", "./"))
                or len(payload) < 3
                or len(payload) > 220
                or _looks_like_sourceish_line(payload)
                or _looks_like_commandish_fragment(payload)
            ):
                continue
            if payload in seen:
                continue
            literals.append(payload)
            seen.add(payload)
            if len(literals) >= limit:
                break
        return literals

    def _hidden_copy_first_candidates(self, *, limit: int = _MAX_COPY_FIRST_CANDIDATES) -> List[str]:
        pool = list(
            dict.fromkeys(self._hidden_exact_visible_matches + self._hidden_visible_text_candidates)
        )
        if not pool:
            return []
        terms = [term.lower() for term in self._instruction_terms()[:18]]
        ranked: List[Tuple[int, int, str]] = []
        for idx, item in enumerate(pool):
            lowered = item.lower()
            source_path = ""
            payload = item
            if lowered.endswith("[path match]"):
                source_path = item.rsplit(" [path match]", 1)[0]
                payload = os.path.basename(source_path)
            elif ": " in item:
                prefix, _, rest = item.partition(": ")
                source_path = prefix.split(":", 1)[0]
                payload = rest
            if source_path and self._hidden_is_target_output_path(source_path):
                continue
            if _looks_like_commandish_fragment(payload):
                continue
            score = self._hidden_literal_payload_score(
                payload=payload,
                source_path=source_path,
                lowered_item=lowered,
                terms=terms,
            )
            if ": " in item and not lowered.endswith("[path match]"):
                score += 3
            if lowered.endswith("[path match]"):
                score -= 1
            ranked.append((score, -idx, item))
        ranked.sort(reverse=True)
        picked = [item for score, _, item in ranked if score > 1][:limit]
        return picked[:limit]

    def _hidden_copy_first_block(self) -> str:
        candidates = self._hidden_copy_first_candidates()
        if not candidates:
            return ""
        return (
            "COPY-FIRST LITERAL CANDIDATES:\n"
            + "\n".join(f"  - {item}" for item in candidates)
            + "\nRULE: Only copy a candidate when it clearly matches the task terms and comes from real source/data evidence, not from a target output file echo, placeholder, or dummy status line."
        )

    def _hidden_copy_first_preferred(self) -> bool:
        if not self._uses_hidden_tests_fallback():
            return False
        if self._hidden_service_targets_present():
            return False
        return bool(self._hidden_copy_first_candidates())

    def _hidden_service_targets_present(self) -> bool:
        return any(target.startswith(("socket: ", "port: ")) for target in self._task_targets)

    def _hidden_socket_probe_candidates(self, value: str, *, limit: int = 6) -> List[str]:
        candidates: List[str] = []
        seen: set[str] = set()

        def _remember(path: str) -> None:
            rendered = str(path or "").strip()
            if not rendered or rendered in seen:
                return
            candidates.append(rendered)
            seen.add(rendered)

        for raw in _extract_socket_targets(value):
            cleaned = str(raw or "").strip()
            if not cleaned:
                continue
            if cleaned.startswith("/"):
                _remember(cleaned)
                base = os.path.basename(cleaned)
            else:
                base = os.path.basename(cleaned)
                if "/" in cleaned:
                    _remember(self._repo_path(cleaned))
                    _remember("/" + cleaned.lstrip("./"))
                else:
                    _remember(cleaned)
            if base:
                for prefix in ("/tmp", "/run", "/var/run"):
                    _remember(f"{prefix}/{base}")
        return candidates[:limit]

    def _hidden_service_handoff_allowed(self) -> bool:
        if self._success_markers:
            return True
        local_verifier_files = self._state.get("repo", {}).get("local_verifier_files") or []
        return bool(local_verifier_files) and not self._local_verifier_mentions_hidden_tests()

    def _hidden_rich_retry_targets_present(self) -> bool:
        return any(
            target.startswith(("producer_candidate: ", "port: ", "socket: "))
            for target in self._task_targets
        ) or bool(
            self._hidden_exact_visible_matches
            or self._hidden_visible_text_candidates
            or self._hidden_process_probe_terms(limit=2)
        )

    def _hidden_unresolved_signal_present(self, text: Optional[str] = None) -> bool:
        rendered = "\n".join(
            part
            for part in (
                text,
                self._last_test_output,
                self._state.get("test", {}).get("last_tail"),
            )
            if part
        )
        lower = rendered.lower()
        if not rendered:
            return False
        return any(
            marker in lower
            for marker in (
                "missing socket",
                "missing port",
                "missing process",
                "multiple_pids=",
                "command not found",
                "traceback",
                "assertionerror",
                "test failed",
                "no such file or directory",
                "[timeout",
                "[error]",
                "producer_failure",
                "smoke_failure",
                "text_target_empty",
                "text_target_no_literal_match",
            )
        )

    def _hidden_service_proof_complete(self, text: Optional[str] = None) -> bool:
        if not self._hidden_service_targets_present():
            return True

        rendered = "\n".join(
            part for part in (text, self._last_test_output, self._state.get("test", {}).get("last_tail")) if part
        )
        lower = rendered.lower()
        if not rendered:
            return False
        if any(marker in lower for marker in ("missing socket", "missing port", "missing process", "multiple_pids=")):
            return False

        sockets_required = any(target.startswith("socket: ") for target in self._task_targets)
        ports_required = any(target.startswith("port: ") for target in self._task_targets)
        process_terms = self._hidden_process_probe_terms(limit=2)

        if sockets_required and "=== socket " not in lower:
            return False
        if ports_required and "=== port " not in lower:
            return False
        if process_terms and "=== process " not in lower:
            return False
        if process_terms and "single_pid_ok=1" not in lower:
            return False
        if process_terms and "process_cmdline_ok=1" not in lower:
            return False
        return True

    def _hidden_advisory_ready_for_handoff(self) -> bool:
        if not self._uses_hidden_tests_fallback():
            return False
        if not self._hidden_service_targets_present():
            return False
        if not self._hidden_service_handoff_allowed():
            return False
        last_rc = self._state.get("test", {}).get("last_rc")
        if last_rc is None or int(last_rc) != 0:
            return False
        if not self._runner_is_rich_hidden_local():
            return False

        rendered = "\n".join(
            part for part in (self._last_test_output, self._state.get("test", {}).get("last_tail")) if part
        )
        lower = rendered.lower()
        if not rendered:
            return False
        if any(
            marker in lower
            for marker in (
                "missing socket",
                "missing port",
                "missing process",
                "missing /",
                "missing ./",
                "command not found",
                "traceback",
                "assertionerror",
                "test failed",
                "no such file or directory",
                "[timeout",
                "[error]",
                "producer_failure",
                "smoke_failure",
                "text_target_empty",
                "text_target_no_literal_match",
            )
        ):
            return False
        return self._hidden_service_proof_complete(rendered)

    def _hidden_process_probe_terms(self, *, limit: int = 4) -> List[str]:
        terms: List[str] = []
        seen: set[str] = set()
        generic = {
            "bash",
            "cat",
            "curl",
            "file",
            "git",
            "grep",
            "head",
            "ls",
            "make",
            "node",
            "npm",
            "npx",
            "pip",
            "pip3",
            "pytest",
            "python",
            "python3",
            "rg",
            "sed",
            "sh",
            "tail",
            "timeout",
            "uv",
            "uvx",
            "wget",
        }

        def _remember(value: str) -> None:
            cleaned = re.sub(r"[^A-Za-z0-9._:+/-]", "", str(value or "").strip().lower())
            if not cleaned or cleaned in generic or cleaned in seen:
                return
            if cleaned in {"monitor", "server", "service", "socket"}:
                return
            terms.append(cleaned)
            seen.add(cleaned)

        def _remember_path_tokens(value: str) -> None:
            base = os.path.basename(str(value or "").strip()).lower()
            stem = os.path.splitext(base)[0]
            for token in re.split(r"[^a-z0-9]+", stem):
                if len(token) < 4 and token != "qemu":
                    continue
                _remember(token)

        for target in self._task_targets:
            tag, _, value = target.partition(": ")
            value = value.strip()
            if not value:
                continue
            if tag == "command":
                _remember(value.split()[0])
            elif tag == "command_line":
                head = _command_head(value).lower()
                if head:
                    _remember(head)
            elif tag == "producer_candidate":
                _remember_path_tokens(value)
            elif tag == "socket":
                _remember_path_tokens(value)
                for token in re.split(r"[^a-z0-9]+", os.path.basename(value).lower()):
                    if token == "qemu":
                        _remember(token)

        for candidate in self._repo_command_candidates[:8]:
            head = _command_head(candidate).lower()
            if head:
                _remember(head)

        return terms[:limit]

    def _hidden_service_focus_block(self) -> str:
        lines: List[str] = []
        process_terms = self._hidden_process_probe_terms(limit=4)
        if process_terms:
            lines.append("PROCESS / SERVICE PROOF TARGETS:")
            for term in process_terms:
                lines.append(f"  - single PID plus /proc/<pid>/cmdline for {term}")
        sockets = [
            target.split(": ", 1)[1]
            for target in self._task_targets
            if target.startswith("socket: ")
        ]
        ports = [
            target.split(": ", 1)[1]
            for target in self._task_targets
            if target.startswith("port: ")
        ]
        if sockets:
            if not lines:
                lines.append("PROCESS / SERVICE PROOF TARGETS:")
            expanded_sockets: List[str] = []
            for path in sockets[:4]:
                expanded_sockets.extend(self._hidden_socket_probe_candidates(path, limit=4))
            for path in list(dict.fromkeys(expanded_sockets or sockets))[:6]:
                lines.append(f"  - required unix socket: {path}")
        if ports:
            if not lines:
                lines.append("PROCESS / SERVICE PROOF TARGETS:")
            for port in ports[:4]:
                lines.append(f"  - required listening port: {port}")
        return "\n".join(lines)

    def _hidden_refresh_needed(self, text: str) -> bool:
        lower = str(text or "").lower()
        return any(
            marker in lower
            for marker in (
                "missing socket",
                "missing port",
                "missing process",
                "missing /",
                "missing ./",
                "command not found",
                "traceback",
                "assertionerror",
                "test failed",
                "no such file or directory",
                "[timeout",
                "[error]",
                "producer_failure",
                "smoke_failure",
                "text_target_empty",
                "text_target_no_literal_match",
            )
        )

    def _should_refresh_hidden_repo_evidence(
        self,
        *,
        turn: int,
        cached_text: str,
        obs_text: str,
    ) -> bool:
        if turn <= 2 or not cached_text:
            return True
        if not self._runner_is_rich_hidden_local():
            return turn <= 4 or self._hidden_refresh_needed(obs_text)
        return self._hidden_refresh_needed(obs_text) and int(
            self._state.get("progress", {}).get("stagnation") or 0
        ) >= 2

    def _should_refresh_hidden_discovery(
        self,
        *,
        turn: int,
        cached_text: str,
        obs_text: str,
    ) -> bool:
        if turn == 1 or not cached_text:
            return True
        if not self._runner_is_rich_hidden_local():
            return turn <= 3 or self._hidden_refresh_needed(obs_text)
        return False

    def _should_refresh_hidden_producer_context(
        self,
        *,
        turn: int,
        cached_text: str,
        obs_text: str,
        hints_changed: bool,
    ) -> bool:
        if turn == 1 or not cached_text or hints_changed:
            return True
        return (not self._runner_is_rich_hidden_local()) and self._hidden_refresh_needed(obs_text)

    def _should_force_hidden_evidence_retry(self, *, turn: int) -> bool:
        if not self._uses_hidden_tests_fallback():
            return False
        if self._runner_is_rich_hidden_local():
            return False
        if self._state.get("test", {}).get("last_success") is True:
            return False
        if bool(self._state.get("progress", {}).get("forced_hidden_evidence_retry")):
            return False
        if turn < max(4, _LOW_SIGNAL_HANDOFF_MIN_TURNS - 1):
            return False
        if int(self._state.get("progress", {}).get("stagnation") or 0) < max(2, _LOW_SIGNAL_HANDOFF_MIN_STAGNATION - 1):
            return False
        return self._hidden_rich_retry_targets_present()

    async def _resolve_instruction_validation_cmd(self, environment) -> Optional[Tuple[str, str]]:
        literal_candidates = self._hidden_candidate_literal_values(limit=4)

        def _probe_lines_for_path(path: str) -> List[str]:
            display = path.replace("'", "")
            is_textish = _path_is_textish(path)
            lines = [
                f"echo '=== {display} ==='",
                f"if [ -e {_shq(path)} ]; then",
                f"  ls -la {_shq(path)}",
            ]
            if is_textish:
                lines.extend(
                    [
                        f"  if [ -f {_shq(path)} ]; then head -120 {_shq(path)} 2>/dev/null || true; fi",
                        f"  if [ -f {_shq(path)} ]; then",
                        f"    if [ ! -s {_shq(path)} ]; then echo 'TEXT_TARGET_EMPTY {display}'; fi",
                        f"    first_line=$(head -5 {_shq(path)} 2>/dev/null | sed '/^$/d' | paste -sd ' ' - | cut -c1-220)",
                        "    if [ -n \"$first_line\" ]; then",
                        f"      printf 'TEXT_TARGET_CONTENT {display} :: %s\\n' \"$first_line\"",
                        "    fi",
                    ]
                )
                if literal_candidates:
                    lines.append("    text_target_literal_hit=0")
                    for idx, literal in enumerate(literal_candidates[:4], 1):
                        lines.append(
                            f"    if grep -Fqx -- {_shq(literal)} {_shq(path)} 2>/dev/null; then "
                            f"echo 'TEXT_TARGET_EXACT_MATCH {display} :: candidate {idx}'; "
                            "text_target_literal_hit=1; fi"
                        )
                    lines.extend(
                        [
                            "    if [ \"$text_target_literal_hit\" = \"0\" ]; then",
                            f"      echo 'TEXT_TARGET_NO_LITERAL_MATCH {display}'",
                            "    fi",
                        ]
                    )
                lines.append("  fi")
            else:
                lines.append(f"  if [ -f {_shq(path)} ]; then file {_shq(path)} 2>/dev/null || true; fi")
                lines.append(f"  if [ -f {_shq(path)} ]; then wc -c {_shq(path)} 2>/dev/null || true; fi")
            lines.extend(
                [
                    "else",
                    f"  echo 'MISSING {display}'",
                    "fi",
                ]
            )
            return lines

        def _probe_lines_for_socket(path: str) -> List[str]:
            display = path.replace("'", "")
            return [
                f"echo '=== socket {display} ==='",
                f"if [ -S {_shq(path)} ]; then ls -la {_shq(path)}; else echo 'MISSING SOCKET {display}'; fi",
                (
                    "(ss -lxnp 2>/dev/null || ss -lxn 2>/dev/null || netstat -lxnp 2>/dev/null || true) "
                    f"| grep -F {_shq(path)} || true"
                ),
            ]

        def _probe_lines_for_port(port: str) -> List[str]:
            lines = [
                f"echo '=== port {port} ==='",
                (
                    "(ss -ltnp 2>/dev/null || netstat -ltnp 2>/dev/null || true) "
                    f"| grep -E '[:.]({re.escape(port)})\\b' || echo 'MISSING PORT {port}'"
                ),
            ]
            if port in {"80", "443", "3000", "5000", "8000", "8080", "8443"}:
                scheme = "https" if port in {"443", "8443"} else "http"
                lines.append(
                    f"if command -v curl >/dev/null 2>&1; then "
                    f"curl -I --max-time 5 {scheme}://127.0.0.1:{port} 2>/dev/null | head -20 || true; "
                    "fi"
                )
            return lines

        def _probe_lines_for_process(term: str) -> List[str]:
            safe_term = re.sub(r"[^A-Za-z0-9._:+/-]", "", str(term or "").strip())
            if not safe_term:
                return []
            display = safe_term.replace("'", "")
            return [
                f"echo '=== process {display} ==='",
                f"pids=$(pgrep -f {_shq(safe_term)} 2>/dev/null | sed -n '1,10p')",
                "if [ -n \"$pids\" ]; then",
                "  pid_count=$(printf '%s\\n' \"$pids\" | sed '/^$/d' | wc -l | tr -d ' ')",
                "  echo PID_COUNT=$pid_count",
                f"  pgrep -af {_shq(safe_term)} 2>/dev/null | sed -n '1,20p'",
                "  if [ \"$pid_count\" = \"1\" ]; then",
                "    echo SINGLE_PID_OK=1",
                "    first_pid=$(printf '%s\\n' \"$pids\" | sed -n '1p')",
                "    if [ -r \"/proc/$first_pid/cmdline\" ]; then",
                "      echo '--- /proc/'\"$first_pid\"'/cmdline ---'",
                "      tr '\\0' ' ' < \"/proc/$first_pid/cmdline\" 2>/dev/null || true",
                "      echo",
                "      echo PROCESS_CMDLINE_OK=1",
                "    fi",
                "  else",
                "    echo MULTIPLE_PIDS=$pid_count",
                "  fi",
                "else",
                f"  echo 'MISSING PROCESS {display}'",
                "fi",
            ]

        command_lines: List[str] = []
        probe_paths: List[str] = []
        producer_paths: List[str] = []
        probe_commands: List[str] = []
        probe_ports: List[str] = []
        probe_sockets: List[str] = []
        seen_command_lines: set[str] = set()
        seen_paths: set[str] = set()
        seen_producers: set[str] = set()
        seen_commands: set[str] = set()
        seen_ports: set[str] = set()
        seen_sockets: set[str] = set()

        def _remember_path(value: str) -> None:
            path = self._repo_path(value)
            if path not in seen_paths:
                probe_paths.append(path)
                seen_paths.add(path)

        def _remember_producer(value: str) -> None:
            path = self._repo_path(value)
            if path in seen_producers:
                return
            if not self._looks_like_hidden_producer_path(path):
                return
            producer_paths.append(path)
            seen_producers.add(path)

        def _remember_port(value: str) -> None:
            port = str(value or "").strip()
            if not port or port in seen_ports:
                return
            if not port.isdigit():
                return
            n = int(port)
            if n <= 0 or n > 65535:
                return
            probe_ports.append(port)
            seen_ports.add(port)

        def _remember_socket(value: str) -> None:
            for path in self._hidden_socket_probe_candidates(value):
                if not path or path in seen_sockets:
                    continue
                probe_sockets.append(path)
                seen_sockets.add(path)

        for target in self._task_targets:
            tag, _, value = target.partition(": ")
            value = value.strip()
            if not value:
                continue
            if tag == "command_line":
                if value not in seen_command_lines and _looks_like_shell_command_line(value):
                    command_lines.append(value)
                    seen_command_lines.add(value)
                for extra_tag, extra_value in _command_line_targets(value):
                    if extra_tag in {"path", "artifact", "file"}:
                        _remember_path(extra_value)
                        _remember_producer(extra_value)
                    elif extra_tag == "socket":
                        _remember_socket(extra_value)
                    elif extra_tag == "command":
                        command = extra_value.split()[0]
                        if command not in seen_commands:
                            probe_commands.append(command)
                            seen_commands.add(command)
            elif tag in {"path", "artifact", "file"}:
                _remember_path(value)
                _remember_producer(value)
            elif tag == "producer_candidate":
                _remember_producer(value)
            elif tag == "command":
                command = value.split()[0]
                if command not in seen_commands:
                    probe_commands.append(command)
                    seen_commands.add(command)
            elif tag == "port":
                _remember_port(value)
            elif tag == "socket":
                _remember_socket(value)

        for observed_path in self._observed_probe_paths[:_OBSERVED_PROBE_PATHS_MAX]:
            if observed_path.endswith(".sock"):
                _remember_socket(observed_path)
                continue
            _remember_path(observed_path)
            _remember_producer(observed_path)

        for item in self._hidden_exact_visible_matches + self._hidden_visible_text_candidates:
            for socket in _extract_socket_targets(item):
                _remember_socket(socket)

        if not command_lines:
            for candidate in self._repo_command_candidates[:4]:
                if candidate in seen_command_lines:
                    continue
                command_lines.append(candidate)
                seen_command_lines.add(candidate)
                for extra_tag, extra_value in _command_line_targets(candidate):
                    if extra_tag in {"path", "artifact", "file"}:
                        _remember_path(extra_value)
                        _remember_producer(extra_value)
                    elif extra_tag == "socket":
                        _remember_socket(extra_value)
                    elif extra_tag == "command":
                        command = extra_value.split()[0]
                        if command not in seen_commands:
                            probe_commands.append(command)
                            seen_commands.add(command)

        if not producer_paths:
            for path in await self._collect_hidden_producer_paths(environment, limit=6):
                _remember_producer(path)

        process_terms = self._hidden_process_probe_terms(limit=4)
        lines: List[str] = ["set +e", self._workspace_cd()]
        evidence_paths = self._hidden_evidence_probe_paths(limit=4)
        used_hidden_evidence = False
        for path in evidence_paths:
            if path not in seen_paths:
                probe_paths.append(path)
                seen_paths.add(path)
                used_hidden_evidence = True
        if command_lines:
            inspect_paths: List[str] = []
            seen_inspect_paths: set[str] = set()
            for command_line in command_lines[:2]:
                for extra_tag, extra_value in _command_line_targets(command_line):
                    if extra_tag not in {"path", "artifact", "file"}:
                        if extra_tag == "socket":
                            _remember_socket(extra_value)
                        continue
                    path = self._repo_path(extra_value)
                    if path in seen_inspect_paths:
                        continue
                    inspect_paths.append(path)
                    seen_inspect_paths.add(path)
            for path in probe_paths[:4]:
                if path in seen_inspect_paths:
                    continue
                inspect_paths.append(path)
                seen_inspect_paths.add(path)
            for path in evidence_paths[:4]:
                if path in seen_inspect_paths:
                    continue
                inspect_paths.append(path)
                seen_inspect_paths.add(path)

            for idx, command_line in enumerate(command_lines[:2], 1):
                runnable = self._normalize_probe_command_line(command_line)
                display = runnable.replace("'", "")
                lines.extend(
                    [
                        f"echo '=== smoke {idx}: {display} ==='",
                        runnable,
                        f"smoke_rc_{idx}=$?",
                        f"echo __SMOKE_RC_{idx}__=$smoke_rc_{idx}",
                        f"if [ \"$smoke_rc_{idx}\" -ne 0 ]; then echo \"SMOKE_FAILURE {idx} rc=$smoke_rc_{idx}\"; fi",
                    ]
                )

            for path in inspect_paths[:4]:
                lines.extend(_probe_lines_for_path(path))
            for socket in probe_sockets[:4]:
                lines.extend(_probe_lines_for_socket(socket))
            for port in probe_ports[:4]:
                lines.extend(_probe_lines_for_port(port))
            for term in process_terms:
                lines.extend(_probe_lines_for_process(term))
            return "\n".join(lines), "local_instruction_smoke"

        if producer_paths:
            for idx, path in enumerate(producer_paths[:2], 1):
                command = self._producer_command_for_path(path)
                if not command:
                    continue
                display = path.replace("'", "")
                lines.extend(
                    [
                        f"echo '=== producer {idx}: {display} ==='",
                        command,
                        f"producer_rc_{idx}=$?",
                        f"echo __PRODUCER_RC_{idx}__=$producer_rc_{idx}",
                        f"if [ \"$producer_rc_{idx}\" -ne 0 ]; then echo \"PRODUCER_FAILURE {idx} rc=$producer_rc_{idx}\"; fi",
                    ]
                )
            for path in probe_paths[:4]:
                lines.extend(_probe_lines_for_path(path))
            for socket in probe_sockets[:4]:
                lines.extend(_probe_lines_for_socket(socket))
            for port in probe_ports[:4]:
                lines.extend(_probe_lines_for_port(port))
            for term in process_terms:
                lines.extend(_probe_lines_for_process(term))
            return "\n".join(lines), "local_instruction_smoke"

        if probe_paths or probe_ports or probe_sockets:
            for path in probe_paths[:4]:
                lines.extend(_probe_lines_for_path(path))
            for socket in probe_sockets[:4]:
                lines.extend(_probe_lines_for_socket(socket))
            for port in probe_ports[:4]:
                lines.extend(_probe_lines_for_port(port))
            for term in process_terms:
                lines.extend(_probe_lines_for_process(term))
            runner = "local_visible_evidence_probe" if used_hidden_evidence else "local_target_probe"
            return "\n".join(lines), runner

        if probe_commands:
            for command in probe_commands[:6]:
                lines.extend(
                    [
                        f"echo '=== command {command} ==='",
                        f"command -v {_shq(command)} >/dev/null 2>&1 && command -v {_shq(command)} || echo 'MISSING {command}'",
                    ]
                )
            return "\n".join(lines), "local_command_probe"

        return None

    def _parse_pytest_nodeid(self, text: str) -> Optional[str]:
        m = re.search(r"^FAILED\s+([^\s]+::[^\s]+)", text, flags=re.MULTILINE)
        if m:
            return m.group(1).strip()
        m = re.search(r"^(?:ERROR|FAILED)\s+([^\s]+::[^\s]+)", text, flags=re.MULTILINE)
        if m:
            return m.group(1).strip()
        return None

    def _tail(self, text: str, max_lines: int = 80, max_chars: int = 2400) -> str:
        lines = (text or "").splitlines()
        tail = "\n".join(lines[-max_lines:])
        if len(tail) > max_chars:
            tail = tail[-max_chars:]
        return tail

    def _output_lines(self, text: str) -> List[str]:
        lines: List[str] = []
        for raw in (text or "").splitlines():
            line = raw.strip()
            if not line or line.startswith("$"):
                continue
            lines.append(line)
        return lines

    def _extract_visible_text_candidates(self, text: str) -> List[str]:
        candidates: List[str] = []
        seen: set[str] = set()
        current_path = ""
        collecting = False

        for raw in str(text or "").splitlines():
            stripped = raw.strip()
            header = re.match(r"^===\s+(.+?)\s+===$", stripped)
            if header:
                current_path = header.group(1).strip()
                collecting = _path_is_textish(current_path) and not _should_ignore_repo_search_path(current_path)
                continue

            if not collecting or not current_path or not stripped:
                continue
            if stripped.startswith(
                (
                    "$ ",
                    "MISSING ",
                    "RC=",
                    "SIGNALS:",
                    "TAIL:",
                    "cmd=",
                    "__SMOKE_RC_",
                    "__PRODUCER_RC_",
                    "[timeout",
                    "[error]",
                )
            ):
                continue
            if stripped.startswith("==="):
                continue
            if re.match(r"^[bcdlps-][rwxstST-]{9}\s+\d+", stripped):
                continue
            if re.fullmatch(r"[0-9eE+.,\\-]+", stripped):
                continue
            if _looks_like_sourceish_line(stripped):
                continue
            if _looks_like_commandish_fragment(stripped):
                continue
            if not re.search(r"[A-Za-z]", stripped) and not re.fullmatch(r"[A-Z0-9]{8,}", stripped):
                continue

            candidate = stripped
            if len(candidate) > 220:
                candidate = candidate[:217] + "..."
            snippet = f"{current_path}: {candidate}"
            if snippet in seen:
                continue
            candidates.append(snippet)
            seen.add(snippet)
            if len(candidates) >= _VISIBLE_TEXT_CANDIDATE_MAX_ITEMS:
                break

        return candidates

    def _remember_visible_text_candidates(self, text: str) -> None:
        merged = list(
            dict.fromkeys(
                self._hidden_visible_text_candidates + self._extract_visible_text_candidates(text)
            )
        )
        self._hidden_visible_text_candidates = merged[:_VISIBLE_TEXT_CANDIDATE_MAX_ITEMS]

    async def _hidden_sparse_repo_evidence(self, environment) -> str:
        listing = await self._safe_exec(
            environment,
            f"{self._workspace_cd()}\n"
            f"find . -maxdepth {_HIDDEN_PRODUCER_SEARCH_MAXDEPTH} {_find_search_prune_clause()} -type f "
            "-print | sort | sed -n '1,400p'",
            _TIMEOUT_PROBE,
        )
        ranked: List[Tuple[int, str, str]] = []
        for raw in self._output_lines(listing):
            line = raw.strip()
            if not line or line.startswith("find:"):
                continue
            path = self._repo_path(line)
            score = self._hidden_evidence_path_score(path)
            if score <= 0:
                continue
            ranked.append((score, self._repo_relpath(path), path))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        selected_paths = [path for _score, _rel, path in ranked[:_HIDDEN_EVIDENCE_MAX_FILES]]
        self._remember_hidden_evidence_paths(selected_paths)
        if not selected_paths:
            return ""

        parts = [
            "VISIBLE HIDDEN-MODE SOURCE / DATA PREVIEW (inspect these repo files before guessing or installing heavy search stacks):"
        ]
        for path in selected_paths:
            out = await self._safe_exec(
                environment,
                f"if [ -f {_shq(path)} ]; then echo '=== {path} ==='; head -80 {_shq(path)} 2>/dev/null || true; fi",
                _TIMEOUT_PROBE,
            )
            cleaned = str(out or "").strip()
            if not cleaned or cleaned.startswith("[error]") or cleaned.startswith("[timeout"):
                continue
            parts.append(cleaned)

        rendered = "\n\n".join(parts).strip()
        if rendered:
            self._remember_visible_text_candidates(rendered)
        return rendered

    def _agent_can_read_tests(self) -> bool:
        # Harbor uploads the official verifier into /tests during the verifier phase,
        # after the agent has already finished running. Agent-side repair turns
        # therefore must not rely on /tests being present.
        return False

    def _local_verifier_mentions_hidden_tests(self) -> bool:
        return bool(self._state.get("repo", {}).get("local_verifier_mentions_tests"))

    def _runner_is_authoritative(self, runner: Optional[str] = None) -> bool:
        value = str(runner or self._state.get("test", {}).get("runner") or "")
        if value in _OFFICIAL_AUTHORITATIVE_RUNNERS:
            return True
        return False

    def _runner_is_local_only(self, runner: Optional[str] = None) -> bool:
        return not self._runner_is_authoritative(runner)

    def _uses_hidden_tests_fallback(self) -> bool:
        return self._runner_is_local_only()

    def _agent_local_validation_enabled(self) -> bool:
        return _ENABLE_AGENT_LOCAL_VALIDATION

    def _disable_agent_local_validation_state(self) -> None:
        self._state["test"]["cmd"] = "disabled: Harbor official verifier runs after agent execution"
        self._state["test"]["runner"] = _POST_AGENT_VERIFIER_HANDOFF_RUNNER
        self._state["test"]["last_rc"] = None
        self._state["test"]["last_success"] = None
        self._state["test"]["last_nodeid"] = None
        self._state["test"]["last_tail"] = None
        self._last_test_output = ""

    def _runner_is_low_signal_local(self, runner: Optional[str] = None) -> bool:
        value = str(runner or self._state.get("test", {}).get("runner") or "")
        if value in _LOW_SIGNAL_LOCAL_RUNNERS:
            return True
        if value in {"local_instruction_smoke", "local_visible_evidence_probe"}:
            if not self._hidden_service_targets_present():
                return True
            return not self._hidden_service_proof_complete()
        return False

    def _runner_is_rich_hidden_local(self, runner: Optional[str] = None) -> bool:
        if not self._runner_is_local_only(runner) or self._runner_is_low_signal_local(runner):
            return False
        if not self._hidden_service_targets_present():
            return False
        return self._hidden_service_proof_complete()

    def _local_verifier_tokens(self) -> List[str]:
        tokens: List[str] = []
        for path in self._state.get("repo", {}).get("local_verifier_files") or []:
            rendered = str(path or "").strip()
            if not rendered:
                continue
            rel = self._repo_relpath(rendered)
            tokens.extend([rendered, rel, f"./{rel}"])
        return list(dict.fromkeys(token for token in tokens if token))

    def _repo_relpath(self, path: str) -> str:
        try:
            return os.path.relpath(self._map_repo_path(path), self._workspace_root_path())
        except ValueError:
            return path

    def _prioritize_verifier_paths(self, paths: List[str]) -> List[str]:
        priority = {
            "test.sh": 0,
            "verify.sh": 1,
            "test_outputs.py": 2,
            "test_output.py": 3,
        }
        unique = list(dict.fromkeys(str(path).strip() for path in paths if str(path).strip()))
        return sorted(unique, key=lambda path: (priority.get(os.path.basename(path), 10), path))

    async def _discover_local_verifier_files(self, environment) -> List[str]:
        listing = await self._safe_exec(
            environment,
            f"{self._workspace_cd()}\n"
            "find . -maxdepth 3 -type f "
            "\\( "
            "-path './test_outputs.py' -o -path './test_output.py' -o -path './test.sh' -o -path './verify.sh' "
            "-o -path './tests/*.py' -o -path './tests/*.sh' -o -path './tests/*/*.py' -o -path './tests/*/*.sh' "
            "-o -path './test/*.py' -o -path './test/*.sh' -o -path './test/*/*.py' -o -path './test/*/*.sh' "
            "\\) | sort | sed -n '1,120p'",
            _TIMEOUT_PROBE,
        )
        files: List[str] = []
        seen: set[str] = set()
        for raw in self._output_lines(listing):
            line = raw.strip()
            if not line or line.startswith("find:"):
                continue
            if line.startswith("./"):
                path = self._repo_path(line)
            elif line.startswith("/"):
                path = self._map_repo_path(line)
            else:
                path = self._repo_path(line)
            if path in seen:
                continue
            files.append(path)
            seen.add(path)
        prioritized = self._prioritize_verifier_paths(files)
        self._state["repo"]["local_verifier_files"] = prioritized[:120]
        return prioritized[:120]

    async def _refresh_tests_visibility(self, environment) -> bool:
        del environment
        self._state["repo"]["tests_files"] = []
        self._state["tools"]["has_verify_sh"] = False
        self._state["tools"]["has_test_sh"] = False
        self._state["repo"]["agent_can_read_tests"] = False
        return False

    async def _read_test_content(self, environment) -> str:
        """Read repo-local verifier-like files for advisory requirements."""
        self._state["repo"]["local_verifier_mentions_tests"] = False
        local_verifier_files = await self._discover_local_verifier_files(environment)
        if not local_verifier_files:
            self._needs = self._infer_needs_from_test_text("", self._assertion_targets)
            self._success_markers = []
            return ""

        test_paths: List[str] = list(local_verifier_files)

        content_parts: List[str] = []

        def _append(path: str, out: str) -> None:
            if not out or out.startswith("[error]") or out.startswith("[timeout") or len(out) < 10:
                return
            content_parts.append(f"--- {path} ---\n{out}")
            self._assertion_targets.extend(_extract_assertion_targets(out))

        for path in self._prioritize_verifier_paths(test_paths)[:18]:
            head = 500 if path.endswith(".py") else 220
            out = await self._safe_exec(
                environment,
                f"cat {_shq(path)} 2>/dev/null | head -{int(head)}",
                _TIMEOUT_PROBE,
            )
            _append(path, out)

        self._assertion_targets = list(dict.fromkeys(self._assertion_targets))[:40]
        combined = "\n\n".join(content_parts)
        self._state["repo"]["local_verifier_mentions_tests"] = bool(
            re.search(r"(^|[^A-Za-z0-9_])\/tests\/", combined)
        )
        self._needs = self._infer_needs_from_test_text(combined, self._assertion_targets)
        self._success_markers = self._extract_success_markers(self._assertion_targets)
        return combined

    def _infer_needs_from_test_text(self, text: str, targets: List[str]) -> Dict[str, bool]:
        lower = (text or "").lower()

        commands: set[str] = set()
        target_values: List[str] = []
        for target in targets:
            tag, _, raw_value = target.partition(": ")
            value = raw_value.strip().lower()
            if not value:
                continue
            target_values.append(value)
            if tag == "command":
                commands.add(value.split()[0])
            elif tag == "command_line":
                head = _command_head(value).lower()
                if head:
                    commands.add(head)

        needs_uv = (
            "uvx" in lower
            or "uv run" in lower
            or "uv pip" in lower
            or any(t.startswith("uv_command:") for t in targets)
            or any(command in {"uv", "uvx"} for command in commands)
        )
        needs_python = (
            "pytest" in lower
            or re.search(r"\bpython\d?\b", lower) is not None
            or any(p.startswith("/tests/") and p.endswith(".py") for p in (self._state.get("repo", {}).get("tests_files") or []))
            or any(command in {"python", "python3"} for command in commands)
            or any(value.endswith(".py") for value in target_values)
        )
        needs_pip = (
            "pip" in lower
            or "pip3" in lower
            or "pytest" in lower
            or any(command in {"pip", "pip3"} for command in commands)
            or needs_uv
        )
        needs_pytest = (
            "pytest" in lower
            or re.search(r"^\s*def\s+test_[A-Za-z0-9_]+\s*\(", text or "", flags=re.MULTILINE)
            is not None
            or any(
                os.path.basename(str(path or "")) in {"test_outputs.py", "test_output.py"}
                for path in (self._state.get("repo", {}).get("local_verifier_files") or [])
            )
            or any(os.path.basename(value) in {"test_outputs.py", "test_output.py"} for value in target_values)
        )

        return {
            "curl": ("curl" in lower) or ("curl" in commands),
            "wget": ("wget" in lower) or ("wget" in commands),
            "git": ("git " in lower) or ("git" in commands),
            "env_shim": "/root/.local/bin/env" in lower,
            "uv": needs_uv,
            "python": needs_python,
            "pip": needs_pip or needs_pytest,
            "pytest": needs_pytest,
        }

    def _extract_success_markers(self, targets: List[str]) -> List[str]:
        markers: List[str] = []
        for target in targets:
            if not target.startswith("expected_output: "):
                continue
            value = target.split(": ", 1)[1]
            if re.search(r"\bTEST PASSED\b", value, flags=re.IGNORECASE):
                markers.append(value)
            elif re.search(r"\bALL TESTS PASSED\b", value, flags=re.IGNORECASE):
                markers.append(value)
        return list(dict.fromkeys(markers))[:5]

    def _merge_needs(self, *sources: Dict[str, bool]) -> Dict[str, bool]:
        merged: Dict[str, bool] = {}
        for source in sources:
            for key, value in (source or {}).items():
                merged[key] = bool(merged.get(key) or value)
        return merged

    def _constrain_hidden_mode_needs(self) -> None:
        if self._agent_can_read_tests():
            return

        explicit_text_parts = [
            self._task_instruction,
            "\n".join(self._state.get("repo", {}).get("local_verifier_files") or []),
            "\n".join(self._assertion_targets),
            "\n".join(self._task_targets),
        ]
        explicit_text = "\n".join(part for part in explicit_text_parts if part)
        lower = explicit_text.lower()

        commands: set[str] = set()
        target_values: List[str] = []
        for target in list(self._task_targets) + list(self._assertion_targets):
            tag, _, raw_value = target.partition(": ")
            value = raw_value.strip().lower()
            if not value:
                continue
            target_values.append(value)
            if tag == "command":
                commands.add(value.split()[0])
            elif tag == "command_line":
                head = _command_head(value).lower()
                if head:
                    commands.add(head)

        explicit_uv = (
            "uvx" in lower
            or "uv run" in lower
            or "uv pip" in lower
            or any(target.startswith("uv_command:") for target in self._assertion_targets)
            or any(command in {"uv", "uvx"} for command in commands)
        )
        explicit_pytest = (
            "pytest" in lower
            or any(
                os.path.basename(str(path or "")) in {"test_outputs.py", "test_output.py"}
                for path in (self._state.get("repo", {}).get("local_verifier_files") or [])
            )
        )
        explicit_python = (
            explicit_pytest
            or re.search(r"\bpython\d?\b", lower) is not None
            or any(command in {"python", "python3"} for command in commands)
            or any(value.endswith(".py") for value in target_values)
        )
        explicit_pip = (
            explicit_uv
            or explicit_pytest
            or re.search(r"\bpip3?\b", lower) is not None
            or any(command in {"pip", "pip3"} for command in commands)
        )
        explicit_curl = ("curl" in lower) or ("curl" in commands)
        explicit_wget = ("wget" in lower) or ("wget" in commands)
        explicit_git = bool(re.search(r"\bgit\b", lower)) or ("git" in commands)

        self._needs["uv"] = bool(self._needs.get("uv") and explicit_uv)
        self._needs["pytest"] = bool(self._needs.get("pytest") and explicit_pytest)
        self._needs["pip"] = bool(self._needs.get("pip") and explicit_pip)
        self._needs["python"] = bool(self._needs.get("python") and (explicit_python or explicit_pip))
        self._needs["curl"] = bool(self._needs.get("curl") and explicit_curl)
        self._needs["wget"] = bool(self._needs.get("wget") and explicit_wget)
        self._needs["git"] = bool(self._needs.get("git") and explicit_git)

    async def _detect_pkg_manager(self, environment) -> Optional[str]:
        if self._pkg_manager is not None:
            return self._pkg_manager
        out = await self._safe_exec(
            environment,
            "if command -v apt-get >/dev/null 2>&1; then echo apt; "
            "elif command -v apk >/dev/null 2>&1; then echo apk; "
            "elif command -v dnf >/dev/null 2>&1; then echo dnf; "
            "elif command -v yum >/dev/null 2>&1; then echo yum; "
            "elif command -v pacman >/dev/null 2>&1; then echo pacman; "
            "else echo none; fi",
            _TIMEOUT_PROBE,
        )
        pm = (out.strip().splitlines()[-1] if out.strip() else "none").strip()
        self._pkg_manager = None if pm == "none" else pm
        self._state["tools"]["pkg_manager"] = self._pkg_manager
        return self._pkg_manager

    async def _install_packages(
        self,
        environment,
        packages: List[str],
        *,
        reason: str,
    ) -> bool:
        packages = [p for p in (packages or []) if str(p).strip()]
        if not packages:
            return False
        pm = await self._detect_pkg_manager(environment)
        if not pm:
            _log(f"no package manager available (wanted {packages} for {reason})")
            return False

        pkg_list = " ".join(_shq(p) for p in packages)
        self._status("installing dependencies", detail=f"{pm}: {reason}")

        if pm == "apt":
            apt_prefix = (
                "DEBIAN_FRONTEND=noninteractive "
                "apt-get -o DPkg::Lock::Timeout=120 -o Acquire::Retries=3"
            )
            if not self._pkg_update_done:
                await self._safe_exec(
                    environment,
                    f"{apt_prefix} update 2>&1",
                    _TIMEOUT_INSTALL,
                )
                self._pkg_update_done = True
            await self._safe_exec(
                environment,
                f"{apt_prefix} install -y --no-install-recommends {pkg_list} 2>&1",
                _TIMEOUT_INSTALL,
            )
            return True
        if pm == "apk":
            await self._safe_exec(environment, f"apk add --no-cache {pkg_list} 2>&1", _TIMEOUT_INSTALL)
            return True
        if pm in {"dnf", "yum"}:
            await self._safe_exec(environment, f"{pm} install -y {pkg_list} 2>&1", _TIMEOUT_INSTALL)
            return True
        if pm == "pacman":
            await self._safe_exec(
                environment,
                f"pacman -Sy --noconfirm {pkg_list} 2>&1",
                _TIMEOUT_INSTALL,
            )
            return True

        _log(f"unknown package manager: {pm}")
        return False

    async def _ensure_command(
        self,
        environment,
        command: str,
        *,
        packages: List[str],
        reason: str,
    ) -> bool:
        probe = await self._safe_exec(
            environment,
            f"command -v {_shq(command)} >/dev/null 2>&1 && echo OK || echo NO",
            _TIMEOUT_PROBE,
        )
        if "OK" in probe:
            return True
        await self._install_packages(environment, packages, reason=reason)
        probe2 = await self._safe_exec(
            environment,
            f"command -v {_shq(command)} >/dev/null 2>&1 && echo OK || echo NO",
            _TIMEOUT_PROBE,
        )
        return "OK" in probe2

    async def _ensure_env_shim(self, environment) -> bool:
        out = await self._safe_exec(
            environment,
            "test -x /root/.local/bin/env && "
            "grep -q 'MEMOHARNESS_ENV_SHIM' /root/.local/bin/env 2>/dev/null && "
            "echo HAS_ENV_SHIM=1 || echo HAS_ENV_SHIM=0",
            _TIMEOUT_PROBE,
        )
        if "HAS_ENV_SHIM=1" in out:
            self._state["tools"]["has_env_shim"] = True
            return True

        cmd = (
            "set -euo pipefail\n"
            "mkdir -p /root/.local/bin\n"
            "rm -f /root/.local/bin/env\n"
            "cat > /root/.local/bin/env <<'EOF'\n"
            "#!/bin/sh\n"
            "# MEMOHARNESS_ENV_SHIM\n"
            "export PATH=\"/root/.local/bin:$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH\"\n"
            "return 0 2>/dev/null || exit 0\n"
            "EOF\n"
            "chmod 0755 /root/.local/bin/env\n"
            "test -x /root/.local/bin/env && "
            "grep -q 'MEMOHARNESS_ENV_SHIM' /root/.local/bin/env 2>/dev/null && "
            "echo HAS_ENV_SHIM=1 || echo HAS_ENV_SHIM=0\n"
        )
        out2 = await self._safe_exec(environment, cmd, _TIMEOUT_CMD)
        ok = "HAS_ENV_SHIM=1" in out2
        self._state["tools"]["has_env_shim"] = ok
        return ok

    async def _ensure_python_aliases(self, environment) -> None:
        py_probe = await self._safe_exec(
            environment,
            "command -v python3 >/dev/null 2>&1 && echo HAS_PYTHON3=1 || echo HAS_PYTHON3=0",
            _TIMEOUT_PROBE,
        )
        if "HAS_PYTHON3=1" in py_probe:
            self._state["tools"]["has_python3"] = True
            await self._safe_exec(
                environment,
                "command -v python >/dev/null 2>&1 || "
                "(ln -sf \"$(command -v python3)\" /usr/local/bin/python 2>/dev/null || true)",
                _TIMEOUT_PROBE,
            )

        pip_probe = await self._safe_exec(
            environment,
            "command -v pip3 >/dev/null 2>&1 && echo HAS_PIP3=1 || echo HAS_PIP3=0",
            _TIMEOUT_PROBE,
        )
        if "HAS_PIP3=1" in pip_probe:
            self._state["tools"]["has_pip3"] = True
            await self._safe_exec(
                environment,
                "command -v pip >/dev/null 2>&1 || "
                "(ln -sf \"$(command -v pip3)\" /usr/local/bin/pip 2>/dev/null || true)",
                _TIMEOUT_PROBE,
            )

    async def _ensure_python_pip(self, environment) -> bool:
        if not self._needs.get("python") and not self._needs.get("pip"):
            await self._ensure_python_aliases(environment)
            return True

        ok_py = await self._ensure_command(
            environment,
            "python3",
            packages=["python3"],
            reason="tests reference python",
        )
        self._state["tools"]["has_python3"] = bool(ok_py)
        ok_pip = True
        if self._needs.get("pip") or self._needs.get("uv"):
            ok_pip = await self._ensure_command(
                environment,
                "pip3",
                packages=["python3-pip"],
                reason="tests reference pip/pytest/uv",
            )
            if not ok_pip and (self._pkg_manager == "apk"):
                ok_pip = await self._ensure_command(
                    environment,
                    "pip3",
                    packages=["py3-pip"],
                    reason="tests reference pip/pytest/uv (alpine)",
                )
        self._state["tools"]["has_pip3"] = bool(ok_pip)
        await self._ensure_python_aliases(environment)

        return ok_py and ok_pip

    async def _ensure_pytest(self, environment) -> bool:
        probe_cmd = (
            "(command -v pytest >/dev/null 2>&1 && echo HAS_PYTEST=1) || "
            "(command -v python3 >/dev/null 2>&1 && python3 -m pytest --version >/dev/null 2>&1 && echo HAS_PYTEST=1) || "
            "echo HAS_PYTEST=0"
        )
        probe = await self._safe_exec(environment, probe_cmd, _TIMEOUT_PROBE)
        if "HAS_PYTEST=1" in probe:
            self._state["tools"]["has_pytest"] = True
            return True

        if not self._needs.get("pytest"):
            self._state["tools"]["has_pytest"] = False
            return False

        await self._ensure_python_pip(environment)
        pm = await self._detect_pkg_manager(environment)
        packages: List[str] = []
        if pm == "apt":
            packages = ["python3-pytest"]
        elif pm == "apk":
            packages = ["py3-pytest"]
        elif pm in {"dnf", "yum"}:
            packages = ["python3-pytest"]
        elif pm == "pacman":
            packages = ["python-pytest"]
        if packages:
            await self._install_packages(
                environment,
                packages,
                reason="verifier-style python tests need pytest",
            )

        probe = await self._safe_exec(environment, probe_cmd, _TIMEOUT_PROBE)
        if "HAS_PYTEST=1" in probe:
            self._state["tools"]["has_pytest"] = True
            return True

        if self._state["tools"].get("has_pip3"):
            out = await self._safe_exec(
                environment,
                "python3 -m pip install -U pytest 2>&1",
                _TIMEOUT_INSTALL,
            )
            if "externally-managed-environment" in out.lower() or "externally managed environment" in out.lower():
                await self._safe_exec(
                    environment,
                    "python3 -m pip install --break-system-packages -U pytest 2>&1",
                    _TIMEOUT_INSTALL,
                )

        probe = await self._safe_exec(environment, probe_cmd, _TIMEOUT_PROBE)
        ok = "HAS_PYTEST=1" in probe
        self._state["tools"]["has_pytest"] = ok
        return ok

    async def _ensure_uv(self, environment) -> bool:
        """Install uv/uvx if verifier scripts reference it."""
        # If uvx is already available, we're done.
        out = await self._safe_exec(
            environment,
            "command -v uvx >/dev/null 2>&1 && uvx --version 2>&1 | head -1 && echo HAS_UVX=1 || echo HAS_UVX=0",
            _TIMEOUT_PROBE,
        )
        if "HAS_UVX=1" in out:
            self._uv_installed = True
            self._state["tools"]["has_uv"] = True
            _log("uvx already available")
            return True

        needs_uv = bool(
            self._needs.get("uv")
            or any(t.startswith("uv_command:") for t in self._assertion_targets)
        )
        if not needs_uv:
            self._state["tools"]["has_uv"] = False
            return False

        await self._ensure_env_shim(environment)
        ok_py = await self._ensure_python_pip(environment)
        if not ok_py:
            _log("cannot install uv: python3/pip3 unavailable")
            self._state["tools"]["has_uv"] = False
            return False

        self._status("installing uv", detail="verifier references uvx/uv")

        install_cmd = "python3 -m pip install -U uv 2>&1"
        install_logs: List[str] = []
        out1 = await self._safe_exec(environment, install_cmd, _TIMEOUT_INSTALL)
        install_logs.append(out1)
        if "externally-managed-environment" in out1.lower() or "externally managed environment" in out1.lower():
            out1 = await self._safe_exec(
                environment,
                "python3 -m pip install --break-system-packages -U uv 2>&1",
                _TIMEOUT_INSTALL,
            )
            install_logs.append(out1)

        # Verify (and make it visible on default PATH for verifiers).
        verify = await self._safe_exec(
            environment,
            "command -v uvx >/dev/null 2>&1 && uvx --version 2>&1 | head -1 && echo HAS_UVX=1 || echo HAS_UVX=0",
            _TIMEOUT_PROBE,
        )
        if "HAS_UVX=1" in verify:
            await self._safe_exec(
                environment,
                "uvx_path=$(command -v uvx) || true; "
                "uv_path=$(command -v uv) || true; "
                "if [ -n \"$uvx_path\" ] && [ ! -x /usr/local/bin/uvx ]; then ln -sf \"$uvx_path\" /usr/local/bin/uvx 2>/dev/null || true; fi; "
                "if [ -n \"$uv_path\" ] && [ ! -x /usr/local/bin/uv ]; then ln -sf \"$uv_path\" /usr/local/bin/uv 2>/dev/null || true; fi; "
                "command -v uvx >/dev/null 2>&1 && echo OK || echo NO",
                _TIMEOUT_PROBE,
            )
            self._uv_installed = True
            self._state["tools"]["has_uv"] = True
            _log("uv installed via pip")
            return True

        # Last-resort installer: uses curl + external downloads (may be blocked).
        have_curl = await self._ensure_command(
            environment,
            "curl",
            packages=["curl", "ca-certificates"],
            reason="uv installer requires curl",
        )
        if have_curl:
            out2 = await self._safe_exec(
                environment,
                "curl -LsSf https://astral.sh/uv/install.sh | sh 2>&1 || true\n"
                "command -v uvx >/dev/null 2>&1 && uvx --version 2>&1 | head -1 && echo HAS_UVX=1 || echo HAS_UVX=0",
                _TIMEOUT_INSTALL,
            )
            install_logs.append(out2)
            if "HAS_UVX=1" in out2:
                self._uv_installed = True
                self._state["tools"]["has_uv"] = True
                _log("uv installed via astral.sh installer")
                return True

        combined_install_log = "\n".join(chunk for chunk in install_logs + [verify] if chunk)
        _log(
            f"uv install failed: {self._tail(combined_install_log, max_lines=12, max_chars=600)}"
        )
        self._state["tools"]["has_uv"] = False
        return False

    async def _ensure_rscript(self, environment) -> bool:
        probe_cmd = (
            "command -v Rscript >/dev/null 2>&1 && echo HAS_RSCRIPT=1 || echo HAS_RSCRIPT=0"
        )
        probe = await self._safe_exec(environment, probe_cmd, _TIMEOUT_PROBE)
        if "HAS_RSCRIPT=1" in probe:
            return True

        pm = await self._detect_pkg_manager(environment)
        package_attempts = {
            "apt": [["r-base-core"], ["r-base"]],
            "apk": [["R"]],
            "dnf": [["R"]],
            "yum": [["R"]],
            "pacman": [["r"]],
        }.get(pm or "", [])

        for packages in package_attempts:
            await self._install_packages(
                environment,
                packages,
                reason="hidden producer script requires Rscript",
            )
            probe = await self._safe_exec(environment, probe_cmd, _TIMEOUT_PROBE)
            if "HAS_RSCRIPT=1" in probe:
                return True
        return False

    async def _deep_discovery(self, environment) -> str:
        """Discover project structure beyond basic bootstrap."""
        discovery_parts: List[str] = []

        # Read README if it exists (often has install/setup instructions)
        for readme in ["README.md", "README.rst", "README.txt", "README"]:
            out = await self._safe_exec(
                environment,
                f"{self._workspace_cd()}\ncat {readme} 2>/dev/null | head -100",
                _TIMEOUT_PROBE,
            )
            if out and not out.startswith("[error]") and not out.startswith("[timeout") and len(out) > 10:
                self._state["repo"]["readme"] = readme
                self._remember_repo_command_candidates(out)
                discovery_parts.append(f"--- {self._repo_path(readme)} (first 100 lines) ---\n{out}")
                break

        # Check for Makefile, setup files, requirements
        for build_file in ["Makefile", "setup.py", "setup.cfg", "pyproject.toml", "requirements.txt"]:
            out = await self._safe_exec(
                environment,
                f"{self._workspace_cd()}\ncat {build_file} 2>/dev/null | head -80",
                _TIMEOUT_PROBE,
            )
            if out and not out.startswith("[error]") and not out.startswith("[timeout") and len(out) > 5:
                if build_file == "Makefile":
                    self._state["repo"]["makefile"] = build_file
                self._remember_repo_command_candidates(out)
                discovery_parts.append(f"--- {self._repo_path(build_file)} (first 80 lines) ---\n{out}")

        # Check for docker/build instructions
        out = await self._safe_exec(
            environment,
            f"{self._workspace_cd()}\nls -la Dockerfile* docker-compose* *.sh 2>/dev/null || true",
            _TIMEOUT_PROBE,
        )
        if out and len(out.strip()) > 0:
            discovery_parts.append(f"--- {self._workspace_root_path()} build files ---\n{out}")

        # List deeper /app structure
        out = await self._safe_exec(
            environment,
            f"{self._workspace_cd()}\nfind . -maxdepth 2 -type f | head -60 2>/dev/null || true",
            _TIMEOUT_PROBE,
        )
        if out and len(out.strip()) > 0:
            discovery_parts.append(f"--- {self._workspace_root_path()} file tree (depth 2) ---\n{out}")

        if not self._agent_can_read_tests():
            hidden_evidence = await self._hidden_sparse_repo_evidence(environment)
            if hidden_evidence:
                discovery_parts.append(hidden_evidence)

        return "\n\n".join(discovery_parts)

    async def _missing_items(self, environment) -> List[str]:
        if not self._assertion_targets:
            return []
        missing: List[str] = []
        for target in self._assertion_targets:
            if target.startswith("path: ") or target.startswith("must_exist: "):
                path = self._map_repo_path(target.split(": ", 1)[1])
                out = await self._safe_exec(
                    environment,
                    f"test -e {_shq(path)} && echo EXISTS || echo MISSING",
                    _TIMEOUT_PROBE,
                )
                if "MISSING" in out:
                    missing.append(path)
            elif target.startswith("file: "):
                path = self._map_repo_path(target.split(": ", 1)[1])
                if path.startswith("/"):
                    out = await self._safe_exec(
                        environment,
                        f"test -e {_shq(path)} && echo EXISTS || echo MISSING",
                        _TIMEOUT_PROBE,
                    )
                    if "MISSING" in out:
                        missing.append(path)
            elif target.startswith("command: "):
                cmd = target.split(": ", 1)[1]
                out = await self._safe_exec(
                    environment,
                    f"command -v {_shq(cmd)} 2>/dev/null && echo FOUND || echo NOT_FOUND",
                    _TIMEOUT_PROBE,
                )
                if "NOT_FOUND" in out:
                    missing.append(f"command '{cmd}' not found")
            elif target.startswith("directory: ") or target.startswith("directory_candidate: "):
                path = self._map_repo_path(target.split(": ", 1)[1])
                out = await self._safe_exec(
                    environment,
                    f"test -d {_shq(path)} && echo EXISTS || echo MISSING",
                    _TIMEOUT_PROBE,
                )
                if "MISSING" in out:
                    missing.append(f"directory {path}")

        return missing

    async def _check_missing_paths(self, environment) -> str:
        """Check if assertion-target paths exist and report missing ones."""
        missing = await self._missing_items(environment)
        if not missing:
            return ""
        return "MISSING PATHS (must be created):\n" + "\n".join(f"  - {m}" for m in missing)

    async def _producer_discovery(self, environment, missing_items: List[str]) -> str:
        """Search /app for references to missing artifacts to find the 'producer' code quickly."""
        if not missing_items:
            return ""

        have_rg = bool(self._state.get("tools", {}).get("has_rg"))
        hints: List[str] = []
        seen_patterns: set[str] = set()

        for item in missing_items:
            if item.startswith("command '"):
                continue
            path = item
            if item.startswith("directory "):
                path = item.split(" ", 1)[1]
            base = os.path.basename(path.strip())
            if not base or base in {"/", "."}:
                continue
            if base in seen_patterns:
                continue
            seen_patterns.add(base)
            if len(seen_patterns) > 5:
                break

            if have_rg:
                cmd = (
                    f"{self._workspace_cd()}\n"
                    f"rg -n -F {_shq(base)} . {_rg_search_excludes()} 2>/dev/null | sed -n '1,200p'"
                )
                out = await self._safe_exec(environment, cmd, _TIMEOUT_PROBE)
            else:
                cmd = (
                    f"{self._workspace_cd()}\n"
                    f"grep -RIn {_grep_search_excludes()} -- {_shq(base)} . 2>/dev/null | sed -n '1,200p'"
                )
                out = await self._safe_exec(environment, cmd, _TIMEOUT_LONG_CMD)
            filtered_lines = self._filter_repo_search_lines(out, max_lines=20)
            if filtered_lines:
                hints.append(
                    f"--- producer search for {base} ---\n" + "\n".join(filtered_lines)
                )

        return "\n\n".join(hints)

    async def _producer_discovery_for_instruction(self, environment) -> str:
        terms = self._instruction_terms()
        self._hidden_exact_visible_matches = []
        if not terms:
            return ""

        turn = int(self._state.get("progress", {}).get("turn") or 0)
        max_terms = 8 if turn <= 1 else 4
        max_path_lines = 12 if turn <= 1 else 8
        max_content_lines = 20 if turn <= 1 else 12
        max_exact_candidates = 12 if turn <= 1 else 8
        have_rg = bool(self._state.get("tools", {}).get("has_rg"))
        hints: List[str] = []
        exact_candidates: List[str] = []
        seen_candidates: set[str] = set()
        for term in terms[:max_terms]:
            if have_rg:
                path_cmd = (
                    f"{self._workspace_cd()}\n"
                    f"rg --files {_rg_search_excludes()} . | rg -n -F {_shq(term)} | sed -n '1,80p'"
                )
                content_cmd = (
                    f"{self._workspace_cd()}\n"
                    f"rg -n -F {_shq(term)} . {_rg_search_excludes()} 2>/dev/null | sed -n '1,120p'"
                )
                path_out = await self._safe_exec(environment, path_cmd, _TIMEOUT_PROBE)
                content_out = await self._safe_exec(environment, content_cmd, _TIMEOUT_PROBE)
            else:
                path_cmd = (
                    f"{self._workspace_cd()}\n"
                    f"find . {_find_search_prune_clause()} -type f -print | "
                    f"grep -n -F -- {_shq(term)} | sed -n '1,80p'"
                )
                content_cmd = (
                    f"{self._workspace_cd()}\n"
                    f"grep -RIn {_grep_search_excludes()} -- {_shq(term)} . 2>/dev/null | sed -n '1,120p'"
                )
                path_out = await self._safe_exec(environment, path_cmd, _TIMEOUT_LONG_CMD)
                content_out = await self._safe_exec(environment, content_cmd, _TIMEOUT_LONG_CMD)

            path_lines = self._filter_repo_search_lines(path_out, max_lines=max_path_lines)
            if path_lines:
                hints.append(
                    f"--- producer path search for {term} ---\n" + "\n".join(path_lines)
                )
                for raw in path_lines:
                    path = _search_hit_path(raw)
                    if not path:
                        continue
                    snippet = f"{path} [path match]"
                    if snippet in seen_candidates:
                        continue
                    exact_candidates.append(_clip_inline(snippet, 240))
                    seen_candidates.add(snippet)
                    if len(exact_candidates) >= max_exact_candidates:
                        break

            content_lines = self._filter_repo_search_lines(content_out, max_lines=max_content_lines)
            if content_lines:
                hints.append(
                    f"--- producer search for {term} ---\n" + "\n".join(content_lines)
                )
            for line in content_lines:
                parts = line.split(":", 2)
                if len(parts) != 3:
                    continue
                path, lineno, payload = parts
                payload = payload.strip()
                if not payload:
                    continue
                if (
                    payload.startswith(("import ", "from ", "def ", "class "))
                    or _looks_like_sourceish_line(payload)
                    or _looks_like_commandish_fragment(payload)
                ):
                    continue
                snippet = f"{path}:{lineno}: {payload}"
                if snippet in seen_candidates:
                    continue
                exact_candidates.append(_clip_inline(snippet, 240))
                seen_candidates.add(snippet)
                if len(exact_candidates) >= max_exact_candidates:
                    break
            if len(exact_candidates) >= max_exact_candidates and len(hints) >= max_terms:
                break

        self._hidden_exact_visible_matches = exact_candidates[:max_exact_candidates]
        sections: List[str] = []
        if exact_candidates:
            sections.append(
                "EXACT VISIBLE MATCH CANDIDATES:\n"
                + "\n".join(f"  - {item}" for item in exact_candidates[:max_exact_candidates])
            )
        if hints:
            sections.append("\n\n".join(hints))
        return "\n\n".join(section for section in sections if section)

    async def _build_hidden_producer_context(self, environment) -> str:
        snippets: List[str] = []
        candidate_paths: List[str] = []
        seen_paths: set[str] = set()

        def _remember(path: str, *, allow_evidence_fallback: bool = False) -> None:
            mapped = self._map_repo_path(path)
            if mapped in seen_paths:
                return
            if not self._is_hidden_source_candidate_path(mapped):
                if not allow_evidence_fallback or not self._is_hidden_evidence_path(mapped):
                    return
            candidate_paths.append(mapped)
            seen_paths.add(mapped)

        for item in self._hidden_exact_visible_matches + self._hidden_visible_text_candidates:
            path = self._hidden_candidate_source_path(item)
            if path:
                _remember(path)

        for path in self._hidden_evidence_paths:
            _remember(path)

        for path in await self._collect_hidden_producer_paths(environment, limit=4):
            _remember(path)

        if not candidate_paths:
            for path in self._hidden_evidence_paths:
                _remember(path, allow_evidence_fallback=True)

        for path in candidate_paths[:4]:
            mapped = self._map_repo_path(path)
            if not self._is_hidden_evidence_path(mapped):
                continue
            out = await self._safe_exec(
                environment,
                f"if [ -f {_shq(mapped)} ]; then echo '--- {mapped} ---'; sed -n '1,160p' {_shq(mapped)}; fi",
                _TIMEOUT_PROBE,
            )
            cleaned = str(out or "").strip()
            if not cleaned or cleaned.startswith("[error]") or cleaned.startswith("[timeout"):
                continue
            snippets.append(_clip_inline(cleaned, 1600))
            if len(snippets) >= 2:
                break
        if not snippets:
            return ""
        return (
            "PRODUCER SOURCE HEADS (inspect these before rewriting a working flow or installing large dependency stacks):\n"
            + "\n\n".join(snippets)
        )

    async def _bootstrap(self, environment) -> str:
        t0 = time.time()
        _log("bootstrap: starting workspace probes...")
        workspace_root = await self._ensure_workspace_root(environment)
        workspace_root = await self._maybe_promote_nested_workspace_root(environment)
        cmds = [
            "command -v rg >/dev/null 2>&1 && echo HAS_RG=1 || echo HAS_RG=0",
            "command -v pytest >/dev/null 2>&1 && echo HAS_PYTEST=1 || echo HAS_PYTEST=0",
            "command -v uvx >/dev/null 2>&1 && echo HAS_UVX=1 || echo HAS_UVX=0",
            "command -v python3 >/dev/null 2>&1 && echo HAS_PYTHON3=1 || echo HAS_PYTHON3=0",
            "command -v pip3 >/dev/null 2>&1 && echo HAS_PIP3=1 || echo HAS_PIP3=0",
        ]
        outs: List[str] = []
        for i, c in enumerate(cmds, 1):
            _log(f"bootstrap cmd {i}/{len(cmds)}: {c[:80]}")
            t1 = time.time()
            result = await self._safe_exec(environment, c, _TIMEOUT_PROBE)
            _log(f"bootstrap cmd {i} done in {_elapsed(t1)} ({len(result)} chars)")
            outs.append(f"$ {c}\n{result}".rstrip())

        boot = "\n\n".join(outs).strip()
        boot_lines = set(self._output_lines(boot))

        self._state["tools"]["has_rg"] = "HAS_RG=1" in boot_lines
        self._state["tools"]["has_pytest"] = "HAS_PYTEST=1" in boot_lines
        self._state["tools"]["has_uv"] = "HAS_UVX=1" in boot_lines
        self._state["tools"]["has_python3"] = "HAS_PYTHON3=1" in boot_lines
        self._state["tools"]["has_pip3"] = "HAS_PIP3=1" in boot_lines

        await self._write_file(environment, _LAST_BOOT_PATH, boot)
        await self._persist_state(environment)
        _log(f"bootstrap: done in {_elapsed(t0)} (pytest={self._state['tools']['has_pytest']}, "
             f"rg={self._state['tools']['has_rg']}, "
             f"uvx={self._state['tools']['has_uv']}, workspace_root={workspace_root})")
        return boot

    async def _resolve_local_validation_cmd(self, environment) -> Tuple[str, str]:
        instruction_probe = await self._resolve_instruction_validation_cmd(environment)
        instruction_probe_fallback: Optional[Tuple[str, str]] = instruction_probe

        local_verifier_files = await self._discover_local_verifier_files(environment)
        has_python3 = bool(self._state["tools"].get("has_python3"))
        has_pytest = bool(self._state["tools"].get("has_pytest"))
        local_verifier_mentions_tests = self._local_verifier_mentions_hidden_tests()
        allow_repo_local_test_runners = (
            _ENABLE_REPO_LOCAL_TEST_RUNNERS and not local_verifier_mentions_tests
        )

        if local_verifier_mentions_tests and instruction_probe_fallback is not None:
            return instruction_probe_fallback

        if allow_repo_local_test_runners:
            for basename, runner in (("test.sh", "app_test_sh"), ("verify.sh", "app_verify_sh")):
                for path in local_verifier_files:
                    if os.path.basename(path) != basename:
                        continue
                    rel_path = self._repo_relpath(path)
                    return f"{self._workspace_cd()}\nbash {_shq(rel_path)} 2>&1", runner

        if (has_python3 or has_pytest) and allow_repo_local_test_runners:
            for basename in ("test_outputs.py", "test_output.py"):
                for path in local_verifier_files:
                    if os.path.basename(path) != basename:
                        continue
                    rel_path = self._repo_relpath(path)
                    return (
                        f"{self._workspace_cd()}\n(pytest -q {_shq(rel_path)} 2>/dev/null || "
                        f"python3 -m pytest -q {_shq(rel_path)})",
                        "app_pytest_test_outputs",
                    )

            if any(self._map_repo_path(path).startswith(self._repo_path("tests/")) for path in local_verifier_files):
                return (
                    f"{self._workspace_cd()}\n(pytest -q --maxfail=1 ./tests 2>/dev/null || "
                    "python3 -m pytest -q --maxfail=1 ./tests)",
                    "app_pytest_tests_dir",
                )

            if any(self._map_repo_path(path).startswith(self._repo_path("test/")) for path in local_verifier_files):
                return (
                    f"{self._workspace_cd()}\n(pytest -q --maxfail=1 ./test 2>/dev/null || "
                    "python3 -m pytest -q --maxfail=1 ./test)",
                    "app_pytest_test_dir",
                )

        if (
            instruction_probe_fallback is not None
            and instruction_probe_fallback[1] in {"local_instruction_smoke", "local_target_probe"}
        ):
            return instruction_probe_fallback

        if (has_python3 or has_pytest) and allow_repo_local_test_runners:
            pytests = await self._safe_exec(
                environment,
                f"{self._workspace_cd()}\nfind . -maxdepth 3 -type f "
                "\\( -name 'test_*.py' -o -path './tests/*.py' -o -path './tests/*/test_*.py' \\) "
                "| sed -n '1,1p'",
                _TIMEOUT_PROBE,
            )
            if pytests.strip():
                return (
                    f"{self._workspace_cd()}\n(pytest -q --maxfail=1 2>/dev/null || "
                    "python3 -m pytest -q --maxfail=1)",
                    "local_repo_pytest",
                )

            pyfiles = await self._safe_exec(
                environment,
                f"{self._workspace_cd()}\nfind . -maxdepth 3 -type f -name '*.py' | sed -n '1,1p'",
                _TIMEOUT_PROBE,
            )
            if pyfiles.strip():
                return (
                    f"{self._workspace_cd()}\nfind . -maxdepth 3 -type f -name '*.py' | sort | sed -n '1,40p' | "
                    "while read -r f; do [ -n \"$f\" ] || continue; python3 -m py_compile \"$f\" || exit $?; done",
                    "local_python_sanity",
                )

        shfiles = await self._safe_exec(
            environment,
            f"{self._workspace_cd()}\nfind . -maxdepth 3 -type f -name '*.sh' | sed -n '1,1p'",
            _TIMEOUT_PROBE,
        )
        if shfiles.strip():
            return (
                f"{self._workspace_cd()}\nfind . -maxdepth 3 -type f -name '*.sh' | sort | sed -n '1,40p' | "
                "while read -r f; do [ -n \"$f\" ] || continue; bash -n \"$f\" || exit $?; done",
                "local_shell_sanity",
            )

        node_probe = await self._safe_exec(
            environment,
            "command -v node >/dev/null 2>&1 && echo HAS_NODE=1 || echo HAS_NODE=0",
            _TIMEOUT_PROBE,
        )
        if "HAS_NODE=1" in node_probe:
            jsfiles = await self._safe_exec(
                environment,
                f"{self._workspace_cd()}\nfind . -maxdepth 3 -type f "
                "\\( -name '*.js' -o -name '*.mjs' -o -name '*.cjs' \\) | sed -n '1,1p'",
                _TIMEOUT_PROBE,
            )
            if jsfiles.strip():
                return (
                    f"{self._workspace_cd()}\nfind . -maxdepth 3 -type f "
                    "\\( -name '*.js' -o -name '*.mjs' -o -name '*.cjs' \\) | sort | sed -n '1,40p' | "
                    "while read -r f; do [ -n \"$f\" ] || continue; node --check \"$f\" || exit $?; done",
                    "local_node_sanity",
                )

        if instruction_probe_fallback is not None:
            return instruction_probe_fallback

        return (
            f"{self._workspace_cd()}\nfind . -maxdepth 2 -type f | sort | sed -n '1,120p'",
            "local_probe",
        )

    async def _resolve_test_cmd(self, environment) -> str:
        if not self._agent_local_validation_enabled():
            self._disable_agent_local_validation_state()
            return str(self._state["test"]["cmd"] or "")

        # Harbor uploads /tests during the verifier phase after the agent has
        # finished. Agent execution therefore does not rely on local validation
        # to decide pass/fail.
        cmd, runner = await self._resolve_local_validation_cmd(environment)

        self._state["test"]["cmd"] = cmd
        self._state["test"]["runner"] = runner
        _log(f"test command resolved: runner={runner} cmd={cmd}")
        return cmd

    def _looks_like_test_failure_output(self, text: str) -> bool:
        lower = (text or "").lower()
        return any(
            marker in lower
            for marker in [
                "test failed",
                "\u2717",
                "assertionerror",
                "traceback",
                "no such file or directory",
                "command not found",
            ]
        )

    def _is_test_success(self, rc: int, out: str) -> bool:
        runner = str(self._state.get("test", {}).get("runner") or "")
        if self._runner_is_local_only(runner):
            return False
        if int(rc) != 0:
            return False
        if self._looks_like_test_failure_output(out):
            # Some verifier scripts exit 0 but report failures in stdout.
            return False
        if self._success_markers and not any(m in (out or "") for m in self._success_markers):
            return False
        return True

    async def _auto_fix_from_test_output(self, environment, out: str) -> List[str]:
        """Apply cheap, deterministic fixes for systemic bootstrap failures."""
        actions: List[str] = []
        lower = (out or "").lower()

        if "/root/.local/bin/env" in lower and "env_shim" not in self._auto_fixes_done:
            ok = await self._ensure_env_shim(environment)
            actions.append(f"ensure /root/.local/bin/env shim: {ok}")
            self._auto_fixes_done.add("env_shim")

        if "curl: command not found" in lower and "curl" not in self._auto_fixes_done:
            ok = await self._ensure_command(
                environment,
                "curl",
                packages=["curl", "ca-certificates"],
                reason="verifier needs curl",
            )
            self._state["tools"]["has_curl"] = ok
            actions.append(f"install curl: {ok}")
            self._auto_fixes_done.add("curl")

        if "uvx: command not found" in lower and "uv" not in self._auto_fixes_done:
            self._needs["uv"] = True
            ok = await self._ensure_uv(environment)
            actions.append(f"install uv/uvx: {ok}")
            self._auto_fixes_done.add("uv")

        if "python3: command not found" in lower and "python3" not in self._auto_fixes_done:
            self._needs["python"] = True
            ok = await self._ensure_python_pip(environment)
            await self._ensure_python_aliases(environment)
            actions.append(f"install python3/pip3: {ok}")
            self._auto_fixes_done.add("python3")

        if "pip3: command not found" in lower and "pip3" not in self._auto_fixes_done:
            self._needs["pip"] = True
            ok = await self._ensure_python_pip(environment)
            await self._ensure_python_aliases(environment)
            actions.append(f"install pip3: {ok}")
            self._auto_fixes_done.add("pip3")

        return actions

    async def _auto_fix_from_command_output(self, environment, out: str) -> List[str]:
        actions: List[str] = []
        lower = (out or "").lower()

        if "python: command not found" in lower and "python_cmd" not in self._auto_fixes_done:
            self._needs["python"] = True
            ok = await self._ensure_python_pip(environment)
            await self._ensure_python_aliases(environment)
            actions.append(f"ensure python command: {ok or self._state['tools'].get('has_python3')}")
            self._auto_fixes_done.add("python_cmd")

        if "python3: command not found" in lower and "python3_cmd" not in self._auto_fixes_done:
            self._needs["python"] = True
            ok = await self._ensure_python_pip(environment)
            await self._ensure_python_aliases(environment)
            actions.append(f"ensure python3 command: {ok or self._state['tools'].get('has_python3')}")
            self._auto_fixes_done.add("python3_cmd")

        if "pip: command not found" in lower and "pip_cmd" not in self._auto_fixes_done:
            self._needs["pip"] = True
            ok = await self._ensure_python_pip(environment)
            await self._ensure_python_aliases(environment)
            actions.append(f"ensure pip command: {ok or self._state['tools'].get('has_pip3')}")
            self._auto_fixes_done.add("pip_cmd")

        if "pip3: command not found" in lower and "pip3_cmd" not in self._auto_fixes_done:
            self._needs["pip"] = True
            ok = await self._ensure_python_pip(environment)
            await self._ensure_python_aliases(environment)
            actions.append(f"ensure pip3 command: {ok or self._state['tools'].get('has_pip3')}")
            self._auto_fixes_done.add("pip3_cmd")

        if "curl: command not found" in lower and "curl" not in self._auto_fixes_done:
            ok = await self._ensure_command(
                environment,
                "curl",
                packages=["curl", "ca-certificates"],
                reason="model commands use curl",
            )
            self._state["tools"]["has_curl"] = ok
            actions.append(f"install curl: {ok}")
            self._auto_fixes_done.add("curl")

        if "uvx: command not found" in lower and "uv" not in self._auto_fixes_done:
            self._needs["uv"] = True
            ok = await self._ensure_uv(environment)
            actions.append(f"install uv/uvx: {ok}")
            self._auto_fixes_done.add("uv")

        if (
            ("missing rscript" in lower or "rscript: command not found" in lower)
            and "rscript" not in self._auto_fixes_done
        ):
            ok = await self._ensure_rscript(environment)
            actions.append(f"install Rscript: {ok}")
            self._auto_fixes_done.add("rscript")

        if "pkill: command not found" in lower and "pkill" not in self._auto_fixes_done:
            ok = await self._ensure_command(
                environment,
                "pkill",
                packages=["procps"],
                reason="model commands restart background services with pkill",
            )
            actions.append(f"install pkill/procps: {ok}")
            self._auto_fixes_done.add("pkill")

        if "rg: command not found" in lower and "rg" not in self._auto_fixes_done:
            ok = await self._ensure_command(
                environment,
                "rg",
                packages=["ripgrep"],
                reason="model commands search the workspace with rg",
            )
            self._state["tools"]["has_rg"] = ok
            actions.append(f"install rg/ripgrep: {ok}")
            self._auto_fixes_done.add("rg")

        if "file: command not found" in lower and "file_cmd" not in self._auto_fixes_done:
            ok = await self._ensure_command(
                environment,
                "file",
                packages=["file"],
                reason="model commands inspect binaries with file",
            )
            actions.append(f"install file utility: {ok}")
            self._auto_fixes_done.add("file_cmd")

        return actions

    def _build_targeted_pytest_cmd(self, target: str) -> str:
        nodeid = str(target or "").strip()
        if not nodeid:
            return str(self._state.get("test", {}).get("cmd") or "")
        quoted = _shq(nodeid)
        return (
            f"{self._workspace_cd()}\n(pytest -q --maxfail=1 {quoted} 2>/dev/null || "
            f"python3 -m pytest -q --maxfail=1 {quoted})"
        )

    async def _run_tests(self, environment, target: Optional[str] = None) -> Tuple[int, str, str]:
        if not self._agent_local_validation_enabled():
            del environment, target
            self._disable_agent_local_validation_state()
            self._state["test"]["last_rc"] = 89
            self._state["test"]["last_success"] = False
            return 89, "", str(self._state["test"]["cmd"] or "")

        base = self._state["test"]["cmd"] or await self._resolve_test_cmd(environment)
        runner = self._state["test"]["runner"]
        is_pytest_runner = "pytest" in str(runner or "")
        can_target_pytest = is_pytest_runner and self._runner_is_authoritative(runner)

        cmd = base
        if target and can_target_pytest:
            cmd = self._build_targeted_pytest_cmd(target)

        t0 = time.time()
        _log(f"running tests: {cmd}")
        rc, out = await self._exec_with_rc(environment, cmd, _TIMEOUT_TEST)
        tail = self._tail(out)

        nodeid = self._parse_pytest_nodeid(out) if is_pytest_runner else None
        success = self._is_test_success(rc, out)
        self._state["test"]["last_rc"] = rc
        self._state["test"]["last_success"] = bool(success)
        self._state["test"]["last_nodeid"] = nodeid
        self._state["test"]["last_tail"] = tail
        if self._runner_is_local_only(runner):
            self._remember_visible_text_candidates(out)
        self._last_test_output = self._build_exec_summary(
            cmd,
            rc,
            out,
            runner=str(runner or ""),
            nodeid=nodeid,
            max_chars=_MAX_TEST_TEXT_CHARS,
            max_lines=160,
        )
        await self._write_file(environment, _LAST_TEST_PATH, self._last_test_output)
        await self._persist_state(environment)
        _log(f"tests done in {_elapsed(t0)}: RC={rc} success={success} nodeid={nodeid}")
        return rc, out, cmd

    def _state_block(self) -> str:
        t = self._state["test"]
        tools = self._state["tools"]
        elapsed = time.time() - self._run_start if self._run_start else 0.0
        remaining = (
            None
            if _MAX_RUNTIME_SECONDS is None
            else max(0.0, _MAX_RUNTIME_SECONDS - elapsed)
        )
        remaining_text = f"{remaining:.0f}s" if remaining is not None else "unlimited"
        parts = [
            "STATE (authoritative):\n"
            f"- workspace_root: {self._workspace_root_path()}\n"
            f"- test_cmd: {t.get('cmd')}\n"
            f"- runner: {t.get('runner')}\n"
            f"- last_rc: {t.get('last_rc')}\n"
            f"- last_nodeid: {t.get('last_nodeid')}\n"
            f"- runner_authoritative: {self._runner_is_authoritative()}\n"
            f"- local_verifier_mentions_tests: {self._state['repo'].get('local_verifier_mentions_tests')}\n"
            f"- has_pytest: {tools.get('has_pytest')}\n"
            f"- has_verify_sh: {tools.get('has_verify_sh')}\n"
            f"- has_test_sh: {tools.get('has_test_sh')}\n"
            f"- agent_can_read_tests: {self._state['repo'].get('agent_can_read_tests')}\n"
            f"- has_uv: {tools.get('has_uv')}\n"
            f"- has_curl: {tools.get('has_curl')}\n"
            f"- has_env_shim: {tools.get('has_env_shim')}\n"
            f"- pkg_manager: {tools.get('pkg_manager')}\n"
            f"- has_make: {tools.get('has_make')}\n"
            f"- has_git: {tools.get('has_git')}\n"
            f"- strategy_phase: {self._state['progress'].get('strategy_phase', 'initial')}\n"
            f"- forced_hidden_evidence_retry: {self._state['progress'].get('forced_hidden_evidence_retry')}\n"
            f"- time_remaining: {remaining_text}\n"
        ]
        if self._success_markers:
            parts.append("SUCCESS MARKERS (stdout must contain one if present):\n")
            for m in self._success_markers[:3]:
                parts.append(f"  - {m}")
        local_verifier_files = self._state["repo"].get("local_verifier_files") or []
        if local_verifier_files:
            parts.append("LOCAL VERIFIER FILES:\n")
            for path in local_verifier_files[:12]:
                parts.append(f"  - {path}")
        repo_command_candidates = self._state["repo"].get("command_candidates") or []
        if repo_command_candidates:
            parts.append("DISCOVERED REPO COMMAND CANDIDATES:\n")
            for command in repo_command_candidates[:8]:
                parts.append(f"  - {command}")
        hidden_evidence_paths = self._state["repo"].get("hidden_evidence_paths") or []
        if hidden_evidence_paths:
            parts.append("VISIBLE SOURCE / DATA PATHS:\n")
            for path in hidden_evidence_paths[:_HIDDEN_EVIDENCE_MAX_FILES]:
                parts.append(f"  - {path}")
        observed_probe_paths = self._state["repo"].get("observed_probe_paths") or []
        if observed_probe_paths:
            parts.append("OBSERVED RUNTIME PROBE PATHS:\n")
            for path in observed_probe_paths[:_OBSERVED_PROBE_PATHS_MAX]:
                parts.append(f"  - {path}")
        if self._assertion_targets:
            parts.append("ASSERTION TARGETS (what the test expects):\n")
            for tgt in self._assertion_targets[:20]:
                parts.append(f"  - {tgt}")
        elif self._task_targets:
            parts.append("TASK TARGETS (from hidden verifier instruction):\n")
            for tgt in self._task_targets[:20]:
                parts.append(f"  - {tgt}")
        parts.append("Artifacts:\n"
                      f"- {self._map_repo_path(_STATE_PATH)}\n"
                      f"- {self._map_repo_path(_LAST_TEST_PATH)}\n"
                      f"- {self._map_repo_path(_LAST_OBS_PATH)}\n"
                      f"- {self._map_repo_path(_LAST_BOOT_PATH)}")
        return "\n".join(parts)

    def _build_model_retry_messages(
        self,
        system_prompt: str,
        instruction: str,
        *,
        hidden_tests_fallback: bool,
        error_text: str,
        last_obs_text: str,
        hidden_producer_context: str,
    ) -> List[dict]:
        parts: List[str] = [
            f"Previous model call failed: {error_text}",
            "Resetting to a compact state summary. Continue from the current workspace state and do not repeat the same long exploratory batch.",
        ]
        if hidden_tests_fallback:
            task_block = self._task_targets_block(include_instruction=True)
            if task_block:
                parts.append(task_block)
            hidden_evidence_paths_block = self._hidden_evidence_paths_block()
            if hidden_evidence_paths_block:
                parts.append(hidden_evidence_paths_block)
            hidden_service_block = self._hidden_service_focus_block()
            if hidden_service_block:
                parts.append(hidden_service_block)
            copy_first_block = self._hidden_copy_first_block()
            if copy_first_block:
                parts.append(copy_first_block)
            if self._hidden_exact_visible_matches:
                parts.append(
                    "TOP EXACT VISIBLE MATCHES:\n"
                    + "\n".join(f"  - {item}" for item in self._hidden_exact_visible_matches[:6])
                )
            if hidden_producer_context:
                parts.append(
                    "TOP PRODUCER SOURCE HEADS:\n"
                    + self._tail(hidden_producer_context, max_lines=60, max_chars=2200)
                )
        elif self._assertion_targets:
            parts.append(
                "ASSERTION TARGETS:\n"
                + "\n".join(f"  - {item}" for item in self._assertion_targets[:20])
            )
        parts.append(self._state_block())
        if last_obs_text:
            parts.append("LATEST COMMAND OUTPUT:\n" + _clip_inline(last_obs_text, 3200))
        last_tail = str(self._state.get("test", {}).get("last_tail") or "").strip()
        if last_tail:
            label = (
                "LATEST LOCAL SMOKE OUTPUT:"
                if hidden_tests_fallback
                else "LATEST TEST OUTPUT:"
            )
            parts.append(f"{label}\n{last_tail}")
        parts.append(
            (
                "Next: use 1-2 native `run_command` tool calls only. "
                if self._native_tool_calling
                else "Next: emit 1-2 <bash> blocks only. "
            )
            + "Prefer exact visible values and existing producer/build flows over broad package installs."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction},
            {"role": "user", "content": "\n\n".join(part for part in parts if part).strip()},
        ]

    def _handoff_to_official_verifier(
        self,
        context,
        detail: str,
        *,
        evidence: str = "",
        info: Optional[str] = None,
    ) -> None:
        self._append_intermediate("note", detail)
        self._status("completed", detail=detail)
        _log(detail)
        summary = str(evidence or self._last_test_output or "No authoritative local test output captured.").strip()
        message = info or "Agent phase skipped local validation and returned control to Harbor's official verifier."
        self._publish_context(context, f"{summary}\n[Info] {message}")

    async def _call_model(
        self,
        messages: List[dict],
        *,
        response_input: Optional[List[dict]] = None,
        previous_response_id: Optional[str] = None,
        system_prompt: str = "",
    ) -> Tuple[dict, int]:
        t0 = time.time()
        use_responses_api = self._openai_api_mode == "responses"
        if use_responses_api:
            msg_tokens_est = len(system_prompt) // 4
            source_items = response_input or build_response_input_from_messages(messages)
            for item in source_items:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if isinstance(content, str):
                    msg_tokens_est += len(content) // 4
                else:
                    msg_tokens_est += len(str(item.get("output", "") or "")) // 4
        else:
            msg_tokens_est = sum(len(m.get("content", "")) for m in messages) // 4
        _log(f"LLM call #{self._num_llm_calls + 1}: ~{msg_tokens_est} prompt tokens, model={self._model} ...")
        detail = f"call={self._num_llm_calls + 1} prompt_tokens~{msg_tokens_est}"

        async def _responses_create(**request):
            return await self._run_with_heartbeat(
                "waiting for model response",
                asyncio.wait_for(
                    self._client.responses.create(**request),
                    timeout=_TIMEOUT_LLM,
                ),
                detail=detail,
            )

        async def _chat_create(**request):
            return await self._run_with_heartbeat(
                "waiting for model response",
                asyncio.wait_for(
                    self._client.chat.completions.create(**request),
                    timeout=_TIMEOUT_LLM,
                ),
                detail=detail,
            )

        def _fallback_notice(from_mode: str, to_mode: str, exc: Exception) -> None:
            fallback_detail = f"{from_mode} -> {to_mode}: {type(exc).__name__}: {exc}"
            self._append_intermediate("llm_api_fallback", fallback_detail)
            _log(f"LLM API fallback {fallback_detail}")

        model_turn, resp, api_mode = await call_openai_model_with_fallback(
            self._client,
            api_mode=self._openai_api_mode,
            model=self._model,
            messages=messages,
            response_input=response_input if use_responses_api else None,
            previous_response_id=previous_response_id if use_responses_api else None,
            system_prompt=system_prompt,
            temperature=_TEMPERATURE,
            max_completion_tokens=_MAX_COMPLETION_TOKENS,
            native_tool_calling=self._native_tool_calling,
            call_responses=_responses_create,
            call_chat_completions=_chat_create,
            on_fallback=_fallback_notice,
        )
        self._openai_api_mode = api_mode
        text = str(model_turn.get("text", "") or "")
        tokens = _usage_tokens(resp)
        _log(f"LLM response in {_elapsed(t0)}: {tokens} tokens, {len(text)} chars")
        return model_turn, tokens

    def _normalize_model_turn(self, model_turn: object) -> Dict[str, Any]:
        if isinstance(model_turn, str):
            return {
                "text": model_turn,
                "tool_calls": [],
                "assistant_message": {"role": "assistant", "content": model_turn or ""},
            }
        if isinstance(model_turn, dict):
            text = str(model_turn.get("text", "") or "")
            tool_calls = list(model_turn.get("tool_calls") or [])
            assistant_message = model_turn.get("assistant_message")
            response_id = str(model_turn.get("response_id", "") or "")
            api_mode = str(model_turn.get("api_mode", "") or self._openai_api_mode)
            if not isinstance(assistant_message, dict):
                assistant_message = build_assistant_tool_message(
                    text=text,
                    tool_calls=tool_calls,
                )
            return {
                "text": text,
                "tool_calls": tool_calls,
                "assistant_message": assistant_message,
                "response_id": response_id,
                "api_mode": api_mode,
            }
        return {
            "text": str(model_turn or ""),
            "tool_calls": [],
            "assistant_message": {"role": "assistant", "content": str(model_turn or "")},
            "response_id": "",
            "api_mode": self._openai_api_mode,
        }

    def _fingerprint_cmds(self, cmds: List[str]) -> str:
        s = "\n".join(c.strip() for c in cmds if c.strip())
        s = re.sub(r"\s+", " ", s).strip()
        return s[:5000]

    def _timeout_for_model_command(self, cmd: str) -> float:
        lower = (cmd or "").lower()
        if any(
            token in lower
            for token in [
                "apt-get ",
                "apt ",
                "apk add",
                "dnf install",
                "yum install",
                "pacman -s",
                "pip install",
                "pip3 install",
                "python -m pip install",
                "python3 -m pip install",
                "uv pip",
                "uvx ",
            ]
        ):
            base_timeout = min(_TIMEOUT_INSTALL, _MODEL_INSTALL_TIMEOUT_CAP)
        elif any(token in lower for token in ["make ", "cmake ", "meson ", "ninja "]):
            base_timeout = _TIMEOUT_LONG_CMD
        else:
            base_timeout = _TIMEOUT_CMD

        if self._run_start and _MAX_RUNTIME_SECONDS is not None:
            remaining = _MAX_RUNTIME_SECONDS - (time.time() - self._run_start)
            available = remaining - _MODEL_COMMAND_RUNTIME_RESERVE
            if available <= 0:
                base_timeout = min(
                    base_timeout,
                    max(
                        _MODEL_COMMAND_HARD_FLOOR,
                        min(_MODEL_COMMAND_LOW_TIME_CAP, remaining / 3.0),
                    ),
                )
            else:
                base_timeout = min(base_timeout, available)

        return max(_MODEL_COMMAND_HARD_FLOOR, base_timeout)

    def _wrap_command_with_timeout(self, cmd: str, timeout_seconds: float) -> str:
        seconds = max(int(timeout_seconds), int(_MODEL_COMMAND_HARD_FLOOR))
        delim = "MEMOHARNESS_TIMEOUT_EOF"
        while delim in cmd:
            delim += "_X"
        return (
            "if command -v timeout >/dev/null 2>&1; then\n"
            f"timeout --preserve-status -k 5s {seconds}s bash <<'{delim}'\n"
            f"{cmd}\n"
            f"{delim}\n"
            "else\n"
            f"{cmd}\n"
            "fi"
        )

    def _is_forbidden_tests_write(self, cmd: str) -> bool:
        """Prevent tampering with official or bundled verifier scripts."""
        rendered = str(cmd or "")
        local_tokens = self._local_verifier_tokens()
        touches_tests = "/tests/" in rendered
        touches_local_verifier = any(token in rendered for token in local_tokens)
        if not touches_tests and not touches_local_verifier:
            return False
        # Allow read-only inspection and execution of verifier entrypoints.
        allowed = [
            r"\bcat\b",
            r"\bls\b",
            r"\bfind\b",
            r"\brg\b",
            r"\bgrep\b",
            r"\bsed\b",
            r"\bnl\b",
            r"\bhead\b",
            r"\btail\b",
            r"\bbash\s+/tests/(?:test|verify)\.sh\b",
            r"\bbash\s+[^\n]*(?:test|verify)\.sh\b",
            r"\bpython3?\s+[^\n]*(?:test_outputs|test_output)\.py\b",
            r"\bpython3?\s+-m\s+pytest\b",
            r"\bpytest\b",
        ]
        destructive = False
        protected_targets = ["/tests/"] + local_tokens
        for token in protected_targets:
            token_re = "/tests/" if token == "/tests/" else re.escape(token)
            patterns = [
                rf"(?:^|\s)(?:>|>>|1>|1>>|2>|2>>)\s*{token_re}(?:\s|$)",
                rf"\b(?:chmod|chown|mv|cp|rm|touch)\b[^\n]*{token_re}",
                rf"\btee\b[^\n]*{token_re}",
                rf"\bsed\s+-i\b[^\n]*{token_re}",
                rf"\bperl\s+-i\b[^\n]*{token_re}",
            ]
            if any(re.search(pattern, rendered) for pattern in patterns):
                destructive = True
                break
        if any(re.search(pat, rendered) for pat in allowed) and not destructive:
            return False
        # Block common write/edit operations.
        return True

    def _is_discouraged_hidden_stack_command(self, cmd: str) -> Optional[str]:
        if not self._hidden_copy_first_preferred():
            return None
        lowered = re.sub(r"\s+", " ", str(cmd or "").lower())
        if not lowered:
            return None
        if not any(term in lowered for term in _DISCOURAGED_HIDDEN_STACK_HINTS):
            return None

        explicit_text = "\n".join(
            [
                self._task_instruction,
                "\n".join(self._task_targets),
                "\n".join(self._assertion_targets),
                "\n".join(self._repo_command_candidates),
                "\n".join(self._hidden_exact_visible_matches),
                "\n".join(self._hidden_visible_text_candidates),
            ]
        ).lower()
        if any(term in explicit_text for term in _DISCOURAGED_HIDDEN_STACK_HINTS):
            return None

        return (
            "hidden-mode exact text/data candidates already exist; do not use embeddings, "
            "semantic-search, or OCR stacks before exhausting visible exact literals and "
            "producer/source paths"
        )

    def _command_summary(self, command: str, max_chars: int = 220) -> str:
        rendered = re.sub(r"\s+", " ", str(command or "").strip())
        if not rendered:
            return "(empty)"
        if len(rendered) <= max_chars:
            return rendered
        return rendered[: max_chars - 3] + "..."

    def _build_exec_summary(
        self,
        command: str,
        rc: int,
        out: str,
        *,
        runner: Optional[str] = None,
        nodeid: Optional[str] = None,
        max_chars: int = 4000,
        max_lines: int = 120,
    ) -> str:
        parts: List[str] = []
        if runner:
            parts.append(f"runner={runner}")
        parts.append(f"cmd={self._command_summary(command)}")
        parts.append(f"rc={int(rc)}")
        if nodeid:
            parts.append(f"nodeid={nodeid}")
        summary = _summarize_output(out, max_chars=max_chars, max_lines=max_lines)
        if summary:
            return " ".join(parts) + "\n" + summary
        return " ".join(parts)

    def _append_intermediate(self, label: str, text: str, *, pre_summarized: bool = False) -> None:
        if text is None:
            return
        rendered = str(text)
        if not rendered:
            return
        if pre_summarized:
            compact = _clip_inline(rendered, _MAX_INTERMEDIATE_CHARS)
        else:
            compact = _summarize_output(rendered, max_chars=_MAX_INTERMEDIATE_CHARS, max_lines=120)
        self._intermediate_outputs.append(f"[{label}] {compact}")
        if len(self._intermediate_outputs) > _MAX_INTERMEDIATE_ITEMS:
            self._intermediate_outputs = self._intermediate_outputs[-_MAX_INTERMEDIATE_ITEMS:]

    def _record_tools(self, commands: List[str]) -> None:
        for command in commands:
            if command is None:
                continue
            rendered = str(command)
            if not rendered:
                continue
            self._tools_invoked.append(_clip_inline(rendered, _MAX_TOOL_TRACE_CHARS))
        if len(self._tools_invoked) > _MAX_TOOL_TRACE_ITEMS:
            self._tools_invoked = self._tools_invoked[-_MAX_TOOL_TRACE_ITEMS:]

    def _latency_ms(self) -> int:
        if not self._run_start:
            return 0
        return int((time.time() - self._run_start) * 1000)

    def _publish_context(self, context, output: str, *, final_output: Optional[str] = None) -> None:
        populate_context(
            context,
            output,
            self._num_llm_calls,
            self._total_tokens,
            latency_ms=self._latency_ms(),
            tools_invoked=self._tools_invoked,
            intermediate_outputs=self._intermediate_outputs,
            status_events=self._status_events,
            current_status=self._current_status,
            final_output=final_output or output,
        )

    def _get_stagnation_hint(self, stag: int, phase: str) -> str:
        """Generate escalating hints based on stagnation level and strategy phase."""
        if stag == 0:
            return ""
        hints = []
        can_read_tests = self._agent_can_read_tests()
        if stag >= 1:
            if can_read_tests:
                hints.append(
                    "SUGGESTION: Re-read the test file to see EXACTLY what it expects. "
                    "Run: cat /tests/test_outputs.py | head -100\n"
                    "Also check the detected repo root in state: pwd && ls -la"
                )
            else:
                hints.append(
                    "SUGGESTION: /tests is not visible from the agent bootstrap context. "
                    "Do NOT invent verifier files; inspect the detected workspace root, README, build files, and repo-local checks instead."
                )
        if stag >= 2:
            hints.append(
                "ESCALATION: Try a different approach. Common fixes:\n"
                "1) If test expects a directory: mkdir -p /path/to/dir\n"
                "2) If test expects a file at a specific path: create it with exact content\n"
                "3) If pip install fails: try 'pip install --no-cache-dir' or 'uv pip install'\n"
                "4) If a command is missing: check if 'apt-get install' or 'pip install' provides it"
            )
        if stag >= 3:
            hints.append(
                "CRITICAL ESCALATION: The current approach is not working. Try:\n"
                "1) Read the README from the repo root: pwd && ls -la && cat README.md\n"
                "2) Read the Makefile/setup.py for build instructions\n"
                "3) Run the install step: pip install -e . or make install\n"
                "4) Check if a service needs to start: python -m http.server, etc.\n"
                "5) If uvx is required: python3 -m pip install -U uv (retry with --break-system-packages on Debian)"
            )
        if stag >= 4:
            if can_read_tests:
                hints.append(
                    "LAST RESORT: Nothing has worked so far. Try:\n"
                    "1) pip install -e . from the repo root\n"
                    "2) Run cat /tests/test.sh or cat /tests/verify.sh to see EXACT test commands\n"
                    "3) Ensure PATH includes ~/.local/bin: export PATH=$HOME/.local/bin:$PATH\n"
                    "4) Verify exact file content matches test assertions (whitespace, case, format)\n"
                    "5) Check if test needs a running server 鈥?start it in background with &"
                )
            else:
                hints.append(
                    "LAST RESORT: /tests is still not visible. Try:\n"
                    "1) Re-read the original task instruction and list the exact deliverables it names\n"
                    "2) Search the detected workspace root for those filenames/keywords to find the producer code or data source\n"
                    "3) Verify the named artifacts directly (ls/head/file) instead of optimizing for `find .`\n"
                    "4) Stop chasing /tests and avoid creating verifier files by hand"
                )
        return "\n".join(hints)

    def _get_urgency_message(self) -> str:
        """Generate urgency message when time is running low."""
        if not self._run_start or _MAX_RUNTIME_SECONDS is None:
            return ""
        remaining = _MAX_RUNTIME_SECONDS - (time.time() - self._run_start)
        if remaining < 120:
            return (
                f"\n\nTIME CRITICAL: Only {remaining:.0f}s remaining. "
                "Focus on the single most likely fix. Do NOT explore; execute the fix NOW."
            )
        if remaining < 300:
            return (
                f"\n\nTime warning: {remaining:.0f}s remaining. "
                "Prioritize the most impactful fix."
            )
        return ""

    def _should_handoff_low_signal_advisory_loop(self, *, turn: int) -> bool:
        if not self._uses_hidden_tests_fallback():
            return False
        if self._runner_is_rich_hidden_local():
            return False
        if not self._runner_is_low_signal_local():
            return False
        rich_hidden_targets = self._hidden_rich_retry_targets_present()
        min_turns = max(_LOCAL_SANITY_MAX_TURNS, _LOW_SIGNAL_HANDOFF_MIN_TURNS + (2 if rich_hidden_targets else 0))
        min_stagnation = _LOW_SIGNAL_HANDOFF_MIN_STAGNATION + (1 if rich_hidden_targets else 0)
        if turn < min_turns:
            return False
        if (
            rich_hidden_targets
            and not bool(self._state.get("progress", {}).get("forced_hidden_evidence_retry"))
        ):
            return False
        if self._hidden_unresolved_signal_present():
            return False
        if int(self._state.get("progress", {}).get("stagnation") or 0) < min_stagnation:
            return False
        return True

    async def _pre_flight_setup(self, environment) -> None:
        """Run verifier-oriented pre-flight: bootstrap tooling and create required directories."""
        await self._detect_pkg_manager(environment)

        # Common verifier assumptions: /root/.local/bin/env exists and is executable.
        await self._ensure_env_shim(environment)
        await self._ensure_python_aliases(environment)

        hidden_verifier_mode = not self._agent_can_read_tests()
        if hidden_verifier_mode:
            _log("pre-flight: Harbor verifier runs post-agent -> skipping agent-side local validation")

        # Install baseline tools if tests reference them.
        if self._needs.get("curl"):
            self._state["tools"]["has_curl"] = await self._ensure_command(
                environment,
                "curl",
                packages=["curl", "ca-certificates"],
                reason="verifier scripts reference curl",
            )
        if self._needs.get("wget"):
            self._state["tools"]["has_wget"] = await self._ensure_command(
                environment,
                "wget",
                packages=["wget", "ca-certificates"],
                reason="verifier scripts reference wget",
            )
        if self._needs.get("git"):
            self._state["tools"]["has_git"] = await self._ensure_command(
                environment,
                "git",
                packages=["git"],
                reason="verifier scripts reference git",
            )
        if self._needs.get("env_shim"):
            await self._ensure_env_shim(environment)

        # Ensure python/pip if tests reference them (or uv needs them).
        await self._ensure_python_pip(environment)
        await self._ensure_pytest(environment)

        # Install uv/uvx if verifier references it (prefer pip wheels over GitHub release downloads).
        await self._ensure_uv(environment)

        # Create commonly required directories that tests check for (safe and deterministic).
        for target in self._assertion_targets:
            if target.startswith("directory: ") or target.startswith("directory_candidate: "):
                path = self._map_repo_path(target.split(": ", 1)[1])
                await self._safe_exec(environment, f"mkdir -p {_shq(path)}", _TIMEOUT_PROBE)

        # Tool availability may have changed; resolve again.
        if self._agent_local_validation_enabled():
            await self._resolve_test_cmd(environment)

    async def run(self, instruction: str, environment, context) -> None:
        self._run_start = time.time()
        self._last_test_output = ""
        self._tools_invoked = []
        self._intermediate_outputs = []
        self._status_events = []
        self._current_status = None
        self._assertion_targets = []
        self._task_targets = []
        self._task_instruction = instruction or ""
        self._test_content_cache = ""
        self._success_markers = []
        self._needs = {}
        self._uv_installed = False
        self._env_setup_done = False
        self._pkg_manager = None
        self._pkg_update_done = False
        self._auto_fixes_done = set()
        self._workspace_root = "/app"
        self._hidden_exact_visible_matches = []
        self._hidden_visible_text_candidates = []
        self._hidden_evidence_paths = []
        self._repo_command_candidates = []
        self._observed_probe_paths = []
        self._state["tools"]["has_pytest"] = None
        self._state["tools"]["has_rg"] = None
        self._state["tools"]["has_curl"] = None
        self._state["tools"]["has_wget"] = None
        self._state["tools"]["has_python3"] = None
        self._state["tools"]["has_pip3"] = None
        self._state["tools"]["has_verify_sh"] = None
        self._state["tools"]["has_test_sh"] = None
        self._state["tools"]["has_uv"] = None
        self._state["tools"]["has_make"] = None
        self._state["tools"]["has_git"] = None
        self._state["tools"]["has_env_shim"] = None
        self._state["tools"]["pkg_manager"] = None
        self._state["repo"]["app_top"] = []
        self._state["repo"]["tests_files"] = []
        self._state["repo"]["agent_can_read_tests"] = False
        self._state["repo"]["local_verifier_files"] = []
        self._state["repo"]["local_verifier_mentions_tests"] = False
        self._state["repo"]["workspace_root"] = "/app"
        self._state["repo"]["workspace_boot_cwd"] = None
        self._state["repo"]["workspace_alias_enabled"] = False
        self._state["repo"]["command_candidates"] = []
        self._state["repo"]["hidden_evidence_paths"] = []
        self._state["repo"]["observed_probe_paths"] = []
        self._state["progress"]["turn"] = 0
        self._state["progress"]["stagnation"] = 0
        self._state["progress"]["last_patch_fingerprint"] = None
        self._state["progress"]["strategy_phase"] = "initial"
        self._state["progress"]["forced_hidden_evidence_retry"] = False
        self._task_targets = _extract_instruction_targets(self._task_instruction)
        self._state["repo"]["instruction_targets"] = self._task_targets[:20]
        _attach_config(context, self._cfg)
        self._status("starting harness run", detail=f"model={self._model}")
        _log(f"=== run start === model={self._model}")

        # --- connectivity diagnostics ---
        self._status("daytona connectivity check")
        _log("diag: testing Daytona exec (echo hello)...")
        t0 = time.time()
        try:
            diag_result = await asyncio.wait_for(environment.exec("echo hello"), timeout=15.0)
            diag_stdout = _extract_stdout(diag_result)
            diag_rc = _extract_rc(diag_result)
            _log(f"diag: Daytona exec OK in {_elapsed(t0)} -> stdout={repr(diag_stdout)[:60]} rc={diag_rc}")
        except Exception as exc:
            _log(f"diag: Daytona exec FAILED in {_elapsed(t0)}: {type(exc).__name__}: {exc}")

        self._status("llm api check")
        _log("diag: testing LLM API (short request)...")
        t0 = time.time()
        try:
            diag_resp = await self._run_with_heartbeat(
                "llm api check",
                asyncio.wait_for(
                    self._client.chat.completions.create(
                        model=self._model,
                        messages=[{"role": "user", "content": "Say OK"}],
                        max_completion_tokens=8,
                    ),
                    timeout=30.0,
                ),
                detail="diagnostic request",
            )
            diag_text = getattr(diag_resp.choices[0].message, "content", "") or ""
            _log(f"diag: LLM API OK in {_elapsed(t0)} -> {repr(diag_text)[:60]}")
        except Exception as exc:
            _log(f"diag: LLM API FAILED in {_elapsed(t0)}: {type(exc).__name__}: {exc}")
        # --- end diagnostics ---

        system_prompt = (
            "You are an automated code-fixing agent operating inside a live Linux terminal.\n"
            "Your goal is to make the verifier/tests pass.\n\n"
            "CRITICAL RULES:\n"
            + (
                "1) Use the native `run_command` tool for every shell command you need to execute.\n"
                "2) Keep actions short: at most 2 tool calls per turn; focus on the next fix.\n"
                if self._native_tool_calling
                else
                "1) Every response MUST contain one or more <bash>...</bash> blocks with shell commands.\n"
                "2) Keep responses short: at most 2 <bash> blocks; focus on the next fix.\n"
            )
            +
            "3) Treat /tests and any verifier-like files (test_outputs.py, test.sh, verify.sh, ./tests, ./test) as READ-ONLY clues: never edit/overwrite/chmod them to make a failure disappear.\n"
            "4) Do NOT bypass the verifier (no rewriting test scripts, no `exit 0` hacks).\n"
            "5) Harbor uploads the official verifier under /tests after the agent finishes, so do NOT rely on /tests during bootstrap or repair turns.\n"
            "6) Make minimal, targeted changes 鈥?do not rewrite entire files.\n"
            "7) After creating/editing files, ALWAYS verify with: cat <path> | head -20\n"
            "8) The harness will NOT run repo-local tests or advisory probes during agent turns; do NOT invent or run Harbor's official /tests verifier yourself.\n"
            "9) When the test expects a FILE at a PATH, create that file BEFORE the harness reruns tests.\n"
            "10) When the test expects a DIRECTORY, create it with mkdir -p.\n"
            "11) Assume outbound network may be restricted; prefer OS packages + pip wheels over GitHub release downloads.\n"
            "12) If a server must run for the test, start it in the background (&) and verify it's listening.\n"
            "13) Prefer `python3` / `python3 -m pip` over `python` / `pip` unless you verified those commands exist.\n"
            "14) The original user instruction is the primary contract; keep it in mind every turn.\n"
            "15) Do not guess from world knowledge if local files can answer it; inspect the detected workspace root first.\n"
            "16) When visible local text/data files already contain candidate answers, copy the best exact visible literal first and only fall back to deterministic transformation if the task clearly requires it; do not jump to embeddings, semantic search, OCR, brute-force, disk/forensics tooling, or speculative inference.\n"
            "17) Ignore harness-generated files under .artifacts as evidence; prefer real repo files, path matches, and visible source lines.\n"
            "18) Do not treat the current contents of the target output file as proof that the answer is correct; corroborate with source/data files, producer stderr/stdout, or real process/socket checks.\n"
            "19) When a visible literal looks generic (for example dummy/sample/placeholder/timestamp text), treat it as a decoy unless the task terms or target pattern clearly match it.\n"
            "20) When a repo-local producer script already exists, inspect its source and latest stderr before reimplementing it or installing large dependency stacks.\n"
            "21) If exact text/data candidates already exist, do not install or import embedding, semantic-search, or OCR stacks (for example sentence-transformers, transformers, FAISS/Chroma, tesseract, OpenCV, easyocr, paddleocr) before exhausting those literals and producer/source paths.\n"
            "22) When a task names ports, sockets, or long-running processes, prove the live service with a single PID, /proc/<pid>/cmdline, and the required ss socket/port evidence before assuming it is correct.\n\n"
            "WORKFLOW (repeat):\n"
            "1) Start from the detected workspace root, README, build metadata, and repo-local verifier-like files.\n"
            "2) Harbor's official verifier is post-agent; do not expect /tests during repair turns and do not invent it.\n"
            "3) The harness will not run local validation between turns; use your own focused inspections or producer checks when helpful.\n"
            "4) From assertions and local evidence, list required paths/commands/expected strings.\n"
            "5) Prefer exact visible local evidence over cleverness: if a small repo-visible file already contains the needed line, password, move list, or artifact source, split hyphenated instruction tokens into meaningful parts, search those parts separately, and copy the best exact visible literal before installing search stacks or launching embeddings/OCR/forensics; ignore generic dummy/sample/status lines and do not use the target output file itself as source evidence.\n"
            "6) When local verifier-like files reference /tests or other hidden paths, treat them as read-only hints, not as patch targets, and inspect existing producer source before writing replacement automation.\n"
            "7) For service tasks, prefer process/socket/port proofs over bare file existence checks.\n"
            "8) When you finish the most likely fix or run out of concrete next moves, stop and let Harbor's official verifier take over.\n\n"
            "COMMAND STYLE:\n"
            "- Always cd the detected workspace root first for repo commands.\n"
            "- Use heredoc for multi-line file creation: cat > /path/to/file <<'EOF' ... EOF\n"
            "- Use sed -i for small edits.\n"
            "- Use python3 -c for complex inline operations when available.\n"
            "- Verify file creation: ls -la <path> && cat <path> | head -20\n"
            "- If pip install fails, try: python3 -m pip install --no-cache-dir or uv pip install\n"
            "- Prefer rg for search: rg -n \"pattern\" .\n"
            "- Keep commands short and idempotent.\n"
            "- Start background services with: python3 server.py &"
        )

        messages: List[dict] = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": instruction})
        use_responses_api = self._openai_api_mode == "responses"
        response_input: List[dict] = (
            [build_response_input_message("user", instruction)]
            if use_responses_api
            else []
        )
        previous_response_id: Optional[str] = None

        def _set_next_response_input(
            user_content: str,
            *,
            prefix_items: Optional[List[dict]] = None,
        ) -> None:
            nonlocal response_input
            response_input = list(prefix_items or [])
            response_input.append(build_response_input_message("user", user_content))

        self._status("workspace bootstrap")
        boot = await self._bootstrap(environment)
        self._append_intermediate("bootstrap", boot)
        if self._agent_local_validation_enabled():
            await self._resolve_test_cmd(environment)
        else:
            self._disable_agent_local_validation_state()

        # Deep discovery: read README, Makefile, etc.
        self._status("deep discovery")
        discovery = await self._deep_discovery(environment)
        self._append_intermediate("discovery", discovery)
        _log(f"deep discovery: {len(discovery)} chars")
        self._needs = self._merge_needs(
            self._needs,
            self._infer_needs_from_test_text(
                "\n".join(filter(None, [self._task_instruction, discovery])),
                self._task_targets,
            ),
        )

        # Read test file contents to discover verifier requirements
        self._status("reading test files")
        test_content = await self._read_test_content(environment)
        self._test_content_cache = test_content
        _log(f"read test content: {len(test_content)} chars, extracted {len(self._assertion_targets)} assertion targets")
        self._needs = self._merge_needs(
            self._needs,
            self._infer_needs_from_test_text(test_content, self._assertion_targets),
        )
        if not self._agent_can_read_tests():
            self._constrain_hidden_mode_needs()

        # Pre-flight: install uv if needed, create missing directories
        self._status("pre-flight setup")
        await self._pre_flight_setup(environment)
        self._env_setup_done = True

        # Ground truth: run full suite once (maxfail=1) to get a concrete failure.
        if self._agent_local_validation_enabled():
            self._status("running initial tests")
        rc = out = cmd = None
        if self._agent_local_validation_enabled():
            rc, out, cmd = await self._run_tests(environment, target=None)
        if self._agent_local_validation_enabled():
            self._append_intermediate("test", self._last_test_output, pre_summarized=True)
        hidden_tests_fallback = self._uses_hidden_tests_fallback()
        local_validation_enabled = self._agent_local_validation_enabled()
        if hidden_tests_fallback and not local_validation_enabled:
            self._append_intermediate("note", _POST_AGENT_VERIFIER_NOTE)
            _log(_POST_AGENT_VERIFIER_NOTE)
        if local_validation_enabled and hidden_tests_fallback:
            detail = (
                "Harbor's official verifier runs after agent execution; "
                "using lightweight local advisory probes between turns"
            )
            self._append_intermediate("note", detail)
            _log(detail)
        if local_validation_enabled and self._state["test"].get("last_success") is True:
            self._status("completed", detail="tests pass on first run")
            _log(f"tests pass on first run 鈥?done in {_elapsed(self._run_start)}")
            self._publish_context(context, "All tests passed (RC=0 + success markers).")
            return

        _log(f"initial test failed (RC={rc}) 鈥?attempting auto-fixes")

        auto_actions: List[str] = []
        if local_validation_enabled:
            auto_actions = await self._auto_fix_from_test_output(environment, out)
        if local_validation_enabled and auto_actions:
            self._append_intermediate("auto_fix", "\n".join(auto_actions))
            self._status("rerunning tests after auto-fix")
            rc2, out2, cmd2 = await self._run_tests(environment, target=None)
            self._append_intermediate("test", self._last_test_output, pre_summarized=True)
            if self._state["test"].get("last_success") is True:
                self._status("completed", detail="tests pass after auto-fix")
                _log(f"tests pass after auto-fix 鈥?done in {_elapsed(self._run_start)}")
                self._publish_context(context, "All tests passed (after auto-fix).")
                return
            rc, out, cmd = rc2, out2, cmd2

        _log("entering agent loop")

        # Check which assertion targets are missing
        missing_items = await self._missing_items(environment)
        missing_report = ""
        if missing_items:
            missing_report = "MISSING PATHS (must be created):\n" + "\n".join(f"  - {m}" for m in missing_items)
        _log(f"missing paths check: {missing_report[:200] if missing_report else 'all present'}")
        producer_hints = await self._producer_discovery(environment, missing_items)
        hidden_repo_evidence = (
            await self._hidden_sparse_repo_evidence(environment)
            if hidden_tests_fallback
            else ""
        )
        hidden_producer_hints = (
            await self._producer_discovery_for_instruction(environment)
            if hidden_tests_fallback
            else ""
        )
        hidden_producer_context = (
            await self._build_hidden_producer_context(environment)
            if hidden_tests_fallback
            else ""
        )
        if hidden_repo_evidence:
            self._remember_observed_probe_paths(hidden_repo_evidence)
        if hidden_producer_hints:
            self._remember_observed_probe_paths(hidden_producer_hints)
        if hidden_producer_context:
            self._remember_observed_probe_paths(hidden_producer_context)
        if hidden_producer_hints:
            self._append_intermediate("hidden_producer_discovery", hidden_producer_hints)
        if hidden_repo_evidence:
            self._append_intermediate(
                "hidden_repo_evidence",
                hidden_repo_evidence,
                pre_summarized=True,
            )
        if hidden_producer_context:
            self._append_intermediate(
                "hidden_producer_context",
                hidden_producer_context,
                pre_summarized=True,
            )

        # Build the initial agent message with test content and failure
        test_info = ""
        if test_content:
            test_info = (
                "TEST FILE CONTENTS (read these to understand what the verifier expects):\n"
                f"{self._tail(test_content, max_lines=150, max_chars=6000)}\n\n"
            )

        assertion_info = ""
        if self._assertion_targets:
            assertion_info = (
                "ASSERTION TARGETS extracted from tests (create these BEFORE running tests):\n"
                + "\n".join(f"  - {t}" for t in self._assertion_targets[:25])
                + "\n\n"
            )

        missing_info = ""
        if missing_report:
            missing_info = f"{missing_report}\n\n"

        producer_info = ""
        if producer_hints:
            producer_info = (
                "PRODUCER DISCOVERY (search in the detected repo root for missing artifacts):\n"
                f"{self._tail(producer_hints, max_lines=80, max_chars=2500)}\n\n"
            )
        hidden_producer_info = ""
        if hidden_producer_hints:
            hidden_producer_info = (
                "PRODUCER DISCOVERY (instruction-derived search terms in the detected repo root):\n"
                f"{self._tail(hidden_producer_hints, max_lines=80, max_chars=2500)}\n\n"
            )
        hidden_producer_context_info = ""
        if hidden_producer_context:
            hidden_producer_context_info = f"{hidden_producer_context}\n\n"
        hidden_repo_evidence_info = ""
        if hidden_repo_evidence:
            hidden_repo_evidence_info = (
                "VISIBLE HIDDEN-MODE SOURCE / DATA PREVIEW:\n"
                f"{self._tail(hidden_repo_evidence, max_lines=90, max_chars=2800)}\n\n"
            )

        discovery_info = ""
        if discovery:
            discovery_info = (
                "PROJECT DISCOVERY (README, build files):\n"
                f"{self._tail(discovery, max_lines=60, max_chars=2000)}\n\n"
            )

        # Include env setup info
        env_info = ""
        env_bits: List[str] = []
        if self._state["tools"].get("has_env_shim"):
            env_bits.append("/root/.local/bin/env shim present")
        if self._state["tools"].get("has_uv"):
            env_bits.append("uv/uvx available")
        if env_bits:
            env_info = "ENV SETUP: " + "; ".join(env_bits) + ".\n\n"

        hidden_verifier_info = ""
        if hidden_tests_fallback:
            hidden_verifier_info = (
                "POST-AGENT VERIFIER MODE:\n"
                "Harbor uploads and runs its official verifier after the agent finishes. "
                "During repair turns, use the original task instruction as the primary contract. "
                "The agent phase does not run harness-managed local validation, so use your own focused "
                "inspections or producer checks when helpful. "
                "Only Harbor's official verifier can "
                "confirm success.\n\n"
            )
            if self._local_verifier_mentions_hidden_tests():
                hidden_verifier_info += (
                    "Bundled verifier-like files under the detected repo root reference Harbor's later /tests paths. "
                    "Treat those files as read-only evidence and do not patch them.\n\n"
                )
        hidden_term_info = ""
        if hidden_tests_fallback:
            hidden_term_block = self._instruction_terms_block()
            if hidden_term_block:
                hidden_term_info = f"{hidden_term_block}\n\n"
        hidden_service_info = ""
        if hidden_tests_fallback:
            hidden_service_block = self._hidden_service_focus_block()
            if hidden_service_block:
                hidden_service_info = f"{hidden_service_block}\n\n"
        hidden_evidence_paths_info = ""
        if hidden_tests_fallback:
            hidden_evidence_paths_block = self._hidden_evidence_paths_block()
            if hidden_evidence_paths_block:
                hidden_evidence_paths_info = f"{hidden_evidence_paths_block}\n\n"
        hidden_visible_text_info = ""
        if hidden_tests_fallback and self._hidden_visible_text_candidates:
            hidden_visible_text_info = (
                "VISIBLE TEXT CANDIDATES FROM FOCUSED INSPECTIONS (copy literal lines from here before "
                "guessing or installing heavy search stacks):\n"
                + "\n".join(
                    f"  - {item}"
                    for item in self._hidden_visible_text_candidates[:_VISIBLE_TEXT_CANDIDATE_MAX_ITEMS]
                )
                + "\n\n"
            )
        copy_first_info = ""
        if hidden_tests_fallback:
            copy_first_block = self._hidden_copy_first_block()
            if copy_first_block:
                copy_first_info = f"{copy_first_block}\n\n"

        if hidden_tests_fallback:
            task_block = self._task_targets_block(include_instruction=True)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Bootstrap complete. /tests was not visible, so solve the task from the original instruction "
                        "plus the local repo evidence from the detected workspace root. "
                        "The agent phase will not run local tests or advisory probes between turns.\n\n"
                        f"{hidden_verifier_info}"
                        f"{task_block}\n\n"
                        f"{discovery_info}"
                        f"{env_info}"
                        f"{hidden_service_info}"
                        f"{hidden_evidence_paths_info}"
                        f"{hidden_repo_evidence_info}"
                        f"{copy_first_info}"
                        f"{hidden_visible_text_info}"
                        f"{hidden_producer_info}"
                        f"{hidden_producer_context_info}"
                        f"{hidden_term_info}"
                        f"{self._state_block()}\n"
                        "Task:\n"
                        "1. Re-read the original user instruction above and identify the required deliverable(s).\n"
                        "2. Inspect the detected workspace root, README/build files, and producer-search hits for the named artifacts/commands.\n"
                        "3. Break hyphenated or slash-separated instruction tokens into searchable parts, then if a probed source file already contains candidate lines or bytes that match those terms, copy the exact visible source value instead of paraphrasing or choosing a semantically related one.\n"
                        "4. Prefer deterministic offline-safe commands; install only small missing tools that the task or repo clearly requires.\n"
                        "5. Execute an existing task-specific producer script before inventing new automation when one is visible, and inspect that producer source plus its latest stderr before rewriting it.\n"
                        "6. Create or repair the deliverable, then verify the named artifact/process with your own focused checks; for services, prefer single-PID plus /proc cmdline and socket/port proofs.\n"
                        "7. When you have finished the strongest fix you can justify from the visible repo evidence, stop and hand control back to Harbor's official verifier.\n"
                        "8. Do not spend turns on embeddings, OCR, or disk/forensics search until the visible literals, sparse file previews, and producer heads above are exhausted; if exact text/data candidates already exist, do not install or import sentence-transformers, transformers, FAISS/Chroma, tesseract, OpenCV, easyocr, or paddleocr first."
                    ),
                }
            )
            if use_responses_api:
                response_input.append(build_response_input_message("user", messages[-1]["content"]))
        else:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Bootstrap complete. Here is the workspace state, test files, and the latest test failure.\n\n"
                        f"{discovery_info}"
                        f"{env_info}"
                        f"{hidden_verifier_info}"
                        f"{test_info}"
                        f"{assertion_info}"
                        f"{missing_info}"
                        f"{producer_info}"
                        f"{self._state_block()}\n"
                        "LATEST TEST OUTPUT (tail):\n"
                        f"{self._state['test']['last_tail']}\n\n"
                        "Task:\n"
                        "1. Read the test assertions above to understand EXACTLY what files/outputs/commands are expected.\n"
                        "2. Check if any expected paths are missing (listed above as MISSING).\n"
                        "3. Install missing dependencies first (pip install, apt-get, uv).\n"
                        "4. Create any missing directories: mkdir -p <path>\n"
                        "5. Create or fix the minimum needed to satisfy each assertion.\n"
                        "6. Verify your changes before running tests.\n"
                        "If last_nodeid is present, focus on that specific test first."
                    ),
                }
            )
            if use_responses_api:
                response_input.append(build_response_input_message("user", messages[-1]["content"]))

        last_fingerprints: List[str] = []
        last_test_fingerprints: List[str] = []
        last_obs_text = self._last_test_output
        model_error_retries = 0
        last_model_error_detail = ""
        try:
            seed_fp = self._fingerprint_cmds(
                [
                    f"RC={self._state['test'].get('last_rc')}",
                    str(self._state['test'].get("last_tail") or ""),
                ]
            )
            if seed_fp:
                last_test_fingerprints.append(seed_fp)
        except Exception:
            pass

        turn = 0
        loop_exit_detail = ""
        while True:
            turn += 1
            self._state["progress"]["turn"] = turn - 1

            remaining = (
                None
                if _MAX_RUNTIME_SECONDS is None
                else max(0.0, _MAX_RUNTIME_SECONDS - (time.time() - self._run_start))
            )
            if remaining is not None and remaining < 60:
                _log("Less than 60s remaining; making a final attempt")
            remaining_label = f"{remaining:.1f}s" if remaining is not None else "unlimited"

            _log(
                f"--- turn {turn} (elapsed {_elapsed(self._run_start)}, "
                f"remaining {remaining_label}, calls={self._num_llm_calls}, "
                f"tokens={self._total_tokens}) ---"
            )

            if not use_responses_api:
                messages = _trim_messages(messages, _HISTORY_KEEP)
            try:
                if use_responses_api:
                    model_turn_raw, used = await self._call_model(
                        messages,
                        response_input=response_input,
                        previous_response_id=previous_response_id,
                        system_prompt=system_prompt,
                    )
                else:
                    model_turn_raw, used = await self._call_model(messages)
            except asyncio.TimeoutError:
                last_model_error_detail = f"timeout after {_TIMEOUT_LLM:.0f}s"
                self._append_intermediate("model_error", last_model_error_detail)
                _log(f"LLM TIMEOUT after {_TIMEOUT_LLM:.0f}s at turn {turn}")
                if model_error_retries < _MAX_MODEL_ERROR_RETRIES:
                    model_error_retries += 1
                    self._status(
                        "recovering from model error",
                        detail=f"retry={model_error_retries} reason=timeout",
                    )
                    retry_messages = self._build_model_retry_messages(
                        system_prompt,
                        instruction,
                        hidden_tests_fallback=hidden_tests_fallback,
                        error_text=last_model_error_detail,
                        last_obs_text=last_obs_text,
                        hidden_producer_context=hidden_producer_context,
                    )
                    if use_responses_api:
                        previous_response_id = None
                        response_input = build_response_input_from_messages(retry_messages)
                    else:
                        messages = retry_messages
                    continue
                if hidden_tests_fallback:
                    detail = (
                        "model call timed out in post-agent advisory mode; "
                        "handing control back to Harbor's verifier"
                    )
                    self._handoff_to_official_verifier(
                        context,
                        detail,
                        evidence=last_obs_text,
                        info="Model calls timed out after compact retries, so MemoHarness returned control to Harbor's official verifier.",
                    )
                    return
                loop_exit_detail = f"model call timed out after compact retries at turn {turn}"
                break
            except Exception as exc:
                last_model_error_detail = f"{type(exc).__name__}: {exc}"
                self._append_intermediate("model_error", last_model_error_detail)
                _log(f"LLM ERROR at turn {turn}: {exc}")
                if model_error_retries < _MAX_MODEL_ERROR_RETRIES:
                    model_error_retries += 1
                    self._status(
                        "recovering from model error",
                        detail=f"retry={model_error_retries} reason={type(exc).__name__}",
                    )
                    retry_messages = self._build_model_retry_messages(
                        system_prompt,
                        instruction,
                        hidden_tests_fallback=hidden_tests_fallback,
                        error_text=last_model_error_detail,
                        last_obs_text=last_obs_text,
                        hidden_producer_context=hidden_producer_context,
                    )
                    if use_responses_api:
                        previous_response_id = None
                        response_input = build_response_input_from_messages(retry_messages)
                    else:
                        messages = retry_messages
                    continue
                if hidden_tests_fallback:
                    detail = (
                        "model call failed repeatedly in post-agent advisory mode; "
                        "handing control back to Harbor's verifier"
                    )
                    self._handoff_to_official_verifier(
                        context,
                        detail,
                        evidence=last_obs_text,
                        info="Model calls failed after compact retries, so MemoHarness returned control to Harbor's official verifier.",
                    )
                    return
                loop_exit_detail = (
                    "model call failed after compact retries at turn "
                    f"{turn}: {last_model_error_detail}"
                )
                break

            self._num_llm_calls += 1
            self._total_tokens += used
            model_error_retries = 0
            last_model_error_detail = ""

            model_turn = self._normalize_model_turn(model_turn_raw)
            text = str(model_turn.get("text", "") or "")
            self._openai_api_mode = str(
                model_turn.get("api_mode") or self._openai_api_mode or "chat_completions"
            )
            use_responses_api = self._openai_api_mode == "responses"
            if use_responses_api:
                previous_response_id = str(
                    model_turn.get("response_id") or previous_response_id or ""
                )
            else:
                previous_response_id = None
            assistant_message = model_turn.get("assistant_message") or {
                "role": "assistant",
                "content": text,
            }
            tool_calls = [
                call for call in list(model_turn.get("tool_calls") or [])
                if isinstance(call, dict)
            ]
            bash_blocks = [b.strip() for b in extract_bash_blocks(text) if b.strip()]
            model_summary = strip_bash_blocks(text).strip()
            if model_summary:
                self._append_intermediate("assistant", model_summary)

            command_blocks: List[Dict[str, Any]] = []
            used_native_tool_calls = bool(tool_calls)
            if tool_calls:
                for tool_call in tool_calls:
                    arguments = tool_call.get("arguments", {})
                    command_blocks.append(
                        {
                            "command": str(
                                arguments.get("command", "") if isinstance(arguments, dict) else ""
                            ).strip(),
                            "tool_call_id": str(tool_call.get("id") or ""),
                            "tool_name": str(tool_call.get("name") or ""),
                        }
                    )
            else:
                command_blocks = [
                    {
                        "command": cmd,
                        "tool_call_id": "",
                        "tool_name": "run_command",
                    }
                    for cmd in bash_blocks
                    if cmd.strip()
                ]

            if not command_blocks:
                if hidden_tests_fallback and not local_validation_enabled:
                    detail = (
                        "model response had no executable commands in post-agent verifier mode; "
                        "handing control back to Harbor's verifier"
                    )
                    evidence = last_obs_text or text or "No command output captured."
                    self._handoff_to_official_verifier(
                        context,
                        detail,
                        evidence=evidence,
                        info="The model stopped emitting executable commands, so MemoHarness returned control to Harbor's official verifier.",
                    )
                    return
                target = self._state["test"].get("last_nodeid")
                await self._run_tests(environment, target=target)
                last_obs_text = self._last_test_output
                obs = (
                    "MODEL ERROR: no native tool calls or legacy <bash> blocks were provided.\n\n"
                    f"{self._state_block()}\n"
                    "LATEST TEST OUTPUT (tail):\n"
                    f"{self._state['test']['last_tail']}\n"
                )
                await self._write_file(environment, _LAST_OBS_PATH, obs)
                prompt_content = (
                    (
                        "You MUST use the native `run_command` tool for shell actions. "
                        if self._native_tool_calling
                        else "You MUST respond with <bash> commands. "
                    )
                    + (
                        "Re-read the original task instruction above, inspect the detected workspace root, and make the smallest change "
                        "that moves the named deliverable forward."
                        if hidden_tests_fallback
                        else "If /tests is visible, start by reading the failing test file with: "
                        "cat /tests/test_outputs.py\nThen create the files or make the changes the test expects."
                    )
                )
                if use_responses_api:
                    messages.append({"role": "assistant", "content": text or "[empty]"})
                    messages.append(
                        {
                            "role": "user",
                            "content": prompt_content,
                        }
                    )
                    _set_next_response_input(prompt_content)
                else:
                    messages.append({"role": "assistant", "content": text or "[empty]"})
                    messages.append(
                        {
                            "role": "user",
                            "content": prompt_content,
                        }
                    )
                self._state["progress"]["stagnation"] += 1
                _log(
                    "turn {0}: no tool calls or bash blocks, stagnation={1}".format(
                        turn,
                        self._state["progress"]["stagnation"],
                    )
                )
                continue

            # Execute each model-requested command directly (not wrapped in bash -lc)
            fp = self._fingerprint_cmds(
                [
                    item["command"] or f"[tool:{item['tool_name']}]"
                    for item in command_blocks
                ]
            )
            if fp in last_fingerprints[-2:]:
                self._state["progress"]["stagnation"] += 1
                stag = self._state["progress"]["stagnation"]
                if hidden_tests_fallback and not local_validation_enabled:
                    detail = (
                        "command batch repeated without new executable progress in post-agent verifier mode; "
                        "handing control back to Harbor's verifier"
                    )
                    self._handoff_to_official_verifier(
                        context,
                        detail,
                        evidence=last_obs_text or "No command output captured.",
                        info="Command batches repeated without new progress, so MemoHarness returned control to Harbor's official verifier.",
                    )
                    return
                stag_hint = self._get_stagnation_hint(stag, self._state["progress"].get("strategy_phase", "initial"))
                prompt_content = (
                    "You repeated the same command batch. Choose a different approach.\n\n"
                    f"{stag_hint}\n\n"
                    f"{self._state_block()}\n"
                    "LATEST TEST OUTPUT (tail):\n"
                    f"{self._state['test']['last_tail']}\n"
                )
                if use_responses_api:
                    messages.append(
                        assistant_message if used_native_tool_calls else {"role": "assistant", "content": text}
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": prompt_content,
                        }
                    )
                    _set_next_response_input(prompt_content)
                else:
                    messages.append(
                        assistant_message if used_native_tool_calls else {"role": "assistant", "content": text}
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": prompt_content,
                        }
                    )
                _log(f"turn {turn}: repeated commands, stagnation={stag}")
                continue
            last_fingerprints.append(fp)
            # Do not reset stagnation just because commands differ; we update it based on test progress.

            # Strategy phase transitions
            if turn <= 2:
                self._state["progress"]["strategy_phase"] = "initial"
            elif turn <= 5:
                self._state["progress"]["strategy_phase"] = "targeted"
            else:
                self._state["progress"]["strategy_phase"] = "escalation"

            command_labels = [
                item["command"] or f"[tool:{item['tool_name']}]"
                for item in command_blocks
            ]
            self._status("executing model commands", detail=f"commands={len(command_blocks)}")
            _log(f"turn {turn}: executing {len(command_blocks)} command(s)")
            response_tool_outputs: List[dict] = []
            if used_native_tool_calls:
                messages.append(assistant_message)
            self._record_tools(command_labels)
            self._append_intermediate("command_batch", " | ".join(command_labels))
            outputs: List[str] = []
            for index, item in enumerate(command_blocks, start=1):
                blk = str(item.get("command") or "")
                tool_name = str(item.get("tool_name") or "")
                tool_call_id = str(item.get("tool_call_id") or f"tool_call_{index}")
                rc_blk = 0
                stdout = ""
                if used_native_tool_calls and tool_name != "run_command":
                    rc_blk = 997
                    stdout = f"[tool error] unknown tool: {tool_name or '(empty)'}."
                elif not blk:
                    rc_blk = 996
                    stdout = "[tool error] missing required `command` argument."
                elif self._is_forbidden_tests_write(blk):
                    rc_blk = 999
                    stdout = (
                        "[blocked] attempted to modify /tests or a verifier-like file. "
                        "Only read or execute verifier scripts; do not edit them."
                    )
                else:
                    discouraged_reason = self._is_discouraged_hidden_stack_command(blk)
                    if discouraged_reason:
                        rc_blk = 998
                        stdout = f"[blocked] {discouraged_reason}."
                    else:
                        timeout_s = self._timeout_for_model_command(blk)
                        rc_blk, stdout = await self._exec_with_rc(
                            environment,
                            self._wrap_command_with_timeout(blk, timeout_s),
                            timeout_s + 20.0,
                        )
                outputs.append(
                    self._build_exec_summary(
                        blk,
                        rc_blk,
                        stdout,
                        max_chars=5000,
                        max_lines=120,
                    )
                )
                if used_native_tool_calls:
                    if use_responses_api:
                        response_tool_outputs.append(
                            build_function_call_output_item(
                                tool_call_id,
                                outputs[-1] or stdout,
                            )
                        )
                    messages.append(
                        build_tool_result_message(
                            tool_call_id,
                            outputs[-1] or stdout,
                        )
                    )
            obs_text = "\n\n".join(chunk for chunk in outputs if chunk).strip()
            if len(obs_text) > _MAX_OBS_TEXT_CHARS:
                obs_text = _clip_inline(obs_text, _MAX_OBS_TEXT_CHARS)
            auto_obs_actions = await self._auto_fix_from_command_output(environment, obs_text)
            if auto_obs_actions:
                auto_fix_block = "AUTO-FIXES:\n" + "\n".join(f"- {item}" for item in auto_obs_actions)
                self._append_intermediate("auto_fix", "\n".join(auto_obs_actions))
                obs_text = (obs_text + "\n\n" + auto_fix_block).strip()
                if len(obs_text) > _MAX_OBS_TEXT_CHARS:
                    obs_text = _clip_inline(obs_text, _MAX_OBS_TEXT_CHARS)
            self._remember_observed_probe_paths(obs_text)
            self._append_intermediate("observation", obs_text, pre_summarized=True)
            await self._write_file(environment, _LAST_OBS_PATH, obs_text)
            last_obs_text = obs_text

            tests_became_visible = False
            if hidden_tests_fallback:
                hints_changed = False
                if self._should_refresh_hidden_repo_evidence(
                    turn=turn,
                    cached_text=hidden_repo_evidence,
                    obs_text=obs_text,
                ):
                    refreshed_hidden_repo_evidence = await self._hidden_sparse_repo_evidence(environment)
                    if refreshed_hidden_repo_evidence != hidden_repo_evidence:
                        hidden_repo_evidence = refreshed_hidden_repo_evidence
                        if hidden_repo_evidence:
                            self._remember_observed_probe_paths(hidden_repo_evidence)
                            self._append_intermediate(
                                "hidden_repo_evidence",
                                hidden_repo_evidence,
                                pre_summarized=True,
                            )
                if self._should_refresh_hidden_discovery(
                    turn=turn,
                    cached_text=hidden_producer_hints,
                    obs_text=obs_text,
                ):
                    refreshed_hidden_hints = await self._producer_discovery_for_instruction(environment)
                    if refreshed_hidden_hints != hidden_producer_hints:
                        hidden_producer_hints = refreshed_hidden_hints
                        hints_changed = True
                        if hidden_producer_hints:
                            self._remember_observed_probe_paths(hidden_producer_hints)
                            self._append_intermediate(
                                "hidden_producer_discovery",
                                hidden_producer_hints,
                            )
                if self._should_refresh_hidden_producer_context(
                    turn=turn,
                    cached_text=hidden_producer_context,
                    obs_text=obs_text,
                    hints_changed=hints_changed,
                ):
                    refreshed_hidden_context = await self._build_hidden_producer_context(environment)
                    if refreshed_hidden_context != hidden_producer_context:
                        hidden_producer_context = refreshed_hidden_context
                        if hidden_producer_context:
                            self._remember_observed_probe_paths(hidden_producer_context)
                            self._append_intermediate(
                                "hidden_producer_context",
                                hidden_producer_context,
                                pre_summarized=True,
                            )
                if local_validation_enabled:
                    await self._resolve_test_cmd(environment)

            # Pre-flight: check if assertion target paths now exist
            missing_after = await self._check_missing_paths(environment)

            if hidden_tests_fallback and not local_validation_enabled:
                if turn >= _LOCAL_SANITY_MAX_TURNS:
                    detail = (
                        f"agent reached {_LOCAL_SANITY_MAX_TURNS} repair turns without agent-side local validation; "
                        "handing control back to Harbor's verifier"
                    )
                    self._handoff_to_official_verifier(
                        context,
                        detail,
                        evidence=last_obs_text or "No command output captured.",
                        info="MemoHarness stopped after the configured repair-turn budget and returned control to Harbor's official verifier.",
                    )
                    return

                stag = self._state["progress"]["stagnation"]
                stag_hint = self._get_stagnation_hint(
                    stag,
                    self._state["progress"].get("strategy_phase", "initial"),
                )
                urgency = self._get_urgency_message()
                missing_info = f"\n\n{missing_after}" if missing_after else ""
                task_reminder = ""
                if self._task_targets:
                    task_reminder = (
                        "\n\nRE-READ TASK CONTRACT (post-agent verifier mode):\n"
                        + "\n".join(f"  - {t}" for t in self._task_targets[:20])
                        + "\n"
                    )
                instruction_terms_reminder = ""
                hidden_term_block = self._instruction_terms_block()
                if hidden_term_block:
                    instruction_terms_reminder = f"\n\n{hidden_term_block}\n"
                service_reminder = ""
                hidden_service_block = self._hidden_service_focus_block()
                if hidden_service_block:
                    service_reminder = f"\n\n{hidden_service_block}\n"
                evidence_paths_reminder = ""
                hidden_evidence_paths_block = self._hidden_evidence_paths_block()
                if hidden_evidence_paths_block:
                    evidence_paths_reminder = f"\n\n{hidden_evidence_paths_block}\n"
                visible_text_reminder = ""
                if self._hidden_visible_text_candidates:
                    visible_text_reminder = (
                        "\n\nVISIBLE TEXT CANDIDATES FROM FOCUSED INSPECTIONS:\n"
                        + "\n".join(
                            f"  - {item}"
                            for item in self._hidden_visible_text_candidates[:_VISIBLE_TEXT_CANDIDATE_MAX_ITEMS]
                        )
                        + "\n"
                    )
                copy_first_reminder = ""
                copy_first_block = self._hidden_copy_first_block()
                if copy_first_block:
                    copy_first_reminder = f"\n\n{copy_first_block}\n"
                hidden_search_reminder = ""
                if hidden_producer_hints:
                    hidden_search_reminder = (
                        "\n\nRE-READ EXACT VISIBLE MATCHES / PRODUCER SEARCH:\n"
                        f"{self._tail(hidden_producer_hints, max_lines=50, max_chars=2200)}\n"
                    )
                hidden_producer_context_reminder = ""
                if hidden_producer_context:
                    hidden_producer_context_reminder = (
                        "\n\nRE-READ PRODUCER SOURCE HEADS:\n"
                        f"{self._tail(hidden_producer_context, max_lines=70, max_chars=2400)}\n"
                    )

                prompt_content = (
                    "COMMAND OUTPUT:\n"
                    f"{obs_text}\n\n"
                    "AGENT-PHASE VALIDATION:\n"
                    "Harbor's official verifier will run after handoff. "
                    "There is no harness-run local validation between turns.\n\n"
                    f"{self._state_block()}\n"
                    f"{missing_info}"
                    f"{task_reminder}"
                    f"{instruction_terms_reminder}"
                    f"{service_reminder}"
                    f"{evidence_paths_reminder}"
                    f"{copy_first_reminder}"
                    f"{visible_text_reminder}"
                    f"{hidden_search_reminder}"
                    f"{hidden_producer_context_reminder}"
                    f"{urgency}"
                    "Next: choose the smallest change that gets the workspace closer to the original task "
                    "instruction. If visible repo evidence already gives the answer or producer fix, apply that "
                    "exact change. When you finish the strongest fix you can justify from the visible workspace "
                    "state, stop and hand control back to Harbor's official verifier.\n"
                    f"{stag_hint}"
                )
                if use_responses_api:
                    if not used_native_tool_calls:
                        messages.append({"role": "assistant", "content": text})
                    messages.append(
                        {
                            "role": "user",
                            "content": prompt_content,
                        }
                    )
                    _set_next_response_input(
                        prompt_content,
                        prefix_items=response_tool_outputs if used_native_tool_calls else None,
                    )
                else:
                    if not used_native_tool_calls:
                        messages.append({"role": "assistant", "content": text})
                    messages.append(
                        {
                            "role": "user",
                            "content": prompt_content,
                        }
                    )
                continue

            target = self._state["test"].get("last_nodeid")
            self._status(
                "rerunning targeted test" if target else "rerunning full test suite",
                detail=target,
            )
            rc_t, out_t, cmd_t = await self._run_tests(environment, target=target)
            self._append_intermediate("test", self._last_test_output, pre_summarized=True)
            last_obs_text = self._last_test_output

            # Stagnation: same failing RC + tail repeatedly => escalate guidance.
            test_fp = self._fingerprint_cmds(
                [
                    f"RC={rc_t}",
                    str(self._state["test"].get("last_tail") or ""),
                ]
            )
            if self._state["test"].get("last_success") is not True:
                if last_test_fingerprints and test_fp == last_test_fingerprints[-1]:
                    self._state["progress"]["stagnation"] += 1
                else:
                    self._state["progress"]["stagnation"] = 0
            last_test_fingerprints.append(test_fp)

            if self._state["test"].get("last_success") is True:
                self._status("verifying full test suite")
                _log("targeted test passed 鈥?verifying full suite...")
                rc_full, out_full, cmd_full = await self._run_tests(environment, target=None)
                self._append_intermediate("test", self._last_test_output, pre_summarized=True)
                if self._state["test"].get("last_success") is True:
                    self._status("completed", detail="full suite passed")
                    _log(f"=== PASS === full suite RC=0 in {_elapsed(self._run_start)} "
                         f"(calls={self._num_llm_calls}, tokens={self._total_tokens})")
                    final_text = strip_bash_blocks(text).strip() or "All tests passed (RC=0)."
                    self._publish_context(context, final_text, final_output=final_text)
                    return

            if self._hidden_advisory_ready_for_handoff():
                detail = (
                    "rich hidden advisory shows clean local producer/service proof; "
                    "handing control back to Harbor's verifier"
                )
                self._append_intermediate("note", detail)
                self._status("completed", detail=detail)
                _log(detail)
                self._publish_context(
                    context,
                    (self._last_test_output or "No authoritative local test output captured.")
                    + "\n[Info] Hidden-mode local checks produced concrete clean evidence, so "
                    + "MemoHarness returned control to Harbor's official verifier without burning "
                    + "more turns on advisory reruns.",
                )
                return

            if self._should_force_hidden_evidence_retry(
                turn=turn,
            ):
                self._state["progress"]["forced_hidden_evidence_retry"] = True
                retry_detail = (
                    "forcing one last evidence-focused hidden retry before low-signal handoff"
                )
                self._append_intermediate("note", retry_detail)
                _log(retry_detail)
                retry_parts: List[str] = [
                    "FINAL EVIDENCE-FOCUSED RETRY BEFORE HANDOFF:",
                    "The latest advisory probe stayed low-signal. Use the best visible literal or the clearest producer/process mismatch now.",
                    self._state_block(),
                ]
                copy_first_block = self._hidden_copy_first_block()
                if copy_first_block:
                    retry_parts.append(copy_first_block)
                hidden_service_block = self._hidden_service_focus_block()
                if hidden_service_block:
                    retry_parts.append(hidden_service_block)
                if hidden_producer_hints:
                    retry_parts.append(
                        "TOP EXACT VISIBLE MATCHES / PRODUCER SEARCH:\n"
                        + self._tail(hidden_producer_hints, max_lines=50, max_chars=2200)
                    )
                if hidden_producer_context:
                    retry_parts.append(
                        "TOP PRODUCER SOURCE HEADS:\n"
                        + self._tail(hidden_producer_context, max_lines=70, max_chars=2400)
                    )
                last_tail = str(self._state.get("test", {}).get("last_tail") or "").strip()
                if last_tail:
                    retry_parts.append(
                        "LATEST LOCAL SMOKE OUTPUT (secondary signal only):\n" + last_tail
                    )
                retry_parts.append(
                    (
                        "Next: use one small native `run_command` tool call that either copies the best exact visible literal into the deliverable or repairs the concrete producer/process mismatch above. "
                        if self._native_tool_calling
                        else "Next: emit one small <bash> block that either copies the best exact visible literal into the deliverable or repairs the concrete producer/process mismatch above. "
                    )
                    + "If a service is involved, prove a single PID plus /proc/<pid>/cmdline and the required port/socket."
                )
                prompt_content = "\n\n".join(
                    part for part in retry_parts if str(part or "").strip()
                )
                if use_responses_api:
                    messages.append(
                        assistant_message if used_native_tool_calls else {"role": "assistant", "content": text}
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": prompt_content,
                        }
                    )
                    _set_next_response_input(
                        prompt_content,
                        prefix_items=response_tool_outputs if used_native_tool_calls else None,
                    )
                else:
                    messages.append(
                        assistant_message if used_native_tool_calls else {"role": "assistant", "content": text}
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": prompt_content,
                        }
                    )
                continue
            if self._should_handoff_low_signal_advisory_loop(
                turn=turn,
            ):
                runner = str(self._state.get("test", {}).get("runner") or "")
                detail = (
                    f"advisory local runner '{runner}' stayed low-signal after repeated reruns; "
                    "handing control back to Harbor's verifier"
                )
                self._append_intermediate("note", detail)
                self._status("completed", detail=detail)
                _log(detail)
                self._publish_context(
                    context,
                    (self._last_test_output or "No authoritative local test output captured.")
                    + "\n[Info] Advisory local checks stayed low-signal, so MemoHarness returned "
                    + "control to Harbor's official verifier instead of spending more turns on probes.",
                )
                return

            stag = self._state["progress"]["stagnation"]
            stag_hint = self._get_stagnation_hint(stag, self._state["progress"].get("strategy_phase", "initial"))
            urgency = self._get_urgency_message()

            # Build user message with context
            missing_info = ""
            if missing_after:
                missing_info = f"\n\n{missing_after}"

            # On stagnation>=2, re-inject test content to refresh context
            test_reminder = ""
            if (tests_became_visible or stag >= 2) and self._test_content_cache:
                # Re-extract assertion targets from test content (they may have changed)
                new_targets = _extract_assertion_targets(self._test_content_cache)
                if new_targets:
                    self._assertion_targets = list(dict.fromkeys(new_targets + self._assertion_targets))[:40]
                    test_reminder = (
                        "\n\nRE-READ TEST ASSERTIONS (these MUST pass):\n"
                        + "\n".join(f"  - {t}" for t in self._assertion_targets[:20])
                        + "\n"
                    )

            task_reminder = ""
            if hidden_tests_fallback and self._task_targets:
                task_reminder = (
                    "\n\nRE-READ TASK CONTRACT (post-agent advisory mode):\n"
                    + "\n".join(f"  - {t}" for t in self._task_targets[:20])
                    + "\n"
                )
            instruction_terms_reminder = ""
            if hidden_tests_fallback:
                hidden_term_block = self._instruction_terms_block()
                if hidden_term_block:
                    instruction_terms_reminder = f"\n\n{hidden_term_block}\n"
            service_reminder = ""
            if hidden_tests_fallback:
                hidden_service_block = self._hidden_service_focus_block()
                if hidden_service_block:
                    service_reminder = f"\n\n{hidden_service_block}\n"
            evidence_paths_reminder = ""
            if hidden_tests_fallback:
                hidden_evidence_paths_block = self._hidden_evidence_paths_block()
                if hidden_evidence_paths_block:
                    evidence_paths_reminder = f"\n\n{hidden_evidence_paths_block}\n"
            visible_text_reminder = ""
            if hidden_tests_fallback and self._hidden_visible_text_candidates:
                visible_text_reminder = (
                    "\n\nVISIBLE TEXT CANDIDATES FROM FOCUSED INSPECTIONS:\n"
                    + "\n".join(
                        f"  - {item}"
                        for item in self._hidden_visible_text_candidates[:_VISIBLE_TEXT_CANDIDATE_MAX_ITEMS]
                    )
                    + "\n"
                )
            hidden_validation_note = ""
            if hidden_tests_fallback and not local_validation_enabled:
                hidden_validation_note = (
                    "AGENT-PHASE VALIDATION:\n"
                    "Harbor's official verifier will run after handoff. "
                    "There is no harness-run local validation between turns.\n\n"
                )
            copy_first_reminder = ""
            if hidden_tests_fallback:
                copy_first_block = self._hidden_copy_first_block()
                if copy_first_block:
                    copy_first_reminder = f"\n\n{copy_first_block}\n"
            hidden_search_reminder = ""
            if hidden_tests_fallback and hidden_producer_hints:
                hidden_search_reminder = (
                    "\n\nRE-READ EXACT VISIBLE MATCHES / PRODUCER SEARCH:\n"
                    f"{self._tail(hidden_producer_hints, max_lines=50, max_chars=2200)}\n"
                )
            hidden_producer_context_reminder = ""
            if hidden_tests_fallback and hidden_producer_context:
                hidden_producer_context_reminder = (
                    "\n\nRE-READ PRODUCER SOURCE HEADS:\n"
                    f"{self._tail(hidden_producer_context, max_lines=70, max_chars=2400)}\n"
                )

            if hidden_tests_fallback:
                prompt_content = (
                    "COMMAND OUTPUT:\n"
                    f"{obs_text}\n\n"
                    f"{hidden_validation_note}"
                    f"{self._state_block()}\n"
                    f"{missing_info}"
                    f"{task_reminder}"
                    f"{instruction_terms_reminder}"
                    f"{service_reminder}"
                    f"{evidence_paths_reminder}"
                    f"{copy_first_reminder}"
                    f"{visible_text_reminder}"
                    f"{hidden_search_reminder}"
                    f"{hidden_producer_context_reminder}"
                    f"{urgency}"
                    "Next: choose the smallest change that gets the workspace closer to the original task "
                    "instruction. If the latest probe already shows an exact source line or value that matches "
                    "the instruction terms, write that exact visible value into the deliverable instead of a "
                    "related guess. If an existing producer source head or latest producer stderr already pinpoints "
                    "the failure, fix that exact producer flow before adding large dependency stacks, then verify "
                    "the named artifact/process with focused local checks. Do not spend turns on embeddings, OCR, or disk/forensics search until the "
                    "visible literals and source/data files above are exhausted; if exact text/data candidates "
                    "already exist, do not install or import sentence-transformers, transformers, FAISS/Chroma, "
                    "tesseract, OpenCV, easyocr, or paddleocr first.\n"
                    f"{stag_hint}"
                )
            else:
                prompt_content = (
                    "COMMAND OUTPUT:\n"
                    f"{obs_text}\n\n"
                    "LATEST TEST OUTPUT (tail):\n"
                    f"{self._state['test']['last_tail']}\n\n"
                    f"{self._state_block()}\n"
                    f"{missing_info}"
                    f"{test_reminder}"
                    f"{urgency}"
                    "Next: propose the minimal patch to fix the failure, then re-run the failing test.\n"
                    f"{stag_hint}"
                )
            if use_responses_api:
                if not used_native_tool_calls:
                    messages.append({"role": "assistant", "content": text})
                messages.append(
                    {
                        "role": "user",
                        "content": prompt_content,
                    }
                )
                _set_next_response_input(
                    prompt_content,
                    prefix_items=response_tool_outputs if used_native_tool_calls else None,
                )
            else:
                if not used_native_tool_calls:
                    messages.append({"role": "assistant", "content": text})
                messages.append(
                    {
                        "role": "user",
                        "content": prompt_content,
                    }
                )

        if (
            _MAX_RUNTIME_SECONDS is not None
            and self._run_start
            and (time.time() - self._run_start) >= _MAX_RUNTIME_SECONDS
        ):
            completion_detail = f"max runtime reached ({int(_MAX_RUNTIME_SECONDS)}s) without RC=0"
            completion_warning = "[Warning] Max runtime reached without RC=0 on full test command."
            completion_label = f"max runtime {int(_MAX_RUNTIME_SECONDS)}s"
        elif loop_exit_detail:
            completion_detail = loop_exit_detail
            completion_warning = f"[Warning] {loop_exit_detail}"
            completion_label = loop_exit_detail
        else:
            completion_detail = "agent loop exited without reaching RC=0"
            completion_warning = "[Warning] Agent loop exited without reaching RC=0 on full test command."
            completion_label = "agent loop exited without success"

        self._status("completed", detail=completion_detail)
        _log(
            f"=== DONE ({completion_label}) === "
            f"elapsed {_elapsed(self._run_start)}, calls={self._num_llm_calls}, "
            f"tokens={self._total_tokens}"
        )
        self._publish_context(
            context,
            (self._last_test_output or "No test output captured.")
            + "\n"
            + completion_warning,
        )
