from __future__ import annotations

import math
import pickle
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

from ..core.models import (
    BenchmarkCase,
    CaseStats,
    FeatureFilter,
    GlobalPattern,
    PerCaseEntry,
    RetrievalRequest,
    RetrievedSlice,
)

# Type aliases for embedding functions
EmbedFn = Callable[[str], list[float]]
SimilarityFn = Callable[[list[float], list[float]], float]


@dataclass
class SimilarCaseResults:
    successful: list[PerCaseEntry]
    failed: list[PerCaseEntry]


class ExperienceBank:
    """
    Stores and retrieves per-case execution records and global patterns.

    An embedding function is required for similarity-based retrieval.
    The recommended approach is to pass an ``openai_client``:

        bank = ExperienceBank(openai_client=client)
    """

    def __init__(
        self,
        embed_fn: Optional[EmbedFn] = None,
        similarity_fn: Optional[SimilarityFn] = None,
        use_semantic_embedding: bool = False,
        openai_client=None,
        embedding_model: str = "text-embedding-3-small",
        min_consecutive_failures: int = 3,
    ) -> None:
        """
        Initialize ExperienceBank.

        Args:
            embed_fn: Custom embedding function. If provided, takes precedence.
            similarity_fn: Custom similarity function. If provided, takes precedence.
            use_semantic_embedding: If True and openai_client is provided, use OpenAI embeddings.
                When openai_client is provided this flag is automatically set.
            openai_client: OpenAI client for semantic embeddings.
            embedding_model: Model name for OpenAI embeddings.
        """
        self._custom_embed_fn = embed_fn
        self._custom_similarity_fn = similarity_fn
        self._openai_embedder = None

        # Auto-enable semantic embedding when a client is provided
        if openai_client is not None:
            use_semantic_embedding = True

        if use_semantic_embedding and openai_client is not None:
            from ..llm.embedder import OpenAIEmbedder
            self._openai_embedder = OpenAIEmbedder(
                client=openai_client,
                model=embedding_model,
            )

        self.entries: list[PerCaseEntry] = []
        self.case_stats: dict[str, CaseStats] = {}
        self.global_patterns: list[GlobalPattern] = []
        self.case_index: dict[str, list[PerCaseEntry]] = defaultdict(list)
        self.dim_index: dict[str, set[str]] = defaultdict(set)
        self.cluster_index: dict[str, set[str]] = defaultdict(set)
        # Tracks how many entries have been added since the last distillation.
        # Used to trigger distillation every N new entries, independent of
        # iteration boundaries.
        self.last_distill_entry_count: int = 0
        self.min_consecutive_failures: int = min_consecutive_failures

    def add_entry(self, entry: PerCaseEntry) -> bool:
        """Add *entry* to the bank and return True if distillation should run.

        Distillation is triggered immediately when any case reaches
        ``min_consecutive_failures`` consecutive failures, before the entry
        is committed.  Callers are responsible for actually running
        ``distill_*`` and then calling ``mark_distill_done()``.
        """
        self.entries.append(entry)
        self.case_index[entry.case_id].append(entry)
        self.case_stats.setdefault(entry.case_id, CaseStats()).update(entry)

        primary_dim = entry.diagnosis.diagnostic_signal.primary_dim
        self.dim_index[primary_dim].add(entry.case_id)

        cluster_id = entry.case_features.cluster_id or entry.case_features.domain
        self.cluster_index[cluster_id].add(entry.case_id)

        self.last_distill_entry_count += 1

        # Immediate trigger: any case just hit the consecutive-failure threshold
        return self._check_consecutive_failure_trigger(entry.case_id)

    def should_distill(self, every_n_entries: int) -> bool:
        """Return True when either trigger condition is met.

        Args:
            every_n_entries: trigger when ``last_distill_entry_count >= every_n_entries``.
        """
        return self.last_distill_entry_count >= every_n_entries

    def mark_distill_done(self) -> None:
        """Reset the entry counter after a distillation run completes."""
        self.last_distill_entry_count = 0

    # ------------------------------------------------------------------
    # Distillation triggers
    # ------------------------------------------------------------------

    def _check_consecutive_failure_trigger(self, case_id: str) -> bool:
        """Return True when *case_id* just reached the consecutive-failure threshold.

        Uses ``self.min_consecutive_failures`` configured at construction time.
        Fires on the entry that caused the count to cross the threshold so
        distillation sees the updated stats immediately.
        """
        stats = self.case_stats.get(case_id)
        if stats is None:
            return False
        return stats.consecutive_failures >= self.min_consecutive_failures

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, request: RetrievalRequest) -> RetrievedSlice:
        matched = [entry for entry in self.entries if self._matches_request(entry, request)]
        if request.sample_by_cluster_k:
            matched = self._sample_by_cluster(matched, request.sample_by_cluster_k)
        elif request.sample_k and len(matched) > request.sample_k:
            matched = random.sample(matched, request.sample_k)

        matched = sorted(matched, key=lambda entry: entry.iteration, reverse=True)
        if request.max_entries is not None:
            matched = matched[:request.max_entries]

        relevant_stats = {entry.case_id: self.case_stats[entry.case_id] for entry in matched}
        patterns = list(self.global_patterns) if request.include_global_patterns else []
        if request.max_global_patterns is not None:
            patterns = patterns[-request.max_global_patterns:]
        return RetrievedSlice(global_patterns=patterns, entries=matched, case_stats=relevant_stats)

    def retrieve_similar_cases(
        self,
        instruction: str,
        top_k_success: int = 10,
        top_k_failure: int = 10,
    ) -> SimilarCaseResults:
        ranked = sorted(
            self.entries,
            key=lambda entry: self._cosine_similarity(
                self._embed_text(instruction),
                self._embed_text(entry.case_features.instruction or entry.case_id),
            ),
            reverse=True,
        )
        successful = [entry for entry in ranked if entry.diagnosis.success][:top_k_success]
        failed = [entry for entry in ranked if not entry.diagnosis.success][:top_k_failure]
        return SimilarCaseResults(successful=successful, failed=failed)

    def retrieve_similar_cases_for_case(
        self,
        case: BenchmarkCase,
        top_k_success: int = 10,
        top_k_failure: int = 10,
    ) -> SimilarCaseResults:
        ranked = sorted(
            self.entries,
            key=lambda entry: self._case_similarity(case, entry),
            reverse=True,
        )
        successful = [entry for entry in ranked if entry.diagnosis.success][:top_k_success]
        failed = [entry for entry in ranked if not entry.diagnosis.success][:top_k_failure]
        return SimilarCaseResults(successful=successful, failed=failed)

    def distill_global_patterns(
        self,
        current_iteration: int,
        min_consecutive_failures: int = 3,
        max_patterns: int = 3,
    ) -> list[GlobalPattern]:
        failing_case_ids = [
            case_id
            for case_id, stats in self.case_stats.items()
            if stats.consecutive_failures >= min_consecutive_failures
        ]
        if not failing_case_ids:
            return []

        grouped: dict[str, list[PerCaseEntry]] = defaultdict(list)
        for case_id in failing_case_ids:
            for entry in self.case_index.get(case_id, []):
                grouped[entry.diagnosis.diagnostic_signal.primary_dim].append(entry)

        distilled: list[GlobalPattern] = []
        for primary_dim, entries in sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True):
            evidence = sorted({entry.case_id for entry in entries})[:5]
            latest_iteration = max(entry.iteration for entry in entries)
            description = self._build_pattern_description(primary_dim, entries)
            effect = self._build_pattern_effect(primary_dim, entries)
            pattern = GlobalPattern(
                pattern_id=f"pattern-{current_iteration}-{primary_dim.lower()}",
                iteration_range=(max(0, latest_iteration - 4), current_iteration),
                description=description,
                evidence=evidence,
                effect=effect,
                primary_dim=primary_dim,
            )
            distilled.append(pattern)
            if len(distilled) >= max_patterns:
                break

        self.global_patterns.extend(distilled)
        return distilled

    def distill_global_patterns_llm(
        self,
        distiller,
        current_iteration: int,
    ) -> list[GlobalPattern]:
        """Use an LLMDistiller to extract global patterns and append them to the bank."""
        patterns = distiller.distill(self, current_iteration)
        self.global_patterns.extend(patterns)
        return patterns

    def save(self, path: "str | Path") -> None:
        """Persist the full bank to *path* (pickle) and a sibling .json for inspection."""
        import dataclasses
        import json as _json
        import os
        import tempfile

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        # --- primary: pickle (fast, lossless, used at runtime) — atomic write ---
        fd, tmp_pkl = tempfile.mkstemp(dir=p.parent, suffix=".pkl.tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                pickle.dump(self, fh)
            os.replace(tmp_pkl, p)
        except BaseException:
            try:
                os.unlink(tmp_pkl)
            except OSError:
                pass
            raise

        # --- secondary: human-readable JSON (for inspection only) ---
        def _to_json(obj):
            """Recursively convert dataclasses / sets / tuples to JSON-safe types."""
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return {k: _to_json(v) for k, v in dataclasses.asdict(obj).items()}
            if isinstance(obj, dict):
                return {str(k): _to_json(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_json(i) for i in obj]
            if isinstance(obj, set):
                return sorted(_to_json(i) for i in obj)
            return obj

        snapshot = {
            "entries": [_to_json(e) for e in self.entries],
            "case_stats": {
                cid: {
                    "consecutive_failures": stats.consecutive_failures,
                    "avg_reward_recent_N": stats.avg_reward_recent_N,
                    "reward_trend": stats.reward_trend,
                    "dim_failure_counts": dict(stats.dim_failure_counts),
                    "history": [_to_json(r) for r in stats.history],
                }
                for cid, stats in self.case_stats.items()
            },
            "global_patterns": [_to_json(gp) for gp in self.global_patterns],
            "meta": {
                "total_entries": len(self.entries),
                "total_patterns": len(self.global_patterns),
                "last_distill_entry_count": self.last_distill_entry_count,
                "min_consecutive_failures": self.min_consecutive_failures,
            },
        }

        json_path = p.with_suffix(".json")
        json_content = _json.dumps(snapshot, indent=2, ensure_ascii=False)
        fd2, tmp_json = tempfile.mkstemp(dir=p.parent, suffix=".json.tmp")
        try:
            with os.fdopen(fd2, "w", encoding="utf-8") as fh:
                fh.write(json_content)
            os.replace(tmp_json, json_path)
        except BaseException:
            try:
                os.unlink(tmp_json)
            except OSError:
                pass
            raise


    @classmethod
    def load(cls, path: "str | Path") -> "ExperienceBank":
        """Load a previously saved bank from *path*."""
        with open(path, "rb") as fh:
            bank = pickle.load(fh)
        if not isinstance(bank, cls):
            raise TypeError(f"Expected ExperienceBank, got {type(bank)}")
        return bank

    def latest_entry_for_case(self, case_id: str) -> Optional[PerCaseEntry]:
        entries = self.case_index.get(case_id)
        return entries[-1] if entries else None

    def _matches_request(self, entry: PerCaseEntry, request: RetrievalRequest) -> bool:
        stats = self.case_stats.get(entry.case_id)
        if request.case_ids and entry.case_id not in request.case_ids:
            return False
        if request.primary_dim and entry.diagnosis.diagnostic_signal.primary_dim != request.primary_dim:
            return False
        if request.iteration_range:
            start, end = request.iteration_range
            if not (start <= entry.iteration <= end):
                return False
        if request.min_consecutive_failures is not None:
            if not stats or stats.consecutive_failures < request.min_consecutive_failures:
                return False
        if request.reward_trend is not None:
            if not stats or stats.reward_trend != request.reward_trend:
                return False
        return all(self._feature_filter_matches(entry, feature_filter) for feature_filter in request.feature_filters)

    def _feature_filter_matches(self, entry: PerCaseEntry, feature_filter: FeatureFilter) -> bool:
        actual = getattr(entry.case_features, feature_filter.field)
        expected = feature_filter.value
        operator = feature_filter.operator
        if operator == "eq":
            return actual == expected
        if operator == "gt":
            return actual > expected
        if operator == "gte":
            return actual >= expected
        if operator == "lt":
            return actual < expected
        if operator == "lte":
            return actual <= expected
        if operator == "in":
            return actual in expected
        return False

    def _sample_by_cluster(self, entries: Iterable[PerCaseEntry], k: int) -> list[PerCaseEntry]:
        by_cluster: dict[str, list[PerCaseEntry]] = defaultdict(list)
        for entry in entries:
            cluster_id = entry.case_features.cluster_id or entry.case_features.domain
            by_cluster[cluster_id].append(entry)

        sampled: list[PerCaseEntry] = []
        for cluster_entries in by_cluster.values():
            sampled.extend(random.sample(cluster_entries, min(k, len(cluster_entries))))
        return sampled

    def _build_pattern_description(self, primary_dim: str, entries: list[PerCaseEntry]) -> str:
        common_domains = Counter(entry.case_features.domain for entry in entries).most_common(2)
        domain_summary = ", ".join(domain for domain, _ in common_domains) or "mixed domains"

        config_summary = self._summarize_dim_config(primary_dim, entries)

        parts = [
            f"Repeated failures anchored on {primary_dim} across {len(set(e.case_id for e in entries))} cases "
            f"in {domain_summary}.",
        ]
        if config_summary:
            parts.append(f"Common config state: {config_summary}.")
        parts.append("The controller should revisit this dimension before tuning others.")
        return " ".join(parts)

    def _build_pattern_effect(self, primary_dim: str, entries: list[PerCaseEntry]) -> str:
        avg_reward = sum(entry.primary_reward for entry in entries) / max(len(entries), 1)
        case_count = len(set(entry.case_id for entry in entries))
        iteration_span = max(entry.iteration for entry in entries) - min(entry.iteration for entry in entries)

        config_hint = self._suggest_dim_action(primary_dim, entries)

        parts = [
            f"Average reward across {case_count} failing cases over {iteration_span + 1} iterations "
            f"is {avg_reward:.2f}.",
        ]
        if config_hint:
            parts.append(config_hint)
        return " ".join(parts)

    def _summarize_dim_config(self, dim: str, entries: list[PerCaseEntry]) -> str:
        """Extract the most common config values for the failing dimension."""
        dim_configs = [getattr(entry.config, dim) for entry in entries]
        if not dim_configs:
            return ""

        all_keys = set()
        for cfg in dim_configs:
            all_keys.update(cfg.keys())

        summaries = []
        for key in sorted(all_keys):
            values = Counter(str(cfg.get(key, "")) for cfg in dim_configs)
            most_common_val, count = values.most_common(1)[0]
            if count >= len(dim_configs) * 0.5:
                summaries.append(f"{key}={most_common_val}")

        return ", ".join(summaries[:4])

    def _suggest_dim_action(self, dim: str, entries: list[PerCaseEntry]) -> str:
        """Produce a concrete suggestion based on the dimension and config state."""
        dim_configs = [getattr(entry.config, dim) for entry in entries]
        if not dim_configs:
            return ""

        suggestions = {
            "D1": self._suggest_d1_action,
            "D2": self._suggest_d2_action,
            "D3": self._suggest_d3_action,
            "D4": self._suggest_d4_action,
            "D5": self._suggest_d5_action,
            "D6": self._suggest_d6_action,
        }
        suggest_fn = suggestions.get(dim)
        return suggest_fn(dim_configs) if suggest_fn else ""

    def _suggest_d1_action(self, configs: list[dict]) -> str:
        if all(not cfg.get("structured_instruction") for cfg in configs):
            return "Consider enabling structured_instruction to improve prompt clarity."
        return ""

    def _suggest_d2_action(self, configs: list[dict]) -> str:
        if all(cfg.get("tool_access") != "enabled" for cfg in configs):
            return "Retrieval/tool access is disabled in all failing runs; consider enabling it."
        if all(cfg.get("tool_access") == "enabled" for cfg in configs):
            return "Tool access is enabled but failures persist; consider changing retrieval_mode or increasing top_k."
        return ""

    def _suggest_d3_action(self, configs: list[dict]) -> str:
        avg_tokens = sum(cfg.get("max_tokens", 256) for cfg in configs) / len(configs)
        if avg_tokens < 512:
            return f"Average max_tokens is {avg_tokens:.0f}; consider increasing the generation budget."
        return ""

    def _suggest_d4_action(self, configs: list[dict]) -> str:
        if all(cfg.get("workflow") == "single_call" for cfg in configs):
            return "All failures use single_call workflow; consider plan_execute_refine for complex cases."
        return ""

    def _suggest_d5_action(self, configs: list[dict]) -> str:
        if all(cfg.get("memory_policy") == "disabled" for cfg in configs):
            return "Memory is disabled; enabling summary_buffer may help multi-turn reasoning."
        return ""

    def _suggest_d6_action(self, configs: list[dict]) -> str:
        if all(cfg.get("validator") == "disabled" for cfg in configs):
            return "Validation is disabled; enabling schema_then_retry may improve output quality."
        return ""

    def _embed_text(self, text: str):
        """Embed text using the best available method.

        Priority: custom embed_fn > OpenAI embedder.
        Raises if no embedder is configured.
        """
        if self._custom_embed_fn is not None:
            return self._custom_embed_fn(text)
        if self._openai_embedder is not None:
            return self._openai_embedder.embed(text)
        raise RuntimeError(
            "ExperienceBank requires an embedder for similarity retrieval. "
            "Pass openai_client= or embed_fn=."
        )

    def _cosine_similarity(self, left, right) -> float:
        """Compute similarity using the best available method.

        Priority: custom similarity_fn > OpenAI cosine.
        Raises if no similarity function is configured.
        """
        if self._custom_similarity_fn is not None:
            return self._custom_similarity_fn(left, right)
        if self._openai_embedder is not None:
            return self._openai_embedder.cosine_similarity(left, right)
        raise RuntimeError(
            "ExperienceBank requires a similarity function. "
            "Pass openai_client= or similarity_fn=."
        )

    def _case_similarity(self, case: BenchmarkCase, entry: PerCaseEntry) -> float:
        features = case.features
        other = entry.case_features
        score = 0.0

        if features.domain == other.domain:
            score += 3.0
        if features.requires_external_knowledge == other.requires_external_knowledge:
            score += 2.5
        if (features.cluster_id or features.domain) == (other.cluster_id or other.domain):
            score += 1.0

        score += max(0.0, 1.0 - abs(features.complexity_estimate - other.complexity_estimate))
        score += max(0.0, 1.0 - abs(features.ambiguity_score - other.ambiguity_score))
        score += max(0.0, 1.0 - abs(features.safety_sensitivity - other.safety_sensitivity))

        instruction = features.instruction or case.prompt
        other_instruction = other.instruction or entry.case_id
        score += self._cosine_similarity(
            self._embed_text(instruction),
            self._embed_text(other_instruction),
        )
        return score
