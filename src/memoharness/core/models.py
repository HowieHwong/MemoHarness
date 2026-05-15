from __future__ import annotations

import copy
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Literal, Optional


DimensionName = Literal["D1", "D2", "D3", "D4", "D5", "D6"]
RewardTrend = Literal["improving", "stable", "degrading"]
PrimaryMetric = Literal["accuracy", "harmlessness", "pass_rate"]
MemoryPolicy = Literal[
    "disabled",            # no cross-call memory
    "sliding_window",      # keep last N messages verbatim
    "summary_buffer",      # keep a compressed summary of history
    "importance_based",    # drop messages scored below a threshold
]

DIMENSIONS: tuple[DimensionName, ...] = ("D1", "D2", "D3", "D4", "D5", "D6")


@dataclass
class CaseFeatures:
    input_length: int
    complexity_estimate: float
    domain: str
    requires_external_knowledge: bool
    safety_sensitivity: float
    ambiguity_score: float
    instruction: str = ""
    cluster_id: Optional[str] = None


@dataclass
class BenchmarkCase:
    case_id: str
    prompt: str
    expected_output: str
    features: CaseFeatures
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HarnessConfig:
    D1: dict[str, Any] = field(default_factory=dict)
    D2: dict[str, Any] = field(default_factory=dict)
    D3: dict[str, Any] = field(default_factory=dict)
    D4: dict[str, Any] = field(default_factory=dict)
    D5: dict[str, Any] = field(default_factory=dict)
    D6: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "HarnessConfig":
        return HarnessConfig(
            D1=copy.deepcopy(self.D1),
            D2=copy.deepcopy(self.D2),
            D3=copy.deepcopy(self.D3),
            D4=copy.deepcopy(self.D4),
            D5=copy.deepcopy(self.D5),
            D6=copy.deepcopy(self.D6),
        )

    def update_dimension(self, dim: DimensionName, updates: dict[str, Any]) -> None:
        current = getattr(self, dim)
        current.update(updates)

    def as_dict(self) -> dict[str, dict[str, Any]]:
        return {dim: dict(getattr(self, dim)) for dim in DIMENSIONS}

    def delta_from(self, previous: "HarnessConfig") -> dict[DimensionName, dict[str, Any]]:
        delta: dict[DimensionName, dict[str, Any]] = {}
        for dim in DIMENSIONS:
            current = getattr(self, dim)
            prior = getattr(previous, dim)
            dim_delta = {
                key: value
                for key, value in current.items()
                if key not in prior or prior[key] != value
            }
            removed_keys = [key for key in prior if key not in current]
            if removed_keys:
                dim_delta["_removed"] = removed_keys
            if dim_delta:
                delta[dim] = dim_delta
        return delta


@dataclass
class Trajectory:
    num_llm_calls: int
    total_tokens: int
    latency_ms: int
    tools_invoked: list[str]
    intermediate_outputs: list[str]
    final_output: str


@dataclass
class ExecutionResult:
    trajectory: Trajectory
    final_output: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiagnosticSignal:
    primary_dim: DimensionName
    secondary_dims: list[DimensionName] = field(default_factory=list)


@dataclass
class Diagnosis:
    success: bool
    analysis: str
    diagnostic_signal: DiagnosticSignal


@dataclass
class PerCaseEntry:
    case_id: str
    iteration: int
    case_features: CaseFeatures
    config: HarnessConfig
    delta_from_prev: dict[DimensionName, dict[str, Any]]
    trajectory: Trajectory
    primary_reward: float
    cost_actual: float
    diagnosis: Diagnosis


@dataclass
class CaseHistoryRecord:
    iteration: int
    success: bool
    primary_reward: float


@dataclass
class CaseStats:
    history: list[CaseHistoryRecord] = field(default_factory=list)
    consecutive_failures: int = 0
    avg_reward_recent_N: float = 0.0
    reward_trend: RewardTrend = "stable"
    dim_failure_counts: dict[DimensionName, int] = field(
        default_factory=lambda: {dim: 0 for dim in DIMENSIONS}
    )

    def update(self, entry: PerCaseEntry, recent_window: int = 5) -> None:
        success = entry.diagnosis.success
        self.history.append(
            CaseHistoryRecord(
                iteration=entry.iteration,
                success=success,
                primary_reward=entry.primary_reward,
            )
        )
        self.consecutive_failures = 0 if success else self.consecutive_failures + 1
        if not success:
            primary_dim = entry.diagnosis.diagnostic_signal.primary_dim
            self.dim_failure_counts[primary_dim] += 1
            for dim in entry.diagnosis.diagnostic_signal.secondary_dims:
                self.dim_failure_counts[dim] += 1

        recent = self.history[-recent_window:]
        self.avg_reward_recent_N = mean(record.primary_reward for record in recent)
        if len(recent) < 2:
            self.reward_trend = "stable"
            return

        deltas = [
            recent[index].primary_reward - recent[index - 1].primary_reward
            for index in range(1, len(recent))
        ]
        avg_delta = mean(deltas)
        if avg_delta > 0.05:
            self.reward_trend = "improving"
        elif avg_delta < -0.05:
            self.reward_trend = "degrading"
        else:
            self.reward_trend = "stable"


@dataclass
class GlobalPattern:
    pattern_id: str
    iteration_range: tuple[int, int]
    description: str
    evidence: list[str]
    effect: str
    primary_dim: DimensionName = "D4"  # 主要锚定的失败维度


@dataclass
class FeatureFilter:
    field: str
    operator: Literal["eq", "gt", "gte", "lt", "lte", "in"]
    value: Any


@dataclass
class RetrievalRequest:
    feature_filters: list[FeatureFilter] = field(default_factory=list)
    min_consecutive_failures: Optional[int] = None
    reward_trend: Optional[RewardTrend] = None
    primary_dim: Optional[DimensionName] = None
    iteration_range: Optional[tuple[int, int]] = None
    case_ids: Optional[list[str]] = None
    sample_k: Optional[int] = None
    sample_by_cluster_k: Optional[int] = None
    include_global_patterns: bool = True
    max_entries: Optional[int] = None
    max_global_patterns: Optional[int] = None


@dataclass
class RetrievedSlice:
    global_patterns: list[GlobalPattern]
    entries: list[PerCaseEntry]
    case_stats: dict[str, CaseStats]


@dataclass
class RewardConfig:
    primary_metric: PrimaryMetric = "accuracy"


@dataclass
class CostConstraint:
    max_tokens_per_case: int
    max_latency_ms: int
    max_llm_calls_per_case: int

    def allows(self, trajectory: Trajectory) -> bool:
        return (
            trajectory.total_tokens <= self.max_tokens_per_case
            and trajectory.latency_ms <= self.max_latency_ms
            and trajectory.num_llm_calls <= self.max_llm_calls_per_case
        )


@dataclass
class ControllerDecision:
    config: HarnessConfig
    rationale: str
    variants: list[HarnessConfig] = field(default_factory=list)


@dataclass
class CandidateRecord:
    iteration: int
    config: HarnessConfig
    primary_reward: float
    cost_actual: float
    constraint_satisfied: bool
    rationale: str


@dataclass
class TrainingReport:
    best_candidate: Optional[CandidateRecord]
    candidates: list[CandidateRecord]
    experience_bank_size: int
    global_patterns: list[GlobalPattern]


@dataclass
class EvaluationCaseResult:
    case_id: str
    config: HarnessConfig
    primary_reward: float
    trajectory: Trajectory
    diagnosis: Diagnosis


@dataclass
class EvaluationReport:
    base_config: HarnessConfig
    results: list[EvaluationCaseResult]


def make_minimal_config() -> HarnessConfig:
    return HarnessConfig(
        D1={
            "examples": False,
            "structured_instruction": False,
            "compression": "none",
        },
        D2={
            "tool_access": "disabled",
            "tool_protocol": "native",
            "retrieval_mode": "none",
            "top_k": 0,
        },
        D3={
            "temperature": 0.0,
            "max_tokens": 256,
            "top_p": 1.0,
            "candidate_count": 1,
        },
        D4={
            "workflow": "single_call",
            "stop_rule": "immediate",
        },
        D5={
            "memory_policy": "disabled",
            "history_keep": 0,
        },
        D6={
            "postprocess": "raw_passthrough",
            "validator": "disabled",
            "fallback": "none",
        },
    )
