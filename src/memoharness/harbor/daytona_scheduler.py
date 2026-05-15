from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

DAYTONA_ASSIGNMENT_STRATEGIES = frozenset(
    {"current", "strict_fill", "strict_balance"}
)


@dataclass(frozen=True)
class PlannedDaytonaShard:
    task_names: list[str]
    daytona_key: str


@dataclass(frozen=True)
class DaytonaRelocationRequest:
    task_name: str
    failed_key: str
    error_kind: str
    allowed_target_loads: tuple[int, ...]


def _chunk_case_ids(case_ids: list[str], chunk_size: int) -> list[list[str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [
        case_ids[index:index + chunk_size]
        for index in range(0, len(case_ids), chunk_size)
    ]


def _build_current_daytona_case_batches(
    case_ids: list[str],
    chunk_size: int,
    exclusive_task_ids: set[str],
) -> list[list[str]]:
    shard_batches = _chunk_case_ids(case_ids, chunk_size)
    if not exclusive_task_ids:
        return shard_batches

    expanded_batches: list[list[str]] = []
    for batch in shard_batches:
        exclusive_batch_ids = [case_id for case_id in batch if case_id in exclusive_task_ids]
        if not exclusive_batch_ids:
            expanded_batches.append(batch)
            continue
        for case_id in exclusive_batch_ids:
            expanded_batches.append([case_id])
        remaining = [case_id for case_id in batch if case_id not in exclusive_task_ids]
        if remaining:
            expanded_batches.append(remaining)
    return expanded_batches


def _partition_case_ids(case_ids: list[str], group_count: int) -> list[list[str]]:
    if group_count <= 0:
        raise ValueError("group_count must be positive")
    if not case_ids:
        return []

    group_count = min(group_count, len(case_ids))
    base_size, remainder = divmod(len(case_ids), group_count)
    partitions: list[list[str]] = []
    start = 0
    for index in range(group_count):
        size = base_size + (1 if index < remainder else 0)
        partitions.append(case_ids[start:start + size])
        start += size
    return partitions


def _reserved_exclusive_key_count(
    *,
    available_key_count: int,
    exclusive_shard_count: int,
    has_shared_work: bool,
) -> int:
    if exclusive_shard_count <= 0 or available_key_count <= 0:
        return 0
    if not has_shared_work:
        return min(exclusive_shard_count, available_key_count)
    return min(exclusive_shard_count, max(available_key_count - 1, 0))


def _assign_round_robin_shards(
    shard_batches: list[list[str]],
    available_keys: list[str],
    *,
    exclusive_task_ids: set[str],
) -> list[PlannedDaytonaShard]:
    if not shard_batches or not available_keys:
        return []

    exclusive_shard_count = sum(
        1
        for batch in shard_batches
        if len(batch) == 1 and batch[0] in exclusive_task_ids
    )
    reserved_exclusive_key_count = _reserved_exclusive_key_count(
        available_key_count=len(available_keys),
        exclusive_shard_count=exclusive_shard_count,
        has_shared_work=exclusive_shard_count < len(shard_batches),
    )
    shared_keys = available_keys[: len(available_keys) - reserved_exclusive_key_count]
    exclusive_keys = available_keys[len(available_keys) - reserved_exclusive_key_count :]
    exclusive_key_index = 0
    shared_key_index = 0

    assignments: list[PlannedDaytonaShard] = []
    for shard_task_batch in shard_batches:
        is_exclusive_shard = (
            len(shard_task_batch) == 1 and shard_task_batch[0] in exclusive_task_ids
        )
        if is_exclusive_shard and exclusive_key_index < len(exclusive_keys):
            daytona_key = exclusive_keys[exclusive_key_index]
            exclusive_key_index += 1
        else:
            key_pool = shared_keys or available_keys
            daytona_key = key_pool[shared_key_index % len(key_pool)]
            shared_key_index += 1
        assignments.append(
            PlannedDaytonaShard(
                task_names=shard_task_batch,
                daytona_key=daytona_key,
            )
        )
    return assignments


def _build_strict_fill_batches(
    case_ids: list[str],
    chunk_size: int,
    exclusive_task_ids: set[str],
) -> list[list[str]]:
    shared_tasks = [case_id for case_id in case_ids if case_id not in exclusive_task_ids]
    exclusive_tasks = [case_id for case_id in case_ids if case_id in exclusive_task_ids]
    return _chunk_case_ids(shared_tasks, chunk_size) + [[case_id] for case_id in exclusive_tasks]


def _build_strict_balance_shared_assignments(
    shared_tasks: list[str],
    shared_keys: list[str],
    *,
    chunk_size: int,
) -> list[PlannedDaytonaShard]:
    if not shared_tasks or not shared_keys:
        return []

    key_bins = _partition_case_ids(shared_tasks, len(shared_keys))
    assignments: list[PlannedDaytonaShard] = []
    for daytona_key, key_batch in zip(shared_keys, key_bins):
        for chunk in _chunk_case_ids(key_batch, chunk_size):
            assignments.append(
                PlannedDaytonaShard(
                    task_names=chunk,
                    daytona_key=daytona_key,
                )
            )
    return assignments


def build_daytona_shard_plan(
    case_ids: list[str],
    *,
    available_keys: list[str],
    max_tasks_per_shard: int,
    exclusive_task_ids: set[str],
    assignment_strategy: str,
) -> list[PlannedDaytonaShard]:
    strategy = str(assignment_strategy or "current").strip().lower()
    if strategy not in DAYTONA_ASSIGNMENT_STRATEGIES:
        raise ValueError(
            "assignment_strategy must be one of: current, strict_fill, strict_balance"
        )
    if not case_ids or not available_keys:
        return []

    if strategy == "current":
        shard_batches = _build_current_daytona_case_batches(
            case_ids,
            max_tasks_per_shard,
            exclusive_task_ids,
        )
        return _assign_round_robin_shards(
            shard_batches,
            available_keys,
            exclusive_task_ids=exclusive_task_ids,
        )

    if strategy == "strict_fill":
        shard_batches = _build_strict_fill_batches(
            case_ids,
            max_tasks_per_shard,
            exclusive_task_ids,
        )
        return _assign_round_robin_shards(
            shard_batches,
            available_keys,
            exclusive_task_ids=exclusive_task_ids,
        )

    shared_tasks = [case_id for case_id in case_ids if case_id not in exclusive_task_ids]
    exclusive_tasks = [case_id for case_id in case_ids if case_id in exclusive_task_ids]
    reserved_exclusive_key_count = _reserved_exclusive_key_count(
        available_key_count=len(available_keys),
        exclusive_shard_count=len(exclusive_tasks),
        has_shared_work=bool(shared_tasks),
    )
    shared_keys = available_keys[: len(available_keys) - reserved_exclusive_key_count]
    exclusive_keys = available_keys[len(available_keys) - reserved_exclusive_key_count :]
    if shared_tasks and not shared_keys:
        shared_keys = list(available_keys)

    assignments = _build_strict_balance_shared_assignments(
        shared_tasks,
        shared_keys,
        chunk_size=max_tasks_per_shard,
    )
    key_loads = {key: 0 for key in available_keys}
    for assignment in assignments:
        key_loads[assignment.daytona_key] = key_loads.get(assignment.daytona_key, 0) + len(
            assignment.task_names
        )

    for index, case_id in enumerate(exclusive_tasks):
        if index < len(exclusive_keys):
            daytona_key = exclusive_keys[index]
        else:
            key_pool = shared_keys or available_keys
            daytona_key = min(
                key_pool,
                key=lambda key: (key_loads.get(key, 0), available_keys.index(key)),
            )
        assignments.append(
            PlannedDaytonaShard(
                task_names=[case_id],
                daytona_key=daytona_key,
            )
        )
        key_loads[daytona_key] = key_loads.get(daytona_key, 0) + 1

    return assignments


def choose_relocation_target_key(
    request: DaytonaRelocationRequest,
    *,
    candidates: list[tuple[str, int]],
    key_priority: dict[str, int],
) -> Optional[str]:
    load_priority = {
        load: index for index, load in enumerate(request.allowed_target_loads)
    }
    matching = [
        (daytona_key, current_load)
        for daytona_key, current_load in candidates
        if current_load in load_priority and daytona_key != request.failed_key
    ]
    if not matching:
        return None
    matching.sort(
        key=lambda item: (
            load_priority[item[1]],
            key_priority.get(item[0], 10**9),
        )
    )
    return matching[0][0]
