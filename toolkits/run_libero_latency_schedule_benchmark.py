"""Benchmark latency-aware LIBERO task scheduling with real env.step calls."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import queue
import random
import time
import traceback
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_TASK_COLUMNS = {"task_id", "task_name", "mean_latency_ms", "njnt", "ngeom"}


@dataclass(frozen=True)
class TaskRecord:
    task_id: int
    task_name: str
    mean_latency_ms: float
    njnt: int
    ngeom: int
    estimated_latency_score: float = 0.0
    extra: dict[str, str] = field(default_factory=dict)


def _parse_int(row: dict[str, str], key: str) -> int:
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer column {key!r}: {row.get(key)!r}") from exc


def _parse_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid float column {key!r}: {row.get(key)!r}") from exc


def load_task_records(path: str | Path) -> list[TaskRecord]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_TASK_COLUMNS - fieldnames)
        if missing:
            raise ValueError(f"missing required columns: {', '.join(missing)}")
        records = []
        for row in reader:
            extra = {
                key: value
                for key, value in row.items()
                if key not in REQUIRED_TASK_COLUMNS and value is not None
            }
            records.append(
                TaskRecord(
                    task_id=_parse_int(row, "task_id"),
                    task_name=row["task_name"],
                    mean_latency_ms=_parse_float(row, "mean_latency_ms"),
                    njnt=_parse_int(row, "njnt"),
                    ngeom=_parse_int(row, "ngeom"),
                    extra=extra,
                )
            )
    return records


def sample_task_records(
    records: list[TaskRecord],
    *,
    num_envs: int,
    seed: int,
) -> list[TaskRecord]:
    if num_envs < 1:
        raise ValueError("num_envs must be >= 1")
    if num_envs > len(records):
        raise ValueError(
            f"num_envs={num_envs} exceeds available task records={len(records)}"
        )
    rng = random.Random(seed)
    return rng.sample(records, num_envs)


def _z_scores(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    std = float(np.std(array))
    if std == 0.0:
        return [0.0 for _ in values]
    mean = float(np.mean(array))
    return [float((value - mean) / std) for value in array]


def estimate_latency_scores(
    records: list[TaskRecord],
    *,
    weight_jnt: float = 0.45,
    weight_geom: float = 0.55,
) -> list[TaskRecord]:
    jnt_scores = _z_scores([float(record.njnt) for record in records])
    geom_scores = _z_scores([float(record.ngeom) for record in records])
    scored = []
    for record, jnt_score, geom_score in zip(records, jnt_scores, geom_scores):
        score = weight_jnt * jnt_score + weight_geom * geom_score
        scored.append(replace(record, estimated_latency_score=float(score)))
    return scored


TASK_ID_BASELINE = "task_id_baseline"
RANDOM_BASELINE = "random_baseline"
RLINF_DEFAULT_UNBOUND_CHUNK = "rlinf_default_unbound_chunk"
RLINF_DEFAULT_BOUND_CHUNK = "rlinf_default_bound_chunk"
RLINF_OPTIMIZED_BOUND_CHUNK = "rlinf_optimized_bound_chunk"
TRAPEZOID_PIPELINE = "trapezoid_pipeline"
PHASE_SHIFTED_TRAPEZOID = "phase_shifted_trapezoid"
HISTORICAL_OPTIMAL = "historical_optimal"
SAME_CORE_LATENCY_ORDER = "same_core_latency_order"
ODD_EVEN_BINPACK = "odd_even_binpack"
WARMUP_ODD_EVEN_BINPACK = "warmup_odd_even_binpack"
MINMAX_BINPACK = "minmax_binpack"
WARMUP_MINMAX_BINPACK = "warmup_minmax_binpack"
EXACT_BINPACK_MAX_TASKS = 18


@dataclass(frozen=True)
class ScheduleItem:
    schedule_name: str
    task: TaskRecord
    core_index: int
    cpu_id: int
    layer_index: int
    order_index: int
    side: str = "baseline"


@dataclass(frozen=True)
class StepEvent:
    schedule_name: str
    round_index: int
    core_index: int
    cpu_id: int
    task_id: int
    task_name: str
    task_step_index: int
    latency_s: float
    round_wall_time_s: float
    idle_time_s: float
    cpu_affinity_applied: bool
    start_time_s: float | None = None
    end_time_s: float | None = None


@dataclass(frozen=True)
class ProcessRunResult:
    events: list[StepEvent]
    errors: list[dict[str, Any]]


@dataclass(frozen=True)
class WarmupLatency:
    task_id: int
    task_name: str
    mean_latency_ms: float
    samples: int


def _require_cpu_ids(cpu_ids: list[int]) -> None:
    if not cpu_ids:
        raise ValueError("cpu_ids must not be empty")


def _layered_plan(
    records: list[TaskRecord],
    *,
    cpu_ids: list[int],
    schedule_name: str,
    side: str = "baseline",
    order_offset: int = 0,
) -> list[ScheduleItem]:
    _require_cpu_ids(cpu_ids)
    items = []
    for index, record in enumerate(records):
        core_index = index % len(cpu_ids)
        layer_index = index // len(cpu_ids)
        items.append(
            ScheduleItem(
                schedule_name=schedule_name,
                task=record,
                core_index=core_index,
                cpu_id=cpu_ids[core_index],
                layer_index=layer_index,
                order_index=order_offset + index,
                side=side,
            )
        )
    return items


def build_task_id_baseline_plan(
    records: list[TaskRecord],
    *,
    cpu_ids: list[int],
) -> list[ScheduleItem]:
    ordered = sorted(records, key=lambda record: record.task_id)
    return _layered_plan(
        ordered,
        cpu_ids=cpu_ids,
        schedule_name=TASK_ID_BASELINE,
    )


def build_random_baseline_plan(
    records: list[TaskRecord],
    *,
    cpu_ids: list[int],
    seed: int,
) -> list[ScheduleItem]:
    ordered = list(records)
    random.Random(seed).shuffle(ordered)
    return _layered_plan(
        ordered,
        cpu_ids=cpu_ids,
        schedule_name=RANDOM_BASELINE,
    )


def build_historical_optimal_plan(
    records: list[TaskRecord],
    *,
    cpu_ids: list[int],
) -> list[ScheduleItem]:
    ordered = sorted(records, key=lambda record: (-record.mean_latency_ms, record.task_id))
    return _layered_plan(
        ordered,
        cpu_ids=cpu_ids,
        schedule_name=HISTORICAL_OPTIMAL,
    )


def build_same_core_latency_order_plan(
    records: list[TaskRecord],
    *,
    cpu_ids: list[int],
) -> list[ScheduleItem]:
    baseline_plan = build_task_id_baseline_plan(records, cpu_ids=cpu_ids)
    grouped = _items_by_core(baseline_plan)
    ordered_by_core = {
        core_index: sorted(
            core_items,
            key=lambda item: (-item.task.estimated_latency_score, item.task.task_id),
        )
        for core_index, core_items in grouped.items()
    }
    items = []
    order_index = 0
    for layer_index in range(max(len(core_items) for core_items in ordered_by_core.values())):
        for core_index, core_items in sorted(ordered_by_core.items()):
            if layer_index >= len(core_items):
                continue
            item = core_items[layer_index]
            items.append(
                replace(
                    item,
                    schedule_name=SAME_CORE_LATENCY_ORDER,
                    layer_index=layer_index,
                    order_index=order_index,
                )
            )
            order_index += 1
    return items


def _binpack_weight(record: TaskRecord, *, min_score: float) -> float:
    del min_score
    return max(record.mean_latency_ms, 0.0)


def _first_fit_decreasing_under_capacity(
    records: list[TaskRecord],
    *,
    cpu_ids: list[int],
    capacity: float,
) -> list[list[TaskRecord]] | None:
    bins: list[list[TaskRecord]] = [[] for _ in cpu_ids]
    loads = [0.0 for _ in cpu_ids]
    for record in records:
        weight = _binpack_weight(record, min_score=0.0)
        placed = False
        for core_index in sorted(range(len(cpu_ids)), key=lambda index: (loads[index], index)):
            if loads[core_index] + weight <= capacity + 1e-9:
                bins[core_index].append(record)
                loads[core_index] += weight
                placed = True
                break
        if not placed:
            return None
    return bins


def _assignment_loads(assignments: list[list[TaskRecord]]) -> list[float]:
    return [
        sum(_binpack_weight(record, min_score=0.0) for record in core_records)
        for core_records in assignments
    ]


def _canonicalize_assignment(
    assignments: list[list[TaskRecord]],
) -> list[list[TaskRecord]]:
    return [
        sorted(
            core_records,
            key=lambda record: (-_binpack_weight(record, min_score=0.0), record.task_id),
        )
        for core_records in assignments
    ]


def _assignment_key(
    assignments: list[list[TaskRecord]],
) -> tuple[float, tuple[float, ...], tuple[int, ...], tuple[tuple[int, ...], ...]]:
    loads = _assignment_loads(assignments)
    canonical = _canonicalize_assignment(assignments)
    layout = tuple(tuple(record.task_id for record in core_records) for core_records in canonical)
    return (
        max(loads, default=0.0),
        tuple(sorted(loads, reverse=True)),
        tuple(sorted((len(core_records) for core_records in assignments), reverse=True)),
        layout,
    )


def _lpt_min_max_assignments(
    records: list[TaskRecord],
    *,
    cpu_ids: list[int],
) -> list[list[TaskRecord]]:
    assignments: list[list[TaskRecord]] = [[] for _ in cpu_ids]
    loads = [0.0 for _ in cpu_ids]
    for record in records:
        core_index = min(range(len(cpu_ids)), key=lambda index: (loads[index], index))
        assignments[core_index].append(record)
        loads[core_index] += _binpack_weight(record, min_score=0.0)
    return assignments


def _paired_two_per_core_assignments(
    records: list[TaskRecord],
    *,
    cpu_ids: list[int],
) -> list[list[TaskRecord]] | None:
    if len(records) != 2 * len(cpu_ids):
        return None
    assignments: list[list[TaskRecord]] = [[] for _ in cpu_ids]
    for core_index in range(len(cpu_ids)):
        assignments[core_index].append(records[core_index])
        assignments[core_index].append(records[-core_index - 1])
    return assignments


def _improve_min_max_assignment(
    assignments: list[list[TaskRecord]],
) -> list[list[TaskRecord]]:
    current = _canonicalize_assignment(assignments)
    while True:
        current_key = _assignment_key(current)
        best_candidate: list[list[TaskRecord]] | None = None
        best_key = current_key
        loads = _assignment_loads(current)
        core_order = sorted(range(len(current)), key=lambda index: (-loads[index], index))

        for source_index in core_order:
            if not current[source_index]:
                continue
            destination_order = sorted(
                (index for index in range(len(current)) if index != source_index),
                key=lambda index: (loads[index], index),
            )
            for record in current[source_index]:
                for destination_index in destination_order:
                    candidate = [list(core_records) for core_records in current]
                    candidate[source_index].remove(record)
                    candidate[destination_index].append(record)
                    candidate = _canonicalize_assignment(candidate)
                    candidate_key = _assignment_key(candidate)
                    if candidate_key < best_key:
                        best_candidate = candidate
                        best_key = candidate_key

        for source_index in core_order:
            if not current[source_index]:
                continue
            for destination_index in range(len(current)):
                if source_index == destination_index or not current[destination_index]:
                    continue
                for source_record in current[source_index]:
                    for destination_record in current[destination_index]:
                        candidate = [list(core_records) for core_records in current]
                        candidate[source_index].remove(source_record)
                        candidate[destination_index].remove(destination_record)
                        candidate[source_index].append(destination_record)
                        candidate[destination_index].append(source_record)
                        candidate = _canonicalize_assignment(candidate)
                        candidate_key = _assignment_key(candidate)
                        if candidate_key < best_key:
                            best_candidate = candidate
                            best_key = candidate_key

        if best_candidate is None:
            return current
        current = best_candidate


def _minimize_max_load_binpack_assignments(
    records: list[TaskRecord],
    *,
    cpu_ids: list[int],
) -> list[list[TaskRecord]]:
    ordered = sorted(
        records,
        key=lambda record: (-_binpack_weight(record, min_score=0.0), record.task_id),
    )
    candidates = [_lpt_min_max_assignments(ordered, cpu_ids=cpu_ids)]
    paired = _paired_two_per_core_assignments(ordered, cpu_ids=cpu_ids)
    if paired is not None:
        candidates.append(paired)
    total_weight = sum(_binpack_weight(record, min_score=0.0) for record in ordered)
    lower = max(
        max((_binpack_weight(record, min_score=0.0) for record in ordered), default=0.0),
        total_weight / len(cpu_ids),
    )
    upper = total_weight
    best = _first_fit_decreasing_under_capacity(
        ordered,
        cpu_ids=cpu_ids,
        capacity=upper,
    )
    for _ in range(48):
        mid = (lower + upper) / 2.0
        candidate = _first_fit_decreasing_under_capacity(
            ordered,
            cpu_ids=cpu_ids,
            capacity=mid,
        )
        if candidate is None:
            lower = mid
        else:
            upper = mid
            best = candidate
    assert best is not None
    candidates.append(best)
    improved_candidates = [
        _improve_min_max_assignment(candidate)
        for candidate in candidates
    ]
    return min(improved_candidates, key=_assignment_key)


def _optimal_binpack_assignments(
    records: list[TaskRecord],
    *,
    cpu_ids: list[int],
) -> list[list[TaskRecord]]:
    if not records:
        return [[] for _ in cpu_ids]
    ordered = sorted(
        records,
        key=lambda record: (-_binpack_weight(record, min_score=0.0), record.task_id),
    )
    bin_count = len(cpu_ids)
    if len(ordered) > EXACT_BINPACK_MAX_TASKS:
        return _minimize_max_load_binpack_assignments(ordered, cpu_ids=cpu_ids)
    best_bins: list[list[TaskRecord]] | None = None
    best_loads: tuple[float, ...] | None = None
    bins: list[list[TaskRecord]] = [[] for _ in range(bin_count)]
    loads = [0.0 for _ in range(bin_count)]

    def candidate_key() -> tuple[float, tuple[float, ...], tuple[tuple[int, ...], ...]]:
        sorted_loads = tuple(sorted(loads, reverse=True))
        task_layout = tuple(tuple(record.task_id for record in bin_items) for bin_items in bins)
        return max(loads), sorted_loads, task_layout

    def is_better_than_best() -> bool:
        if best_loads is None:
            return True
        current_key = candidate_key()
        best_layout = tuple(
            tuple(record.task_id for record in bin_items)
            for bin_items in (best_bins or [])
        )
        best_key = max(best_loads), tuple(sorted(best_loads, reverse=True)), best_layout
        return current_key < best_key

    def search(record_index: int) -> None:
        nonlocal best_bins, best_loads
        if record_index >= len(ordered):
            if is_better_than_best():
                best_bins = [list(bin_items) for bin_items in bins]
                best_loads = tuple(loads)
            return
        record = ordered[record_index]
        weight = _binpack_weight(record, min_score=0.0)
        seen_loads: set[float] = set()
        for core_index in sorted(range(bin_count), key=lambda index: (loads[index], index)):
            if loads[core_index] in seen_loads:
                continue
            seen_loads.add(loads[core_index])
            if best_loads is not None and loads[core_index] + weight > max(best_loads):
                continue
            bins[core_index].append(record)
            loads[core_index] += weight
            search(record_index + 1)
            loads[core_index] -= weight
            bins[core_index].pop()

    search(0)
    assert best_bins is not None
    return best_bins


def _binpacked_phase_plan(
    records: list[TaskRecord],
    *,
    cpu_ids: list[int],
    schedule_name: str,
    side: str,
    order_offset: int,
) -> list[ScheduleItem]:
    assignments = _optimal_binpack_assignments(records, cpu_ids=cpu_ids)
    items = []
    order_index = order_offset
    for layer_index in range(max((len(core_records) for core_records in assignments), default=0)):
        for core_index, core_records in enumerate(assignments):
            if layer_index >= len(core_records):
                continue
            items.append(
                ScheduleItem(
                    schedule_name=schedule_name,
                    task=core_records[layer_index],
                    core_index=core_index,
                    cpu_id=cpu_ids[core_index],
                    layer_index=layer_index,
                    order_index=order_index,
                    side=side,
                )
            )
            order_index += 1
    return items


def _binpacked_plan_from_records(
    records: list[TaskRecord],
    *,
    cpu_ids: list[int],
    schedule_name: str,
    side: str,
) -> list[ScheduleItem]:
    assignments = _optimal_binpack_assignments(records, cpu_ids=cpu_ids)
    items = []
    order_index = 0
    for layer_index in range(max((len(core_records) for core_records in assignments), default=0)):
        for core_index, core_records in enumerate(assignments):
            if layer_index >= len(core_records):
                continue
            items.append(
                ScheduleItem(
                    schedule_name=schedule_name,
                    task=core_records[layer_index],
                    core_index=core_index,
                    cpu_id=cpu_ids[core_index],
                    layer_index=layer_index,
                    order_index=order_index,
                    side=side,
                )
            )
            order_index += 1
    return items


def build_minmax_binpack_plan(
    records: list[TaskRecord],
    *,
    cpu_ids: list[int],
) -> list[ScheduleItem]:
    _require_cpu_ids(cpu_ids)
    ordered = sorted(
        records,
        key=lambda record: (-record.estimated_latency_score, record.task_id),
    )
    return _binpacked_plan_from_records(
        ordered,
        cpu_ids=cpu_ids,
        schedule_name=MINMAX_BINPACK,
        side="minmax",
    )


def build_warmup_minmax_binpack_plan(
    records: list[TaskRecord],
    *,
    cpu_ids: list[int],
    warmup_latency_ms: dict[int, float],
) -> list[ScheduleItem]:
    _require_cpu_ids(cpu_ids)
    missing_task_ids = [
        record.task_id
        for record in records
        if record.task_id not in warmup_latency_ms
    ]
    if missing_task_ids:
        raise ValueError(f"missing warmup latency for task_id={missing_task_ids[0]}")
    warmup_records = [
        replace(record, mean_latency_ms=float(warmup_latency_ms[record.task_id]))
        for record in records
    ]
    ordered = sorted(
        warmup_records,
        key=lambda record: (-record.mean_latency_ms, record.task_id),
    )
    record_by_task_id = {record.task_id: record for record in records}
    warmup_plan = _binpacked_plan_from_records(
        ordered,
        cpu_ids=cpu_ids,
        schedule_name=WARMUP_MINMAX_BINPACK,
        side="minmax",
    )
    return [
        replace(item, task=record_by_task_id[item.task.task_id])
        for item in warmup_plan
    ]


def build_odd_even_binpack_plan(
    records: list[TaskRecord],
    *,
    cpu_ids: list[int],
) -> list[ScheduleItem]:
    _require_cpu_ids(cpu_ids)
    ordered = sorted(
        records,
        key=lambda record: (-record.estimated_latency_score, record.task_id),
    )
    side_by_task_id = {
        record.task_id: "odd" if index % 2 == 0 else "even"
        for index, record in enumerate(ordered)
    }
    assignments = _optimal_binpack_assignments(ordered, cpu_ids=cpu_ids)
    items = []
    order_index = 0
    for layer_index in range(max((len(core_records) for core_records in assignments), default=0)):
        for core_index, core_records in enumerate(assignments):
            if layer_index >= len(core_records):
                continue
            record = core_records[layer_index]
            items.append(
                ScheduleItem(
                    schedule_name=ODD_EVEN_BINPACK,
                    task=record,
                    core_index=core_index,
                    cpu_id=cpu_ids[core_index],
                    layer_index=layer_index,
                    order_index=order_index,
                    side=side_by_task_id[record.task_id],
                )
            )
            order_index += 1
    return items


def build_warmup_odd_even_binpack_plan(
    records: list[TaskRecord],
    *,
    cpu_ids: list[int],
    warmup_latency_ms: dict[int, float],
) -> list[ScheduleItem]:
    _require_cpu_ids(cpu_ids)
    missing_task_ids = [
        record.task_id
        for record in records
        if record.task_id not in warmup_latency_ms
    ]
    if missing_task_ids:
        raise ValueError(f"missing warmup latency for task_id={missing_task_ids[0]}")
    warmup_records = [
        replace(record, mean_latency_ms=float(warmup_latency_ms[record.task_id]))
        for record in records
    ]
    ordered = sorted(
        warmup_records,
        key=lambda record: (-record.mean_latency_ms, record.task_id),
    )
    side_by_task_id = {
        record.task_id: "odd" if index % 2 == 0 else "even"
        for index, record in enumerate(ordered)
    }
    assignments = _optimal_binpack_assignments(ordered, cpu_ids=cpu_ids)
    record_by_task_id = {record.task_id: record for record in records}
    items = []
    order_index = 0
    for layer_index in range(max((len(core_records) for core_records in assignments), default=0)):
        for core_index, core_records in enumerate(assignments):
            if layer_index >= len(core_records):
                continue
            warmup_record = core_records[layer_index]
            items.append(
                ScheduleItem(
                    schedule_name=WARMUP_ODD_EVEN_BINPACK,
                    task=record_by_task_id[warmup_record.task_id],
                    core_index=core_index,
                    cpu_id=cpu_ids[core_index],
                    layer_index=layer_index,
                    order_index=order_index,
                    side=side_by_task_id[warmup_record.task_id],
                )
            )
            order_index += 1
    return items


def build_trapezoid_pipeline_plan(
    records: list[TaskRecord],
    *,
    cpu_ids: list[int],
) -> list[ScheduleItem]:
    _require_cpu_ids(cpu_ids)
    if len(records) % 2 != 0:
        raise ValueError("trapezoid_pipeline requires an even number of tasks")
    ordered = sorted(
        records,
        key=lambda record: (-record.estimated_latency_score, record.task_id),
    )
    split = len(ordered) // 2
    long_half = ordered[:split]
    short_half = list(reversed(ordered[split:]))
    long_items = _layered_plan(
        long_half,
        cpu_ids=cpu_ids,
        schedule_name=TRAPEZOID_PIPELINE,
        side="long",
    )
    short_items = _layered_plan(
        short_half,
        cpu_ids=cpu_ids,
        schedule_name=TRAPEZOID_PIPELINE,
        side="short",
        order_offset=len(long_items),
    )
    return long_items + short_items


def build_phase_shifted_trapezoid_plan(
    records: list[TaskRecord],
    *,
    cpu_ids: list[int],
) -> list[ScheduleItem]:
    plan = build_trapezoid_pipeline_plan(records, cpu_ids=cpu_ids)
    long_items_by_core = {
        item.core_index: item
        for item in plan
        if item.side == "long" and item.layer_index == 0
    }
    sorted_long_cores = sorted(
        long_items_by_core,
        key=lambda core_index: (
            -long_items_by_core[core_index].task.estimated_latency_score,
            long_items_by_core[core_index].task.task_id,
            core_index,
        ),
    )
    short_first_count = 0 if len(sorted_long_cores) == 1 else len(sorted_long_cores) // 2
    short_first_cores = set(sorted_long_cores[:short_first_count])
    shifted = []
    for item in plan:
        schedule_name = PHASE_SHIFTED_TRAPEZOID
        if item.core_index not in short_first_cores:
            order_index = item.order_index
        elif item.side == "long":
            order_index = item.order_index + len(plan)
        else:
            order_index = item.order_index - len(plan)
        shifted.append(
            replace(
                item,
                schedule_name=schedule_name,
                order_index=order_index,
            )
        )
    return sorted(shifted, key=lambda item: item.order_index)


def _items_by_core(plan: list[ScheduleItem]) -> dict[int, list[ScheduleItem]]:
    grouped: dict[int, list[ScheduleItem]] = {}
    for item in sorted(plan, key=lambda value: (value.core_index, value.order_index)):
        grouped.setdefault(item.core_index, []).append(item)
    return grouped


def _validate_schedule_inputs(plan: list[ScheduleItem], *, steps_per_env: int) -> None:
    if not plan:
        raise ValueError("plan must not be empty")
    if steps_per_env < 1:
        raise ValueError("steps_per_env must be >= 1")
    invalid_cpu_ids = [item.cpu_id for item in plan if item.cpu_id < 0]
    if invalid_cpu_ids:
        raise ValueError(f"cpu_id must be >= 0: {invalid_cpu_ids[0]}")
    task_ids = [item.task.task_id for item in plan]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("duplicate task_id in schedule plan")


def apply_cpu_affinity(cpu_id: int | None) -> bool:
    if cpu_id is None or not hasattr(os, "sched_setaffinity"):
        return False
    try:
        os.sched_setaffinity(0, {cpu_id})
    except (OSError, ValueError):
        return False
    return True


def build_worker_plans(plan: list[ScheduleItem]) -> dict[int, list[ScheduleItem]]:
    return _items_by_core(plan)


def _is_random_baseline_name(schedule_name: str) -> bool:
    return schedule_name == RANDOM_BASELINE or schedule_name.startswith(
        f"{RANDOM_BASELINE}_"
    )


def _next_item_for_core(
    items: list[ScheduleItem],
    per_task_counts: dict[int, int],
    *,
    steps_per_env: int,
    cursor: int,
) -> tuple[ScheduleItem | None, int]:
    if not items:
        return None, cursor
    for offset in range(len(items)):
        index = (cursor + offset) % len(items)
        item = items[index]
        if per_task_counts.get(item.task.task_id, 0) < steps_per_env:
            return item, (index + 1) % len(items)
    return None, cursor


def _next_round_commands(
    grouped: dict[int, list[ScheduleItem]],
    per_task_counts: dict[int, int],
    cursors: dict[int, int],
    *,
    steps_per_env: int,
) -> list[tuple[int, ScheduleItem]]:
    commands = []
    for core_index in sorted(grouped):
        item, next_cursor = _next_item_for_core(
            grouped[core_index],
            per_task_counts,
            steps_per_env=steps_per_env,
            cursor=cursors[core_index],
        )
        cursors[core_index] = next_cursor
        if item is None:
            continue
        per_task_counts[item.task.task_id] += 1
        commands.append((core_index, item))
    return commands


def _worker_loop(
    *,
    core_index: int,
    items: list[ScheduleItem],
    steps_per_env: int,
    env_factory: Any,
    dummy_action: list[float],
    bind_cpu_affinity: bool,
    command_queue: Any,
    result_queue: Any,
) -> None:
    del steps_per_env
    cpu_id = items[0].cpu_id if items else None
    affinity_applied = False
    envs: dict[int, Any] = {}
    task_counts = {item.task.task_id: 0 for item in items}
    current_item: ScheduleItem | None = None
    current_step_index: int | None = None
    current_phase = "affinity"
    try:
        affinity_applied = apply_cpu_affinity(cpu_id) if bind_cpu_affinity else False
        current_phase = "env_init"
        for item in items:
            current_item = item
            current_step_index = task_counts[item.task.task_id]
            if item.task.task_id not in envs:
                envs[item.task.task_id] = env_factory(item)
        current_item = None
        current_phase = "command_wait"
        result_queue.put(
            {
                "event": "ready",
                "core_index": core_index,
                "cpu_id": cpu_id,
                "cpu_affinity_applied": affinity_applied,
            }
        )
        while True:
            command = command_queue.get()
            if command == "stop":
                break
            round_index, item = command
            current_item = item
            env = envs[item.task.task_id]
            task_step_index = task_counts[item.task.task_id]
            current_step_index = task_step_index
            current_phase = "step"
            start = time.perf_counter()
            env.step(np.asarray(dummy_action, dtype=np.float32))
            end = time.perf_counter()
            latency_s = max(float(end - start), 0.0)
            task_counts[item.task.task_id] = task_step_index + 1
            result_queue.put(
                {
                    "event": "step",
                    "round_index": round_index,
                    "core_index": core_index,
                    "cpu_id": cpu_id,
                    "task_id": item.task.task_id,
                    "task_name": item.task.task_name,
                    "task_step_index": task_step_index,
                    "latency_s": latency_s,
                    "start_time_s": float(start),
                    "end_time_s": float(end),
                    "cpu_affinity_applied": affinity_applied,
                }
            )
            current_item = None
            current_step_index = None
            current_phase = "command_wait"
    except Exception as exc:
        error = {
            "event": "error",
            "phase": current_phase,
            "core_index": core_index,
            "cpu_id": cpu_id,
            "error_type": exc.__class__.__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        if current_item is not None:
            error.update(
                {
                    "task_id": current_item.task.task_id,
                    "task_name": current_item.task.task_name,
                    "task_step_index": current_step_index,
                }
            )
        result_queue.put(error)
    finally:
        for env in envs.values():
            close = getattr(env, "close", None)
            if close is not None:
                close()


def _chunk_worker_loop(
    *,
    core_index: int,
    items: list[ScheduleItem],
    env_factory: Any,
    dummy_action: list[float],
    bind_cpu_affinity: bool,
    command_queue: Any,
    result_queue: Any,
) -> None:
    cpu_id = items[0].cpu_id if items else None
    affinity_applied = False
    envs: dict[int, Any] = {}
    task_counts = {item.task.task_id: 0 for item in items}
    current_item: ScheduleItem | None = None
    current_step_index: int | None = None
    current_phase = "affinity"
    try:
        affinity_applied = apply_cpu_affinity(cpu_id) if bind_cpu_affinity else False
        current_phase = "env_init"
        for item in items:
            current_item = item
            current_step_index = task_counts[item.task.task_id]
            if item.task.task_id not in envs:
                envs[item.task.task_id] = env_factory(item)
        current_item = None
        current_phase = "command_wait"
        result_queue.put(
            {
                "event": "ready",
                "core_index": core_index,
                "cpu_id": cpu_id,
                "cpu_affinity_applied": affinity_applied,
            }
        )
        while True:
            command = command_queue.get()
            if command == "stop":
                break
            command_type = command[0]
            if command_type == "step":
                _, round_index, generation_index, chunk_step_index, item = command
                current_item = item
                env = envs[item.task.task_id]
                task_step_index = task_counts[item.task.task_id]
                current_step_index = task_step_index
                current_phase = "chunk_step"
                start = time.perf_counter()
                env.step(np.asarray(dummy_action, dtype=np.float32))
                end = time.perf_counter()
                latency_s = max(float(end - start), 0.0)
                task_counts[item.task.task_id] = task_step_index + 1
                result_queue.put(
                    {
                        "event": "step",
                        "round_index": round_index,
                        "generation_index": generation_index,
                        "chunk_step_index": chunk_step_index,
                        "core_index": core_index,
                        "cpu_id": cpu_id,
                        "task_id": item.task.task_id,
                        "task_name": item.task.task_name,
                        "task_step_index": task_step_index,
                        "latency_s": latency_s,
                        "start_time_s": float(start),
                        "end_time_s": float(end),
                        "cpu_affinity_applied": affinity_applied,
                    }
                )
            elif command_type == "chunk":
                _, round_index, generation_index, chunk_size, item = command
                current_item = item
                env = envs[item.task.task_id]
                for local_chunk_step_index in range(chunk_size):
                    task_step_index = task_counts[item.task.task_id]
                    current_step_index = task_step_index
                    current_phase = "chunk_step"
                    start = time.perf_counter()
                    env.step(np.asarray(dummy_action, dtype=np.float32))
                    end = time.perf_counter()
                    latency_s = max(float(end - start), 0.0)
                    task_counts[item.task.task_id] = task_step_index + 1
                    result_queue.put(
                        {
                            "event": "step",
                            "round_index": round_index,
                            "generation_index": generation_index,
                            "chunk_step_index": local_chunk_step_index,
                            "core_index": core_index,
                            "cpu_id": cpu_id,
                            "task_id": item.task.task_id,
                            "task_name": item.task.task_name,
                            "task_step_index": task_step_index,
                            "latency_s": latency_s,
                            "start_time_s": float(start),
                            "end_time_s": float(end),
                            "cpu_affinity_applied": affinity_applied,
                        }
                    )
            else:
                raise ValueError(f"unknown chunk worker command: {command_type!r}")
            current_item = None
            current_step_index = None
            current_phase = "command_wait"
    except Exception as exc:
        error = {
            "event": "error",
            "phase": current_phase,
            "core_index": core_index,
            "cpu_id": cpu_id,
            "error_type": exc.__class__.__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        if current_item is not None:
            error.update(
                {
                    "task_id": current_item.task.task_id,
                    "task_name": current_item.task.task_name,
                    "task_step_index": current_step_index,
                }
            )
        result_queue.put(error)
    finally:
        for env in envs.values():
            close = getattr(env, "close", None)
            if close is not None:
                close()


def _command_diagnostic(command: tuple[int, ScheduleItem]) -> dict[str, Any]:
    core_index, item = command
    return {
        "core_index": core_index,
        "cpu_id": item.cpu_id,
        "task_id": item.task.task_id,
        "task_name": item.task.task_name,
    }


def _trapezoid_items_by_layer_and_core(
    plan: list[ScheduleItem],
) -> dict[int, dict[int, list[ScheduleItem]]]:
    side_order = {"long": 0, "short": 1}
    grouped: dict[int, dict[int, list[ScheduleItem]]] = {}
    for item in sorted(
        plan,
        key=lambda value: (
            value.layer_index,
            value.core_index,
            side_order.get(value.side, 2),
            value.order_index,
        ),
    ):
        grouped.setdefault(item.layer_index, {}).setdefault(item.core_index, []).append(item)
    return grouped


def _items_by_side_and_core(
    plan: list[ScheduleItem],
    *,
    sides: list[str],
) -> dict[str, dict[int, list[ScheduleItem]]]:
    grouped: dict[str, dict[int, list[ScheduleItem]]] = {side: {} for side in sides}
    valid_sides = set(sides)
    for item in sorted(
        plan,
        key=lambda value: (value.side, value.core_index, value.order_index),
    ):
        if item.side not in valid_sides:
            continue
        grouped[item.side].setdefault(item.core_index, []).append(item)
    return grouped


def _process_exitcode_diagnostics(
    processes: list[tuple[int, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "pid": process.pid,
            "core_index": core_index,
            "exitcode": process.exitcode,
            "is_alive": process.is_alive(),
        }
        for core_index, process in processes
    ]


def _close_mp_queue(mp_queue: Any) -> None:
    close = getattr(mp_queue, "close", None)
    if close is not None:
        try:
            close()
        except Exception:
            pass
    join_thread = getattr(mp_queue, "join_thread", None)
    if join_thread is not None:
        try:
            join_thread()
        except Exception:
            pass


def _coerce_worker_latency(raw_result: dict[str, Any], *, schedule_name: str) -> float:
    raw_latency = raw_result.get("latency_s")
    try:
        latency_s = float(raw_latency)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "invalid worker latency for "
            f"schedule_name={schedule_name}, round_index={raw_result.get('round_index')}, "
            f"core_index={raw_result.get('core_index')}, "
            f"task_id={raw_result.get('task_id')}: {raw_latency!r}"
        ) from exc
    if not np.isfinite(latency_s) or latency_s < 0.0:
        raise ValueError(
            "invalid worker latency for "
            f"schedule_name={schedule_name}, round_index={raw_result.get('round_index')}, "
            f"core_index={raw_result.get('core_index')}, "
            f"task_id={raw_result.get('task_id')}: {latency_s!r}"
        )
    return latency_s


def _worker_latency_error(exc: Exception, *, schedule_name: str) -> dict[str, Any]:
    return {
        "event": "error",
        "schedule_name": schedule_name,
        "error_type": exc.__class__.__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }


def _worker_startup_diagnostic(core_index: int, items: list[ScheduleItem]) -> dict[str, Any]:
    return {
        "core_index": core_index,
        "cpu_id": items[0].cpu_id if items else None,
        "task_ids": [item.task.task_id for item in items],
        "task_names": [item.task.task_name for item in items],
    }


def _startup_timeout_error(
    *,
    schedule_name: str,
    pending_workers: list[dict[str, Any]],
    processes: list[tuple[int, Any]],
    startup_timeout_s: float,
) -> dict[str, Any]:
    return {
        "event": "timeout",
        "phase": "startup",
        "schedule_name": schedule_name,
        "pending_workers": pending_workers,
        "process_exitcodes": _process_exitcode_diagnostics(processes),
        "error_type": "TimeoutError",
        "error": (
            "timed out waiting for process worker startup "
            f"after {startup_timeout_s:.3f}s"
        ),
    }


def run_schedule_with_process_workers(
    plan: list[ScheduleItem],
    *,
    steps_per_env: int,
    env_factory: Any,
    dummy_action: list[float],
    subprocess_timeout_s: float = 300.0,
    startup_timeout_s: float | None = None,
    mp_context: Any | None = None,
    bind_cpu_affinity: bool = True,
) -> ProcessRunResult:
    _validate_schedule_inputs(plan, steps_per_env=steps_per_env)
    schedule_name = plan[0].schedule_name
    grouped = build_worker_plans(plan)
    per_task_counts = {item.task.task_id: 0 for item in plan}
    cursors = dict.fromkeys(grouped, 0)
    run_start_time_s: float | None = None
    ctx = mp_context or mp.get_context("spawn")
    result_queue = ctx.Queue()
    command_queues: dict[int, Any] = {}
    processes: list[tuple[int, Any]] = []
    events: list[StepEvent] = []
    startup_timeout_s = subprocess_timeout_s if startup_timeout_s is None else startup_timeout_s
    try:
        for core_index, items in sorted(grouped.items()):
            command_queue = ctx.Queue()
            command_queues[core_index] = command_queue
            process = ctx.Process(
                target=_worker_loop,
                kwargs={
                    "core_index": core_index,
                    "items": items,
                    "steps_per_env": steps_per_env,
                    "env_factory": env_factory,
                    "dummy_action": dummy_action,
                    "bind_cpu_affinity": bind_cpu_affinity,
                    "command_queue": command_queue,
                    "result_queue": result_queue,
                },
            )
            process.start()
            processes.append((core_index, process))

        ready_cores: set[int] = set()
        startup_pending_workers = [
            _worker_startup_diagnostic(core_index, items)
            for core_index, items in sorted(grouped.items())
        ]
        startup_deadline = time.monotonic() + startup_timeout_s
        while len(ready_cores) < len(grouped):
            remaining_s = max(startup_deadline - time.monotonic(), 0.0)
            if remaining_s <= 0.0:
                return ProcessRunResult(
                    events=events,
                    errors=[
                        _startup_timeout_error(
                            schedule_name=schedule_name,
                            pending_workers=[
                                worker
                                for worker in startup_pending_workers
                                if worker["core_index"] not in ready_cores
                            ],
                            processes=processes,
                            startup_timeout_s=startup_timeout_s,
                        )
                    ],
                )
            try:
                result = result_queue.get(timeout=remaining_s)
            except queue.Empty:
                return ProcessRunResult(
                    events=events,
                    errors=[
                        _startup_timeout_error(
                            schedule_name=schedule_name,
                            pending_workers=[
                                worker
                                for worker in startup_pending_workers
                                if worker["core_index"] not in ready_cores
                            ],
                            processes=processes,
                            startup_timeout_s=startup_timeout_s,
                        )
                    ],
                )
            event_type = result.get("event")
            if event_type == "ready":
                ready_cores.add(int(result["core_index"]))
                continue
            if event_type == "error":
                result.setdefault("schedule_name", schedule_name)
                return ProcessRunResult(events=events, errors=[result])
            return ProcessRunResult(
                events=events,
                errors=[
                    {
                        "event": "error",
                        "phase": "startup",
                        "schedule_name": schedule_name,
                        "error_type": "ValueError",
                        "error": f"unexpected worker result during startup: {result!r}",
                    }
                ],
            )

        next_round_index = 0
        pending_commands_by_core: dict[int, dict[str, Any]] = {}
        active_layer_core_items: dict[int, list[ScheduleItem]] = {}
        active_layer_core_task_index: dict[int, int] = {}
        active_trapezoid_layer_index: int | None = None
        active_trapezoid_step_index = 0
        trapezoid_layers = (
            _trapezoid_items_by_layer_and_core(plan)
            if schedule_name == TRAPEZOID_PIPELINE
            else None
        )
        round_barrier_grouped = (
            grouped
            if schedule_name == TASK_ID_BASELINE or _is_random_baseline_name(schedule_name)
            else None
        )
        active_round_commands = 0
        pending_trapezoid_layers = (
            sorted(trapezoid_layers)
            if trapezoid_layers is not None
            else []
        )
        step_barrier_grouped = (
            grouped
            if schedule_name
            in {
                ODD_EVEN_BINPACK,
                WARMUP_ODD_EVEN_BINPACK,
                MINMAX_BINPACK,
                WARMUP_MINMAX_BINPACK,
            }
            else None
        )
        active_step_grouped: dict[int, list[ScheduleItem]] = {}
        active_step_core_task_index: dict[int, int] = {}
        active_step_index = 0

        def dispatch_item(core_index: int, item: ScheduleItem) -> None:
            nonlocal next_round_index
            per_task_counts[item.task.task_id] += 1
            round_index = next_round_index
            next_round_index += 1
            command_queues[core_index].put((round_index, item))
            pending_commands_by_core[core_index] = {
                **_command_diagnostic((core_index, item)),
                "round_index": round_index,
            }

        def dispatch_next(core_index: int) -> None:
            item, next_cursor = _next_item_for_core(
                grouped[core_index],
                per_task_counts,
                steps_per_env=steps_per_env,
                cursor=cursors[core_index],
            )
            cursors[core_index] = next_cursor
            if item is None:
                return
            dispatch_item(core_index, item)

        def dispatch_round_barrier_round() -> None:
            nonlocal active_round_commands
            active_round_commands = 0
            if round_barrier_grouped is None:
                return
            for core_index in sorted(round_barrier_grouped):
                item, next_cursor = _next_item_for_core(
                    round_barrier_grouped[core_index],
                    per_task_counts,
                    steps_per_env=steps_per_env,
                    cursor=cursors[core_index],
                )
                cursors[core_index] = next_cursor
                if item is None:
                    continue
                dispatch_item(core_index, item)
                active_round_commands += 1

        def dispatch_next_trapezoid_core(core_index: int) -> None:
            items = active_layer_core_items.get(core_index)
            if not items:
                return
            task_index = active_layer_core_task_index.get(core_index, 0)
            if task_index < len(items):
                item = items[task_index]
                active_layer_core_task_index[core_index] = task_index + 1
                dispatch_item(core_index, item)
                return
            active_layer_core_items.pop(core_index, None)
            active_layer_core_task_index.pop(core_index, None)

        def dispatch_next_trapezoid_layer() -> None:
            nonlocal active_trapezoid_layer_index, active_trapezoid_step_index
            if not pending_trapezoid_layers:
                active_trapezoid_layer_index = None
                return
            layer_index = pending_trapezoid_layers.pop(0)
            active_trapezoid_layer_index = layer_index
            active_trapezoid_step_index = 0
            active_layer_core_items.clear()
            active_layer_core_task_index.clear()
            assert trapezoid_layers is not None
            for core_index, items in sorted(trapezoid_layers[layer_index].items()):
                active_layer_core_items[core_index] = items
                active_layer_core_task_index[core_index] = 0
                dispatch_next_trapezoid_core(core_index)

        def dispatch_next_trapezoid_step() -> None:
            nonlocal active_trapezoid_step_index
            if trapezoid_layers is None:
                return
            active_trapezoid_step_index += 1
            if active_trapezoid_step_index >= steps_per_env:
                dispatch_next_trapezoid_layer()
                return
            assert active_trapezoid_layer_index is not None
            assert trapezoid_layers is not None
            active_layer_core_items.clear()
            active_layer_core_task_index.clear()
            for core_index, items in sorted(trapezoid_layers[active_trapezoid_layer_index].items()):
                active_layer_core_items[core_index] = items
                active_layer_core_task_index[core_index] = 0
                dispatch_next_trapezoid_core(core_index)

        def dispatch_next_step_barrier_core(core_index: int) -> None:
            items = active_step_grouped.get(core_index)
            if not items:
                return
            task_index = active_step_core_task_index.get(core_index, 0)
            while task_index < len(items):
                item = items[task_index]
                if per_task_counts[item.task.task_id] == active_step_index:
                    active_step_core_task_index[core_index] = task_index + 1
                    dispatch_item(core_index, item)
                    return
                task_index += 1
            active_step_grouped.pop(core_index, None)
            active_step_core_task_index.pop(core_index, None)

        def dispatch_current_barrier_step() -> None:
            active_step_grouped.clear()
            active_step_core_task_index.clear()
            if step_barrier_grouped is None:
                return
            for core_index, items in sorted(step_barrier_grouped.items()):
                active_step_grouped[core_index] = items
                active_step_core_task_index[core_index] = 0
                dispatch_next_step_barrier_core(core_index)

        def advance_barrier_step() -> None:
            nonlocal active_step_index
            active_step_index += 1
            if active_step_index >= steps_per_env:
                active_step_grouped.clear()
                active_step_core_task_index.clear()
                return
            dispatch_current_barrier_step()

        run_start_time_s = time.perf_counter()
        if trapezoid_layers is not None:
            dispatch_next_trapezoid_layer()
        elif step_barrier_grouped is not None:
            dispatch_current_barrier_step()
        elif round_barrier_grouped is not None:
            dispatch_round_barrier_round()
        else:
            for core_index in sorted(grouped):
                dispatch_next(core_index)

        while pending_commands_by_core:
            deadline = time.monotonic() + subprocess_timeout_s
            remaining_s = max(deadline - time.monotonic(), 0.0)
            try:
                result = result_queue.get(timeout=remaining_s)
            except queue.Empty:
                return ProcessRunResult(
                    events=events,
                    errors=[
                        {
                            "event": "timeout",
                            "schedule_name": schedule_name,
                            "round_index": min(
                                command["round_index"]
                                for command in pending_commands_by_core.values()
                            ),
                            "pending_commands": list(pending_commands_by_core.values()),
                            "process_exitcodes": _process_exitcode_diagnostics(processes),
                            "error_type": "TimeoutError",
                            "error": (
                                "timed out waiting for process worker result "
                                f"after {subprocess_timeout_s:.3f}s"
                            ),
                        }
                    ],
                )
            if result.get("event") == "error":
                result.setdefault("schedule_name", schedule_name)
                return ProcessRunResult(events=events, errors=[result])
            if result.get("event") != "step":
                return ProcessRunResult(
                    events=events,
                    errors=[
                        {
                            "event": "error",
                            "schedule_name": schedule_name,
                            "error_type": "ValueError",
                            "error": f"unexpected worker result: {result!r}",
                        }
                    ],
                )

            try:
                latency_s = _coerce_worker_latency(result, schedule_name=schedule_name)
            except ValueError as exc:
                return ProcessRunResult(
                    events=events,
                    errors=[_worker_latency_error(exc, schedule_name=schedule_name)],
                )

            core_index = int(result["core_index"])
            pending_commands_by_core.pop(core_index, None)
            round_index = int(result["round_index"])
            start_time_s = max(float(result["start_time_s"]) - run_start_time_s, 0.0)
            end_time_s = max(float(result["end_time_s"]) - run_start_time_s, 0.0)
            events.append(
                StepEvent(
                    schedule_name=schedule_name,
                    round_index=round_index,
                    core_index=core_index,
                    cpu_id=int(result["cpu_id"]),
                    task_id=int(result["task_id"]),
                    task_name=str(result["task_name"]),
                    task_step_index=int(result["task_step_index"]),
                    latency_s=latency_s,
                    round_wall_time_s=latency_s,
                    idle_time_s=0.0,
                    cpu_affinity_applied=bool(result["cpu_affinity_applied"]),
                    start_time_s=start_time_s,
                    end_time_s=end_time_s,
                )
            )
            if step_barrier_grouped is not None:
                dispatch_next_step_barrier_core(core_index)
                if not pending_commands_by_core and not active_step_grouped:
                    advance_barrier_step()
            elif round_barrier_grouped is not None:
                active_round_commands -= 1
                if active_round_commands == 0:
                    dispatch_round_barrier_round()
            elif trapezoid_layers is None:
                dispatch_next(core_index)
            else:
                dispatch_next_trapezoid_core(core_index)
                if not pending_commands_by_core and active_layer_core_items:
                    for pending_core_index in sorted(active_layer_core_items):
                        dispatch_next_trapezoid_core(pending_core_index)
                if not pending_commands_by_core and not active_layer_core_items:
                    dispatch_next_trapezoid_step()
        return ProcessRunResult(events=events, errors=[])
    finally:
        for command_queue in command_queues.values():
            try:
                command_queue.put("stop")
            except Exception:
                pass
        for _, process in processes:
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        for command_queue in command_queues.values():
            _close_mp_queue(command_queue)
        _close_mp_queue(result_queue)


def _record_chunk_worker_result(
    result: dict[str, Any],
    *,
    schedule_name: str,
    run_start_time_s: float,
) -> StepEvent:
    latency_s = _coerce_worker_latency(result, schedule_name=schedule_name)
    start_time_s = max(float(result["start_time_s"]) - run_start_time_s, 0.0)
    end_time_s = max(float(result["end_time_s"]) - run_start_time_s, 0.0)
    return StepEvent(
        schedule_name=schedule_name,
        round_index=int(result["round_index"]),
        core_index=int(result["core_index"]),
        cpu_id=int(result["cpu_id"]),
        task_id=int(result["task_id"]),
        task_name=str(result["task_name"]),
        task_step_index=int(result["task_step_index"]),
        latency_s=latency_s,
        round_wall_time_s=latency_s,
        idle_time_s=0.0,
        cpu_affinity_applied=bool(result["cpu_affinity_applied"]),
        start_time_s=start_time_s,
        end_time_s=end_time_s,
    )


def _get_chunk_worker_step(
    result_queue: Any,
    *,
    schedule_name: str,
    run_start_time_s: float,
    subprocess_timeout_s: float,
    processes: list[tuple[int, Any]],
    pending_commands: list[dict[str, Any]],
) -> tuple[StepEvent | None, dict[str, Any] | None]:
    deadline = time.monotonic() + subprocess_timeout_s
    remaining_s = max(deadline - time.monotonic(), 0.0)
    try:
        result = result_queue.get(timeout=remaining_s)
    except queue.Empty:
        return None, {
            "event": "timeout",
            "schedule_name": schedule_name,
            "pending_commands": pending_commands,
            "process_exitcodes": _process_exitcode_diagnostics(processes),
            "error_type": "TimeoutError",
            "error": (
                "timed out waiting for chunk process worker result "
                f"after {subprocess_timeout_s:.3f}s"
            ),
        }
    if result.get("event") == "error":
        result.setdefault("schedule_name", schedule_name)
        return None, result
    if result.get("event") != "step":
        return None, {
            "event": "error",
            "schedule_name": schedule_name,
            "error_type": "ValueError",
            "error": f"unexpected worker result: {result!r}",
        }
    try:
        return (
            _record_chunk_worker_result(
                result,
                schedule_name=schedule_name,
                run_start_time_s=run_start_time_s,
            ),
            None,
        )
    except ValueError as exc:
        return None, _worker_latency_error(exc, schedule_name=schedule_name)


def _start_chunk_workers(
    plan: list[ScheduleItem],
    *,
    env_factory: Any,
    dummy_action: list[float],
    bind_cpu_affinity: bool,
    subprocess_timeout_s: float,
    startup_timeout_s: float | None,
    mp_context: Any | None,
    schedule_name: str,
) -> tuple[Any, dict[int, Any], list[tuple[int, Any]], list[dict[str, Any]]]:
    grouped = build_worker_plans(plan)
    ctx = mp_context or mp.get_context("spawn")
    result_queue = ctx.Queue()
    command_queues: dict[int, Any] = {}
    processes: list[tuple[int, Any]] = []
    startup_timeout_s = subprocess_timeout_s if startup_timeout_s is None else startup_timeout_s
    for core_index, items in sorted(grouped.items()):
        command_queue = ctx.Queue()
        command_queues[core_index] = command_queue
        process = ctx.Process(
            target=_chunk_worker_loop,
            kwargs={
                "core_index": core_index,
                "items": items,
                "env_factory": env_factory,
                "dummy_action": dummy_action,
                "bind_cpu_affinity": bind_cpu_affinity,
                "command_queue": command_queue,
                "result_queue": result_queue,
            },
        )
        process.start()
        processes.append((core_index, process))

    ready_cores: set[int] = set()
    pending_workers = [
        _worker_startup_diagnostic(core_index, items)
        for core_index, items in sorted(grouped.items())
    ]
    startup_deadline = time.monotonic() + startup_timeout_s
    while len(ready_cores) < len(grouped):
        remaining_s = max(startup_deadline - time.monotonic(), 0.0)
        if remaining_s <= 0.0:
            return (
                result_queue,
                command_queues,
                processes,
                [
                    _startup_timeout_error(
                        schedule_name=schedule_name,
                        pending_workers=[
                            worker
                            for worker in pending_workers
                            if worker["core_index"] not in ready_cores
                        ],
                        processes=processes,
                        startup_timeout_s=startup_timeout_s,
                    )
                ],
            )
        try:
            result = result_queue.get(timeout=remaining_s)
        except queue.Empty:
            return (
                result_queue,
                command_queues,
                processes,
                [
                    _startup_timeout_error(
                        schedule_name=schedule_name,
                        pending_workers=[
                            worker
                            for worker in pending_workers
                            if worker["core_index"] not in ready_cores
                        ],
                        processes=processes,
                        startup_timeout_s=startup_timeout_s,
                    )
                ],
            )
        event_type = result.get("event")
        if event_type == "ready":
            ready_cores.add(int(result["core_index"]))
            continue
        if event_type == "error":
            result.setdefault("schedule_name", schedule_name)
            return result_queue, command_queues, processes, [result]
        return (
            result_queue,
            command_queues,
            processes,
            [
                {
                    "event": "error",
                    "phase": "startup",
                    "schedule_name": schedule_name,
                    "error_type": "ValueError",
                    "error": f"unexpected worker result during startup: {result!r}",
                }
            ],
        )
    return result_queue, command_queues, processes, []


def _stop_chunk_workers(
    command_queues: dict[int, Any],
    result_queue: Any,
    processes: list[tuple[int, Any]],
) -> None:
    for command_queue in command_queues.values():
        try:
            command_queue.put("stop")
        except Exception:
            pass
    for _, process in processes:
        process.join(timeout=1.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
    for command_queue in command_queues.values():
        _close_mp_queue(command_queue)
    _close_mp_queue(result_queue)


def run_chunk_barrier_with_process_workers(
    plan: list[ScheduleItem],
    *,
    num_action_chunks: int,
    chunk_size: int,
    env_factory: Any,
    dummy_action: list[float],
    generation_latency_s: float = 0.0,
    subprocess_timeout_s: float = 300.0,
    startup_timeout_s: float | None = None,
    mp_context: Any | None = None,
    bind_cpu_affinity: bool = True,
    schedule_name: str = RLINF_DEFAULT_BOUND_CHUNK,
) -> ProcessRunResult:
    _validate_schedule_inputs(plan, steps_per_env=max(num_action_chunks * chunk_size, 1))
    if num_action_chunks < 1:
        raise ValueError("num_action_chunks must be >= 1")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    result_queue, command_queues, processes, startup_errors = _start_chunk_workers(
        plan,
        env_factory=env_factory,
        dummy_action=dummy_action,
        bind_cpu_affinity=bind_cpu_affinity,
        subprocess_timeout_s=subprocess_timeout_s,
        startup_timeout_s=startup_timeout_s,
        mp_context=mp_context,
        schedule_name=schedule_name,
    )
    events: list[StepEvent] = []
    try:
        if startup_errors:
            return ProcessRunResult(events=events, errors=startup_errors)
        grouped = build_worker_plans(plan)
        run_start_time_s = time.perf_counter()
        round_index = 0
        for generation_index in range(num_action_chunks):
            if generation_latency_s > 0.0:
                time.sleep(generation_latency_s)
            for chunk_step_index in range(chunk_size):
                pending_commands = []
                for core_index, items in sorted(grouped.items()):
                    for item in items:
                        command_queues[core_index].put(
                            (
                                "step",
                                round_index,
                                generation_index,
                                chunk_step_index,
                                item,
                            )
                        )
                        pending_commands.append(
                            {
                                **_command_diagnostic((core_index, item)),
                                "round_index": round_index,
                                "generation_index": generation_index,
                                "chunk_step_index": chunk_step_index,
                            }
                        )
                for _ in pending_commands:
                    event, error = _get_chunk_worker_step(
                        result_queue,
                        schedule_name=schedule_name,
                        run_start_time_s=run_start_time_s,
                        subprocess_timeout_s=subprocess_timeout_s,
                        processes=processes,
                        pending_commands=pending_commands,
                    )
                    if error is not None:
                        return ProcessRunResult(events=events, errors=[error])
                    assert event is not None
                    events.append(event)
                round_index += 1
        return ProcessRunResult(events=events, errors=[])
    finally:
        _stop_chunk_workers(command_queues, result_queue, processes)


def run_chunk_independent_with_process_workers(
    plan: list[ScheduleItem],
    *,
    num_action_chunks: int,
    chunk_size: int,
    env_factory: Any,
    dummy_action: list[float],
    generation_latency_s: float = 0.0,
    subprocess_timeout_s: float = 300.0,
    startup_timeout_s: float | None = None,
    mp_context: Any | None = None,
    bind_cpu_affinity: bool = True,
    schedule_name: str = RLINF_OPTIMIZED_BOUND_CHUNK,
) -> ProcessRunResult:
    _validate_schedule_inputs(plan, steps_per_env=max(num_action_chunks * chunk_size, 1))
    if num_action_chunks < 1:
        raise ValueError("num_action_chunks must be >= 1")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    result_queue, command_queues, processes, startup_errors = _start_chunk_workers(
        plan,
        env_factory=env_factory,
        dummy_action=dummy_action,
        bind_cpu_affinity=bind_cpu_affinity,
        subprocess_timeout_s=subprocess_timeout_s,
        startup_timeout_s=startup_timeout_s,
        mp_context=mp_context,
        schedule_name=schedule_name,
    )
    events: list[StepEvent] = []
    try:
        if startup_errors:
            return ProcessRunResult(events=events, errors=startup_errors)
        grouped = build_worker_plans(plan)
        run_start_time_s = time.perf_counter()
        round_index = 0
        for generation_index in range(num_action_chunks):
            if generation_latency_s > 0.0:
                time.sleep(generation_latency_s)
            pending_commands = []
            for core_index, items in sorted(grouped.items()):
                for item in items:
                    command_queues[core_index].put(
                        ("chunk", round_index, generation_index, chunk_size, item)
                    )
                    pending_commands.append(
                        {
                            **_command_diagnostic((core_index, item)),
                            "round_index": round_index,
                            "generation_index": generation_index,
                            "chunk_size": chunk_size,
                        }
                    )
            expected_events = len(pending_commands) * chunk_size
            for _ in range(expected_events):
                event, error = _get_chunk_worker_step(
                    result_queue,
                    schedule_name=schedule_name,
                    run_start_time_s=run_start_time_s,
                    subprocess_timeout_s=subprocess_timeout_s,
                    processes=processes,
                    pending_commands=pending_commands,
                )
                if error is not None:
                    return ProcessRunResult(events=events, errors=[error])
                assert event is not None
                events.append(event)
            round_index += 1
        return ProcessRunResult(events=events, errors=[])
    finally:
        _stop_chunk_workers(command_queues, result_queue, processes)


def run_schedule_with_step_function(
    plan: list[ScheduleItem],
    *,
    steps_per_env: int,
    step_fn: Any,
    cpu_affinity_by_core: dict[int, bool] | None = None,
) -> list[StepEvent]:
    _validate_schedule_inputs(plan, steps_per_env=steps_per_env)
    step_barrier_grouped = (
        _items_by_core(plan)
        if plan[0].schedule_name
        in {
            ODD_EVEN_BINPACK,
            WARMUP_ODD_EVEN_BINPACK,
            MINMAX_BINPACK,
            WARMUP_MINMAX_BINPACK,
        }
        else None
    )
    per_task_counts = {item.task.task_id: 0 for item in plan}
    events: list[StepEvent] = []
    round_index = 0
    if step_barrier_grouped is None:
        grouped = _items_by_core(plan)
        cursors = dict.fromkeys(grouped, 0)
        while any(count < steps_per_env for count in per_task_counts.values()):
            round_results = []
            for core_index, items in grouped.items():
                item, next_cursor = _next_item_for_core(
                    items,
                    per_task_counts,
                    steps_per_env=steps_per_env,
                    cursor=cursors[core_index],
                )
                cursors[core_index] = next_cursor
                if item is None:
                    continue
                task_step_index = per_task_counts[item.task.task_id]
                raw_latency = step_fn(item, task_step_index)
                try:
                    latency_s = float(raw_latency)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "invalid latency for "
                        f"task_id={item.task.task_id}, core_index={item.core_index}, "
                        f"step_index={task_step_index}: {raw_latency!r}"
                    ) from exc
                if not np.isfinite(latency_s) or latency_s < 0.0:
                    raise ValueError(
                        "invalid latency for "
                        f"task_id={item.task.task_id}, core_index={item.core_index}, "
                        f"step_index={task_step_index}: {latency_s!r}"
                    )
                per_task_counts[item.task.task_id] = task_step_index + 1
                round_results.append((item, task_step_index, latency_s))
            if not round_results:
                break
            round_wall_time_s = max(latency for _, _, latency in round_results)
            for item, task_step_index, latency_s in round_results:
                affinity = True
                if cpu_affinity_by_core is not None:
                    affinity = bool(cpu_affinity_by_core.get(item.core_index, True))
                events.append(
                    StepEvent(
                        schedule_name=item.schedule_name,
                        round_index=round_index,
                        core_index=item.core_index,
                        cpu_id=item.cpu_id,
                        task_id=item.task.task_id,
                        task_name=item.task.task_name,
                        task_step_index=task_step_index,
                        latency_s=latency_s,
                        round_wall_time_s=round_wall_time_s,
                        idle_time_s=max(round_wall_time_s - latency_s, 0.0),
                        cpu_affinity_applied=affinity,
                    )
                )
            round_index += 1
        return events

    for active_step_index in range(steps_per_env):
        grouped = step_barrier_grouped
        core_task_indexes = dict.fromkeys(grouped, 0)
        while True:
            round_results = []
            for core_index, items in grouped.items():
                item = None
                task_index = core_task_indexes[core_index]
                while task_index < len(items):
                    candidate = items[task_index]
                    if per_task_counts[candidate.task.task_id] == active_step_index:
                        item = candidate
                        break
                    task_index += 1
                core_task_indexes[core_index] = task_index + 1
                if item is None:
                    continue
                task_step_index = per_task_counts[item.task.task_id]
                raw_latency = step_fn(item, task_step_index)
                try:
                    latency_s = float(raw_latency)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "invalid latency for "
                        f"task_id={item.task.task_id}, core_index={item.core_index}, "
                        f"step_index={task_step_index}: {raw_latency!r}"
                    ) from exc
                if not np.isfinite(latency_s) or latency_s < 0.0:
                    raise ValueError(
                        "invalid latency for "
                        f"task_id={item.task.task_id}, core_index={item.core_index}, "
                        f"step_index={task_step_index}: {latency_s!r}"
                    )
                per_task_counts[item.task.task_id] = task_step_index + 1
                round_results.append((item, task_step_index, latency_s))
            if not round_results:
                break
            round_wall_time_s = max(latency for _, _, latency in round_results)
            for item, task_step_index, latency_s in round_results:
                affinity = True
                if cpu_affinity_by_core is not None:
                    affinity = bool(cpu_affinity_by_core.get(item.core_index, True))
                events.append(
                    StepEvent(
                        schedule_name=item.schedule_name,
                        round_index=round_index,
                        core_index=item.core_index,
                        cpu_id=item.cpu_id,
                        task_id=item.task.task_id,
                        task_name=item.task.task_name,
                        task_step_index=task_step_index,
                        latency_s=latency_s,
                        round_wall_time_s=round_wall_time_s,
                        idle_time_s=max(round_wall_time_s - latency_s, 0.0),
                        cpu_affinity_applied=affinity,
                    )
                )
            round_index += 1
    return events


def measure_warmup_latency_with_step_function(
    records: list[TaskRecord],
    *,
    steps_per_task: int,
    step_fn: Any,
) -> list[WarmupLatency]:
    if steps_per_task < 1:
        raise ValueError("steps_per_task must be >= 1")
    results = []
    for record in records:
        item = ScheduleItem(
            schedule_name="warmup_profile",
            task=record,
            core_index=0,
            cpu_id=0,
            layer_index=0,
            order_index=0,
        )
        latencies = [
            float(step_fn(item, step_index)) * 1000.0
            for step_index in range(steps_per_task)
        ]
        results.append(
            WarmupLatency(
                task_id=record.task_id,
                task_name=record.task_name,
                mean_latency_ms=float(np.mean(np.asarray(latencies))),
                samples=steps_per_task,
            )
        )
    return results


def measure_warmup_latency_serial(
    records: list[TaskRecord],
    *,
    env_factory: Any,
    dummy_action: list[float],
    steps_per_task: int,
) -> list[WarmupLatency]:
    if steps_per_task < 1:
        raise ValueError("steps_per_task must be >= 1")
    results = []
    action = np.asarray(dummy_action, dtype=np.float32)
    for order_index, record in enumerate(records):
        item = ScheduleItem(
            schedule_name="warmup_profile",
            task=record,
            core_index=0,
            cpu_id=0,
            layer_index=0,
            order_index=order_index,
        )
        env = env_factory(item)
        try:
            latencies = []
            for _ in range(steps_per_task):
                start = time.perf_counter()
                env.step(action)
                end = time.perf_counter()
                latencies.append(max(float(end - start), 0.0) * 1000.0)
            results.append(
                WarmupLatency(
                    task_id=record.task_id,
                    task_name=record.task_name,
                    mean_latency_ms=float(np.mean(np.asarray(latencies))),
                    samples=steps_per_task,
                )
            )
        finally:
            close = getattr(env, "close", None)
            if close is not None:
                close()
    return results


def write_warmup_latencies(path: Path, latencies: list[WarmupLatency]) -> None:
    fieldnames = ["task_id", "task_name", "mean_latency_ms", "samples"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for latency in latencies:
            writer.writerow(_to_jsonable(latency))


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def compute_schedule_summary(
    schedule_name: str,
    events: list[StepEvent],
    *,
    failed: bool = False,
) -> dict[str, Any]:
    if failed:
        status = "failed"
    elif any(not event.cpu_affinity_applied for event in events):
        status = "degraded"
    else:
        status = "completed"
    if not events:
        return {
            "schedule_name": schedule_name,
            "status": status,
            "total_steps": 0,
            "makespan_s": 0.0,
            "steps_per_second": 0.0,
            "mean_core_idle_ratio": None,
            "cpu_affinity_success_rate": None,
        }
    rounds: dict[int, list[StepEvent]] = {}
    for event in events:
        rounds.setdefault(event.round_index, []).append(event)
    total_cores = len({event.core_index for event in events})
    timed_events = [
        event
        for event in events
        if event.start_time_s is not None and event.end_time_s is not None
    ]
    if timed_events:
        makespan_s = float(
            max(event.end_time_s for event in timed_events if event.end_time_s is not None)
            - min(event.start_time_s for event in timed_events if event.start_time_s is not None)
        )
    else:
        makespan_s = float(
            sum(
                max(event.round_wall_time_s for event in round_events)
                for round_events in rounds.values()
            )
        )
    latencies = [event.latency_s for event in events]
    idle_ratios: list[float] = []
    for round_events in rounds.values():
        round_wall_time_s = max(event.round_wall_time_s for event in round_events)
        per_round_idle_ratios = [
            0.0 if round_wall_time_s == 0.0 else event.idle_time_s / round_wall_time_s
            for event in round_events
        ]
        missing_cores = total_cores - len({event.core_index for event in round_events})
        if missing_cores > 0:
            per_round_idle_ratios.extend([1.0 if round_wall_time_s > 0.0 else 0.0] * missing_cores)
        idle_ratios.append(float(np.mean(np.asarray(per_round_idle_ratios))))
    if timed_events and makespan_s > 0.0 and total_cores > 0:
        busy_core_seconds = sum(event.latency_s for event in events)
        mean_core_idle_ratio = max(
            1.0 - busy_core_seconds / (makespan_s * total_cores),
            0.0,
        )
    else:
        mean_core_idle_ratio = float(np.mean(np.asarray(idle_ratios)))
    affinity_rate = sum(1 for event in events if event.cpu_affinity_applied) / len(events)
    return {
        "schedule_name": schedule_name,
        "status": status,
        "total_steps": len(events),
        "makespan_s": makespan_s,
        "steps_per_second": 0.0 if makespan_s == 0.0 else len(events) / makespan_s,
        "mean_step_latency_s": float(np.mean(np.asarray(latencies))),
        "median_step_latency_s": float(np.median(np.asarray(latencies))),
        "p90_step_latency_s": _percentile(latencies, 90),
        "p95_step_latency_s": _percentile(latencies, 95),
        "p99_step_latency_s": _percentile(latencies, 99),
        "mean_core_idle_ratio": float(mean_core_idle_ratio),
        "p90_round_idle_ratio": _percentile(idle_ratios, 90),
        "p99_round_idle_ratio": _percentile(idle_ratios, 99),
        "cpu_affinity_success_rate": float(affinity_rate),
    }


@dataclass
class LiberoEnvFactory:
    suite: str
    camera_height: int
    camera_width: int
    libero_type: str
    seed: int
    warmup_steps: int
    dummy_action: list[float]
    task_spec_cache: dict[int, Any] = field(default_factory=dict)
    init_state_cache: dict[int, Any] = field(default_factory=dict)

    def __call__(self, item: ScheduleItem) -> Any:
        from toolkits.profile_libero_step_latency import (
            ProfileConfig,
            build_task_trial_specs,
        )
        from toolkits.profile_libero_step_latency import (
            make_libero_env_factory as make_profile_env_factory,
        )

        if item.task.task_id not in self.task_spec_cache:
            config = ProfileConfig(
                suite=self.suite,
                task_ids=str(item.task.task_id),
                trials_per_task=1,
                specific_trial_ids=None,
                warmup_steps=self.warmup_steps,
                measure_steps=1,
                cpu_id=item.cpu_id,
                cpu_ids=None,
                camera_height=self.camera_height,
                camera_width=self.camera_width,
                libero_type=self.libero_type,
                seed=self.seed,
                output_dir=Path("."),
                dummy_action=self.dummy_action,
                stop_on_done=False,
                subprocess_timeout_s=None,
                mujoco_profiler=False,
            )
            specs, init_states = build_task_trial_specs(config)
            self.task_spec_cache[item.task.task_id] = (config, specs[0])
            self.init_state_cache[item.task.task_id] = init_states[0]
        config, spec = self.task_spec_cache[item.task.task_id]
        env = make_profile_env_factory(config, spec)()
        if hasattr(env, "seed"):
            env.seed(spec.seed)
        env.reset()
        init_state = self.init_state_cache[item.task.task_id]
        if init_state is not None and hasattr(env, "set_init_state"):
            env.set_init_state(init_state)
        for _ in range(self.warmup_steps):
            env.step(np.asarray(self.dummy_action, dtype=np.float32))
        env.reset()
        if init_state is not None and hasattr(env, "set_init_state"):
            env.set_init_state(init_state)
        return env


class CsvLatencyEnv:
    def __init__(self, latency_s: float) -> None:
        self.latency_s = latency_s

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        del action
        time.sleep(self.latency_s)
        return {}, 0.0, False, {}

    def close(self) -> None:
        return None


class CsvLatencyEnvFactory:
    def __call__(self, item: ScheduleItem) -> CsvLatencyEnv:
        return CsvLatencyEnv(latency_s=item.task.mean_latency_ms / 1000.0)


def make_libero_env_factory(
    *,
    suite: str,
    camera_height: int,
    camera_width: int,
    libero_type: str,
    seed: int,
    warmup_steps: int,
    dummy_action: list[float],
) -> Any:
    return LiberoEnvFactory(
        suite=suite,
        camera_height=camera_height,
        camera_width=camera_width,
        libero_type=libero_type,
        seed=seed,
        warmup_steps=warmup_steps,
        dummy_action=dummy_action,
    )


@dataclass(frozen=True)
class BenchmarkResult:
    schedule_name: str
    events: list[StepEvent]
    summary: dict[str, Any]
    errors: list[dict[str, Any]]


class BenchmarkRunner:
    def __init__(
        self,
        *,
        steps_per_env: int,
        step_fn: Any | None = None,
        env_factory: Any | None = None,
        dummy_action: list[float] | None = None,
        subprocess_timeout_s: float = 300.0,
        startup_timeout_s: float | None = None,
        bind_cpu_affinity: bool = True,
    ) -> None:
        self.steps_per_env = steps_per_env
        self.step_fn = step_fn
        self.env_factory = env_factory
        self.dummy_action = dummy_action
        self.subprocess_timeout_s = subprocess_timeout_s
        self.startup_timeout_s = startup_timeout_s
        self.bind_cpu_affinity = bind_cpu_affinity

    def run(self, schedule_name: str, plan: list[ScheduleItem]) -> BenchmarkResult:
        try:
            if self.step_fn is not None:
                events = run_schedule_with_step_function(
                    plan,
                    steps_per_env=self.steps_per_env,
                    step_fn=self.step_fn,
                )
                errors: list[dict[str, Any]] = []
            else:
                if self.env_factory is None:
                    raise RuntimeError("env_factory is required when step_fn is not provided")
                if self.dummy_action is None:
                    raise RuntimeError("dummy_action is required when step_fn is not provided")
                process_result = run_schedule_with_process_workers(
                    plan,
                    steps_per_env=self.steps_per_env,
                    env_factory=self.env_factory,
                    dummy_action=self.dummy_action,
                    subprocess_timeout_s=self.subprocess_timeout_s,
                    startup_timeout_s=self.startup_timeout_s,
                    bind_cpu_affinity=self.bind_cpu_affinity,
                )
                events = process_result.events
                errors = process_result.errors
            summary = compute_schedule_summary(schedule_name, events, failed=bool(errors))
            return BenchmarkResult(
                schedule_name=schedule_name,
                events=events,
                summary=summary,
                errors=errors,
            )
        except Exception as exc:
            summary = compute_schedule_summary(schedule_name, [], failed=True)
            return BenchmarkResult(
                schedule_name=schedule_name,
                events=[],
                summary=summary,
                errors=[
                    {
                        "event": "error",
                        "schedule_name": schedule_name,
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                ],
            )


class ChunkBenchmarkRunner:
    def __init__(
        self,
        *,
        num_action_chunks: int,
        chunk_size: int,
        env_factory: Any,
        dummy_action: list[float],
        generation_latency_s: float,
        subprocess_timeout_s: float = 300.0,
        startup_timeout_s: float | None = None,
    ) -> None:
        self.num_action_chunks = num_action_chunks
        self.chunk_size = chunk_size
        self.env_factory = env_factory
        self.dummy_action = dummy_action
        self.generation_latency_s = generation_latency_s
        self.subprocess_timeout_s = subprocess_timeout_s
        self.startup_timeout_s = startup_timeout_s

    def run(self, schedule_name: str, plan: list[ScheduleItem]) -> BenchmarkResult:
        try:
            if schedule_name == RLINF_DEFAULT_UNBOUND_CHUNK:
                process_result = run_chunk_barrier_with_process_workers(
                    plan,
                    num_action_chunks=self.num_action_chunks,
                    chunk_size=self.chunk_size,
                    env_factory=self.env_factory,
                    dummy_action=self.dummy_action,
                    generation_latency_s=self.generation_latency_s,
                    subprocess_timeout_s=self.subprocess_timeout_s,
                    startup_timeout_s=self.startup_timeout_s,
                    bind_cpu_affinity=False,
                    schedule_name=schedule_name,
                )
            elif schedule_name == RLINF_DEFAULT_BOUND_CHUNK:
                process_result = run_chunk_barrier_with_process_workers(
                    plan,
                    num_action_chunks=self.num_action_chunks,
                    chunk_size=self.chunk_size,
                    env_factory=self.env_factory,
                    dummy_action=self.dummy_action,
                    generation_latency_s=self.generation_latency_s,
                    subprocess_timeout_s=self.subprocess_timeout_s,
                    startup_timeout_s=self.startup_timeout_s,
                    bind_cpu_affinity=True,
                    schedule_name=schedule_name,
                )
            elif schedule_name == RLINF_OPTIMIZED_BOUND_CHUNK:
                process_result = run_chunk_independent_with_process_workers(
                    plan,
                    num_action_chunks=self.num_action_chunks,
                    chunk_size=self.chunk_size,
                    env_factory=self.env_factory,
                    dummy_action=self.dummy_action,
                    generation_latency_s=self.generation_latency_s,
                    subprocess_timeout_s=self.subprocess_timeout_s,
                    startup_timeout_s=self.startup_timeout_s,
                    bind_cpu_affinity=True,
                    schedule_name=schedule_name,
                )
            else:
                raise ValueError(f"unsupported chunk schedule: {schedule_name}")
            summary = compute_schedule_summary(
                schedule_name,
                process_result.events,
                failed=bool(process_result.errors),
            )
            return BenchmarkResult(
                schedule_name=schedule_name,
                events=process_result.events,
                summary=summary,
                errors=process_result.errors,
            )
        except Exception as exc:
            return BenchmarkResult(
                schedule_name=schedule_name,
                events=[],
                summary=compute_schedule_summary(schedule_name, [], failed=True),
                errors=[
                    {
                        "event": "error",
                        "schedule_name": schedule_name,
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                ],
            )


def parse_int_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("integer list must not be empty")
    return [int(item) for item in items]


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_to_jsonable(value), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_selected_tasks(path: Path, records: list[TaskRecord]) -> None:
    fixed_fieldnames = [
        "task_id",
        "task_name",
        "mean_latency_ms",
        "njnt",
        "ngeom",
        "estimated_latency_score",
    ]
    optional_fieldnames = sorted({key for record in records for key in record.extra})
    fieldnames = fixed_fieldnames + optional_fieldnames
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {key: getattr(record, key) for key in fixed_fieldnames}
            row.update({key: record.extra.get(key, "") for key in optional_fieldnames})
            writer.writerow(row)


def write_schedule_plan(path: Path, plan: list[ScheduleItem]) -> None:
    fieldnames = [
        "schedule_name",
        "order_index",
        "core_index",
        "cpu_id",
        "layer_index",
        "side",
        "task_id",
        "task_name",
        "estimated_latency_score",
        "mean_latency_ms",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in plan:
            writer.writerow(
                {
                    "schedule_name": item.schedule_name,
                    "order_index": item.order_index,
                    "core_index": item.core_index,
                    "cpu_id": item.cpu_id,
                    "layer_index": item.layer_index,
                    "side": item.side,
                    "task_id": item.task.task_id,
                    "task_name": item.task.task_name,
                    "estimated_latency_score": item.task.estimated_latency_score,
                    "mean_latency_ms": item.task.mean_latency_ms,
                }
            )


def write_step_events(path: Path, events: list[StepEvent]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(_to_jsonable(event), sort_keys=True) + "\n")


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for summary in summaries for key in summary})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def compute_comparison_metrics(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = next(
        (
            summary
            for summary in summaries
            if summary.get("schedule_name") == TASK_ID_BASELINE
        ),
        None,
    )
    baseline_sps = _as_optional_float(
        baseline.get("steps_per_second") if baseline is not None else None
    )
    baseline_idle = _as_optional_float(
        baseline.get("mean_core_idle_ratio") if baseline is not None else None
    )
    baseline_comparison: dict[str, dict[str, float | None]] = {}
    for summary in summaries:
        schedule_name = str(summary.get("schedule_name", ""))
        if schedule_name == TASK_ID_BASELINE:
            continue
        steps_per_second = _as_optional_float(summary.get("steps_per_second"))
        idle_ratio = _as_optional_float(summary.get("mean_core_idle_ratio"))
        bubble_reduction = None
        if (
            baseline_idle is not None
            and baseline_idle != 0.0
            and idle_ratio is not None
        ):
            bubble_reduction = (baseline_idle - idle_ratio) / baseline_idle
        baseline_comparison[schedule_name] = {
            "speedup_vs_task_id_baseline": _safe_ratio(
                steps_per_second,
                baseline_sps,
            ),
            "bubble_reduction_vs_task_id_baseline": bubble_reduction,
        }

    random_summaries = [
        summary
        for summary in summaries
        if _is_random_baseline_name(str(summary.get("schedule_name", "")))
    ]
    random_steps_per_second = [
        value
        for value in (
            _as_optional_float(summary.get("steps_per_second"))
            for summary in random_summaries
        )
        if value is not None
    ]
    random_idle_ratios = [
        value
        for value in (
            _as_optional_float(summary.get("mean_core_idle_ratio"))
            for summary in random_summaries
        )
        if value is not None
    ]
    random_aggregate = None
    if random_summaries:
        random_aggregate = {
            "count": len(random_summaries),
            "mean_steps_per_second": (
                float(np.mean(np.asarray(random_steps_per_second)))
                if random_steps_per_second
                else None
            ),
            "median_steps_per_second": (
                float(np.median(np.asarray(random_steps_per_second)))
                if random_steps_per_second
                else None
            ),
            "best_steps_per_second": (
                max(random_steps_per_second) if random_steps_per_second else None
            ),
            "worst_steps_per_second": (
                min(random_steps_per_second) if random_steps_per_second else None
            ),
            "mean_core_idle_ratio": (
                float(np.mean(np.asarray(random_idle_ratios)))
                if random_idle_ratios
                else None
            ),
        }
    return {
        "baseline_comparison": baseline_comparison,
        "random_aggregate": random_aggregate,
    }


def add_comparison_metrics_to_summaries(summaries: list[dict[str, Any]]) -> None:
    comparison = compute_comparison_metrics(summaries)["baseline_comparison"]
    for summary in summaries:
        schedule_metrics = comparison.get(str(summary.get("schedule_name", "")))
        if schedule_metrics is None:
            continue
        summary.update(schedule_metrics)


def _format_optional_float(value: Any) -> str:
    number = _as_optional_float(value)
    if number is None:
        return ""
    return f"{number:.6f}"


def _task_ids_for_side(items: list[ScheduleItem], side: str) -> str:
    task_ids = [
        str(item.task.task_id)
        for item in sorted(items, key=lambda value: value.order_index)
        if item.side == side
    ]
    return ", ".join(task_ids)


def _append_trapezoid_mapping_evidence(
    lines: list[str],
    plans: dict[str, list[ScheduleItem]] | None,
) -> None:
    if not plans or TRAPEZOID_PIPELINE not in plans:
        return
    lines.extend(["", "## Trapezoid Mapping Evidence", ""])
    lines.append(
        "The trapezoid plan pairs long and short task groups on the same core. "
        "Within each group step, runtime dispatch is per-core asynchronous: a core "
        "starts its paired short task as soon as its own long task finishes, without "
        "waiting for the full long group to complete. The next step of that group, "
        "and the next long+short layer, start only after every core has completed "
        "the current long+short step."
    )
    lines.append("")
    lines.append("| core_index | cpu_id | long task ids | short task ids |")
    lines.append("|---:|---:|---|---|")
    grouped = _items_by_core(plans[TRAPEZOID_PIPELINE])
    for core_index, items in sorted(grouped.items()):
        cpu_id = items[0].cpu_id if items else ""
        lines.append(
            "| {core_index} | {cpu_id} | {long_tasks} | {short_tasks} |".format(
                core_index=core_index,
                cpu_id=cpu_id,
                long_tasks=_task_ids_for_side(items, "long"),
                short_tasks=_task_ids_for_side(items, "short"),
            )
        )


def write_comparison_report(
    path: Path,
    summaries: list[dict[str, Any]],
    *,
    plans: dict[str, list[ScheduleItem]] | None = None,
) -> None:
    metrics = compute_comparison_metrics(summaries)
    lines = ["# LIBERO Latency Schedule Benchmark", "", "## Raw Schedule Metrics", ""]
    lines.append("| schedule | status | steps/sec | idle ratio |")
    lines.append("|---|---:|---:|---:|")
    for summary in summaries:
        lines.append(
            "| {schedule} | {status} | {sps:.6f} | {idle} |".format(
                schedule=summary["schedule_name"],
                status=summary["status"],
                sps=float(summary.get("steps_per_second") or 0.0),
                idle=summary.get("mean_core_idle_ratio"),
            )
        )
    lines.extend(["", "## Baseline Comparison", ""])
    lines.append(
        "| schedule | speedup_vs_task_id_baseline | "
        "bubble_reduction_vs_task_id_baseline |"
    )
    lines.append("|---|---:|---:|")
    for schedule_name, values in metrics["baseline_comparison"].items():
        lines.append(
            "| {schedule} | {speedup} | {bubble} |".format(
                schedule=schedule_name,
                speedup=_format_optional_float(
                    values.get("speedup_vs_task_id_baseline")
                ),
                bubble=_format_optional_float(
                    values.get("bubble_reduction_vs_task_id_baseline")
                ),
            )
        )
    random_aggregate = metrics["random_aggregate"]
    if random_aggregate is not None:
        lines.extend(["", "## Random Baseline Aggregate", ""])
        lines.append(
            "| count | mean steps/sec | median steps/sec | best steps/sec | "
            "worst steps/sec | mean idle ratio |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|")
        lines.append(
            "| {count} | {mean_sps} | {median_sps} | {best_sps} | {worst_sps} | "
            "{mean_idle} |".format(
                count=random_aggregate["count"],
                mean_sps=_format_optional_float(
                    random_aggregate["mean_steps_per_second"]
                ),
                median_sps=_format_optional_float(
                    random_aggregate["median_steps_per_second"]
                ),
                best_sps=_format_optional_float(
                    random_aggregate["best_steps_per_second"]
                ),
                worst_sps=_format_optional_float(
                    random_aggregate["worst_steps_per_second"]
                ),
                mean_idle=_format_optional_float(
                    random_aggregate["mean_core_idle_ratio"]
                ),
            )
        )
    _append_trapezoid_mapping_evidence(lines, plans)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-csv", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu-ids", required=True)
    parser.add_argument("--steps-per-env", type=int, default=100)
    parser.add_argument(
        "--chunk-benchmark",
        action="store_true",
        help=(
            "Run RLinf action-chunk benchmarks only: default unbound, default bound, "
            "and bound with independent per-env chunk execution."
        ),
    )
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--num-action-chunks", type=int, default=1)
    parser.add_argument("--generation-latency-ms", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-baseline-repeats", type=int, default=1)
    parser.add_argument("--suite", default="libero_90")
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument(
        "--libero-type",
        choices=["standard", "pro", "plus"],
        default="standard",
    )
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--subprocess-timeout-s", type=float, default=300.0)
    parser.add_argument("--startup-timeout-s", type=float, default=None)
    parser.add_argument("--dummy-action", default="0,0,0,0,0,0,-1")
    parser.add_argument(
        "--fake-latency-from-csv",
        action="store_true",
        help="Use CSV mean_latency_ms as fake latency for unit/local smoke tests.",
    )
    parser.add_argument(
        "--include-same-core-latency-order",
        action="store_true",
        help=(
            "Also run a controlled schedule that keeps task_id baseline task-to-core "
            "assignment and only reorders tasks within each core by estimated latency."
        ),
    )
    parser.add_argument(
        "--include-odd-even-binpack",
        action="store_true",
        help=(
            "Also run a schedule that sorts tasks by estimated latency, splits 1-based "
            "odd/even positions into two blocking phases, and bin-packs each phase "
            "across fixed cores."
        ),
    )
    parser.add_argument(
        "--include-minmax-binpack",
        action="store_true",
        help=(
            "Also run a schedule that bin-packs all selected tasks across fixed cores "
            "to minimize maximum per-core estimated latency."
        ),
    )
    parser.add_argument(
        "--include-warmup-odd-even-binpack",
        action="store_true",
        help=(
            "Also run odd/even bin packing sorted and weighted by measured warmup "
            "latency from the selected tasks."
        ),
    )
    parser.add_argument(
        "--include-warmup-minmax-binpack",
        action="store_true",
        help=(
            "Also run all-task min-max bin packing weighted by measured warmup "
            "latency from the selected tasks."
        ),
    )
    parser.add_argument(
        "--warmup-profile-steps",
        type=int,
        default=3,
        help="Number of measured warmup env.step calls per task for warmup bin packing.",
    )
    parser.add_argument(
        "--no-cpu-affinity",
        action="store_true",
        help=(
            "Keep the same number of worker slots but do not bind workers to specific "
            "CPU cores; let the OS scheduler place the processes."
        ),
    )
    parser.add_argument(
        "--skip-trapezoid-plans",
        action="store_true",
        help="Do not run trapezoid_pipeline or phase_shifted_trapezoid plans.",
    )
    return parser


def _dummy_action_from_arg(value: str) -> list[float]:
    try:
        action = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--dummy-action must be a comma-separated list of floats") from exc
    if not action:
        raise ValueError("--dummy-action must not be empty")
    return action


def _fake_step_fn(item: ScheduleItem, step_index: int) -> float:
    del step_index
    return item.task.mean_latency_ms / 1000.0


def _parse_cpu_ids_or_exit(parser: argparse.ArgumentParser, value: str) -> list[int]:
    try:
        cpu_ids = parse_int_list(value)
    except ValueError as exc:
        parser.error(f"--cpu-ids: {exc}")
    invalid_cpu_ids = [cpu_id for cpu_id in cpu_ids if cpu_id < 0]
    if invalid_cpu_ids:
        parser.error(f"--cpu-ids must be >= 0: {invalid_cpu_ids[0]}")
    if len(set(cpu_ids)) != len(cpu_ids):
        parser.error("--cpu-ids must not contain duplicates")
    return cpu_ids


def _dummy_action_or_exit(parser: argparse.ArgumentParser, value: str) -> list[float]:
    try:
        return _dummy_action_from_arg(value)
    except ValueError as exc:
        parser.error(str(exc))


def _validate_cli_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.num_envs < 1:
        parser.error("--num-envs must be >= 1")
    if args.num_envs % 2 != 0:
        parser.error("--num-envs must be even for trapezoid_pipeline")
    if args.steps_per_env < 1:
        parser.error("--steps-per-env must be >= 1")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be >= 1")
    if args.num_action_chunks < 1:
        parser.error("--num-action-chunks must be >= 1")
    if args.generation_latency_ms < 0.0:
        parser.error("--generation-latency-ms must be >= 0")
    if args.random_baseline_repeats < 0:
        parser.error("--random-baseline-repeats must be >= 0")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be >= 0")
    if args.warmup_profile_steps < 1:
        parser.error("--warmup-profile-steps must be >= 1")
    if args.subprocess_timeout_s <= 0.0:
        parser.error("--subprocess-timeout-s must be > 0")
    if args.startup_timeout_s is not None and args.startup_timeout_s <= 0.0:
        parser.error("--startup-timeout-s must be > 0")


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _validate_cli_args(parser, args)
    cpu_ids = _parse_cpu_ids_or_exit(parser, args.cpu_ids)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    errors_path = output_dir / "errors.jsonl"
    errors_path.unlink(missing_ok=True)
    records = estimate_latency_scores(
        sample_task_records(
            load_task_records(args.task_csv),
            num_envs=args.num_envs,
            seed=args.seed,
        )
    )
    run_config = {
        key: _to_jsonable(value) for key, value in vars(args).items()
    }
    run_config["cpu_ids"] = cpu_ids
    write_json(output_dir / "run_config.json", run_config)
    write_selected_tasks(output_dir / "selected_tasks.csv", records)

    if args.chunk_benchmark:
        dummy_action = _dummy_action_or_exit(parser, args.dummy_action)
        if args.fake_latency_from_csv:
            env_factory = CsvLatencyEnvFactory()
        else:
            env_factory = make_libero_env_factory(
                suite=args.suite,
                camera_height=args.camera_height,
                camera_width=args.camera_width,
                libero_type=args.libero_type,
                seed=args.seed,
                warmup_steps=args.warmup_steps,
                dummy_action=dummy_action,
            )
        base_plan = build_task_id_baseline_plan(records, cpu_ids=cpu_ids)
        chunk_schedule_names = [
            RLINF_DEFAULT_UNBOUND_CHUNK,
            RLINF_DEFAULT_BOUND_CHUNK,
            RLINF_OPTIMIZED_BOUND_CHUNK,
        ]
        plans = [
            [replace(item, schedule_name=schedule_name) for item in base_plan]
            for schedule_name in chunk_schedule_names
        ]
        runner = ChunkBenchmarkRunner(
            num_action_chunks=args.num_action_chunks,
            chunk_size=args.chunk_size,
            env_factory=env_factory,
            dummy_action=dummy_action,
            generation_latency_s=args.generation_latency_ms / 1000.0,
            subprocess_timeout_s=args.subprocess_timeout_s,
            startup_timeout_s=args.startup_timeout_s,
        )
        summaries = []
        errors = []
        plans_by_name: dict[str, list[ScheduleItem]] = {}
        for plan in plans:
            schedule_name = plan[0].schedule_name
            plans_by_name[schedule_name] = plan
            write_schedule_plan(output_dir / f"schedule_plan_{schedule_name}.csv", plan)
            result = runner.run(schedule_name, plan)
            write_step_events(output_dir / f"step_events_{schedule_name}.jsonl", result.events)
            summaries.append(result.summary)
            errors.extend(result.errors)
        write_summary_csv(output_dir / "schedule_summary.csv", summaries)
        write_json(output_dir / "schedule_summary.json", summaries)
        write_comparison_report(
            output_dir / "comparison_report.md",
            summaries,
            plans=plans_by_name,
        )
        if errors:
            with errors_path.open("w", encoding="utf-8") as handle:
                for error in errors:
                    handle.write(json.dumps(_to_jsonable(error), sort_keys=True) + "\n")
        return 0

    plans: list[list[ScheduleItem]] = [build_task_id_baseline_plan(records, cpu_ids=cpu_ids)]
    if args.fake_latency_from_csv:
        plans.append(build_historical_optimal_plan(records, cpu_ids=cpu_ids))
    if args.include_same_core_latency_order:
        plans.append(build_same_core_latency_order_plan(records, cpu_ids=cpu_ids))
    if args.include_odd_even_binpack:
        plans.append(build_odd_even_binpack_plan(records, cpu_ids=cpu_ids))
    if args.include_minmax_binpack:
        plans.append(build_minmax_binpack_plan(records, cpu_ids=cpu_ids))
    warmup_latencies: list[WarmupLatency] | None = None
    if args.fake_latency_from_csv:
        if args.include_warmup_odd_even_binpack or args.include_warmup_minmax_binpack:
            warmup_latencies = measure_warmup_latency_with_step_function(
                records,
                steps_per_task=args.warmup_profile_steps,
                step_fn=_fake_step_fn,
            )
        runner = BenchmarkRunner(
            steps_per_env=args.steps_per_env,
            step_fn=_fake_step_fn,
        )
    else:
        dummy_action = _dummy_action_or_exit(parser, args.dummy_action)
        env_factory = make_libero_env_factory(
            suite=args.suite,
            camera_height=args.camera_height,
            camera_width=args.camera_width,
            libero_type=args.libero_type,
            seed=args.seed,
            warmup_steps=args.warmup_steps,
            dummy_action=dummy_action,
        )
        if args.include_warmup_odd_even_binpack or args.include_warmup_minmax_binpack:
            warmup_env_factory = make_libero_env_factory(
                suite=args.suite,
                camera_height=args.camera_height,
                camera_width=args.camera_width,
                libero_type=args.libero_type,
                seed=args.seed,
                warmup_steps=args.warmup_steps,
                dummy_action=dummy_action,
            )
            warmup_latencies = measure_warmup_latency_serial(
                records,
                env_factory=warmup_env_factory,
                dummy_action=dummy_action,
                steps_per_task=args.warmup_profile_steps,
            )
        runner = BenchmarkRunner(
            steps_per_env=args.steps_per_env,
            env_factory=env_factory,
            dummy_action=dummy_action,
            subprocess_timeout_s=args.subprocess_timeout_s,
            startup_timeout_s=args.startup_timeout_s,
            bind_cpu_affinity=not args.no_cpu_affinity,
        )
    if warmup_latencies is not None:
        write_warmup_latencies(output_dir / "warmup_latency.csv", warmup_latencies)
        warmup_latency_ms = {
            latency.task_id: latency.mean_latency_ms
            for latency in warmup_latencies
        }
        if args.include_warmup_odd_even_binpack:
            plans.append(
                build_warmup_odd_even_binpack_plan(
                    records,
                    cpu_ids=cpu_ids,
                    warmup_latency_ms=warmup_latency_ms,
                )
            )
        if args.include_warmup_minmax_binpack:
            plans.append(
                build_warmup_minmax_binpack_plan(
                    records,
                    cpu_ids=cpu_ids,
                    warmup_latency_ms=warmup_latency_ms,
                )
            )
    if not args.skip_trapezoid_plans:
        plans.extend(
            [
                build_trapezoid_pipeline_plan(records, cpu_ids=cpu_ids),
                build_phase_shifted_trapezoid_plan(records, cpu_ids=cpu_ids),
            ]
        )
    for repeat in range(args.random_baseline_repeats):
        plans.append(
            build_random_baseline_plan(
                records,
                cpu_ids=cpu_ids,
                seed=args.seed + repeat + 1,
            )
        )

    summaries = []
    errors = []
    plans_by_name: dict[str, list[ScheduleItem]] = {}
    for plan in plans:
        schedule_name = plan[0].schedule_name
        if schedule_name == RANDOM_BASELINE and args.random_baseline_repeats > 1:
            random_index = sum(
                1
                for summary in summaries
                if summary["schedule_name"].startswith(RANDOM_BASELINE)
            )
            schedule_name = f"{RANDOM_BASELINE}_{random_index}"
            plan = [replace(item, schedule_name=schedule_name) for item in plan]
        plans_by_name[schedule_name] = plan
        write_schedule_plan(output_dir / f"schedule_plan_{schedule_name}.csv", plan)
        result = runner.run(schedule_name, plan)
        write_step_events(output_dir / f"step_events_{schedule_name}.jsonl", result.events)
        summaries.append(result.summary)
        errors.extend(result.errors)

    add_comparison_metrics_to_summaries(summaries)
    write_summary_csv(output_dir / "schedule_summary.csv", summaries)
    write_json(output_dir / "schedule_summary.json", summaries)
    write_comparison_report(
        output_dir / "comparison_report.md",
        summaries,
        plans=plans_by_name,
    )
    if errors:
        with errors_path.open("w", encoding="utf-8") as handle:
            for error in errors:
                handle.write(json.dumps(_to_jsonable(error), sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
