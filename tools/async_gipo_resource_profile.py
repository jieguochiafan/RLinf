#!/usr/bin/env python3
"""Aggregate async GIPO CPU, torch, and phase resource profiles."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
import shutil
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

CPU_FIELDS = (
    "timestamp",
    "datetime",
    "cpu",
    "util_pct",
    "thread_count",
    "migration_count",
)
GPU_WORKER_FIELDS = (
    "timestamp",
    "datetime",
    "worker",
    "component",
    "rank",
    "gpu_uuid",
    "gpu_label",
    "kernel_busy_pct",
    "est_sm_occupancy_pct",
    "kernel_count",
)
GPU_DEVICE_FIELDS = (
    "timestamp",
    "datetime",
    "gpu_uuid",
    "gpu_label",
    "kernel_busy_pct",
)
PHASE_FIELDS = ("phase", "step", "component", "rank", "start", "end", "source")
OCCUPANCY_KEY = "est. achieved occupancy %"
DEVICE_KEYS = ("device", "Device Id", "device_id")
PHASE_KEY_FIELDS = (
    "rank",
    "mode",
    "epoch",
    "chunk_step",
    "stage",
    "phase",
    "step",
    "trajectory_id",
    "episode",
    "batch",
    "pid",
)


@dataclass(frozen=True)
class KernelInterval:
    """A wall-clock kernel interval assigned to one worker and physical GPU."""

    worker: str
    component: str
    rank: int | None
    gpu_uuid: str
    gpu_label: str
    start_s: float
    end_s: float
    occupancy_pct: float | None


@dataclass(frozen=True)
class PhaseWindow:
    """A paired wall-clock phase interval."""

    phase: str
    step: int | str | None
    component: str
    rank: int | str | None
    start: float
    end: float
    source: str
    pid: int | None = None
    session_key: str = ""


@dataclass(frozen=True)
class TraceDescriptor:
    """One unique manifest entry without an attached trace payload."""

    worker: str
    session_id: str
    component: str
    rank: int | None
    pid: int | None
    devices: dict[int, dict[str, Any]]
    trace_path: Path
    manifest_index: int
    chunk_index: int | None


class IntervalUnionAccumulator:
    """Maintain sorted disjoint intervals with incremental merging."""

    def __init__(self) -> None:
        self.intervals: list[tuple[float, float]] = []

    def add(self, start: float, end: float) -> None:
        if end <= start:
            return
        index = bisect.bisect_left(self.intervals, (start, end))
        if index > 0 and self.intervals[index - 1][1] >= start:
            index -= 1
            start = min(start, self.intervals[index][0])
            end = max(end, self.intervals[index][1])
            del self.intervals[index]
        while index < len(self.intervals) and self.intervals[index][0] <= end:
            start = min(start, self.intervals[index][0])
            end = max(end, self.intervals[index][1])
            del self.intervals[index]
        self.intervals.insert(index, (start, end))

    @property
    def duration(self) -> float:
        return sum(end - start for start, end in self.intervals)


class P2Median:
    """Estimate the median online with five P² markers."""

    def __init__(self) -> None:
        self._count = 0
        self._initial: list[float] = []
        self._heights: list[float] = []
        self._positions: list[int] = []
        self._desired: list[float] = []
        self._increments = [0.0, 0.25, 0.5, 0.75, 1.0]

    def update(self, value: float) -> None:
        self._count += 1
        if self._count <= 5:
            self._initial.append(value)
            if self._count == 5:
                self._heights = sorted(self._initial)
                self._positions = [1, 2, 3, 4, 5]
                self._desired = [1.0, 2.0, 3.0, 4.0, 5.0]
                self._initial.clear()
            return
        if value < self._heights[0]:
            self._heights[0] = value
            cell = 0
        elif value >= self._heights[4]:
            self._heights[4] = value
            cell = 3
        else:
            cell = next(
                index
                for index in range(4)
                if self._heights[index] <= value < self._heights[index + 1]
            )
        for index in range(cell + 1, 5):
            self._positions[index] += 1
        for index, increment in enumerate(self._increments):
            self._desired[index] += increment
        for index in range(1, 4):
            delta = self._desired[index] - self._positions[index]
            direction = 1 if delta >= 1 else -1 if delta <= -1 else 0
            if direction == 0:
                continue
            if not (
                self._positions[index + 1] - self._positions[index] > 1
                if direction > 0
                else self._positions[index - 1] - self._positions[index] < -1
            ):
                continue
            candidate = self._parabolic(index, direction)
            if not self._heights[index - 1] < candidate < self._heights[index + 1]:
                neighbor = index + direction
                candidate = self._heights[index] + direction * (
                    self._heights[neighbor] - self._heights[index]
                ) / (self._positions[neighbor] - self._positions[index])
            self._heights[index] = candidate
            self._positions[index] += direction

    def _parabolic(self, index: int, direction: int) -> float:
        lower_position = self._positions[index - 1]
        position = self._positions[index]
        upper_position = self._positions[index + 1]
        lower_height = self._heights[index - 1]
        height = self._heights[index]
        upper_height = self._heights[index + 1]
        return height + direction / (upper_position - lower_position) * (
            (position - lower_position + direction)
            * (upper_height - height)
            / (upper_position - position)
            + (upper_position - position - direction)
            * (height - lower_height)
            / (position - lower_position)
        )

    def result(self) -> float | None:
        if self._count == 0:
            return None
        if self._count < 5:
            ordered = sorted(self._initial)
            middle = len(ordered) // 2
            if len(ordered) % 2:
                return ordered[middle]
            return (ordered[middle - 1] + ordered[middle]) / 2.0
        return self._heights[2]

    @property
    def state_size(self) -> int:
        return (
            len(self._initial)
            + len(self._heights)
            + len(self._positions)
            + len(self._desired)
            + len(self._increments)
        )


@dataclass
class GpuWorkerBin:
    """Streaming GPU statistics for one worker/device/bin."""

    union: IntervalUnionAccumulator
    occupancy_weighted_sum: float = 0.0
    occupancy_weight: float = 0.0
    kernel_count: int = 0


def _visit_gpu_bin() -> None:
    """Instrumentation hook for GPU bin-visit complexity tests."""


@dataclass
class TorchParseState:
    """Mutable coverage state shared by streaming trace consumers."""

    coverage: dict[str, Any]
    warnings: list[str]
    blocks: dict[tuple[str, str], list[dict[str, Any]]]
    session_records: dict[tuple[str, str], dict[str, Any]]
    worker_records: dict[str, dict[str, Any]]


def interval_union_duration(
    intervals: Iterable[KernelInterval],
    bin_start: float,
    bin_end: float,
) -> float:
    """Return the clipped union duration of kernel intervals within a bin."""
    clipped = sorted(
        (max(interval.start_s, bin_start), min(interval.end_s, bin_end))
        for interval in intervals
        if interval.end_s > bin_start and interval.start_s < bin_end
    )
    clipped = [(start, end) for start, end in clipped if end > start]
    if not clipped:
        return 0.0
    duration = 0.0
    current_start, current_end = clipped[0]
    for start, end in clipped[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        duration += current_end - current_start
        current_start, current_end = start, end
    return duration + current_end - current_start


def _load_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Malformed JSON in {path}: line {error.lineno}, column {error.colno}"
        ) from error


def _load_trace(path: Path) -> Any:
    """Load one trace payload through an injectable lifecycle boundary."""
    return _load_json(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Malformed JSON in {path} at line {line_number}: {error.msg}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"Malformed JSON in {path} at line {line_number}: expected object"
                )
            records.append(record)
    return records


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _gpu_identity(device: dict[str, Any], local_index: int) -> tuple[str, str]:
    raw_visible_id = device.get("visible_id")
    visible_id = "" if raw_visible_id is None else str(raw_visible_id).strip()
    raw_uuid = device.get("uuid")
    gpu_uuid = "" if raw_uuid is None else str(raw_uuid).strip()
    gpu_uuid = gpu_uuid or visible_id or str(local_index)
    gpu_label = visible_id or str(local_index)
    return gpu_uuid, gpu_label


def _device_index(args: dict[str, Any]) -> int | None:
    for key in DEVICE_KEYS:
        if key in args:
            local_index = _as_int(args[key])
            if local_index is not None:
                return local_index
    return None


def _torch_coverage() -> dict[str, Any]:
    return {
        "workers": [],
        "missing_trace_files": [],
        "unmapped_device_events": 0,
        "missing_occupancy_events": 0,
        "total_trace_bytes": 0,
        "observed_gaps": [],
        "overlaps": [],
        "out_of_order_chunks": [],
        "trace_blocks": [],
        "worker_devices": [],
    }


def _interval_pair_union_duration(intervals: Iterable[tuple[float, float]]) -> float:
    sorted_intervals = sorted((start, end) for start, end in intervals if end > start)
    if not sorted_intervals:
        return 0.0
    total = 0.0
    current_start, current_end = sorted_intervals[0]
    for start, end in sorted_intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        total += current_end - current_start
        current_start, current_end = start, end
    return total + current_end - current_start


def _effective_coverage(intervals: Iterable[tuple[float, float]]) -> dict[str, Any]:
    valid = [(start, end) for start, end in intervals if end > start]
    if not valid:
        return {"start": None, "end": None, "effective_coverage_ratio": 0.0}
    start = min(interval[0] for interval in valid)
    end = max(interval[1] for interval in valid)
    span = end - start
    return {
        "start": start,
        "end": end,
        "effective_coverage_ratio": (
            _interval_pair_union_duration(valid) / span if span > 0 else 0.0
        ),
    }


def _trace_coverage_window(
    trace_events: list[Any], base_ns: int
) -> tuple[float, float] | None:
    marker_starts: list[float] = []
    marker_ends: list[float] = []
    event_starts: list[float] = []
    event_ends: list[float] = []
    base_s = base_ns / 1_000_000_000.0
    for event in trace_events:
        if not isinstance(event, dict):
            continue
        timestamp_us = _as_float(event.get("ts"))
        if timestamp_us is None:
            continue
        timestamp_s = base_s + timestamp_us / 1_000_000.0
        name = str(event.get("name", ""))
        if name == "Iteration Start: PyTorch Profiler":
            marker_starts.append(timestamp_s)
        elif name == "Record Window End":
            marker_ends.append(timestamp_s)
        event_starts.append(timestamp_s)
        duration_us = _as_float(event.get("dur"))
        event_ends.append(
            timestamp_s
            + (
                duration_us / 1_000_000.0
                if event.get("ph") == "X" and duration_us and duration_us > 0
                else 0.0
            )
        )
    if marker_starts and marker_ends:
        start = min(marker_starts)
        end = max(marker_ends)
        if end > start:
            return start, end
    if event_starts:
        start = min(event_starts)
        end = max(event_ends)
        if end > start:
            return start, end
    return None


def _legacy_load_torch_intervals(
    torch_dir: Path,
) -> tuple[list[KernelInterval], dict[str, Any], list[str]]:
    """Load every manifested torch trace below ``torch_dir``."""
    intervals: list[KernelInterval] = []
    coverage = _torch_coverage()
    warnings: list[str] = []
    blocks_by_worker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    worker_records: dict[str, dict[str, Any]] = {}

    for anchor_path in sorted(torch_dir.rglob("time_anchor.json")):
        worker_dir = anchor_path.parent
        manifest_path = worker_dir / "trace_manifest.jsonl"
        worker = worker_dir.relative_to(torch_dir).as_posix()
        anchor = _load_json(anchor_path)
        if not isinstance(anchor, dict):
            raise ValueError(f"Malformed JSON in {anchor_path}: expected object")
        component = str(anchor.get("component", "unknown"))
        rank = _as_int(anchor.get("rank"))
        devices = {
            _as_int(device.get("local_index")): device
            for device in anchor.get("cuda_devices", [])
            if isinstance(device, dict)
            and _as_int(device.get("local_index")) is not None
        }
        gpu_mappings: list[dict[str, Any]] = []
        for local_index, device in sorted(devices.items()):
            gpu_uuid, gpu_label = _gpu_identity(device, local_index)
            mapping = {
                "worker": worker,
                "component": component,
                "rank": rank,
                "local_index": local_index,
                "visible_id": (
                    ""
                    if device.get("visible_id") is None
                    else str(device.get("visible_id")).strip()
                ),
                "name": (
                    ""
                    if device.get("name") is None
                    else str(device.get("name")).strip()
                ),
                "gpu_uuid": gpu_uuid,
                "gpu_label": gpu_label,
            }
            gpu_mappings.append(mapping)
            coverage["worker_devices"].append(mapping)
        worker_record = {
            "worker": worker,
            "component": component,
            "rank": rank,
            "gpu_mappings": gpu_mappings,
            "chunk_count": 0,
            "start": None,
            "end": None,
            "effective_coverage_ratio": 0.0,
        }
        worker_records[worker] = worker_record
        coverage["workers"].append(worker_record)
        if not manifest_path.is_file():
            warnings.append(f"Missing trace manifest: {manifest_path}")
            continue

        previous_manifest_start: float | None = None
        for manifest_index, record in enumerate(_read_jsonl(manifest_path)):
            trace_name = record.get("trace_file")
            if not trace_name:
                warnings.append(
                    f"Manifest record {manifest_index} has no trace_file: {manifest_path}"
                )
                continue
            trace_path = worker_dir / str(trace_name)
            if not trace_path.is_file():
                coverage["missing_trace_files"].append(str(trace_path))
                warnings.append(f"Missing trace file: {trace_path}")
                continue
            coverage["total_trace_bytes"] += trace_path.stat().st_size
            trace = _load_json(trace_path)
            if not isinstance(trace, dict):
                raise ValueError(f"Malformed JSON in {trace_path}: expected object")
            base_ns = _as_int(trace.get("baseTimeNanoseconds"))
            if base_ns is None:
                warnings.append(f"Skipping {trace_path}: missing baseTimeNanoseconds")
                continue
            trace_events = trace.get("traceEvents", [])
            if not isinstance(trace_events, list):
                raise ValueError(
                    f"Malformed JSON in {trace_path}: traceEvents must be a list"
                )

            trace_window = _trace_coverage_window(trace_events, base_ns)
            if trace_window is None:
                warnings.append(f"Trace has no coverage window: {trace_path}")
            else:
                block = {
                    "worker": worker,
                    "trace_file": str(trace_path),
                    "manifest_index": manifest_index,
                    "start": trace_window[0],
                    "end": trace_window[1],
                }
                coverage["trace_blocks"].append(block)
                blocks_by_worker[worker].append(block)
                if (
                    previous_manifest_start is not None
                    and block["start"] < previous_manifest_start
                ):
                    coverage["out_of_order_chunks"].append(
                        {
                            "worker": worker,
                            "trace_file": str(trace_path),
                            "start": block["start"],
                            "previous_start": previous_manifest_start,
                        }
                    )
                previous_manifest_start = block["start"]
                anchor_wall_ns = _as_int(anchor.get("wall_time_ns"))
                if anchor_wall_ns is not None:
                    anchor_s = anchor_wall_ns / 1_000_000_000.0
                    if abs(block["start"] - anchor_s) > 86_400:
                        warnings.append(
                            f"Trace time is more than one day from anchor for {trace_path}"
                        )

            for event in trace_events:
                if not isinstance(event, dict):
                    continue
                categories = {
                    category.strip()
                    for category in str(event.get("cat", "")).split(",")
                }
                duration_us = _as_float(event.get("dur"))
                timestamp_us = _as_float(event.get("ts"))
                if (
                    event.get("ph") != "X"
                    or "kernel" not in categories
                    or duration_us is None
                    or duration_us <= 0
                    or timestamp_us is None
                ):
                    continue
                start_s = base_ns / 1_000_000_000.0 + timestamp_us / 1_000_000.0
                end_s = start_s + duration_us / 1_000_000.0
                args = event.get("args", {})
                if not isinstance(args, dict):
                    args = {}
                occupancy = _as_float(args.get(OCCUPANCY_KEY))
                if occupancy is None:
                    coverage["missing_occupancy_events"] += 1
                local_index = _device_index(args)
                device = devices.get(local_index)
                if device is None:
                    coverage["unmapped_device_events"] += 1
                    continue
                gpu_uuid, gpu_label = _gpu_identity(device, local_index)
                intervals.append(
                    KernelInterval(
                        worker=worker,
                        component=component,
                        rank=rank,
                        gpu_uuid=gpu_uuid,
                        gpu_label=gpu_label,
                        start_s=start_s,
                        end_s=end_s,
                        occupancy_pct=occupancy,
                    )
                )

    for worker, blocks in blocks_by_worker.items():
        sorted_blocks = sorted(blocks, key=lambda block: (block["start"], block["end"]))
        worker_coverage = _effective_coverage(
            (block["start"], block["end"]) for block in sorted_blocks
        )
        worker_records[worker].update(worker_coverage)
        worker_records[worker]["chunk_count"] = len(sorted_blocks)
        if not sorted_blocks:
            continue
        running_max_end = sorted_blocks[0]["end"]
        for current in sorted_blocks[1:]:
            if current["start"] > running_max_end:
                coverage["observed_gaps"].append(
                    {
                        "worker": worker,
                        "start": running_max_end,
                        "end": current["start"],
                        "duration_s": current["start"] - running_max_end,
                    }
                )
            elif current["start"] < running_max_end:
                coverage["overlaps"].append(
                    {
                        "worker": worker,
                        "start": current["start"],
                        "end": min(running_max_end, current["end"]),
                    }
                )
            running_max_end = max(running_max_end, current["end"])
    return intervals, coverage, warnings


def _new_torch_state() -> TorchParseState:
    return TorchParseState(
        coverage=_torch_coverage(),
        warnings=[],
        blocks=defaultdict(list),
        session_records={},
        worker_records={},
    )


def _normalized_devices(metadata: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        local_index: device
        for device in metadata.get("cuda_devices", [])
        if isinstance(device, dict)
        and (local_index := _as_int(device.get("local_index"))) is not None
    }


def _load_session_metadata(
    worker_dir: Path,
    session_id: str,
    session_count: int,
    anchor: dict[str, Any] | None,
    warnings: list[str],
) -> dict[str, Any]:
    session_path = worker_dir / f"session_{session_id}.json"
    if session_id != "__legacy__" and session_path.is_file():
        metadata = _load_json(session_path)
        if not isinstance(metadata, dict):
            raise ValueError(f"Malformed JSON in {session_path}: expected object")
        metadata_session_id = str(metadata.get("session_id", ""))
        if metadata_session_id and metadata_session_id != session_id:
            raise ValueError(
                f"Session id mismatch in {session_path}: "
                f"{metadata_session_id!r} != {session_id!r}"
            )
        return metadata
    if session_count == 1 and anchor is not None:
        warnings.append(
            f"Using legacy time_anchor.json fallback for session {session_id} "
            f"in {worker_dir}"
        )
        metadata = dict(anchor)
        metadata["session_id"] = session_id
        return metadata
    raise ValueError(
        f"Missing session metadata for {session_id} in multi-session worker {worker_dir}"
    )


def _validate_manifest_identity(
    record: dict[str, Any], metadata: dict[str, Any], manifest_path: Path
) -> None:
    for field in ("pid", "component", "rank"):
        if field not in record or record[field] is None:
            continue
        expected = metadata.get(field)
        actual = record[field]
        if field in {"pid", "rank"}:
            expected = _as_int(expected)
            actual = _as_int(actual)
        else:
            expected = str(expected)
            actual = str(actual)
        if actual != expected:
            raise ValueError(
                f"Manifest {field} mismatch in {manifest_path}: "
                f"{actual!r} != {expected!r}"
            )


def _mapping_record(
    worker: str,
    session_id: str,
    component: str,
    rank: int | None,
    local_index: int,
    device: dict[str, Any],
) -> dict[str, Any]:
    gpu_uuid, gpu_label = _gpu_identity(device, local_index)
    return {
        "worker": worker,
        "session_id": session_id,
        "component": component,
        "rank": rank,
        "local_index": local_index,
        "visible_id": (
            ""
            if device.get("visible_id") is None
            else str(device.get("visible_id")).strip()
        ),
        "name": ("" if device.get("name") is None else str(device.get("name")).strip()),
        "gpu_uuid": gpu_uuid,
        "gpu_label": gpu_label,
    }


def _iter_trace_chunks(
    torch_dir: Path, state: TorchParseState
) -> Iterable[TraceDescriptor]:
    manifest_paths = sorted(torch_dir.rglob("trace_manifest.jsonl"))
    manifest_dirs = {path.parent for path in manifest_paths}
    for anchor_path in sorted(torch_dir.rglob("time_anchor.json")):
        if anchor_path.parent not in manifest_dirs:
            state.warnings.append(
                f"Missing trace manifest: {anchor_path.parent / 'trace_manifest.jsonl'}"
            )

    for manifest_path in manifest_paths:
        worker_dir = manifest_path.parent
        worker = worker_dir.relative_to(torch_dir).as_posix()
        records = _read_jsonl(manifest_path)
        raw_session_ids = [
            str(record.get("session_id"))
            if record.get("session_id") not in {None, ""}
            else "__legacy__"
            for record in records
        ]
        session_ids = sorted(set(raw_session_ids)) or ["__legacy__"]
        anchor_path = worker_dir / "time_anchor.json"
        anchor: dict[str, Any] | None = None
        if anchor_path.is_file():
            loaded_anchor = _load_json(anchor_path)
            if not isinstance(loaded_anchor, dict):
                raise ValueError(f"Malformed JSON in {anchor_path}: expected object")
            anchor = loaded_anchor

        contexts: dict[str, tuple[dict[str, Any], dict[int, dict[str, Any]]]] = {}
        sessions: list[dict[str, Any]] = []
        worker_record = {
            "worker": worker,
            "component": None,
            "rank": None,
            "gpu_mappings": [],
            "sessions": sessions,
            "chunk_count": 0,
            "start": None,
            "end": None,
            "effective_coverage_ratio": 0.0,
        }
        state.worker_records[worker] = worker_record
        state.coverage["workers"].append(worker_record)
        for session_id in session_ids:
            metadata = _load_session_metadata(
                worker_dir,
                session_id,
                len(session_ids),
                anchor,
                state.warnings,
            )
            component = str(metadata.get("component", "unknown"))
            rank = _as_int(metadata.get("rank"))
            devices = _normalized_devices(metadata)
            mappings = [
                _mapping_record(
                    worker,
                    session_id,
                    component,
                    rank,
                    local_index,
                    device,
                )
                for local_index, device in sorted(devices.items())
            ]
            session_record = {
                "session_id": session_id,
                "component": component,
                "rank": rank,
                "pid": _as_int(metadata.get("pid")),
                "gpu_mappings": mappings,
                "chunk_count": 0,
                "start": None,
                "end": None,
                "effective_coverage_ratio": 0.0,
            }
            contexts[session_id] = (metadata, devices)
            sessions.append(session_record)
            state.session_records[(worker, session_id)] = session_record
            state.coverage["worker_devices"].extend(mappings)
            worker_record["gpu_mappings"].extend(mappings)
            if worker_record["component"] is None:
                worker_record["component"] = component
                worker_record["rank"] = rank

        exact_records: set[str] = set()
        chunk_records: dict[tuple[str, int | None], tuple[str, str]] = {}
        for manifest_index, (record, session_id) in enumerate(
            zip(records, raw_session_ids, strict=True)
        ):
            canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
            if canonical in exact_records:
                state.warnings.append(
                    f"Skipping duplicate manifest record in {manifest_path} "
                    f"at index {manifest_index}"
                )
                continue
            exact_records.add(canonical)
            chunk_index = _as_int(record.get("chunk_index"))
            trace_name = str(record.get("trace_file", ""))
            chunk_key = (session_id, chunk_index)
            previous = chunk_records.get(chunk_key)
            if previous is not None and previous != (trace_name, canonical):
                raise ValueError(
                    f"conflicting manifest chunk in {manifest_path}: "
                    f"session={session_id} chunk={chunk_index}"
                )
            chunk_records[chunk_key] = (trace_name, canonical)
            metadata, devices = contexts[session_id]
            _validate_manifest_identity(record, metadata, manifest_path)
            if not trace_name:
                state.warnings.append(
                    f"Manifest record {manifest_index} has no trace_file: {manifest_path}"
                )
                continue
            trace_path = worker_dir / trace_name
            if not trace_path.is_file():
                state.coverage["missing_trace_files"].append(str(trace_path))
                state.warnings.append(f"Missing trace file: {trace_path}")
                continue
            state.coverage["total_trace_bytes"] += trace_path.stat().st_size
            yield TraceDescriptor(
                worker=worker,
                session_id=session_id,
                component=str(metadata.get("component", "unknown")),
                rank=_as_int(metadata.get("rank")),
                pid=_as_int(metadata.get("pid")),
                devices=devices,
                trace_path=trace_path,
                manifest_index=manifest_index,
                chunk_index=chunk_index,
            )


def _record_chunk_coverage(
    chunk: TraceDescriptor,
    trace_events: list[Any],
    base_ns: int,
    state: TorchParseState,
) -> tuple[float, float] | None:
    trace_window = _trace_coverage_window(trace_events, base_ns)
    if trace_window is None:
        state.warnings.append(f"Trace has no coverage window: {chunk.trace_path}")
        return None
    block = {
        "worker": chunk.worker,
        "session_id": chunk.session_id,
        "trace_file": str(chunk.trace_path),
        "manifest_index": chunk.manifest_index,
        "chunk_index": chunk.chunk_index,
        "start": trace_window[0],
        "end": trace_window[1],
    }
    state.coverage["trace_blocks"].append(block)
    state.blocks[(chunk.worker, chunk.session_id)].append(block)
    return trace_window


def _finalize_torch_state(state: TorchParseState) -> None:
    worker_blocks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (worker, session_id), blocks in state.blocks.items():
        for previous, current in zip(blocks, blocks[1:]):
            if current["start"] < previous["start"]:
                state.coverage["out_of_order_chunks"].append(
                    {
                        "worker": worker,
                        "session_id": session_id,
                        "trace_file": current["trace_file"],
                        "start": current["start"],
                        "previous_start": previous["start"],
                        "chunk_index": current["chunk_index"],
                        "previous_chunk_index": previous["chunk_index"],
                        "manifest_index": current["manifest_index"],
                        "previous_manifest_index": previous["manifest_index"],
                    }
                )
        sorted_blocks = sorted(blocks, key=lambda item: (item["start"], item["end"]))
        session_record = state.session_records[(worker, session_id)]
        session_record.update(
            _effective_coverage((item["start"], item["end"]) for item in sorted_blocks)
        )
        session_record["chunk_count"] = len(sorted_blocks)
        worker_blocks[worker].extend(sorted_blocks)
        if not sorted_blocks:
            continue
        running_max_end = sorted_blocks[0]["end"]
        for current in sorted_blocks[1:]:
            if current["start"] > running_max_end:
                state.coverage["observed_gaps"].append(
                    {
                        "worker": worker,
                        "session_id": session_id,
                        "start": running_max_end,
                        "end": current["start"],
                        "duration_s": current["start"] - running_max_end,
                    }
                )
            elif current["start"] < running_max_end:
                state.coverage["overlaps"].append(
                    {
                        "worker": worker,
                        "session_id": session_id,
                        "start": current["start"],
                        "end": min(running_max_end, current["end"]),
                    }
                )
            running_max_end = max(running_max_end, current["end"])
    for worker, record in state.worker_records.items():
        blocks = worker_blocks.get(worker, [])
        record.update(
            _effective_coverage((item["start"], item["end"]) for item in blocks)
        )
        record["chunk_count"] = len(blocks)


def _load_chunk_kernels(
    chunk: TraceDescriptor, state: TorchParseState
) -> tuple[tuple[float, float] | None, list[KernelInterval]]:
    trace = _load_trace(chunk.trace_path)
    trace_events: list[Any] | None = None
    try:
        if not isinstance(trace, dict):
            raise ValueError(f"Malformed JSON in {chunk.trace_path}: expected object")
        base_ns = _as_int(trace.get("baseTimeNanoseconds"))
        if base_ns is None:
            state.warnings.append(
                f"Skipping {chunk.trace_path}: missing baseTimeNanoseconds"
            )
            return None, []
        trace_events = trace.get("traceEvents", [])
        if not isinstance(trace_events, list):
            raise ValueError(
                f"Malformed JSON in {chunk.trace_path}: traceEvents must be a list"
            )
        trace_window = _record_chunk_coverage(chunk, trace_events, base_ns, state)
        kernels = sorted(
            (
                kernel
                for event in trace_events
                if (kernel := _kernel_from_event(event, base_ns, chunk, state.coverage))
                is not None
            ),
            key=lambda kernel: (kernel.start_s, kernel.end_s),
        )
        return trace_window, kernels
    finally:
        del trace_events
        del trace


def load_torch_intervals(
    torch_dir: Path,
) -> tuple[list[KernelInterval], dict[str, Any], list[str]]:
    """Load torch intervals for small callers while resolving every session."""
    state = _new_torch_state()
    intervals: list[KernelInterval] = []
    for chunk in _iter_trace_chunks(torch_dir, state):
        _, chunk_kernels = _load_chunk_kernels(chunk, state)
        intervals.extend(chunk_kernels)
        del chunk_kernels
    _finalize_torch_state(state)
    return intervals, state.coverage, state.warnings


def _kernel_from_event(
    event: Any,
    base_ns: int,
    chunk: TraceDescriptor,
    coverage: dict[str, Any],
) -> KernelInterval | None:
    if not isinstance(event, dict):
        return None
    categories = {category.strip() for category in str(event.get("cat", "")).split(",")}
    duration_us = _as_float(event.get("dur"))
    timestamp_us = _as_float(event.get("ts"))
    if (
        event.get("ph") != "X"
        or "kernel" not in categories
        or duration_us is None
        or duration_us <= 0
        or timestamp_us is None
    ):
        return None
    args = event.get("args", {})
    if not isinstance(args, dict):
        args = {}
    occupancy = _as_float(args.get(OCCUPANCY_KEY))
    if occupancy is None:
        coverage["missing_occupancy_events"] += 1
    local_index = _device_index(args)
    device = chunk.devices.get(local_index)
    if device is None or local_index is None:
        coverage["unmapped_device_events"] += 1
        return None
    gpu_uuid, gpu_label = _gpu_identity(device, local_index)
    start_s = base_ns / 1_000_000_000.0 + timestamp_us / 1_000_000.0
    return KernelInterval(
        worker=chunk.worker,
        component=chunk.component,
        rank=chunk.rank,
        gpu_uuid=gpu_uuid,
        gpu_label=gpu_label,
        start_s=start_s,
        end_s=start_s + duration_us / 1_000_000.0,
        occupancy_pct=occupancy,
    )


def _parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _load_cpu_samples(
    path: Path, num_cpus: int, warnings: list[str]
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "interval_start",
            "interval_end",
            "pid",
            "tid",
            "cpu",
            "cpu_time_s",
            "migrated",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CPU columns in {path}: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            start = _as_float(row.get("interval_start"))
            end = _as_float(row.get("interval_end"))
            cpu = _as_int(row.get("cpu"))
            cpu_time = _as_float(row.get("cpu_time_s"))
            if (
                start is None
                or end is None
                or end <= start
                or cpu_time is None
                or cpu is None
                or not 0 <= cpu < num_cpus
            ):
                warnings.append(f"Skipping invalid CPU sample {path}:{line_number}")
                continue
            samples.append(
                {
                    "start": start,
                    "end": end,
                    "pid": _as_int(row.get("pid")),
                    "tid": _as_int(row.get("tid")),
                    "cpu": cpu,
                    "cpu_time": cpu_time,
                    "migrated": _parse_bool(row.get("migrated")),
                }
            )
    return samples


def _strict_float(
    row: dict[str, Any], field: str, path: Path, line_number: int
) -> float:
    try:
        value = float(row.get(field, ""))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid {field} in {path} at line {line_number}: {row.get(field)!r}"
        ) from error
    if not math.isfinite(value):
        raise ValueError(
            f"Invalid {field} in {path} at line {line_number}: must be finite"
        )
    return value


def _aggregate_cpu_csv(path: Path, bin_s: float, num_cpus: int) -> dict[str, Any]:
    cpu_times: dict[tuple[int, int], float] = defaultdict(float)
    threads: dict[tuple[int, int], set[tuple[int | None, int | None]]] = defaultdict(
        set
    )
    migrations: dict[tuple[int, int], int] = defaultdict(int)
    median_estimator = P2Median()
    source_union = IntervalUnionAccumulator()
    start: float | None = None
    end: float | None = None
    sample_count = 0
    migrated_count = 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "interval_start",
            "interval_end",
            "pid",
            "tid",
            "cpu",
            "cpu_time_s",
            "migrated",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CPU columns in {path}: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            interval_start = _strict_float(row, "interval_start", path, line_number)
            interval_end = _strict_float(row, "interval_end", path, line_number)
            cpu_time = _strict_float(row, "cpu_time_s", path, line_number)
            cpu = _as_int(row.get("cpu"))
            if interval_end <= interval_start:
                raise ValueError(
                    f"Invalid CPU interval in {path} at line {line_number}: "
                    "interval_end must be greater than interval_start"
                )
            if cpu_time < 0:
                raise ValueError(
                    f"Invalid cpu_time_s in {path} at line {line_number}: "
                    "must be non-negative"
                )
            if cpu is None or not 0 <= cpu < num_cpus:
                raise ValueError(
                    f"Invalid cpu in {path} at line {line_number}: {row.get('cpu')!r}"
                )
            duration = interval_end - interval_start
            first_bin = math.floor(interval_start / bin_s)
            last_bin = math.floor(math.nextafter(interval_end, -math.inf) / bin_s)
            for index in range(first_bin, last_bin + 1):
                bin_start = _timestamp(index, bin_s)
                overlap = _overlap(
                    interval_start, interval_end, bin_start, bin_start + bin_s
                )
                if overlap <= 0:
                    continue
                key = (index, cpu)
                cpu_times[key] += cpu_time * overlap / duration
                threads[key].add((_as_int(row.get("pid")), _as_int(row.get("tid"))))
            migrated = _parse_bool(row.get("migrated"))
            if migrated:
                migrations[(last_bin, cpu)] += 1
                migrated_count += 1
            sample_count += 1
            median_estimator.update(duration)
            source_union.add(interval_start, interval_end)
            start = interval_start if start is None else min(start, interval_start)
            end = interval_end if end is None else max(end, interval_end)
    return {
        "cpu_times": cpu_times,
        "threads": threads,
        "migrations": migrations,
        "sample_count": sample_count,
        "migrated_count": migrated_count,
        "median_estimator": median_estimator,
        "source_union": source_union,
        "start": start,
        "end": end,
    }


def _cpu_rows_from_aggregate(
    aggregate: dict[str, Any],
    *,
    window_start: float,
    window_end: float,
    bin_s: float,
    num_cpus: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    over_100_bins = 0
    for index in _bin_indexes(window_start, window_end, bin_s):
        timestamp = _timestamp(index, bin_s)
        for cpu in range(num_cpus):
            key = (index, cpu)
            util_pct = aggregate["cpu_times"].get(key, 0.0) / bin_s * 100.0
            if util_pct > 100.0:
                over_100_bins += 1
            rows.append(
                {
                    "timestamp": timestamp,
                    "datetime": _datetime(timestamp),
                    "cpu": cpu,
                    "util_pct": util_pct,
                    "thread_count": len(aggregate["threads"].get(key, set())),
                    "migration_count": aggregate["migrations"].get(key, 0),
                }
            )
    sample_count = aggregate["sample_count"]
    return rows, {
        "start": aggregate["start"],
        "end": aggregate["end"],
        "migration_ratio": (
            aggregate["migrated_count"] / sample_count if sample_count else 0.0
        ),
        "over_100_bins": over_100_bins,
    }


def _phase_key(event: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        json.dumps(event.get(field), sort_keys=True) for field in PHASE_KEY_FIELDS
    )


def _pair_events(
    events: list[dict[str, Any]],
    *,
    component: str,
    default_phase: str,
    runner_events: bool = False,
) -> tuple[list[PhaseWindow], list[dict[str, Any]], list[dict[str, Any]]]:
    queues: dict[tuple[str, tuple[str, ...]], deque[dict[str, Any]]] = defaultdict(
        deque
    )
    windows: list[PhaseWindow] = []
    invalid_starts: list[dict[str, Any]] = []
    unmatched_ends: list[dict[str, Any]] = []

    def diagnostic(
        event: dict[str, Any], key: tuple[str, tuple[str, ...]]
    ) -> dict[str, Any]:
        record = {
            name: value for name, value in event.items() if not name.startswith("_")
        }
        event_component = component
        if runner_events and key[0] == "actor_training":
            event_component = "training"
        record.update(
            {
                "component": event_component,
                "wall": (
                    wall_ns / 1_000_000_000.0
                    if (wall_ns := _as_int(event.get("wall_ns"))) is not None
                    else None
                ),
                "source": str(event.get("_source", "")),
                "key": [key[0], *key[1]],
            }
        )
        return record

    for event in events:
        event_name = str(event.get("event", ""))
        if runner_events:
            if event_name not in {
                "step.start",
                "step.end",
                "actor_training.start",
                "actor_training.end",
            }:
                continue
            phase, boundary = event_name.rsplit(".", 1)
            pid = _as_int(event.get("pid"))
            session_identity = (
                f"pid:{pid}"
                if pid is not None
                else (
                    f"legacy:{event.get('_source', '')}:"
                    f"{event.get('_legacy_session', 0)}"
                )
            )
            key = (
                phase,
                (session_identity, json.dumps(event.get("step"))),
            )
        else:
            if event_name not in {"start", "end"}:
                continue
            boundary = event_name
            phase = str(event.get("phase") or default_phase)
            key = (phase, _phase_key(event))
        if boundary == "start":
            if _as_int(event.get("wall_ns")) is None:
                invalid_starts.append(diagnostic(event, key))
                continue
            queues[key].append(event)
            continue
        if not queues[key]:
            unmatched_ends.append(diagnostic(event, key))
            continue
        start_event = queues[key][0]
        start_ns = _as_int(start_event.get("wall_ns"))
        end_ns = _as_int(event.get("wall_ns"))
        if start_ns is None or end_ns is None or end_ns <= start_ns:
            unmatched_ends.append(diagnostic(event, key))
            continue
        queues[key].popleft()
        window_component = (
            "training" if runner_events and phase == "actor_training" else component
        )
        windows.append(
            PhaseWindow(
                phase=phase,
                step=start_event.get("step", start_event.get("chunk_step")),
                component=window_component,
                rank=start_event.get("rank"),
                start=start_ns / 1_000_000_000.0,
                end=end_ns / 1_000_000_000.0,
                source=str(start_event.get("_source", "")),
                pid=_as_int(start_event.get("pid")),
                session_key=key[1][0],
            )
        )
    unmatched_starts = invalid_starts + [
        diagnostic(event, key) for key, queue in queues.items() for event in queue
    ]
    return windows, unmatched_starts, unmatched_ends


def _events_from_path(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    paths = [path] if path.is_file() else sorted(path.glob("*.jsonl"))
    events: list[dict[str, Any]] = []
    for event_path in paths:
        for event in _read_jsonl(event_path):
            event["_source"] = str(event_path)
            events.append(event)
    return events


def _annotate_legacy_runner_sessions(events: list[dict[str, Any]]) -> None:
    generations: dict[str, int] = defaultdict(int)
    last_step_start: dict[str, int] = {}
    for event in events:
        if _as_int(event.get("pid")) is not None:
            continue
        source = str(event.get("_source", ""))
        if event.get("event") == "step.start":
            step = _as_int(event.get("step"))
            previous = last_step_start.get(source)
            if step is not None and previous is not None and step <= previous:
                generations[source] += 1
            if step is not None:
                last_step_start[source] = step
        event["_legacy_session"] = generations[source]


def _resolve_timestamp_dir(run_dir: Path, name: str) -> Path:
    direct = run_dir / name
    return direct if direct.exists() else run_dir / "logs" / name


def _phase_source_coverage(
    path: Path,
    events: list[dict[str, Any]],
    windows: list[PhaseWindow],
) -> dict[str, Any]:
    timestamps = [
        wall_ns / 1_000_000_000.0
        for event in events
        if (wall_ns := _as_int(event.get("wall_ns"))) is not None
    ]
    record: dict[str, Any] = {"path": str(path)}
    if not timestamps:
        record.update(
            {
                "start": None,
                "end": None,
                "effective_coverage_ratio": 0.0,
                "warning": (
                    f"Missing phase source: {path}"
                    if not path.exists()
                    else f"No valid phase timestamps: {path}"
                ),
            }
        )
        return record
    start = min(timestamps)
    end = max(timestamps)
    duration = end - start
    record.update(
        {
            "start": start,
            "end": end,
            "effective_coverage_ratio": (
                _interval_pair_union_duration(
                    (window.start, window.end) for window in windows
                )
                / duration
                if duration > 0
                else 0.0
            ),
        }
    )
    return record


def load_phase_windows(run_dir: Path) -> tuple[list[PhaseWindow], dict[str, Any]]:
    """Pair runner, generation, and environment timestamp streams FIFO."""
    profile_dir = run_dir / "resource_profile"
    runner_path = profile_dir / "runner_events.jsonl"
    generation_path = _resolve_timestamp_dir(run_dir, "rollout_generation_timestamps")
    env_path = _resolve_timestamp_dir(run_dir, "env_sim_timestamps")
    windows: list[PhaseWindow] = []
    unmatched_starts: list[dict[str, Any]] = []
    unmatched_ends: list[dict[str, Any]] = []
    source_coverage: dict[str, dict[str, Any]] = {}
    sources = (
        ("runner", runner_path, "runner", "step", True),
        (
            "generation",
            generation_path,
            "generation",
            "rollout_generation",
            False,
        ),
        ("env", env_path, "env", "env_simulation", False),
    )
    for source_name, path, component, default_phase, is_runner in sources:
        events = _events_from_path(path)
        if is_runner:
            _annotate_legacy_runner_sessions(events)
        paired, starts, ends = _pair_events(
            events,
            component=component,
            default_phase=default_phase,
            runner_events=is_runner,
        )
        windows.extend(paired)
        unmatched_starts.extend(starts)
        unmatched_ends.extend(ends)
        source_coverage[source_name] = _phase_source_coverage(path, events, paired)
    windows.sort(key=lambda window: (window.start, window.end, window.phase))
    return windows, {
        "unmatched_starts": unmatched_starts,
        "unmatched_ends": unmatched_ends,
        "sources": source_coverage,
    }


def _bin_indexes(window_start: float, window_end: float, bin_s: float) -> range:
    first = math.floor(window_start / bin_s)
    stop = math.ceil(window_end / bin_s)
    if stop <= first:
        stop = first + 1
    return range(first, stop)


def _timestamp(index: int, bin_s: float) -> float:
    return index * bin_s


def _datetime(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(
        timespec="microseconds"
    )


def _overlap(start: float, end: float, bin_start: float, bin_end: float) -> float:
    return max(0.0, min(end, bin_end) - max(start, bin_start))


def _write_csv(
    path: Path, fields: tuple[str, ...], rows: Iterable[dict[str, Any]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _cpu_rows(
    samples: list[dict[str, Any]],
    *,
    window_start: float,
    window_end: float,
    bin_s: float,
    num_cpus: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cpu_times: dict[tuple[int, int], float] = defaultdict(float)
    threads: dict[tuple[int, int], set[tuple[int | None, int | None]]] = defaultdict(
        set
    )
    migrations: dict[tuple[int, int], int] = defaultdict(int)
    indexes = _bin_indexes(window_start, window_end, bin_s)
    for sample in samples:
        duration = sample["end"] - sample["start"]
        for index in indexes:
            bin_start = _timestamp(index, bin_s)
            overlap = _overlap(
                sample["start"], sample["end"], bin_start, bin_start + bin_s
            )
            if overlap <= 0:
                continue
            key = (index, sample["cpu"])
            cpu_times[key] += sample["cpu_time"] * overlap / duration
            threads[key].add((sample["pid"], sample["tid"]))
        if sample["migrated"]:
            migration_timestamp = math.nextafter(sample["end"], -math.inf)
            migration_index = math.floor(migration_timestamp / bin_s)
            if migration_index in indexes:
                migrations[(migration_index, sample["cpu"])] += 1

    rows: list[dict[str, Any]] = []
    over_100_bins = 0
    for index in indexes:
        timestamp = _timestamp(index, bin_s)
        for cpu in range(num_cpus):
            key = (index, cpu)
            util_pct = cpu_times[key] / bin_s * 100.0
            if util_pct > 100.0:
                over_100_bins += 1
            rows.append(
                {
                    "timestamp": timestamp,
                    "datetime": _datetime(timestamp),
                    "cpu": cpu,
                    "util_pct": util_pct,
                    "thread_count": len(threads[key]),
                    "migration_count": migrations[key],
                }
            )
    migrated_count = sum(bool(sample["migrated"]) for sample in samples)
    return rows, {
        "start": min((sample["start"] for sample in samples), default=None),
        "end": max((sample["end"] for sample in samples), default=None),
        "migration_ratio": migrated_count / len(samples) if samples else 0.0,
        "over_100_bins": over_100_bins,
    }


def _gpu_worker_rows(
    intervals: list[KernelInterval],
    worker_devices: list[dict[str, Any]],
    *,
    window_start: float,
    window_end: float,
    bin_s: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[KernelInterval]] = defaultdict(list)
    for interval in intervals:
        grouped[
            (
                interval.worker,
                interval.component,
                interval.rank,
                interval.gpu_uuid,
                interval.gpu_label,
            )
        ].append(interval)
    for device in worker_devices:
        grouped.setdefault(
            (
                device["worker"],
                device["component"],
                device["rank"],
                device["gpu_uuid"],
                device["gpu_label"],
            ),
            [],
        )

    rows: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: tuple(str(value) for value in item)):
        worker, component, rank, gpu_uuid, gpu_label = key
        device_intervals = grouped[key]
        for index in _bin_indexes(window_start, window_end, bin_s):
            bin_start = _timestamp(index, bin_s)
            bin_end = bin_start + bin_s
            overlapping = [
                interval
                for interval in device_intervals
                if interval.end_s > bin_start and interval.start_s < bin_end
            ]
            occupancy_weight = 0.0
            occupancy_duration = 0.0
            for interval in overlapping:
                overlap = _overlap(interval.start_s, interval.end_s, bin_start, bin_end)
                if interval.occupancy_pct is not None:
                    occupancy_weight += interval.occupancy_pct * overlap
                    occupancy_duration += overlap
            rows.append(
                {
                    "timestamp": bin_start,
                    "datetime": _datetime(bin_start),
                    "worker": worker,
                    "component": component,
                    "rank": "" if rank is None else rank,
                    "gpu_uuid": gpu_uuid,
                    "gpu_label": gpu_label,
                    "kernel_busy_pct": min(
                        100.0,
                        interval_union_duration(overlapping, bin_start, bin_end)
                        / bin_s
                        * 100.0,
                    ),
                    "est_sm_occupancy_pct": (
                        occupancy_weight / occupancy_duration
                        if occupancy_duration > 0
                        else math.nan
                    ),
                    "kernel_count": len(overlapping),
                }
            )
    return rows


def _gpu_device_rows(
    intervals: list[KernelInterval],
    worker_devices: list[dict[str, Any]],
    *,
    window_start: float,
    window_end: float,
    bin_s: float,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[KernelInterval]] = defaultdict(list)
    labels: dict[str, str] = {}
    for interval in intervals:
        grouped[interval.gpu_uuid].append(interval)
        labels.setdefault(interval.gpu_uuid, interval.gpu_label)
    for device in worker_devices:
        grouped.setdefault(device["gpu_uuid"], [])
        labels.setdefault(device["gpu_uuid"], device["gpu_label"])
    rows: list[dict[str, Any]] = []
    for gpu_uuid, device_intervals in sorted(grouped.items()):
        gpu_label = labels[gpu_uuid]
        for index in _bin_indexes(window_start, window_end, bin_s):
            bin_start = _timestamp(index, bin_s)
            bin_end = bin_start + bin_s
            rows.append(
                {
                    "timestamp": bin_start,
                    "datetime": _datetime(bin_start),
                    "gpu_uuid": gpu_uuid,
                    "gpu_label": gpu_label,
                    "kernel_busy_pct": min(
                        100.0,
                        interval_union_duration(device_intervals, bin_start, bin_end)
                        / bin_s
                        * 100.0,
                    ),
                }
            )
    return rows


def _aggregate_streaming_torch(
    torch_dir: Path,
    *,
    bin_s: float,
    clip_window: tuple[float, float] | None,
) -> tuple[
    dict[tuple[Any, ...], GpuWorkerBin],
    dict[tuple[str, int], IntervalUnionAccumulator],
    set[tuple[Any, ...]],
    dict[str, str],
    TorchParseState,
]:
    state = _new_torch_state()
    worker_bins: dict[tuple[Any, ...], GpuWorkerBin] = {}
    device_bins: dict[tuple[str, int], IntervalUnionAccumulator] = {}
    known_workers: set[tuple[Any, ...]] = set()
    device_labels: dict[str, str] = {}
    for chunk in _iter_trace_chunks(torch_dir, state):
        trace_window, chunk_kernels = _load_chunk_kernels(chunk, state)
        if clip_window is not None and trace_window is not None:
            if trace_window[0] < clip_window[0] or trace_window[1] > clip_window[1]:
                state.warnings.append(
                    f"Trace {chunk.trace_path} is outside primary resource window "
                    f"[{clip_window[0]}, {clip_window[1]}]"
                )
        for kernel in chunk_kernels:
            start = kernel.start_s
            end = kernel.end_s
            if clip_window is not None:
                start = max(start, clip_window[0])
                end = min(end, clip_window[1])
                if end <= start:
                    continue
            worker_key = (
                kernel.worker,
                kernel.component,
                kernel.rank,
                kernel.gpu_uuid,
                kernel.gpu_label,
            )
            known_workers.add(worker_key)
            device_labels.setdefault(kernel.gpu_uuid, kernel.gpu_label)
            first_bin = math.floor(start / bin_s)
            last_bin = math.floor(math.nextafter(end, -math.inf) / bin_s)
            for index in range(first_bin, last_bin + 1):
                _visit_gpu_bin()
                bin_start = _timestamp(index, bin_s)
                clipped_start = max(start, bin_start)
                clipped_end = min(end, bin_start + bin_s)
                if clipped_end <= clipped_start:
                    continue
                worker_bin_key = (*worker_key, index)
                worker_bin = worker_bins.get(worker_bin_key)
                if worker_bin is None:
                    worker_bin = GpuWorkerBin(IntervalUnionAccumulator())
                    worker_bins[worker_bin_key] = worker_bin
                worker_bin.union.add(clipped_start, clipped_end)
                overlap = clipped_end - clipped_start
                if kernel.occupancy_pct is not None:
                    worker_bin.occupancy_weighted_sum += kernel.occupancy_pct * overlap
                    worker_bin.occupancy_weight += overlap
                worker_bin.kernel_count += 1
                device_bin = device_bins.setdefault(
                    (kernel.gpu_uuid, index), IntervalUnionAccumulator()
                )
                device_bin.add(clipped_start, clipped_end)
        del chunk_kernels
    _finalize_torch_state(state)
    for mapping in state.coverage["worker_devices"]:
        worker_key = (
            mapping["worker"],
            mapping["component"],
            mapping["rank"],
            mapping["gpu_uuid"],
            mapping["gpu_label"],
        )
        known_workers.add(worker_key)
        device_labels.setdefault(mapping["gpu_uuid"], mapping["gpu_label"])
    return worker_bins, device_bins, known_workers, device_labels, state


def _streaming_gpu_rows(
    worker_bins: dict[tuple[Any, ...], GpuWorkerBin],
    device_bins: dict[tuple[str, int], IntervalUnionAccumulator],
    known_workers: set[tuple[Any, ...]],
    device_labels: dict[str, str],
    *,
    window_start: float,
    window_end: float,
    bin_s: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indexes = _bin_indexes(window_start, window_end, bin_s)
    worker_rows: list[dict[str, Any]] = []
    for worker_key in sorted(
        known_workers, key=lambda key: tuple(str(value) for value in key)
    ):
        worker, component, rank, gpu_uuid, gpu_label = worker_key
        for index in indexes:
            timestamp = _timestamp(index, bin_s)
            aggregate = worker_bins.get((*worker_key, index))
            worker_rows.append(
                {
                    "timestamp": timestamp,
                    "datetime": _datetime(timestamp),
                    "worker": worker,
                    "component": component,
                    "rank": "" if rank is None else rank,
                    "gpu_uuid": gpu_uuid,
                    "gpu_label": gpu_label,
                    "kernel_busy_pct": (
                        min(100.0, aggregate.union.duration / bin_s * 100.0)
                        if aggregate is not None
                        else 0.0
                    ),
                    "est_sm_occupancy_pct": (
                        aggregate.occupancy_weighted_sum / aggregate.occupancy_weight
                        if aggregate is not None and aggregate.occupancy_weight > 0
                        else math.nan
                    ),
                    "kernel_count": aggregate.kernel_count if aggregate else 0,
                }
            )
    device_rows: list[dict[str, Any]] = []
    for gpu_uuid, gpu_label in sorted(device_labels.items()):
        for index in indexes:
            timestamp = _timestamp(index, bin_s)
            aggregate = device_bins.get((gpu_uuid, index))
            device_rows.append(
                {
                    "timestamp": timestamp,
                    "datetime": _datetime(timestamp),
                    "gpu_uuid": gpu_uuid,
                    "gpu_label": gpu_label,
                    "kernel_busy_pct": (
                        min(100.0, aggregate.duration / bin_s * 100.0)
                        if aggregate is not None
                        else 0.0
                    ),
                }
            )
    return worker_rows, device_rows


def _phase_rows(windows: list[PhaseWindow]) -> list[dict[str, Any]]:
    return [
        {
            "phase": window.phase,
            "step": "" if window.step is None else window.step,
            "component": window.component,
            "rank": "" if window.rank is None else window.rank,
            "start": window.start,
            "end": window.end,
            "source": window.source,
        }
        for window in windows
    ]


def _clip_phase_windows(
    windows: list[PhaseWindow],
    window_start: float,
    window_end: float,
    warnings: list[str],
) -> list[PhaseWindow]:
    clipped: list[PhaseWindow] = []
    for window in windows:
        start = max(window.start, window_start)
        end = min(window.end, window_end)
        if start != window.start or end != window.end:
            warnings.append(
                f"Phase {window.phase} from {window.source} is outside primary "
                f"resource window [{window_start}, {window_end}]"
            )
        if end <= start:
            continue
        clipped.append(
            PhaseWindow(
                phase=window.phase,
                step=window.step,
                component=window.component,
                rank=window.rank,
                start=start,
                end=end,
                source=window.source,
                pid=window.pid,
                session_key=window.session_key,
            )
        )
    return clipped


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _publish_generation(profile_dir: Path, staging_dir: Path) -> None:
    generation_id = staging_dir.name.removeprefix(".resource_profile_generation_")
    derived_path = profile_dir / "derived"
    backup_path = profile_dir / f".resource_profile_backup_{generation_id}"
    staged_derived = staging_dir / "derived"
    staged_metadata = staging_dir / "metadata.json"
    had_derived = derived_path.exists()
    published_derived = False
    try:
        if had_derived:
            os.replace(derived_path, backup_path)
        os.replace(staged_derived, derived_path)
        published_derived = True
        os.replace(staged_metadata, profile_dir / "metadata.json")
    except Exception:
        if published_derived and derived_path.exists():
            shutil.rmtree(derived_path)
        if backup_path.exists():
            os.replace(backup_path, derived_path)
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    shutil.rmtree(backup_path, ignore_errors=True)
    shutil.rmtree(staging_dir, ignore_errors=True)


def process_run(
    run_dir: str | Path,
    *,
    bin_s: float = 1.0,
    num_cpus: int = 112,
) -> dict[str, Any]:
    """Generate normalized resource-profile CSVs and coverage metadata."""
    run_dir = Path(run_dir).resolve()
    if not isinstance(bin_s, (int, float)) or not math.isfinite(bin_s) or bin_s <= 0:
        raise ValueError("bin_s must be finite and positive")
    if isinstance(num_cpus, bool) or not isinstance(num_cpus, int) or num_cpus <= 0:
        raise ValueError("num_cpus must be a positive integer")
    bin_s = float(bin_s)
    profile_dir = run_dir / "resource_profile"
    cpu_path = profile_dir / "cpu" / "thread_core_samples.csv"
    torch_dir = profile_dir / "torch"
    if not cpu_path.is_file():
        raise FileNotFoundError(f"Required CPU input not found: {cpu_path}")
    if not torch_dir.is_dir():
        raise FileNotFoundError(
            f"Required torch input directory not found: {torch_dir}"
        )

    cpu_aggregate = _aggregate_cpu_csv(cpu_path, bin_s, num_cpus)
    phase_windows, phase_coverage = load_phase_windows(run_dir)
    primary_window: tuple[float, float] | None = None
    if cpu_aggregate["start"] is not None and cpu_aggregate["end"] is not None:
        primary_window = (cpu_aggregate["start"], cpu_aggregate["end"])
    runner_steps = [
        window
        for window in phase_windows
        if window.phase == "step" and window.component == "runner"
    ]
    if primary_window is None and runner_steps:
        steps_by_session: dict[str, list[PhaseWindow]] = defaultdict(list)
        for window in runner_steps:
            steps_by_session[window.session_key].append(window)
        latest_session = max(
            steps_by_session.values(),
            key=lambda windows: max(item.end for item in windows),
        )
        primary_window = (
            min(window.start for window in latest_session),
            max(window.end for window in latest_session),
        )
    (
        worker_bins,
        device_bins,
        known_workers,
        device_labels,
        torch_state,
    ) = _aggregate_streaming_torch(
        torch_dir,
        bin_s=bin_s,
        clip_window=primary_window,
    )
    warnings = list(torch_state.warnings)
    torch_coverage = torch_state.coverage
    if primary_window is not None:
        window_start, window_end = primary_window
        output_phases = _clip_phase_windows(
            phase_windows, window_start, window_end, warnings
        )
    else:
        torch_window = _effective_coverage(
            (block["start"], block["end"]) for block in torch_coverage["trace_blocks"]
        )
        if torch_window["start"] is None or torch_window["end"] is None:
            raise ValueError(
                "No valid CPU, runner step, or torch resource window found"
            )
        window_start = torch_window["start"]
        window_end = torch_window["end"]
        output_phases = _clip_phase_windows(
            phase_windows, window_start, window_end, warnings
        )
    cpu_rows, cpu_coverage = _cpu_rows_from_aggregate(
        cpu_aggregate,
        window_start=window_start,
        window_end=window_end,
        bin_s=bin_s,
        num_cpus=num_cpus,
    )
    gpu_worker_rows, gpu_device_rows = _streaming_gpu_rows(
        worker_bins,
        device_bins,
        known_workers,
        device_labels,
        window_start=window_start,
        window_end=window_end,
        bin_s=bin_s,
    )
    worker_devices = torch_coverage["worker_devices"]
    cpu_span = (
        cpu_aggregate["end"] - cpu_aggregate["start"]
        if cpu_aggregate["start"] is not None and cpu_aggregate["end"] is not None
        else 0.0
    )
    cpu_source = {
        "path": str(cpu_path),
        "start": cpu_aggregate["start"],
        "end": cpu_aggregate["end"],
        "effective_coverage_ratio": (
            cpu_aggregate["source_union"].duration / cpu_span if cpu_span > 0 else 0.0
        ),
    }
    if not cpu_aggregate["sample_count"]:
        cpu_source["warning"] = f"No valid CPU samples: {cpu_path}"
    torch_source = {
        "path": str(torch_dir),
        **_effective_coverage(
            (block["start"], block["end"]) for block in torch_coverage["trace_blocks"]
        ),
    }
    if not torch_coverage["trace_blocks"]:
        torch_source["warning"] = f"No torch trace coverage: {torch_dir}"
    source_coverage = {
        "cpu": cpu_source,
        "torch": torch_source,
        **phase_coverage["sources"],
    }
    coverage = {
        "cpu": cpu_coverage,
        "torch": torch_coverage,
        "phases": phase_coverage,
        "sources": source_coverage,
        "warnings": warnings,
    }

    generation_id = uuid4().hex
    staging_dir = profile_dir / f".resource_profile_generation_{generation_id}"
    derived_dir = staging_dir / "derived"
    derived_dir.mkdir(parents=True, exist_ok=False)
    try:
        _write_csv(derived_dir / "cpu_core_1s.csv", CPU_FIELDS, cpu_rows)
        _write_csv(
            derived_dir / "gpu_worker_1s.csv", GPU_WORKER_FIELDS, gpu_worker_rows
        )
        _write_csv(
            derived_dir / "gpu_device_1s.csv", GPU_DEVICE_FIELDS, gpu_device_rows
        )
        _write_csv(
            derived_dir / "phase_windows.csv", PHASE_FIELDS, _phase_rows(output_phases)
        )
        _write_json(derived_dir / "coverage.json", coverage)

        generation_path = _resolve_timestamp_dir(
            run_dir, "rollout_generation_timestamps"
        )
        env_path = _resolve_timestamp_dir(run_dir, "env_sim_timestamps")
        metadata = {
            "run_dir": str(run_dir),
            "bin_s": bin_s,
            "num_cpus": num_cpus,
            "generated_at": datetime.now(tz=timezone.utc).isoformat(
                timespec="microseconds"
            ),
            "cpu_sample_interval_s": cpu_aggregate["median_estimator"].result(),
            "cpu_sample_interval_method": "p2_median",
            "workers": torch_coverage["workers"],
            "gpu_devices": worker_devices,
            "total_trace_bytes": torch_coverage["total_trace_bytes"],
            "sources": {
                "cpu": str(cpu_path),
                "torch": str(torch_dir),
                "runner_events": str(profile_dir / "runner_events.jsonl"),
                "rollout_generation_timestamps": str(generation_path),
                "env_sim_timestamps": str(env_path),
            },
        }
        _write_json(staging_dir / "metadata.json", metadata)
        _publish_generation(profile_dir, staging_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    return coverage


def generate_resource_profile(
    run_dir: str | Path,
    *,
    bin_s: float = 1.0,
    num_cpus: int = 112,
) -> dict[str, Any]:
    """Compatibility wrapper around the streaming run processor."""
    return process_run(run_dir, bin_s=bin_s, num_cpus=num_cpus)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate async GIPO resource profiling artifacts."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--bin-s", type=float, default=1.0)
    parser.add_argument("--num-cpus", type=int, default=112)
    return parser.parse_args()


def main() -> None:
    """Run the resource profile aggregation CLI."""
    args = _parse_args()
    try:
        generate_resource_profile(
            args.run_dir,
            bin_s=args.bin_s,
            num_cpus=args.num_cpus,
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"async_gipo_resource_profile: error: {error}") from error


if __name__ == "__main__":
    main()
