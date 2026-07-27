#!/usr/bin/env python3
"""Summarize RoboCasa env/generation/training spans against CPU/GPU samples."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

TS_PREFIX_RE = re.compile(
    r"^\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6})\]"
)
METRIC_RE = re.compile(r"(?P<name>[A-Za-z_/]+)=(?P<value>[0-9.eE+-]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", required=True, type=Path)
    parser.add_argument("--num-cpus", type=float, default=112.0)
    return parser.parse_args()


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def wall_ns_to_s(wall_ns: int) -> float:
    return wall_ns / 1_000_000_000.0


def normalize_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def read_jsonl_events(directory: Path, event_filter: set[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not directory.exists():
        return events
    for path in sorted(directory.glob("*.jsonl")):
        with path.open(errors="replace") as handle:
            for raw in handle:
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if event.get("event") in event_filter and "wall_ns" in event:
                    events.append(event)
    return events


def pair_start_end_events(
    events: list[dict[str, Any]],
    key_fields: tuple[str, ...],
) -> list[tuple[float, float, dict[str, Any]]]:
    starts: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    spans: list[tuple[float, float, dict[str, Any]]] = []
    for event in sorted(events, key=lambda item: int(item["wall_ns"])):
        key = tuple(normalize_key(event.get(field)) for field in key_fields)
        if event["event"] == "start":
            starts.setdefault(key, []).append(event)
            continue
        start_list = starts.get(key)
        if not start_list:
            continue
        start = start_list.pop(0)
        start_s = wall_ns_to_s(int(start["wall_ns"]))
        end_s = wall_ns_to_s(int(event["wall_ns"]))
        if end_s > start_s:
            meta = dict(start)
            meta.update({f"end_{k}": v for k, v in event.items()})
            spans.append((start_s, end_s, meta))
    return spans


def merge_intervals(
    intervals: list[tuple[float, float, dict[str, Any]]],
) -> list[tuple[float, float]]:
    if not intervals:
        return []
    sorted_intervals = sorted((start, end) for start, end, _ in intervals)
    merged: list[tuple[float, float]] = []
    cur_start, cur_end = sorted_intervals[0]
    for start, end in sorted_intervals[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))
    return merged


def interval_duration(intervals: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in intervals)


def parse_cpu(path: Path, num_cpus: float) -> dict[str, np.ndarray]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No CPU samples found in {path}")
    out: dict[str, np.ndarray] = {
        "timestamp": np.asarray([float(row["timestamp"]) for row in rows]),
        "total_util_pct": np.asarray([float(row["total_util_pct"]) for row in rows]),
        "total_cores": np.asarray([float(row["total_cores"]) for row in rows]),
    }
    for column in ("EnvWorker_cores", "RolloutWorker_cores", "ActorWorker_cores"):
        if column in rows[0]:
            out[column] = np.asarray([float(row[column]) for row in rows])
            out[f"{column}_pct"] = out[column] / num_cpus * 100.0
    return out


def parse_gpu_timestamp(raw_ts: str) -> float:
    return datetime.strptime(raw_ts.strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()


def parse_gpu(path: Path) -> dict[str, np.ndarray]:
    timestamps: list[float] = []
    utils: list[float] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"No GPU CSV header found in {path}")
        util_column = next(
            (
                name
                for name in reader.fieldnames
                if name.strip().startswith("utilization.gpu")
            ),
            None,
        )
        if util_column is None:
            raise RuntimeError(f"No GPU utilization column found in {path}")
        grouped: dict[float, list[float]] = {}
        for row in reader:
            ts = parse_gpu_timestamp(row["timestamp"])
            grouped.setdefault(ts, []).append(float(row[util_column].strip()))
    for ts in sorted(grouped):
        timestamps.append(ts)
        utils.append(float(mean(grouped[ts])))
    return {
        "timestamp": np.asarray(timestamps),
        "gpu_util_pct": np.asarray(utils),
    }


def parse_log_metadata(path: Path) -> dict[str, float]:
    last_ts: float | None = None
    metrics: dict[str, float] = {}
    for raw in path.read_text(errors="replace").splitlines():
        line = strip_ansi(raw)
        ts_match = TS_PREFIX_RE.match(line)
        if ts_match:
            last_ts = datetime.strptime(
                ts_match.group("ts"), "%Y-%m-%d %H:%M:%S.%f"
            ).timestamp()
        for match in METRIC_RE.finditer(line):
            metrics[match.group("name")] = float(match.group("value"))
    metadata = dict(metrics)
    if last_ts is not None:
        metadata["metrics_wall_time"] = last_ts
    return metadata


def sample_mask(timestamps: np.ndarray, intervals: list[tuple[float, float]]) -> np.ndarray:
    mask = np.zeros_like(timestamps, dtype=bool)
    for start, end in intervals:
        mask |= (timestamps >= start) & (timestamps <= end)
    return mask


def describe_samples(
    timestamps: np.ndarray,
    values: np.ndarray,
    intervals: list[tuple[float, float]],
) -> dict[str, float]:
    mask = sample_mask(timestamps, intervals)
    selected = values[mask]
    if selected.size == 0:
        return {"samples": 0, "avg": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "samples": float(selected.size),
        "avg": float(np.mean(selected)),
        "p95": float(np.percentile(selected, 95)),
        "max": float(np.max(selected)),
    }


def print_phase(
    name: str,
    intervals: list[tuple[float, float]],
    cpu: dict[str, np.ndarray],
    gpu: dict[str, np.ndarray],
) -> None:
    if not intervals:
        print(f"{name}: unavailable")
        return
    cpu_stats = describe_samples(
        cpu["timestamp"], cpu["total_util_pct"], intervals
    )
    gpu_stats = describe_samples(
        gpu["timestamp"], gpu["gpu_util_pct"], intervals
    )
    print(
        f"{name}: duration={interval_duration(intervals):.3f}s "
        f"cpu_avg={cpu_stats['avg']:.2f}% cpu_p95={cpu_stats['p95']:.2f}% "
        f"cpu_max={cpu_stats['max']:.2f}% gpu_avg={gpu_stats['avg']:.2f}% "
        f"gpu_p95={gpu_stats['p95']:.2f}% gpu_max={gpu_stats['max']:.2f}% "
        f"cpu_samples={int(cpu_stats['samples'])} gpu_samples={int(gpu_stats['samples'])}"
    )


def main() -> None:
    args = parse_args()
    profile_dir = args.profile_dir.resolve()
    cpu = parse_cpu(profile_dir / "cpu_highres.csv", args.num_cpus)
    gpu = parse_gpu(profile_dir / "gpu_highres.csv")
    metadata = parse_log_metadata(profile_dir / "train.log")

    env_events = read_jsonl_events(profile_dir / "logs" / "env_sim_timestamps", {"start", "end"})
    env_spans = pair_start_end_events(
        env_events, ("rank", "epoch", "chunk_step", "stage")
    )
    env_intervals = merge_intervals(env_spans)

    generation_events = read_jsonl_events(
        profile_dir / "logs" / "rollout_generation_timestamps", {"start", "end"}
    )
    generation_spans = pair_start_end_events(
        generation_events,
        ("rank", "mode", "epoch", "chunk_step", "stage", "phase"),
    )
    generation_intervals = merge_intervals(generation_spans)

    training_intervals: list[tuple[float, float]] = []
    metrics_ts = metadata.get("metrics_wall_time")
    training_s = metadata.get("actor/run_training")
    if metrics_ts is not None and training_s is not None:
        training_intervals = [(metrics_ts - training_s, metrics_ts)]

    print(f"PROFILE_DIR={profile_dir}")
    print(
        f"ENV_SPANS={len(env_spans)} ENV_UNION_INTERVALS={len(env_intervals)} "
        f"GENERATION_SPANS={len(generation_spans)} "
        f"GENERATION_UNION_INTERVALS={len(generation_intervals)}"
    )
    print_phase("env_simulation", env_intervals, cpu, gpu)
    print_phase("model_generation", generation_intervals, cpu, gpu)
    if training_intervals:
        print_phase("actor_training", training_intervals, cpu, gpu)
    else:
        print("actor_training: unavailable")


if __name__ == "__main__":
    main()
