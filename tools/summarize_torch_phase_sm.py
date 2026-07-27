#!/usr/bin/env python3
"""Summarize torch-profiler phase GPU busy and estimated SM occupancy."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TraceEvent:
    name: str
    category: str
    ts_us: float
    dur_us: float
    args: dict[str, Any]

    @property
    def end_us(self) -> float:
        return self.ts_us + self.dur_us


def load_trace_events(path: Path) -> list[TraceEvent]:
    data = json.loads(path.read_text())
    events: list[TraceEvent] = []
    for raw in data.get("traceEvents", []):
        if raw.get("ph") != "X":
            continue
        if "ts" not in raw or "dur" not in raw:
            continue
        events.append(
            TraceEvent(
                name=str(raw.get("name", "")),
                category=str(raw.get("cat", "")),
                ts_us=float(raw["ts"]),
                dur_us=float(raw["dur"]),
                args=dict(raw.get("args", {})),
            )
        )
    return events


def overlap_us(left: TraceEvent, right: TraceEvent) -> float:
    return max(0.0, min(left.end_us, right.end_us) - max(left.ts_us, right.ts_us))


def weighted_average(values_and_weights: list[tuple[float, float]]) -> float:
    total_weight = sum(weight for _, weight in values_and_weights)
    if total_weight <= 0:
        return 0.0
    return sum(value * weight for value, weight in values_and_weights) / total_weight


def summarize_phase(events: list[TraceEvent], phase_name: str) -> dict[str, float]:
    phases = [event for event in events if event.name == phase_name]
    kernels = [event for event in events if event.category == "kernel"]
    phase_wall_us = sum(event.dur_us for event in phases)
    kernel_overlap_us = 0.0
    occupancy_samples: list[tuple[float, float]] = []

    for phase in phases:
        for kernel in kernels:
            overlap = overlap_us(phase, kernel)
            if overlap <= 0:
                continue
            kernel_overlap_us += overlap
            occupancy = kernel.args.get("est. achieved occupancy %")
            if occupancy is not None:
                occupancy_samples.append((float(occupancy), overlap))

    return {
        "phase_count": float(len(phases)),
        "phase_wall_us": phase_wall_us,
        "kernel_overlap_us": kernel_overlap_us,
        "gpu_busy_pct": 0.0
        if phase_wall_us <= 0
        else 100.0 * kernel_overlap_us / phase_wall_us,
        "est_sm_occupancy_pct": weighted_average(occupancy_samples),
    }


def infer_phase_name(trace_dir: Path) -> str:
    if trace_dir.name.startswith("env_rank"):
        return "env.step"
    if trace_dir.name.startswith("rollout_rank"):
        return "generation"
    raise ValueError(f"Cannot infer phase name from directory: {trace_dir}")


def summarize_trace_dir(trace_dir: Path, phase_name: str | None = None) -> list[dict[str, Any]]:
    phase = phase_name or infer_phase_name(trace_dir)
    rows: list[dict[str, Any]] = []
    for trace_path in sorted(trace_dir.glob("*.pt.trace.json")):
        summary = summarize_phase(load_trace_events(trace_path), phase)
        rows.append(
            {
                "trace_dir": str(trace_dir),
                "trace_file": trace_path.name,
                "phase": phase,
                **summary,
            }
        )
    return rows


def find_phase_trace_dirs(torch_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in torch_dir.iterdir()
        if path.is_dir()
        and (path.name.startswith("env_rank") or path.name.startswith("rollout_rank"))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("torch_dir", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for trace_dir in find_phase_trace_dirs(args.torch_dir):
        rows.extend(summarize_trace_dir(trace_dir))

    output = args.output or args.torch_dir / "env_generation_sm_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trace_dir",
        "trace_file",
        "phase",
        "phase_count",
        "phase_wall_us",
        "kernel_overlap_us",
        "gpu_busy_pct",
        "est_sm_occupancy_pct",
    ]
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
