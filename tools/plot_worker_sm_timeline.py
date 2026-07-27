#!/usr/bin/env python3
"""Plot per-worker estimated SM occupancy on a shared torch-profiler timeline."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Event:
    name: str
    category: str
    ts_us: float
    dur_us: float
    args: dict[str, Any]

    @property
    def end_us(self) -> float:
        return self.ts_us + self.dur_us


def load_events(path: Path) -> list[Event]:
    data = json.loads(path.read_text())
    events: list[Event] = []
    for raw in data.get("traceEvents", []):
        if raw.get("ph") != "X" or "ts" not in raw or "dur" not in raw:
            continue
        events.append(
            Event(
                name=str(raw.get("name", "")),
                category=str(raw.get("cat", "")),
                ts_us=float(raw["ts"]),
                dur_us=float(raw["dur"]),
                args=dict(raw.get("args", {})),
            )
        )
    return events


def overlap_us(left: Event, right: Event) -> float:
    return max(0.0, min(left.end_us, right.end_us) - max(left.ts_us, right.ts_us))


def weighted_mean(samples: list[tuple[float, float]]) -> float:
    total = sum(weight for _, weight in samples)
    if total <= 0:
        return 0.0
    return sum(value * weight for value, weight in samples) / total


def infer_phases(trace_dir: Path) -> list[str]:
    if trace_dir.name.startswith("env_rank"):
        return ["env.step"]
    if trace_dir.name.startswith("rollout_rank"):
        return ["generation"]
    if trace_dir.name.startswith("rank"):
        return ["fwd", "loss", "backward", "optimizer.step"]
    raise ValueError(f"Unsupported trace directory: {trace_dir}")


def display_phase(phase: str) -> str:
    if phase in {"fwd", "loss", "backward", "optimizer.step"}:
        return "training"
    return phase


def collect_points(torch_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    trace_dirs = sorted(
        path
        for path in torch_dir.iterdir()
        if path.is_dir()
        and (
            path.name.startswith("env_rank")
            or path.name.startswith("rollout_rank")
            or path.name.startswith("rank")
        )
    )
    global_start_us: float | None = None
    raw_rows: list[dict[str, Any]] = []
    for trace_dir in trace_dirs:
        phases = infer_phases(trace_dir)
        worker = trace_dir.name
        trace_files = sorted(trace_dir.glob("*.pt.trace.json"))
        if not trace_files:
            continue
        events = load_events(trace_files[0])
        kernels = [event for event in events if event.category == "kernel"]
        for phase in phases:
            phase_events = [event for event in events if event.name == phase]
            for index, phase_event in enumerate(phase_events):
                occupancy_samples: list[tuple[float, float]] = []
                kernel_overlap = 0.0
                for kernel in kernels:
                    overlap = overlap_us(phase_event, kernel)
                    if overlap <= 0:
                        continue
                    kernel_overlap += overlap
                    occupancy = kernel.args.get("est. achieved occupancy %")
                    if occupancy is not None:
                        occupancy_samples.append((float(occupancy), overlap))
                row = {
                    "worker": worker,
                    "phase": display_phase(phase),
                    "phase_detail": phase,
                    "step_index": index,
                    "start_us": phase_event.ts_us,
                    "duration_us": phase_event.dur_us,
                    "gpu_busy_pct": 0.0
                    if phase_event.dur_us <= 0
                    else 100.0 * kernel_overlap / phase_event.dur_us,
                    "est_sm_occupancy_pct": weighted_mean(occupancy_samples),
                }
                raw_rows.append(row)
                global_start_us = (
                    phase_event.ts_us
                    if global_start_us is None
                    else min(global_start_us, phase_event.ts_us)
                )
    if global_start_us is None:
        return []
    for row in raw_rows:
        row = dict(row)
        row["time_s"] = (float(row["start_us"]) - global_start_us) / 1_000_000.0
        rows.append(row)
    return rows


def write_points_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "worker",
        "phase",
        "phase_detail",
        "step_index",
        "time_s",
        "duration_us",
        "gpu_busy_pct",
        "est_sm_occupancy_pct",
    ]
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fieldnames} for row in rows)


def plot_timeline(
    rows: list[dict[str, Any]],
    output_pdf: Path,
    output_png: Path,
    *,
    metric: str,
    ylabel: str,
    ylim: tuple[float, float],
) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(8.0, 3.0))
    colors = {
        "generation": "#EA5455",
        "env.step": "#2D4059",
        "training": "#16A3A6",
    }
    markers = {
        "generation": "o",
        "env.step": "s",
        "training": "^",
    }
    for phase in ("generation", "env.step", "training"):
        phase_rows = [row for row in rows if row["phase"] == phase]
        workers = sorted({str(row["worker"]) for row in phase_rows})
        for worker_idx, worker in enumerate(workers):
            worker_rows = sorted(
                (row for row in phase_rows if row["worker"] == worker),
                key=lambda row: float(row["time_s"]),
            )
            if not worker_rows:
                continue
            x = np.array([float(row["time_s"]) for row in worker_rows])
            y = np.array([float(row[metric]) for row in worker_rows])
            alpha = 0.35 if phase == "env.step" else 0.75
            label = phase if worker_idx == 0 else None
            ax.plot(
                x,
                y,
                color=colors[phase],
                marker=markers[phase],
                markersize=2.5,
                linewidth=1.0,
                alpha=alpha,
                label=label,
                zorder=10 if phase == "generation" else 5,
            )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, color="gray", alpha=0.45)
    ax.legend(
        ncol=2,
        fontsize=9,
        handletextpad=0.3,
        handlelength=1.2,
        columnspacing=0.8,
        loc="upper right",
        frameon=True,
    )
    fig.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("torch_dir", type=Path)
    parser.add_argument("--output-prefix", type=Path, default=None)
    args = parser.parse_args()

    prefix = args.output_prefix or args.torch_dir / "worker_sm_timeline"
    rows = collect_points(args.torch_dir)
    if not rows:
        raise RuntimeError(f"No env/generation phase points found under {args.torch_dir}")
    write_points_csv(rows, prefix.with_suffix(".csv"))
    plot_timeline(
        rows,
        prefix.with_suffix(".pdf"),
        prefix.with_suffix(".png"),
        metric="est_sm_occupancy_pct",
        ylabel="Estimated SM occupancy (%)",
        ylim=(-2, 40),
    )
    plot_timeline(
        rows,
        prefix.with_name(f"{prefix.name}_gpu_busy").with_suffix(".pdf"),
        prefix.with_name(f"{prefix.name}_gpu_busy").with_suffix(".png"),
        metric="gpu_busy_pct",
        ylabel="GPU busy (%)",
        ylim=(-2, 105),
    )


if __name__ == "__main__":
    main()
