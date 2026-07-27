"""Plot Gantt-style timelines from LIBERO latency benchmark step events."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-libero-gantt")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def _load_events(path: Path) -> list[dict[str, Any]]:
    events = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            if event.get("start_time_s") is None or event.get("end_time_s") is None:
                continue
            events.append(event)
    return events


def _task_color(task_id: int) -> tuple[float, float, float, float]:
    cmap = plt.get_cmap("tab20")
    return cmap(task_id % cmap.N)


def _plot_schedule(
    ax: Any,
    events: list[dict[str, Any]],
    title: str,
    *,
    x_max_s: float,
) -> None:
    cores = sorted({int(event["core_index"]) for event in events})
    y_by_core = {core_index: row for row, core_index in enumerate(cores)}
    for event in events:
        core_index = int(event["core_index"])
        start = float(event["start_time_s"])
        end = float(event["end_time_s"])
        duration = max(end - start, 0.0)
        task_id = int(event["task_id"])
        step_index = int(event["task_step_index"])
        ax.broken_barh(
            [(start, duration)],
            (y_by_core[core_index] - 0.38, 0.76),
            facecolors=_task_color(task_id),
            edgecolors="none",
            alpha=0.95 if step_index == 0 else 0.72,
        )
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.set_ylabel("core")
    ax.set_yticks([y_by_core[core] for core in cores])
    ax.set_yticklabels([str(core) for core in cores], fontsize=7)
    ax.grid(axis="x", color="#d0d0d0", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_xlim(0.0, x_max_s)


def plot_gantt(
    baseline_events_path: Path,
    warmup_events_path: Path,
    output_path: Path,
) -> None:
    baseline_events = _load_events(baseline_events_path)
    warmup_events = _load_events(warmup_events_path)
    if not baseline_events:
        raise ValueError(f"no timed events loaded from {baseline_events_path}")
    if not warmup_events:
        raise ValueError(f"no timed events loaded from {warmup_events_path}")

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(18, 12),
        sharex=False,
        constrained_layout=True,
    )
    x_max_s = 1.02 * max(
        max(float(event["end_time_s"]) for event in baseline_events),
        max(float(event["end_time_s"]) for event in warmup_events),
    )
    _plot_schedule(axes[0], baseline_events, "task_id_baseline", x_max_s=x_max_s)
    _plot_schedule(
        axes[1],
        warmup_events,
        "warmup_minmax_binpack",
        x_max_s=x_max_s,
    )
    axes[1].set_xlabel("elapsed time (s)")
    fig.suptitle(
        "LIBERO Step Timeline: Baseline vs Warmup Min-Max Binpack",
        fontsize=15,
        fontweight="bold",
    )
    fig.legend(
        handles=[
            Patch(facecolor="#555555", alpha=0.95, label="step 0 opacity"),
            Patch(facecolor="#555555", alpha=0.72, label="later steps opacity"),
        ],
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot Gantt-style LIBERO latency benchmark timelines."
    )
    parser.add_argument("--baseline-events", type=Path, required=True)
    parser.add_argument("--warmup-events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    plot_gantt(args.baseline_events, args.warmup_events, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
