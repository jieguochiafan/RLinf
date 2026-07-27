#!/usr/bin/env python3
"""Plot worker host RSS and per-process GPU memory timelines."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROLE_COLORS = {
    "EnvWorker": "#FFD460",
    "RolloutWorker": "#EA5455",
    "ActorWorker": "#2D4059",
    "CompileWorker": "#16A3A6",
    "Main": "#76BA99",
    "raylet": "#6C5B7B",
    "gcs_server": "#B83B5E",
    "Other": "darkgray",
}


def read_memory_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def aggregate_by_role(
    rows: list[dict[str, str]],
    *,
    metric: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    by_time_role: dict[tuple[float, str], float] = defaultdict(float)
    for row in rows:
        raw_value = row.get(metric, "")
        if raw_value in {"", None}:
            continue
        try:
            timestamp = float(row["timestamp"])
            value = float(raw_value)
        except (KeyError, ValueError):
            continue
        by_time_role[(timestamp, row.get("role", "Other") or "Other")] += value

    by_role: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (timestamp, role), value in sorted(by_time_role.items()):
        by_role[role].append((timestamp, value))

    if not by_role:
        return {}
    t0 = min(timestamp for points in by_role.values() for timestamp, _ in points)
    return {
        role: (
            np.asarray([timestamp - t0 for timestamp, _ in points], dtype=float),
            np.asarray([value for _, value in points], dtype=float),
        )
        for role, points in by_role.items()
    }


def top_workers(
    rows: list[dict[str, str]],
    *,
    metric: str,
    limit: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    by_pid: dict[str, list[tuple[float, float]]] = defaultdict(list)
    labels: dict[str, str] = {}
    for row in rows:
        raw_value = row.get(metric, "")
        if raw_value in {"", None}:
            continue
        try:
            timestamp = float(row["timestamp"])
            value = float(raw_value)
        except (KeyError, ValueError):
            continue
        pid = row.get("pid", "")
        role = row.get("role", "Other")
        labels[pid] = f"{role}:{pid}"
        by_pid[pid].append((timestamp, value))

    ranked = sorted(
        by_pid,
        key=lambda pid: max(value for _, value in by_pid[pid]),
        reverse=True,
    )[:limit]
    if not ranked:
        return {}
    t0 = min(timestamp for pid in ranked for timestamp, _ in by_pid[pid])
    return {
        labels[pid]: (
            np.asarray([timestamp - t0 for timestamp, _ in by_pid[pid]], dtype=float),
            np.asarray([value for _, value in by_pid[pid]], dtype=float),
        )
        for pid in ranked
    }


def plot_series(
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    output_prefix: Path,
    *,
    title: str,
    ylabel: str,
) -> None:
    if not series:
        return
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    for label, (times, values) in series.items():
        ax.plot(
            times,
            values,
            label=label,
            color=ROLE_COLORS.get(label, None),
            linewidth=1.4,
            alpha=0.9,
        )
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, color="gray", alpha=0.45)
    ax.legend(
        ncol=2,
        fontsize=8,
        handletextpad=0.3,
        handlelength=1.2,
        columnspacing=0.8,
        loc="upper right",
        frameon=True,
    )
    fig.tight_layout()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_role_summary(
    rows: list[dict[str, str]],
    output: Path,
) -> None:
    by_role_metric: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        role = row.get("role", "Other") or "Other"
        for metric in ("rss_mib", "gpu_memory_mib"):
            raw_value = row.get(metric, "")
            if raw_value in {"", None}:
                continue
            try:
                by_role_metric[(role, metric)].append(float(raw_value))
            except ValueError:
                continue

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "role",
                "metric",
                "avg_mib",
                "p95_mib",
                "max_mib",
                "samples",
            ],
        )
        writer.writeheader()
        for (role, metric), values in sorted(by_role_metric.items()):
            arr = np.asarray(values, dtype=float)
            writer.writerow(
                {
                    "role": role,
                    "metric": metric,
                    "avg_mib": f"{float(np.mean(arr)):.6f}",
                    "p95_mib": f"{float(np.percentile(arr, 95)):.6f}",
                    "max_mib": f"{float(np.max(arr)):.6f}",
                    "samples": len(values),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--top-workers", type=int, default=8)
    parser.add_argument("--output-prefix", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    rows = read_memory_rows(run_dir / "memory_workers.csv")
    prefix = args.output_prefix or run_dir / "memory_worker_timeline"

    plot_series(
        aggregate_by_role(rows, metric="rss_mib"),
        prefix.with_name(f"{prefix.name}_rss_by_role"),
        title="Host RSS by worker role",
        ylabel="RSS (MiB)",
    )
    plot_series(
        aggregate_by_role(rows, metric="gpu_memory_mib"),
        prefix.with_name(f"{prefix.name}_gpu_by_role"),
        title="GPU memory by worker role",
        ylabel="GPU memory (MiB)",
    )
    plot_series(
        top_workers(rows, metric="rss_mib", limit=args.top_workers),
        prefix.with_name(f"{prefix.name}_rss_top_workers"),
        title="Top worker host RSS",
        ylabel="RSS (MiB)",
    )
    plot_series(
        top_workers(rows, metric="gpu_memory_mib", limit=args.top_workers),
        prefix.with_name(f"{prefix.name}_gpu_top_workers"),
        title="Top worker GPU memory",
        ylabel="GPU memory (MiB)",
    )
    write_role_summary(rows, prefix.with_name(f"{prefix.name}_summary").with_suffix(".csv"))


if __name__ == "__main__":
    main()
