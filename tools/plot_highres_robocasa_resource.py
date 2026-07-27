#!/usr/bin/env python3
"""Draw high-resolution role-level RoboCasa CPU/GPU utilization traces."""

from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path

import matplotlib
from matplotlib import font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


CPU_ROLES = [
    ("EnvWorker_cores", "Env CPU", "#FFD460"),
    ("RolloutWorker_cores", "Generation CPU", "#EA5455"),
    ("ActorWorker_cores", "Training CPU", "#2D4059"),
    ("Other_cores", "Other CPU", "darkgray"),
]
TS_PREFIX_RE = re.compile(r"^\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6})\]")
METRIC_RE = re.compile(r"(?P<name>[A-Za-z_/]+)=(?P<value>[0-9.eE+-]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", required=True, type=Path)
    parser.add_argument("--num-cpus", type=float, default=112.0)
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument("--output-stem", default="resource_utilization_highres_roles")
    parser.add_argument("--fig-width", type=float, default=6.2)
    parser.add_argument("--total-only", action="store_true")
    return parser.parse_args()


def parse_cpu(path: Path, num_cpus: float) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No CPU samples found in {path}")
    t0 = float(rows[0]["timestamp"])
    xs = np.asarray([float(row["timestamp"]) - t0 for row in rows])
    series: dict[str, np.ndarray] = {}
    for column, _, _ in CPU_ROLES:
        if column in rows[0]:
            series[column] = np.asarray([float(row[column]) / num_cpus * 100.0 for row in rows])
    total = np.asarray([float(row["total_util_pct"]) for row in rows])
    series["total_util_pct"] = total
    elapsed = np.asarray([float(row["elapsed_s"]) for row in rows])
    stats = {
        "cpu_t0": t0,
        "slot_median": float(np.median(elapsed)),
        "slot_p95": float(np.percentile(elapsed, 95)),
        "total_avg": float(np.mean(total)),
        "total_max": float(np.max(total)),
    }
    return xs, series, stats


def parse_gpu_timestamp(raw_ts: str) -> float:
    ts = datetime.strptime(raw_ts.strip(), "%Y/%m/%d %H:%M:%S.%f")
    return ts.timestamp()


def parse_gpu(path: Path, cpu_t0: float) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[float, list[float]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"No GPU CSV header found in {path}")
        util_column = next(
            (name for name in reader.fieldnames if name.strip().startswith("utilization.gpu")),
            None,
        )
        if util_column is None:
            raise RuntimeError(f"No GPU utilization column found in {path}: {reader.fieldnames}")
        for row in reader:
            ts = parse_gpu_timestamp(row["timestamp"])
            bucket = round((ts - cpu_t0) * 10.0) / 10.0
            grouped.setdefault(bucket, []).append(float(row[util_column]))
    xs = np.asarray(sorted(grouped))
    ys = np.asarray([float(np.mean(grouped[x])) for x in xs])
    return xs, ys


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def parse_log_metadata(path: Path, cpu_t0: float) -> dict[str, float]:
    metadata: dict[str, float] = {}
    last_ts: float | None = None
    rollout_start: float | None = None
    rollout_end: float | None = None
    metrics: dict[str, float] = {}
    for raw in path.read_text(errors="replace").splitlines():
        line = strip_ansi(raw)
        ts_match = TS_PREFIX_RE.match(line)
        if ts_match:
            last_ts = datetime.strptime(
                ts_match.group("ts"), "%Y-%m-%d %H:%M:%S.%f"
            ).timestamp()
        if "Generating Rollout Epochs:" in line and "0/1" in line and rollout_start is None:
            rollout_start = last_ts
        if "Generating Rollout Epochs:" in line and "100%" in line:
            rollout_end = last_ts
        for match in METRIC_RE.finditer(line):
            metrics[match.group("name")] = float(match.group("value"))
    if rollout_start is not None:
        metadata["rollout_start"] = rollout_start - cpu_t0
    if rollout_end is not None:
        metadata["rollout_end"] = rollout_end - cpu_t0
    if "step" in metrics:
        metadata["step_seconds"] = metrics["step"]
    if "actor/run_training" in metrics:
        metadata["training_seconds"] = metrics["actor/run_training"]
    if "rollout/predict" in metrics:
        metadata["generation_seconds"] = metrics["rollout/predict"]
    if "env/env_interact_step" in metrics:
        metadata["env_seconds"] = metrics["env/env_interact_step"]
    return metadata


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def main() -> None:
    args = parse_args()
    profile_dir = args.profile_dir.resolve()
    cpu_x, cpu_series, stats = parse_cpu(profile_dir / "cpu_highres.csv", args.num_cpus)
    gpu_x, gpu_y = parse_gpu(profile_dir / "gpu_highres.csv", stats["cpu_t0"])
    metadata = parse_log_metadata(profile_dir / "train.log", stats["cpu_t0"])

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    font_family = "Arial" if "Arial" in available_fonts else "DejaVu Sans"
    font = {"family": font_family, "weight": "normal", "size": 12}
    plt.rcParams.update(
        {
            "font.family": font_family,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )

    fig, ax_gpu = plt.subplots(figsize=(args.fig_width, 2.9))
    ax_cpu = ax_gpu.twinx()

    rollout_start = metadata.get("rollout_start")
    rollout_end = metadata.get("rollout_end")
    x_max = max(float(np.max(cpu_x)), float(np.max(gpu_x)) if len(gpu_x) else 0.0)
    if not args.total_only and rollout_start is not None and rollout_end is not None:
        ax_gpu.axvspan(rollout_start, rollout_end, color="#9DBDFF", alpha=0.13, zorder=-4)
        ax_gpu.text(
            (rollout_start + rollout_end) / 2,
            0.97,
            "Rollout",
            transform=ax_gpu.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
        )
    if not args.total_only and "training_seconds" in metadata:
        train_end = x_max
        train_start = max(0.0, train_end - metadata["training_seconds"])
        ax_gpu.axvspan(train_start, train_end, color="#76BA99", alpha=0.11, zorder=-4)
        ax_gpu.text(
            (train_start + train_end) / 2,
            0.97,
            "Training",
            transform=ax_gpu.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
        )

    gpu_line = ax_gpu.plot(
        gpu_x,
        smooth(gpu_y, args.smooth_window),
        color="#EA5455",
        linewidth=1.7,
        label="GPU",
        zorder=20,
    )[0]
    cpu_handles = []
    if not args.total_only:
        for column, label, color in CPU_ROLES:
            values = cpu_series.get(column)
            if values is None:
                continue
            if column == "Other_cores" and float(np.max(values)) < 0.5:
                continue
            handle = ax_cpu.plot(
                cpu_x,
                smooth(values, args.smooth_window),
                color=color,
                linewidth=1.4 if column != "Other_cores" else 1.0,
                linestyle="--" if column == "Other_cores" else "-",
                label=label,
                zorder=18 if column != "Other_cores" else 12,
            )[0]
            cpu_handles.append(handle)

    total_line = ax_cpu.plot(
        cpu_x,
        smooth(cpu_series["total_util_pct"], args.smooth_window),
        color="#2D4059",
        linewidth=1.7 if args.total_only else 1.2,
        alpha=1.0 if args.total_only else 0.65,
        label="CPU" if args.total_only else "CPU total",
        zorder=16,
    )[0]

    if not args.total_only and rollout_start is not None and rollout_end is not None:
        gen_s = metadata.get("generation_seconds", 0.0)
        env_s = metadata.get("env_seconds", 0.0)
        if gen_s > 0 and env_s > 0:
            y = 0.055
            h = 0.052
            strip_transform = ax_gpu.get_xaxis_transform()
            strip_start = rollout_start + 0.5
            ax_gpu.broken_barh(
                [(strip_start, gen_s)],
                (y, h),
                transform=strip_transform,
                facecolors="#EA5455",
                edgecolors="black",
                linewidth=0.5,
                zorder=30,
            )
            ax_gpu.broken_barh(
                [(strip_start + gen_s + 0.5, env_s)],
                (y, h),
                transform=strip_transform,
                facecolors="#FFD460",
                edgecolors="black",
                linewidth=0.5,
                zorder=30,
            )
            ax_gpu.text(
                strip_start + gen_s / 2,
                y + h / 2,
                "generation",
                transform=strip_transform,
                ha="center",
                va="center",
                fontsize=7.5,
                zorder=31,
            )
            ax_gpu.text(
                strip_start + gen_s + 0.5 + env_s / 2,
                y + h / 2,
                "env",
                transform=strip_transform,
                ha="center",
                va="center",
                fontsize=7.5,
                zorder=31,
            )

    ax_gpu.set_xlabel("Time (s)", fontdict=font)
    ax_gpu.set_ylabel("GPU Util. (%)", fontdict=font, color="#EA5455")
    ax_cpu.set_ylabel("CPU Util. (%)", fontdict=font, color="#2D4059")
    ax_gpu.tick_params(axis="y", labelsize=9, colors="#EA5455")
    ax_cpu.tick_params(axis="y", labelsize=9, colors="#2D4059")
    ax_gpu.tick_params(axis="x", labelsize=9)
    ax_gpu.grid(axis="y", linestyle="--", linewidth=0.5, color="gray", alpha=0.6, zorder=-1)
    ax_gpu.set_xlim(0, x_max)
    ax_gpu.set_ylim(0, max(40.0, min(100.0, float(np.max(gpu_y)) * 1.25)))
    ax_cpu.set_ylim(0, max(30.0, min(100.0, float(np.max(cpu_series["total_util_pct"])) * 1.25)))

    if args.total_only:
        legend_handles = [gpu_line, total_line]
    else:
        legend_handles = [
            gpu_line,
            total_line,
            *cpu_handles[:3],
            Patch(facecolor="#9DBDFF", alpha=0.3, edgecolor="none", label="Rollout"),
            Patch(facecolor="#76BA99", alpha=0.3, edgecolor="none", label="Training"),
        ]
    ax_gpu.legend(
        handles=legend_handles,
        ncol=2 if args.total_only else 3,
        fontsize=9 if args.total_only else 8,
        handletextpad=0.2,
        handlelength=1.2,
        columnspacing=0.6,
        loc="upper right",
        frameon=True,
        edgecolor="black",
    )

    output_pdf = profile_dir / f"{args.output_stem}.pdf"
    output_png = profile_dir / f"{args.output_stem}.png"
    fig.tight_layout()
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, bbox_inches="tight", dpi=260)
    print(f"WROTE={output_pdf}")
    print(f"WROTE={output_png}")
    print(f"CPU_AVG={stats['total_avg']:.2f}% CPU_MAX={stats['total_max']:.2f}%")
    print(f"GPU_AVG={np.mean(gpu_y):.2f}% GPU_MAX={np.max(gpu_y):.2f}%")
    print(f"SLOT_MEDIAN={stats['slot_median']:.4f}s SLOT_P95={stats['slot_p95']:.4f}s")
    for key in ("generation_seconds", "env_seconds", "training_seconds"):
        if key in metadata:
            print(f"{key.upper()}={metadata[key]:.3f}s")


if __name__ == "__main__":
    main()
