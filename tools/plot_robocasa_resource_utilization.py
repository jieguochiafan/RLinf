#!/usr/bin/env python3
"""Draw a KVCompress-style CPU/GPU utilization trace for RoboCasa profiling."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
from matplotlib import font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle


SAMPLE_RE = re.compile(r"^=== (?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ===$")
PID_RE = re.compile(r"\bpid=(\d+)\)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", required=True, type=Path)
    parser.add_argument("--start", default="2026-06-03 16:16:58")
    parser.add_argument("--end", default="2026-06-03 16:17:36")
    parser.add_argument("--num-cpus", type=float, default=112.0)
    parser.add_argument(
        "--output-stem",
        default="resource_utilization_dual_axis",
        help="Output stem inside the profile directory.",
    )
    parser.add_argument("--annotate-phases", action="store_true")
    parser.add_argument("--rollout-seconds", type=float, default=27.754)
    parser.add_argument("--generation-seconds", type=float, default=3.411)
    parser.add_argument("--env-step-seconds", type=float, default=3.529)
    parser.add_argument("--step-seconds", type=float, default=43.561)
    parser.add_argument("--no-smooth", action="store_true")
    parser.add_argument("--fig-width", type=float, default=4.8)
    return parser.parse_args()


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def parse_env_group_pids(profile_dir: Path) -> set[int]:
    env_group_pids: set[int] = set()
    log_path = profile_dir / "train.log"
    if not log_path.exists():
        return env_group_pids
    for line in log_path.read_text(errors="replace").splitlines():
        clean = strip_ansi(line)
        if "EnvGroup(rank=" in clean and "pid=" in clean:
            match = PID_RE.search(clean)
            if match:
                env_group_pids.add(int(match.group(1)))
    return env_group_pids


def classify(pid: int, comm: str, cmd: str, env_group_pids: set[int]) -> str | None:
    text = f"{comm} {cmd}"
    if pid in env_group_pids or "ray::EnvWorker" in text or "ray::EnvWorker.init_worker" in text:
        return "EnvWorker"
    if "ray::MultiStepRolloutWorker" in text or "ray::RolloutGroup" in text:
        return "RolloutWorker"
    if "ray::EmbodiedFSDPActor" in text or "ray::ActorGroup" in text:
        return "ActorWorker"
    if "compile_worker" in text:
        return "CompileWorker"
    if "train_embodied_agent.py" in text:
        return "Main"
    if comm == "raylet" or " raylet " in f" {text} ":
        return "raylet"
    if comm == "gcs_server" or "gcs_server" in text:
        return "gcs_server"
    return None


def parse_cpu_samples(
    cpu_path: Path,
    start: datetime,
    end: datetime,
    num_cpus: float,
    env_group_pids: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    current_ts: datetime | None = None
    current_total = 0.0
    duplicate_counts: dict[datetime, int] = defaultdict(int)

    def flush() -> None:
        nonlocal current_ts, current_total
        if current_ts is None or not (start <= current_ts <= end):
            return
        offset = duplicate_counts[current_ts] * 0.5
        duplicate_counts[current_ts] += 1
        xs.append((current_ts - start).total_seconds() + offset)
        ys.append(current_total / num_cpus * 100.0)

    with cpu_path.open(errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            match = SAMPLE_RE.match(line)
            if match:
                flush()
                current_ts = datetime.strptime(match.group("ts"), "%Y-%m-%d %H:%M:%S")
                current_total = 0.0
                continue
            if current_ts is None or not (start <= current_ts <= end):
                continue
            parts = line.split(None, 5)
            if len(parts) < 6:
                continue
            try:
                pid = int(parts[0])
            except ValueError:
                continue
            role = classify(pid, parts[4], parts[5], env_group_pids)
            if role is None:
                continue
            try:
                current_total += float(parts[2]) / 100.0
            except ValueError:
                continue
    flush()
    return np.asarray(xs), np.asarray(ys)


def parse_gpu_timestamp(raw_ts: str) -> datetime:
    return datetime.strptime(raw_ts.strip(), "%Y/%m/%d %H:%M:%S.%f")


def parse_gpu_samples(
    gpu_path: Path, start: datetime, end: datetime
) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[datetime, list[float]] = defaultdict(list)
    with gpu_path.open(errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ts = parse_gpu_timestamp(row["timestamp"])
            if not (start <= ts <= end):
                continue
            bucket = ts.replace(microsecond=int(ts.microsecond / 500000) * 500000)
            grouped[bucket].append(float(row["utilization.gpu"]))
    xs = []
    ys = []
    for ts in sorted(grouped):
        xs.append((ts - start).total_seconds())
        ys.append(float(np.mean(grouped[ts])))
    return np.asarray(xs), np.asarray(ys)


def moving_average(values: np.ndarray, window: int = 3) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def main() -> None:
    args = parse_args()
    profile_dir = args.profile_dir.resolve()
    start = datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S")
    end = datetime.strptime(args.end, "%Y-%m-%d %H:%M:%S")

    env_group_pids = parse_env_group_pids(profile_dir)
    cpu_x, cpu_y = parse_cpu_samples(
        profile_dir / "cpu_samples.txt",
        start,
        end,
        args.num_cpus,
        env_group_pids,
    )
    gpu_x, gpu_y = parse_gpu_samples(profile_dir / "gpu_samples.csv", start, end)
    if len(cpu_x) == 0 or len(gpu_x) == 0:
        raise RuntimeError("No CPU or GPU samples found in the requested window.")

    if not args.no_smooth:
        cpu_y = moving_average(cpu_y, 3)
        gpu_y = moving_average(gpu_y, 3)

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    font_family = "Arial" if "Arial" in available_fonts else "DejaVu Sans"
    font = {"family": font_family, "weight": "normal", "size": 13}
    plt.rcParams.update(
        {
            "font.family": font_family,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )

    fig, ax_gpu = plt.subplots(figsize=(args.fig_width, 2.6))
    ax_cpu = ax_gpu.twinx()

    gpu_line = ax_gpu.plot(
        gpu_x,
        gpu_y,
        color="#EA5455",
        linewidth=1.8,
        marker="s",
        markersize=3.0 if args.no_smooth else 3.5,
        markevery=1 if args.no_smooth else max(1, len(gpu_x) // 8),
        label="GPU",
        zorder=10,
    )[0]
    cpu_line = ax_cpu.plot(
        cpu_x,
        cpu_y,
        color="#2D4059",
        linewidth=1.8,
        marker="o",
        markersize=3.0 if args.no_smooth else 3.5,
        markevery=1 if args.no_smooth else max(1, len(cpu_x) // 8),
        label="CPU",
        zorder=9,
    )[0]

    if args.annotate_phases:
        rollout_end = min(args.rollout_seconds, (end - start).total_seconds())
        step_end = min(args.step_seconds, (end - start).total_seconds())
        ax_gpu.axvspan(0, rollout_end, color="#9DBDFF", alpha=0.13, zorder=-3)
        if step_end > rollout_end:
            ax_gpu.axvspan(rollout_end, step_end, color="#76BA99", alpha=0.09, zorder=-3)
        ax_gpu.axvline(rollout_end, color="gray", linestyle="--", linewidth=0.8, zorder=2)
        ax_gpu.text(
            rollout_end / 2,
            0.94,
            "Rollout",
            transform=ax_gpu.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5),
            zorder=20,
        )
        if step_end > rollout_end:
            ax_gpu.text(
                rollout_end + (step_end - rollout_end) / 2,
                0.94,
                "Training",
                transform=ax_gpu.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=10,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5),
                zorder=20,
            )

        strip_start = 1.0
        strip_gap = 0.45
        strip_y = 0.055
        strip_h = 0.055
        gen_w = args.generation_seconds
        env_w = args.env_step_seconds
        env_x = strip_start + gen_w + strip_gap
        strip_transform = ax_gpu.get_xaxis_transform()
        ax_gpu.add_patch(
            Rectangle(
                (strip_start, strip_y),
                gen_w,
                strip_h,
                transform=strip_transform,
                facecolor="#EA5455",
                edgecolor="black",
                linewidth=0.6,
                alpha=0.85,
                zorder=30,
            )
        )
        ax_gpu.add_patch(
            Rectangle(
                (env_x, strip_y),
                env_w,
                strip_h,
                transform=strip_transform,
                facecolor="#FFD460",
                edgecolor="black",
                linewidth=0.6,
                alpha=0.9,
                zorder=30,
            )
        )
        ax_gpu.text(
            strip_start + gen_w / 2,
            strip_y + strip_h / 2,
            "generation",
            transform=strip_transform,
            ha="center",
            va="center",
            fontsize=8,
            zorder=31,
        )
        ax_gpu.text(
            env_x + env_w / 2,
            strip_y + strip_h / 2,
            "env",
            transform=strip_transform,
            ha="center",
            va="center",
            fontsize=8,
            zorder=31,
        )
        ax_gpu.text(
            strip_start,
            strip_y + strip_h + 0.015,
            "cumulative time inside rollout",
            transform=strip_transform,
            ha="left",
            va="bottom",
            fontsize=7.5,
            color="dimgray",
            zorder=31,
        )

    ax_gpu.fill_between(gpu_x, gpu_y, 0, color="#EA5455", alpha=0.16, zorder=1)
    ax_cpu.fill_between(cpu_x, cpu_y, 0, color="#2D4059", alpha=0.10, zorder=0)

    ax_gpu.set_xlabel("Time (s)", fontdict=font)
    ax_gpu.set_ylabel("GPU Util. (%)", fontdict=font, color="#EA5455")
    ax_cpu.set_ylabel("CPU Util. (%)", fontdict=font, color="#2D4059")
    ax_gpu.tick_params(axis="y", labelsize=10, colors="#EA5455")
    ax_cpu.tick_params(axis="y", labelsize=10, colors="#2D4059")
    ax_gpu.tick_params(axis="x", labelsize=10)
    ax_gpu.set_xlim(0, max((end - start).total_seconds(), float(np.max(gpu_x))))
    ax_gpu.set_ylim(0, max(45, min(100, float(np.max(gpu_y)) * 1.25)))
    ax_cpu.set_ylim(0, max(35, min(100, float(np.max(cpu_y)) * 1.35)))
    ax_gpu.grid(axis="y", linestyle="--", linewidth=0.5, color="gray", alpha=0.6, zorder=-1)
    legend_handles = [gpu_line, cpu_line]
    if args.annotate_phases:
        legend_handles.extend(
            [
                Patch(facecolor="#9DBDFF", alpha=0.25, edgecolor="none", label="Rollout"),
                Patch(facecolor="#FFD460", edgecolor="black", label="Env"),
            ]
        )
    ax_gpu.legend(
        handles=legend_handles,
        ncol=2,
        fontsize=9,
        handletextpad=0.2,
        handlelength=1.4,
        columnspacing=0.8,
        loc="upper right",
        frameon=True,
        edgecolor="black",
    )

    output_pdf = profile_dir / f"{args.output_stem}.pdf"
    output_png = profile_dir / f"{args.output_stem}.png"
    fig.tight_layout()
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, bbox_inches="tight", dpi=240)
    print(f"WROTE={output_pdf}")
    print(f"WROTE={output_png}")
    print(f"CPU_AVG={np.mean(cpu_y):.2f}% CPU_MAX={np.max(cpu_y):.2f}%")
    print(f"GPU_AVG={np.mean(gpu_y):.2f}% GPU_MAX={np.max(gpu_y):.2f}%")


if __name__ == "__main__":
    main()
