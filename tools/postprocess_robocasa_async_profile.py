#!/usr/bin/env python3
"""Stream post-process large RoboCasa async profile CSVs.

The raw worker memory/core CSVs can grow to tens of GB when a run hangs.  This
script keeps memory bounded by aggregating rows while streaming the files.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROLE_ORDER = [
    "EnvWorker",
    "RolloutWorker",
    "ActorWorker",
    "CompileWorker",
    "Main",
    "raylet",
    "gcs_server",
    "Idle",
    "Other",
]

ROLE_COLORS = {
    "EnvWorker": "#FFD460",
    "RolloutWorker": "#EA5455",
    "ActorWorker": "#2D4059",
    "CompileWorker": "#16A3A6",
    "Main": "#76BA99",
    "raylet": "#6C5B7B",
    "gcs_server": "#B83B5E",
    "Idle": "lightgray",
    "Other": "darkgray",
}


@dataclass
class PidStats:
    role: str = "Other"
    comm: str = ""
    cmdline: str = ""
    max_rss_mib: float = 0.0
    max_gpu_mib: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pid-min", type=int, default=None)
    parser.add_argument("--pid-max", type=int, default=None)
    parser.add_argument("--bin-s", type=float, default=2.0)
    parser.add_argument(
        "--active-end-s",
        type=float,
        default=360.0,
        help="Active-window end relative to run start.",
    )
    parser.add_argument("--top-workers", type=int, default=16)
    parser.add_argument("--num-cpus", type=float, default=112.0)
    return parser.parse_args()


def role_sort_key(role: str) -> tuple[int, str]:
    try:
        return (ROLE_ORDER.index(role), role)
    except ValueError:
        return (len(ROLE_ORDER), role)


def normalize_role(role: str, comm: str, cmdline: str) -> str:
    text = f"{comm} {cmdline}"
    if "AsyncEnvWorker" in text or "EnvGroup" in text:
        return "EnvWorker"
    if "AsyncMultiStepRolloutWorker" in text or "RolloutWorker" in text:
        return "RolloutWorker"
    if "AsyncPPOEmbodiedFSDPActor" in text or "ActorWorker" in text:
        return "ActorWorker"
    if "train_async.py" in text or "train_embodied_agent.py" in text:
        return "Main"
    if "ray::IDLE" in text:
        return "Idle"
    return role or "Other"


def pid_allowed(pid: int, pid_min: int | None, pid_max: int | None) -> bool:
    if pid_min is not None and pid < pid_min:
        return False
    if pid_max is not None and pid > pid_max:
        return False
    return True


def infer_t0(run_dir: Path) -> float:
    with (run_dir / "cpu_highres.csv").open(newline="") as handle:
        row = next(csv.DictReader(handle))
    return float(row["timestamp"])


def bin_start(rel_s: float, bin_s: float) -> float:
    return math.floor(rel_s / bin_s) * bin_s


def parse_gpu_timestamp(raw_ts: str) -> float:
    return datetime.strptime(raw_ts.strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_cpu_gpu(
    run_dir: Path,
    output_dir: Path,
    *,
    t0: float,
    bin_s: float,
    active_end_s: float,
    num_cpus: float,
) -> None:
    by_bucket_cpu: dict[float, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    cpu_columns = [
        "total_cores",
        "EnvWorker_cores",
        "RolloutWorker_cores",
        "ActorWorker_cores",
        "CompileWorker_cores",
        "Main_cores",
        "raylet_cores",
        "gcs_server_cores",
        "Other_cores",
    ]
    with (run_dir / "cpu_highres.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            rel_s = float(row["timestamp"]) - t0
            if rel_s < 0 or rel_s > active_end_s:
                continue
            bucket = bin_start(rel_s, bin_s)
            for column in cpu_columns:
                by_bucket_cpu[bucket][column].append(float(row.get(column, 0.0) or 0.0))

    by_bucket_gpu: dict[float, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with (run_dir / "gpu_highres.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        util_column = next(
            name
            for name in reader.fieldnames or []
            if name.strip().startswith("utilization.gpu")
        )
        mem_column = next(
            name
            for name in reader.fieldnames or []
            if name.strip().startswith("memory.used")
        )
        for row in reader:
            rel_s = parse_gpu_timestamp(row["timestamp"]) - t0
            if rel_s < 0 or rel_s > active_end_s:
                continue
            bucket = bin_start(rel_s, bin_s)
            by_bucket_gpu[bucket]["gpu_util_pct"].append(float(row[util_column]))
            by_bucket_gpu[bucket]["gpu_mem_mib"].append(float(row[mem_column]))

    rows: list[dict[str, object]] = []
    for bucket in sorted(set(by_bucket_cpu) | set(by_bucket_gpu)):
        cpu = by_bucket_cpu.get(bucket, {})
        gpu = by_bucket_gpu.get(bucket, {})
        row: dict[str, object] = {"time_s": f"{bucket:.3f}"}
        for column in cpu_columns:
            values = cpu.get(column, [])
            row[column] = f"{float(np.mean(values)):.6f}" if values else ""
        gpu_util = gpu.get("gpu_util_pct", [])
        gpu_mem = gpu.get("gpu_mem_mib", [])
        row["gpu_util_avg_pct"] = f"{float(np.mean(gpu_util)):.6f}" if gpu_util else ""
        row["gpu_util_max_pct"] = f"{float(np.max(gpu_util)):.6f}" if gpu_util else ""
        row["gpu_mem_total_gib"] = (
            f"{float(np.sum(gpu_mem) / 1024.0):.6f}" if gpu_mem else ""
        )
        row["cpu_total_pct_of_configured"] = (
            ""
            if not cpu.get("total_cores")
            else f"{float(np.mean(cpu['total_cores']) / num_cpus * 100.0):.6f}"
        )
        rows.append(row)

    fieldnames = [
        "time_s",
        *cpu_columns,
        "cpu_total_pct_of_configured",
        "gpu_util_avg_pct",
        "gpu_util_max_pct",
        "gpu_mem_total_gib",
    ]
    csv_path = output_dir / "active_cpu_gpu_roles.csv"
    write_rows(csv_path, fieldnames, rows)
    plot_cpu_gpu_roles(rows, output_dir / "active_cpu_gpu_roles", cpu_columns)


def plot_cpu_gpu_roles(
    rows: list[dict[str, object]],
    output_prefix: Path,
    cpu_columns: list[str],
) -> None:
    if not rows:
        return
    times = np.array([float(row["time_s"]) for row in rows])
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 5.2), sharex=True)
    role_columns = [
        ("EnvWorker_cores", "Env CPU", "#FFD460"),
        ("RolloutWorker_cores", "Rollout CPU", "#EA5455"),
        ("ActorWorker_cores", "Actor CPU", "#2D4059"),
        ("CompileWorker_cores", "Compile CPU", "#16A3A6"),
        ("Other_cores", "Other CPU", "darkgray"),
    ]
    for column, label, color in role_columns:
        values = np.array([
            float(row[column]) if row.get(column) not in {"", None} else np.nan
            for row in rows
        ])
        if np.all(np.isnan(values)) or np.nanmax(values) <= 0:
            continue
        axes[0].plot(times, values, label=label, color=color, linewidth=1.2)
    axes[0].set_ylabel("CPU cores")
    axes[0].legend(ncol=3, fontsize=8, loc="upper right")

    gpu_avg = np.array([
        float(row["gpu_util_avg_pct"])
        if row.get("gpu_util_avg_pct") not in {"", None}
        else np.nan
        for row in rows
    ])
    gpu_max = np.array([
        float(row["gpu_util_max_pct"])
        if row.get("gpu_util_max_pct") not in {"", None}
        else np.nan
        for row in rows
    ])
    axes[1].plot(times, gpu_avg, label="GPU avg", color="#EA5455", linewidth=1.3)
    axes[1].plot(times, gpu_max, label="GPU max", color="#2D4059", linewidth=1.0)
    axes[1].set_ylabel("GPU util. (%)")
    axes[1].set_ylim(0, 105)
    axes[1].legend(ncol=2, fontsize=8, loc="upper right")

    gpu_mem = np.array([
        float(row["gpu_mem_total_gib"])
        if row.get("gpu_mem_total_gib") not in {"", None}
        else np.nan
        for row in rows
    ])
    axes[2].plot(times, gpu_mem, label="GPU memory", color="#16A3A6", linewidth=1.3)
    axes[2].set_ylabel("GPU mem. (GiB)")
    axes[2].set_xlabel("Time from run start (s)")
    for ax in axes:
        ax.grid(axis="y", linestyle="--", linewidth=0.5, color="gray", alpha=0.45)
    fig.tight_layout()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def flush_memory_sample(
    timestamp: float,
    t0: float,
    bin_s: float,
    role_totals: dict[str, dict[str, float]],
    buckets: dict[float, dict[str, dict[str, float]]],
    bucket_counts: dict[float, int],
    *,
    max_rel_s: float | None,
) -> None:
    rel_s = timestamp - t0
    if rel_s < 0:
        return
    if max_rel_s is not None and rel_s > max_rel_s:
        return
    bucket = bin_start(rel_s, bin_s)
    bucket_counts[bucket] += 1
    for role, totals in role_totals.items():
        for metric, value in totals.items():
            buckets[bucket][role][metric] += value


def aggregate_memory(
    run_dir: Path,
    output_dir: Path,
    *,
    t0: float,
    bin_s: float,
    active_end_s: float,
    pid_min: int | None,
    pid_max: int | None,
    top_workers: int,
) -> None:
    full_buckets: dict[float, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    active_buckets: dict[float, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    full_counts: dict[float, int] = defaultdict(int)
    active_counts: dict[float, int] = defaultdict(int)
    pid_stats: dict[int, PidStats] = defaultdict(PidStats)

    current_ts: float | None = None
    sample_totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def flush() -> None:
        if current_ts is None:
            return
        flush_memory_sample(
            current_ts,
            t0,
            bin_s,
            sample_totals,
            full_buckets,
            full_counts,
            max_rel_s=None,
        )
        flush_memory_sample(
            current_ts,
            t0,
            bin_s,
            sample_totals,
            active_buckets,
            active_counts,
            max_rel_s=active_end_s,
        )

    with (run_dir / "memory_workers.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                pid = int(row["pid"])
                timestamp = float(row["timestamp"])
            except (KeyError, ValueError):
                continue
            if not pid_allowed(pid, pid_min, pid_max):
                continue
            role = normalize_role(
                row.get("role", "Other"), row.get("comm", ""), row.get("cmdline", "")
            )
            if current_ts is not None and timestamp != current_ts:
                flush()
                sample_totals = defaultdict(lambda: defaultdict(float))
            current_ts = timestamp
            rss_mib = parse_float(row.get("rss_mib", ""))
            gpu_mib = parse_float(row.get("gpu_memory_mib", ""))
            sample_totals[role]["rss_mib"] += rss_mib
            sample_totals[role]["gpu_mib"] += gpu_mib

            stats = pid_stats[pid]
            stats.role = role
            stats.comm = row.get("comm", "")
            stats.cmdline = row.get("cmdline", "")[:180]
            stats.max_rss_mib = max(stats.max_rss_mib, rss_mib)
            stats.max_gpu_mib = max(stats.max_gpu_mib, gpu_mib)
    flush()

    write_memory_role_csv(output_dir / "memory_roles_full.csv", full_buckets, full_counts)
    write_memory_role_csv(
        output_dir / "memory_roles_active.csv", active_buckets, active_counts
    )
    plot_memory_roles(
        output_dir / "memory_roles_active.csv",
        output_dir / "memory_roles_active",
        title="Active worker memory by role",
    )
    plot_memory_roles(
        output_dir / "memory_roles_full.csv",
        output_dir / "memory_roles_full",
        title="Full worker memory by role",
    )
    write_pid_summary(output_dir / "memory_worker_summary.csv", pid_stats)
    top_pids = select_top_pids(pid_stats, top_workers)
    if top_pids:
        aggregate_top_worker_memory(
            run_dir,
            output_dir,
            t0=t0,
            bin_s=bin_s,
            active_end_s=active_end_s,
            pid_min=pid_min,
            pid_max=pid_max,
            top_pids=top_pids,
        )


def parse_float(value: str | None) -> float:
    if value in {None, ""}:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def write_memory_role_csv(
    output: Path,
    buckets: dict[float, dict[str, dict[str, float]]],
    counts: dict[float, int],
) -> None:
    roles = sorted(
        {role for role_values in buckets.values() for role in role_values},
        key=role_sort_key,
    )
    fieldnames = [
        "time_s",
        *[f"{role}_rss_gib" for role in roles],
        *[f"{role}_gpu_gib" for role in roles],
        "total_rss_gib",
        "total_gpu_gib",
        "samples",
    ]
    rows: list[dict[str, object]] = []
    for bucket in sorted(buckets):
        samples = max(1, counts[bucket])
        row: dict[str, object] = {"time_s": f"{bucket:.3f}", "samples": samples}
        total_rss = 0.0
        total_gpu = 0.0
        for role in roles:
            rss = buckets[bucket][role].get("rss_mib", 0.0) / samples / 1024.0
            gpu = buckets[bucket][role].get("gpu_mib", 0.0) / samples / 1024.0
            row[f"{role}_rss_gib"] = f"{rss:.6f}"
            row[f"{role}_gpu_gib"] = f"{gpu:.6f}"
            total_rss += rss
            total_gpu += gpu
        row["total_rss_gib"] = f"{total_rss:.6f}"
        row["total_gpu_gib"] = f"{total_gpu:.6f}"
        rows.append(row)
    write_rows(output, fieldnames, rows)


def plot_memory_roles(csv_path: Path, output_prefix: Path, *, title: str) -> None:
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    roles = [
        name[: -len("_rss_gib")]
        for name in rows[0]
        if name.endswith("_rss_gib") and not name.startswith("total_")
    ]
    times = np.array([float(row["time_s"]) for row in rows])
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 4.2), sharex=True)
    for role in roles:
        rss = np.array([float(row[f"{role}_rss_gib"]) for row in rows])
        gpu = np.array([float(row[f"{role}_gpu_gib"]) for row in rows])
        color = ROLE_COLORS.get(role, None)
        if np.max(rss) > 0.01:
            axes[0].plot(times, rss, label=role, color=color, linewidth=1.2)
        if np.max(gpu) > 0.01:
            axes[1].plot(times, gpu, label=role, color=color, linewidth=1.2)
    axes[0].set_ylabel("Host RSS (GiB)")
    axes[1].set_ylabel("GPU mem. (GiB)")
    axes[1].set_xlabel("Time from run start (s)")
    axes[0].set_title(title, fontsize=10)
    for ax in axes:
        ax.grid(axis="y", linestyle="--", linewidth=0.5, color="gray", alpha=0.45)
        ax.legend(ncol=3, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_pid_summary(output: Path, pid_stats: dict[int, PidStats]) -> None:
    rows = []
    for pid, stats in sorted(
        pid_stats.items(),
        key=lambda item: (item[1].max_gpu_mib, item[1].max_rss_mib),
        reverse=True,
    ):
        rows.append(
            {
                "pid": pid,
                "role": stats.role,
                "comm": stats.comm,
                "max_rss_gib": f"{stats.max_rss_mib / 1024.0:.6f}",
                "max_gpu_gib": f"{stats.max_gpu_mib / 1024.0:.6f}",
                "cmdline": stats.cmdline,
            }
        )
    write_rows(
        output,
        ["pid", "role", "comm", "max_rss_gib", "max_gpu_gib", "cmdline"],
        rows,
    )


def select_top_pids(pid_stats: dict[int, PidStats], limit: int) -> set[int]:
    if limit <= 0:
        return set()
    ranked = sorted(
        pid_stats,
        key=lambda pid: (pid_stats[pid].max_gpu_mib, pid_stats[pid].max_rss_mib),
        reverse=True,
    )
    return set(ranked[:limit])


def aggregate_top_worker_memory(
    run_dir: Path,
    output_dir: Path,
    *,
    t0: float,
    bin_s: float,
    active_end_s: float,
    pid_min: int | None,
    pid_max: int | None,
    top_pids: set[int],
) -> None:
    pid_buckets: dict[float, dict[int, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    counts: dict[float, int] = defaultdict(int)
    current_ts: float | None = None
    sample_values: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def flush() -> None:
        if current_ts is None:
            return
        rel_s = current_ts - t0
        if rel_s < 0 or rel_s > active_end_s:
            return
        bucket = bin_start(rel_s, bin_s)
        counts[bucket] += 1
        for pid, values in sample_values.items():
            for metric, value in values.items():
                pid_buckets[bucket][pid][metric] += value

    with (run_dir / "memory_workers.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                pid = int(row["pid"])
                timestamp = float(row["timestamp"])
            except (KeyError, ValueError):
                continue
            if not pid_allowed(pid, pid_min, pid_max):
                continue
            if timestamp - t0 > active_end_s:
                break
            if current_ts is not None and timestamp != current_ts:
                flush()
                sample_values = defaultdict(lambda: defaultdict(float))
            current_ts = timestamp
            if pid not in top_pids:
                continue
            sample_values[pid]["rss_mib"] += parse_float(row.get("rss_mib", ""))
            sample_values[pid]["gpu_mib"] += parse_float(row.get("gpu_memory_mib", ""))
    flush()

    fieldnames = ["time_s"]
    for pid in sorted(top_pids):
        fieldnames.extend([f"{pid}_rss_gib", f"{pid}_gpu_gib"])
    rows: list[dict[str, object]] = []
    for bucket in sorted(pid_buckets):
        samples = max(1, counts[bucket])
        row: dict[str, object] = {"time_s": f"{bucket:.3f}"}
        for pid in sorted(top_pids):
            row[f"{pid}_rss_gib"] = (
                f"{pid_buckets[bucket][pid].get('rss_mib', 0.0) / samples / 1024.0:.6f}"
            )
            row[f"{pid}_gpu_gib"] = (
                f"{pid_buckets[bucket][pid].get('gpu_mib', 0.0) / samples / 1024.0:.6f}"
            )
        rows.append(row)
    output = output_dir / "memory_top_workers_active.csv"
    write_rows(output, fieldnames, rows)
    plot_top_worker_memory(output, output_dir / "memory_top_workers_active")


def plot_top_worker_memory(csv_path: Path, output_prefix: Path) -> None:
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    pids = sorted({name.split("_", 1)[0] for name in rows[0] if name.endswith("_gpu_gib")})
    times = np.array([float(row["time_s"]) for row in rows])
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 4.4), sharex=True)
    for pid in pids:
        rss = np.array([float(row[f"{pid}_rss_gib"]) for row in rows])
        gpu = np.array([float(row[f"{pid}_gpu_gib"]) for row in rows])
        label = pid
        if np.max(rss) > 0.01:
            axes[0].plot(times, rss, label=label, linewidth=1.0)
        if np.max(gpu) > 0.01:
            axes[1].plot(times, gpu, label=label, linewidth=1.0)
    axes[0].set_ylabel("Host RSS (GiB)")
    axes[1].set_ylabel("GPU mem. (GiB)")
    axes[1].set_xlabel("Time from run start (s)")
    for ax in axes:
        ax.grid(axis="y", linestyle="--", linewidth=0.5, color="gray", alpha=0.45)
        ax.legend(ncol=4, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def summarize_env_affinity(
    run_dir: Path,
    output_dir: Path,
    *,
    pid_min: int | None,
    pid_max: int | None,
) -> None:
    worker_rows = []
    core_counts: dict[int, int] = defaultdict(int)
    with (run_dir / "cpu" / "worker_cpu_cores.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                pid = int(row["pid"])
            except ValueError:
                continue
            if not pid_allowed(pid, pid_min, pid_max):
                continue
            label = row.get("label", "")
            if "AsyncEnvWorker" not in label:
                continue
            cores = [int(core) for core in row.get("cores", "").split() if core]
            for core in cores:
                core_counts[core] += 1
            worker_rows.append(
                {
                    "pid": pid,
                    "core_count": len(cores),
                    "cores": " ".join(str(core) for core in cores),
                    "label": label[:120],
                }
            )
    write_rows(output_dir / "env_worker_affinity.csv", ["pid", "core_count", "cores", "label"], worker_rows)
    core_rows = [
        {"cpu": core, "env_worker_count": count}
        for core, count in sorted(core_counts.items())
    ]
    write_rows(output_dir / "env_core_assignment_counts.csv", ["cpu", "env_worker_count"], core_rows)


def aggregate_env_core_util(
    run_dir: Path,
    output_dir: Path,
    *,
    t0: float,
    bin_s: float,
    active_end_s: float,
    pid_min: int | None,
    pid_max: int | None,
) -> None:
    by_bucket_core_max: dict[tuple[float, int], float] = defaultdict(float)
    by_pid: dict[int, list[float]] = defaultdict(list)
    raw_path = run_dir / "cpu" / "worker_cpu_core_util.csv"
    with raw_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                pid = int(row["pid"])
                timestamp = float(row["timestamp"])
                core = int(row["cpu"])
                active_pct = float(row["active_pct"])
            except (KeyError, ValueError):
                continue
            rel_s = timestamp - t0
            if rel_s > active_end_s:
                break
            if rel_s < 0 or not pid_allowed(pid, pid_min, pid_max):
                continue
            if "AsyncEnvWorker" not in row.get("worker_label", ""):
                continue
            bucket = bin_start(rel_s, bin_s)
            key = (bucket, core)
            by_bucket_core_max[key] = max(by_bucket_core_max[key], active_pct)
            by_pid[pid].append(active_pct)

    times = sorted({bucket for bucket, _ in by_bucket_core_max})
    cores = sorted({core for _, core in by_bucket_core_max})
    rows: list[dict[str, object]] = []
    for bucket in times:
        values = [by_bucket_core_max.get((bucket, core), 0.0) for core in cores]
        arr = np.asarray(values, dtype=float)
        rows.append(
            {
                "time_s": f"{bucket:.3f}",
                "mean_env_core_active_pct": f"{float(np.mean(arr)):.6f}",
                "p90_env_core_active_pct": f"{float(np.percentile(arr, 90)):.6f}",
                "max_env_core_active_pct": f"{float(np.max(arr)):.6f}",
                "core_count": len(cores),
            }
        )
    write_rows(
        output_dir / "env_core_util_active_summary.csv",
        [
            "time_s",
            "mean_env_core_active_pct",
            "p90_env_core_active_pct",
            "max_env_core_active_pct",
            "core_count",
        ],
        rows,
    )
    pid_rows = []
    for pid, values in sorted(by_pid.items(), key=lambda item: np.mean(item[1]), reverse=True):
        arr = np.asarray(values, dtype=float)
        pid_rows.append(
            {
                "pid": pid,
                "avg_active_pct": f"{float(np.mean(arr)):.6f}",
                "p95_active_pct": f"{float(np.percentile(arr, 95)):.6f}",
                "max_active_pct": f"{float(np.max(arr)):.6f}",
                "samples": len(values),
            }
        )
    write_rows(
        output_dir / "env_worker_core_util_active_summary.csv",
        ["pid", "avg_active_pct", "p95_active_pct", "max_active_pct", "samples"],
        pid_rows,
    )
    plot_env_core_heatmap(by_bucket_core_max, times, cores, output_dir / "env_core_util_active_heatmap")


def plot_env_core_heatmap(
    values: dict[tuple[float, int], float],
    times: list[float],
    cores: list[int],
    output_prefix: Path,
) -> None:
    if not times or not cores:
        return
    matrix = np.zeros((len(cores), len(times)), dtype=float)
    for i, core in enumerate(cores):
        for j, time_s in enumerate(times):
            matrix[i, j] = values.get((time_s, core), 0.0)
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    image = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="YlOrRd",
        vmin=0,
        vmax=100,
        extent=[times[0], times[-1], cores[0] - 0.5, cores[-1] + 0.5],
    )
    ax.set_xlabel("Time from run start (s)")
    ax.set_ylabel("CPU core")
    colorbar = fig.colorbar(image, ax=ax, pad=0.015)
    colorbar.set_label("Env core active (%)")
    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_file_sizes(run_dir: Path, output_dir: Path) -> None:
    rows = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "bytes": path.stat().st_size,
                    "mib": f"{path.stat().st_size / 1024.0 / 1024.0:.3f}",
                    "path": str(path.relative_to(run_dir)),
                }
            )
    rows.sort(key=lambda row: int(row["bytes"]), reverse=True)
    write_rows(output_dir / "profile_file_sizes.csv", ["bytes", "mib", "path"], rows)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir or run_dir / "derived_profile"
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = infer_t0(run_dir)
    aggregate_cpu_gpu(
        run_dir,
        output_dir,
        t0=t0,
        bin_s=args.bin_s,
        active_end_s=args.active_end_s,
        num_cpus=args.num_cpus,
    )
    aggregate_memory(
        run_dir,
        output_dir,
        t0=t0,
        bin_s=args.bin_s,
        active_end_s=args.active_end_s,
        pid_min=args.pid_min,
        pid_max=args.pid_max,
        top_workers=args.top_workers,
    )
    summarize_env_affinity(
        run_dir,
        output_dir,
        pid_min=args.pid_min,
        pid_max=args.pid_max,
    )
    aggregate_env_core_util(
        run_dir,
        output_dir,
        t0=t0,
        bin_s=args.bin_s,
        active_end_s=args.active_end_s,
        pid_min=args.pid_min,
        pid_max=args.pid_max,
    )
    write_file_sizes(run_dir, output_dir)
    print(f"WROTE={output_dir}")


if __name__ == "__main__":
    main()
