#!/usr/bin/env python3
"""Parse RoboCasa/OpenPI profiling samples and generate CPU/GPU summaries."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SAMPLE_RE = re.compile(r"^=== (?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ===$")
PID_RE = re.compile(r"\bpid=(\d+)\)")
METRIC_RE = re.compile(r"([A-Za-z0-9_/]+)=([0-9.]+)")


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = int(round((len(sorted_values) - 1) * pct / 100.0))
    return sorted_values[max(0, min(idx, len(sorted_values) - 1))]


def summarize(values: list[float]) -> str:
    if not values:
        return "avg=0.00, p50=0.00, p95=0.00, max=0.00, n=0"
    avg = sum(values) / len(values)
    return (
        f"avg={avg:.2f}, p50={percentile(values, 50):.2f}, "
        f"p95={percentile(values, 95):.2f}, max={max(values):.2f}, n={len(values)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", required=True, type=Path)
    parser.add_argument("--config", required=True)
    parser.add_argument("--pinned-cores", type=int, default=112)
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=0.0,
        help="Use the last active window of this duration instead of inferring from metrics.",
    )
    return parser.parse_args()


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def parse_log(profile_dir: Path) -> tuple[set[int], list[str], dict[str, float]]:
    log_path = profile_dir / "train.log"
    env_group_pids: set[int] = set()
    evidence: list[str] = []
    metrics: dict[str, float] = {}
    for line in log_path.read_text(errors="replace").splitlines():
        clean = strip_ansi(line)
        if "EnvGroup(rank=" in clean and "pid=" in clean:
            match = PID_RE.search(clean)
            if match:
                env_group_pids.add(int(match.group(1)))
        if "Env CPU binding after train env setup" in clean:
            evidence.append(clean)
        if "│" in clean and "=" in clean:
            for key, value in METRIC_RE.findall(clean):
                metrics[key] = float(value)
    return env_group_pids, evidence, metrics


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
    profile_dir: Path, env_group_pids: set[int]
) -> tuple[list[dict[str, object]], Counter[str]]:
    samples: list[dict[str, object]] = []
    affinity_counts: Counter[str] = Counter()
    current: dict[str, object] | None = None
    cpu_path = profile_dir / "cpu_samples.txt"
    with cpu_path.open(errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            match = SAMPLE_RE.match(line)
            if match:
                if current is not None:
                    samples.append(current)
                current = {
                    "ts": datetime.strptime(match.group("ts"), "%Y-%m-%d %H:%M:%S"),
                    "roles": defaultdict(float),
                    "env_worker_pids": set(),
                }
                continue
            if current is None or not line.strip():
                continue
            parts = line.split(None, 5)
            if len(parts) < 6:
                continue
            try:
                pid = int(parts[0])
                psr = parts[1]
                cpu_pct = float(parts[2])
            except ValueError:
                continue
            comm = parts[4]
            cmd = parts[5]
            role = classify(pid, comm, cmd, env_group_pids)
            if role is None:
                continue
            roles = current["roles"]
            assert isinstance(roles, defaultdict)
            roles[role] += cpu_pct / 100.0
            if role == "EnvWorker":
                env_pids = current["env_worker_pids"]
                assert isinstance(env_pids, set)
                env_pids.add(pid)
            affinity_counts[psr] += 1
    if current is not None:
        samples.append(current)
    return samples, affinity_counts


def select_rollout_window(
    samples: list[dict[str, object]], metrics: dict[str, float], window_seconds: float = 0.0
) -> tuple[datetime, datetime]:
    rollout_seconds = window_seconds or metrics.get("step") or metrics.get("generate_rollouts") or 0.0
    if not samples:
        raise ValueError("no CPU samples parsed")
    end = samples[-1]["ts"]
    assert isinstance(end, datetime)
    start = end - timedelta(seconds=rollout_seconds + 5)
    # Prefer the high-utilization window if final samples include shutdown/idleness.
    rollout_samples = [
        sample
        for sample in samples
        if isinstance(sample["ts"], datetime) and start <= sample["ts"] <= end
    ]
    if rollout_samples:
        active = [
            sample
            for sample in rollout_samples
            if sum(sample["roles"].values()) >= 5.0  # type: ignore[union-attr]
        ]
        if active:
            active_start = active[0]["ts"]
            active_end = active[-1]["ts"]
            assert isinstance(active_start, datetime)
            assert isinstance(active_end, datetime)
            return active_start, active_end
    return start, end


def parse_gpu_timestamp(raw_ts: str) -> datetime | None:
    raw_ts = raw_ts.strip()
    for fmt in (
        "%Y/%m/%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(raw_ts, fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(raw_ts)
    except ValueError:
        return None


def write_plot(
    samples: list[dict[str, object]],
    output: Path,
    title: str,
    pinned_cores: int,
    start: datetime | None = None,
    end: datetime | None = None,
) -> None:
    selected = []
    for sample in samples:
        ts = sample["ts"]
        assert isinstance(ts, datetime)
        if start is not None and ts < start:
            continue
        if end is not None and ts > end:
            continue
        selected.append(sample)
    if not selected:
        return
    times = [sample["ts"] for sample in selected]
    roles = ["EnvWorker", "RolloutWorker", "ActorWorker", "CompileWorker", "Main"]
    plt.figure(figsize=(14, 7))
    total = []
    for sample in selected:
        role_values = sample["roles"]
        total.append(sum(role_values.values()))  # type: ignore[union-attr]
    plt.plot(times, total, label="Total", linewidth=2)
    for role in roles:
        plt.plot(
            times,
            [sample["roles"].get(role, 0.0) for sample in selected],  # type: ignore[union-attr]
            label=role,
            linewidth=1,
        )
    plt.axhline(pinned_cores, color="black", linestyle="--", linewidth=1, label=f"{pinned_cores} pinned cores")
    plt.title(title)
    plt.ylabel("CPU cores (sum thread %CPU / 100)")
    plt.xlabel("time")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def gpu_summary(profile_dir: Path, start: datetime, end: datetime) -> list[str]:
    path = profile_dir / "gpu_samples.csv"
    if not path.exists():
        return []
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with path.open(errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_ts = row.get("timestamp") or row.get("time") or ""
            ts = parse_gpu_timestamp(raw_ts)
            if ts is None:
                continue
            if not (start <= ts <= end):
                continue
            gpu = row.get("index") or row.get("gpu") or row.get("gpu_index") or "unknown"
            util_raw = (
                row.get("utilization.gpu [%]")
                or row.get("utilization.gpu")
                or row.get("util")
                or row.get("gpu_util")
            )
            mem_raw = (
                row.get("memory.used [MiB]")
                or row.get("memory.used")
                or row.get("memory_used")
                or row.get("mem_used")
            )
            if util_raw is not None:
                values[gpu]["util"].append(float(str(util_raw).replace("%", "").strip()))
            if mem_raw is not None:
                values[gpu]["mem"].append(float(str(mem_raw).replace("MiB", "").strip()))
    lines = []
    for gpu in sorted(values, key=lambda item: int(item) if item.isdigit() else item):
        util = values[gpu]["util"]
        mem = values[gpu]["mem"]
        if util:
            lines.append(
                f"gpu{gpu}: util_avg={sum(util)/len(util):.1f}%, util_max={max(util):.1f}%, "
                f"mem_used_avg={sum(mem)/len(mem):.0f}MiB, mem_used_max={max(mem):.0f}MiB"
            )
    return lines


def parse_gpu_samples(profile_dir: Path) -> list[dict[str, object]]:
    path = profile_dir / "gpu_samples.csv"
    if not path.exists():
        return []
    samples: list[dict[str, object]] = []
    with path.open(errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_ts = row.get("timestamp") or row.get("time") or ""
            ts = parse_gpu_timestamp(raw_ts)
            if ts is None:
                continue
            gpu = (row.get("index") or row.get("gpu") or row.get("gpu_index") or "unknown").strip()
            util_raw = row.get("utilization.gpu [%]") or row.get("utilization.gpu") or row.get("util") or row.get("gpu_util")
            mem_raw = row.get("memory.used [MiB]") or row.get("memory.used") or row.get("memory_used") or row.get("mem_used")
            power_raw = row.get("power.draw [W]") or row.get("power.draw") or row.get("power")
            try:
                util = float(str(util_raw).replace("%", "").strip()) if util_raw is not None else 0.0
                mem = float(str(mem_raw).replace("MiB", "").strip()) if mem_raw is not None else 0.0
                power = float(str(power_raw).replace("W", "").strip()) if power_raw is not None else 0.0
            except ValueError:
                continue
            samples.append({"ts": ts, "gpu": gpu, "util": util, "mem": mem, "power": power})
    return samples


def write_gpu_plot(
    gpu_samples: list[dict[str, object]],
    output: Path,
    title: str,
    start: datetime | None = None,
    end: datetime | None = None,
) -> None:
    selected = []
    for sample in gpu_samples:
        ts = sample["ts"]
        assert isinstance(ts, datetime)
        if start is not None and ts < start:
            continue
        if end is not None and ts > end:
            continue
        selected.append(sample)
    if not selected:
        return
    by_gpu: dict[str, list[dict[str, object]]] = defaultdict(list)
    for sample in selected:
        by_gpu[str(sample["gpu"])].append(sample)
    plt.figure(figsize=(14, 7))
    for gpu in sorted(by_gpu, key=lambda item: int(item) if item.isdigit() else item):
        rows = by_gpu[gpu]
        plt.plot([row["ts"] for row in rows], [row["util"] for row in rows], label=f"gpu{gpu}", linewidth=1)
    plt.title(title)
    plt.ylabel("GPU utilization (%)")
    plt.xlabel("time")
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper right", ncol=2)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    profile_dir = args.profile_dir.resolve()
    status = (profile_dir / "status.txt").read_text(errors="replace").strip()
    env_group_pids, evidence, metrics = parse_log(profile_dir)
    samples, affinity_counts = parse_cpu_samples(profile_dir, env_group_pids)
    start, end = select_rollout_window(samples, metrics, args.window_seconds)
    rollout_samples = [
        sample
        for sample in samples
        if isinstance(sample["ts"], datetime) and start <= sample["ts"] <= end
    ]
    role_values: dict[str, list[float]] = defaultdict(list)
    totals: list[float] = []
    env_worker_pids: set[int] = set()
    for sample in rollout_samples:
        roles = sample["roles"]
        totals.append(sum(roles.values()))  # type: ignore[union-attr]
        for role in ["EnvWorker", "RolloutWorker", "ActorWorker", "CompileWorker", "Main", "raylet", "gcs_server"]:
            role_values[role].append(roles.get(role, 0.0))  # type: ignore[union-attr]
        env_pids = sample["env_worker_pids"]
        env_worker_pids.update(env_pids)  # type: ignore[arg-type]
    whole_totals = [sum(sample["roles"].values()) for sample in samples]  # type: ignore[union-attr]
    write_plot(
        samples,
        profile_dir / "cpu_utilization_timeseries.png",
        args.config,
        args.pinned_cores,
    )
    write_plot(
        samples,
        profile_dir / "cpu_utilization_rollout_window.png",
        f"{args.config} (rollout window)",
        args.pinned_cores,
        start,
        end,
    )
    gpu_samples = parse_gpu_samples(profile_dir)
    write_gpu_plot(
        gpu_samples,
        profile_dir / "gpu_utilization_timeseries.png",
        f"{args.config} GPU utilization",
    )
    write_gpu_plot(
        gpu_samples,
        profile_dir / "gpu_utilization_epoch_window.png",
        f"{args.config} GPU utilization (epoch window)",
        start,
        end,
    )
    summary_lines = [
        f"PROFILE_DIR={profile_dir}",
        f"CONFIG={args.config}",
        status,
        f"ROLLOUT_WINDOW={start:%Y-%m-%d %H:%M:%S}..{end:%Y-%m-%d %H:%M:%S} samples={len(rollout_samples)}",
        f"ENV_WORKER_PIDS_OBSERVED_IN_WINDOW={len(env_worker_pids)}",
        f"AFFINITY_CHECK: ray_affinity_counts={dict(affinity_counts.most_common(16))}",
        "CPU_CORES=sum(ps thread %CPU)/100; EnvWorker includes ray::EnvWorker/init_worker processes",
        "",
        f"rollout_total: {summarize(totals)}",
    ]
    for role in ["EnvWorker", "RolloutWorker", "ActorWorker", "CompileWorker", "Main", "raylet", "gcs_server"]:
        summary_lines.append(f"{role}: {summarize(role_values[role])}")
    summary_lines.append(f"whole_run_total: {summarize(whole_totals)}")
    if metrics:
        metric_keys = [
            "generate_rollouts",
            "rollout/generate_one_epoch",
            "env/interact",
            "env/env_interact_step",
            "rollout/predict",
            "step",
            "num_trajectories",
            "return",
        ]
        summary_lines.extend(["", "Runtime metrics:"])
        for key in metric_keys:
            if key in metrics:
                summary_lines.append(f"{key}={metrics[key]}")
    gpu_lines = gpu_summary(profile_dir, start, end)
    if gpu_lines:
        summary_lines.extend(["", "GPU rollout utilization by index:", *gpu_lines])
    if evidence:
        summary_lines.extend(["", "Binding evidence examples:", *evidence[:8]])
    (profile_dir / "profile_summary.txt").write_text("\n".join(summary_lines) + "\n")
    print("\n".join(summary_lines[:30]))
    print(f"WROTE={profile_dir / 'profile_summary.txt'}")
    print(f"WROTE={profile_dir / 'cpu_utilization_timeseries.png'}")
    print(f"WROTE={profile_dir / 'cpu_utilization_rollout_window.png'}")
    print(f"WROTE={profile_dir / 'gpu_utilization_timeseries.png'}")
    print(f"WROTE={profile_dir / 'gpu_utilization_epoch_window.png'}")


if __name__ == "__main__":
    main()
