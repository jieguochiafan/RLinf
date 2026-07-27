#!/usr/bin/env python3
"""Sample per-worker host RSS and GPU memory from procfs and nvidia-smi."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

try:
    from highres_cpu_monitor import RoleLogParser, classify, read_cmdline, read_comm
except ModuleNotFoundError:
    from tools.highres_cpu_monitor import (
        RoleLogParser,
        classify,
        read_cmdline,
        read_comm,
    )


@dataclass(frozen=True)
class ComputeAppMemory:
    gpu_uuid: str
    used_memory_mib: float


def parse_status_rss_mib(text: str) -> float:
    for line in text.splitlines():
        if not line.startswith("VmRSS:"):
            continue
        fields = line.split()
        if len(fields) < 2:
            return 0.0
        return float(fields[1]) / 1024.0
    return 0.0


def read_rss_mib(pid: int) -> float | None:
    try:
        return parse_status_rss_mib(Path(f"/proc/{pid}/status").read_text())
    except OSError:
        return None


def parse_nvidia_compute_apps(text: str) -> dict[int, ComputeAppMemory]:
    rows: dict[int, ComputeAppMemory] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("pid"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            used_memory_mib = float(re.sub(r"[^0-9.]+", "", parts[2]) or 0.0)
        except ValueError:
            continue
        rows[pid] = ComputeAppMemory(
            gpu_uuid=parts[1],
            used_memory_mib=used_memory_mib,
        )
    return rows


def parse_gpu_index_uuid_map(text: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("index"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        mapping[parts[1]] = parts[0]
    return mapping


def run_nvidia_smi(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", *args],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""


def read_gpu_uuid_index_map() -> dict[str, str]:
    return parse_gpu_index_uuid_map(
        run_nvidia_smi(
            [
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ]
        )
    )


def read_gpu_memory_by_pid() -> dict[int, ComputeAppMemory]:
    return parse_nvidia_compute_apps(
        run_nvidia_smi(
            [
                "--query-compute-apps=pid,gpu_uuid,used_memory",
                "--format=csv,noheader,nounits",
            ]
        )
    )


def discover_processes(pid_roles: dict[int, str]) -> list[tuple[int, str, str, str]]:
    rows: list[tuple[int, str, str, str]] = []
    for proc_entry in Path("/proc").iterdir():
        if not proc_entry.name.isdigit():
            continue
        pid = int(proc_entry.name)
        cmdline = read_cmdline(pid)
        comm = read_comm(pid)
        role = classify(pid, comm, cmdline, pid_roles)
        if role in {None, "Ignore"} and pid not in pid_roles:
            text = f"{comm} {cmdline}"
            if not any(
                token in text
                for token in (
                    "train_embodied_agent.py",
                    "train_async.py",
                    "ray::",
                    "ActorGroup",
                    "RolloutGroup",
                    "EnvGroup",
                    "EnvWorker",
                    "EmbodiedFSDPActor",
                    "AsyncPPOEmbodiedFSDPActor",
                    "MultiStepRolloutWorker",
                    "compile_worker",
                    "raylet",
                    "gcs_server",
                )
            ):
                continue
        if role in {None, "Ignore"}:
            role = "Other"
        rows.append((pid, role, comm, cmdline))
    return sorted(rows)


def run_sampler(
    output: Path,
    stop_file: Path,
    *,
    interval: float,
    role_log: Path | None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    role_parser = RoleLogParser(role_log)
    gpu_indices_by_uuid = read_gpu_uuid_index_map()
    start_wall = time.time()
    with output.open("w", encoding="utf-8", newline="", buffering=1) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "timestamp",
                "time_s",
                "pid",
                "role",
                "comm",
                "rss_mib",
                "gpu_index",
                "gpu_uuid",
                "gpu_memory_mib",
                "cmdline",
            ]
        )
        while not stop_file.exists():
            timestamp = time.time()
            pid_roles = role_parser.refresh()
            gpu_memory_by_pid = read_gpu_memory_by_pid()
            for pid, role, comm, cmdline in discover_processes(pid_roles):
                rss_mib = read_rss_mib(pid)
                if rss_mib is None:
                    continue
                gpu_memory = gpu_memory_by_pid.get(pid)
                gpu_uuid = gpu_memory.gpu_uuid if gpu_memory is not None else ""
                writer.writerow(
                    [
                        f"{timestamp:.6f}",
                        f"{timestamp - start_wall:.6f}",
                        pid,
                        role,
                        comm,
                        f"{rss_mib:.6f}",
                        gpu_indices_by_uuid.get(gpu_uuid, ""),
                        gpu_uuid,
                        ""
                        if gpu_memory is None
                        else f"{gpu_memory.used_memory_mib:.6f}",
                        cmdline[:300],
                    ]
                )
            time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stop-file", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--role-log", type=Path)
    args = parser.parse_args()

    run_sampler(
        output=args.output,
        stop_file=args.stop_file,
        interval=args.interval,
        role_log=args.role_log,
    )


if __name__ == "__main__":
    main()
