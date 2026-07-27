#!/usr/bin/env python3
"""Sample active CPU-core utilization for matching worker processes."""

from __future__ import annotations

import argparse
import csv
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

CLK_TCK = os.sysconf(os.sysconf_names["SC_CLK_TCK"])


@dataclass(frozen=True)
class WorkerProcess:
    pid: int
    label: str
    cores: tuple[int, ...]


@dataclass(frozen=True)
class ProcSample:
    timestamp: float
    proc_jiffies: int
    core_times: dict[int, tuple[int, int]]


def parse_proc_stat_core_times(text: str) -> dict[int, tuple[int, int]]:
    """Return ``(busy, total)`` jiffies per CPU core from /proc/stat text."""
    times: dict[int, tuple[int, int]] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5 or not re.fullmatch(r"cpu\d+", parts[0]):
            continue
        cpu_id = int(parts[0][3:])
        values = [int(value) for value in parts[1:]]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        total = sum(values)
        times[cpu_id] = (total - idle, total)
    return times


def parse_proc_task_jiffies(stat_text: str) -> int:
    """Return utime+stime from a /proc/<pid>/stat style line."""
    close_paren = stat_text.rfind(")")
    if close_paren < 0:
        raise ValueError(f"Malformed proc stat line: {stat_text!r}")
    fields = stat_text[close_paren + 2 :].split()
    if len(fields) <= 12:
        raise ValueError(f"Malformed proc stat line: {stat_text!r}")
    return int(fields[11]) + int(fields[12])


def compute_worker_core_utilization(
    before: ProcSample,
    after: ProcSample,
    worker: WorkerProcess,
) -> dict[int, float]:
    """Compute hardware busy percent for each core in a worker's affinity set."""
    if not worker.cores:
        return dict.fromkeys(worker.cores, 0.0)
    utilization: dict[int, float] = {}
    for core in worker.cores:
        before_busy, before_total = before.core_times.get(core, (0, 0))
        after_busy, after_total = after.core_times.get(core, (0, 0))
        busy_delta = max(0, after_busy - before_busy)
        total_delta = max(0, after_total - before_total)
        utilization[core] = 0.0 if total_delta == 0 else 100.0 * busy_delta / total_delta
    return utilization


def read_proc_jiffies(pid: int) -> int | None:
    total = 0
    task_dir = Path(f"/proc/{pid}/task")
    try:
        tids = [entry.name for entry in task_dir.iterdir()]
    except OSError:
        return None
    for tid in tids:
        try:
            total += parse_proc_task_jiffies(Path(f"/proc/{pid}/task/{tid}/stat").read_text())
        except OSError:
            continue
    return total


def read_core_times() -> dict[int, tuple[int, int]]:
    return parse_proc_stat_core_times(Path("/proc/stat").read_text())


def read_affinity(pid: int) -> tuple[int, ...]:
    try:
        return tuple(sorted(os.sched_getaffinity(pid)))
    except OSError:
        return ()


def read_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()


def discover_workers(pattern: str) -> list[WorkerProcess]:
    regex = re.compile(pattern)
    workers: list[WorkerProcess] = []
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        pid = int(proc_dir.name)
        cmdline = read_cmdline(pid)
        if not cmdline or not regex.search(cmdline):
            continue
        cores = read_affinity(pid)
        if cores:
            workers.append(WorkerProcess(pid=pid, label=cmdline[:160], cores=cores))
    return sorted(workers, key=lambda worker: worker.pid)


def sample_worker(worker: WorkerProcess) -> ProcSample | None:
    proc_jiffies = read_proc_jiffies(worker.pid)
    if proc_jiffies is None:
        return None
    return ProcSample(
        timestamp=time.time(),
        proc_jiffies=proc_jiffies,
        core_times=read_core_times(),
    )


def run_sampler(outdir: Path, interval: float, pattern: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "worker_cpu_core_util.csv"
    workers_path = outdir / "worker_cpu_cores.csv"
    known_workers: dict[int, WorkerProcess] = {}
    previous: dict[int, ProcSample] = {}
    with (
        csv_path.open("w", encoding="utf-8", newline="", buffering=1) as util_file,
        workers_path.open("w", encoding="utf-8", newline="", buffering=1) as workers_file,
    ):
        writer = csv.writer(util_file)
        workers_writer = csv.writer(workers_file)
        writer.writerow(
            [
                "timestamp",
                "pid",
                "worker_label",
                "cpu",
                "active_pct",
                "proc_delta_jiffies",
                "core_busy_delta_jiffies",
                "core_total_delta_jiffies",
            ]
        )
        workers_writer.writerow(["timestamp", "pid", "cores", "label"])
        while True:
            for worker in discover_workers(pattern):
                if worker.pid not in known_workers:
                    known_workers[worker.pid] = worker
                    workers_writer.writerow(
                        [
                            time.time(),
                            worker.pid,
                            " ".join(map(str, worker.cores)),
                            worker.label,
                        ]
                    )
                else:
                    worker = known_workers[worker.pid]

                if not Path(f"/proc/{worker.pid}").exists():
                    known_workers.pop(worker.pid, None)
                    previous.pop(worker.pid, None)
                    continue

                current = sample_worker(worker)
                if current is None:
                    continue
                prev = previous.get(worker.pid)
                if prev is not None:
                    util = compute_worker_core_utilization(prev, current, worker)
                    for core, active_pct in util.items():
                        prev_busy, prev_total = prev.core_times.get(core, (0, 0))
                        curr_busy, curr_total = current.core_times.get(core, (0, 0))
                        writer.writerow(
                            [
                                current.timestamp,
                                worker.pid,
                                worker.label,
                                core,
                                f"{active_pct:.6f}",
                                current.proc_jiffies - prev.proc_jiffies,
                                curr_busy - prev_busy,
                                curr_total - prev_total,
                            ]
                        )
                previous[worker.pid] = current
            time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outdir", type=Path, required=True)
    parser.add_argument("-i", "--interval", type=float, default=0.1)
    parser.add_argument(
        "-p",
        "--pattern",
        default=r"ray::|EnvWorker|RolloutWorker|ActorWorker|train_embodied_agent",
    )
    args = parser.parse_args()
    run_sampler(args.outdir, args.interval, args.pattern)


if __name__ == "__main__":
    main()
