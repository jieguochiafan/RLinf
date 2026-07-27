#!/usr/bin/env python3
"""High-resolution role-level CPU monitor using /proc jiffies deltas."""

from __future__ import annotations

import argparse
import csv
import os
import re
import time
from collections import defaultdict
from pathlib import Path


ROLE_ORDER = [
    "EnvWorker",
    "RolloutWorker",
    "ActorWorker",
    "CompileWorker",
    "Main",
    "raylet",
    "gcs_server",
    "Other",
]

INHERITED_ROLES = {"EnvWorker", "RolloutWorker", "ActorWorker"}

LOG_ROLE_RE = re.compile(
    r"\((?P<group>EnvGroup|RolloutGroup|MultiStepRolloutWorker|ActorGroup|EmbodiedFSDPActor)[^)]*\)\s+pid=(?P<pid>\d+)"
)
LOG_ROLE_MAP = {
    "EnvGroup": "EnvWorker",
    "RolloutGroup": "RolloutWorker",
    "MultiStepRolloutWorker": "RolloutWorker",
    "ActorGroup": "ActorWorker",
    "EmbodiedFSDPActor": "ActorWorker",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stop-file", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--refresh-interval", type=float, default=0.5)
    parser.add_argument("--role-log", type=Path)
    parser.add_argument("--num-cpus", type=float, default=112.0)
    parser.add_argument(
        "--mode",
        choices=("process", "thread"),
        default="process",
        help="Use process-level aggregate jiffies or per-thread jiffies.",
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def read_cmdline(pid: int) -> str:
    try:
        data = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return data.replace(b"\x00", b" ").decode(errors="replace")


def read_comm(pid: int, tid: int | None = None) -> str:
    path = Path(f"/proc/{pid}/task/{tid}/comm") if tid is not None else Path(f"/proc/{pid}/comm")
    return read_text(path).strip()


def read_ppid(pid: int) -> int | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(errors="replace")
    except OSError:
        return None
    close = stat.rfind(")")
    if close < 0:
        return None
    fields = stat[close + 2 :].split()
    try:
        # proc_stat fields after comm start at field 3, so ppid is index 1.
        return int(fields[1])
    except (IndexError, ValueError):
        return None


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


class RoleLogParser:
    def __init__(self, path: Path | None):
        self.path = path
        self.offset = 0
        self.pid_roles: dict[int, str] = {}

    def refresh(self) -> dict[int, str]:
        if self.path is None:
            return self.pid_roles
        try:
            size = self.path.stat().st_size
        except OSError:
            return self.pid_roles
        if size < self.offset:
            self.offset = 0
        try:
            with self.path.open("r", errors="replace") as handle:
                handle.seek(self.offset)
                for raw in handle:
                    clean = strip_ansi(raw)
                    for match in LOG_ROLE_RE.finditer(clean):
                        role = LOG_ROLE_MAP[match.group("group")]
                        self.pid_roles[int(match.group("pid"))] = role
                self.offset = handle.tell()
        except OSError:
            return self.pid_roles
        return self.pid_roles


def classify(pid: int, comm: str, cmdline: str, pid_roles: dict[int, str]) -> str | None:
    if pid in pid_roles:
        return pid_roles[pid]
    text = f"{comm} {cmdline}"
    if "ray::IDLE" in text:
        return "Ignore"
    if "ray::EnvWorker" in text or "ray::EnvGroup" in text or "EnvGroup" in text:
        return "EnvWorker"
    if "ray::MultiStepRolloutWorker" in text or "ray::RolloutGroup" in text or "RolloutGroup" in text:
        return "RolloutWorker"
    if "ray::EmbodiedFSDPActor" in text or "ray::ActorGroup" in text or "ActorGroup" in text:
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


def resolve_descendant_roles(
    pids: list[int],
    pid_roles: dict[int, str],
    ppid_by_pid: dict[int, int | None],
) -> dict[int, str]:
    resolved = dict(pid_roles)
    pid_set = set(pids)

    def inherited_role(pid: int, seen: set[int]) -> str | None:
        if pid in resolved:
            role = resolved[pid]
            return role if role in INHERITED_ROLES else None
        if pid in seen:
            return None
        seen.add(pid)
        parent = ppid_by_pid.get(pid)
        if parent is None or parent not in pid_set:
            return None
        role = inherited_role(parent, seen)
        if role is not None:
            resolved[pid] = role
        return role

    for pid in pids:
        inherited_role(pid, set())
    return resolved


def parse_stat_jiffies(path: Path) -> int | None:
    try:
        stat = path.read_text(errors="replace")
    except OSError:
        return None
    close = stat.rfind(")")
    if close < 0:
        return None
    fields = stat[close + 2 :].split()
    try:
        # proc_stat fields after comm start at field 3, so utime/stime are indices 11/12.
        return int(fields[11]) + int(fields[12])
    except (IndexError, ValueError):
        return None


def parse_thread_jiffies(pid: int, tid: int) -> int | None:
    return parse_stat_jiffies(Path(f"/proc/{pid}/task/{tid}/stat"))


def parse_process_jiffies(pid: int) -> int | None:
    return parse_stat_jiffies(Path(f"/proc/{pid}/stat"))


def iter_processes(pid_roles: dict[int, str]) -> list[tuple[int, int, str]]:
    rows: list[tuple[int, int, str]] = []
    candidates: list[tuple[int, str, str]] = []
    ppid_by_pid: dict[int, int | None] = {}
    for proc_entry in Path("/proc").iterdir():
        if not proc_entry.name.isdigit():
            continue
        pid = int(proc_entry.name)
        cmdline = read_cmdline(pid)
        pid_comm = read_comm(pid)
        text = f"{pid_comm} {cmdline}"
        ppid_by_pid[pid] = read_ppid(pid)
        candidates.append((pid, pid_comm, cmdline))
        if not any(
            token in text
            for token in (
                "train_embodied_agent.py",
                "ray::",
                "ActorGroup",
                "RolloutGroup",
                "EnvGroup",
                "EnvWorker",
                "EmbodiedFSDPActor",
                "MultiStepRolloutWorker",
                "compile_worker",
                "raylet",
                "gcs_server",
            )
        ) and pid not in pid_roles:
            continue
    resolved_roles = resolve_descendant_roles(
        [pid for pid, _, _ in candidates],
        pid_roles,
        ppid_by_pid,
    )
    for pid, pid_comm, cmdline in candidates:
        role = classify(pid, pid_comm, cmdline, resolved_roles) or "Other"
        if role == "Ignore":
            continue
        if role == "Other" and pid not in resolved_roles:
            text = f"{pid_comm} {cmdline}"
            if not any(
                token in text
                for token in (
                    "train_embodied_agent.py",
                    "ray::",
                    "ActorGroup",
                    "RolloutGroup",
                    "EnvGroup",
                    "EnvWorker",
                    "EmbodiedFSDPActor",
                    "MultiStepRolloutWorker",
                    "compile_worker",
                    "raylet",
                    "gcs_server",
                )
            ):
                continue
        rows.append((pid, pid, role))
    return rows


def iter_threads(pid_roles: dict[int, str]) -> list[tuple[int, int, str]]:
    rows: list[tuple[int, int, str]] = []
    for proc_entry in Path("/proc").iterdir():
        if not proc_entry.name.isdigit():
            continue
        pid = int(proc_entry.name)
        cmdline = read_cmdline(pid)
        pid_comm = read_comm(pid)
        if not any(
            token in f"{pid_comm} {cmdline}"
            for token in (
                "train_embodied_agent.py",
                "ray::",
                "ActorGroup",
                "RolloutGroup",
                "EnvGroup",
                "EnvWorker",
                "EmbodiedFSDPActor",
                "MultiStepRolloutWorker",
                "compile_worker",
                "raylet",
                "gcs_server",
            )
        ) and pid not in pid_roles:
            continue
        task_dir = proc_entry / "task"
        try:
            task_entries = list(task_dir.iterdir())
        except OSError:
            continue
        for task_entry in task_entries:
            if not task_entry.name.isdigit():
                continue
            tid = int(task_entry.name)
            comm = read_comm(pid, tid) or pid_comm
            role = classify(pid, comm, cmdline, pid_roles) or "Other"
            if role == "Ignore":
                continue
            rows.append((pid, tid, role))
    return rows


def snapshot(
    cached_threads: list[tuple[int, int, str]] | None,
    pid_roles: dict[int, str],
    mode: str,
) -> tuple[dict[tuple[int, int], tuple[int, str]], int, list[tuple[int, int, str]]]:
    data: dict[tuple[int, int], tuple[int, str]] = {}
    other_count = 0
    iterator = iter_processes if mode == "process" else iter_threads
    threads = iterator(pid_roles) if cached_threads is None else cached_threads
    live_threads: list[tuple[int, int, str]] = []
    for pid, tid, role in threads:
        if pid in pid_roles and role != pid_roles[pid]:
            role = pid_roles[pid]
        jiffies = parse_process_jiffies(pid) if mode == "process" else parse_thread_jiffies(pid, tid)
        if jiffies is None:
            continue
        live_threads.append((pid, tid, role))
        if role == "Other":
            other_count += 1
        data[(pid, tid)] = (jiffies, role)
    return data, other_count, live_threads


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    clk_tck = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    role_log = RoleLogParser(args.role_log)
    start_wall = time.time()
    prev_time = time.monotonic()
    pid_roles = role_log.refresh()
    prev, _, cached_threads = snapshot(None, pid_roles, args.mode)
    last_refresh = prev_time
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "timestamp",
                "time_s",
                "elapsed_s",
                "total_cores",
                "total_util_pct",
                *[f"{role}_cores" for role in ROLE_ORDER],
                "tracked_items",
                "other_items",
            ]
        )
        while not args.stop_file.exists():
            time.sleep(args.interval)
            now = time.monotonic()
            pid_roles = role_log.refresh()
            refresh_threads = now - last_refresh >= args.refresh_interval
            curr, other_count, cached_threads = snapshot(
                None if refresh_threads else cached_threads,
                pid_roles,
                args.mode,
            )
            if refresh_threads:
                last_refresh = now
            elapsed = max(now - prev_time, 1e-9)
            role_cores: dict[str, float] = defaultdict(float)
            for key, (jiffies, role) in curr.items():
                prev_item = prev.get(key)
                if prev_item is None:
                    continue
                prev_jiffies, prev_role = prev_item
                role = role if role != "Other" else prev_role
                delta = max(jiffies - prev_jiffies, 0)
                role_cores[role] += delta / clk_tck / elapsed
            total_cores = sum(role_cores.values())
            writer.writerow(
                [
                    f"{time.time():.6f}",
                    f"{time.time() - start_wall:.6f}",
                    f"{elapsed:.6f}",
                    f"{total_cores:.6f}",
                    f"{total_cores / args.num_cpus * 100.0:.6f}",
                    *[f"{role_cores.get(role, 0.0):.6f}" for role in ROLE_ORDER],
                    len(curr),
                    other_count,
                ]
            )
            handle.flush()
            prev = curr
            prev_time = now


if __name__ == "__main__":
    main()
