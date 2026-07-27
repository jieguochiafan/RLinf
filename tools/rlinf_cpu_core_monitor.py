#!/usr/bin/env python3
"""Record per-thread CPU time for RLinf and its Ray worker process trees."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from types import FrameType

CSV_FIELDS = (
    "interval_start",
    "interval_end",
    "pid",
    "tid",
    "component",
    "rank",
    "cpu",
    "cpu_time_s",
    "migrated",
)

_SIGNAL_RECEIVED = False


@dataclass(frozen=True)
class ThreadSnapshot:
    """A thread's cumulative CPU time and latest processor."""

    jiffies: int
    processor: int
    starttime: int


@dataclass(frozen=True)
class ThreadDelta:
    """CPU time consumed between two thread snapshots."""

    cpu: int
    cpu_time_s: float
    migrated: bool


@dataclass(frozen=True)
class ProcessIdentity:
    """RLinf component metadata attached to a process."""

    pid: int
    component: str
    rank: int | None


def parse_task_stat(text: str) -> ThreadSnapshot:
    """Parse jiffies and processor from a proc task stat record."""
    close_paren = text.rfind(")")
    open_paren = text.find("(")
    if open_paren < 0 or close_paren <= open_paren:
        raise ValueError(f"Malformed proc stat line: {text!r}")
    fields = text[close_paren + 1 :].split()
    try:
        return ThreadSnapshot(
            jiffies=int(fields[11]) + int(fields[12]),
            processor=int(fields[36]),
            starttime=int(fields[19]),
        )
    except (IndexError, ValueError) as error:
        raise ValueError(f"Malformed proc stat line: {text!r}") from error


def parse_environ(data: bytes) -> dict[str, str]:
    """Parse a null-delimited proc environment without failing on bad bytes."""
    environ: dict[str, str] = {}
    for raw_entry in data.split(b"\0"):
        if b"=" not in raw_entry:
            continue
        raw_key, raw_value = raw_entry.split(b"=", 1)
        try:
            key = raw_key.decode("utf-8")
            value = raw_value.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if key:
            environ[key] = value
    return environ


def compute_thread_delta(
    before: ThreadSnapshot,
    after: ThreadSnapshot,
    clk_tck: int,
) -> ThreadDelta | None:
    """Compute a thread delta attributed to its processor after sampling."""
    if clk_tck <= 0:
        raise ValueError("clk_tck must be positive")
    if before.starttime != after.starttime:
        return None
    delta_jiffies = max(0, after.jiffies - before.jiffies)
    return ThreadDelta(
        cpu=after.processor,
        cpu_time_s=delta_jiffies / clk_tck,
        migrated=before.processor != after.processor,
    )


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return None


def _read_bytes(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except OSError:
        return None


def _parse_ppid(stat_text: str) -> int:
    close_paren = stat_text.rfind(")")
    open_paren = stat_text.find("(")
    if open_paren < 0 or close_paren <= open_paren:
        raise ValueError(f"Malformed proc stat line: {stat_text!r}")
    fields = stat_text[close_paren + 1 :].split()
    try:
        return int(fields[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"Malformed proc stat line: {stat_text!r}") from error


def _component_from_text(text: str) -> str | None:
    lower_text = text.lower()
    if any(token in lower_text for token in ("envgroup", "envworker")):
        return "env"
    if any(
        token in lower_text
        for token in ("rolloutgroup", "rolloutworker", "generationworker")
    ):
        return "generation"
    if any(
        token in lower_text
        for token in ("actorgroup", "actorworker", "embodiedfsdpactor")
    ):
        return "training"
    return None


def _is_training_main(text: str) -> bool:
    lower_text = text.lower()
    entrypoints = (
        "train_embodied_agent.py",
        "train_async.py",
        "train_offline_rl.py",
        "train_reasoning.py",
        "train_coding_agent.py",
    )
    return any(entrypoint in lower_text for entrypoint in entrypoints) or bool(
        re.search(
            r"examples/(?:agent|reasoning|sft|reward)/[^\0 ]*/?train[^\0 ]*\.py",
            lower_text,
        )
    )


def _parse_rank(environ: dict[str, str], text: str) -> int | None:
    candidates = [environ.get("RANK")]
    worker_name = environ.get("WORKER_NAME", "")
    if worker_name:
        candidates.append(worker_name.rsplit(":", 1)[-1])
    match = re.search(r"(?:rank[=:_-]?|:)(\d+)(?:\D|$)", text, re.IGNORECASE)
    if match:
        candidates.append(match.group(1))
    for value in candidates:
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return None


def _explicit_identity(
    pid: int,
    environ: dict[str, str],
    comm: str,
    cmdline: str,
) -> ProcessIdentity | None:
    env_text = " ".join(
        value
        for value in (environ.get("GROUP_NAME"), environ.get("WORKER_NAME"))
        if value
    )
    fallback_text = f"{comm} {cmdline}"
    if env_text:
        component = _component_from_text(env_text) or "other"
        return ProcessIdentity(pid, component, _parse_rank(environ, fallback_text))
    component = _component_from_text(fallback_text)
    if component is not None:
        return ProcessIdentity(pid, component, _parse_rank(environ, fallback_text))
    if _is_training_main(fallback_text):
        return ProcessIdentity(pid, "main", _parse_rank(environ, fallback_text))
    return None


def discover_rlinf_processes(
    proc_root: Path = Path("/proc"),
    run_id: str | None = None,
) -> dict[int, ProcessIdentity]:
    """Discover RLinf roots and recursively attach their identity to descendants."""
    process_data: dict[int, tuple[int, ProcessIdentity | None]] = {}
    run_id_conflicts: set[int] = set()
    try:
        proc_entries = list(proc_root.iterdir())
    except OSError as error:
        raise OSError(f"Cannot enumerate proc root {proc_root}: {error}") from error

    for proc_dir in proc_entries:
        if not proc_dir.name.isdigit() or not proc_dir.is_dir():
            continue
        pid = int(proc_dir.name)
        stat_text = _read_text(proc_dir / "stat")
        if stat_text is None:
            continue
        try:
            ppid = _parse_ppid(stat_text)
        except ValueError:
            continue
        raw_environ = _read_bytes(proc_dir / "environ")
        environ = parse_environ(raw_environ) if raw_environ is not None else {}
        raw_cmdline = _read_bytes(proc_dir / "cmdline")
        cmdline = (
            raw_cmdline.replace(b"\0", b" ").decode("utf-8", errors="replace")
            if raw_cmdline is not None
            else ""
        )
        comm = _read_text(proc_dir / "comm") or ""
        identity = _explicit_identity(pid, environ, comm.strip(), cmdline.strip())
        if run_id:
            process_run_id = environ.get("RLINF_RESOURCE_PROFILE_RUN_ID")
            if process_run_id and process_run_id != run_id:
                run_id_conflicts.add(pid)
                identity = None
            elif process_run_id != run_id:
                identity = None
            elif identity is None:
                identity = ProcessIdentity(
                    pid, "other", _parse_rank(environ, f"{comm} {cmdline}")
                )
        process_data[pid] = (ppid, identity)

    identities = {
        pid: identity
        for pid, (_, identity) in process_data.items()
        if identity is not None
    }
    changed = True
    while changed:
        changed = False
        for pid, (ppid, explicit_identity) in process_data.items():
            if pid in identities or pid in run_id_conflicts or ppid not in identities:
                continue
            parent = identities[ppid]
            identities[pid] = ProcessIdentity(pid, parent.component, parent.rank)
            changed = True
    return dict(sorted(identities.items()))


def snapshot_threads(
    identities: dict[int, ProcessIdentity],
    proc_root: Path = Path("/proc"),
) -> dict[tuple[int, int], tuple[ThreadSnapshot, ProcessIdentity]]:
    """Read every available thread snapshot for the supplied processes."""
    snapshots: dict[tuple[int, int], tuple[ThreadSnapshot, ProcessIdentity]] = {}
    for pid, identity in identities.items():
        task_root = proc_root / str(pid) / "task"
        try:
            task_entries = list(task_root.iterdir())
        except OSError:
            continue
        for task_dir in task_entries:
            if not task_dir.name.isdigit():
                continue
            tid = int(task_dir.name)
            stat_text = _read_text(task_dir / "stat")
            if stat_text is None:
                continue
            try:
                snapshot = parse_task_stat(stat_text)
            except ValueError:
                continue
            snapshots[(pid, tid)] = (snapshot, identity)
    return snapshots


def _handle_signal(_signum: int, _frame: FrameType | None) -> None:
    global _SIGNAL_RECEIVED
    _SIGNAL_RECEIVED = True


def _stop_requested(stop_file: Path) -> bool:
    return _SIGNAL_RECEIVED or stop_file.exists()


def run_monitor(
    output: Path,
    stop_file: Path,
    interval: float = 0.1,
    refresh_interval: float = 1.0,
    proc_root: Path = Path("/proc"),
    clk_tck: int | None = None,
    run_id: str | None = None,
) -> None:
    """Sample RLinf threads until a stop file or process signal is observed."""
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("interval must be finite and positive")
    if not math.isfinite(refresh_interval) or refresh_interval <= 0:
        raise ValueError("refresh_interval must be finite and positive")
    if clk_tck is None:
        clk_tck = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    if clk_tck <= 0:
        raise ValueError("clk_tck must be positive")

    output.parent.mkdir(parents=True, exist_ok=True)
    previous: dict[tuple[int, int], tuple[ThreadSnapshot, ProcessIdentity]] = {}
    previous_wall: float | None = None
    identities: dict[int, ProcessIdentity] = {}
    next_deadline = time.monotonic()
    next_refresh = next_deadline
    with output.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        output_file.flush()
        while not _stop_requested(stop_file):
            delay = next_deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
                if _stop_requested(stop_file):
                    break
            sample_start = time.monotonic()
            if sample_start >= next_refresh:
                discovered = discover_rlinf_processes(proc_root, run_id=run_id)
                changed_pids = {
                    pid
                    for pid in identities.keys() | discovered.keys()
                    if identities.get(pid) != discovered.get(pid)
                }
                if changed_pids:
                    previous = {
                        key: value
                        for key, value in previous.items()
                        if key[0] not in changed_pids
                    }
                identities = discovered
                next_refresh = sample_start + refresh_interval
            current = snapshot_threads(identities, proc_root)
            current_wall = time.time()
            invalidated_pids: set[int] = set()
            if previous_wall is not None:
                invalidated_pids = {
                    pid
                    for (pid, tid), (after, _) in current.items()
                    if (before_entry := previous.get((pid, tid))) is not None
                    and before_entry[0].starttime != after.starttime
                }
                for (pid, tid), (after, identity) in sorted(current.items()):
                    if pid in invalidated_pids:
                        continue
                    before_entry = previous.get((pid, tid))
                    if before_entry is None:
                        continue
                    delta = compute_thread_delta(before_entry[0], after, clk_tck)
                    if delta is None:
                        continue
                    writer.writerow(
                        {
                            "interval_start": previous_wall,
                            "interval_end": current_wall,
                            "pid": pid,
                            "tid": tid,
                            "component": identity.component,
                            "rank": "" if identity.rank is None else identity.rank,
                            "cpu": delta.cpu,
                            "cpu_time_s": delta.cpu_time_s,
                            "migrated": str(delta.migrated).lower(),
                        }
                    )
            if invalidated_pids:
                identities = {
                    pid: identity
                    for pid, identity in identities.items()
                    if pid not in invalidated_pids
                }
                previous = {
                    key: value
                    for key, value in previous.items()
                    if key[0] not in invalidated_pids
                }
                current = {
                    key: value
                    for key, value in current.items()
                    if key[0] not in invalidated_pids
                }
            output_file.flush()
            previous = current
            previous_wall = current_wall
            identities = {
                pid: identity
                for pid, identity in identities.items()
                if (proc_root / str(pid)).exists()
            }
            sample_end = time.monotonic()
            scheduled_next = next_deadline + interval
            next_deadline = (
                sample_end + interval
                if sample_end >= scheduled_next
                else scheduled_next
            )


def _positive_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def parse_args() -> argparse.Namespace:
    """Parse command-line monitor settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--interval", type=_positive_finite_float, default=0.1)
    parser.add_argument("--refresh-interval", type=_positive_finite_float, default=1.0)
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"))
    parser.add_argument(
        "--run-id", default=os.environ.get("RLINF_RESOURCE_PROFILE_RUN_ID", "")
    )
    return parser.parse_args()


def main() -> None:
    """Run the command-line monitor and report fatal output errors."""
    args = parse_args()
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    try:
        run_monitor(
            args.output,
            args.stop_file,
            interval=args.interval,
            refresh_interval=args.refresh_interval,
            proc_root=args.proc_root,
            run_id=args.run_id or None,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"rlinf_cpu_core_monitor: {error}") from error


if __name__ == "__main__":
    main()
