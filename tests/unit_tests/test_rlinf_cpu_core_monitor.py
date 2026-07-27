from __future__ import annotations

import csv
import importlib.util
import io
import signal
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "tools" / "rlinf_cpu_core_monitor.py"
)
SPEC = importlib.util.spec_from_file_location("rlinf_cpu_core_monitor", SCRIPT_PATH)
assert SPEC is not None
monitor = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = monitor
SPEC.loader.exec_module(monitor)

ProcessIdentity = monitor.ProcessIdentity
ThreadSnapshot = monitor.ThreadSnapshot
compute_thread_delta = monitor.compute_thread_delta
discover_rlinf_processes = monitor.discover_rlinf_processes
parse_environ = monitor.parse_environ
parse_task_stat = monitor.parse_task_stat
run_monitor = monitor.run_monitor
snapshot_threads = monitor.snapshot_threads


def _stat(
    pid: int,
    comm: str,
    *,
    ppid: int = 1,
    utime: int = 0,
    stime: int = 0,
    starttime: int = 1,
    processor: int = 0,
) -> str:
    fields = ["S", str(ppid), *(["0"] * 35)]
    fields[11] = str(utime)
    fields[12] = str(stime)
    fields[19] = str(starttime)
    fields[36] = str(processor)
    return f"{pid} ({comm}) {' '.join(fields)}\n"


def _write_process(
    proc_root: Path,
    pid: int,
    *,
    comm: str,
    cmdline: str,
    environ: dict[str, str] | None = None,
    ppid: int = 1,
    threads: dict[int, tuple[int, int, int]] | None = None,
) -> None:
    proc_dir = proc_root / str(pid)
    proc_dir.mkdir(parents=True)
    (proc_dir / "stat").write_text(_stat(pid, comm, ppid=ppid))
    (proc_dir / "comm").write_text(f"{comm}\n")
    (proc_dir / "cmdline").write_bytes(cmdline.replace(" ", "\0").encode())
    env = environ or {}
    (proc_dir / "environ").write_bytes(
        b"\0".join(f"{key}={value}".encode() for key, value in env.items())
        + (b"\0" if env else b"")
    )
    for tid, (utime, stime, processor) in (threads or {}).items():
        task_dir = proc_dir / "task" / str(tid)
        task_dir.mkdir(parents=True)
        (task_dir / "stat").write_text(
            _stat(
                tid,
                comm,
                ppid=ppid,
                utime=utime,
                stime=stime,
                processor=processor,
            )
        )


def test_parse_task_stat_handles_comm_with_spaces() -> None:
    snapshot = parse_task_stat(
        _stat(42, "ray worker name", utime=17, stime=8, processor=23)
    )

    assert snapshot == ThreadSnapshot(jiffies=25, processor=23, starttime=1)


def test_parse_task_stat_rejects_malformed_input() -> None:
    with pytest.raises(ValueError, match="Malformed"):
        parse_task_stat("42 missing-parentheses")


def test_parse_environ_ignores_entries_without_equals() -> None:
    raw = b"GROUP_NAME=EnvGroup\0RANK=3\0BROKEN\0VALUE=a=b\0\xff=bad\0"

    assert parse_environ(raw) == {
        "GROUP_NAME": "EnvGroup",
        "RANK": "3",
        "VALUE": "a=b",
    }


def test_compute_thread_delta_attributes_time_to_later_processor() -> None:
    delta = compute_thread_delta(
        ThreadSnapshot(jiffies=100, processor=2, starttime=10),
        ThreadSnapshot(jiffies=125, processor=7, starttime=10),
        clk_tck=100,
    )

    assert delta.cpu == 7
    assert delta.cpu_time_s == pytest.approx(0.25)
    assert delta.migrated is True


def test_compute_thread_delta_clamps_negative_jiffies() -> None:
    delta = compute_thread_delta(
        ThreadSnapshot(jiffies=100, processor=4, starttime=10),
        ThreadSnapshot(jiffies=80, processor=4, starttime=10),
        clk_tck=100,
    )

    assert delta.cpu_time_s == 0.0
    assert delta.migrated is False


def test_compute_thread_delta_rejects_reused_thread_id() -> None:
    assert (
        compute_thread_delta(
            ThreadSnapshot(jiffies=100, processor=4, starttime=10),
            ThreadSnapshot(jiffies=500, processor=7, starttime=20),
            clk_tck=100,
        )
        is None
    )


def test_discovery_prefers_environment_and_excludes_unrelated(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    _write_process(
        proc_root,
        100,
        comm="python",
        cmdline="python examples/embodiment/train_async.py",
    )
    _write_process(
        proc_root,
        200,
        comm="ray::ActorGroup",
        cmdline="ray::ActorGroup",
        environ={"GROUP_NAME": "EnvGroup", "WORKER_NAME": "EnvGroup:3", "RANK": "3"},
        ppid=100,
    )
    _write_process(proc_root, 300, comm="bash", cmdline="bash unrelated.sh")

    identities = discover_rlinf_processes(proc_root)

    assert identities == {
        100: ProcessIdentity(pid=100, component="main", rank=None),
        200: ProcessIdentity(pid=200, component="env", rank=3),
    }


def test_discovery_inherits_identity_for_all_descendants(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    _write_process(
        proc_root,
        200,
        comm="ray::RolloutGroup",
        cmdline="ray::RolloutGroup",
        environ={"GROUP_NAME": "RolloutGroup", "RANK": "5"},
    )
    _write_process(proc_root, 201, comm="python", cmdline="python child.py", ppid=200)
    _write_process(proc_root, 202, comm="helper", cmdline="helper", ppid=201)

    identities = discover_rlinf_processes(proc_root)

    expected = ProcessIdentity(pid=200, component="generation", rank=5)
    assert identities[200] == expected
    assert identities[201] == ProcessIdentity(201, "generation", 5)
    assert identities[202] == ProcessIdentity(202, "generation", 5)


def test_discovery_uses_worker_name_rank_and_actor_fallback(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    _write_process(
        proc_root,
        400,
        comm="ray::EmbodiedFSDPActor",
        cmdline="ray::EmbodiedFSDPActor",
        environ={"WORKER_NAME": "ActorGroup:7"},
    )

    assert discover_rlinf_processes(proc_root) == {
        400: ProcessIdentity(400, "training", 7)
    }


def test_discovery_excludes_generic_ray_root_but_includes_confirmed_descendant(
    tmp_path: Path,
) -> None:
    proc_root = tmp_path / "proc"
    _write_process(
        proc_root,
        500,
        comm="ray::RewardWorker",
        cmdline="ray::RewardWorker",
    )
    _write_process(
        proc_root,
        600,
        comm="python",
        cmdline="python worker.py",
        environ={"GROUP_NAME": "RewardGroup", "WORKER_NAME": "RewardGroup:2"},
    )
    _write_process(
        proc_root,
        800,
        comm="python",
        cmdline="python examples/embodiment/train_async.py",
    )
    _write_process(
        proc_root,
        801,
        comm="ray::RewardWorker",
        cmdline="ray::RewardWorker",
        ppid=800,
    )
    _write_process(proc_root, 700, comm="python", cmdline="python unrelated.py")

    assert discover_rlinf_processes(proc_root) == {
        600: ProcessIdentity(600, "other", 2),
        800: ProcessIdentity(800, "main", None),
        801: ProcessIdentity(801, "main", None),
    }


def test_discovery_filters_roots_by_run_id_and_keeps_descendants(
    tmp_path: Path,
) -> None:
    proc_root = tmp_path / "proc"
    _write_process(
        proc_root,
        100,
        comm="python",
        cmdline="python examples/embodiment/train_async.py",
        environ={"RLINF_RESOURCE_PROFILE_RUN_ID": "job-a"},
    )
    _write_process(
        proc_root,
        101,
        comm="ray::RewardWorker",
        cmdline="ray::RewardWorker",
        ppid=100,
    )
    _write_process(
        proc_root,
        200,
        comm="ray::EnvGroup",
        cmdline="ray::EnvGroup",
        environ={
            "GROUP_NAME": "EnvGroup",
            "RANK": "4",
            "RLINF_RESOURCE_PROFILE_RUN_ID": "job-b",
        },
        ppid=100,
    )

    assert discover_rlinf_processes(proc_root, run_id="job-a") == {
        100: ProcessIdentity(100, "main", None),
        101: ProcessIdentity(101, "main", None),
    }


def test_discovery_raises_when_proc_root_is_unavailable(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="Cannot enumerate proc root"):
        discover_rlinf_processes(tmp_path / "missing-proc")


def test_discovery_raises_when_proc_root_cannot_be_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    original_iterdir = Path.iterdir

    def deny_proc_root(path: Path):
        if path == proc_root:
            raise PermissionError("denied")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", deny_proc_root)

    with pytest.raises(OSError, match="Cannot enumerate proc root"):
        discover_rlinf_processes(proc_root)


def test_discovery_and_snapshot_skip_malformed_or_unreadable_proc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc_root = tmp_path / "proc"
    _write_process(
        proc_root,
        10,
        comm="ray::EnvGroup",
        cmdline="ray::EnvGroup",
        environ={"GROUP_NAME": "EnvGroup", "RANK": "1"},
        threads={10: (4, 6, 2), 11: (1, 1, 3)},
    )
    _write_process(
        proc_root,
        20,
        comm="ray::ActorGroup",
        cmdline="ray::ActorGroup",
        threads={20: (1, 2, 4)},
    )
    (proc_root / "20" / "stat").write_text("malformed")
    (proc_root / "10" / "task" / "11" / "stat").write_text("malformed")
    original_read_bytes = Path.read_bytes

    def deny_one_environ(path: Path) -> bytes:
        if path == proc_root / "10" / "environ":
            raise PermissionError("denied")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", deny_one_environ)

    identities = discover_rlinf_processes(proc_root)
    snapshots = snapshot_threads(identities, proc_root)

    assert identities == {10: ProcessIdentity(10, "env", None)}
    assert snapshots == {
        (10, 10): (ThreadSnapshot(10, 2, 1), ProcessIdentity(10, "env", None))
    }


def test_snapshot_skips_pid_that_disappears_after_discovery(tmp_path: Path) -> None:
    identity = ProcessIdentity(10, "env", 1)

    assert snapshot_threads({10: identity}, tmp_path / "proc") == {}


def test_monitor_writes_only_threads_seen_in_consecutive_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "nested" / "threads.csv"
    stop_file = tmp_path / "stop"
    env = ProcessIdentity(10, "env", 3)
    snapshots = iter(
        [
            {(10, 10): (ThreadSnapshot(100, 1, 1), env)},
            {
                (10, 10): (ThreadSnapshot(120, 2, 1), env),
                (10, 20): (ThreadSnapshot(50, 4, 2), env),
            },
            {(10, 20): (ThreadSnapshot(75, 4, 2), env)},
        ]
    )
    wall_times = iter([10.0, 10.1, 10.2])
    sample_count = 0

    def take_snapshot(_ids, _root):
        nonlocal sample_count
        sample_count += 1
        return next(snapshots)

    monkeypatch.setattr(
        monitor, "discover_rlinf_processes", lambda _root, run_id=None: {10: env}
    )
    monkeypatch.setattr(
        monitor,
        "snapshot_threads",
        take_snapshot,
    )
    monkeypatch.setattr(monitor, "_stop_requested", lambda _path: sample_count >= 3)
    monkeypatch.setattr(monitor.time, "time", lambda: next(wall_times))
    monkeypatch.setattr(monitor.time, "monotonic", lambda: 1.0)
    monkeypatch.setattr(monitor.time, "sleep", lambda _delay: None)

    run_monitor(
        output, stop_file, interval=0.1, proc_root=tmp_path / "proc", clk_tck=100
    )

    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["tid"]) for row in rows] == [10, 20]
    assert [
        (float(row["interval_start"]), float(row["interval_end"])) for row in rows
    ] == [
        (10.0, 10.1),
        (10.1, 10.2),
    ]
    assert [int(row["cpu"]) for row in rows] == [2, 4]
    assert [float(row["cpu_time_s"]) for row in rows] == pytest.approx([0.2, 0.25])
    assert [row["migrated"] for row in rows] == ["true", "false"]


class _FlushCountingFile(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.flush_count = 0

    def flush(self) -> None:
        self.flush_count += 1
        super().flush()

    def close(self) -> None:
        pass


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start
        self.sleep_delays: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, delay: float) -> None:
        self.sleep_delays.append(delay)
        self.now += delay


def test_monitor_flushes_header_and_every_sampling_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "monitor.csv"
    output_file = _FlushCountingFile()
    sample_count = 0

    def take_snapshot(_ids, _root):
        nonlocal sample_count
        sample_count += 1
        return {}

    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: output_file)
    monkeypatch.setattr(
        monitor, "discover_rlinf_processes", lambda _root, run_id=None: {}
    )
    monkeypatch.setattr(monitor, "snapshot_threads", take_snapshot)
    monkeypatch.setattr(monitor, "_stop_requested", lambda _path: sample_count >= 3)
    monkeypatch.setattr(monitor.time, "monotonic", lambda: 1.0)
    monkeypatch.setattr(monitor.time, "time", lambda: 2.0)
    monkeypatch.setattr(monitor.time, "sleep", lambda _delay: None)

    run_monitor(output, tmp_path / "stop", proc_root=tmp_path / "proc")

    assert output_file.flush_count == 4


def test_monitor_refreshes_discovery_less_often_than_sampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _FakeClock()
    discover_count = 0
    sample_count = 0

    def discover(_root, run_id=None):
        nonlocal discover_count
        discover_count += 1
        return {}

    def snapshot(_ids, _root):
        nonlocal sample_count
        sample_count += 1
        return {}

    monkeypatch.setattr(monitor, "discover_rlinf_processes", discover)
    monkeypatch.setattr(monitor, "snapshot_threads", snapshot)
    monkeypatch.setattr(monitor, "_stop_requested", lambda _path: sample_count >= 5)
    monkeypatch.setattr(monitor.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(monitor.time, "time", clock.monotonic)
    monkeypatch.setattr(monitor.time, "sleep", clock.sleep)

    run_monitor(
        tmp_path / "monitor.csv",
        tmp_path / "stop",
        interval=0.1,
        refresh_interval=1.0,
        proc_root=tmp_path / "proc",
    )

    assert sample_count == 5
    assert discover_count == 1


def test_monitor_refresh_removes_reused_pid_from_cached_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc_root = tmp_path / "proc"
    (proc_root / "10").mkdir(parents=True)
    identity = ProcessIdentity(10, "env", 3)
    clock = _FakeClock()
    discover_results = iter([{10: identity}, {}])
    sample_count = 0

    def discover(_root, run_id=None):
        return next(discover_results)

    def snapshot(identities, _root):
        nonlocal sample_count
        sample_count += 1
        if 10 not in identities:
            return {}
        if sample_count <= 2:
            thread = ThreadSnapshot(sample_count * 10, 1, 1)
        else:
            thread = ThreadSnapshot(sample_count * 10, 1, 2)
        return {(10, 10): (thread, identities[10])}

    monkeypatch.setattr(monitor, "discover_rlinf_processes", discover)
    monkeypatch.setattr(monitor, "snapshot_threads", snapshot)
    monkeypatch.setattr(monitor, "_stop_requested", lambda _path: sample_count >= 4)
    monkeypatch.setattr(monitor.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(monitor.time, "time", clock.monotonic)
    monkeypatch.setattr(monitor.time, "sleep", clock.sleep)

    output = tmp_path / "monitor.csv"
    run_monitor(
        output,
        tmp_path / "stop",
        interval=0.1,
        refresh_interval=0.15,
        proc_root=proc_root,
        clk_tck=100,
        run_id="job-a",
    )

    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert float(rows[0]["cpu_time_s"]) == pytest.approx(0.1)


def test_monitor_invalidates_reused_pid_between_discovery_refreshes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc_root = tmp_path / "proc"
    (proc_root / "10").mkdir(parents=True)
    identity = ProcessIdentity(10, "env", 3)
    clock = _FakeClock()
    discover_count = 0
    sample_count = 0

    def discover(_root, run_id=None):
        nonlocal discover_count
        discover_count += 1
        return {10: identity} if discover_count == 1 else {}

    def snapshot(identities, _root):
        nonlocal sample_count
        sample_count += 1
        if 10 not in identities:
            return {}
        starttime = 1 if sample_count == 1 else 2
        return {
            (10, 9): (ThreadSnapshot(sample_count * 10, 1, 9), identities[10]),
            (10, 10): (
                ThreadSnapshot(sample_count * 10, 1, starttime),
                identities[10],
            ),
        }

    monkeypatch.setattr(monitor, "discover_rlinf_processes", discover)
    monkeypatch.setattr(monitor, "snapshot_threads", snapshot)
    monkeypatch.setattr(monitor, "_stop_requested", lambda _path: sample_count >= 3)
    monkeypatch.setattr(monitor.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(monitor.time, "time", clock.monotonic)
    monkeypatch.setattr(monitor.time, "sleep", clock.sleep)

    output = tmp_path / "monitor.csv"
    run_monitor(
        output,
        tmp_path / "stop",
        interval=0.1,
        refresh_interval=1.0,
        proc_root=proc_root,
        clk_tck=100,
    )

    with output.open(newline="") as handle:
        assert list(csv.DictReader(handle)) == []
    assert discover_count == 1


def test_monitor_does_not_busy_scan_after_interval_overrun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _FakeClock()
    sample_count = 0

    def snapshot(_ids, _root):
        nonlocal sample_count
        sample_count += 1
        clock.now += 0.35
        return {}

    monkeypatch.setattr(
        monitor, "discover_rlinf_processes", lambda _root, run_id=None: {}
    )
    monkeypatch.setattr(monitor, "snapshot_threads", snapshot)
    monkeypatch.setattr(monitor, "_stop_requested", lambda _path: sample_count >= 2)
    monkeypatch.setattr(monitor.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(monitor.time, "time", clock.monotonic)
    monkeypatch.setattr(monitor.time, "sleep", clock.sleep)

    run_monitor(
        tmp_path / "monitor.csv",
        tmp_path / "stop",
        interval=0.1,
        proc_root=tmp_path / "proc",
    )

    assert clock.sleep_delays == pytest.approx([0.1])


def test_monitor_keeps_scheduled_deadline_when_sample_finishes_early(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _FakeClock()
    sample_count = 0

    def snapshot(_ids, _root):
        nonlocal sample_count
        sample_count += 1
        clock.now += 0.02
        return {}

    monkeypatch.setattr(
        monitor, "discover_rlinf_processes", lambda _root, run_id=None: {}
    )
    monkeypatch.setattr(monitor, "snapshot_threads", snapshot)
    monkeypatch.setattr(monitor, "_stop_requested", lambda _path: sample_count >= 2)
    monkeypatch.setattr(monitor.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(monitor.time, "time", clock.monotonic)
    monkeypatch.setattr(monitor.time, "sleep", clock.sleep)

    run_monitor(
        tmp_path / "monitor.csv",
        tmp_path / "stop",
        interval=0.1,
        proc_root=tmp_path / "proc",
    )

    assert clock.sleep_delays == pytest.approx([0.08])


def test_monitor_records_interval_end_after_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = ProcessIdentity(10, "env", 0)
    snapshots = iter(
        [
            {(10, 10): (ThreadSnapshot(10, 1, 1), env)},
            {(10, 10): (ThreadSnapshot(20, 1, 1), env)},
        ]
    )
    sample_count = 0
    wall_clock = 100.0

    def snapshot(_ids, _root):
        nonlocal sample_count, wall_clock
        sample_count += 1
        wall_clock += 0.03
        return next(snapshots)

    monkeypatch.setattr(
        monitor, "discover_rlinf_processes", lambda _root, run_id=None: {10: env}
    )
    monkeypatch.setattr(monitor, "snapshot_threads", snapshot)
    monkeypatch.setattr(monitor, "_stop_requested", lambda _path: sample_count >= 2)
    monkeypatch.setattr(monitor.time, "monotonic", lambda: sample_count * 0.1)
    monkeypatch.setattr(monitor.time, "time", lambda: wall_clock)
    monkeypatch.setattr(monitor.time, "sleep", lambda _delay: None)

    output = tmp_path / "monitor.csv"
    run_monitor(output, tmp_path / "stop", proc_root=tmp_path / "proc", clk_tck=100)

    with output.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert float(row["interval_start"]) == pytest.approx(100.03)
    assert float(row["interval_end"]) == pytest.approx(100.06)


def test_monitor_stops_after_sleep_without_another_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _FakeClock()
    sample_count = 0
    stopped = False

    def snapshot(_ids, _root):
        nonlocal sample_count
        sample_count += 1
        return {}

    def sleep(delay: float) -> None:
        nonlocal stopped
        clock.sleep(delay)
        stopped = True

    monkeypatch.setattr(
        monitor, "discover_rlinf_processes", lambda _root, run_id=None: {}
    )
    monkeypatch.setattr(monitor, "snapshot_threads", snapshot)
    monkeypatch.setattr(monitor, "_stop_requested", lambda _path: stopped)
    monkeypatch.setattr(monitor.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(monitor.time, "time", clock.monotonic)
    monkeypatch.setattr(monitor.time, "sleep", sleep)

    run_monitor(
        tmp_path / "monitor.csv", tmp_path / "stop", proc_root=tmp_path / "proc"
    )

    assert sample_count == 1


def test_stop_file_immediately_exits_after_writing_header(tmp_path: Path) -> None:
    output = tmp_path / "monitor.csv"
    stop_file = tmp_path / "stop"
    stop_file.touch()

    run_monitor(output, stop_file, proc_root=tmp_path / "proc")

    assert output.read_text().splitlines() == [
        "interval_start,interval_end,pid,tid,component,rank,cpu,cpu_time_s,migrated"
    ]


def test_parse_args_accepts_all_monitor_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "monitor",
            "--output",
            str(tmp_path / "out.csv"),
            "--stop-file",
            str(tmp_path / "stop"),
            "--interval",
            "0.25",
            "--proc-root",
            str(tmp_path / "proc"),
            "--refresh-interval",
            "2.5",
            "--run-id",
            "job-a",
        ],
    )

    args = monitor.parse_args()

    assert args.output == tmp_path / "out.csv"
    assert args.stop_file == tmp_path / "stop"
    assert args.interval == 0.25
    assert args.proc_root == tmp_path / "proc"
    assert args.refresh_interval == 2.5
    assert args.run_id == "job-a"


def test_parse_args_defaults_run_id_from_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RLINF_RESOURCE_PROFILE_RUN_ID", "job-from-env")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "monitor",
            "--output",
            str(tmp_path / "out.csv"),
            "--stop-file",
            str(tmp_path / "stop"),
        ],
    )

    assert monitor.parse_args().run_id == "job-from-env"


@pytest.mark.parametrize("option", ["--interval", "--refresh-interval"])
@pytest.mark.parametrize("value", ["nan", "inf", "0", "-0.1"])
def test_parse_args_rejects_nonfinite_or_nonpositive_intervals(
    option: str, value: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "monitor",
            "--output",
            str(tmp_path / "out.csv"),
            "--stop-file",
            str(tmp_path / "stop"),
            option,
            value,
        ],
    )

    with pytest.raises(SystemExit):
        monitor.parse_args()


@pytest.mark.parametrize(
    ("interval", "refresh_interval"),
    [(float("nan"), 1.0), (float("inf"), 1.0), (0.1, 0.0)],
)
def test_monitor_rejects_invalid_intervals(
    interval: float, refresh_interval: float, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="must be finite and positive"):
        run_monitor(
            tmp_path / "out.csv",
            tmp_path / "stop",
            interval=interval,
            refresh_interval=refresh_interval,
        )


def test_main_exits_nonzero_for_missing_proc_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "monitor",
            "--output",
            str(tmp_path / "out.csv"),
            "--stop-file",
            str(tmp_path / "stop"),
            "--proc-root",
            str(tmp_path / "missing-proc"),
        ],
    )
    stop_checks = iter([False, True])
    monkeypatch.setattr(monitor, "_stop_requested", lambda _path: next(stop_checks))
    monkeypatch.setattr(monitor.signal, "signal", lambda *_args: None)

    with pytest.raises(SystemExit, match="Cannot enumerate proc root"):
        monitor.main()


class _FailingWriteFile(io.StringIO):
    def write(self, _text: str) -> int:
        raise OSError("disk failure")


@pytest.mark.parametrize("failure_point", ["mkdir", "write"])
def test_main_exits_nonzero_for_output_creation_or_write_errors(
    failure_point: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "nested" / "out.csv"
    stop_file = tmp_path / "stop"
    stop_file.touch()
    monkeypatch.setattr(
        sys,
        "argv",
        ["monitor", "--output", str(output), "--stop-file", str(stop_file)],
    )
    monkeypatch.setattr(monitor.signal, "signal", lambda *_args: None)
    if failure_point == "mkdir":
        monkeypatch.setattr(
            Path, "mkdir", lambda *_args, **_kwargs: _raise_disk_error()
        )
    else:
        monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: _FailingWriteFile())

    with pytest.raises(SystemExit, match="disk failure"):
        monitor.main()


def _raise_disk_error() -> None:
    raise OSError("disk failure")


def test_signal_handler_requests_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(monitor, "_SIGNAL_RECEIVED", False)

    monitor._handle_signal(signal.SIGTERM, None)

    assert monitor._stop_requested(Path("unused")) is True
