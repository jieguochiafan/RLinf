from __future__ import annotations

from tools.worker_cpu_core_util import (
    ProcSample,
    WorkerProcess,
    compute_worker_core_utilization,
    parse_proc_stat_core_times,
    parse_proc_task_jiffies,
)


def test_parse_proc_stat_core_totals_reads_per_core_jiffies() -> None:
    text = """cpu  1 2 3 4 5 6 7 8 9 10
cpu0  10 20 30 40 50 60 70 80 90 100
cpu1  1 2 3 4 5 6 7 8 9 10
"""

    assert parse_proc_stat_core_times(text) == {0: (460, 550), 1: (46, 55)}


def test_parse_proc_task_jiffies_handles_comm_with_spaces() -> None:
    stat = "42 (ray worker name) S 1 2 3 4 5 6 7 8 9 10 11 12 13 14"

    assert parse_proc_task_jiffies(stat) == 23


def test_compute_worker_core_utilization_distributes_active_time_by_core_delta() -> None:
    worker = WorkerProcess(pid=42, label="env", cores=(2, 3))
    before = ProcSample(
        timestamp=1.0,
        proc_jiffies=100,
        core_times={2: (100, 1000), 3: (200, 2000)},
    )
    after = ProcSample(
        timestamp=2.0,
        proc_jiffies=150,
        core_times={2: (120, 1100), 3: (280, 2100)},
    )

    util = compute_worker_core_utilization(before, after, worker)

    assert util[2] == 20.0
    assert util[3] == 80.0
