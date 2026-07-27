from __future__ import annotations

import csv
import gc
import importlib.util
import inspect
import json
import math
import subprocess
import sys
import time
import weakref
from dataclasses import fields
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "tools" / "async_gipo_resource_profile.py"
)
SPEC = importlib.util.spec_from_file_location(
    "async_gipo_resource_profile", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
PROFILE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROFILE
SPEC.loader.exec_module(PROFILE)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _write_trace(
    path: Path,
    *,
    base_ns: int | None,
    events: list[dict[str, object]],
) -> None:
    payload: dict[str, object] = {"traceEvents": events}
    if base_ns is not None:
        payload["baseTimeNanoseconds"] = base_ns
    path.write_text(json.dumps(payload))


def _kernel(
    ts_us: float,
    dur_us: float,
    *,
    device_key: str = "device",
    device: int = 0,
    occupancy: float | None = None,
) -> dict[str, object]:
    args: dict[str, object] = {device_key: device}
    if occupancy is not None:
        args["est. achieved occupancy %"] = occupancy
    return {
        "name": "kernel",
        "ph": "X",
        "cat": "kernel",
        "ts": ts_us,
        "dur": dur_us,
        "args": args,
    }


def _write_required_inputs(run_dir: Path) -> tuple[Path, Path]:
    profile_dir = run_dir / "resource_profile"
    cpu_path = profile_dir / "cpu" / "thread_core_samples.csv"
    cpu_path.parent.mkdir(parents=True)
    with cpu_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "interval_start",
                "interval_end",
                "pid",
                "tid",
                "component",
                "rank",
                "cpu",
                "cpu_time_s",
                "migrated",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "interval_start": 1000.5,
                "interval_end": 1001.5,
                "pid": 10,
                "tid": 11,
                "component": "env",
                "rank": 0,
                "cpu": 1,
                "cpu_time_s": 1.0,
                "migrated": "false",
            }
        )
        writer.writerow(
            {
                "interval_start": 1000.0,
                "interval_end": 1001.0,
                "pid": 12,
                "tid": 13,
                "component": "generation",
                "rank": 0,
                "cpu": 1,
                "cpu_time_s": 1.2,
                "migrated": "true",
            }
        )

    worker_dir = profile_dir / "torch" / "rollout_rank0"
    worker_dir.mkdir(parents=True)
    (worker_dir / "time_anchor.json").write_text(
        json.dumps(
            {
                "component": "generation",
                "rank": 0,
                "wall_time_ns": 1_000_100_000_000,
                "perf_counter_ns": 999,
                "cuda_devices": [
                    {
                        "local_index": 0,
                        "visible_id": "4",
                        "name": "Test GPU",
                        "uuid": "GPU-uuid4",
                    }
                ],
            }
        )
    )
    _write_trace(
        worker_dir / "chunk0.pt.trace.json",
        base_ns=1_000_000_000_000,
        events=[
            _kernel(0, 750_000, occupancy=25),
            _kernel(250_000, 750_000, device_key="Device Id", occupancy=75),
            {"name": "trace end", "ph": "i", "ts": 1_000_000},
        ],
    )
    _write_trace(
        worker_dir / "chunk1.pt.trace.json",
        base_ns=1_001_000_000_000,
        events=[
            _kernel(0, 500_000, device_key="device_id"),
            {"name": "trace end", "ph": "i", "ts": 500_000},
        ],
    )
    _write_trace(
        worker_dir / "missing_base.pt.trace.json",
        base_ns=None,
        events=[_kernel(0, 100_000, occupancy=90)],
    )
    _write_jsonl(
        worker_dir / "trace_manifest.jsonl",
        [
            {"chunk_index": 0, "trace_file": "chunk0.pt.trace.json"},
            {"chunk_index": 1, "trace_file": "chunk1.pt.trace.json"},
            {"chunk_index": 2, "trace_file": "missing_base.pt.trace.json"},
            {"chunk_index": 3, "trace_file": "not_written.pt.trace.json"},
        ],
    )
    return cpu_path, worker_dir


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_cpu_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "interval_start",
                "interval_end",
                "pid",
                "tid",
                "component",
                "rank",
                "cpu",
                "cpu_time_s",
                "migrated",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_interval_union_duration_clips_and_merges_overlaps() -> None:
    intervals = [
        PROFILE.KernelInterval("a", "generation", 0, "u", "4", -1, 0.75, 10),
        PROFILE.KernelInterval("a", "generation", 0, "u", "4", 0.25, 1.25, 20),
        PROFILE.KernelInterval("a", "generation", 0, "u", "4", 1.5, 3, 30),
    ]

    assert PROFILE.interval_union_duration(intervals, 0, 2) == pytest.approx(1.75)


def test_end_to_end_aggregates_cpu_gpu_phases_and_coverage(tmp_path: Path) -> None:
    _write_required_inputs(tmp_path)
    profile_dir = tmp_path / "resource_profile"
    _write_jsonl(
        profile_dir / "runner_events.jsonl",
        [
            {"event": "step.start", "step": 7, "wall_ns": 1_000_000_000_000},
            {
                "event": "actor_training.start",
                "step": 7,
                "wall_ns": 1_000_200_000_000,
            },
            {"event": "step.end", "step": 7, "wall_ns": 1_001_800_000_000},
        ],
    )
    _write_jsonl(
        tmp_path / "rollout_generation_timestamps" / "rank0.jsonl",
        [
            {
                "event": "start",
                "wall_ns": 1_000_100_000_000,
                "rank": 0,
                "epoch": 1,
                "chunk_step": 2,
                "stage": "rollout",
                "phase": "decode",
            },
            {
                "event": "end",
                "wall_ns": 1_000_900_000_000,
                "rank": 0,
                "epoch": 1,
                "chunk_step": 2,
                "stage": "rollout",
                "phase": "decode",
            },
            {
                "event": "end",
                "wall_ns": 1_001_000_000_000,
                "rank": 1,
                "epoch": 1,
            },
        ],
    )
    _write_jsonl(
        tmp_path / "env_sim_timestamps" / "rank0.jsonl",
        [
            {
                "event": "start",
                "wall_ns": 1_000_300_000_000,
                "rank": 0,
                "epoch": 1,
                "chunk_step": 2,
                "stage": "step",
            },
            {
                "event": "end",
                "wall_ns": 1_000_600_000_000,
                "rank": 0,
                "epoch": 1,
                "chunk_step": 2,
                "stage": "step",
            },
        ],
    )

    coverage = PROFILE.generate_resource_profile(tmp_path, bin_s=1.0, num_cpus=3)

    derived = profile_dir / "derived"
    for name in (
        "cpu_core_1s.csv",
        "gpu_worker_1s.csv",
        "gpu_device_1s.csv",
        "phase_windows.csv",
        "coverage.json",
    ):
        assert (derived / name).is_file()
    assert (profile_dir / "metadata.json").is_file()

    worker_rows = _read_csv(derived / "gpu_worker_1s.csv")
    bucket_1000 = next(row for row in worker_rows if float(row["timestamp"]) == 1000)
    bucket_1001 = next(row for row in worker_rows if float(row["timestamp"]) == 1001)
    assert bucket_1000["worker"] == "rollout_rank0"
    assert bucket_1000["gpu_uuid"] == "GPU-uuid4"
    assert bucket_1000["gpu_label"] == "4"
    assert float(bucket_1000["kernel_busy_pct"]) == pytest.approx(100)
    assert float(bucket_1000["est_sm_occupancy_pct"]) == pytest.approx(50)
    assert int(bucket_1000["kernel_count"]) == 2
    assert math.isnan(float(bucket_1001["est_sm_occupancy_pct"]))

    device_rows = _read_csv(derived / "gpu_device_1s.csv")
    assert float(device_rows[0]["kernel_busy_pct"]) == pytest.approx(100)

    cpu_rows = _read_csv(derived / "cpu_core_1s.csv")
    assert len(cpu_rows) == 6
    by_bucket_cpu = {
        (float(row["timestamp"]), int(row["cpu"])): row for row in cpu_rows
    }
    assert float(by_bucket_cpu[(1000.0, 1)]["util_pct"]) == pytest.approx(170)
    assert float(by_bucket_cpu[(1001.0, 1)]["util_pct"]) == pytest.approx(50)
    assert float(by_bucket_cpu[(1000.0, 0)]["util_pct"]) == 0
    assert int(by_bucket_cpu[(1001.0, 2)]["thread_count"]) == 0

    phases = _read_csv(derived / "phase_windows.csv")
    assert {row["phase"] for row in phases} == {"step", "decode", "env_simulation"}
    step_phase = next(row for row in phases if row["phase"] == "step")
    assert step_phase["step"] == "7"
    assert float(step_phase["start"]) == pytest.approx(1000.0)
    assert float(step_phase["end"]) == pytest.approx(1001.5)

    saved_coverage = json.loads((derived / "coverage.json").read_text())
    assert coverage == saved_coverage
    assert coverage["cpu"]["migration_ratio"] == pytest.approx(0.5)
    assert coverage["cpu"]["over_100_bins"] == 1
    assert len(coverage["torch"]["workers"]) == 1
    assert len(coverage["torch"]["missing_trace_files"]) == 1
    assert coverage["torch"]["missing_occupancy_events"] == 1
    assert len(coverage["phases"]["unmatched_starts"]) == 1
    assert len(coverage["phases"]["unmatched_ends"]) == 1
    assert any("baseTimeNanoseconds" in warning for warning in coverage["warnings"])

    metadata = json.loads((profile_dir / "metadata.json").read_text())
    assert metadata["run_dir"] == str(tmp_path.resolve())
    assert metadata["bin_s"] == 1.0
    assert metadata["num_cpus"] == 3
    assert "generated_at" in metadata
    assert metadata["sources"]["cpu"].endswith("thread_core_samples.csv")


def test_base_time_not_wall_perf_offset_controls_kernel_timestamp(
    tmp_path: Path,
) -> None:
    _, worker_dir = _write_required_inputs(tmp_path)
    intervals, _, _ = PROFILE.load_torch_intervals(worker_dir.parent)

    assert min(interval.start_s for interval in intervals) == pytest.approx(1000.0)
    assert max(interval.end_s for interval in intervals) == pytest.approx(1001.5)


def test_gpu_busy_is_capped_and_occupancy_is_overlap_weighted(
    tmp_path: Path,
) -> None:
    _write_required_inputs(tmp_path)

    PROFILE.generate_resource_profile(tmp_path, bin_s=1.0, num_cpus=2)

    derived = tmp_path / "resource_profile" / "derived"
    worker_row = _read_csv(derived / "gpu_worker_1s.csv")[0]
    device_row = _read_csv(derived / "gpu_device_1s.csv")[0]
    assert float(worker_row["kernel_busy_pct"]) == pytest.approx(100)
    assert float(device_row["kernel_busy_pct"]) == pytest.approx(100)
    assert float(worker_row["est_sm_occupancy_pct"]) == pytest.approx(50)
    assert int(worker_row["kernel_count"]) == 2


def test_gpu_worker_bin_without_valid_occupancy_writes_nan(tmp_path: Path) -> None:
    _write_required_inputs(tmp_path)

    PROFILE.generate_resource_profile(tmp_path, bin_s=1.0, num_cpus=2)

    rows = _read_csv(tmp_path / "resource_profile" / "derived" / "gpu_worker_1s.csv")
    row = next(item for item in rows if float(item["timestamp"]) == 1001)
    assert math.isnan(float(row["est_sm_occupancy_pct"]))


def test_known_gpu_devices_are_zero_filled_when_traces_have_no_kernels(
    tmp_path: Path,
) -> None:
    _, worker_dir = _write_required_inputs(tmp_path)
    for trace_name, base_ns in (
        ("chunk0.pt.trace.json", 1_000_000_000_000),
        ("chunk1.pt.trace.json", 1_001_000_000_000),
    ):
        _write_trace(
            worker_dir / trace_name,
            base_ns=base_ns,
            events=[{"ph": "X", "cat": "cpu_op", "ts": 0, "dur": 10}],
        )

    PROFILE.generate_resource_profile(tmp_path, bin_s=1.0, num_cpus=2)

    derived = tmp_path / "resource_profile" / "derived"
    worker_rows = _read_csv(derived / "gpu_worker_1s.csv")
    device_rows = _read_csv(derived / "gpu_device_1s.csv")
    assert len(worker_rows) == 2
    assert len(device_rows) == 2
    assert all(float(row["kernel_busy_pct"]) == 0 for row in worker_rows)
    assert all(int(row["kernel_count"]) == 0 for row in worker_rows)
    assert all(math.isnan(float(row["est_sm_occupancy_pct"])) for row in worker_rows)
    assert all(float(row["kernel_busy_pct"]) == 0 for row in device_rows)


def test_gpu_device_unions_workers_by_physical_uuid(tmp_path: Path) -> None:
    _write_required_inputs(tmp_path)
    torch_dir = tmp_path / "resource_profile" / "torch"
    worker_dir = torch_dir / "rank1"
    worker_dir.mkdir()
    (worker_dir / "time_anchor.json").write_text(
        json.dumps(
            {
                "component": "training",
                "rank": 1,
                "wall_time_ns": 1_000_000_000_000,
                "cuda_devices": [
                    {
                        "local_index": 0,
                        "visible_id": "5",
                        "name": "Same GPU",
                        "uuid": "GPU-uuid4",
                    }
                ],
            }
        )
    )
    _write_trace(
        worker_dir / "chunk.pt.trace.json",
        base_ns=1_000_000_000_000,
        events=[_kernel(0, 1_000_000, occupancy=60)],
    )
    _write_jsonl(
        worker_dir / "trace_manifest.jsonl",
        [{"chunk_index": 0, "trace_file": "chunk.pt.trace.json"}],
    )

    PROFILE.generate_resource_profile(tmp_path, bin_s=1.0, num_cpus=2)

    rows = _read_csv(tmp_path / "resource_profile" / "derived" / "gpu_device_1s.csv")
    bucket_rows = [row for row in rows if float(row["timestamp"]) == 1000]
    assert len(bucket_rows) == 1
    assert float(bucket_rows[0]["kernel_busy_pct"]) == pytest.approx(100)


def test_all_manifest_chunks_are_loaded_and_missing_base_is_skipped(
    tmp_path: Path,
) -> None:
    _, worker_dir = _write_required_inputs(tmp_path)

    intervals, torch_coverage, warnings = PROFILE.load_torch_intervals(
        worker_dir.parent
    )

    assert len(intervals) == 3
    assert {interval.start_s for interval in intervals} == {1000.0, 1000.25, 1001.0}
    assert len(torch_coverage["missing_trace_files"]) == 1
    assert torch_coverage["missing_occupancy_events"] == 1
    expected_bytes = sum(
        (worker_dir / name).stat().st_size
        for name in (
            "chunk0.pt.trace.json",
            "chunk1.pt.trace.json",
            "missing_base.pt.trace.json",
        )
    )
    assert torch_coverage["total_trace_bytes"] == expected_bytes
    assert any("baseTimeNanoseconds" in warning for warning in warnings)


def test_cpu_interval_splits_across_bins_and_zero_fills_all_cores(
    tmp_path: Path,
) -> None:
    _write_required_inputs(tmp_path)

    PROFILE.generate_resource_profile(tmp_path, bin_s=1.0, num_cpus=3)

    rows = _read_csv(tmp_path / "resource_profile" / "derived" / "cpu_core_1s.csv")
    by_bucket_cpu = {(float(row["timestamp"]), int(row["cpu"])): row for row in rows}
    assert len(rows) == 6
    assert float(by_bucket_cpu[(1001.0, 1)]["util_pct"]) == pytest.approx(50)
    assert float(by_bucket_cpu[(1000.0, 0)]["util_pct"]) == 0
    assert int(by_bucket_cpu[(1001.0, 2)]["thread_count"]) == 0


def test_cpu_coverage_reports_migrations_and_over_100_bins(tmp_path: Path) -> None:
    _write_required_inputs(tmp_path)

    coverage = PROFILE.generate_resource_profile(tmp_path, bin_s=1.0, num_cpus=3)

    assert coverage["cpu"]["migration_ratio"] == pytest.approx(0.5)
    assert coverage["cpu"]["over_100_bins"] == 1


def test_phase_pairing_is_fifo_and_counts_unmatched_events(tmp_path: Path) -> None:
    _write_required_inputs(tmp_path)
    runner_path = tmp_path / "resource_profile" / "runner_events.jsonl"
    _write_jsonl(
        runner_path,
        [
            {"event": "step.start", "step": 2, "wall_ns": 1_000_000_000_000},
            {"event": "step.start", "step": 2, "wall_ns": 1_000_100_000_000},
            {"event": "step.end", "step": 2, "wall_ns": 1_000_200_000_000},
            {
                "event": "actor_training.end",
                "step": 2,
                "wall_ns": 1_000_300_000_000,
            },
        ],
    )

    windows, coverage = PROFILE.load_phase_windows(tmp_path)

    assert len(windows) == 1
    assert windows[0].phase == "step"
    assert windows[0].start == pytest.approx(1000.1)
    assert windows[0].end == pytest.approx(1000.2)
    assert len(coverage["unmatched_starts"]) == 1
    assert len(coverage["unmatched_ends"]) == 1


def test_unmapped_local_device_is_counted_and_skipped(tmp_path: Path) -> None:
    _, worker_dir = _write_required_inputs(tmp_path)
    _write_trace(
        worker_dir / "chunk0.pt.trace.json",
        base_ns=1_000_000_000_000,
        events=[_kernel(0, 10, device=9, occupancy=50)],
    )

    intervals, torch_coverage, _ = PROFILE.load_torch_intervals(worker_dir.parent)

    assert len(intervals) == 1
    assert torch_coverage["unmapped_device_events"] == 1


def test_malformed_trace_json_reports_its_path(tmp_path: Path) -> None:
    _, worker_dir = _write_required_inputs(tmp_path)
    trace_path = worker_dir / "chunk0.pt.trace.json"
    trace_path.write_text("{bad trace")

    with pytest.raises(ValueError, match=str(trace_path)):
        PROFILE.load_torch_intervals(worker_dir.parent)


def test_malformed_jsonl_reports_path_and_line(tmp_path: Path) -> None:
    _write_required_inputs(tmp_path)
    event_path = tmp_path / "rollout_generation_timestamps" / "rank0.jsonl"
    event_path.parent.mkdir(parents=True)
    event_path.write_text('{"event": "start"}\nnot-json\n')

    with pytest.raises(ValueError) as error:
        PROFILE.generate_resource_profile(tmp_path, bin_s=1.0, num_cpus=2)

    assert str(event_path) in str(error.value)
    assert "line 2" in str(error.value)


def test_cli_rejects_missing_required_inputs(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(tmp_path)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "thread_core_samples.csv" in result.stderr


def test_metadata_includes_sampling_workers_devices_and_trace_bytes(
    tmp_path: Path,
) -> None:
    _, worker_dir = _write_required_inputs(tmp_path)

    coverage = PROFILE.generate_resource_profile(tmp_path, bin_s=1.0, num_cpus=3)

    metadata = json.loads((tmp_path / "resource_profile" / "metadata.json").read_text())
    assert metadata["cpu_sample_interval_s"] == pytest.approx(1.0)
    assert metadata["workers"][0]["worker"] == "rollout_rank0"
    assert metadata["workers"][0]["chunk_count"] == 2
    assert metadata["gpu_devices"] == [
        {
            "worker": "rollout_rank0",
            "session_id": "__legacy__",
            "component": "generation",
            "rank": 0,
            "local_index": 0,
            "visible_id": "4",
            "name": "Test GPU",
            "gpu_uuid": "GPU-uuid4",
            "gpu_label": "4",
        }
    ]
    expected_bytes = sum(
        (worker_dir / name).stat().st_size
        for name in (
            "chunk0.pt.trace.json",
            "chunk1.pt.trace.json",
            "missing_base.pt.trace.json",
        )
    )
    assert metadata["total_trace_bytes"] == expected_bytes
    assert metadata["total_trace_bytes"] == coverage["torch"]["total_trace_bytes"]
    assert set(metadata["sources"]) == {
        "cpu",
        "torch",
        "runner_events",
        "rollout_generation_timestamps",
        "env_sim_timestamps",
    }


def test_coverage_uses_worker_objects_unmatched_events_and_source_records(
    tmp_path: Path,
) -> None:
    _write_required_inputs(tmp_path)
    profile_dir = tmp_path / "resource_profile"
    _write_jsonl(
        profile_dir / "runner_events.jsonl",
        [
            {"event": "step.start", "step": 1, "wall_ns": 1_000_000_000_000},
            {"event": "step.end", "step": 1, "wall_ns": 1_001_000_000_000},
            {
                "event": "actor_training.start",
                "step": 1,
                "wall_ns": 1_001_100_000_000,
            },
        ],
    )
    _write_jsonl(
        tmp_path / "rollout_generation_timestamps" / "rank0.jsonl",
        [
            {"event": "start", "rank": 0, "epoch": 1, "wall_ns": 1_000_000_000_000},
            {"event": "end", "rank": 0, "epoch": 1, "wall_ns": 1_000_500_000_000},
        ],
    )
    _write_jsonl(
        tmp_path / "env_sim_timestamps" / "rank0.jsonl",
        [{"event": "end", "rank": 0, "epoch": 1, "wall_ns": 1_000_500_000_000}],
    )

    coverage = PROFILE.generate_resource_profile(tmp_path, bin_s=1.0, num_cpus=2)

    worker = coverage["torch"]["workers"][0]
    assert set(worker) >= {
        "worker",
        "component",
        "rank",
        "gpu_mappings",
        "chunk_count",
        "start",
        "end",
        "effective_coverage_ratio",
    }
    assert worker["chunk_count"] == 2
    assert worker["start"] == pytest.approx(1000.0)
    assert worker["end"] == pytest.approx(1001.5)
    assert worker["effective_coverage_ratio"] == pytest.approx(1.0)
    assert coverage["phases"]["unmatched_starts"][0]["event"] == "actor_training.start"
    assert coverage["phases"]["unmatched_ends"][0]["event"] == "end"
    assert coverage["phases"]["unmatched_starts"][0]["component"] == "training"
    assert "key" in coverage["phases"]["unmatched_ends"][0]
    for source_name in ("cpu", "torch", "runner", "generation", "env"):
        source = coverage["sources"][source_name]
        assert source["start"] is not None
        assert source["end"] is not None
        assert "effective_coverage_ratio" in source


def test_empty_kernel_trace_uses_profiler_markers_for_chunk_coverage(
    tmp_path: Path,
) -> None:
    _, worker_dir = _write_required_inputs(tmp_path)
    _write_trace(
        worker_dir / "chunk0.pt.trace.json",
        base_ns=1_000_000_000_000,
        events=[
            {"name": "Iteration Start: PyTorch Profiler", "ph": "i", "ts": 100},
            {"name": "cpu only", "ph": "X", "cat": "cpu_op", "ts": 200, "dur": 10},
            {"name": "Record Window End", "ph": "i", "ts": 900},
        ],
    )

    intervals, coverage, warnings = PROFILE.load_torch_intervals(worker_dir.parent)

    assert not any(interval.start_s < 1001 for interval in intervals)
    worker = coverage["workers"][0]
    assert worker["start"] == pytest.approx(1000.0001)
    assert worker["end"] == pytest.approx(1001.5)
    assert worker["chunk_count"] == 2
    assert not any("no coverage" in warning for warning in warnings)


def test_nested_chunk_coverage_uses_running_max_for_gaps_and_overlaps(
    tmp_path: Path,
) -> None:
    _, worker_dir = _write_required_inputs(tmp_path)
    chunks = {
        "chunk0.pt.trace.json": (1_000_000_000_000, [0, 2_000_000]),
        "chunk1.pt.trace.json": (1_000_000_000_000, [500_000, 1_000_000]),
        "missing_base.pt.trace.json": (1_000_000_000_000, [3_000_000, 4_000_000]),
    }
    for name, (base_ns, timestamps) in chunks.items():
        _write_trace(
            worker_dir / name,
            base_ns=base_ns,
            events=[{"name": "cpu", "ph": "i", "ts": ts} for ts in timestamps],
        )

    _, coverage, _ = PROFILE.load_torch_intervals(worker_dir.parent)

    assert len(coverage["overlaps"]) == 1
    assert coverage["overlaps"][0]["start"] == pytest.approx(1000.5)
    assert len(coverage["observed_gaps"]) == 1
    assert coverage["observed_gaps"][0]["start"] == pytest.approx(1002.0)
    assert coverage["observed_gaps"][0]["end"] == pytest.approx(1003.0)
    assert len(coverage["missing_trace_files"]) == 1


def test_chunk_fallback_coverage_uses_complete_event_end(
    tmp_path: Path,
) -> None:
    _, worker_dir = _write_required_inputs(tmp_path)
    _write_trace(
        worker_dir / "chunk0.pt.trace.json",
        base_ns=1_000_000_000_000,
        events=[
            _kernel(0, 1_000_000, occupancy=50),
            {"name": "later event", "ph": "i", "ts": 250_000},
        ],
    )

    _, coverage, _ = PROFILE.load_torch_intervals(worker_dir.parent)

    first_block = min(coverage["trace_blocks"], key=lambda block: block["start"])
    assert first_block["start"] == pytest.approx(1000.0)
    assert first_block["end"] == pytest.approx(1001.0)


def test_device_mapping_tries_fallback_keys_and_handles_null_uuid(
    tmp_path: Path,
) -> None:
    _, worker_dir = _write_required_inputs(tmp_path)
    anchor = json.loads((worker_dir / "time_anchor.json").read_text())
    anchor["cuda_devices"][0]["uuid"] = None
    anchor["cuda_devices"][0]["visible_id"] = ""
    (worker_dir / "time_anchor.json").write_text(json.dumps(anchor))
    _write_trace(
        worker_dir / "chunk0.pt.trace.json",
        base_ns=1_000_000_000_000,
        events=[
            _kernel(0, 10, device_key="device", occupancy=50)
            | {
                "args": {
                    "device": "bad",
                    "Device Id": 0,
                    "est. achieved occupancy %": 50,
                }
            }
        ],
    )

    intervals, _, _ = PROFILE.load_torch_intervals(worker_dir.parent)

    mapped = next(interval for interval in intervals if interval.start_s == 1000.0)
    assert mapped.gpu_uuid == "0"
    assert mapped.gpu_label == "0"


def test_unmapped_kernel_without_occupancy_increments_both_counts(
    tmp_path: Path,
) -> None:
    _, worker_dir = _write_required_inputs(tmp_path)
    _write_trace(
        worker_dir / "chunk0.pt.trace.json",
        base_ns=1_000_000_000_000,
        events=[_kernel(0, 10, device=99)],
    )

    _, coverage, _ = PROFILE.load_torch_intervals(worker_dir.parent)

    assert coverage["unmapped_device_events"] == 1
    assert coverage["missing_occupancy_events"] == 2


def test_cross_bin_migration_is_counted_once_in_interval_end_bucket(
    tmp_path: Path,
) -> None:
    cpu_path, _ = _write_required_inputs(tmp_path)
    _write_cpu_rows(
        cpu_path,
        [
            {
                "interval_start": 1000.25,
                "interval_end": 1001.75,
                "pid": 1,
                "tid": 2,
                "component": "env",
                "rank": 0,
                "cpu": 0,
                "cpu_time_s": 1.5,
                "migrated": "true",
            },
            {
                "interval_start": 1000.25,
                "interval_end": 1001.0,
                "pid": 3,
                "tid": 4,
                "component": "env",
                "rank": 0,
                "cpu": 1,
                "cpu_time_s": 0.75,
                "migrated": "true",
            },
        ],
    )

    PROFILE.generate_resource_profile(tmp_path, bin_s=1.0, num_cpus=2)

    rows = _read_csv(tmp_path / "resource_profile" / "derived" / "cpu_core_1s.csv")
    migrations = {
        (float(row["timestamp"]), int(row["cpu"])): int(row["migration_count"])
        for row in rows
    }
    assert sum(migrations.values()) == 2
    assert migrations[(1001.0, 0)] == 1
    assert migrations[(1000.0, 1)] == 1


def test_invalid_phase_end_preserves_start_and_reports_both_unmatched(
    tmp_path: Path,
) -> None:
    _write_required_inputs(tmp_path)
    runner_path = tmp_path / "resource_profile" / "runner_events.jsonl"
    _write_jsonl(
        runner_path,
        [
            {"event": "step.start", "step": 4, "wall_ns": 1_001_000_000_000},
            {"event": "step.end", "step": 4, "wall_ns": 1_000_000_000_000},
        ],
    )

    windows, coverage = PROFILE.load_phase_windows(tmp_path)

    assert windows == []
    assert coverage["unmatched_starts"][0]["event"] == "step.start"
    assert coverage["unmatched_starts"][0]["step"] == 4
    assert coverage["unmatched_ends"][0]["event"] == "step.end"
    assert coverage["unmatched_ends"][0]["wall_ns"] == 1_000_000_000_000
    assert coverage["unmatched_ends"][0]["wall"] == pytest.approx(1000.0)
    assert coverage["unmatched_starts"][0]["source"] == str(runner_path)


def _write_session_metadata(
    worker_dir: Path,
    session_id: str,
    *,
    pid: int,
    component: str,
    rank: int,
    visible_id: str,
    gpu_uuid: str,
) -> None:
    (worker_dir / f"session_{session_id}.json").write_text(
        json.dumps(
            {
                "session_id": session_id,
                "component": component,
                "rank": rank,
                "pid": pid,
                "wall_time_ns": 1_000_000_000_000,
                "perf_counter_ns": 10,
                "cuda_devices": [
                    {
                        "local_index": 0,
                        "visible_id": visible_id,
                        "name": f"GPU {visible_id}",
                        "uuid": gpu_uuid,
                    }
                ],
                "torch_profiler_schedule": {
                    "wait": 1,
                    "warmup": 1,
                    "active": 1,
                    "repeat": 1,
                },
            }
        )
    )


def test_cpu_aggregation_visits_only_overlapping_bins(
    tmp_path: Path, monkeypatch
) -> None:
    cpu_path, _ = _write_required_inputs(tmp_path)
    _write_cpu_rows(
        cpu_path,
        [
            {
                "interval_start": 1000.0 + index * 0.5,
                "interval_end": 1000.1 + index * 0.5,
                "pid": 1,
                "tid": index,
                "component": "env",
                "rank": 0,
                "cpu": index % 2,
                "cpu_time_s": 0.05,
                "migrated": "false",
            }
            for index in range(200)
        ],
    )
    overlap_calls = 0
    original_overlap = PROFILE._overlap

    def counting_overlap(*args):
        nonlocal overlap_calls
        overlap_calls += 1
        return original_overlap(*args)

    monkeypatch.setattr(PROFILE, "_overlap", counting_overlap)

    PROFILE.generate_resource_profile(tmp_path, bin_s=1.0, num_cpus=2)

    assert overlap_calls < 1_000


def test_streaming_gpu_smoke_visits_near_kernel_count(
    tmp_path: Path, monkeypatch
) -> None:
    cpu_path, worker_dir = _write_required_inputs(tmp_path)
    _write_cpu_rows(
        cpu_path,
        [
            {
                "interval_start": 1000.0,
                "interval_end": 1100.0,
                "pid": 1,
                "tid": 1,
                "component": "main",
                "rank": 0,
                "cpu": 0,
                "cpu_time_s": 1.0,
                "migrated": "false",
            }
        ],
    )
    kernel_count = 10_000
    events = [
        _kernel((index % 100) * 1_000_000 + (index // 100) * 100, 50, occupancy=50)
        for index in range(kernel_count)
    ]
    events.append({"name": "trace end", "ph": "i", "ts": 100_000_000})
    _write_trace(
        worker_dir / "chunk0.pt.trace.json",
        base_ns=1_000_000_000_000,
        events=events,
    )
    _write_jsonl(
        worker_dir / "trace_manifest.jsonl",
        [{"chunk_index": 0, "trace_file": "chunk0.pt.trace.json"}],
    )
    visits = 0

    def visit() -> None:
        nonlocal visits
        visits += 1

    monkeypatch.setattr(PROFILE, "_visit_gpu_bin", visit)
    started = time.monotonic()

    PROFILE.generate_resource_profile(tmp_path, bin_s=1.0, num_cpus=2)

    assert visits <= kernel_count * 2
    assert time.monotonic() - started < 5.0


def test_two_trace_sessions_use_their_own_gpu_mapping(tmp_path: Path) -> None:
    _, worker_dir = _write_required_inputs(tmp_path)
    _write_session_metadata(
        worker_dir,
        "s1",
        pid=101,
        component="generation",
        rank=0,
        visible_id="4",
        gpu_uuid="GPU-session-1",
    )
    _write_session_metadata(
        worker_dir,
        "s2",
        pid=202,
        component="generation",
        rank=0,
        visible_id="5",
        gpu_uuid="GPU-session-2",
    )
    for session_id, base_ns in (("s1", 1_000_000_000_000), ("s2", 1_001_000_000_000)):
        _write_trace(
            worker_dir / f"{session_id}.pt.trace.json",
            base_ns=base_ns,
            events=[_kernel(0, 100_000, occupancy=50), {"ph": "i", "ts": 100_000}],
        )
    _write_jsonl(
        worker_dir / "trace_manifest.jsonl",
        [
            {
                "session_id": "s1",
                "chunk_index": 0,
                "trace_file": "s1.pt.trace.json",
                "pid": 101,
                "component": "generation",
                "rank": 0,
            },
            {
                "session_id": "s2",
                "chunk_index": 0,
                "trace_file": "s2.pt.trace.json",
                "pid": 202,
                "component": "generation",
                "rank": 0,
            },
        ],
    )

    PROFILE.generate_resource_profile(tmp_path, bin_s=1.0, num_cpus=2)

    rows = _read_csv(tmp_path / "resource_profile" / "derived" / "gpu_worker_1s.csv")
    assert {row["gpu_uuid"] for row in rows} == {"GPU-session-1", "GPU-session-2"}


def test_duplicate_manifest_record_is_warned_and_not_double_counted(
    tmp_path: Path,
) -> None:
    _, worker_dir = _write_required_inputs(tmp_path)
    _write_session_metadata(
        worker_dir,
        "s1",
        pid=101,
        component="generation",
        rank=0,
        visible_id="4",
        gpu_uuid="GPU-session-1",
    )
    trace_path = worker_dir / "s1.pt.trace.json"
    _write_trace(
        trace_path,
        base_ns=1_000_000_000_000,
        events=[_kernel(0, 100_000, occupancy=50), {"ph": "i", "ts": 100_000}],
    )
    record = {
        "session_id": "s1",
        "chunk_index": 0,
        "trace_file": trace_path.name,
        "pid": 101,
        "component": "generation",
        "rank": 0,
    }
    _write_jsonl(worker_dir / "trace_manifest.jsonl", [record, record])

    coverage = PROFILE.generate_resource_profile(tmp_path, bin_s=1.0, num_cpus=2)

    rows = _read_csv(tmp_path / "resource_profile" / "derived" / "gpu_worker_1s.csv")
    assert sum(int(row["kernel_count"]) for row in rows) == 1
    assert coverage["torch"]["total_trace_bytes"] == trace_path.stat().st_size
    assert any(
        "duplicate manifest" in warning.lower() for warning in coverage["warnings"]
    )


def test_conflicting_manifest_chunk_raises_value_error(tmp_path: Path) -> None:
    _, worker_dir = _write_required_inputs(tmp_path)
    _write_session_metadata(
        worker_dir,
        "s1",
        pid=101,
        component="generation",
        rank=0,
        visible_id="4",
        gpu_uuid="GPU-session-1",
    )
    records = [
        {
            "session_id": "s1",
            "chunk_index": 0,
            "trace_file": "a.pt.trace.json",
            "pid": 101,
            "component": "generation",
            "rank": 0,
        },
        {
            "session_id": "s1",
            "chunk_index": 0,
            "trace_file": "b.pt.trace.json",
            "pid": 101,
            "component": "generation",
            "rank": 0,
        },
    ]
    _write_jsonl(worker_dir / "trace_manifest.jsonl", records)

    with pytest.raises(ValueError, match="conflicting manifest"):
        PROFILE.generate_resource_profile(tmp_path, bin_s=1.0, num_cpus=2)


def test_multiple_sessions_without_session_metadata_raise(tmp_path: Path) -> None:
    _, worker_dir = _write_required_inputs(tmp_path)
    _write_jsonl(
        worker_dir / "trace_manifest.jsonl",
        [
            {"session_id": "s1", "chunk_index": 0, "trace_file": "a.json"},
            {"session_id": "s2", "chunk_index": 0, "trace_file": "b.json"},
        ],
    )

    with pytest.raises(ValueError, match="session metadata"):
        PROFILE.generate_resource_profile(tmp_path, bin_s=1.0, num_cpus=2)


def test_manifest_identity_must_match_session_metadata(tmp_path: Path) -> None:
    _, worker_dir = _write_required_inputs(tmp_path)
    _write_session_metadata(
        worker_dir,
        "s1",
        pid=101,
        component="generation",
        rank=0,
        visible_id="4",
        gpu_uuid="GPU-session-1",
    )
    _write_jsonl(
        worker_dir / "trace_manifest.jsonl",
        [
            {
                "session_id": "s1",
                "chunk_index": 0,
                "trace_file": "a.json",
                "pid": 999,
                "component": "generation",
                "rank": 0,
            }
        ],
    )

    with pytest.raises(ValueError, match="pid"):
        PROFILE.generate_resource_profile(tmp_path, bin_s=1.0, num_cpus=2)


def test_phase_pairing_isolated_by_pid_and_bad_start_does_not_block(
    tmp_path: Path,
) -> None:
    _write_required_inputs(tmp_path)
    path = tmp_path / "rollout_generation_timestamps" / "events.jsonl"
    _write_jsonl(
        path,
        [
            {"event": "start", "pid": 1, "rank": 0, "wall_ns": 1_000_000_000_000},
            {"event": "end", "pid": 2, "rank": 0, "wall_ns": 1_000_100_000_000},
            {"event": "start", "pid": 3, "rank": 0, "wall_ns": "bad"},
            {"event": "start", "pid": 3, "rank": 0, "wall_ns": 1_000_200_000_000},
            {"event": "end", "pid": 3, "rank": 0, "wall_ns": 1_000_300_000_000},
        ],
    )

    windows, coverage = PROFILE.load_phase_windows(tmp_path)

    generation = [window for window in windows if window.component == "generation"]
    assert len(generation) == 1
    assert generation[0].start == pytest.approx(1000.2)
    assert len(coverage["unmatched_starts"]) == 2
    assert len(coverage["unmatched_ends"]) == 1


def test_chunk_fallback_uses_duration_for_complete_events(tmp_path: Path) -> None:
    _, worker_dir = _write_required_inputs(tmp_path)
    _write_trace(
        worker_dir / "chunk0.pt.trace.json",
        base_ns=1_000_000_000_000,
        events=[{"name": "cpu", "ph": "X", "ts": 0, "dur": 1_000_000}],
    )

    _, coverage, _ = PROFILE.load_torch_intervals(worker_dir.parent)

    first = min(coverage["trace_blocks"], key=lambda block: block["start"])
    assert first["start"] == pytest.approx(1000.0)
    assert first["end"] == pytest.approx(1001.0)


@pytest.mark.parametrize("bin_s", [math.nan, math.inf, -math.inf])
def test_bin_size_must_be_finite(tmp_path: Path, bin_s: float) -> None:
    _write_required_inputs(tmp_path)

    with pytest.raises(ValueError, match="bin_s must be finite and positive"):
        PROFILE.generate_resource_profile(tmp_path, bin_s=bin_s, num_cpus=2)


def test_num_cpus_must_be_a_positive_integer(tmp_path: Path) -> None:
    _write_required_inputs(tmp_path)

    with pytest.raises(ValueError, match="num_cpus must be a positive integer"):
        PROFILE.generate_resource_profile(tmp_path, bin_s=1.0, num_cpus=2.5)


@pytest.mark.parametrize("cpu_time", [-0.1, math.inf, math.nan])
def test_invalid_cpu_time_raises_with_path_and_line(
    tmp_path: Path, cpu_time: float
) -> None:
    cpu_path, _ = _write_required_inputs(tmp_path)
    _write_cpu_rows(
        cpu_path,
        [
            {
                "interval_start": 1000.0,
                "interval_end": 1001.0,
                "pid": 1,
                "tid": 1,
                "component": "env",
                "rank": 0,
                "cpu": 0,
                "cpu_time_s": cpu_time,
                "migrated": "false",
            }
        ],
    )

    with pytest.raises(ValueError) as error:
        PROFILE.generate_resource_profile(tmp_path, bin_s=1.0, num_cpus=2)

    assert str(cpu_path) in str(error.value)
    assert "line 2" in str(error.value)


def test_primary_cpu_window_clips_stale_trace_session(tmp_path: Path) -> None:
    cpu_path, worker_dir = _write_required_inputs(tmp_path)
    _write_cpu_rows(
        cpu_path,
        [
            {
                "interval_start": 1000.0,
                "interval_end": 1002.0,
                "pid": 1,
                "tid": 1,
                "component": "main",
                "rank": 0,
                "cpu": 0,
                "cpu_time_s": 1.0,
                "migrated": "false",
            }
        ],
    )
    _write_trace(
        worker_dir / "chunk0.pt.trace.json",
        base_ns=500_000_000_000,
        events=[_kernel(0, 100_000, occupancy=50), {"ph": "i", "ts": 100_000}],
    )
    _write_jsonl(
        worker_dir / "trace_manifest.jsonl",
        [{"chunk_index": 0, "trace_file": "chunk0.pt.trace.json"}],
    )

    coverage = PROFILE.generate_resource_profile(tmp_path, bin_s=1.0, num_cpus=2)

    rows = _read_csv(tmp_path / "resource_profile" / "derived" / "gpu_worker_1s.csv")
    assert {float(row["timestamp"]) for row in rows} == {1000.0, 1001.0}
    assert all(float(row["kernel_busy_pct"]) == 0 for row in rows)
    assert any(
        "outside primary resource window" in item for item in coverage["warnings"]
    )


def test_generation_transaction_failure_retains_old_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    _write_required_inputs(tmp_path)
    profile_dir = tmp_path / "resource_profile"
    old_derived = profile_dir / "derived"
    old_derived.mkdir()
    (old_derived / "sentinel.txt").write_text("old")
    old_metadata = profile_dir / "metadata.json"
    old_metadata.write_text('{"old": true}\n')
    original_replace = PROFILE.os.replace

    def fail_metadata_replace(source, destination):
        if Path(destination) == old_metadata:
            raise OSError("metadata publish failed")
        return original_replace(source, destination)

    monkeypatch.setattr(PROFILE.os, "replace", fail_metadata_replace)

    with pytest.raises(OSError, match="metadata publish failed"):
        PROFILE.generate_resource_profile(tmp_path, bin_s=1.0, num_cpus=2)

    assert (old_derived / "sentinel.txt").read_text() == "old"
    assert old_metadata.read_text() == '{"old": true}\n'
    assert not list(profile_dir.glob(".resource_profile_generation_*"))
    assert not list(profile_dir.glob(".resource_profile_backup_*"))


def test_trace_chunk_handler_writes_session_metadata(tmp_path: Path) -> None:
    from rlinf.utils.profile_timeline import TraceChunkHandler

    handler = TraceChunkHandler(tmp_path, component="generation", rank=3)

    metadata = json.loads((tmp_path / f"session_{handler.session_id}.json").read_text())
    assert metadata["session_id"] == handler.session_id
    assert metadata["component"] == "generation"
    assert metadata["rank"] == 3
    assert metadata["pid"] > 0
    assert "cuda_devices" in metadata
    assert "torch_profiler_schedule" in metadata


def test_runner_phase_events_do_not_pair_across_pids(tmp_path: Path) -> None:
    _write_required_inputs(tmp_path)
    _write_jsonl(
        tmp_path / "resource_profile" / "runner_events.jsonl",
        [
            {
                "event": "step.start",
                "step": 1,
                "pid": 10,
                "wall_ns": 1_000_000_000_000,
            },
            {
                "event": "step.end",
                "step": 1,
                "pid": 20,
                "wall_ns": 1_001_000_000_000,
            },
        ],
    )

    windows, coverage = PROFILE.load_phase_windows(tmp_path)

    assert not [window for window in windows if window.phase == "step"]
    assert {event["pid"] for event in coverage["unmatched_starts"]} == {10}
    assert {event["pid"] for event in coverage["unmatched_ends"]} == {20}


def test_legacy_runner_restart_does_not_pair_with_stale_start(tmp_path: Path) -> None:
    _write_required_inputs(tmp_path)
    _write_jsonl(
        tmp_path / "resource_profile" / "runner_events.jsonl",
        [
            {"event": "step.start", "step": 1, "wall_ns": 1_000_000_000_000},
            {"event": "step.start", "step": 1, "wall_ns": 2_000_000_000_000},
            {"event": "step.end", "step": 1, "wall_ns": 2_001_000_000_000},
        ],
    )

    windows, coverage = PROFILE.load_phase_windows(tmp_path)

    steps = [window for window in windows if window.phase == "step"]
    assert len(steps) == 1
    assert steps[0].start == pytest.approx(2000.0)
    assert len(coverage["unmatched_starts"]) == 1
    assert coverage["unmatched_starts"][0]["wall"] == pytest.approx(1000.0)


def test_cpu_window_is_authoritative_over_stale_runner_events(tmp_path: Path) -> None:
    cpu_path, _ = _write_required_inputs(tmp_path)
    _write_cpu_rows(
        cpu_path,
        [
            {
                "interval_start": 1000.0,
                "interval_end": 1002.0,
                "pid": 1,
                "tid": 1,
                "component": "main",
                "rank": 0,
                "cpu": 0,
                "cpu_time_s": 1.0,
                "migrated": "false",
            }
        ],
    )
    _write_jsonl(
        tmp_path / "resource_profile" / "runner_events.jsonl",
        [
            {"event": "step.start", "step": 1, "pid": 9, "wall_ns": 500_000_000_000},
            {"event": "step.end", "step": 1, "pid": 9, "wall_ns": 600_000_000_000},
        ],
    )

    coverage = PROFILE.generate_resource_profile(tmp_path, bin_s=1.0, num_cpus=2)

    rows = _read_csv(tmp_path / "resource_profile" / "derived" / "cpu_core_1s.csv")
    assert {float(row["timestamp"]) for row in rows} == {1000.0, 1001.0}
    assert any(
        "outside primary resource window" in item for item in coverage["warnings"]
    )


def test_empty_cpu_uses_latest_complete_runner_pid_session(tmp_path: Path) -> None:
    cpu_path, _ = _write_required_inputs(tmp_path)
    _write_cpu_rows(cpu_path, [])
    _write_jsonl(
        tmp_path / "resource_profile" / "runner_events.jsonl",
        [
            {"event": "step.start", "step": 1, "pid": 1, "wall_ns": 100_000_000_000},
            {"event": "step.end", "step": 1, "pid": 1, "wall_ns": 101_000_000_000},
            {"event": "step.start", "step": 1, "pid": 2, "wall_ns": 200_000_000_000},
            {"event": "step.end", "step": 1, "pid": 2, "wall_ns": 201_000_000_000},
            {"event": "step.start", "step": 2, "pid": 2, "wall_ns": 202_000_000_000},
            {"event": "step.end", "step": 2, "pid": 2, "wall_ns": 203_000_000_000},
        ],
    )

    PROFILE.generate_resource_profile(tmp_path, bin_s=1.0, num_cpus=2)

    rows = _read_csv(tmp_path / "resource_profile" / "derived" / "cpu_core_1s.csv")
    assert {float(row["timestamp"]) for row in rows} == {200.0, 201.0, 202.0}


def test_p2_median_has_constant_state_and_bounded_error(tmp_path: Path) -> None:
    estimator = PROFILE.P2Median()
    for value in range(1, 100_001):
        estimator.update(float(value))

    assert estimator.state_size <= 20
    assert estimator.result() == pytest.approx(50_000.5, rel=0.01)

    cpu_path, _ = _write_required_inputs(tmp_path)
    aggregate = PROFILE._aggregate_cpu_csv(cpu_path, bin_s=1.0, num_cpus=2)
    assert "durations" not in aggregate
    assert aggregate["median_estimator"].state_size <= 20


def test_metadata_marks_p2_cpu_interval_method(tmp_path: Path) -> None:
    _write_required_inputs(tmp_path)

    PROFILE.generate_resource_profile(tmp_path, bin_s=1.0, num_cpus=2)

    metadata = json.loads((tmp_path / "resource_profile" / "metadata.json").read_text())
    assert metadata["cpu_sample_interval_method"] == "p2_median"


def test_trace_iterator_yields_descriptor_without_trace_payload() -> None:
    assert "trace" not in {field.name for field in fields(PROFILE.TraceDescriptor)}
    source = inspect.getsource(PROFILE.process_run)
    assert "load_torch_intervals" not in source
    assert "cpu_samples" not in source


def test_streaming_releases_trace_before_loading_next(
    tmp_path: Path, monkeypatch
) -> None:
    _, worker_dir = _write_required_inputs(tmp_path)
    marker_refs: list[weakref.ReferenceType] = []

    class Marker:
        pass

    def tracking_load(path: Path):
        if marker_refs:
            gc.collect()
            assert marker_refs[-1]() is None
        payload = json.loads(path.read_text())
        marker = Marker()
        payload["_lifetime_marker"] = marker
        marker_refs.append(weakref.ref(marker))
        return payload

    monkeypatch.setattr(PROFILE, "_load_trace", tracking_load)

    PROFILE.generate_resource_profile(tmp_path, bin_s=1.0, num_cpus=2)

    assert len(marker_refs) == 3


def test_reverse_manifest_trace_order_is_reported(tmp_path: Path) -> None:
    _, worker_dir = _write_required_inputs(tmp_path)
    _write_trace(
        worker_dir / "late.json",
        base_ns=1_001_000_000_000,
        events=[{"ph": "i", "ts": 0}, {"ph": "i", "ts": 100_000}],
    )
    _write_trace(
        worker_dir / "early.json",
        base_ns=1_000_000_000_000,
        events=[{"ph": "i", "ts": 0}, {"ph": "i", "ts": 100_000}],
    )
    _write_jsonl(
        worker_dir / "trace_manifest.jsonl",
        [
            {"chunk_index": 0, "trace_file": "late.json"},
            {"chunk_index": 1, "trace_file": "early.json"},
        ],
    )

    _, coverage, _ = PROFILE.load_torch_intervals(worker_dir.parent)

    assert len(coverage["out_of_order_chunks"]) == 1
    record = coverage["out_of_order_chunks"][0]
    assert record["previous_chunk_index"] == 0
    assert record["chunk_index"] == 1
