from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import rlinf.utils.profile_timeline as profile_timeline
from rlinf.utils.profile_timeline import (
    TraceChunkHandler,
    build_cuda_metadata,
    build_time_anchor,
    torch_profiler_schedule_kwargs,
    write_time_anchor,
)


class _FakeCudaProperties:
    def __init__(self, name: str, uuid: str) -> None:
        self.name = name
        self.uuid = uuid


class _FakeCuda:
    def is_available(self) -> bool:
        return True

    def current_device(self) -> int:
        return 0

    def device_count(self) -> int:
        return 2

    def get_device_properties(self, local_index: int) -> _FakeCudaProperties:
        return _FakeCudaProperties(
            name=f"Fake GPU {local_index}",
            uuid=f"GPU-fake-{local_index}",
        )


class _FakeProfiler:
    def __init__(self, step_num: int) -> None:
        self.step_num = step_num
        self.export_calls = 0

    def export_chrome_trace(self, path: str) -> None:
        self.export_calls += 1
        with open(path, "w", encoding="utf-8") as trace_file:
            trace_file.write(f'{{"step": {self.step_num}}}\n')


class _FailingCuda(_FakeCuda):
    def current_device(self) -> int:
        raise RuntimeError("current device unavailable")

    def get_device_properties(self, local_index: int) -> _FakeCudaProperties:
        if local_index == 0:
            raise RuntimeError("device properties unavailable")
        return _FakeCudaProperties(name=123, uuid=456)  # type: ignore[arg-type]


class _FailingExportProfiler(_FakeProfiler):
    def export_chrome_trace(self, path: str) -> None:
        del path
        self.export_calls += 1
        raise RuntimeError("trace export failed")


class _OSErrorExportProfiler(_FakeProfiler):
    def export_chrome_trace(self, path: str) -> None:
        del path
        self.export_calls += 1
        raise OSError("trace export failed")


class _MissingExportProfiler(_FakeProfiler):
    def export_chrome_trace(self, path: str) -> None:
        super().export_chrome_trace(path)
        Path(path).unlink()


def test_build_time_anchor_has_wall_and_perf_clocks() -> None:
    anchor = build_time_anchor(component="env", rank=3, pid=1234)

    assert anchor["component"] == "env"
    assert anchor["rank"] == 3
    assert anchor["pid"] == 1234
    assert isinstance(anchor["wall_time_ns"], int)
    assert isinstance(anchor["perf_counter_ns"], int)
    assert anchor["clock"] == "time.time_ns/perf_counter_ns"


def test_write_time_anchor_persists_json(tmp_path) -> None:
    anchor = write_time_anchor(str(tmp_path), component="generation", rank=1)

    written = json.loads((tmp_path / "time_anchor.json").read_text())

    assert written == anchor
    assert written["component"] == "generation"
    assert written["rank"] == 1


def test_env_and_generation_anchors_share_clock_schema(tmp_path) -> None:
    env_anchor = write_time_anchor(str(tmp_path / "env_rank0"), component="env", rank=0)
    gen_anchor = write_time_anchor(
        str(tmp_path / "rollout_rank0"), component="generation", rank=0
    )

    comparable_keys = {"wall_time_ns", "perf_counter_ns", "clock"}

    assert comparable_keys <= env_anchor.keys()
    assert comparable_keys <= gen_anchor.keys()
    assert env_anchor["clock"] == gen_anchor["clock"]


def test_torch_profiler_schedule_kwargs_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("RLINF_TORCH_PROFILE_WAIT", "1")
    monkeypatch.setenv("RLINF_TORCH_PROFILE_WARMUP", "2")
    monkeypatch.setenv("RLINF_TORCH_PROFILE_ACTIVE", "3")
    monkeypatch.delenv("RLINF_TORCH_PROFILE_REPEAT", raising=False)

    assert torch_profiler_schedule_kwargs() == {
        "wait": 1,
        "warmup": 2,
        "active": 3,
        "repeat": 1,
    }


def test_torch_profiler_schedule_kwargs_reads_repeat(monkeypatch) -> None:
    monkeypatch.setenv("RLINF_TORCH_PROFILE_REPEAT", "0")

    assert torch_profiler_schedule_kwargs()["repeat"] == 0


def test_build_cuda_metadata_maps_visible_devices(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")

    metadata = build_cuda_metadata(cuda=_FakeCuda())

    assert metadata == {
        "cuda_visible_devices": ["4", "5"],
        "cuda_current_device": 0,
        "cuda_devices": [
            {
                "local_index": 0,
                "visible_id": "4",
                "name": "Fake GPU 0",
                "uuid": "GPU-fake-0",
            },
            {
                "local_index": 1,
                "visible_id": "5",
                "name": "Fake GPU 1",
                "uuid": "GPU-fake-1",
            },
        ],
    }


def test_build_cuda_metadata_tolerates_query_failures(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")

    metadata = build_cuda_metadata(cuda=_FailingCuda())

    assert metadata["cuda_current_device"] is None
    assert metadata["cuda_devices"] == [
        {
            "local_index": 0,
            "visible_id": "4",
            "name": "",
            "uuid": "",
        },
        {
            "local_index": 1,
            "visible_id": "5",
            "name": "123",
            "uuid": "456",
        },
    ]
    json.dumps(metadata)


def test_write_time_anchor_failure_preserves_existing_file(
    monkeypatch, tmp_path
) -> None:
    anchor_path = tmp_path / "time_anchor.json"
    anchor_path.write_text('{"existing": true}\n')
    monkeypatch.setattr(
        profile_timeline,
        "build_cuda_metadata",
        lambda: {"cuda_devices": []},
    )

    def fail_replace(_source, _destination) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(profile_timeline.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        write_time_anchor(str(tmp_path), component="env", rank=0)

    assert anchor_path.read_text() == '{"existing": true}\n'
    assert not list(tmp_path.glob(".*.tmp"))


def test_trace_chunk_handler_exports_incrementing_manifest(tmp_path) -> None:
    handler = TraceChunkHandler(tmp_path, component="generation", rank=1)
    first_profiler = _FakeProfiler(step_num=2)
    second_profiler = _FakeProfiler(step_num=4)

    handler(first_profiler)
    handler(second_profiler)

    trace_files = sorted(tmp_path.glob("*.pt.trace.json"))
    assert [path.name for path in trace_files] == [
        f"{handler.session_id}_chunk_000000_step_2.pt.trace.json",
        f"{handler.session_id}_chunk_000001_step_4.pt.trace.json",
    ]
    manifest = [
        json.loads(line)
        for line in (tmp_path / "trace_manifest.jsonl").read_text().splitlines()
    ]
    assert [entry["chunk_index"] for entry in manifest] == [0, 1]
    assert [entry["step_num"] for entry in manifest] == [2, 4]
    assert [entry["trace_file"] for entry in manifest] == [
        path.name for path in trace_files
    ]
    assert all(entry["component"] == "generation" for entry in manifest)
    assert all(entry["rank"] == 1 for entry in manifest)
    assert all(entry["session_id"] == handler.session_id for entry in manifest)
    assert all(entry["size_bytes"] > 0 for entry in manifest)


def test_trace_chunk_handler_disables_after_export_failure(tmp_path) -> None:
    handler = TraceChunkHandler(tmp_path, component="env", rank=0)
    profiler = _FailingExportProfiler(step_num=2)

    handler(profiler)
    handler(profiler)

    assert profiler.export_calls == 1
    assert not list(tmp_path.glob("*.pt.trace.json"))
    assert not list(tmp_path.glob(".*.tmp"))


def test_trace_chunk_handler_cleanup_failure_is_best_effort(
    monkeypatch, tmp_path
) -> None:
    handler = TraceChunkHandler(tmp_path, component="env", rank=0)
    profiler = _OSErrorExportProfiler(step_num=2)
    monkeypatch.setattr(
        Path,
        "unlink",
        MagicMock(side_effect=PermissionError("cleanup denied")),
    )

    handler(profiler)

    assert handler._disabled is True
    assert profiler.export_calls == 1


def test_trace_chunk_handler_disables_after_stat_failure(tmp_path) -> None:
    handler = TraceChunkHandler(tmp_path, component="env", rank=0)
    profiler = _MissingExportProfiler(step_num=2)

    handler(profiler)
    handler(profiler)

    assert profiler.export_calls == 1
    assert not (tmp_path / "trace_manifest.jsonl").exists()


def test_trace_chunk_handler_preserves_manifest_after_update_failure(
    monkeypatch, tmp_path
) -> None:
    manifest_path = tmp_path / "trace_manifest.jsonl"
    manifest_path.write_text('{"existing": true}\n')
    handler = TraceChunkHandler(tmp_path, component="generation", rank=1)
    profiler = _FakeProfiler(step_num=3)
    original_replace = profile_timeline.os.replace

    def fail_manifest_replace(source, destination) -> None:
        if Path(destination) == manifest_path:
            raise OSError("manifest replace failed")
        original_replace(source, destination)

    monkeypatch.setattr(profile_timeline.os, "replace", fail_manifest_replace)

    handler(profiler)
    handler(profiler)

    assert profiler.export_calls == 1
    assert manifest_path.read_text() == '{"existing": true}\n'
    assert not list(tmp_path.glob(".*.tmp"))
