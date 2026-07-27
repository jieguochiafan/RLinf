from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

import rlinf.workers.actor.async_ppo_fsdp_worker as actor_profiler_module
from rlinf.workers.actor.async_ppo_fsdp_worker import (
    create_actor_torch_profiler,
)
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class _FailingStopProfiler:
    def stop(self) -> None:
        raise RuntimeError("profiler stop failed")

    def key_averages(self):
        raise RuntimeError("summary failed")


class _FailingWorkerStopProfiler:
    def stop(self) -> None:
        raise RuntimeError("profiler stop failed")


class _FailingStepProfiler:
    def __init__(self) -> None:
        self.step_calls = 0
        self.stop_calls = 0

    def step(self) -> None:
        self.step_calls += 1
        raise RuntimeError("profiler step failed")

    def stop(self) -> None:
        self.stop_calls += 1


class _FailingStartProfiler:
    def __init__(self, *, stop_raises: bool = False) -> None:
        self.start_calls = 0
        self.stop_calls = 0
        self.stop_raises = stop_raises

    def start(self) -> None:
        self.start_calls += 1
        raise RuntimeError("profiler start failed")

    def stop(self) -> None:
        self.stop_calls += 1
        if self.stop_raises:
            raise RuntimeError("partial profiler stop failed")


def test_create_actor_torch_profiler_disabled_without_env(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("RLINF_TORCH_PROFILE", raising=False)
    monkeypatch.setenv("RLINF_TORCH_PROFILE_DIR", str(tmp_path))

    profiler, section, output_dir = create_actor_torch_profiler(rank=2)

    assert profiler is None
    assert output_dir is None
    with section("fwd"):
        pass
    assert not (tmp_path / "rank2").exists()


def test_create_actor_torch_profiler_writes_rank_anchor(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RLINF_TORCH_PROFILE", "1")
    monkeypatch.setenv("RLINF_TORCH_PROFILE_DIR", str(tmp_path))

    profiler, _, output_dir = create_actor_torch_profiler(rank=2, start=False)

    assert profiler is not None
    assert output_dir == tmp_path / "rank2"
    anchor = json.loads((tmp_path / "rank2" / "time_anchor.json").read_text())
    assert anchor["component"] == "training"
    assert anchor["rank"] == 2


def test_create_actor_torch_profiler_uses_env_schedule(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("RLINF_TORCH_PROFILE", "1")
    monkeypatch.setenv("RLINF_TORCH_PROFILE_DIR", str(tmp_path))
    monkeypatch.setenv("RLINF_TORCH_PROFILE_WAIT", "1")
    monkeypatch.setenv("RLINF_TORCH_PROFILE_WARMUP", "1")
    monkeypatch.setenv("RLINF_TORCH_PROFILE_ACTIVE", "2")

    profiler, _, _ = create_actor_torch_profiler(rank=2, start=False)

    assert profiler is not None
    assert profiler.schedule(0).name == "NONE"
    assert profiler.schedule(1).name == "WARMUP"
    assert profiler.schedule(2).name == "RECORD"
    assert profiler.schedule(3).name == "RECORD_AND_SAVE"
    assert profiler.schedule(4).name == "NONE"


def test_create_actor_torch_profiler_repeats_until_stopped(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("RLINF_TORCH_PROFILE", "1")
    monkeypatch.setenv("RLINF_TORCH_PROFILE_DIR", str(tmp_path))
    monkeypatch.setenv("RLINF_TORCH_PROFILE_WAIT", "0")
    monkeypatch.setenv("RLINF_TORCH_PROFILE_WARMUP", "0")
    monkeypatch.setenv("RLINF_TORCH_PROFILE_ACTIVE", "2")
    monkeypatch.setenv("RLINF_TORCH_PROFILE_REPEAT", "0")

    profiler, _, _ = create_actor_torch_profiler(rank=2, start=False)

    assert profiler is not None
    assert profiler.schedule(0).name == "RECORD"
    assert profiler.schedule(1).name == "RECORD_AND_SAVE"
    assert profiler.schedule(2).name == "RECORD"
    assert profiler.schedule(3).name == "RECORD_AND_SAVE"


def test_create_actor_torch_profiler_disables_after_initialization_failure(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("RLINF_TORCH_PROFILE", "1")
    monkeypatch.setenv("RLINF_TORCH_PROFILE_DIR", str(tmp_path))
    monkeypatch.setattr(
        actor_profiler_module,
        "write_time_anchor",
        MagicMock(side_effect=OSError("anchor failed")),
    )

    profiler, section, output_dir = create_actor_torch_profiler(rank=2)

    assert profiler is None
    assert output_dir is None
    with section("fwd"):
        pass


def test_actor_profiler_cleanup_failure_does_not_escape(tmp_path) -> None:
    profiler = _FailingStopProfiler()

    actor_profiler_module.stop_actor_torch_profiler(profiler, tmp_path)


def test_actor_profiler_step_failure_disables_and_stops_profiler() -> None:
    profiler = _FailingStepProfiler()

    active_profiler = actor_profiler_module.step_actor_torch_profiler(profiler)

    assert active_profiler is None
    assert profiler.step_calls == 1
    assert profiler.stop_calls == 1


def test_actor_profiler_start_failure_stops_partial_profiler(
    monkeypatch, tmp_path
) -> None:
    partial_profiler = _FailingStartProfiler(stop_raises=True)
    monkeypatch.setenv("RLINF_TORCH_PROFILE", "1")
    monkeypatch.setenv("RLINF_TORCH_PROFILE_DIR", str(tmp_path))
    monkeypatch.setattr(actor_profiler_module, "write_time_anchor", MagicMock())
    monkeypatch.setattr(
        "torch.profiler.profile",
        MagicMock(return_value=partial_profiler),
    )

    profiler, _, output_dir = create_actor_torch_profiler(rank=2)

    assert profiler is None
    assert output_dir is None
    assert partial_profiler.start_calls == 1
    assert partial_profiler.stop_calls == 1


@pytest.mark.parametrize(
    ("worker_cls", "anchor_target"),
    [
        (EnvWorker, "rlinf.workers.env.env_worker.write_time_anchor"),
        (
            MultiStepRolloutWorker,
            "rlinf.workers.rollout.hf.huggingface_worker.write_time_anchor",
        ),
    ],
)
def test_worker_torch_profiler_initialization_failure_disables_profiler(
    monkeypatch, tmp_path, worker_cls, anchor_target
) -> None:
    monkeypatch.setenv("RLINF_TORCH_PROFILE", "1")
    monkeypatch.setenv("RLINF_TORCH_PROFILE_DIR", str(tmp_path))
    monkeypatch.setattr(anchor_target, MagicMock(side_effect=OSError("anchor failed")))
    worker = object.__new__(worker_cls)
    worker._rank = 2
    worker._torch_profiler = None
    worker._torch_profiler_dir = None
    worker._torch_profiler_step_enabled = False
    worker.log_warning = MagicMock()

    worker._start_torch_profiler()

    assert worker._torch_profiler is None
    assert worker._torch_profiler_dir is None
    assert worker._torch_profiler_step_enabled is False
    worker.log_warning.assert_called_once()


@pytest.mark.parametrize("worker_cls", [EnvWorker, MultiStepRolloutWorker])
def test_worker_torch_profiler_stop_failure_does_not_escape(
    worker_cls, tmp_path
) -> None:
    worker = object.__new__(worker_cls)
    worker._torch_profiler = _FailingWorkerStopProfiler()
    worker._torch_profiler_dir = str(tmp_path)
    worker._torch_profiler_step_enabled = True
    worker.log_warning = MagicMock()

    worker._stop_torch_profiler()

    assert worker._torch_profiler is None
    assert worker._torch_profiler_dir is None
    assert worker._torch_profiler_step_enabled is False
    worker.log_warning.assert_called_once()


@pytest.mark.parametrize("worker_cls", [EnvWorker, MultiStepRolloutWorker])
def test_worker_torch_profiler_step_failure_disables_and_stops_profiler(
    worker_cls,
) -> None:
    profiler = _FailingStepProfiler()
    worker = object.__new__(worker_cls)
    worker._torch_profiler = profiler
    worker._torch_profiler_dir = "/tmp/unused"
    worker._torch_profiler_step_enabled = True
    worker.log_warning = MagicMock()

    worker._step_torch_profiler()

    assert profiler.step_calls == 1
    assert profiler.stop_calls == 1
    assert worker._torch_profiler is None
    assert worker._torch_profiler_step_enabled is False
    worker.log_warning.assert_called_once()


@pytest.mark.parametrize(
    ("worker_cls", "anchor_target"),
    [
        (EnvWorker, "rlinf.workers.env.env_worker.write_time_anchor"),
        (
            MultiStepRolloutWorker,
            "rlinf.workers.rollout.hf.huggingface_worker.write_time_anchor",
        ),
    ],
)
def test_worker_torch_profiler_start_failure_stops_partial_profiler(
    monkeypatch, tmp_path, worker_cls, anchor_target
) -> None:
    partial_profiler = _FailingStartProfiler(stop_raises=True)
    monkeypatch.setenv("RLINF_TORCH_PROFILE", "1")
    monkeypatch.setenv("RLINF_TORCH_PROFILE_DIR", str(tmp_path))
    monkeypatch.setattr(anchor_target, MagicMock())
    monkeypatch.setattr(
        "torch.profiler.profile",
        MagicMock(return_value=partial_profiler),
    )
    worker = object.__new__(worker_cls)
    worker._rank = 2
    worker._torch_profiler = None
    worker._torch_profiler_dir = None
    worker._torch_profiler_step_enabled = False
    worker.log_warning = MagicMock()

    worker._start_torch_profiler()

    assert partial_profiler.start_calls == 1
    assert partial_profiler.stop_calls == 1
    assert worker._torch_profiler is None
    assert worker._torch_profiler_dir is None
    assert worker._torch_profiler_step_enabled is False
