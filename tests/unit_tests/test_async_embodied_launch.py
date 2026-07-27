from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rlinf.runners.async_ppo_embodied_runner import AsyncPPOEmbodiedRunner
from rlinf.workers.rollout.hf.async_huggingface_worker import (
    AsyncMultiStepRolloutWorker,
)


def test_train_async_launches_long_running_workers_with_concurrency() -> None:
    source = Path("examples/embodiment/train_async.py").read_text()

    rollout_launch = source.split(
        "rollout_group = AsyncMultiStepRolloutWorker.create_group(cfg).launch(",
        maxsplit=1,
    )[1].split(")", maxsplit=1)[0]
    env_launch = source.split(
        "env_group = AsyncEnvWorker.create_group(cfg).launch(",
        maxsplit=1,
    )[1].split(")", maxsplit=1)[0]

    assert "max_concurrency=" in rollout_launch
    assert "max_concurrency=" in env_launch


class _DoneHandle:
    def __init__(self, result=None, error=None):
        self._result = result if result is not None else [None]
        self._error = error

    def wait(self):
        if self._error is not None:
            raise self._error
        return self._result

    def consume_durations(self, return_per_rank=False):
        result = ({}, [{}])
        return result if return_per_rank else {}


class _FakeActor:
    worker_group_name = "ActorGroup"

    def __init__(self, training_error=None):
        self.training_calls = 0
        self.training_error = training_error

    def set_global_step(self, _step):
        return _DoneHandle()

    def sync_model_to_rollout(self):
        return _DoneHandle()

    def recv_rollout_trajectories(self, input_channel):
        return _DoneHandle()

    def compute_proximal_logprobs(self):
        return _DoneHandle()

    def compute_advantages_and_returns(self):
        return _DoneHandle([{"reward": 1.0}])

    def run_training(self):
        self.training_calls += 1
        return _DoneHandle([{"loss": 0.1}], error=self.training_error)


class _FakeRollout:
    worker_group_name = "RolloutGroup"

    def __init__(self):
        self.generate_calls = 0

    def set_global_step(self, _step):
        return _DoneHandle()

    def sync_model_from_actor(self):
        return _DoneHandle()

    def generate(self, **_kwargs):
        self.generate_calls += 1
        return _DoneHandle()

    def stop(self):
        return _DoneHandle()


class _FakeEnv:
    worker_group_name = "EnvGroup"

    def __init__(self):
        self.interact_calls = 0

    def set_global_step(self, _step):
        return _DoneHandle()

    def interact(self, **_kwargs):
        self.interact_calls += 1
        return _DoneHandle()

    def stop(self):
        return _DoneHandle()


class _NoopMetricLogger:
    def log(self, *args, **kwargs):
        pass

    def finish(self):
        pass


class _NoopTimer:
    def __call__(self, _name):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def consume_durations(self):
        return {}


def _make_fake_runner(*, max_steps=1, training_error=None):
    runner = object.__new__(AsyncPPOEmbodiedRunner)
    runner.global_step = 0
    runner.max_steps = max_steps
    runner.cfg = SimpleNamespace(
        runner=SimpleNamespace(
            val_check_interval=-1,
            save_interval=-1,
            logger=SimpleNamespace(log_path="."),
        )
    )
    runner.actor = _FakeActor(training_error=training_error)
    runner.rollout = _FakeRollout()
    runner.env = _FakeEnv()
    runner.reward = None
    runner.reward_channel = None
    runner.env_channel = object()
    runner.rollout_channel = object()
    runner.actor_channel = object()
    runner.env_metric_channel = object()
    runner.rollout_metric_channel = object()
    runner.recompute_logprobs = True
    runner.metric_logger = _NoopMetricLogger()
    runner.stop_logging = False
    runner.log_queue = SimpleNamespace(join=lambda: None)
    runner.log_thread = SimpleNamespace(join=lambda timeout=None: None)
    runner.timer = _NoopTimer()
    runner.update_rollout_weights = lambda: None
    runner.get_env_metrics = lambda: ({}, [], [])
    runner.get_rollout_metrics = lambda: ({}, [])
    runner._aggregate_numeric_metrics = lambda metrics: metrics[0] if metrics else {}
    runner._log_ranked_metrics = lambda *args, **kwargs: None
    runner.print_metrics_table_async = lambda *args, **kwargs: None
    runner._save_checkpoint = lambda: None
    return runner


def test_resource_profile_writer_disabled_does_not_create_file(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("RLINF_RESOURCE_PROFILE", raising=False)
    runner = _make_fake_runner()
    runner.cfg.runner.logger.log_path = str(tmp_path)

    runner._write_resource_profile_event("step.start", step=1)

    assert not (tmp_path / "resource_profile").exists()


def test_resource_profile_writer_writes_json_lines(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RLINF_RESOURCE_PROFILE", "1")
    monkeypatch.setattr("rlinf.runners.async_ppo_embodied_runner.os.getpid", lambda: 77)
    wall_times = iter([101, 202])
    monkeypatch.setattr(
        "rlinf.runners.async_ppo_embodied_runner.time.time_ns",
        lambda: next(wall_times),
    )
    runner = _make_fake_runner()
    runner.cfg.runner.logger.log_path = str(tmp_path)

    runner._write_resource_profile_event("step.start", step=3)
    runner._write_resource_profile_event("step.end", step=3)
    runner._close_resource_profile_events()

    path = tmp_path / "resource_profile" / "runner_events.jsonl"
    lines = path.read_text().splitlines()
    assert [json.loads(line) for line in lines] == [
        {"event": "step.start", "pid": 77, "step": 3, "wall_ns": 101},
        {"event": "step.end", "pid": 77, "step": 3, "wall_ns": 202},
    ]
    assert lines[0] == '{"event": "step.start", "pid": 77, "step": 3, "wall_ns": 101}'


def test_resource_profile_write_failure_disables_future_events(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("RLINF_RESOURCE_PROFILE", "1")

    class _FailingHandle:
        def __init__(self):
            self.write_calls = 0
            self.close_calls = 0

        def write(self, _line):
            self.write_calls += 1
            raise OSError("disk full")

        def flush(self):
            raise AssertionError("flush must not follow a failed write")

        def close(self):
            self.close_calls += 1

    handle = _FailingHandle()
    open_calls = 0

    def fake_open(_path, *args, **kwargs):
        nonlocal open_calls
        open_calls += 1
        return handle

    monkeypatch.setattr(Path, "open", fake_open)
    runner = _make_fake_runner()
    runner.cfg.runner.logger.log_path = str(tmp_path)
    runner.logger = SimpleNamespace(warning=lambda *args, **kwargs: None)

    runner._write_resource_profile_event("step.start", step=1)
    runner._write_resource_profile_event("step.end", step=1)

    assert open_calls == 1
    assert handle.write_calls == 1
    assert handle.close_calls == 1


def test_async_ppo_runner_restarts_rollout_each_training_step() -> None:
    runner = _make_fake_runner(max_steps=3)

    runner.run()

    assert runner.rollout.generate_calls == runner.max_steps
    assert runner.actor.training_calls == runner.max_steps
    assert runner.env.interact_calls == 1


def test_async_ppo_runner_records_resource_profile_event_order() -> None:
    runner = _make_fake_runner()
    events = []
    runner._write_resource_profile_event = lambda event, *, step: events.append(
        (event, step)
    )
    runner._close_resource_profile_events = lambda: None

    runner.run()

    assert events == [
        ("step.start", 1),
        ("actor_training.start", 1),
        ("actor_training.end", 1),
        ("step.end", 1),
    ]


def test_async_ppo_runner_records_end_events_when_actor_wait_fails() -> None:
    error = RuntimeError("actor training failed")
    runner = _make_fake_runner(training_error=error)
    events = []
    runner._write_resource_profile_event = lambda event, *, step: events.append(
        (event, step)
    )
    runner._close_resource_profile_events = lambda: None

    with pytest.raises(RuntimeError, match="actor training failed") as exc_info:
        runner.run()

    assert exc_info.value is error
    assert events == [
        ("step.start", 1),
        ("actor_training.start", 1),
        ("actor_training.end", 1),
        ("step.end", 1),
    ]


def test_async_rollout_generate_returns_after_one_ppo_step() -> None:
    worker = object.__new__(AsyncMultiStepRolloutWorker)
    worker._rank = 2
    worker._background_weight_sync_active = False
    worker.rollout_epoch = 3
    worker.finished_episodes = 5
    worker.total_num_train_envs = 7
    worker.generate_one_epoch_calls = 0

    async def generate_one_epoch(_input_channel, _output_channel):
        worker.generate_one_epoch_calls += 1
        await asyncio.sleep(0.001)

    async def wait_if_stale():
        return None

    worker.generate_one_epoch = generate_one_epoch
    worker.wait_if_stale = wait_if_stale
    worker.pop_execution_times = lambda: {"generate": 1.5}

    metric_results = []

    class _MetricChannel:
        def put(self, item, async_op=False):
            metric_results.append((item, async_op))

    async def run_generate():
        await asyncio.wait_for(
            worker._generate(object(), object(), _MetricChannel()),
            timeout=0.1,
        )

    asyncio.run(run_generate())

    assert worker.generate_one_epoch_calls == worker.rollout_epoch
    assert worker.finished_episodes == 5 + 7 * worker.rollout_epoch
    assert metric_results == [
        (
            {
                "rank": 2,
                "time": {"time/rollout/generate": 1.5},
            },
            True,
        )
    ]


def test_async_rollout_generate_can_be_restarted() -> None:
    worker = object.__new__(AsyncMultiStepRolloutWorker)
    worker._generate_task = None
    worker._rank = 0
    worker._background_weight_sync_active = False
    worker.rollout_epoch = 2
    worker.finished_episodes = None
    worker.total_num_train_envs = 4
    worker.generate_one_epoch_calls = 0

    async def generate_one_epoch(_input_channel, _output_channel):
        worker.generate_one_epoch_calls += 1
        await asyncio.sleep(0.001)

    async def wait_if_stale():
        return None

    worker.generate_one_epoch = generate_one_epoch
    worker.wait_if_stale = wait_if_stale
    worker.pop_execution_times = lambda: {}

    class _MetricChannel:
        def put(self, item, async_op=False):
            pass

    async def run_generate_twice():
        for _ in range(2):
            await asyncio.wait_for(
                worker.generate(object(), object(), _MetricChannel()),
                timeout=0.1,
            )

    asyncio.run(run_generate_twice())

    assert worker.generate_one_epoch_calls == 2 * worker.rollout_epoch
    assert worker._generate_task is None


def test_train_async_selects_gipo_runner_for_gipo_loss():
    source = Path("examples/embodiment/train_async.py").read_text()

    assert "gipo_actor_critic" in source
    assert "AsyncGIPOEmbodiedRunner" in source
    assert "AsyncGIPOEmbodiedFSDPActor" in source


def test_async_gipo_example_config_contains_required_sections():
    path = Path("examples/embodiment/config/libero_spatial_async_gipo_openpi_pi05.yaml")
    text = path.read_text()

    assert "loss_type: gipo_actor_critic" in text
    assert "replay_buffer:" in text
    assert "gipo:" in text
    assert "target_batch_size:" in text
    assert "max_wait_time_s:" in text
