import asyncio
from contextlib import contextmanager

import torch
from omegaconf import OmegaConf

from rlinf.data.embodied_io_struct import RolloutResult
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class FakeProfiler:
    def __init__(self):
        self.events = []

    def event(self, event, **fields):
        self.events.append((event, fields))

    @contextmanager
    def span(self, event, **fields):
        self.events.append((event + ".start", fields))
        yield
        self.events.append((event + ".end", fields))

    def record_metrics(self, event, metrics, **fields):
        self.events.append((event, {"metrics": metrics, **fields}))

    def flush(self):
        self.events.append(("flush", {}))


class FakeChannel:
    def __init__(self, item):
        self.item = item
        self.puts = []

    def get(self, key=None):
        return self.item

    def put(self, item, key=None, **kwargs):
        self.puts.append((key, item, kwargs))


class FakeEnv:
    def __init__(self):
        self.timestamp_contexts = []

    def chunk_step(self, chunk_actions, denoising_curvature=None):
        del chunk_actions, denoising_curvature
        obs = {"states": torch.zeros(2, 4)}
        dones = torch.zeros(2, 1, dtype=torch.bool)
        rewards = torch.zeros(2, 1)
        infos = {}
        return [obs], rewards, dones, dones, [infos]

    def set_sim_timestamp_context(self, context):
        self.timestamp_contexts.append(context)

    def get_last_chunk_profile(self):
        return {"substep_count": 2, "child_step_s": 0.12}


class FakeAsyncWait:
    def __init__(self, item):
        self.item = item

    async def async_wait(self):
        return self.item


class FakeAsyncChannel:
    def __init__(self, item):
        self.item = item
        self.puts = []

    def get(self, key=None, async_op=False):
        assert async_op is True
        return FakeAsyncWait(self.item)

    def put(self, item, key=None, **kwargs):
        self.puts.append((key, item, kwargs))


def _make_env_worker_for_recv():
    worker = object.__new__(EnvWorker)
    worker._rank = 0
    worker._timer_metrics = {}
    worker.src_rank_map = {"rollout_train": [(0, 2)]}
    worker.rollout_profiler = FakeProfiler()
    return worker


def _make_env_worker_for_step():
    worker = object.__new__(EnvWorker)
    worker._rank = 0
    worker._timer_metrics = {}
    worker.cfg = OmegaConf.create(
        {
            "runner": {"logger": {"log_path": "/tmp/unused"}},
            "env": {
                "train": {
                    "env_type": "libero",
                    "auto_reset": True,
                    "ignore_terminations": False,
                    "wm_env_type": None,
                }
            },
            "actor": {
                "model": {
                    "model_type": "openpi",
                    "num_action_chunks": 1,
                    "action_dim": 12,
                    "policy_setup": None,
                }
            },
        }
    )
    worker.log_sim_timestamps = False
    worker.log_sim_affinity_interval = 0
    worker.stage_num = 1
    worker.train_num_envs_per_stage = 2
    worker.use_external_reward_model = False
    worker._torch_profiler = None
    worker._torch_profiler_step_enabled = False
    worker.env_list = [FakeEnv()]
    worker.rollout_profiler = FakeProfiler()
    return worker


def _make_rollout_worker_for_recv():
    worker = object.__new__(MultiStepRolloutWorker)
    worker._rank = 0
    worker._timer_metrics = {}
    worker.src_ranks = {"train": [(0, 2)]}
    worker.rollout_profiler = FakeProfiler()
    return worker


def test_env_worker_profiles_recv_rollout_results():
    worker = _make_env_worker_for_recv()
    result = RolloutResult(
        actions=torch.zeros(2, 5, 12),
        prev_logprobs=None,
        prev_values=None,
        bootstrap_values=None,
        forward_inputs={"action": torch.zeros(2, 5, 12)},
    )

    worker.recv_rollout_results(FakeChannel(result), mode="train")

    event_names = [name for name, _fields in worker.rollout_profiler.events]
    assert "env.recv_rollout_results.start" in event_names
    assert "env.recv_rollout_results.end" in event_names


def test_env_worker_profiles_env_interact_step():
    worker = _make_env_worker_for_step()

    worker.env_interact_step(
        torch.zeros(2, 1, 12),
        stage_id=0,
        epoch=3,
        chunk_step_idx=4,
    )

    event_names = [name for name, _fields in worker.rollout_profiler.events]
    assert "env.prepare_actions.start" in event_names
    assert "env.prepare_actions.end" in event_names
    assert "env.chunk_step.start" in event_names
    assert "env.chunk_step.end" in event_names
    assert "env.chunk_profile" in event_names


def test_env_worker_profiles_send_env_batch():
    worker = object.__new__(EnvWorker)
    worker._rank = 0
    worker._timer_metrics = {}
    worker.dst_rank_map = {"rollout_train": [(0, 1), (1, 1)]}
    worker.rollout_profiler = FakeProfiler()
    channel = FakeChannel(None)

    worker.send_env_batch(
        channel,
        {"obs": {"states": torch.zeros(2, 4), "task_descriptions": ["a", "b"]}},
        mode="train",
    )

    event_names = [name for name, _fields in worker.rollout_profiler.events]
    assert "env.send_env_batch.start" in event_names
    assert "env.send_env_batch.end" in event_names
    assert len(channel.puts) == 2


def test_env_worker_initializes_rollout_profiler_from_cfg(monkeypatch, tmp_path):
    captured = {}

    def fake_factory(cfg, *, component, rank):
        captured["component"] = component
        captured["rank"] = rank
        return FakeProfiler()

    monkeypatch.setattr(
        "rlinf.workers.env.env_worker.make_rollout_profiler",
        fake_factory,
    )
    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create({"runner": {"logger": {"log_path": str(tmp_path)}}})
    worker._rank = 4

    worker._init_rollout_profiler()

    assert captured == {"component": "env", "rank": 4}
    assert isinstance(worker.rollout_profiler, FakeProfiler)


def test_rollout_worker_profiles_recv_env_output():
    worker = _make_rollout_worker_for_recv()
    obs_batch = {
        "obs": {
            "states": torch.zeros(2, 4),
            "task_descriptions": ["a", "b"],
        },
        "final_obs": None,
    }

    asyncio.run(worker.recv_env_output(FakeAsyncChannel(obs_batch), mode="train"))

    event_names = [name for name, _fields in worker.rollout_profiler.events]
    assert "rollout.recv_env_output.start" in event_names
    assert "rollout.recv_env_output.end" in event_names


def test_rollout_worker_profiles_send_rollout_result():
    worker = object.__new__(MultiStepRolloutWorker)
    worker._rank = 0
    worker._timer_metrics = {}
    worker.dst_ranks = {"train": [(0, 1), (1, 1)]}
    worker.rollout_profiler = FakeProfiler()
    channel = FakeAsyncChannel(None)

    worker.send_rollout_result(
        channel,
        RolloutResult(
            actions=torch.zeros(2, 1, 12),
            prev_logprobs=torch.zeros(2, 1),
            prev_values=torch.zeros(2, 1),
            forward_inputs={"action": torch.zeros(2, 1, 12)},
        ),
        mode="train",
    )

    event_names = [name for name, _fields in worker.rollout_profiler.events]
    assert "rollout.send_rollout_result.start" in event_names
    assert "rollout.send_rollout_result.end" in event_names
    assert len(channel.puts) == 2


def test_rollout_worker_initializes_rollout_profiler_from_cfg(monkeypatch, tmp_path):
    captured = {}

    def fake_factory(cfg, *, component, rank):
        captured["component"] = component
        captured["rank"] = rank
        return FakeProfiler()

    monkeypatch.setattr(
        "rlinf.workers.rollout.hf.huggingface_worker.make_rollout_profiler",
        fake_factory,
    )
    worker = object.__new__(MultiStepRolloutWorker)
    worker.cfg = OmegaConf.create({"runner": {"logger": {"log_path": str(tmp_path)}}})
    worker._rank = 5

    worker._init_rollout_profiler()

    assert captured == {"component": "rollout", "rank": 5}
    assert isinstance(worker.rollout_profiler, FakeProfiler)

