import asyncio
from queue import Queue
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from rlinf.data.embodied_async import AsyncTrajectoryEnvelope
from rlinf.runners.async_gipo_embodied_runner import AsyncGIPOEmbodiedRunner
from rlinf.workers.actor.async_gipo_fsdp_worker import AsyncGIPOEmbodiedFSDPActor


class _Buffer:
    def __init__(self):
        self.items = []

    def add_trajectories(self, items):
        self.items.extend(items)


def test_gipo_actor_drains_envelopes_into_replay_buffer():
    actor = object.__new__(AsyncGIPOEmbodiedFSDPActor)
    actor._recv_queue = Queue()
    actor.replay_buffer = _Buffer()
    actor._recv_queue.put(
        AsyncTrajectoryEnvelope(
            env_rank=0,
            stage_id=0,
            segment_type="fixed_horizon",
            auto_reset=False,
            trajectory="traj",
            completed_at=1.0,
            last_policy_version=2,
        )
    )

    actor._drain_received_trajectories()

    assert actor.replay_buffer.items == ["traj"]


class _ReadyBuffer:
    def __init__(self):
        self.size = 1

    async def is_ready_async(self, min_size):
        return True

    def sample_trajectory_batch(self, num_trajectories):
        return {
            "prev_logprobs": torch.zeros(2, 1, 1),
            "prev_values": torch.zeros(3, 1, 1),
            "rewards": torch.ones(2, 1, 1),
            "dones": torch.zeros(3, 1, 1, dtype=torch.bool),
            "terminations": torch.zeros(3, 1, 1, dtype=torch.bool),
            "truncations": torch.zeros(3, 1, 1, dtype=torch.bool),
            "versions": torch.zeros(2, 1, 1),
            "forward_inputs": {"action": torch.zeros(2, 1, 1)},
        }


def test_gipo_actor_loads_replay_rollout_batch_before_training():
    actor = object.__new__(AsyncGIPOEmbodiedFSDPActor)
    actor.replay_buffer = _ReadyBuffer()
    actor._drain_received_trajectories = lambda max_trajectories=None: 0
    actor.cfg = OmegaConf.create(
        {
            "actor": {"recv_drain_max_trajectories": 256},
            "algorithm": {
                "replay_buffer": {"min_buffer_size": 1},
                "gipo": {"target_batch_segments": 1},
            },
        }
    )
    actor.load_batch = lambda batch: {"reward": 1.0}

    metrics = asyncio.run(actor._prepare_gipo_replay_batch())

    assert metrics == {"reward": 1.0}
    assert actor.rollout_batch["prev_logprobs"].shape == (2, 1, 1)


class _DoneHandle:
    def __init__(self, value=None):
        self.value = value if value is not None else [None]

    def wait(self):
        return self.value

    def consume_durations(self, return_per_rank=False):
        return ({}, [{}]) if return_per_rank else {}


class _Service:
    worker_group_name = "Group"

    def __init__(self):
        self.started = 0
        self.stopped = 0
        self.trained = 0

    def set_global_step(self, step):
        return _DoneHandle()

    def sync_model_from_actor(self):
        return _DoneHandle()

    def sync_model_to_rollout(self):
        return _DoneHandle()

    def serve_inference_gipo(self, **kwargs):
        self.started += 1
        return _DoneHandle()

    def interact_gipo_async(self, **kwargs):
        self.started += 1
        return _DoneHandle()

    def recv_trajectories_async(self, **kwargs):
        self.started += 1
        return _DoneHandle()

    def run_training(self):
        self.trained += 1
        return _DoneHandle([{"loss": 0.1}])

    def stop(self):
        self.stopped += 1
        return _DoneHandle()


def test_async_gipo_runner_starts_and_stops_services():
    runner = object.__new__(AsyncGIPOEmbodiedRunner)
    runner.global_step = 0
    runner.max_steps = 1
    runner.cfg = SimpleNamespace(
        runner=SimpleNamespace(save_interval=-1, val_check_interval=-1)
    )
    runner.actor = _Service()
    runner.rollout = _Service()
    runner.env = _Service()
    runner.metric_logger = SimpleNamespace(log=lambda *a, **k: None, finish=lambda: None)
    runner.timer = SimpleNamespace(
        consume_durations=lambda: {}, __call__=lambda self, name: self
    )
    runner.update_rollout_weights = lambda: None
    runner._aggregate_numeric_metrics = lambda metrics: metrics[0] if metrics else {}
    runner.print_metrics_table_async = lambda *a, **k: None
    runner._save_checkpoint = lambda: None
    runner._drain_gipo_metrics = lambda: {}
    runner._create_gipo_channels_for_test()

    runner.run()

    assert runner.actor.trained == 1
    assert runner.env.stopped == 1
    assert runner.rollout.stopped == 1
