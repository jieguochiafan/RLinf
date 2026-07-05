import asyncio

import torch

from rlinf.data.embodied_async import build_inference_response_key
from rlinf.data.embodied_io_struct import (
    EmbodiedRolloutResult,
    EnvOutput,
    RolloutResult,
)
from rlinf.workers.env.async_env_worker import AsyncEnvWorker


def test_gipo_fixed_horizon_does_not_flush_on_done_before_horizon():
    worker = object.__new__(AsyncEnvWorker)
    worker.cfg = type("Cfg", (), {})()
    worker.cfg.env = type("EnvCfg", (), {})()
    worker.cfg.env.train = type(
        "TrainCfg", (), {"auto_reset": False, "max_episode_steps": 10}
    )()
    worker.stage_num = 1
    worker.train_num_envs_per_stage = 2
    worker.n_train_chunk_steps = 3
    worker.rollout_results = [EmbodiedRolloutResult(max_episode_length=10)]
    worker.compute_bootstrap_rewards = (
        lambda env_output, bootstrap_values, reward_model_output: env_output.rewards
    )

    dones = torch.zeros(2, 1, dtype=torch.bool)
    dones[0, 0] = True
    env_output = EnvOutput(
        obs={"states": torch.zeros(2, 3)},
        rewards=torch.ones(2, 1),
        dones=dones,
        terminations=dones.clone(),
        truncations=torch.zeros_like(dones),
    )
    rollout_result = RolloutResult(
        actions=torch.zeros(2, 2),
        prev_logprobs=torch.zeros(2, 2),
        prev_values=torch.zeros(2, 1),
        versions=torch.zeros(2, 2),
        forward_inputs={"action": torch.zeros(2, 2)},
    )

    should_flush = worker._append_gipo_step_and_check_flush(
        stage_id=0,
        env_output=env_output,
        rollout_result=rollout_result,
        chunk_step_idx=0,
    )

    assert not should_flush
    assert len(worker.rollout_results[0].dones) == 1


class _ResponseChannel:
    def __init__(self, response):
        self.response = response
        self.keys = []

    def get(self, key=None, async_op=False):
        self.keys.append((key, async_op))
        return _AsyncWork(self.response)


class _AsyncWork:
    def __init__(self, value):
        self.value = value

    async def async_wait(self):
        return self.value


def test_env_waits_on_request_specific_response_key():
    worker = object.__new__(AsyncEnvWorker)
    response = object()
    channel = _ResponseChannel(response)

    got = asyncio.run(worker._recv_gipo_inference_response(channel, 3, 1, "req"))

    assert got is response
    assert channel.keys == [(build_inference_response_key(3, 1, "req"), True)]
