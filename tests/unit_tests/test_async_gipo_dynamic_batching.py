import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from rlinf.data.embodied_async import InferenceRequest
from rlinf.data.embodied_io_struct import RolloutResult
from rlinf.workers.rollout.hf.async_batching import DynamicBatchState
from rlinf.workers.rollout.hf.async_huggingface_worker import (
    AsyncMultiStepRolloutWorker,
)


def test_target_batch_size_triggers_flush():
    state = DynamicBatchState(target_batch_size=4, max_wait_time_s=1.0)
    state.mark_first_request(now=10.0)

    assert state.should_flush(queue_size=4, now=10.1)


def test_max_wait_time_triggers_flush_before_target_size():
    state = DynamicBatchState(target_batch_size=8, max_wait_time_s=0.05)
    state.mark_first_request(now=10.0)

    assert state.should_flush(queue_size=2, now=10.06)


def test_empty_queue_never_flushes():
    state = DynamicBatchState(target_batch_size=1, max_wait_time_s=0.0)
    state.mark_first_request(now=10.0)

    assert not state.should_flush(queue_size=0, now=11.0)


def test_gipo_split_rollout_result_preserves_request_sizes():
    worker = object.__new__(AsyncMultiStepRolloutWorker)
    rollout_result = RolloutResult(
        actions=torch.arange(6, dtype=torch.float32).reshape(3, 2),
        prev_logprobs=torch.zeros(3, 2),
        prev_values=torch.zeros(3, 1),
        forward_inputs={"action": torch.zeros(3, 2)},
        versions=torch.ones(3, 2),
    )

    pieces = worker._split_gipo_rollout_result_by_sizes(rollout_result, [1, 2])

    assert pieces[0].actions.shape == (1, 2)
    assert pieces[1].actions.shape == (2, 2)
    assert pieces[1].forward_inputs["action"].shape == (2, 2)


class _AsyncWork:
    def __init__(self, value):
        self.value = value

    async def async_wait(self):
        return self.value


class _QueueChannel:
    def __init__(self, values=None):
        self.values = list(values or [])
        self.puts = []

    def get(self, async_op=False, key=None):
        assert async_op
        return _AsyncWork(self.values.pop(0))

    def get_nowait(self):
        if not self.values:
            raise asyncio.QueueEmpty
        return self.values.pop(0)

    def put(self, item, key=None, async_op=False):
        self.puts.append((key, item, async_op))


async def _flush_once(worker, requests):
    response_channel = _QueueChannel()
    await worker._flush_gipo_inference_requests(requests, response_channel)
    return response_channel.puts


def test_gipo_flush_sends_one_response_per_request():
    worker = object.__new__(AsyncMultiStepRolloutWorker)
    worker._rank = 0
    worker.version = 5
    worker.predict = lambda obs, profile_context=None: (
        torch.zeros(len(obs["states"]), 2),
        {
            "prev_logprobs": torch.zeros(len(obs["states"]), 2),
            "prev_values": torch.zeros(len(obs["states"]), 1),
            "forward_inputs": {"action": torch.zeros(len(obs["states"]), 2)},
        },
    )

    reqs = [
        InferenceRequest(
            request_id="r0",
            env_rank=0,
            stage_id=0,
            env_ids=[0],
            obs={"states": torch.zeros(1, 3)},
        ),
        InferenceRequest(
            request_id="r1",
            env_rank=0,
            stage_id=0,
            env_ids=[1, 2],
            obs={"states": torch.zeros(2, 3)},
        ),
    ]
    puts = asyncio.run(_flush_once(worker, reqs))

    assert len(puts) == 2
    assert puts[0][0] == "action:0:0:r0"
    assert puts[1][1].rollout_result.actions.shape == (2, 2)


class _CancellingRequestChannel:
    def get_nowait(self):
        raise asyncio.CancelledError


def test_gipo_inference_service_starts_and_stops_torch_profiler():
    worker = object.__new__(AsyncMultiStepRolloutWorker)
    worker.cfg = SimpleNamespace(
        algorithm={"async_inference": {"target_batch_size": 1, "max_wait_time_s": 0.0}}
    )
    worker._background_weight_sync_active = False
    worker._start_torch_profiler = MagicMock()
    worker._stop_torch_profiler = MagicMock()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            worker._serve_inference_gipo(
                _CancellingRequestChannel(),
                _QueueChannel(),
                _QueueChannel(),
            )
        )

    worker._start_torch_profiler.assert_called_once_with()
    worker._stop_torch_profiler.assert_called_once_with()


def test_gipo_action_generation_phase_uses_generation_profiler(monkeypatch):
    worker = object.__new__(AsyncMultiStepRolloutWorker)
    worker._torch_profiler = object()
    context = object()
    record_function = MagicMock(return_value=context)
    monkeypatch.setattr("torch.profiler.record_function", record_function)

    assert (
        worker._profile_generation_context({"phase": "gipo_action_generation"})
        is context
    )
    assert worker._should_step_generation_profiler(
        {"phase": "gipo_action_generation"}
    )
    record_function.assert_called_once_with("generation")
