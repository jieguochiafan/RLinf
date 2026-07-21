# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import time
from typing import Any

import torch
from omegaconf.omegaconf import DictConfig

from rlinf.data.embodied_async import InferenceRequest, InferenceResponse
from rlinf.data.embodied_io_struct import RolloutResult
from rlinf.scheduler import Channel
from rlinf.workers.rollout.hf.async_batching import DynamicBatchState
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class AsyncMultiStepRolloutWorker(MultiStepRolloutWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._generate_task: asyncio.Task = None
        self.staleness_threshold = cfg.algorithm.get("staleness_threshold", None)
        self.num_envs_per_stage = (
            self.cfg.env.train.total_num_envs
            // self._world_size
            // self.num_pipeline_stages
        )
        assert not self.enable_offload, (
            "Offload not supported in AsyncMultiStepRolloutWorker"
        )

        self._background_weight_sync_active = self.cfg.actor.get(
            "sync_weight_no_wait", False
        )
        self._weight_sync_requested = False
        self._weight_sync_work = None
        self._weight_sync_apply_total = 0
        self._weight_sync_coalesced_total = 0
        self._weight_sync_request_total = 0

    async def generate(
        self,
        input_channel: Channel,
        output_channel: Channel,
        metric_channel: Channel,
    ):
        assert self._generate_task is None, (
            "generate task is not None but generate function is called."
        )
        self._generate_task = asyncio.create_task(
            self._generate(input_channel, output_channel, metric_channel)
        )
        try:
            await self._generate_task
        except asyncio.CancelledError:
            pass

    async def _generate(
        self,
        input_channel: Channel,
        output_channel: Channel,
        metric_channel: Channel,
    ):
        while True:
            if self._background_weight_sync_active:
                await self._poll_background_weight_sync()
            await self.wait_if_stale()
            for _ in range(self.rollout_epoch):
                await self.generate_one_epoch(input_channel, output_channel)
            if self.finished_episodes is not None:
                self.finished_episodes += self.total_num_train_envs * self.rollout_epoch
            rollout_metrics = self.pop_execution_times()
            rollout_metrics = {
                f"time/rollout/{k}": v for k, v in rollout_metrics.items()
            }
            metric_channel.put(
                {"rank": self._rank, "time": rollout_metrics},
                async_op=True,
            )

    async def wait_if_stale(self) -> None:
        if self.staleness_threshold is None:
            return
        assert self.finished_episodes is not None, (
            "finished_episodes should be initialized."
        )

        def has_capacity() -> bool:
            capacity = (
                (self.staleness_threshold + self.version + 1)
                * self.total_num_train_envs
                * self.rollout_epoch
            )
            return (
                self.finished_episodes + self.total_num_train_envs * self.rollout_epoch
                <= capacity
            )

        if has_capacity():
            return
        with self.timeline.span(
            "rollout.staleness_wait",
            policy_version=self.version,
            finished_episodes=self.finished_episodes,
        ):
            while not has_capacity():
                await asyncio.sleep(0.01)

    def stop(self):
        if self._generate_task is not None and not self._generate_task.done():
            self._generate_task.cancel()

    async def _recv_and_apply_actor_sync(self) -> int:
        await super().sync_model_from_actor()
        return self.version

    def _start_background_weight_sync_if_needed(self):
        if (
            not self._background_weight_sync_active
            or not self._weight_sync_requested
            or self._weight_sync_work is not None
        ):
            return

        self._weight_sync_requested = False
        self._weight_sync_work = asyncio.create_task(self._recv_and_apply_actor_sync())

    async def _poll_background_weight_sync(self):
        self._start_background_weight_sync_if_needed()
        if self._weight_sync_work is None:
            return

        if not self._weight_sync_work.done():
            return

        await self._weight_sync_work
        self._weight_sync_work = None
        self._weight_sync_apply_total += 1

        self._start_background_weight_sync_if_needed()

    async def request_actor_sync_model(self):
        self._weight_sync_request_total += 1
        if self._weight_sync_requested or self._weight_sync_work is not None:
            self._weight_sync_coalesced_total += 1
        self._weight_sync_requested = True
        self._start_background_weight_sync_if_needed()

    def _split_gipo_rollout_result_by_sizes(
        self, rollout_result: RolloutResult, sizes: list[int]
    ) -> list[RolloutResult]:
        return self._split_rollout_result(rollout_result, sizes)

    def _merge_gipo_request_obs(
        self, requests: list[InferenceRequest]
    ) -> dict[str, Any]:
        obs_batches = [{"obs": request.obs, "final_obs": None} for request in requests]
        return self._merge_obs_batches(obs_batches)["obs"]

    async def _flush_gipo_inference_requests(
        self,
        requests: list[InferenceRequest],
        response_channel: Channel,
    ) -> None:
        if not requests:
            return
        merged_obs = self._merge_gipo_request_obs(requests)
        actions, result = self.predict(
            merged_obs,
            profile_context={"phase": "gipo_action_generation"},
        )
        rollout_result = RolloutResult(
            actions=actions,
            prev_logprobs=result.get("prev_logprobs"),
            prev_values=result.get("prev_values"),
            forward_inputs=result.get("forward_inputs", {}),
            versions=torch.full_like(
                result["prev_logprobs"],
                float(self.version),
                dtype=torch.float32,
            ),
        )
        sizes = [len(request.env_ids) for request in requests]
        pieces = self._split_gipo_rollout_result_by_sizes(rollout_result, sizes)
        for request, piece in zip(requests, pieces, strict=True):
            response_channel.put(
                InferenceResponse(
                    request_id=request.request_id,
                    rollout_rank=self._rank,
                    actions=piece.actions,
                    rollout_result=piece,
                    policy_version=int(self.version),
                ),
                key=request.response_key,
                async_op=True,
            )

    async def serve_inference_gipo(
        self,
        request_channel: Channel,
        response_channel: Channel,
        metric_channel: Channel,
    ):
        assert self._generate_task is None, "GIPO inference service is already running."
        self._generate_task = asyncio.create_task(
            self._serve_inference_gipo(
                request_channel,
                response_channel,
                metric_channel,
            )
        )
        try:
            await self._generate_task
        except asyncio.CancelledError:
            pass
        finally:
            self._generate_task = None

    async def _serve_inference_gipo(
        self,
        request_channel: Channel,
        response_channel: Channel,
        metric_channel: Channel,
    ) -> None:
        async_cfg = self.cfg.algorithm.get("async_inference", {})
        batch_state = DynamicBatchState(
            target_batch_size=async_cfg.get("target_batch_size", 1),
            max_wait_time_s=async_cfg.get("max_wait_time_s", 0.0),
        )
        pending: list[InferenceRequest] = []
        while True:
            if self._background_weight_sync_active:
                await self._poll_background_weight_sync()
            try:
                request = request_channel.get_nowait()
            except asyncio.QueueEmpty:
                if batch_state.should_flush(len(pending), time.perf_counter()):
                    await self._flush_gipo_inference_requests(pending, response_channel)
                    pending.clear()
                    batch_state.reset()
                await asyncio.sleep(0)
                continue

            pending.append(request)
            batch_state.mark_first_request(time.perf_counter())
            if batch_state.should_flush(len(pending), time.perf_counter()):
                await self._flush_gipo_inference_requests(pending, response_channel)
                pending.clear()
                batch_state.reset()
