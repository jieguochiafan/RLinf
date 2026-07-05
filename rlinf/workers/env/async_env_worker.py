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

from omegaconf.omegaconf import DictConfig

from rlinf.data.embodied_async import (
    AsyncTrajectoryEnvelope,
    InferenceRequest,
    build_inference_request_id,
    build_inference_response_key,
)
from rlinf.data.embodied_io_struct import ChunkStepResult, EnvOutput, RolloutResult
from rlinf.scheduler import Channel
from rlinf.workers.env.env_worker import EnvWorker


class AsyncEnvWorker(EnvWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._interact_task: asyncio.Task = None
        assert not self.enable_offload, "Offload not supported in AsyncEnvWorker"

    async def interact(
        self,
        input_channel: Channel,
        rollout_channel: Channel,
        reward_channel: Channel | None,
        actor_channel: Channel | None,
        metric_channel: Channel,
    ):
        assert self._interact_task is None or self._interact_task.done(), (
            "Previous interact task is still running while a new interact call is made."
        )
        self._interact_task = asyncio.create_task(
            self._interact(
                input_channel,
                rollout_channel,
                reward_channel,
                actor_channel,
                metric_channel,
            )
        )
        try:
            await self._interact_task
        except asyncio.CancelledError:
            pass

    async def _interact(
        self,
        input_channel: Channel,
        rollout_channel: Channel,
        reward_channel: Channel | None,
        actor_channel: Channel | None,
        metric_channel: Channel,
    ):
        while True:
            env_metrics = await self._run_interact_once(
                input_channel,
                rollout_channel,
                reward_channel,
                actor_channel,
                cooperative_yield=True,
            )

            env_metrics = {f"env/{k}": v for k, v in env_metrics.items()}
            env_interact_time_metrics = self.pop_execution_times()
            env_interact_time_metrics = {
                f"time/env/{k}": v for k, v in env_interact_time_metrics.items()
            }
            metrics = {
                "rank": self._rank,
                "env": env_metrics,
                "time": env_interact_time_metrics,
            }
            metric_channel.put(metrics, async_op=True)

    async def stop(self):
        if self._interact_task is not None and not self._interact_task.done():
            self._interact_task.cancel()

    def _append_gipo_step_and_check_flush(
        self,
        stage_id: int,
        env_output: EnvOutput,
        rollout_result: RolloutResult,
        chunk_step_idx: int,
    ) -> bool:
        rewards = self.compute_bootstrap_rewards(
            env_output,
            rollout_result.bootstrap_values,
            reward_model_output=None,
        )
        step_result = ChunkStepResult(
            actions=rollout_result.forward_inputs.get(
                "action", rollout_result.actions
            ),
            prev_logprobs=rollout_result.prev_logprobs,
            prev_values=rollout_result.prev_values,
            forward_inputs=rollout_result.forward_inputs,
            versions=rollout_result.versions,
            dones=env_output.dones,
            truncations=env_output.truncations,
            terminations=env_output.terminations,
            rewards=rewards,
        )
        self.rollout_results[stage_id].append_step_result(step_result)
        if not self.cfg.env.train.auto_reset:
            return chunk_step_idx + 1 >= self.n_train_chunk_steps
        return bool(env_output.dones is not None and env_output.dones[:, -1].any().item())

    async def _recv_gipo_inference_response(
        self,
        response_channel: Channel,
        env_rank: int,
        stage_id: int,
        request_id: str,
    ):
        key = build_inference_response_key(env_rank, stage_id, request_id)
        response = await response_channel.get(key=key, async_op=True).async_wait()
        if getattr(response, "has_error", False):
            raise RuntimeError(
                f"GIPO inference failed for {request_id}: {response.error}"
            )
        return response

    async def interact_gipo_async(
        self,
        request_channel: Channel,
        response_channel: Channel,
        trajectory_channel: Channel,
        metric_channel: Channel,
    ):
        assert self._interact_task is None or self._interact_task.done(), (
            "Previous GIPO interact task is still running."
        )
        self._interact_task = asyncio.create_task(
            self._interact_gipo_async(
                request_channel,
                response_channel,
                trajectory_channel,
                metric_channel,
            )
        )
        try:
            await self._interact_task
        except asyncio.CancelledError:
            pass

    async def _interact_gipo_async(
        self,
        request_channel: Channel,
        response_channel: Channel,
        trajectory_channel: Channel,
        metric_channel: Channel,
    ) -> None:
        while True:
            for chunk_step_idx in range(self.n_train_chunk_steps):
                for stage_id in range(self.stage_num):
                    env_output = EnvOutput(
                        obs=self.last_obs_list[stage_id],
                    )
                    request_id = build_inference_request_id(
                        env_rank=self._rank,
                        stage_id=stage_id,
                        local_step=chunk_step_idx,
                    )
                    request_channel.put(
                        InferenceRequest(
                            request_id=request_id,
                            env_rank=self._rank,
                            stage_id=stage_id,
                            env_ids=list(range(self.train_num_envs_per_stage)),
                            obs=env_output.obs,
                            policy_version_hint=None,
                        ),
                        async_op=True,
                    )
                    response = await self._recv_gipo_inference_response(
                        response_channel,
                        self._rank,
                        stage_id,
                        request_id,
                    )
                    next_output, _ = self.env_interact_step(
                        response.actions,
                        stage_id,
                        forward_inputs=response.rollout_result.forward_inputs,
                        chunk_step_idx=chunk_step_idx,
                    )
                    should_flush = self._append_gipo_step_and_check_flush(
                        stage_id,
                        next_output,
                        response.rollout_result,
                        chunk_step_idx,
                    )
                    self.last_obs_list[stage_id] = next_output.obs
                    if should_flush:
                        trajectory = self.rollout_results[stage_id].to_trajectory()
                        trajectory_channel.put(
                            AsyncTrajectoryEnvelope(
                                env_rank=self._rank,
                                stage_id=stage_id,
                                segment_type="fixed_horizon"
                                if not self.cfg.env.train.auto_reset
                                else "episode",
                                auto_reset=self.cfg.env.train.auto_reset,
                                trajectory=trajectory,
                                completed_at=time.perf_counter(),
                                last_policy_version=response.policy_version,
                            ),
                            async_op=True,
                        )
                        self.rollout_results[stage_id].clear()
