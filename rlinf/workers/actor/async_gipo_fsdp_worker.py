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
import json
import os
import queue
import threading
from typing import Any

import torch

from rlinf.data.embodied_async import AsyncTrajectoryEnvelope
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.workers.actor.async_ppo_fsdp_worker import AsyncPPOEmbodiedFSDPActor


class AsyncGIPOEmbodiedFSDPActor(AsyncPPOEmbodiedFSDPActor):
    """Async GIPO actor that trains from trajectory replay segments."""

    should_stop = False

    def setup_gipo_replay_buffer(self) -> None:
        seed = self.cfg.actor.get("seed", 1234)
        auto_save_path = self.cfg.algorithm.replay_buffer.get("auto_save_path", None)
        if auto_save_path is None:
            auto_save_path = os.path.join(
                self.cfg.runner.logger.log_path,
                f"gipo_replay_buffer/rank_{self._rank}",
            )
        else:
            auto_save_path = os.path.join(auto_save_path, f"rank_{self._rank}")
        self.replay_buffer = TrajectoryReplayBuffer(
            seed=seed,
            enable_cache=self.cfg.algorithm.replay_buffer.enable_cache,
            cache_size=self.cfg.algorithm.replay_buffer.cache_size,
            sample_window_size=self.cfg.algorithm.replay_buffer.sample_window_size,
            auto_save=self.cfg.algorithm.replay_buffer.get("auto_save", False),
            auto_save_path=auto_save_path,
            trajectory_format=self.cfg.algorithm.replay_buffer.get(
                "trajectory_format", "pt"
            ),
        )

    def init_worker(self) -> None:
        super().init_worker()
        self.setup_gipo_replay_buffer()
        self._recv_queue = queue.Queue()
        self._rollout_trajectory_timestamps = []
        self._last_rollout_trajectory_timestamps = []
        self._trajectory_timestamp_file = None

    def _write_rollout_trajectory_timestamp(self, payload: dict[str, Any]) -> None:
        if self._trajectory_timestamp_file is None:
            log_path = self.cfg.runner.logger.get("log_path", "../results")
            output_dir = os.path.join(log_path, "rollout_trajectory_timestamps")
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, f"actor_rank_{self._rank}.jsonl")
            self._trajectory_timestamp_file = open(
                path,
                "a",
                encoding="utf-8",
                buffering=1,
            )
        self._trajectory_timestamp_file.write(json.dumps(payload, sort_keys=True) + "\n")

    def _record_rollout_trajectory_timestamp(
        self,
        envelope: AsyncTrajectoryEnvelope,
    ) -> None:
        if not hasattr(self, "_rollout_trajectory_timestamps"):
            self._rollout_trajectory_timestamps = []
        if not hasattr(self, "_last_rollout_trajectory_timestamps"):
            self._last_rollout_trajectory_timestamps = []

        payload = envelope.timing_payload()
        self._rollout_trajectory_timestamps.append(payload)
        self._last_rollout_trajectory_timestamps.append(payload)
        if hasattr(self, "cfg") and hasattr(self.cfg, "runner"):
            self._write_rollout_trajectory_timestamp(payload)

    def _consume_rollout_trajectory_timing_metrics(self) -> dict[str, float]:
        timestamps = getattr(self, "_last_rollout_trajectory_timestamps", [])
        self._last_rollout_trajectory_timestamps = []
        if not timestamps:
            return {}

        durations = [float(payload["duration_s"]) for payload in timestamps]
        started_at = [float(payload["started_at"]) for payload in timestamps]
        completed_at = [float(payload["completed_at"]) for payload in timestamps]
        return {
            "rollout/trajectory_count": float(len(timestamps)),
            "rollout/trajectory_duration_mean": sum(durations) / len(durations),
            "rollout/trajectory_duration_min": min(durations),
            "rollout/trajectory_duration_max": max(durations),
            "rollout/trajectory_collect_window": max(completed_at) - min(started_at),
        }

    def _drain_received_trajectories(self, max_trajectories: int | None = None) -> int:
        recv_list = []
        while True:
            if max_trajectories is not None and len(recv_list) >= max_trajectories:
                break
            try:
                envelope = self._recv_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(envelope, AsyncTrajectoryEnvelope):
                self._record_rollout_trajectory_timestamp(envelope)
                recv_list.append(envelope.trajectory)
            else:
                recv_list.append(envelope)
        if recv_list:
            self.replay_buffer.add_trajectories(recv_list)
        return len(recv_list)

    def _get_common_gipo_step_time(self, local_step_time: int) -> int:
        if not (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ):
            return local_step_time

        step_time = torch.tensor(local_step_time, dtype=torch.long, device=self.device)
        torch.distributed.all_reduce(step_time, op=torch.distributed.ReduceOp.MAX)
        return int(step_time.item())

    def _pad_gipo_tensor_time_dim(
        self,
        tensor: torch.Tensor,
        target_time: int,
        *,
        pad_value: float | bool = 0,
    ) -> torch.Tensor:
        if tensor.shape[0] == target_time:
            return tensor
        pad_shape = (target_time - tensor.shape[0], *tensor.shape[1:])
        padding = torch.full(
            pad_shape,
            fill_value=pad_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        return torch.cat([tensor, padding], dim=0)

    def _pad_gipo_replay_batch_to_step_time(
        self,
        batch: dict,
        target_step_time: int,
    ) -> None:
        boundary_fields = ("dones", "terminations", "truncations", "prev_values")
        step_fields = (
            "actions",
            "intervene_flags",
            "rewards",
            "prev_logprobs",
            "versions",
            "loss_mask",
        )

        def _pad_nested_step_tensors(values: dict) -> None:
            for key, value in list(values.items()):
                if isinstance(value, torch.Tensor):
                    values[key] = self._pad_gipo_tensor_time_dim(
                        value,
                        target_step_time,
                        pad_value=False if value.dtype == torch.bool else 0,
                    )
                elif isinstance(value, dict):
                    _pad_nested_step_tensors(value)

        for field_name in step_fields:
            value = batch.get(field_name)
            if isinstance(value, torch.Tensor):
                batch[field_name] = self._pad_gipo_tensor_time_dim(
                    value,
                    target_step_time,
                    pad_value=(
                        False
                        if field_name in ("intervene_flags", "loss_mask")
                        else 0
                    ),
                )

        for field_name in boundary_fields:
            value = batch.get(field_name)
            if isinstance(value, torch.Tensor):
                batch[field_name] = self._pad_gipo_tensor_time_dim(
                    value,
                    target_step_time + 1,
                )

        for dict_name in ("forward_inputs", "curr_obs", "next_obs"):
            values = batch.get(dict_name)
            if not values:
                continue
            _pad_nested_step_tensors(values)

    async def _wait_for_replay_buffer_ready(self, min_buffer_size: int) -> None:
        while not self.should_stop:
            self._drain_received_trajectories(
                max_trajectories=self.cfg.actor.get("recv_drain_max_trajectories", 256)
                if hasattr(self.cfg, "actor")
                else None
            )
            if await self.replay_buffer.is_ready_async(min_buffer_size):
                return
            await asyncio.sleep(1.0)

    async def _prepare_gipo_replay_batch(self) -> dict:
        min_buffer_size = self.cfg.algorithm.replay_buffer.get("min_buffer_size", 1)
        await self._wait_for_replay_buffer_ready(min_buffer_size)
        target_segments = self.cfg.algorithm.get("gipo", {}).get(
            "target_batch_segments", 1
        )
        batch = self.replay_buffer.sample_trajectory_batch(target_segments)
        local_step_time = int(batch["prev_logprobs"].shape[0])
        target_step_time = self._get_common_gipo_step_time(local_step_time)
        self._pad_gipo_replay_batch_to_step_time(batch, target_step_time)
        self.rollout_batch = batch
        return self.load_batch(batch)

    async def run_training(self):
        await self._prepare_gipo_replay_batch()
        training_metrics = await asyncio.to_thread(
            AsyncPPOEmbodiedFSDPActor.run_training,
            self,
        )
        timing_metrics = self._consume_rollout_trajectory_timing_metrics()
        if timing_metrics:
            training_metrics.update(timing_metrics)
        return training_metrics

    async def recv_trajectories_async(self, input_channel):
        if getattr(self, "_recv_queue", None) is None:
            self._recv_queue = queue.Queue()
        if getattr(self, "_recv_thread", None) is None or not self._recv_thread.is_alive():
            self._recv_thread = threading.Thread(
                target=self._recv_thread_main,
                args=(input_channel,),
                daemon=True,
            )
            self._recv_thread.start()

    def _recv_thread_main(self, input_channel):
        while not self.should_stop:
            envelope = input_channel.get()
            self._recv_queue.put(envelope)

    async def stop(self):
        self.should_stop = True
        recv_thread = getattr(self, "_recv_thread", None)
        if recv_thread is not None and recv_thread.is_alive():
            await asyncio.to_thread(recv_thread.join, 5)
        timestamp_file = getattr(self, "_trajectory_timestamp_file", None)
        if timestamp_file is not None:
            timestamp_file.close()
            self._trajectory_timestamp_file = None
