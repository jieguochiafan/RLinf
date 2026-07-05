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
import os
import queue
import threading

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
                recv_list.append(envelope.trajectory)
            else:
                recv_list.append(envelope)
        if recv_list:
            self.replay_buffer.add_trajectories(recv_list)
        return len(recv_list)

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
        self.rollout_batch = batch
        return self.load_batch(batch)

    async def run_training(self):
        await self._prepare_gipo_replay_batch()
        return await asyncio.to_thread(AsyncPPOEmbodiedFSDPActor.run_training, self)

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
