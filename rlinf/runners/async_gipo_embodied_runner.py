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

import contextlib
import time

from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Channel
from rlinf.utils.runner_utils import check_progress


class AsyncGIPOEmbodiedRunner(EmbodiedRunner):
    """Runner for async GIPO embodied training services."""

    def __init__(self, cfg, actor, rollout, env, reward=None, critic=None):
        super().__init__(cfg, actor, rollout, env, critic, reward)
        self._create_gipo_channels()

    def _create_gipo_channels(self) -> None:
        self.gipo_request_channel = Channel.create("GIPOInferenceRequest")
        self.gipo_response_channel = Channel.create("GIPOInferenceResponse")
        self.gipo_trajectory_channel = Channel.create("GIPOTrajectory")
        self.gipo_metric_channel = Channel.create("GIPOMetric")

    def _create_gipo_channels_for_test(self) -> None:
        self.gipo_request_channel = object()
        self.gipo_response_channel = object()
        self.gipo_trajectory_channel = object()
        self.gipo_metric_channel = object()

    def _drain_gipo_metrics(self) -> dict:
        return {}

    def run(self) -> None:
        start_step = self.global_step
        start_time = time.time()

        self.actor.set_global_step(self.global_step).wait()
        self.rollout.set_global_step(self.global_step).wait()
        self.env.set_global_step(self.global_step).wait()
        self.update_rollout_weights()

        env_handle = self.env.interact_gipo_async(
            request_channel=self.gipo_request_channel,
            response_channel=self.gipo_response_channel,
            trajectory_channel=self.gipo_trajectory_channel,
            metric_channel=self.gipo_metric_channel,
        )
        rollout_handle = self.rollout.serve_inference_gipo(
            request_channel=self.gipo_request_channel,
            response_channel=self.gipo_response_channel,
            metric_channel=self.gipo_metric_channel,
        )
        actor_recv_handle = self.actor.recv_trajectories_async(
            input_channel=self.gipo_trajectory_channel,
        )

        while self.global_step < self.max_steps:
            timer_context = (
                self.timer("actor_training")
                if callable(self.timer)
                else contextlib.nullcontext()
            )
            with timer_context:
                actor_training_handle = self.actor.run_training()
                training_metrics = actor_training_handle.wait()

            self.global_step += 1
            self.actor.set_global_step(self.global_step).wait()
            self.rollout.set_global_step(self.global_step).wait()
            self.env.set_global_step(self.global_step).wait()
            self.update_rollout_weights()

            time_metrics = self.timer.consume_durations()
            time_metrics = {f"time/{key}": value for key, value in time_metrics.items()}
            train_metrics = {
                f"train/{key}": value
                for key, value in self._aggregate_numeric_metrics(
                    training_metrics
                ).items()
            }
            gipo_metrics = self._drain_gipo_metrics()
            self.metric_logger.log(train_metrics, self.global_step)
            if gipo_metrics:
                self.metric_logger.log(gipo_metrics, self.global_step)
            self.metric_logger.log(time_metrics, self.global_step)

            logging_metrics = {**time_metrics, **train_metrics, **gipo_metrics}
            self.print_metrics_table_async(
                self.global_step - 1,
                self.max_steps,
                start_time,
                logging_metrics,
                start_step,
            )

            _, save_model, _ = check_progress(
                self.global_step,
                self.max_steps,
                self.cfg.runner.val_check_interval,
                self.cfg.runner.save_interval,
                1.0,
                run_time_exceeded=False,
            )
            if save_model:
                self._save_checkpoint()

        self.metric_logger.finish()
        self.env.stop().wait()
        self.rollout.stop().wait()
        self.actor.stop().wait()
        env_handle.wait()
        rollout_handle.wait()
        actor_recv_handle.wait()
