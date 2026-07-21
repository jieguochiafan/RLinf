# Copyright 2026 The RLinf Authors.
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

from unittest.mock import MagicMock

import torch

from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def test_merge_obs_batches_preserves_max_reset_time() -> None:
    obs_batches = [
        {
            "obs": {"states": torch.zeros(2, 3)},
            "final_obs": None,
            "reset_time_s": 4.5,
        },
        {
            "obs": {"states": torch.ones(1, 3)},
            "final_obs": None,
            "reset_time_s": 7.25,
        },
    ]

    merged = MultiStepRolloutWorker._merge_obs_batches(obs_batches)

    assert merged["obs"]["states"].shape == (3, 3)
    assert merged["reset_time_s"] == 7.25


def test_rollout_progress_logs_permanent_epoch_timing() -> None:
    worker = object.__new__(MultiStepRolloutWorker)
    worker._rank = 0
    worker.log_info = MagicMock()

    worker._log_rollout_progress(
        epoch_idx=1,
        total_epochs=8,
        epoch_time_s=36.7204,
        reset_time_s=1.7244,
    )

    worker.log_info.assert_called_once_with(
        "[rollout-progress] rollout_epoch=2/8 total=36.720s reset=1.724s"
    )
