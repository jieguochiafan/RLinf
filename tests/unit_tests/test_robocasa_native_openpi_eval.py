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

import numpy as np

from toolkits.standalone_eval_scripts.robocasa.native_openpi_eval import (
    extract_policy_observation,
)


def test_extract_policy_observation_uses_canonical_16d_order() -> None:
    observation = {
        "robot0_agentview_left_image": np.zeros((2, 3, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.ones((2, 3, 3), dtype=np.uint8),
        "robot0_base_to_eef_pos": np.arange(0, 3, dtype=np.float32),
        "robot0_base_to_eef_quat": np.arange(3, 7, dtype=np.float32),
        "robot0_base_pos": np.arange(7, 10, dtype=np.float32),
        "robot0_base_quat": np.arange(10, 14, dtype=np.float32),
        "robot0_gripper_qpos": np.arange(14, 16, dtype=np.float32),
    }

    result = extract_policy_observation(observation, "close the drawer")

    np.testing.assert_array_equal(
        result["states"].numpy()[0], np.arange(16, dtype=np.float32)
    )
    assert result["main_images"].shape == (1, 2, 3, 3)
    assert result["wrist_images"].shape == (1, 2, 3, 3)
    assert result["extra_view_images"] is None
    assert result["task_descriptions"] == ["close the drawer"]
