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

"""Optional rollout post-processing (bootstrap shaping, trainer-side hooks)."""

from rlinf_rollout.postprocess.base import (
    TrajectoryPostprocessor,
    load_trajectory_postprocessor,
)
from rlinf_rollout.postprocess.bootstrap import (
    BootstrapRewardShaper,
    estimate_bootstrap_values,
)

__all__ = [
    "BootstrapRewardShaper",
    "TrajectoryPostprocessor",
    "estimate_bootstrap_values",
    "load_trajectory_postprocessor",
]
