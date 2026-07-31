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

"""Configuration for the standalone rollout system."""

from rlinf_rollout.config.models import (
    EMBODIED_MODEL,
    SupportedModel,
    torch_dtype_from_precision,
)
from rlinf_rollout.config.rollout import (
    DEFAULT_ROLLOUT_CONFIG,
    FORBIDDEN_SECTIONS,
    RolloutConfig,
    RolloutConfigError,
    RolloutMode,
    build_rollout_config,
    validate_rollout_config,
)

__all__ = [
    "DEFAULT_ROLLOUT_CONFIG",
    "EMBODIED_MODEL",
    "FORBIDDEN_SECTIONS",
    "RolloutConfig",
    "RolloutConfigError",
    "RolloutMode",
    "SupportedModel",
    "build_rollout_config",
    "torch_dtype_from_precision",
    "validate_rollout_config",
]
