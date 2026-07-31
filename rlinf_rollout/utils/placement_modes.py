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

"""Placement / weight-sync mode enums, free of any Ray import.

Split out of :mod:`rlinf_rollout.utils.placement` (which pulls in the vendored
scheduler and therefore Ray) so that pure-logic consumers — notably
:mod:`rlinf_rollout.weight_sync.llm`, whose rank mapping is plain arithmetic — stay
importable and unit-testable without a cluster runtime.
:mod:`rlinf_rollout.utils.placement` re-exports these names, so existing imports
keep working.
"""

from enum import Enum, auto

__all__ = [
    "PlacementMode",
    "RolloutSyncMode",
    "placement_mode_to_rollout_sync_mode",
]


class PlacementMode(Enum):
    """
    Component placement mode represents the way to place components on GPUs.

    COLLOCATED: All components share the same set of GPUs.
    DISAGGREGATED: Each component has its own dedicated set of GPUs.
    HYBRID: Hybrid placement mode that allows components to run on any sets of GPUs.
    AUTO: Automatically choose the placement mode based on the component placement.
    """

    COLLOCATED = auto()
    DISAGGREGATED = auto()
    HYBRID = auto()
    AUTO = auto()


class RolloutSyncMode(Enum):
    """
    Rollout sync mode represents the way to synchronize rollout model weights.

    This mode is only used in reasoning scenarios.

    COLLOCATED: Used when rollout and actor components share the same set of GPUs.
        No inter-rank communication is required, and synchronization is typically
        conducted via CUDA IPC for optimal performance.

    DISAGGREGATED: Used when rollout and actor components use different sets of GPUs.
        Inter-rank communication is required, and synchronization is typically
        conducted via collective communication operations, such as NCCL.

    A key difference between modes is the rank mapping data structure:
    - COLLOCATED: rank mapping uses format `dict[int, tuple[int, int]]`
    - DISAGGREGATED: rank mapping uses format `dict[int, list[tuple[int, int]]]`"""

    COLLOCATED = auto()
    DISAGGREGATED = auto()


def placement_mode_to_rollout_sync_mode(
    placement_mode: PlacementMode,
) -> RolloutSyncMode:
    """Map placement mode to rollout sync mode in general cases.

    In special scenarios, the rollout sync mode is not the same as the placement mode. Thus, rollout sync mode should assigned separately, do not use this function in such scenarios.

    Args:
        placement_mode (PlacementMode): The placement mode.

    Returns:
        RolloutSyncMode: The corresponding rollout sync mode.
    """
    return (
        RolloutSyncMode.COLLOCATED
        if placement_mode == PlacementMode.COLLOCATED
        else RolloutSyncMode.DISAGGREGATED
    )
