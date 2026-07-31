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

"""Frozen ``v1`` protocol of the standalone rollout system.

Three interfaces make the rollout system usable from any training framework:

1. Weights in — :class:`WeightReceiver` consumes :class:`WeightUpdateRequest`
   and answers with :class:`WeightUpdateAck`.
2. Data out — :class:`TrajectorySink` accepts :class:`Trajectory` (embodied) or
   :class:`RolloutResult` (LLM), partitioned according to :class:`ConsumerSpec`.
3. Work in — :class:`TaskSource` yields :class:`RolloutTask`.

Every message inherits :class:`SchemaBase` and therefore carries
``schema_version == SCHEMA_VERSION``. The field set of this version is frozen
and locked by ``rlinf_rollout/tests/test_api_v1_schema.py``.
"""

from .common import SCHEMA_VERSION, Metadata, SchemaBase, SchemaError
from .task import (
    EpisodeSpec,
    PromptSpec,
    RolloutMode,
    RolloutTask,
    SamplingParams,
    TaskKind,
    TaskSource,
)
from .trajectory import (
    ActionChunkResult,
    ConsumerSpec,
    FinishReason,
    PartitionAxis,
    PolicyInputs,
    RolloutResult,
    TensorLayout,
    Trajectory,
    TrajectorySink,
)
from .weight import (
    SourceTopology,
    TensorSpec,
    WeightReceiver,
    WeightSyncMode,
    WeightTransport,
    WeightUpdateAck,
    WeightUpdateRequest,
    WeightUpdateStatus,
)

__all__ = [
    "SCHEMA_VERSION",
    # common
    "Metadata",
    "SchemaBase",
    "SchemaError",
    # weight
    "SourceTopology",
    "TensorSpec",
    "WeightReceiver",
    "WeightSyncMode",
    "WeightTransport",
    "WeightUpdateAck",
    "WeightUpdateRequest",
    "WeightUpdateStatus",
    # trajectory / outputs
    "ActionChunkResult",
    "ConsumerSpec",
    "FinishReason",
    "PartitionAxis",
    "PolicyInputs",
    "RolloutResult",
    "TensorLayout",
    "Trajectory",
    "TrajectorySink",
    # tasks
    "EpisodeSpec",
    "PromptSpec",
    "RolloutMode",
    "RolloutTask",
    "SamplingParams",
    "TaskKind",
    "TaskSource",
]
