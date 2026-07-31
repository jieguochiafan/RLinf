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

"""Training-side SDK for a running rollout service.

Typical use from any training framework::

    from rlinf_rollout.client import RolloutClient

    client = RolloutClient("rollout")
    client.wait_until_ready()

    for step in range(max_steps):
        client.push_checkpoint("/shared/ckpt/step_%d.pt" % step, version=step)
        for payload in client.get_trajectories(max_items=8, timeout=60):
            train_on(payload)  # api/v1 Trajectory / RolloutResult

Nothing in this package exposes a Ray object; the service is addressed by name.
"""

from rlinf_rollout.client.rollout_client import (
    RolloutClient,
    RolloutClientError,
    make_prompt_tasks,
)
from rlinf_rollout.client.weight_sender import (
    CheckpointWeightPublisher,
    describe_collective_sender,
)

__all__ = [
    "CheckpointWeightPublisher",
    "RolloutClient",
    "RolloutClientError",
    "describe_collective_sender",
    "make_prompt_tasks",
]
