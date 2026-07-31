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

"""Remotely addressable shell around :class:`ControlPlaneState`.

Launched by :class:`~rlinf_rollout.serve.service.RolloutService` as a single-rank
CPU worker named :func:`~rlinf_rollout.serve.protocol.control_group_name`. Both
the client SDK and the controller talk to it through the vendored scheduler's
worker-group API, so neither side ever handles a Ray object.

Every method is a one-line forward to :class:`ControlPlaneState`; the logic and
its tests live there, free of Ray.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from rlinf_rollout.api.v1 import RolloutTask, WeightUpdateRequest
from rlinf_rollout.scheduler import Worker
from rlinf_rollout.serve.control import ControlPlaneState
from rlinf_rollout.serve.protocol import (
    DEFAULT_SERVICE_NAME,
    ServiceStatus,
    TaskAck,
    WeightPushResult,
)

__all__ = ["ControlPlaneWorker"]


class ControlPlaneWorker(Worker):
    """Single-rank CPU worker exposing a service's control plane.

    Args:
        service_name: Logical service name.
        accepted_task_kinds: Task-kind values this service serves, or ``None``.
        max_pending_tasks: Cap on the task queue (``0`` = unbounded).
    """

    def __init__(
        self,
        service_name: str = DEFAULT_SERVICE_NAME,
        accepted_task_kinds: Optional[Sequence[Any]] = None,
        max_pending_tasks: int = 0,
    ) -> None:
        Worker.__init__(self)
        self._state = ControlPlaneState(
            service_name,
            accepted_task_kinds=accepted_task_kinds,
            max_pending_tasks=max_pending_tasks,
        )

    # --------------------------------------------------------------- tasks in

    def submit_tasks(self, tasks: Sequence[RolloutTask]) -> TaskAck:
        """Queue ``tasks``; see :meth:`ControlPlaneState.submit_tasks`."""
        return self._state.submit_tasks(tasks)

    def next_tasks(self, max_num_tasks: int) -> list[RolloutTask]:
        """Pop tasks; see :meth:`ControlPlaneState.next_tasks`."""
        return self._state.next_tasks(max_num_tasks)

    def num_pending_tasks(self) -> int:
        """Number of queued tasks."""
        return self._state.num_pending_tasks

    def report_task_done(self, task_id: str, error: Optional[str] = None) -> None:
        """Record completion of a task."""
        self._state.report_task_done(task_id, error)

    # ------------------------------------------------------------- weights in

    def request_weight_update(self, request: WeightUpdateRequest) -> None:
        """Queue a weight update for the controller."""
        self._state.request_weight_update(request)

    def next_weight_request(self) -> Optional[WeightUpdateRequest]:
        """Pop the oldest queued weight update."""
        return self._state.next_weight_request()

    def publish_weight_result(self, result: WeightPushResult) -> None:
        """Store the outcome of a weight update."""
        self._state.publish_weight_result(result)

    def get_weight_result(self, version: int) -> Optional[WeightPushResult]:
        """Return the stored result for ``version``, if any."""
        return self._state.get_weight_result(version)

    # ----------------------------------------------------------------- status

    def publish_status(self, status: ServiceStatus) -> None:
        """Replace the published status snapshot."""
        self._state.publish_status(status)

    def get_status(self) -> ServiceStatus:
        """Return the latest published status snapshot."""
        return self._state.get_status()

    # ------------------------------------------------------------------- stop

    def request_stop(self) -> None:
        """Ask the service to drain and shut down."""
        self._state.request_stop()

    def stop_requested(self) -> bool:
        """Whether a stop was requested."""
        return self._state.stop_requested

    def describe_state(self) -> dict[str, Any]:
        """Return a compact snapshot of every queue."""
        return self._state.describe()
