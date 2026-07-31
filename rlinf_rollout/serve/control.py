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

"""The service's control plane state: a mailbox between clients and controller.

A rollout service is a long-running process tree. Clients live in *other*
processes (usually another framework's trainer), so the control plane is a tiny
CPU-only worker that both sides reach by name:

* a client resolves it with the well-known group name
  :func:`~rlinf_rollout.serve.protocol.control_group_name` and calls its methods
  (submit tasks, push weights, read status, request a stop);
* the service's :class:`~rlinf_rollout.serve.controller.RolloutController` polls
  it from the driver process, executes the work and publishes results back.

This module holds the logic — a plain Python object with no Ray, torch-CUDA or
accelerator dependency, so it is unit-testable on its own. The remotely
addressable shell lives in :mod:`rlinf_rollout.serve.control_worker`.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from typing import Any, Optional, Sequence

from rlinf_rollout.api.v1 import RolloutTask, WeightUpdateRequest
from rlinf_rollout.serve.protocol import (
    DEFAULT_SERVICE_NAME,
    ServiceStatus,
    TaskAck,
    WeightPushResult,
)

__all__ = ["MAX_RETAINED_WEIGHT_RESULTS", "ControlPlaneState"]

#: How many weight-push results are kept around for clients to poll.
MAX_RETAINED_WEIGHT_RESULTS = 64


class ControlPlaneState:
    """Queues and last-known state of one rollout service.

    Args:
        service_name: Logical service name, echoed in the initial status.
        accepted_task_kinds: Task kinds this service can run, as
            :class:`~rlinf_rollout.api.v1.TaskKind` members or their string
            values. Tasks of any other kind are rejected with a reason instead of
            being silently queued. ``None`` accepts everything.
        max_pending_tasks: Cap on the task queue; further submissions are
            rejected with a back-pressure reason. ``0`` means unbounded.
    """

    def __init__(
        self,
        service_name: str = DEFAULT_SERVICE_NAME,
        *,
        accepted_task_kinds: Optional[Sequence[Any]] = None,
        max_pending_tasks: int = 0,
    ) -> None:
        self._service_name = service_name
        self._accepted_task_kinds = (
            None
            if accepted_task_kinds is None
            else {str(getattr(kind, "value", kind)) for kind in accepted_task_kinds}
        )
        self._max_pending_tasks = int(max_pending_tasks)

        self._tasks: deque[RolloutTask] = deque()
        self._weight_requests: deque[WeightUpdateRequest] = deque()
        self._weight_results: OrderedDict[int, WeightPushResult] = OrderedDict()
        self._status = ServiceStatus(service_name=service_name)
        self._stop_requested = False
        self._num_tasks_submitted = 0
        self._num_tasks_completed = 0
        self._task_errors: dict[str, str] = {}

    @property
    def service_name(self) -> str:
        """Logical name of the service this control plane belongs to."""
        return self._service_name

    @property
    def accepted_task_kinds(self) -> Optional[set[str]]:
        """Task-kind values this service serves, or ``None`` for "anything"."""
        return (
            None
            if self._accepted_task_kinds is None
            else set(self._accepted_task_kinds)
        )

    # --------------------------------------------------------------- tasks in

    def submit_tasks(self, tasks: Sequence[RolloutTask]) -> TaskAck:
        """Queue ``tasks`` for the controller, rejecting the ones we cannot run.

        Args:
            tasks: Tasks to enqueue. Queue order is submission order;
                :attr:`~rlinf_rollout.api.v1.RolloutTask.priority` is applied by
                :meth:`next_tasks`.

        Returns:
            A :class:`~rlinf_rollout.serve.protocol.TaskAck` listing accepted ids
            and per-task rejection reasons.
        """
        accepted: list[str] = []
        rejected: dict[str, str] = {}
        for task in tasks:
            kind = task.kind.value
            if self._stop_requested:
                rejected[task.task_id] = "service is shutting down"
            elif self._accepted_task_kinds is not None and (
                kind not in self._accepted_task_kinds
            ):
                rejected[task.task_id] = (
                    f"task kind {kind!r} is not served by {self._service_name!r} "
                    f"(accepted: {sorted(self._accepted_task_kinds)})"
                )
            elif 0 < self._max_pending_tasks <= len(self._tasks):
                rejected[task.task_id] = (
                    f"task queue is full ({self._max_pending_tasks} pending)"
                )
            else:
                self._tasks.append(task)
                accepted.append(task.task_id)

        self._num_tasks_submitted += len(accepted)
        return TaskAck(
            num_accepted=len(accepted),
            task_ids=tuple(accepted),
            rejected=rejected,
            num_pending=len(self._tasks),
        )

    def next_tasks(self, max_num_tasks: int) -> list[RolloutTask]:
        """Pop up to ``max_num_tasks`` queued tasks, highest priority first.

        Args:
            max_num_tasks: Upper bound on the returned batch size.

        Returns:
            The dequeued tasks, possibly empty. Ties keep submission order.
        """
        if max_num_tasks <= 0 or not self._tasks:
            return []
        ordered = sorted(self._tasks, key=lambda task: -task.priority)
        taken = ordered[:max_num_tasks]
        self._tasks = deque(ordered[max_num_tasks:])
        return taken

    @property
    def num_pending_tasks(self) -> int:
        """Number of queued, not-yet-dispatched tasks."""
        return len(self._tasks)

    @property
    def num_tasks_submitted(self) -> int:
        """Number of tasks accepted since start."""
        return self._num_tasks_submitted

    @property
    def num_tasks_completed(self) -> int:
        """Number of tasks the controller reported as finished."""
        return self._num_tasks_completed

    @property
    def task_errors(self) -> dict[str, str]:
        """Task id -> failure reason for every task that finished with an error."""
        return dict(self._task_errors)

    def report_task_done(self, task_id: str, error: Optional[str] = None) -> None:
        """Record completion of a dispatched task.

        Args:
            task_id: Id of the finished task.
            error: Failure reason, or ``None`` on success.
        """
        self._num_tasks_completed += 1
        if error:
            self._task_errors[task_id] = error

    # ------------------------------------------------------------- weights in

    def request_weight_update(self, request: WeightUpdateRequest) -> None:
        """Queue a weight update for the controller to execute.

        Args:
            request: The client's update description.
        """
        self._weight_requests.append(request)

    def next_weight_request(self) -> Optional[WeightUpdateRequest]:
        """Pop the oldest queued weight update, or ``None`` when idle."""
        if not self._weight_requests:
            return None
        return self._weight_requests.popleft()

    @property
    def num_pending_weight_requests(self) -> int:
        """Number of weight updates waiting for the controller."""
        return len(self._weight_requests)

    def publish_weight_result(self, result: WeightPushResult) -> None:
        """Store the outcome of a weight update for the client to poll.

        Only the most recent :data:`MAX_RETAINED_WEIGHT_RESULTS` versions are
        kept; a client that never polls cannot grow the service's memory.

        Args:
            result: Aggregated per-receiver outcome.
        """
        self._weight_results[result.version] = result
        self._weight_results.move_to_end(result.version)
        while len(self._weight_results) > MAX_RETAINED_WEIGHT_RESULTS:
            self._weight_results.popitem(last=False)

    def get_weight_result(self, version: int) -> Optional[WeightPushResult]:
        """Return the stored result for ``version``, or ``None`` if not yet known."""
        return self._weight_results.get(version)

    # ----------------------------------------------------------------- status

    def publish_status(self, status: ServiceStatus) -> None:
        """Replace the published status snapshot.

        Args:
            status: Snapshot produced by the controller.
        """
        self._status = status

    def get_status(self) -> ServiceStatus:
        """Return the latest published status snapshot."""
        return self._status

    # ------------------------------------------------------------------- stop

    def request_stop(self) -> None:
        """Ask the service to drain and shut down."""
        self._stop_requested = True

    @property
    def stop_requested(self) -> bool:
        """Whether a client asked the service to stop."""
        return self._stop_requested

    def describe(self) -> dict[str, Any]:
        """Return a compact snapshot of every queue, for logs and diagnostics."""
        return {
            "service_name": self._service_name,
            "num_pending_tasks": self.num_pending_tasks,
            "num_tasks_submitted": self._num_tasks_submitted,
            "num_tasks_completed": self._num_tasks_completed,
            "num_pending_weight_requests": self.num_pending_weight_requests,
            "num_weight_results": len(self._weight_results),
            "stop_requested": self._stop_requested,
            "state": self._status.state.value,
        }
