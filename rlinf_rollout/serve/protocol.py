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

"""Control-plane messages exchanged between a client SDK and a running service.

These types sit *on top of* the frozen data-plane protocol in
:mod:`rlinf_rollout.api.v1`: a client submits :class:`~rlinf_rollout.api.v1.RolloutTask`
objects and :class:`~rlinf_rollout.api.v1.WeightUpdateRequest` objects, and reads
:class:`~rlinf_rollout.api.v1.Trajectory` / :class:`~rlinf_rollout.api.v1.RolloutResult`
payloads back. Everything here only describes the *service management* surface —
"who am I talking to", "was my task accepted", "did my weights land".

They reuse :class:`~rlinf_rollout.api.v1.SchemaBase`, so every message carries
``schema_version`` and gets the same strict ``from_dict`` decoding. This module
imports nothing heavier than the ``api.v1`` package, so a client can depend on it
without pulling in Ray, engines or simulators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from rlinf_rollout.api.v1 import Metadata, SchemaBase, SchemaError, WeightUpdateAck

__all__ = [
    "DEFAULT_SERVICE_NAME",
    "ServiceEndpoints",
    "ServiceState",
    "ServiceStatus",
    "TaskAck",
    "WeightPushResult",
    "WeightPushStatus",
    "control_group_name",
]

#: Service name used when neither the config nor the CLI provides one.
DEFAULT_SERVICE_NAME = "rollout"


def control_group_name(service_name: str = DEFAULT_SERVICE_NAME) -> str:
    """Return the worker-group name of a service's control plane.

    This is the single well-known name a client needs: everything else
    (channels, HTTP endpoints) is discovered through
    :attr:`ServiceStatus.endpoints`.

    Args:
        service_name: Logical service name, ``rollout`` by default.

    Returns:
        The control-plane worker-group name.
    """
    return f"{service_name}_control"


class ServiceState(str, Enum):
    """Lifecycle state of a rollout service.

    Attributes:
        STARTING: Workers are being launched / initialized.
        RUNNING: Serving; the resident loops are live.
        DRAINING: A stop was requested; in-flight work is finishing.
        STOPPED: The resident loops returned and workers were told to stop.
        FAILED: The service aborted; :attr:`ServiceStatus.error` explains why.
    """

    STARTING = "starting"
    RUNNING = "running"
    DRAINING = "draining"
    STOPPED = "stopped"
    FAILED = "failed"


class WeightPushStatus(str, Enum):
    """Outcome of a client-initiated weight push.

    Attributes:
        ACCEPTED: The request was handed to the rollout side, which will apply it
            in the background (``rollout.weight_sync.no_wait``). No per-receiver
            ack is available yet.
        APPLIED: Every receiver acknowledged the update.
        FAILED: At least one receiver failed; :attr:`WeightPushResult.error`
            explains why.
    """

    ACCEPTED = "accepted"
    APPLIED = "applied"
    FAILED = "failed"


@dataclass(kw_only=True)
class ServiceEndpoints(SchemaBase):
    """Names/addresses a client uses to reach a service's data plane.

    Channel names refer to vendored-scheduler channels; a client connects to them
    by name and never sees a Ray object.

    Args:
        control_group: Worker-group name of the control plane.
        output_channel: Channel carrying ``api/v1`` payloads
            (:class:`~rlinf_rollout.api.v1.Trajectory` for embodied,
            :class:`~rlinf_rollout.api.v1.RolloutResult` for LLM).
        num_output_partitions: Partition count the sink was built with, i.e.
            ``sink.num_shards``.
        rollout_group: Worker-group name of the rollout/engine workers.
        env_group: Worker-group name of the env workers (embodied only).
        http_endpoints: Optional OpenAI-compatible HTTP endpoints, keyed by role
            (e.g. ``{"online_router": "http://host:8081"}``).
    """

    control_group: str = ""
    output_channel: str = ""
    num_output_partitions: int = 1
    rollout_group: str = ""
    env_group: Optional[str] = None
    http_endpoints: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.num_output_partitions < 1:
            raise SchemaError(
                f"num_output_partitions must be positive, got "
                f"{self.num_output_partitions}."
            )


@dataclass(kw_only=True)
class ServiceStatus(SchemaBase):
    """Snapshot of a running service, published by its controller.

    Args:
        service_name: Logical service name.
        kind: Rollout chain, ``embodied`` or ``llm``.
        mode: ``collect`` or ``eval``.
        state: Lifecycle state.
        endpoints: How to reach the data plane.
        served_version: Weight version the rollout side currently serves, or
            ``-1`` before the first update (always ``-1`` in eval mode).
        num_tasks_pending: Tasks queued in the control plane.
        num_tasks_completed: Tasks the controller finished.
        num_outputs_published: ``api/v1`` payloads handed to the sink.
        num_weight_updates: Weight updates applied since start.
        uptime_seconds: Seconds since the controller started serving.
        metrics: Latest scalar metrics (eval metrics, queue sizes, ...).
        error: Failure reason when ``state`` is :attr:`ServiceState.FAILED`.
        metadata: Free-form annotations.
    """

    service_name: str = DEFAULT_SERVICE_NAME
    kind: str = ""
    mode: str = ""
    state: ServiceState = ServiceState.STARTING
    endpoints: ServiceEndpoints = field(default_factory=ServiceEndpoints)
    served_version: int = -1
    num_tasks_pending: int = 0
    num_tasks_completed: int = 0
    num_outputs_published: int = 0
    num_weight_updates: int = 0
    uptime_seconds: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    metadata: Metadata = field(default_factory=Metadata)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.state = ServiceState(self.state)
        if self.state is ServiceState.FAILED and not self.error:
            raise SchemaError("error must be set when state is 'failed'.")

    @property
    def is_serving(self) -> bool:
        """Whether the service currently accepts work."""
        return self.state is ServiceState.RUNNING


@dataclass(kw_only=True)
class TaskAck(SchemaBase):
    """Result of :meth:`rlinf_rollout.client.RolloutClient.submit_tasks`.

    Args:
        num_accepted: Tasks queued for execution.
        task_ids: Ids of the accepted tasks, in submission order.
        rejected: Task id -> reason for every task the service refused.
        num_pending: Queue depth after this submission.
        metadata: Free-form annotations.
    """

    num_accepted: int = 0
    task_ids: tuple[str, ...] = ()
    rejected: dict[str, str] = field(default_factory=dict)
    num_pending: int = 0
    metadata: Metadata = field(default_factory=Metadata)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.num_accepted != len(self.task_ids):
            raise SchemaError(
                f"num_accepted ({self.num_accepted}) disagrees with task_ids "
                f"({len(self.task_ids)})."
            )


@dataclass(kw_only=True)
class WeightPushResult(SchemaBase):
    """Aggregated outcome of one weight push, one entry per receiver.

    Args:
        version: Version the client asked for.
        status: Aggregate outcome, see :class:`WeightPushStatus`.
        served_version: Version the rollout side serves afterwards. For the
            receiver-driven collective transport this is authoritative and may
            exceed :attr:`version`.
        acks: Per-receiver acknowledgements; empty for
            :attr:`WeightPushStatus.ACCEPTED`.
        elapsed_seconds: Wall-clock duration measured by the controller.
        error: Failure reason when ``status`` is
            :attr:`WeightPushStatus.FAILED`.
        metadata: Free-form annotations.
    """

    version: int
    status: WeightPushStatus = WeightPushStatus.APPLIED
    served_version: int = -1
    acks: tuple[WeightUpdateAck, ...] = ()
    elapsed_seconds: float = 0.0
    error: Optional[str] = None
    metadata: Metadata = field(default_factory=Metadata)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.status = WeightPushStatus(self.status)
        if self.status is WeightPushStatus.FAILED and not self.error:
            raise SchemaError("error must be set when status is 'failed'.")

    def describe(self) -> dict[str, Any]:
        """Return a compact, log-friendly summary."""
        return {
            "version": self.version,
            "status": self.status.value,
            "served_version": self.served_version,
            "num_acks": len(self.acks),
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "error": self.error,
        }
