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

"""Weight-update protocol: interface 1 of the standalone rollout system.

The trainer (any framework) describes *where* new weights come from and *how*
they are encoded; the rollout side applies them and acknowledges with the
version it now serves. No trainer-specific group name or rank layout is baked
into rollout code: the sender topology travels inside
:class:`WeightUpdateRequest`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .common import Metadata, SchemaBase, SchemaError


class WeightSyncMode(str, Enum):
    """Encoding of the weight payload.

    Attributes:
        BUCKET: Full state dict, flattened into fixed-size buckets.
        PATCH: Only parameters that changed since the receiver's snapshot,
            optionally delta-encoded and compressed.
    """

    BUCKET = "bucket"
    PATCH = "patch"


class WeightTransport(str, Enum):
    """How bytes travel from trainer to rollout.

    Attributes:
        COLLECTIVE: Accelerator collective (e.g. NCCL broadcast/send) driven by
            the vendored scheduler; sender layout is given by
            :class:`SourceTopology`.
        CHECKPOINT: Rollout loads a checkpoint from a shared path.
        HTTP_PUSH: Trainer pushes tensors over HTTP to the rollout server.
    """

    COLLECTIVE = "collective"
    CHECKPOINT = "checkpoint"
    HTTP_PUSH = "http_push"


class WeightUpdateStatus(str, Enum):
    """Outcome of a weight update.

    Attributes:
        APPLIED: New weights are live on the rollout side.
        SKIPPED: Request was ignored (e.g. version not newer than current).
        FAILED: Update did not complete; ``error`` carries the reason.
    """

    APPLIED = "applied"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(kw_only=True)
class TensorSpec(SchemaBase):
    """Description of one tensor in the payload.

    Args:
        name: Parameter name in the policy state dict.
        shape: Logical (unsharded) shape.
        dtype: Torch dtype name without the ``torch.`` prefix, e.g. ``bfloat16``.
        shard_dim: Dimension the sender shards this tensor along, or ``None``
            when the tensor is sent whole.
        num_bytes: Payload size in bytes, after compression when applicable.
    """

    name: str
    shape: tuple[int, ...] = ()
    dtype: str = ""
    shard_dim: Optional[int] = None
    num_bytes: Optional[int] = None


@dataclass(kw_only=True)
class SourceTopology(SchemaBase):
    """Sender-side layout, described by the trainer instead of assumed.

    Args:
        group_name: Sender worker-group name in the sender's own namespace.
        src_ranks: Sender ranks participating in the transfer. For a broadcast
            this is the single root rank.
        world_size: Total sender ranks (``0`` when not applicable).
        parallel_sizes: Sender model-parallel sizes, e.g.
            ``{"tp": 4, "pp": 2, "dp": 8}``. Empty for single-rank senders.
        rank_map: Optional mapping from receiver rank (as string) to the sender
            rank it must pair with, for point-to-point resharding.
        endpoint: Transport-specific address (e.g. HTTP base URL); unused by
            collective transports.
    """

    group_name: str = ""
    src_ranks: tuple[int, ...] = (0,)
    world_size: int = 0
    parallel_sizes: dict[str, int] = field(default_factory=dict)
    rank_map: dict[str, int] = field(default_factory=dict)
    endpoint: Optional[str] = None


@dataclass(kw_only=True)
class WeightUpdateRequest(SchemaBase):
    """Trainer -> rollout request to install a new policy version.

    Args:
        version: Monotonically increasing weight version. Flows back out with
            every rollout output so consumers can reason about staleness.
        mode: Payload encoding, see :class:`WeightSyncMode`.
        transport: Byte transport, see :class:`WeightTransport`.
        source: Sender topology description.
        tensors: Optional tensor manifest. May be empty when the receiver can
            derive it (e.g. bucket mode negotiated at handshake time).
        checkpoint_path: Required for :attr:`WeightTransport.CHECKPOINT`.
        compression: Compressor name applied to the payload (``"none"`` when
            uncompressed).
        bucket_size_bytes: Bucket size for :attr:`WeightSyncMode.BUCKET`.
        blocking: Whether the trainer waits for the ack before continuing.
        metadata: Free-form annotations.
    """

    version: int
    mode: WeightSyncMode = WeightSyncMode.BUCKET
    transport: WeightTransport = WeightTransport.COLLECTIVE
    source: SourceTopology = field(default_factory=SourceTopology)
    tensors: tuple[TensorSpec, ...] = ()
    checkpoint_path: Optional[str] = None
    compression: str = "none"
    bucket_size_bytes: Optional[int] = None
    blocking: bool = True
    metadata: Metadata = field(default_factory=Metadata)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.mode = WeightSyncMode(self.mode)
        self.transport = WeightTransport(self.transport)
        if self.version < 0:
            raise SchemaError(f"version must be non-negative, got {self.version}")
        if self.transport is WeightTransport.CHECKPOINT and not self.checkpoint_path:
            raise SchemaError(
                "checkpoint_path is required when transport is 'checkpoint'."
            )
        if self.transport is WeightTransport.COLLECTIVE and not self.source.src_ranks:
            raise SchemaError(
                "source.src_ranks must not be empty for collective transport."
            )


@dataclass(kw_only=True)
class WeightUpdateAck(SchemaBase):
    """Rollout -> trainer acknowledgement of a weight update.

    Args:
        version: ``version`` of the request being acknowledged.
        status: Outcome, see :class:`WeightUpdateStatus`.
        served_version: Version the receiver serves after handling the request.
            Equals ``version`` on success, the previous version on skip/failure.
        receiver_id: Identity of the acknowledging receiver (e.g.
            ``"rollout:3"``).
        num_tensors_applied: Number of tensors written into the live model.
        num_bytes_received: Bytes received for this update.
        elapsed_seconds: Wall-clock duration of the update.
        error: Failure reason when ``status`` is
            :attr:`WeightUpdateStatus.FAILED`.
        metadata: Free-form annotations.
    """

    version: int
    status: WeightUpdateStatus = WeightUpdateStatus.APPLIED
    served_version: int = -1
    receiver_id: str = ""
    num_tensors_applied: int = 0
    num_bytes_received: int = 0
    elapsed_seconds: float = 0.0
    error: Optional[str] = None
    metadata: Metadata = field(default_factory=Metadata)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.status = WeightUpdateStatus(self.status)
        if self.status is WeightUpdateStatus.FAILED and not self.error:
            raise SchemaError("error must be set when status is 'failed'.")


class WeightReceiver(ABC):
    """Rollout-side endpoint that installs new policy weights.

    Backends implement this over different transports (collective broadcast via
    the vendored weight syncer, checkpoint polling, HTTP push). Implementations
    must be idempotent for repeated versions and must not reference any
    trainer-specific worker-group name: everything they need arrives in
    :class:`WeightUpdateRequest`.
    """

    @property
    @abstractmethod
    def served_version(self) -> int:
        """Weight version currently served, or ``-1`` before the first update."""

    @abstractmethod
    async def recv(self, request: WeightUpdateRequest) -> WeightUpdateAck:
        """Receive and apply one weight update.

        Args:
            request: Update description produced by the trainer.

        Returns:
            The acknowledgement, including the version served afterwards.
        """

    async def prepare(self, request: WeightUpdateRequest) -> None:
        """Perform one-time handshake/negotiation before the first :meth:`recv`.

        The default implementation is a no-op.

        Args:
            request: The first update request, used to negotiate topology.
        """
        del request

    async def close(self) -> None:
        """Release transport resources. The default implementation is a no-op."""

    def describe(self) -> dict[str, Any]:
        """Return backend diagnostics (transport, served version, ...)."""
        return {
            "schema_version": SchemaBase.SCHEMA_VERSION,
            "receiver": type(self).__name__,
            "served_version": self.served_version,
        }
