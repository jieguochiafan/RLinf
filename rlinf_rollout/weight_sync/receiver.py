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

"""Collective (NCCL) :class:`~rlinf_rollout.api.v1.WeightReceiver` backend.

The training repo's ``MultiStepRolloutWorker.setup_weight_sync`` hardcoded the
sender's group name (``cfg.actor.group_name``), its root rank (``0``) and derived
its world size from ``HybridComponentPlacement.get_world_size("actor")``. Here that
layout arrives in :class:`~rlinf_rollout.api.v1.SourceTopology`, so the rollout side
holds no trainer-specific knowledge.

Pull semantics: the collective transport is receiver-driven — the rollout worker
blocks on a broadcast and the authoritative version arrives inside the payload.
``request.version`` is therefore only the *minimum expected* version;
:attr:`WeightUpdateAck.served_version` is authoritative.
"""

import gc
import time
from typing import TYPE_CHECKING, Any, Optional, Sequence

import torch

from rlinf_rollout.api.v1 import (
    WeightReceiver,
    WeightTransport,
    WeightUpdateAck,
    WeightUpdateRequest,
    WeightUpdateStatus,
)

if TYPE_CHECKING:
    from rlinf_rollout.weight_sync.base import WeightSyncer

__all__ = ["CollectiveWeightReceiver"]


class CollectiveWeightReceiver(WeightReceiver):
    """Applies weights broadcast over an accelerator collective.

    Args:
        worker: The vendored-scheduler ``Worker`` that owns the collective
            endpoints (used for ``broadcast`` / ``send``).
        syncer: Payload codec, see :class:`rlinf_rollout.weight_sync.WeightSyncer`.
        model: Live policy the weights are written into.
        receiver_group_name: Worker-group name of the receiving (rollout) group.
        receiver_ranks: Ranks of the receiving group taking part in the broadcast.
        is_handshake_sender: Whether this rank sends handshake replies back to the
            sender group (exactly one receiver rank should do so).
        receiver_id: Human-readable identity reported in the ack.
        empty_cache: Whether to drop cached accelerator memory after each update.
    """

    def __init__(
        self,
        *,
        worker: Any,
        syncer: "WeightSyncer",
        model: torch.nn.Module,
        receiver_group_name: str,
        receiver_ranks: Sequence[int],
        is_handshake_sender: bool,
        receiver_id: str = "",
        empty_cache: bool = True,
    ) -> None:
        self._worker = worker
        self._syncer = syncer
        self._model = model
        self._receiver_group_name = receiver_group_name
        self._receiver_ranks = list(receiver_ranks)
        self._is_handshake_sender = is_handshake_sender
        self._receiver_id = receiver_id or receiver_group_name
        self._empty_cache = empty_cache
        self._served_version = -1

    @property
    def served_version(self) -> int:
        """Weight version currently served, or ``-1`` before the first update."""
        return self._served_version

    def _recv_fn(self, request: WeightUpdateRequest):
        src_group = request.source.group_name
        src_rank = int(request.source.src_ranks[0])
        options = self._syncer.comm_options

        async def recv() -> Any:
            return await self._worker.broadcast(
                None,
                groups=[
                    (src_group, src_rank),
                    (self._receiver_group_name, self._receiver_ranks),
                ],
                src=(src_group, src_rank),
                async_op=True,
                options=options,
            ).async_wait()

        return recv

    def _send_fn(self, request: WeightUpdateRequest):
        src_group = request.source.group_name
        options = self._syncer.comm_options
        sender_world_size = int(request.source.world_size) or len(
            request.source.src_ranks
        )

        async def send(data: Any) -> None:
            if not self._is_handshake_sender:
                return
            for dst_rank in range(sender_world_size):
                await self._worker.send(
                    data,
                    dst_group_name=src_group,
                    dst_rank=dst_rank,
                    async_op=True,
                    options=options,
                ).async_wait()

        return send

    @staticmethod
    def _check_transport(request: WeightUpdateRequest) -> None:
        if request.transport is not WeightTransport.COLLECTIVE:
            raise NotImplementedError(
                f"{CollectiveWeightReceiver.__name__} only supports "
                f"{WeightTransport.COLLECTIVE.value} transport, got "
                f"{request.transport.value}."
            )
        if not request.source.group_name:
            raise ValueError(
                "request.source.group_name must name the sending worker group."
            )

    async def prepare(self, request: WeightUpdateRequest) -> None:
        """Run the one-time receiver handshake with the sender group."""
        self._check_transport(request)
        if self._syncer.receiver_initialized():
            return
        await self._syncer.init_receiver(
            state_dict=self._model.state_dict(),
            recv=self._recv_fn(request),
            send=self._send_fn(request),
        )

    async def recv(self, request: WeightUpdateRequest) -> WeightUpdateAck:
        """Receive one broadcast payload and write it into the live model."""
        started = time.perf_counter()
        try:
            self._check_transport(request)
            await self.prepare(request)
            applied_version = await self._syncer.apply(
                self._model, self._recv_fn(request)
            )
        except Exception as exc:  # noqa: BLE001 - reported through the ack
            return WeightUpdateAck(
                version=request.version,
                status=WeightUpdateStatus.FAILED,
                served_version=self._served_version,
                receiver_id=self._receiver_id,
                elapsed_seconds=time.perf_counter() - started,
                error=f"{type(exc).__name__}: {exc}",
            )

        self._served_version = int(applied_version)
        if hasattr(self._model, "set_global_step"):
            self._model.set_global_step(self._served_version)

        gc.collect()
        if self._empty_cache:
            platform = getattr(self._worker, "torch_platform", None)
            if platform is not None:
                platform.empty_cache()

        return WeightUpdateAck(
            version=request.version,
            status=WeightUpdateStatus.APPLIED,
            served_version=self._served_version,
            receiver_id=self._receiver_id,
            elapsed_seconds=time.perf_counter() - started,
        )

    def describe(self) -> dict[str, Any]:
        """Return backend diagnostics."""
        info = super().describe()
        info.update(
            {
                "transport": WeightTransport.COLLECTIVE.value,
                "syncer": type(self._syncer).__name__,
                "receiver_group_name": self._receiver_group_name,
                "receiver_ranks": tuple(self._receiver_ranks),
                "is_handshake_sender": self._is_handshake_sender,
            }
        )
        return info

    @property
    def syncer(self) -> "WeightSyncer":
        """The payload codec in use."""
        return self._syncer

    @property
    def comm_options(self) -> Optional[Any]:
        """Collective options the syncer requires, forwarded by the worker."""
        return self._syncer.comm_options
