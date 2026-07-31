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

"""Trainer-side helpers for the weight-push half of the client SDK.

Two transports, two very different couplings:

* **Checkpoint** — :class:`CheckpointWeightPublisher` writes a state dict to a
  shared directory and hands back the matching
  :class:`~rlinf_rollout.api.v1.WeightUpdateRequest`. No collective, no
  co-scheduling; the trainer can even be a different process tree.

* **Collective** — the fast path. It cannot be wrapped here, because the
  broadcast must be issued *by the trainer's own worker group* over the trainer's
  own collective: only that process holds the sharded state dict and the group
  handle. :func:`describe_collective_sender` documents the exact contract, and the
  ``sender`` argument of :meth:`~rlinf_rollout.client.RolloutClient.push_weights`
  is where the trainer plugs its coroutine in. A sketch::

      from rlinf_rollout.weight_sync import WeightSyncer

      syncer = WeightSyncer.create(my_weight_sync_cfg)  # same cfg as the service


      async def sender(request):
          # runs inside the trainer's worker; ``self`` is that worker
          async def send(data):
              await self.broadcast(
                  data,
                  groups=[
                      (request.source.group_name, request.source.src_ranks[0]),
                      (rollout_group_name, rollout_ranks),
                  ],
                  src=(request.source.group_name, request.source.src_ranks[0]),
                  async_op=True,
                  options=syncer.comm_options,
              ).async_wait()

          await syncer.sync(self.model.state_dict(), send, request.version)


      client.push_weights(request, sender=sender)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Union

from rlinf_rollout.api.v1 import (
    SourceTopology,
    WeightTransport,
    WeightUpdateRequest,
)

__all__ = ["CheckpointWeightPublisher", "describe_collective_sender"]


class CheckpointWeightPublisher:
    """Writes versioned state dicts and builds checkpoint weight requests.

    Args:
        directory: Directory the rollout workers can read. Created on demand.
        prefix: File-name prefix; files are ``<prefix>_v<version>.pt``.
        keep_last: How many published checkpoints to keep on disk; ``0`` keeps
            everything.
        save_fn: ``(state_dict, path) -> None`` writer. Defaults to
            :func:`torch.save`, imported lazily so this module stays light.
        group_name: Sender identity recorded in the request's topology.
    """

    def __init__(
        self,
        directory: Union[str, Path],
        *,
        prefix: str = "weights",
        keep_last: int = 2,
        save_fn: Optional[Callable[[Any, Path], None]] = None,
        group_name: str = "client",
    ) -> None:
        self.directory = Path(directory)
        self.prefix = prefix
        self.keep_last = int(keep_last)
        self._save_fn = save_fn
        self.group_name = group_name
        self._published: list[Path] = []

    def path_for(self, version: int) -> Path:
        """Return the checkpoint path used for ``version``."""
        return self.directory / f"{self.prefix}_v{int(version)}.pt"

    def publish(self, state_dict: Any, version: int) -> WeightUpdateRequest:
        """Write ``state_dict`` and return the request describing it.

        Args:
            state_dict: Mapping of parameter name to tensor.
            version: Monotonically increasing weight version.

        Returns:
            A ``CHECKPOINT``-transport
            :class:`~rlinf_rollout.api.v1.WeightUpdateRequest` for
            :meth:`~rlinf_rollout.client.RolloutClient.push_weights`.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path_for(version)
        save_fn = self._save_fn
        if save_fn is None:
            import torch

            def save_fn(payload: Any, target: Path) -> None:
                torch.save(payload, target)

        save_fn(state_dict, path)
        self._published.append(path)
        self._prune()
        return WeightUpdateRequest(
            version=int(version),
            transport=WeightTransport.CHECKPOINT,
            checkpoint_path=str(path),
            source=SourceTopology(group_name=self.group_name),
        )

    def _prune(self) -> None:
        """Delete checkpoints beyond :attr:`keep_last`."""
        if self.keep_last <= 0:
            return
        while len(self._published) > self.keep_last:
            stale = self._published.pop(0)
            try:
                stale.unlink(missing_ok=True)
            except OSError:
                continue

    @property
    def published_paths(self) -> tuple[Path, ...]:
        """Checkpoints still on disk, oldest first."""
        return tuple(self._published)


def describe_collective_sender() -> dict[str, str]:
    """Return the contract a trainer's collective sender must satisfy.

    Returned as data (rather than prose only) so a trainer integration can assert
    against it and so the requirements show up in ``--help``-style tooling.

    Returns:
        Mapping of requirement name to explanation.
    """
    return {
        "group_name": (
            "must equal rollout.weight_sync.source.group_name in the service "
            "config; the receivers open their collective against that name."
        ),
        "src_rank": (
            "the broadcast root, i.e. request.source.src_ranks[0]; must match "
            "rollout.weight_sync.source.src_rank."
        ),
        "world_size": (
            "request.source.world_size must be the sender group's size: the "
            "receiver replies to every sender rank during the handshake."
        ),
        "syncer": (
            "the sender must use the same WeightSyncer type/settings as "
            "rollout.weight_sync (bucket vs patch, compression, bucket size)."
        ),
        "ordering": (
            "queue the request first (RolloutClient.push_weights does this), then "
            "broadcast: the receivers are receiver-driven and block on the "
            "collective until the sender arrives."
        ),
        "version": (
            "the version travels inside the payload; the ack's served_version is "
            "authoritative and may exceed request.version."
        ),
    }
