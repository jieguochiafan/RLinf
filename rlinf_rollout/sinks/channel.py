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

"""Channel-backed :class:`~rlinf_rollout.api.v1.TrajectorySink` implementations.

Replaces the training repo's direct ``actor_channel.put(trajectory)`` calls. The
env worker no longer knows how many shards a trainer wants, nor which channel key
each shard belongs to: it asks the sink for its :class:`ConsumerSpec`, splits the
collection buffer accordingly and pushes v1 payloads.
"""

from typing import Any, Optional, Sequence, Union

from rlinf_rollout.api.v1 import (
    ConsumerSpec,
    PartitionAxis,
    RolloutResult,
    Trajectory,
    TrajectorySink,
)

__all__ = ["ChannelTrajectorySink", "NullTrajectorySink"]


class ChannelTrajectorySink(TrajectorySink):
    """Forwards payloads to a vendored scheduler ``Channel``.

    Args:
        channel: The channel to push to.
        consumer_spec: Delivery contract declared by the consumer. When omitted a
            single unpartitioned stream is assumed.
        keys: Optional per-partition channel keys, indexed by partition. When
            omitted, payloads go to the channel's default key.
        weight: Priority weight passed to ``Channel.put``.
    """

    def __init__(
        self,
        channel: Any,
        consumer_spec: Optional[ConsumerSpec] = None,
        *,
        keys: Optional[Sequence[Any]] = None,
        weight: int = 0,
    ) -> None:
        self._channel = channel
        self._consumer_spec = consumer_spec or ConsumerSpec(
            num_partitions=1, axis=PartitionAxis.NONE
        )
        self._keys = list(keys) if keys is not None else None
        if (
            self._keys is not None
            and len(self._keys) != self._consumer_spec.num_partitions
        ):
            raise ValueError(
                f"keys has {len(self._keys)} entries but consumer_spec declares "
                f"{self._consumer_spec.num_partitions} partitions."
            )
        self._weight = weight
        self._partition_cursor = 0
        self._pending: list[Any] = []

    @property
    def consumer_spec(self) -> ConsumerSpec:
        """Delivery contract declared by the downstream consumer."""
        return self._consumer_spec

    async def put(
        self,
        item: Union[Trajectory, RolloutResult],
        *,
        partition: Optional[int] = None,
    ) -> None:
        """Push one payload, round-robining over partitions when unspecified.

        Args:
            item: A v1 trajectory or LLM rollout result.
            partition: Explicit partition index; defaults to round-robin so a
                caller that emits ``num_partitions`` shards in order lands them on
                the matching keys.
        """
        num_partitions = self._consumer_spec.num_partitions
        if partition is None:
            partition = self._partition_cursor % num_partitions
            self._partition_cursor += 1
        elif not 0 <= partition < num_partitions:
            raise IndexError(
                f"partition {partition} out of range for {num_partitions} partitions."
            )

        put_kwargs: dict[str, Any] = {"async_op": True, "weight": self._weight}
        if self._keys is not None:
            put_kwargs["key"] = self._keys[partition]
        work = self._channel.put(item, **put_kwargs)
        if work is not None:
            self._pending.append(work)

    async def flush(self) -> None:
        """Wait for all outstanding async puts to complete."""
        pending, self._pending = self._pending, []
        for work in pending:
            wait = getattr(work, "async_wait", None)
            if wait is not None:
                await wait()
            else:
                work.wait()

    async def close(self) -> None:
        """Flush outstanding puts; the channel itself is owned by the caller."""
        await self.flush()


class NullTrajectorySink(TrajectorySink):
    """Drops everything. Used for eval-only runs and for tests."""

    def __init__(self, consumer_spec: Optional[ConsumerSpec] = None) -> None:
        self._consumer_spec = consumer_spec or ConsumerSpec(
            num_partitions=1, axis=PartitionAxis.NONE
        )
        self.num_dropped = 0

    @property
    def consumer_spec(self) -> ConsumerSpec:
        """Delivery contract declared by the downstream consumer."""
        return self._consumer_spec

    async def put(
        self,
        item: Union[Trajectory, RolloutResult],
        *,
        partition: Optional[int] = None,
    ) -> None:
        """Count and discard ``item``."""
        del item, partition
        self.num_dropped += 1
