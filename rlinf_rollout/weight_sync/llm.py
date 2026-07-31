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

"""Weight-sync plumbing for the LLM engines (SGLang / vLLM).

The training repo derived the sender's rank layout from
``ModelParallelComponentPlacement`` — i.e. from ``cfg.actor.model.*`` and the actor's
GPU allocation. A standalone rollout system has neither, so the layout arrives in
:class:`rlinf_rollout.api.v1.SourceTopology` and is projected here into the
receiver-side pairing the engines need:

======================================  =============================================
training repo                           rollout system
======================================  =============================================
``placement.actor_tp_size``             ``SourceTopology.parallel_sizes["tp"]``
``placement.actor_pp_size``             ``SourceTopology.parallel_sizes["pp"]``
``placement.actor_world_size``          ``SourceTopology.world_size``
``cfg.actor.group_name``                ``SourceTopology.group_name``
``cfg.actor.training_backend != fsdp``  ``EngineWeightSyncSetup.presharded_weights``
``RankMapper.rollout_to_source_map(placement)``
                                        ``EngineWeightSyncSetup.source_rank_for(dp, tp)``
======================================  =============================================

A trainer that reshards in a way this module cannot express fills
:attr:`~rlinf_rollout.api.v1.SourceTopology.rank_map` instead, keyed by
``"<dp_rank>,<tp_rank>"``; the derived mapping is then bypassed entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from rlinf_rollout.api.v1 import SourceTopology
from rlinf_rollout.utils.placement_modes import RolloutSyncMode

__all__ = [
    "CollocateRankMapper",
    "DisaggRankMapper",
    "EngineWeightSyncSetup",
    "RankMapper",
    "SourceLayout",
    "parse_rank_map",
    "format_rollout_rank",
]


def format_rollout_rank(dp_rank: int, tp_rank: int) -> str:
    """Encode a 2D rollout rank as the string key used by ``SourceTopology.rank_map``."""
    return f"{int(dp_rank)},{int(tp_rank)}"


def parse_rank_map(rank_map: dict[str, int]) -> dict[tuple[int, int], int]:
    """Decode ``SourceTopology.rank_map`` into ``{(dp_rank, tp_rank): source_rank}``.

    Args:
        rank_map: Mapping whose keys are ``"<dp_rank>,<tp_rank>"``.

    Returns:
        The decoded mapping, empty when ``rank_map`` is empty.

    Raises:
        ValueError: If a key is not a ``"<int>,<int>"`` pair.
    """
    decoded: dict[tuple[int, int], int] = {}
    for key, src_rank in rank_map.items():
        parts = str(key).split(",")
        if len(parts) != 2:
            raise ValueError(
                f"SourceTopology.rank_map keys must be '<dp_rank>,<tp_rank>', got {key!r}."
            )
        try:
            dp_rank, tp_rank = int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise ValueError(
                f"SourceTopology.rank_map keys must be '<dp_rank>,<tp_rank>', got {key!r}."
            ) from exc
        decoded[(dp_rank, tp_rank)] = int(src_rank)
    return decoded


@dataclass(frozen=True)
class SourceLayout:
    """Model-parallel layout of the weight sender, read off a :class:`SourceTopology`.

    Args:
        group_name: Sender worker-group name (was ``cfg.actor.group_name``).
        world_size: Total sender ranks (was ``placement.actor_world_size``).
        tp_size: Sender tensor-parallel size (was ``placement.actor_tp_size``).
        pp_size: Sender pipeline-parallel size (was ``placement.actor_pp_size``).
        rank_map: Explicit ``{(dp_rank, tp_rank): source_rank}`` override supplied by
            the trainer; empty means "derive it".
    """

    group_name: str
    world_size: int
    tp_size: int = 1
    pp_size: int = 1
    rank_map: dict[tuple[int, int], int] = field(default_factory=dict)

    @classmethod
    def from_topology(cls, source: SourceTopology) -> "SourceLayout":
        """Project an api/v1 :class:`SourceTopology` onto this layout.

        Args:
            source: Sender description carried by a ``WeightUpdateRequest``.

        Returns:
            The projected layout.

        Raises:
            ValueError: If the topology names no group or no world size.
        """
        if not source.group_name:
            raise ValueError(
                "SourceTopology.group_name must name the weight-sending worker group."
            )
        world_size = int(source.world_size)
        if world_size <= 0:
            raise ValueError(
                "SourceTopology.world_size must be positive for LLM weight sync; "
                "the sender's rank layout cannot be derived without it."
            )
        return cls(
            group_name=str(source.group_name),
            world_size=world_size,
            tp_size=int(source.parallel_sizes.get("tp", 1)),
            pp_size=int(source.parallel_sizes.get("pp", 1)),
            rank_map=parse_rank_map(source.rank_map),
        )


class RankMapper:
    """Pairs sender ranks with rollout engine ranks for weight transfer.

    The algorithms are unchanged from the training repo's
    ``rlinf/workers/rollout/utils.py``; only the inputs moved from a placement
    object to a :class:`SourceLayout`.
    """

    @classmethod
    def source_rank_to_rollout_rank_map(
        cls,
        source: SourceLayout,
        rollout_tp_size: int,
        rollout_world_size: int,
        sync_mode: RolloutSyncMode,
    ):
        """Map each sender 1D rank to the rollout 2D rank(s) it feeds."""
        return cls._get_rank_mapper(sync_mode).source_to_rollout_map(
            source.tp_size,
            source.pp_size,
            source.world_size,
            rollout_tp_size,
            rollout_world_size,
        )

    @classmethod
    def rollout_rank_to_source_rank_map(
        cls,
        source: SourceLayout,
        rollout_tp_size: int,
        rollout_world_size: int,
        sync_mode: RolloutSyncMode,
    ) -> dict[tuple[int, int], int]:
        """Map each rollout ``(dp_rank, tp_rank)`` to the sender rank it pairs with.

        An explicit :attr:`SourceLayout.rank_map` short-circuits the derivation.
        """
        if source.rank_map:
            return dict(source.rank_map)
        return cls._get_rank_mapper(sync_mode).rollout_to_source_map(
            source.tp_size,
            source.pp_size,
            source.world_size,
            rollout_tp_size,
            rollout_world_size,
        )

    @staticmethod
    def _get_rank_mapper(sync_mode: RolloutSyncMode):
        """Return the mapper class matching ``sync_mode``."""
        if sync_mode == RolloutSyncMode.COLLOCATED:
            return CollocateRankMapper
        elif sync_mode == RolloutSyncMode.DISAGGREGATED:
            return DisaggRankMapper
        else:
            raise ValueError(f"Unsupported rollout sync mode: {sync_mode}.")


class CollocateRankMapper(RankMapper):
    """Sender and rollout share GPUs; weights travel via CUDA IPC handles."""

    @classmethod
    def source_to_rollout_map(
        cls,
        source_tp_size: int,
        source_pp_size: int,
        source_world_size: int,
        rollout_tp_size: int,
        rollout_world_size: int,
    ) -> dict[int, tuple[int, int]]:
        """Get the global mapping from sender 1D rank to rollout 2D rank as dict."""
        # rank -> (dp, tp)
        if source_tp_size == 1:
            return {
                rank: (rank // rollout_tp_size, rank % rollout_tp_size)
                for rank in range(source_world_size)
            }
        rank_map = {}
        for source_rank in range(source_world_size):
            rank_map[source_rank] = cls._source_rank_to_rollout_rank(
                source_rank,
                source_tp_size,
                rollout_tp_size,
            )
        return rank_map

    @classmethod
    def rollout_to_source_map(
        cls,
        source_tp_size: int,
        source_pp_size: int,
        source_world_size: int,
        rollout_tp_size: int,
        rollout_world_size: int,
    ):
        """Get the global mapping from rollout 2D rank to sender 1D rank as dict."""
        rank_map = cls.source_to_rollout_map(
            source_tp_size,
            source_pp_size,
            source_world_size,
            rollout_tp_size,
            rollout_world_size,
        )
        return {v: k for k, v in rank_map.items()}

    @staticmethod
    def _source_rank_to_rollout_rank(
        source_rank: int,
        source_tp_size: int,
        rollout_tp_size: int,
    ):
        """Get the mapping from sender 1D rank to rollout 2D rank."""
        num_rollout_dp_ranks_per_source_tp_group = source_tp_size // rollout_tp_size

        source_tp_rank = source_rank % source_tp_size

        source_tp_group_id = source_rank // source_tp_size
        rollout_start_dp_rank = (
            source_tp_group_id * num_rollout_dp_ranks_per_source_tp_group
        )

        weight_dst_dp_rank_in_rollout = (
            rollout_start_dp_rank
            + source_tp_rank % num_rollout_dp_ranks_per_source_tp_group
        )

        weight_dst_tp_rank_in_rollout = (
            source_tp_rank // num_rollout_dp_ranks_per_source_tp_group
        )

        return (weight_dst_dp_rank_in_rollout, weight_dst_tp_rank_in_rollout)


class DisaggRankMapper(RankMapper):
    """Sender and rollout own disjoint GPUs; weights travel over a collective.

    Assumes ``source_tp_size = n * rollout_tp_size``.
    """

    @classmethod
    def source_to_rollout_map(
        cls,
        source_tp_size: int,
        source_pp_size: int,
        source_world_size: int,
        rollout_tp_size: int,
        rollout_world_size: int,
    ) -> dict[int, list[tuple[int, int]]]:
        """Only ranks in the sender's dp=0 group send weights to the rollout LLM."""
        source_model_parallel_size = source_tp_size
        assert rollout_world_size >= source_model_parallel_size, (
            f"rollout_world_size ({rollout_world_size}) should more than source_model_parallel_size ({source_model_parallel_size})"
        )

        assert rollout_world_size % source_model_parallel_size == 0, (
            f"rollout_world_size ({rollout_world_size}) should be a multiple of source_model_parallel_size ({source_model_parallel_size})"
        )

        source_dp = source_world_size // source_tp_size
        stride = source_model_parallel_size // rollout_tp_size

        rank_map = {}
        for source_rank in range(source_world_size):
            if source_rank > rollout_world_size:
                rank_map[source_rank] = []
                continue
            gen_dp, gen_tp = cls._source_rank_to_rollout_rank(
                source_rank,
                source_tp_size,
                rollout_tp_size,
            )
            if source_world_size <= rollout_world_size:
                rank_map[source_rank] = [
                    (gen_dp + i * stride * source_dp, gen_tp)
                    for i in range(rollout_world_size // source_world_size)
                ]
            elif source_rank < rollout_world_size:
                rank_map[source_rank] = [(gen_dp, gen_tp)]
            else:
                rank_map[source_rank] = []

        return rank_map

    @classmethod
    def rollout_to_source_map(
        cls,
        source_tp_size: int,
        source_pp_size: int,
        source_world_size: int,
        rollout_tp_size: int,
        rollout_world_size: int,
    ) -> dict[tuple[int, int], int]:
        """Invert :meth:`source_to_rollout_map`."""
        rank_map = cls.source_to_rollout_map(
            source_tp_size,
            source_pp_size,
            source_world_size,
            rollout_tp_size,
            rollout_world_size,
        )
        result_map = {}
        for source_rank, rollout_2d_ranks in rank_map.items():
            for rollout_2d_rank in rollout_2d_ranks:
                result_map[rollout_2d_rank] = source_rank
        return result_map

    @staticmethod
    def _source_rank_to_rollout_rank(
        source_rank: int,
        source_tp_size: int,
        rollout_tp_size: int,
    ) -> tuple[int, int]:
        """Get the mapping from sender 1D rank to rollout 2D rank."""
        assert source_tp_size % rollout_tp_size == 0, (
            "source_tp_size must be a multiple of rollout_tp_size"
        )

        num_rollout_dp_ranks_per_source_tp_group = source_tp_size // rollout_tp_size
        source_tp_rank = source_rank % source_tp_size
        source_tp_group_id = source_rank // source_tp_size
        rollout_start_dp_rank = (
            source_tp_group_id * num_rollout_dp_ranks_per_source_tp_group
        )
        weight_dst_dp_rank_in_rollout = (
            rollout_start_dp_rank
            + source_tp_rank % num_rollout_dp_ranks_per_source_tp_group
        )
        weight_dst_tp_rank_in_rollout = (
            source_tp_rank // num_rollout_dp_ranks_per_source_tp_group
        )

        return (weight_dst_dp_rank_in_rollout, weight_dst_tp_rank_in_rollout)


@dataclass(frozen=True)
class EngineWeightSyncSetup:
    """Everything an LLM engine process needs to receive weights.

    This dataclass is what crosses the process boundary into the SGLang scheduler
    and the vLLM worker, replacing the ``(placement, full_trainer_cfg)`` pair the
    training repo passed. It is plain data (picklable, no Ray or OmegaConf
    objects) so it survives ``spawn``.

    Args:
        source: Sender layout, projected from a ``SourceTopology``.
        rollout_tp_size: Tensor-parallel size of one rollout engine.
        rollout_world_size: Total accelerators across all rollout engines.
        sync_mode: ``COLLOCATED`` (CUDA IPC handles) or ``DISAGGREGATED``
            (tensors received directly over the collective).
        presharded_weights: Whether the sender already ships per-TP-rank shards
            (was ``cfg.actor.training_backend != "fsdp"``).
        validate_first_sync: Whether to snapshot weight norms at load time and
            compare them after the first sync. Callers must pass ``False`` when
            the trainer resumes from a checkpoint, since the engine's HF weights
            then cannot match the trainer's.
    """

    source: SourceLayout
    rollout_tp_size: int
    rollout_world_size: int
    sync_mode: RolloutSyncMode = RolloutSyncMode.DISAGGREGATED
    presharded_weights: bool = False
    validate_first_sync: bool = False

    @classmethod
    def from_topology(
        cls,
        source: SourceTopology,
        *,
        rollout_tp_size: int,
        rollout_world_size: int,
        sync_mode: RolloutSyncMode = RolloutSyncMode.DISAGGREGATED,
        presharded_weights: bool = False,
        validate_first_sync: bool = False,
    ) -> "EngineWeightSyncSetup":
        """Build a setup straight from an api/v1 :class:`SourceTopology`."""
        return cls(
            source=SourceLayout.from_topology(source),
            rollout_tp_size=int(rollout_tp_size),
            rollout_world_size=int(rollout_world_size),
            sync_mode=sync_mode,
            presharded_weights=bool(presharded_weights),
            validate_first_sync=bool(validate_first_sync),
        )

    @property
    def source_group_name(self) -> str:
        """Worker-group name of the weight sender."""
        return self.source.group_name

    @property
    def is_collocated(self) -> bool:
        """Whether weights arrive as CUDA IPC handles rather than tensors."""
        return self.sync_mode == RolloutSyncMode.COLLOCATED

    def rollout_rank_to_source_rank(self) -> dict[tuple[int, int], int]:
        """Full ``{(dp_rank, tp_rank): source_rank}`` mapping for this setup."""
        return RankMapper.rollout_rank_to_source_rank_map(
            self.source,
            self.rollout_tp_size,
            self.rollout_world_size,
            self.sync_mode,
        )

    def source_rank_for(self, dp_rank: int, tp_rank: int) -> int:
        """Return the sender rank that feeds rollout rank ``(dp_rank, tp_rank)``.

        Raises:
            KeyError: If no sender rank is paired with that rollout rank.
        """
        rank_map = self.rollout_rank_to_source_rank()
        key = (int(dp_rank), int(tp_rank))
        if key not in rank_map:
            raise KeyError(
                f"rollout rank {key} has no weight source; known pairs: "
                f"{sorted(rank_map)}"
            )
        return rank_map[key]

    def describe(self) -> dict[str, Optional[object]]:
        """Return a log-friendly summary of the setup."""
        return {
            "source_group_name": self.source.group_name,
            "source_world_size": self.source.world_size,
            "source_tp_size": self.source.tp_size,
            "source_pp_size": self.source.pp_size,
            "rollout_tp_size": self.rollout_tp_size,
            "rollout_world_size": self.rollout_world_size,
            "sync_mode": self.sync_mode.name,
            "presharded_weights": self.presharded_weights,
            "validate_first_sync": self.validate_first_sync,
            "explicit_rank_map": bool(self.source.rank_map),
        }
