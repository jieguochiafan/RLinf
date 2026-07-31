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

"""Placement for the standalone LLM rollout system.

``ModelParallelComponentPlacement`` (vendored in :mod:`rlinf_rollout.utils.placement`)
cannot be used by a rollout-only deployment: it *requires* an ``actor`` entry in
``cluster.component_placement``, infers the placement mode from whether the actor and
rollout GPU sets overlap, and reads the actor's parallel sizes from
``cfg.actor.model.*``. None of that exists once the trainer lives in another process
tree.

:class:`RolloutComponentPlacement` keeps the same read surface the LLM engines and
workers rely on (``is_collocated`` / ``is_pipeline`` / ``is_auto``,
``rollout_{dp,tp,world}_size``, ``rollout_sync_mode``) but:

* only the ``rollout`` component is required — ``reward`` and any auxiliary
  components stay optional;
* the mode is *declared* by ``rollout.placement_mode`` instead of inferred from the
  trainer's GPU allocation;
* the collocated stride, which the training repo computed as
  ``actor_tp_size // rollout_tp_size``, comes from the weight sender's topology
  (``rollout.weight_sync.source.parallel_sizes.tp``).
"""

from __future__ import annotations

import logging
from typing import Optional

from omegaconf import DictConfig, OmegaConf

from rlinf_rollout.scheduler import (
    Cluster,
    ComponentPlacement,
    PackedPlacementStrategy,
)
from rlinf_rollout.utils.placement_modes import (
    PlacementMode,
    RolloutSyncMode,
    placement_mode_to_rollout_sync_mode,
)

__all__ = ["ROLLOUT_PLACEMENT_MODES", "RolloutComponentPlacement"]

#: Accepted values of ``rollout.placement_mode``.
ROLLOUT_PLACEMENT_MODES: dict[str, PlacementMode] = {
    "collocated": PlacementMode.COLLOCATED,
    "disaggregated": PlacementMode.DISAGGREGATED,
    "auto": PlacementMode.AUTO,
}


class RolloutComponentPlacement(ComponentPlacement):
    """Places LLM rollout engines (and optional helper components) on accelerators.

    Args:
        config: Rollout-system config. Must carry ``cluster.component_placement``
            with a ``rollout`` entry and the ``rollout.tensor_parallel_size`` /
            ``rollout.pipeline_parallel_size`` sizes.
        cluster: Active cluster.

    Raises:
        AssertionError: If the ``rollout`` component is missing, its accelerator
            ranks are not contiguous, or the parallel sizes do not divide the
            allocation.
        ValueError: If ``rollout.placement_mode`` is not one of
            :data:`ROLLOUT_PLACEMENT_MODES`.
    """

    def __init__(self, config: DictConfig, cluster: Cluster):
        super().__init__(config, cluster)

        self._rollout_gpus = self._get_component_hardware("rollout")
        self._reward_gpus = self._get_component_hardware("reward")
        self._cluster_num_gpus = cluster.num_accelerators

        assert self._rollout_gpus is not None, (
            "Rollout accelerators must be specified in the component_placement config."
        )
        assert self._rollout_gpus == list(
            range(self._rollout_gpus[0], self._rollout_gpus[-1] + 1)
        ), f"Rollout accelerators {self._rollout_gpus} must be contiguous."

        self._rollout_num_gpus = len(self._rollout_gpus)
        self._reward_num_gpus = len(self._reward_gpus) if self._reward_gpus else 0

        mode_name = str(
            OmegaConf.select(config, "rollout.placement_mode", default="disaggregated")
        )
        if mode_name not in ROLLOUT_PLACEMENT_MODES:
            raise ValueError(
                f"rollout.placement_mode must be one of "
                f"{sorted(ROLLOUT_PLACEMENT_MODES)}, got {mode_name!r}."
            )
        self._placement_mode = ROLLOUT_PLACEMENT_MODES[mode_name]
        self._rollout_sync_mode = placement_mode_to_rollout_sync_mode(
            self._placement_mode
        )
        logging.info("Rollout placement mode: %s", self._placement_mode.name)

        assert self.rollout_tp_size <= self.rollout_world_size, (
            f"Rollout TP size {self.rollout_tp_size} must be less than or equal to "
            f"Rollout world size {self.rollout_world_size}."
        )
        assert self._rollout_num_gpus % self.num_gpus_per_engine == 0, (
            f"Rollout accelerators ({self._rollout_num_gpus}) must be divisible by "
            f"tensor_parallel_size * pipeline_parallel_size "
            f"({self.num_gpus_per_engine})."
        )

        self._generate_placements()

    # ------------------------------------------------------------------ sizes

    @property
    def num_gpus_per_engine(self) -> int:
        """Accelerators one rollout engine process occupies."""
        return int(
            OmegaConf.select(self._config, "rollout.tensor_parallel_size", default=1)
        ) * int(
            OmegaConf.select(self._config, "rollout.pipeline_parallel_size", default=1)
        )

    @property
    def rollout_dp_size(self) -> int:
        """Number of rollout engine processes."""
        return self._rollout_num_gpus // self.num_gpus_per_engine

    @property
    def rollout_tp_size(self) -> int:
        """Tensor-parallel size of one rollout engine."""
        return int(
            OmegaConf.select(self._config, "rollout.tensor_parallel_size", default=1)
        )

    @property
    def rollout_world_size(self) -> int:
        """Total accelerators across all rollout engines."""
        return self._rollout_num_gpus

    @property
    def reward_world_size(self) -> int:
        """Accelerators allocated to the optional reward component."""
        return self._reward_num_gpus

    # ------------------------------------------------------------------ modes

    @property
    def placement_mode(self) -> PlacementMode:
        """Declared placement mode."""
        return self._placement_mode

    @property
    def rollout_sync_mode(self) -> RolloutSyncMode:
        """How weights reach the engines: CUDA IPC handles or collective tensors."""
        return self._rollout_sync_mode

    @property
    def is_collocated(self) -> bool:
        """Whether the trainer shares these accelerators with the rollout engines."""
        return self._placement_mode == PlacementMode.COLLOCATED

    @property
    def is_disaggregated(self) -> bool:
        """Whether the rollout engines own their accelerators exclusively."""
        return self._placement_mode == PlacementMode.DISAGGREGATED

    @property
    def is_auto(self) -> bool:
        """Whether the dynamic scheduler may scale the engines up and down."""
        return self._placement_mode == PlacementMode.AUTO

    @property
    def is_pipeline(self) -> bool:
        """Whether generation may stream results out before the batch completes."""
        return self.is_disaggregated or self.is_auto

    @property
    def has_dedicated_inference(self) -> bool:
        """Always ``False``: recomputing logprobs is a trainer concern."""
        return False

    # ------------------------------------------------------------- placements

    @property
    def source_tp_size(self) -> int:
        """Tensor-parallel size of the weight sender, per the handshake topology.

        Replaces ``ModelParallelComponentPlacement.actor_tp_size``, which read
        ``cfg.actor.model.tensor_model_parallel_size``.
        """
        return int(
            OmegaConf.select(
                self._config,
                "rollout.weight_sync.source.parallel_sizes.tp",
                default=1,
            )
        )

    def _collocated_stride(self) -> int:
        """Stride that lines engine ranks up with the sender's TP shards."""
        source_tp_size = self.source_tp_size
        rollout_tp_size = self.rollout_tp_size
        if source_tp_size <= rollout_tp_size:
            return 1
        assert source_tp_size % rollout_tp_size == 0, (
            f"Weight source TP size ({source_tp_size}) must be divisible by rollout "
            f"TP size ({rollout_tp_size}); set "
            f"rollout.weight_sync.source.parallel_sizes.tp accordingly."
        )
        return source_tp_size // rollout_tp_size

    def _generate_placements(self) -> None:
        """Build the packed strategies for the rollout (and reward) components."""
        stride = self._collocated_stride() if self.is_collocated else 1
        self._placements["rollout"] = PackedPlacementStrategy(
            self._rollout_gpus[0],
            self._rollout_gpus[-1],
            num_hardware_per_process=self.num_gpus_per_engine,
            stride=stride,
        )
        if self._reward_gpus:
            self._placements["reward"] = PackedPlacementStrategy(
                self._reward_gpus[0], self._reward_gpus[-1]
            )

    def _get_component_hardware(self, component_name: str) -> Optional[list[int]]:
        """Hardware ranks of ``component_name``, or ``None`` when not placed."""
        if component_name not in self._component_rank_map:
            return None
        return super().get_hardware_ranks(component_name)
