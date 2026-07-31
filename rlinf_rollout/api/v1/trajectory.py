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

"""Rollout output protocol: interface 2 of the standalone rollout system.

Two payload families share one delivery interface:

- Embodied: :class:`Trajectory`, a time-major batch of transitions produced by
  env workers, with :class:`ActionChunkResult` as the per-chunk policy output
  exchanged internally between rollout and env workers.
- LLM: :class:`RolloutResult`, a batch of generated sequences.

Differences from the in-tree RLinf structs this is distilled from:

- No splitting by trainer world size. The consumer declares its partitioning via
  :class:`ConsumerSpec`; the sink implementation honors it.
- ``forward_inputs`` is replaced by the self-describing :class:`PolicyInputs`,
  so a consumer can tell what a policy produced without sharing code.
- Training-only quantities (advantages, returns, reference logprobs, bootstrap
  values) are absent; they belong to the trainer or to an optional
  post-processing plugin.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Optional, Union

import torch

from .common import Metadata, SchemaBase, SchemaError


class TensorLayout(str, Enum):
    """Axis order of batched trajectory tensors.

    Attributes:
        TIME_MAJOR: ``[T, B, ...]`` — step axis first (embodied default).
        BATCH_MAJOR: ``[B, T, ...]`` — batch axis first.
        FLAT: ``[B, ...]`` — a single step, no time axis.
    """

    TIME_MAJOR = "time_major"
    BATCH_MAJOR = "batch_major"
    FLAT = "flat"


class PartitionAxis(str, Enum):
    """Axis a consumer wants its data partitioned along.

    Attributes:
        ENV: Parallel-environment axis of an embodied trajectory.
        SEQUENCE: Sequence axis of an LLM rollout result.
        NONE: Deliver whole payloads, unpartitioned.
    """

    ENV = "env"
    SEQUENCE = "sequence"
    NONE = "none"


class FinishReason(str, Enum):
    """Why a generated sequence stopped.

    Attributes:
        STOP: A stop condition/token was reached.
        LENGTH: The length budget was exhausted.
        ABORT: Generation was aborted and the sequence is incomplete.
    """

    STOP = "stop"
    LENGTH = "length"
    ABORT = "abort"


@dataclass(kw_only=True)
class PolicyInputs(SchemaBase):
    """Self-describing tensors a policy needs to recompute its own forward pass.

    Replaces the implicit ``forward_inputs`` contract: the producing policy
    states its identity, the tensor layout, and which keys a consumer must feed
    back into the model.

    Args:
        producer: Policy identity that produced these tensors, e.g.
            ``"openvla_oft"`` or ``"openpi"``.
        layout: Axis order of every tensor in :attr:`tensors`.
        tensors: Named tensors, e.g. tokenized inputs, masks, sampled actions.
        required_keys: Subset of :attr:`tensors` keys that a training forward
            pass must receive. Empty means "all keys".
    """

    producer: str = ""
    layout: TensorLayout = TensorLayout.TIME_MAJOR
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    required_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.layout = TensorLayout(self.layout)
        missing = [key for key in self.required_keys if key not in self.tensors]
        if missing:
            raise SchemaError(
                f"PolicyInputs from {self.producer!r} declares required_keys "
                f"{missing} that are absent from tensors."
            )

    def describe(self) -> dict[str, dict[str, Any]]:
        """Return per-key shape/dtype description of :attr:`tensors`."""
        return {
            key: {"shape": tuple(value.shape), "dtype": str(value.dtype)}
            for key, value in self.tensors.items()
        }


@dataclass(kw_only=True)
class ActionChunkResult(SchemaBase):
    """Policy output for one (chunked) action step.

    Produced by a rollout worker and consumed by an env worker. Tensors are
    batch-major over parallel environments (``[B, ...]``).

    Args:
        actions: Actions to execute, ``[B, num_action_chunks * action_dim]``.
        prev_logprobs: Log-probabilities of :attr:`actions` under the acting
            policy, when the policy exposes them.
        prev_values: Value estimates for the current observation, ``[B, 1]``.
        intervene_flags: Per-chunk flag marking externally supplied actions
            (expert/teleop), ``[B, num_action_chunks]``.
        versions: Weight version that produced each row, ``[B, 1]``.
        policy_inputs: Self-describing forward inputs for this step.
        metadata: Free-form annotations.
    """

    actions: Optional[torch.Tensor] = None
    prev_logprobs: Optional[torch.Tensor] = None
    prev_values: Optional[torch.Tensor] = None
    intervene_flags: Optional[torch.Tensor] = None
    versions: Optional[torch.Tensor] = None
    policy_inputs: Optional[PolicyInputs] = None
    metadata: Metadata = field(default_factory=Metadata)

    @property
    def batch_size(self) -> int:
        """Number of parallel environments in this chunk step, ``0`` if empty."""
        return 0 if self.actions is None else int(self.actions.shape[0])


@dataclass(kw_only=True)
class Trajectory(SchemaBase):
    """Embodied trajectory: a batch of transitions over parallel environments.

    Tensors follow :attr:`layout`; for the default time-major layout the leading
    axes are ``[num_steps, num_envs, ...]``. ``terminations`` / ``truncations`` /
    ``dones`` may carry extra leading rows when the collector records episode
    boundaries in addition to steps; :meth:`validate` tolerates that.

    Args:
        trajectory_id: Unique id of this payload.
        layout: Axis order of the tensor fields.
        num_steps: Number of collected chunk steps.
        num_envs: Number of parallel environments.
        max_episode_length: Episode-length cap used by the env.
        model_weights_id: Stable digest of :attr:`versions`, identifying the
            weight mix that generated the data.
        actions: Executed actions.
        rewards: Per-step rewards.
        dones: Episode-end flags.
        terminations: Task-completion flags.
        truncations: Time-limit flags.
        prev_logprobs: Acting-policy log-probabilities.
        prev_values: Acting-policy value estimates.
        intervene_flags: Flags marking externally supplied actions.
        versions: Weight version per element.
        curr_obs: Observation tensors before each step.
        next_obs: Observation tensors after each step.
        policy_inputs: Self-describing forward inputs for training.
        metadata: Free-form annotations.
    """

    trajectory_id: str = ""
    layout: TensorLayout = TensorLayout.TIME_MAJOR
    num_steps: int = 0
    num_envs: int = 0
    max_episode_length: int = 0
    model_weights_id: str = ""

    actions: Optional[torch.Tensor] = None
    rewards: Optional[torch.Tensor] = None
    dones: Optional[torch.Tensor] = None
    terminations: Optional[torch.Tensor] = None
    truncations: Optional[torch.Tensor] = None
    prev_logprobs: Optional[torch.Tensor] = None
    prev_values: Optional[torch.Tensor] = None
    intervene_flags: Optional[torch.Tensor] = None
    versions: Optional[torch.Tensor] = None

    curr_obs: dict[str, torch.Tensor] = field(default_factory=dict)
    next_obs: dict[str, torch.Tensor] = field(default_factory=dict)
    policy_inputs: Optional[PolicyInputs] = None
    metadata: Metadata = field(default_factory=Metadata)

    #: Fields whose leading axes must match ``(num_steps, num_envs)`` exactly.
    STEP_ALIGNED_FIELDS: ClassVar[tuple[str, ...]] = (
        "actions",
        "rewards",
        "prev_logprobs",
        "intervene_flags",
        "versions",
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.layout = TensorLayout(self.layout)
        if self.num_steps < 0 or self.num_envs < 0:
            raise SchemaError(
                f"num_steps/num_envs must be non-negative, got "
                f"{self.num_steps}/{self.num_envs}."
            )

    @property
    def num_transitions(self) -> int:
        """Total number of transitions, i.e. ``num_steps * num_envs``."""
        return self.num_steps * self.num_envs

    def validate(self) -> None:
        """Check declared shape metadata against the actual tensors.

        Raises:
            SchemaError: If a step-aligned tensor disagrees with
                ``(num_steps, num_envs)``, or if the layout is unsupported for
                shape checking.
        """
        if self.layout is TensorLayout.TIME_MAJOR:
            expected = (self.num_steps, self.num_envs)
        elif self.layout is TensorLayout.BATCH_MAJOR:
            expected = (self.num_envs, self.num_steps)
        else:
            raise SchemaError(
                f"Cannot validate trajectory shapes for layout {self.layout}."
            )

        for name in self.STEP_ALIGNED_FIELDS:
            tensor = getattr(self, name)
            if tensor is None:
                continue
            actual = tuple(int(dim) for dim in tensor.shape[:2])
            if actual != expected:
                raise SchemaError(
                    f"Trajectory field {name!r} has leading dims {actual}, "
                    f"expected {expected} for layout {self.layout.value}."
                )


@dataclass(kw_only=True)
class RolloutResult(SchemaBase):
    """LLM rollout output: a batch of generated sequences.

    Only generation-side facts live here. Rewards are optional and set solely
    when a reward plugin runs inside the rollout system; advantages, returns and
    reference log-probabilities are trainer-side concerns and intentionally
    absent.

    Args:
        num_sequences: Number of sequences in this batch.
        group_size: Number of samples generated per prompt.
        request_ids: Task id of each sequence, aligned with the other lists.
        prompt_ids: Prompt token ids per sequence.
        prompt_lengths: Prompt length per sequence.
        response_ids: Generated token ids per sequence.
        response_lengths: Generated length per sequence.
        finish_reasons: Stop reason per sequence.
        response_masks: Per-token mask over the response marking model-generated
            tokens (``0`` for injected tool/environment tokens).
        rollout_logprobs: Engine-reported log-probabilities per response token.
        prompt_texts: Optional decoded prompts.
        response_texts: Optional decoded responses.
        answers: Optional reference answers carried through from the task.
        image_data: Optional raw image payloads (bytes or URLs) per sequence.
        multi_modal_inputs: Optional preprocessed multi-modal inputs per
            sequence.
        versions: Weight version that generated each sequence.
        rewards: Optional per-sequence reward, when computed in-system.
        metadata: Free-form annotations.
    """

    num_sequences: int
    group_size: int = 1
    request_ids: tuple[str, ...] = ()
    prompt_ids: list[list[int]] = field(default_factory=list)
    prompt_lengths: list[int] = field(default_factory=list)
    response_ids: list[list[int]] = field(default_factory=list)
    response_lengths: list[int] = field(default_factory=list)
    finish_reasons: list[FinishReason] = field(default_factory=list)
    response_masks: Optional[list[list[int]]] = None
    rollout_logprobs: Optional[list[list[float]]] = None
    prompt_texts: Optional[list[str]] = None
    response_texts: Optional[list[str]] = None
    answers: Optional[list[Any]] = None
    image_data: Optional[list[list[Union[bytes, str]]]] = None
    multi_modal_inputs: Optional[list[Optional[dict]]] = None
    versions: tuple[int, ...] = ()
    rewards: Optional[list[float]] = None
    metadata: Metadata = field(default_factory=Metadata)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.num_sequences < 0:
            raise SchemaError(
                f"num_sequences must be non-negative, got {self.num_sequences}."
            )
        if self.group_size <= 0:
            raise SchemaError(f"group_size must be positive, got {self.group_size}.")
        if self.num_sequences % self.group_size != 0:
            raise SchemaError(
                f"num_sequences ({self.num_sequences}) must be divisible by "
                f"group_size ({self.group_size})."
            )
        self.finish_reasons = [FinishReason(reason) for reason in self.finish_reasons]

    @property
    def num_prompts(self) -> int:
        """Number of distinct prompts represented in this batch."""
        return self.num_sequences // self.group_size

    @property
    def is_end(self) -> list[bool]:
        """Per-sequence flag telling whether generation completed normally."""
        return [reason is not FinishReason.ABORT for reason in self.finish_reasons]


@dataclass(kw_only=True)
class ConsumerSpec(SchemaBase):
    """Consumer-declared delivery contract for rollout outputs.

    The rollout system never guesses a trainer's data-parallel layout; the
    consumer states how many partitions it wants and along which axis.

    Args:
        num_partitions: Number of partitions each payload is split into.
        axis: Axis to split along, see :class:`PartitionAxis`.
        partition_sizes: Explicit, possibly uneven partition sizes. Empty means
            even chunks.
        layout: Tensor layout the consumer expects.
        required_keys: Keys the consumer needs in
            :attr:`Trajectory.policy_inputs`.
    """

    num_partitions: int = 1
    axis: PartitionAxis = PartitionAxis.NONE
    partition_sizes: tuple[int, ...] = ()
    layout: TensorLayout = TensorLayout.TIME_MAJOR
    required_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.axis = PartitionAxis(self.axis)
        self.layout = TensorLayout(self.layout)
        if self.num_partitions <= 0:
            raise SchemaError(
                f"num_partitions must be positive, got {self.num_partitions}."
            )
        if self.partition_sizes and len(self.partition_sizes) != self.num_partitions:
            raise SchemaError(
                f"partition_sizes has {len(self.partition_sizes)} entries but "
                f"num_partitions is {self.num_partitions}."
            )
        if self.axis is PartitionAxis.NONE and self.num_partitions != 1:
            raise SchemaError(
                "num_partitions must be 1 when axis is 'none' (no partitioning)."
            )


class TrajectorySink(ABC):
    """Destination for rollout outputs.

    Implementations forward payloads to a trainer (channel, queue, HTTP, disk).
    The rollout side only ever calls :meth:`put`; splitting/reordering is the
    sink's business, driven by :attr:`consumer_spec`.
    """

    @property
    @abstractmethod
    def consumer_spec(self) -> ConsumerSpec:
        """Delivery contract declared by the downstream consumer."""

    @abstractmethod
    async def put(self, item: Union[Trajectory, RolloutResult]) -> None:
        """Hand one payload to the consumer.

        Args:
            item: Embodied trajectory or LLM rollout result.
        """

    async def flush(self) -> None:
        """Block until buffered payloads are delivered. Default: no-op."""

    async def close(self) -> None:
        """Release sink resources. Default: no-op."""
