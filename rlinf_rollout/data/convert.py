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

"""Conversions between the internal collection buffers and the frozen v1 schema.

``rlinf_rollout.data.embodied_io_struct`` is the *internal* representation used by
the env/rollout workers. Everything that leaves the rollout system goes through
``rlinf_rollout.api.v1``, so consumers never depend on internal layout choices.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch

from rlinf_rollout.api.v1 import (
    ActionChunkResult,
    FinishReason,
    PolicyInputs,
    TensorLayout,
)
from rlinf_rollout.api.v1 import (
    RolloutResult as ApiRolloutResult,
)
from rlinf_rollout.api.v1 import (
    Trajectory as ApiTrajectory,
)
from rlinf_rollout.data.embodied_io_struct import RolloutResult as InternalRolloutResult
from rlinf_rollout.data.embodied_io_struct import Trajectory as InternalTrajectory

if TYPE_CHECKING:
    # Type-only: ``data.io_struct`` pulls in the Ray-backed scheduler, and this
    # module must stay importable (and testable) without a running cluster.
    from rlinf_rollout.data.io_struct import RolloutResult as InternalLLMRolloutResult

__all__ = [
    "chunk_result_to_api",
    "rollout_result_to_api",
    "trajectory_to_api",
]

#: Internal fields probed (in order) to infer ``(num_steps, num_envs)``.
_SHAPE_PROBE_FIELDS = (
    "actions",
    "rewards",
    "prev_logprobs",
    "versions",
    "intervene_flags",
)


def _infer_steps_and_envs(trajectory: InternalTrajectory) -> tuple[int, int]:
    for name in _SHAPE_PROBE_FIELDS:
        tensor = getattr(trajectory, name, None)
        if isinstance(tensor, torch.Tensor) and tensor.dim() >= 2:
            return int(tensor.shape[0]), int(tensor.shape[1])
    return 0, 0


def trajectory_to_api(
    trajectory: InternalTrajectory,
    *,
    producer: str,
    trajectory_id: str = "",
    required_keys: tuple[str, ...] = (),
    validate: bool = False,
) -> ApiTrajectory:
    """Convert an internal time-major trajectory into the v1 schema.

    Args:
        trajectory: Internal trajectory produced by
            ``EmbodiedRolloutResult.to_trajectory``/``to_splited_trajectories``.
        producer: Policy identity, used for
            :attr:`rlinf_rollout.api.v1.PolicyInputs.producer`.
        trajectory_id: Optional stable id for the payload.
        required_keys: Policy-input keys a consumer must receive.
        validate: Whether to run :meth:`ApiTrajectory.validate`. Off by default:
            an epoch's terminal observation contributes rewards / done flags but
            no action, so per-field step counts legitimately differ by one and
            strict ``(num_steps, num_envs)`` checking would reject valid data.

    Returns:
        The v1 :class:`~rlinf_rollout.api.v1.Trajectory`.
    """
    num_steps, num_envs = _infer_steps_and_envs(trajectory)
    forward_inputs = trajectory.forward_inputs or {}
    policy_inputs = (
        PolicyInputs(
            producer=producer,
            layout=TensorLayout.TIME_MAJOR,
            tensors={
                key: value
                for key, value in forward_inputs.items()
                if isinstance(value, torch.Tensor)
            },
            required_keys=required_keys,
        )
        if forward_inputs
        else None
    )

    api_trajectory = ApiTrajectory(
        trajectory_id=trajectory_id,
        layout=TensorLayout.TIME_MAJOR,
        num_steps=num_steps,
        num_envs=num_envs,
        max_episode_length=int(trajectory.max_episode_length or 0),
        model_weights_id=trajectory.model_weights_id or "",
        actions=trajectory.actions,
        rewards=trajectory.rewards,
        dones=trajectory.dones,
        terminations=trajectory.terminations,
        truncations=trajectory.truncations,
        prev_logprobs=trajectory.prev_logprobs,
        prev_values=trajectory.prev_values,
        intervene_flags=trajectory.intervene_flags,
        versions=trajectory.versions,
        curr_obs=trajectory.curr_obs or {},
        next_obs=trajectory.next_obs or {},
        policy_inputs=policy_inputs,
    )
    if validate:
        api_trajectory.validate()
    return api_trajectory


def chunk_result_to_api(
    result: InternalRolloutResult,
    *,
    producer: str,
    required_keys: tuple[str, ...] = (),
    metadata: Optional[dict[str, Any]] = None,
) -> ActionChunkResult:
    """Convert an internal per-chunk rollout result into the v1 schema.

    Args:
        result: Internal ``RolloutResult`` emitted by the rollout worker.
        producer: Policy identity for :class:`PolicyInputs`.
        required_keys: Policy-input keys a consumer must receive.
        metadata: Extra annotations to attach.

    Returns:
        The v1 :class:`~rlinf_rollout.api.v1.ActionChunkResult`. Note that
        ``bootstrap_values`` has no v1 counterpart: it is a post-processing
        quantity, carried in ``metadata`` when present.
    """
    forward_inputs = result.forward_inputs or {}
    policy_inputs = (
        PolicyInputs(
            producer=producer,
            layout=TensorLayout.FLAT,
            tensors={
                key: value
                for key, value in forward_inputs.items()
                if isinstance(value, torch.Tensor)
            },
            required_keys=required_keys,
        )
        if forward_inputs
        else None
    )
    chunk = ActionChunkResult(
        actions=result.actions,
        prev_logprobs=result.prev_logprobs,
        prev_values=result.prev_values,
        intervene_flags=result.intervene_flags,
        versions=result.versions,
        policy_inputs=policy_inputs,
    )
    if metadata:
        chunk.metadata.extra.update(metadata)
    return chunk


def rollout_result_to_api(
    result: "InternalLLMRolloutResult",
    *,
    request_ids: tuple[str, ...] = (),
    versions: tuple[int, ...] = (),
    producer: str = "",
    metadata: Optional[dict[str, Any]] = None,
) -> ApiRolloutResult:
    """Convert an engine's LLM rollout result into the v1 schema.

    The internal struct doubles as the trainer's batch container, so it carries
    fields that are trainer-side by definition (``advantages``, ``ref_logprobs``,
    ``values``, ``returns``, ``recomputed_logprobs``). Those are dropped here: the
    v1 payload only states what the *generation* produced.

    ``is_end`` (``finish_reason == "stop"``) is widened into
    :class:`~rlinf_rollout.api.v1.FinishReason`. Engines only hand a sequence group
    over once every sequence has completed, so a non-``stop`` reason means the
    length budget ran out, not an abort.

    Args:
        result: Internal ``rlinf_rollout.data.io_struct.RolloutResult``.
        request_ids: Task id per sequence; empty when the caller does not track it.
        versions: Weight version per sequence.
        producer: Identity of the engine worker, recorded in the metadata.
        metadata: Extra annotations to attach.

    Returns:
        The v1 :class:`~rlinf_rollout.api.v1.RolloutResult`.
    """
    rewards = result.rewards
    if isinstance(rewards, torch.Tensor):
        rewards = rewards.detach().float().reshape(-1).tolist()

    api_result = ApiRolloutResult(
        num_sequences=int(result.num_sequence),
        group_size=int(result.group_size),
        request_ids=tuple(request_ids),
        prompt_ids=list(result.prompt_ids),
        prompt_lengths=list(result.prompt_lengths),
        response_ids=[list(ids) for ids in result.response_ids],
        response_lengths=list(result.response_lengths),
        finish_reasons=[
            FinishReason.STOP if is_end else FinishReason.LENGTH
            for is_end in result.is_end
        ],
        response_masks=result.response_mask,
        rollout_logprobs=result.rollout_logprobs,
        prompt_texts=result.prompt_texts,
        response_texts=result.response_texts,
        answers=result.answers,
        image_data=result.image_data,
        multi_modal_inputs=result.multi_modal_inputs,
        versions=tuple(int(version) for version in versions),
        rewards=rewards,
    )
    if producer:
        api_result.metadata.producer = producer
    if metadata:
        api_result.metadata.extra.update(metadata)
    return api_result
