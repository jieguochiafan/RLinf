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

"""Task input protocol: interface 3 of the standalone rollout system.

A :class:`RolloutTask` is the unit of work pulled by the rollout controller. It
carries either an LLM prompt batch (:class:`PromptSpec`) or an embodied
episode/evaluation spec (:class:`EpisodeSpec`), never both.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

from .common import Metadata, SchemaBase, SchemaError


class TaskKind(str, Enum):
    """Kind of work a task describes.

    Attributes:
        LLM_GENERATION: Generate responses for a prompt batch.
        EMBODIED_EPISODE: Collect training episodes in a simulator or on
            hardware.
        EMBODIED_EVAL: Run evaluation episodes; outputs are metrics-oriented and
            must not be used for training.
    """

    LLM_GENERATION = "llm_generation"
    EMBODIED_EPISODE = "embodied_episode"
    EMBODIED_EVAL = "embodied_eval"


class RolloutMode(str, Enum):
    """Whether a task collects training data or measures performance.

    Attributes:
        TRAIN: Sampling behavior configured for exploration.
        EVAL: Deterministic/greedy behavior for measurement.
    """

    TRAIN = "train"
    EVAL = "eval"


@dataclass(kw_only=True)
class SamplingParams(SchemaBase):
    """Generation knobs for LLM tasks.

    Args:
        n: Samples per prompt.
        temperature: Softmax temperature; ``0.0`` means greedy.
        top_p: Nucleus sampling mass.
        top_k: Top-k cutoff; ``-1`` disables it.
        max_new_tokens: Hard cap on generated tokens.
        min_new_tokens: Minimum tokens before a stop condition may fire.
        repetition_penalty: Penalty applied to already-generated tokens.
        stop: Stop strings.
        stop_token_ids: Stop token ids.
        seed: Per-task RNG seed; ``None`` leaves engine defaults.
        return_logprobs: Ask the engine for per-token log-probabilities.
    """

    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_new_tokens: Optional[int] = None
    min_new_tokens: int = 0
    repetition_penalty: float = 1.0
    stop: tuple[str, ...] = ()
    stop_token_ids: tuple[int, ...] = ()
    seed: Optional[int] = None
    return_logprobs: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n <= 0:
            raise SchemaError(f"n must be positive, got {self.n}.")
        if self.temperature < 0.0:
            raise SchemaError(
                f"temperature must be non-negative, got {self.temperature}."
            )


@dataclass(kw_only=True)
class PromptSpec(SchemaBase):
    """Prompt batch for an LLM generation task.

    Either :attr:`input_ids` or :attr:`prompt_texts` must be provided; when both
    are present, :attr:`input_ids` wins and the texts are informational.

    Args:
        input_ids: Tokenized prompts.
        prompt_texts: Raw prompt strings.
        image_data: Raw image payloads (bytes or URLs) per prompt.
        multi_modal_inputs: Preprocessed multi-modal inputs per prompt.
        answers: Reference answers per prompt, echoed back in the output.
    """

    input_ids: Optional[list[list[int]]] = None
    prompt_texts: Optional[list[str]] = None
    image_data: Optional[list[list[Union[bytes, str]]]] = None
    multi_modal_inputs: Optional[list[Optional[dict]]] = None
    answers: Optional[list[Any]] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.input_ids and not self.prompt_texts:
            raise SchemaError("PromptSpec requires input_ids or prompt_texts.")

    @property
    def num_prompts(self) -> int:
        """Number of prompts in this batch."""
        if self.input_ids:
            return len(self.input_ids)
        return len(self.prompt_texts or [])


@dataclass(kw_only=True)
class EpisodeSpec(SchemaBase):
    """Episode/evaluation spec for an embodied task.

    Args:
        env_type: Registered environment type, e.g. ``"maniskill"``.
        env_ids: Task/scene identifiers to instantiate; empty means "use the env
            configuration's default".
        num_envs: Number of parallel environments to drive.
        group_size: Number of episodes sharing one initial state, used by
            group-based advantage estimators downstream.
        num_episodes: Episodes to collect in total; ``0`` means "run until the
            controller stops the task".
        num_chunk_steps: Chunk steps to run per episode; ``0`` means "until the
            env reports done".
        max_episode_steps: Episode-length cap; ``0`` defers to the env.
        seeds: Explicit per-environment seeds.
        auto_reset: Whether the env resets finished episodes automatically.
        env_config: Env-specific options passed through to the env constructor.
    """

    env_type: str
    env_ids: tuple[str, ...] = ()
    num_envs: int = 1
    group_size: int = 1
    num_episodes: int = 0
    num_chunk_steps: int = 0
    max_episode_steps: int = 0
    seeds: tuple[int, ...] = ()
    auto_reset: bool = True
    env_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.env_type:
            raise SchemaError("EpisodeSpec requires a non-empty env_type.")
        if self.num_envs <= 0:
            raise SchemaError(f"num_envs must be positive, got {self.num_envs}.")
        if self.group_size <= 0:
            raise SchemaError(f"group_size must be positive, got {self.group_size}.")
        if self.num_envs % self.group_size != 0:
            raise SchemaError(
                f"num_envs ({self.num_envs}) must be divisible by group_size "
                f"({self.group_size})."
            )
        if self.seeds and len(self.seeds) != self.num_envs:
            raise SchemaError(
                f"seeds has {len(self.seeds)} entries but num_envs is {self.num_envs}."
            )


@dataclass(kw_only=True)
class RolloutTask(SchemaBase):
    """One unit of rollout work.

    Args:
        kind: Task kind, see :class:`TaskKind`.
        task_id: Unique id; auto-generated when empty.
        mode: Train or eval behavior.
        prompts: Payload for :attr:`TaskKind.LLM_GENERATION`.
        sampling: Generation knobs for LLM tasks.
        episode: Payload for embodied task kinds.
        min_weight_version: Refuse to run before the rollout side serves at
            least this weight version; ``None`` disables the check.
        priority: Higher values are scheduled first.
        metadata: Free-form annotations.
    """

    kind: TaskKind
    task_id: str = ""
    mode: RolloutMode = RolloutMode.TRAIN
    prompts: Optional[PromptSpec] = None
    sampling: Optional[SamplingParams] = None
    episode: Optional[EpisodeSpec] = None
    min_weight_version: Optional[int] = None
    priority: int = 0
    metadata: Metadata = field(default_factory=Metadata)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.kind = TaskKind(self.kind)
        self.mode = RolloutMode(self.mode)
        if not self.task_id:
            self.task_id = str(uuid.uuid4())
        if self.kind is TaskKind.LLM_GENERATION:
            if self.prompts is None:
                raise SchemaError("LLM_GENERATION tasks require 'prompts'.")
            if self.episode is not None:
                raise SchemaError("LLM_GENERATION tasks must not carry 'episode'.")
        else:
            if self.episode is None:
                raise SchemaError(f"{self.kind.value} tasks require 'episode'.")
            if self.prompts is not None:
                raise SchemaError(f"{self.kind.value} tasks must not carry 'prompts'.")
        if self.kind is TaskKind.EMBODIED_EVAL and self.mode is not RolloutMode.EVAL:
            raise SchemaError("EMBODIED_EVAL tasks must use mode='eval'.")


class TaskSource(ABC):
    """Pull-based source of rollout tasks.

    The controller repeatedly calls :meth:`next_batch`; the implementation may
    read a dataset, drain a queue, or block on an HTTP endpoint. Sources are
    responsible for their own ordering and back-pressure.
    """

    @abstractmethod
    async def next_batch(self, max_num_tasks: int) -> list[RolloutTask]:
        """Return up to ``max_num_tasks`` tasks.

        Args:
            max_num_tasks: Upper bound on the batch size.

        Returns:
            Tasks to run; an empty list means "nothing available right now",
            which is distinct from exhaustion (see :meth:`exhausted`).
        """

    async def exhausted(self) -> bool:
        """Whether the source will never produce another task. Default: never."""
        return False

    async def report_done(self, task_id: str, error: Optional[str] = None) -> None:
        """Acknowledge completion of a task. Default: no-op.

        Args:
            task_id: Id of the finished task.
            error: Failure reason, or ``None`` on success.
        """
        del task_id, error

    async def close(self) -> None:
        """Release source resources. Default: no-op."""
