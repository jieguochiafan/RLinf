from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidatePair:
    """Actor/rollout MPS SM allocation candidate."""

    actor_sm: int
    rollout_sm: int

    @property
    def candidate_id(self) -> str:
        """Stable id used for reports and profile files."""
        return f"actor{self.actor_sm}_rollout{self.rollout_sm}"


@dataclass(frozen=True)
class ConfigSummary:
    """Config fields required by resource orchestration."""

    total_num_envs: int
    episode_env_steps: int
    chunk_size: int
    chunk_steps_per_env: int
    rollout_epoch: int
    rollout_chunk_count: int
    update_epoch: int
    actor_global_batch_size: int
    actor_micro_batch_size: int
    pipeline_stage_num: int
    resource_pool_mode: str


@dataclass(frozen=True)
class StageThroughput:
    """Measured throughput for each orchestration stage.

    actor_chunk_steps_per_sec is full actor training-stage throughput and
    already includes the repeated update_epoch training cost.
    """

    env_chunk_steps_per_sec: float
    model_chunk_steps_per_sec: float
    actor_chunk_steps_per_sec: float
    pipeline_samples_per_sec: float | None = None


@dataclass(frozen=True)
class CandidateEstimate:
    """Timing estimate for a resource orchestration candidate."""

    candidate: CandidatePair
    throughput: StageThroughput
    rollout_chunk_count: int
    rollout_time_s: float
    training_time_s: float
    epoch_time_s: float
    balance_gap_s: float
    bottleneck_stage: str
