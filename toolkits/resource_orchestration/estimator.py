from __future__ import annotations

import math

from toolkits.resource_orchestration.types import (
    CandidateEstimate,
    CandidatePair,
    ConfigSummary,
    StageThroughput,
)


def _require_positive(throughput: StageThroughput) -> None:
    values = (
        throughput.env_chunk_steps_per_sec,
        throughput.model_chunk_steps_per_sec,
        throughput.actor_chunk_steps_per_sec,
    )
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("stage throughput values must be finite and positive")


def estimate_candidate(
    candidate: CandidatePair,
    summary: ConfigSummary,
    throughput: StageThroughput,
) -> CandidateEstimate:
    """Estimate rollout/training timing for a candidate allocation."""
    _require_positive(throughput)

    rollout_bottleneck_tput = min(
        throughput.env_chunk_steps_per_sec,
        throughput.model_chunk_steps_per_sec,
    )
    bottleneck_stage = (
        "env"
        if throughput.env_chunk_steps_per_sec <= throughput.model_chunk_steps_per_sec
        else "model"
    )
    rollout_time_s = summary.rollout_chunk_count / rollout_bottleneck_tput
    training_time_s = summary.rollout_chunk_count / throughput.actor_chunk_steps_per_sec

    return CandidateEstimate(
        candidate=candidate,
        throughput=throughput,
        rollout_chunk_count=summary.rollout_chunk_count,
        rollout_time_s=rollout_time_s,
        training_time_s=training_time_s,
        epoch_time_s=max(rollout_time_s, training_time_s),
        balance_gap_s=abs(rollout_time_s - training_time_s),
        bottleneck_stage=bottleneck_stage,
    )
