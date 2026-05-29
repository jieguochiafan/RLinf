import pytest

from toolkits.resource_orchestration.estimator import estimate_candidate
from toolkits.resource_orchestration.types import (
    CandidatePair,
    ConfigSummary,
    StageThroughput,
)


def _summary(rollout_chunk_count: int = 128) -> ConfigSummary:
    return ConfigSummary(
        total_num_envs=8,
        episode_env_steps=40,
        chunk_size=5,
        chunk_steps_per_env=8,
        rollout_epoch=2,
        rollout_chunk_count=rollout_chunk_count,
        update_epoch=4,
        actor_global_batch_size=16,
        actor_micro_batch_size=4,
        pipeline_stage_num=2,
        resource_pool_mode="mps",
    )


def test_estimate_candidate_uses_slowest_rollout_stage() -> None:
    candidate = CandidatePair(actor_sm=30, rollout_sm=70)
    throughput = StageThroughput(
        env_chunk_steps_per_sec=32,
        model_chunk_steps_per_sec=16,
        actor_chunk_steps_per_sec=64,
    )

    estimate = estimate_candidate(candidate, _summary(), throughput)

    assert estimate.candidate == candidate
    assert estimate.throughput == throughput
    assert estimate.rollout_chunk_count == 128
    assert estimate.rollout_time_s == 8
    assert estimate.training_time_s == 2
    assert estimate.epoch_time_s == 8
    assert estimate.balance_gap_s == 6
    assert estimate.bottleneck_stage == "model"


def test_estimate_candidate_rejects_zero_throughput() -> None:
    throughput = StageThroughput(
        env_chunk_steps_per_sec=0,
        model_chunk_steps_per_sec=16,
        actor_chunk_steps_per_sec=64,
    )

    with pytest.raises(ValueError, match="throughput"):
        estimate_candidate(CandidatePair(actor_sm=30, rollout_sm=70), _summary(), throughput)


@pytest.mark.parametrize("invalid_tput", [float("nan"), float("inf")])
def test_estimate_candidate_rejects_non_finite_throughput(
    invalid_tput: float,
) -> None:
    throughput = StageThroughput(
        env_chunk_steps_per_sec=32,
        model_chunk_steps_per_sec=invalid_tput,
        actor_chunk_steps_per_sec=64,
    )

    with pytest.raises(ValueError, match="throughput"):
        estimate_candidate(CandidatePair(actor_sm=30, rollout_sm=70), _summary(), throughput)
