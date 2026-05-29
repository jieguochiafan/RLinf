import pytest

from toolkits.resource_orchestration.selector import select_best_candidate
from toolkits.resource_orchestration.types import (
    CandidateEstimate,
    CandidatePair,
    StageThroughput,
)


def _estimate(
    actor_sm: int,
    rollout_sm: int,
    rollout_time_s: float,
    training_time_s: float,
) -> CandidateEstimate:
    return CandidateEstimate(
        candidate=CandidatePair(actor_sm=actor_sm, rollout_sm=rollout_sm),
        throughput=StageThroughput(
            env_chunk_steps_per_sec=1,
            model_chunk_steps_per_sec=1,
            actor_chunk_steps_per_sec=1,
        ),
        rollout_chunk_count=1,
        rollout_time_s=rollout_time_s,
        training_time_s=training_time_s,
        epoch_time_s=max(rollout_time_s, training_time_s),
        balance_gap_s=abs(rollout_time_s - training_time_s),
        bottleneck_stage="test",
    )


def test_select_best_candidate_minimizes_epoch_time() -> None:
    slower = _estimate(actor_sm=30, rollout_sm=70, rollout_time_s=10, training_time_s=4)
    faster = _estimate(actor_sm=40, rollout_sm=60, rollout_time_s=8, training_time_s=7)

    result = select_best_candidate((slower, faster))

    assert result.selected == faster
    assert result.ranked == (faster, slower)


def test_select_best_candidate_uses_balance_within_tolerance() -> None:
    faster_unbalanced = _estimate(
        actor_sm=30,
        rollout_sm=70,
        rollout_time_s=10,
        training_time_s=1,
    )
    balanced = _estimate(
        actor_sm=40, rollout_sm=60, rollout_time_s=9.9, training_time_s=9
    )

    result = select_best_candidate((faster_unbalanced, balanced), tolerance=0.03)

    assert result.selected == balanced
    assert result.ranked == (balanced, faster_unbalanced)


def test_select_best_candidate_prefers_higher_actor_sm_after_balance_tie() -> None:
    faster_lower_actor = _estimate(
        actor_sm=40,
        rollout_sm=60,
        rollout_time_s=10,
        training_time_s=8,
    )
    slower_higher_actor = _estimate(
        actor_sm=50,
        rollout_sm=50,
        rollout_time_s=10.2,
        training_time_s=8.2,
    )

    result = select_best_candidate(
        (faster_lower_actor, slower_higher_actor), tolerance=0.03
    )

    assert faster_lower_actor.balance_gap_s == slower_higher_actor.balance_gap_s
    assert result.selected == slower_higher_actor
    assert result.ranked == (slower_higher_actor, faster_lower_actor)


def test_select_best_candidate_prefers_higher_actor_sm_after_ties() -> None:
    lower_actor = _estimate(
        actor_sm=40, rollout_sm=60, rollout_time_s=8, training_time_s=7
    )
    higher_actor = _estimate(
        actor_sm=50, rollout_sm=50, rollout_time_s=8, training_time_s=7
    )

    result = select_best_candidate((lower_actor, higher_actor))

    assert result.selected == higher_actor
    assert result.ranked == (higher_actor, lower_actor)


def test_select_best_candidate_rejects_empty_estimates() -> None:
    with pytest.raises(ValueError, match="estimate"):
        select_best_candidate(())


def test_select_best_candidate_rejects_negative_tolerance() -> None:
    estimate = _estimate(
        actor_sm=30, rollout_sm=70, rollout_time_s=10, training_time_s=4
    )

    with pytest.raises(ValueError, match="tolerance"):
        select_best_candidate((estimate,), tolerance=-0.01)
