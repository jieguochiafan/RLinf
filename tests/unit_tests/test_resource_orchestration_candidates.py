import pytest

from toolkits.resource_orchestration.candidates import (
    default_candidate_pairs,
    parse_candidate_pairs,
)
from toolkits.resource_orchestration.types import CandidatePair


def test_parse_candidate_pairs_accepts_actor_rollout_pairs() -> None:
    assert parse_candidate_pairs("30:70,40:60") == (
        CandidatePair(actor_sm=30, rollout_sm=70),
        CandidatePair(actor_sm=40, rollout_sm=60),
    )


def test_default_candidate_pairs_are_complementary() -> None:
    assert default_candidate_pairs() == (
        CandidatePair(actor_sm=20, rollout_sm=80),
        CandidatePair(actor_sm=30, rollout_sm=70),
        CandidatePair(actor_sm=40, rollout_sm=60),
        CandidatePair(actor_sm=50, rollout_sm=50),
        CandidatePair(actor_sm=60, rollout_sm=40),
        CandidatePair(actor_sm=70, rollout_sm=30),
        CandidatePair(actor_sm=80, rollout_sm=20),
    )


@pytest.mark.parametrize("raw", ["30", "30-70", "30:70:0", "15:85", "60:50"])
def test_parse_candidate_pairs_rejects_invalid_values(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_candidate_pairs(raw)
