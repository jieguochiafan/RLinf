from __future__ import annotations

from rlinf.scheduler.resource_pool.gpu_binding import validate_sm_percent
from toolkits.resource_orchestration.types import CandidatePair


def _validate_pair(pair: CandidatePair) -> CandidatePair:
    validate_sm_percent(pair.actor_sm)
    validate_sm_percent(pair.rollout_sm)
    if pair.actor_sm + pair.rollout_sm > 100:
        raise ValueError(
            "candidate pair exceeds one GPU SM budget: "
            f"actor_sm={pair.actor_sm}, rollout_sm={pair.rollout_sm}"
        )
    return pair


def default_candidate_pairs() -> tuple[CandidatePair, ...]:
    """Return default complementary actor/rollout MPS pairs."""
    return tuple(
        CandidatePair(actor_sm=actor_sm, rollout_sm=100 - actor_sm)
        for actor_sm in range(20, 90, 10)
    )


def parse_candidate_pairs(raw: str | None) -> tuple[CandidatePair, ...]:
    """Parse comma-separated actor:rollout SM candidate pairs."""
    if raw is None or not raw.strip():
        return default_candidate_pairs()

    pairs: list[CandidatePair] = []
    for item in raw.split(","):
        token = item.strip()
        parts = token.split(":")
        if len(parts) != 2:
            raise ValueError(f"candidate pair must use actor:rollout syntax, got {token!r}")
        try:
            actor_sm = int(parts[0])
            rollout_sm = int(parts[1])
        except ValueError as exc:
            raise ValueError(f"candidate pair values must be integers, got {token!r}") from exc
        pairs.append(_validate_pair(CandidatePair(actor_sm=actor_sm, rollout_sm=rollout_sm)))

    if not pairs:
        raise ValueError("at least one candidate pair is required")
    return tuple(pairs)
