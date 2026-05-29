from __future__ import annotations

from toolkits.resource_orchestration.types import CandidateEstimate, SelectionResult


def select_best_candidate(
    estimates: tuple[CandidateEstimate, ...] | list[CandidateEstimate],
    tolerance: float = 0.03,
) -> SelectionResult:
    """Select the best resource orchestration candidate estimate."""
    if not estimates:
        raise ValueError("estimates must not be empty")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    best_epoch_time_s = min(estimate.epoch_time_s for estimate in estimates)

    def sort_key(estimate: CandidateEstimate) -> tuple[float, float, float, int]:
        within_tolerance = estimate.epoch_time_s <= best_epoch_time_s * (1 + tolerance)
        epoch_penalty = 0.0 if within_tolerance else estimate.epoch_time_s
        return (
            epoch_penalty,
            estimate.balance_gap_s,
            -estimate.candidate.actor_sm,
            estimate.epoch_time_s,
        )

    ranked = tuple(sorted(estimates, key=sort_key))
    return SelectionResult(selected=ranked[0], ranked=ranked)
