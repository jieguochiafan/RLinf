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
