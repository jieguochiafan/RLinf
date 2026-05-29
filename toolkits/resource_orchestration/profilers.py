from __future__ import annotations

from typing import Protocol

from toolkits.resource_orchestration.types import CandidatePair, StageThroughput


class ThroughputProfiler(Protocol):
    """Profiles measured stage throughput for a candidate allocation."""

    def profile(self, candidate: CandidatePair) -> StageThroughput:
        """Return measured throughput for a candidate allocation."""
