from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

TrainingProfileRunner = Callable[..., Mapping[str, float]]

TRAINING_PROFILE_RUNNER: TrainingProfileRunner | None = None


def run_training_profile(
    cfg: Any,
    actor_sm: int,
    warmup_steps: int,
    measure_steps: int,
    rollout_chunk_count: int,
) -> dict[str, float]:
    """Profile actor training throughput through an injected backend.

    This module intentionally does not synthesize production throughput. Until
    a real actor profiling backend is wired in, callers must inject one by
    setting TRAINING_PROFILE_RUNNER.
    """
    if TRAINING_PROFILE_RUNNER is None:
        raise RuntimeError(
            "actor training profiling backend is not implemented; configure "
            "toolkits.training_eval.run.TRAINING_PROFILE_RUNNER or wire a real "
            "training profiling backend before running profile-based resource "
            "orchestration"
        )

    metrics = TRAINING_PROFILE_RUNNER(
        cfg=cfg,
        actor_sm=actor_sm,
        warmup_steps=warmup_steps,
        measure_steps=measure_steps,
        rollout_chunk_count=rollout_chunk_count,
    )
    try:
        actor_chunk_steps_per_sec = float(metrics["actor_chunk_steps_per_sec"])
    except KeyError as exc:
        raise ValueError(
            "training profile metrics must include actor_chunk_steps_per_sec"
        ) from exc

    if (
        not math.isfinite(actor_chunk_steps_per_sec)
        or actor_chunk_steps_per_sec <= 0.0
    ):
        raise ValueError(
            "actor_chunk_steps_per_sec must be finite and positive"
        )

    return {"actor_chunk_steps_per_sec": actor_chunk_steps_per_sec}
