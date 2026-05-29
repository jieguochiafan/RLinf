from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest


def test_run_training_profile_is_importable() -> None:
    from toolkits.training_eval.run import run_training_profile

    assert callable(run_training_profile)


def test_run_training_profile_calls_injected_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from toolkits.training_eval import run

    cfg = SimpleNamespace(name="cfg")
    calls: list[dict[str, Any]] = []

    def runner(**kwargs: Any) -> dict[str, float]:
        calls.append(kwargs)
        return {"actor_chunk_steps_per_sec": 12.5}

    monkeypatch.setattr(run, "TRAINING_PROFILE_RUNNER", runner)

    metrics = run.run_training_profile(
        cfg=cfg,
        actor_sm=70,
        warmup_steps=1,
        measure_steps=3,
        rollout_chunk_count=40,
    )

    assert calls == [
        {
            "cfg": cfg,
            "actor_sm": 70,
            "warmup_steps": 1,
            "measure_steps": 3,
            "rollout_chunk_count": 40,
        }
    ]
    assert metrics == {"actor_chunk_steps_per_sec": 12.5}


@pytest.mark.parametrize("throughput", [0.0, -1.0, math.inf, math.nan])
def test_run_training_profile_rejects_invalid_throughput(
    monkeypatch: pytest.MonkeyPatch,
    throughput: float,
) -> None:
    from toolkits.training_eval import run

    def runner(**_kwargs: Any) -> dict[str, float]:
        return {"actor_chunk_steps_per_sec": throughput}

    monkeypatch.setattr(run, "TRAINING_PROFILE_RUNNER", runner)

    with pytest.raises(ValueError, match="actor_chunk_steps_per_sec.*finite.*positive"):
        run.run_training_profile(
            cfg=SimpleNamespace(),
            actor_sm=70,
            warmup_steps=1,
            measure_steps=3,
            rollout_chunk_count=40,
        )


def test_run_training_profile_raises_without_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from toolkits.training_eval import run

    monkeypatch.setattr(run, "TRAINING_PROFILE_RUNNER", None)

    with pytest.raises(
        RuntimeError,
        match="actor training profiling backend.*not implemented",
    ):
        run.run_training_profile(
            cfg=SimpleNamespace(),
            actor_sm=70,
            warmup_steps=1,
            measure_steps=3,
            rollout_chunk_count=40,
        )
