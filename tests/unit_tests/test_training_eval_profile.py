from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest
from omegaconf import OmegaConf


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


def test_run_training_profile_runs_default_mlp_policy_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from toolkits.training_eval import run

    monkeypatch.setattr(run, "TRAINING_PROFILE_RUNNER", None)

    metrics = run.run_training_profile(
        cfg=_mlp_profile_cfg(update_epoch=2),
        actor_sm=70,
        warmup_steps=1,
        measure_steps=2,
        rollout_chunk_count=8,
    )

    assert metrics["actor_chunk_steps_per_sec"] > 0.0


def test_run_training_profile_default_backend_rejects_complex_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from toolkits.training_eval import run

    cfg = _mlp_profile_cfg(update_epoch=1)
    cfg.actor.model.model_type = "openpi"
    monkeypatch.setattr(run, "TRAINING_PROFILE_RUNNER", None)

    with pytest.raises(RuntimeError, match="default training profile backend.*mlp_policy"):
        run.run_training_profile(
            cfg=cfg,
            actor_sm=70,
            warmup_steps=1,
            measure_steps=1,
            rollout_chunk_count=8,
        )


def _mlp_profile_cfg(update_epoch: int) -> Any:
    return OmegaConf.create(
        {
            "runner": {"task_type": "embodied"},
            "algorithm": {
                "update_epoch": update_epoch,
                "loss_type": "actor_critic",
                "reward_type": "action_level",
                "logprob_type": "action_level",
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.2,
                "clip_ratio_c": 3.0,
                "value_clip": 1.0,
                "huber_delta": 10.0,
            },
            "actor": {
                "global_batch_size": 4,
                "micro_batch_size": 2,
                "model": {
                    "model_type": "mlp_policy",
                    "obs_dim": 3,
                    "action_dim": 2,
                    "num_action_chunks": 1,
                    "add_value_head": True,
                    "add_q_head": False,
                },
                "optim": {
                    "lr": 1.0e-3,
                    "adam_beta1": 0.9,
                    "adam_beta2": 0.999,
                    "adam_eps": 1.0e-8,
                    "weight_decay": 0.0,
                },
            },
            "env": {"train": {"max_episode_steps": 8}},
        }
    )
