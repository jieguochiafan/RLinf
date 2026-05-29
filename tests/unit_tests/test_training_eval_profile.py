from __future__ import annotations

import builtins
import importlib
import math
import sys
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


def test_run_training_profile_injected_runner_does_not_require_torch_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "toolkits.training_eval.run"
    existing_module = sys.modules.pop(module_name, None)
    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    try:
        run = importlib.import_module(module_name)
        monkeypatch.setattr(
            run,
            "TRAINING_PROFILE_RUNNER",
            lambda **_kwargs: {"actor_chunk_steps_per_sec": 3.0},
        )

        metrics = run.run_training_profile(
            cfg=SimpleNamespace(),
            actor_sm=70,
            warmup_steps=1,
            measure_steps=1,
            rollout_chunk_count=4,
        )
    finally:
        sys.modules.pop(module_name, None)
        if existing_module is not None:
            sys.modules[module_name] = existing_module

    assert metrics == {"actor_chunk_steps_per_sec": 3.0}


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


def test_run_training_profile_restores_mps_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from toolkits.training_eval import run

    monkeypatch.setattr(run, "TRAINING_PROFILE_RUNNER", None)
    monkeypatch.setenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "25")
    observed_mps: list[str | None] = []

    def fake_profile_iteration(*_args: Any, **_kwargs: Any) -> None:
        observed_mps.append(run.os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"))

    monkeypatch.setattr(run, "_run_profile_iteration", fake_profile_iteration)

    metrics = run.run_training_profile(
        cfg=_mlp_profile_cfg(update_epoch=1),
        actor_sm=70,
        warmup_steps=0,
        measure_steps=1,
        rollout_chunk_count=4,
    )

    assert metrics["actor_chunk_steps_per_sec"] > 0.0
    assert observed_mps == ["70"]
    assert run.os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] == "25"


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


@pytest.mark.parametrize(
    ("global_batch_size", "micro_batch_size", "rollout_chunk_count"),
    [
        (3, 2, 6),
        (4, 2, 6),
    ],
)
def test_run_training_profile_rejects_non_production_batch_partitioning(
    monkeypatch: pytest.MonkeyPatch,
    global_batch_size: int,
    micro_batch_size: int,
    rollout_chunk_count: int,
) -> None:
    from toolkits.training_eval import run

    cfg = _mlp_profile_cfg(update_epoch=1)
    cfg.actor.global_batch_size = global_batch_size
    cfg.actor.micro_batch_size = micro_batch_size
    monkeypatch.setattr(run, "TRAINING_PROFILE_RUNNER", None)

    with pytest.raises(ValueError, match="batch.*divisible|rollout_chunk_count"):
        run.run_training_profile(
            cfg=cfg,
            actor_sm=70,
            warmup_steps=0,
            measure_steps=1,
            rollout_chunk_count=rollout_chunk_count,
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
