from __future__ import annotations

from pathlib import Path

from tools.plot_worker_sm_timeline import infer_phases


def test_infer_phases_includes_training_for_actor_rank_dirs() -> None:
    assert infer_phases(Path("rank0")) == ["fwd", "loss", "backward", "optimizer.step"]


def test_infer_phases_keeps_env_and_generation_dirs() -> None:
    assert infer_phases(Path("env_rank3")) == ["env.step"]
    assert infer_phases(Path("rollout_rank7")) == ["generation"]
