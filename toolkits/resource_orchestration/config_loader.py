from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from rlinf.scheduler.resource_pool.bindings import WorkerResourceBinding
from toolkits.resource_orchestration.types import ConfigSummary

ComponentBindings = dict[str, list[WorkerResourceBinding]]


def _select_int(cfg: DictConfig, path: str, default: int | None = None) -> int:
    value = OmegaConf.select(cfg, path, default=default)
    if value is None:
        raise ValueError(f"missing required config value: {path}")
    return int(value)


def load_hydra_config(
    config_path: str,
    config_name: str,
    overrides: tuple[str, ...] = (),
) -> DictConfig:
    """Load a Hydra config from an absolute or relative config directory."""
    abs_config_path = str(Path(config_path).resolve())
    with initialize_config_dir(version_base="1.1", config_dir=abs_config_path):
        return compose(config_name=config_name, overrides=list(overrides))


def build_config_summary(cfg: DictConfig) -> ConfigSummary:
    """Extract orchestration fields from a loaded RLinf config."""
    mode = str(OmegaConf.select(cfg, "cluster.resource_pool.gpu.mode", default=""))
    if mode != "mps":
        raise ValueError(
            f"profile-based orchestration v1 requires MPS resource_pool, got {mode!r}"
        )

    total_num_envs = _select_int(cfg, "env.train.total_num_envs")
    episode_env_steps = _select_int(
        cfg,
        "env.train.max_steps_per_rollout_epoch",
        default=OmegaConf.select(cfg, "env.train.max_episode_steps"),
    )
    chunk_size = _select_int(cfg, "actor.model.num_action_chunks")
    if chunk_size <= 0:
        raise ValueError(f"actor.model.num_action_chunks must be > 0, got {chunk_size}")
    if episode_env_steps % chunk_size != 0:
        raise ValueError(
            "env.train.max_steps_per_rollout_epoch must be divisible by "
            "actor.model.num_action_chunks"
        )

    chunk_steps_per_env = episode_env_steps // chunk_size
    rollout_epoch = _select_int(cfg, "algorithm.rollout_epoch", default=1)
    update_epoch = _select_int(cfg, "algorithm.update_epoch", default=1)

    return ConfigSummary(
        total_num_envs=total_num_envs,
        episode_env_steps=episode_env_steps,
        chunk_size=chunk_size,
        chunk_steps_per_env=chunk_steps_per_env,
        rollout_epoch=rollout_epoch,
        rollout_chunk_count=total_num_envs * rollout_epoch * chunk_steps_per_env,
        update_epoch=update_epoch,
        actor_global_batch_size=_select_int(cfg, "actor.global_batch_size"),
        actor_micro_batch_size=_select_int(cfg, "actor.micro_batch_size"),
        pipeline_stage_num=_select_int(cfg, "rollout.pipeline_stage_num", default=1),
        resource_pool_mode=mode,
    )


def load_plan_bindings(plan_path: str | Path) -> ComponentBindings:
    """Load worker resource bindings from an allocation plan JSON file."""
    payload = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    bindings: dict[str, list[WorkerResourceBinding]] = defaultdict(list)
    for item in payload.get("bindings", []):
        binding = WorkerResourceBinding.from_json(json.dumps(item))
        bindings[binding.component].append(binding)
    return {
        component: sorted(component_bindings, key=lambda binding: binding.rank)
        for component, component_bindings in sorted(bindings.items())
    }


def load_base_bindings(cfg: DictConfig, base_plan: str | None) -> ComponentBindings:
    """Load the base binding plan from CLI input or config."""
    plan_path = base_plan or OmegaConf.select(
        cfg, "cluster.resource_pool.allocation_plan_path"
    )
    if not plan_path:
        raise ValueError(
            "base resource bindings require --base-plan or "
            "cluster.resource_pool.allocation_plan_path"
        )
    return load_plan_bindings(plan_path)
