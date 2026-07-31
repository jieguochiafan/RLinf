# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Self-contained configuration for the embodied rollout system.

The training repo's env / rollout workers read ``cfg.actor.*``, ``cfg.algorithm.*``
and ``cfg.runner.*``. Those sections belong to a trainer, not to a rollout service,
so this module defines the rollout system's own schema:

===============================  ==========================================
training-repo key                rollout-system key
===============================  ==========================================
``actor.model``                  ``policy.model``
``actor.group_name``             ``rollout.weight_sync.source.group_name``
``actor.sync_weight_no_wait``    ``rollout.weight_sync.no_wait``
``runner.only_eval``             ``rollout.mode: eval``
``runner.enable_decoupled_mode`` ``rollout.decoupled``
``runner.ckpt_path``             ``policy.ckpt_path``
``runner.expert_ckpt_path``      ``rollout.plugins.dagger.expert_ckpt_path``
``algorithm.loss_type=rlt_ac``   ``rollout.plugins.rlt.enabled``
``algorithm.loss_type=dagger``   ``rollout.plugins.dagger.enabled``
``algorithm.adv_type=opd``       ``rollout.plugins.opd.enabled``
``algorithm.dagger.*``           ``rollout.plugins.dagger.*``
``algorithm.staleness_threshold``  ``rollout.staleness_threshold``
``algorithm.gamma``              ``rollout.postprocess.bootstrap.gamma``
``algorithm.bootstrap_type``     ``rollout.postprocess.bootstrap.type``
``weight_syncer``                ``rollout.weight_sync``
actor world size (traj split)    ``sink.num_shards``
===============================  ==========================================

Advantage / return computation is intentionally absent: it is a trainer concern and
is reachable only through the optional ``rollout.postprocess.trajectory_postprocessor``
hook (see :mod:`rlinf_rollout.postprocess`).
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

__all__ = [
    "DEFAULT_ROLLOUT_CONFIG",
    "RolloutConfig",
    "RolloutConfigError",
    "RolloutMode",
    "build_rollout_config",
    "validate_rollout_config",
]

#: Config sections that belong to a trainer and must not leak into the rollout system.
FORBIDDEN_SECTIONS: tuple[str, ...] = ("actor", "algorithm", "critic", "runner")


class RolloutMode:
    """Rollout service mode."""

    COLLECT = "collect"
    """Collect training trajectories and emit them through a ``TrajectorySink``."""

    EVAL = "eval"
    """Evaluate a fixed policy; no weight sync and no trajectory output."""

    ALL = (COLLECT, EVAL)


class RolloutConfigError(ValueError):
    """Raised when a rollout config is missing required keys or is inconsistent."""


#: Defaults merged under the user config by :func:`build_rollout_config`.
DEFAULT_ROLLOUT_CONFIG: dict[str, Any] = {
    "policy": {
        "ckpt_path": None,
    },
    "rollout": {
        "group_name": "rollout",
        "mode": RolloutMode.COLLECT,
        "decoupled": False,
        "pipeline_stage_num": 1,
        "rollout_queue_size": 0,
        "collect_transitions": False,
        "collect_prev_infos": True,
        "enable_offload": False,
        "enable_cuda_graph": False,
        "enable_torch_compile": False,
        "torch_compile_mode": "max-autotune-no-cudagraphs",
        "staleness_threshold": None,
        "rlt_feature_model": None,
        "expert_model": None,
        "sampling_params": None,
        "weight_sync": {
            "no_wait": False,
            "source": {
                "group_name": None,
                "src_rank": 0,
            },
        },
        "plugins": {
            "dagger": {
                "enabled": False,
                "expert_ckpt_path": None,
                "init_beta": 0.5,
                "beta_schedule": "exponential",
                "beta_min": 0.05,
                "beta_decay": 0.99,
                "only_save_expert": True,
                "online_lerobot": {
                    "enabled": False,
                    "only_success": False,
                },
            },
            "rlt": {
                "enabled": False,
                "schedule": {
                    "enable": False,
                    "warmup_post_collect_updates": 0,
                },
            },
            "opd": {"enabled": False},
        },
        "postprocess": {
            "bootstrap": {
                "enabled": True,
                "type": "standard",
                "gamma": 1.0,
            },
            "trajectory_postprocessor": None,
        },
    },
    "env": {
        "group_name": "env",
    },
    "sink": {
        "num_shards": None,
        "enabled": True,
    },
}


def build_rollout_config(cfg: DictConfig) -> DictConfig:
    """Merge rollout defaults under ``cfg`` and validate the result.

    Delegates to :func:`validate_rollout_config`, which raises
    :class:`RolloutConfigError` when the config is invalid.

    Args:
        cfg: User-provided rollout config (typically loaded from YAML by Hydra
            or ``OmegaConf.load``).

    Returns:
        A new ``DictConfig`` with defaults filled in.
    """
    merged = OmegaConf.merge(OmegaConf.create(DEFAULT_ROLLOUT_CONFIG), cfg)
    validate_rollout_config(merged)
    return merged


def validate_rollout_config(cfg: DictConfig) -> None:
    """Validate a rollout config, raising :class:`RolloutConfigError` on problems."""
    present_forbidden = [name for name in FORBIDDEN_SECTIONS if name in cfg]
    if present_forbidden:
        raise RolloutConfigError(
            f"Trainer-owned config section(s) {present_forbidden} are not accepted by "
            "the rollout system. See rlinf_rollout/config/rollout.py for the key "
            "mapping (e.g. actor.model -> policy.model, "
            "algorithm.gamma -> rollout.postprocess.bootstrap.gamma)."
        )

    mode = OmegaConf.select(cfg, "rollout.mode")
    if mode not in RolloutMode.ALL:
        raise RolloutConfigError(
            f"rollout.mode must be one of {RolloutMode.ALL}, got {mode!r}."
        )

    if OmegaConf.select(cfg, "policy.model") is None:
        raise RolloutConfigError("policy.model must be provided.")
    for key in ("model_type", "num_action_chunks", "action_dim"):
        if OmegaConf.select(cfg, f"policy.model.{key}") is None:
            raise RolloutConfigError(f"policy.model.{key} must be provided.")

    if OmegaConf.select(cfg, "rollout.model") is None:
        raise RolloutConfigError(
            "rollout.model must be provided (at least precision and model_path)."
        )

    stage_num = OmegaConf.select(cfg, "rollout.pipeline_stage_num")
    if not isinstance(stage_num, int) or stage_num < 1:
        raise RolloutConfigError(
            f"rollout.pipeline_stage_num must be a positive int, got {stage_num!r}."
        )

    train_cfg = OmegaConf.select(cfg, "env.train")
    eval_cfg = OmegaConf.select(cfg, "env.eval")
    if mode == RolloutMode.COLLECT and train_cfg is None:
        raise RolloutConfigError(
            "env.train must be provided when rollout.mode=collect."
        )
    if mode == RolloutMode.EVAL and eval_cfg is None:
        raise RolloutConfigError("env.eval must be provided when rollout.mode=eval.")

    num_action_chunks = int(OmegaConf.select(cfg, "policy.model.num_action_chunks"))
    for name, env_cfg in (("train", train_cfg), ("eval", eval_cfg)):
        if env_cfg is None:
            continue
        for key in ("env_type", "total_num_envs", "max_steps_per_rollout_epoch"):
            if OmegaConf.select(env_cfg, key) is None:
                raise RolloutConfigError(f"env.{name}.{key} must be provided.")
        total_num_envs = int(env_cfg.total_num_envs)
        if total_num_envs % stage_num != 0:
            raise RolloutConfigError(
                f"env.{name}.total_num_envs ({total_num_envs}) must be divisible by "
                f"rollout.pipeline_stage_num ({stage_num})."
            )
        max_steps = int(env_cfg.max_steps_per_rollout_epoch)
        if max_steps % num_action_chunks != 0:
            raise RolloutConfigError(
                f"env.{name}.max_steps_per_rollout_epoch ({max_steps}) must be "
                f"divisible by policy.model.num_action_chunks ({num_action_chunks})."
            )

    if mode == RolloutMode.COLLECT:
        source_group = OmegaConf.select(cfg, "rollout.weight_sync.source.group_name")
        if source_group is None:
            raise RolloutConfigError(
                "rollout.weight_sync.source.group_name must name the weight-sending "
                "worker group when rollout.mode=collect."
            )
        if OmegaConf.select(cfg, "rollout.weight_sync.type") is None:
            raise RolloutConfigError(
                "rollout.weight_sync.type must be 'bucket' or 'patch' (see "
                "rlinf_rollout.weight_sync.WeightSyncer.create)."
            )

    num_shards = OmegaConf.select(cfg, "sink.num_shards")
    if num_shards is not None and (not isinstance(num_shards, int) or num_shards < 1):
        raise RolloutConfigError(
            f"sink.num_shards must be a positive int or null, got {num_shards!r}."
        )

    bootstrap_type = OmegaConf.select(cfg, "rollout.postprocess.bootstrap.type")
    if bootstrap_type not in ("standard", "done"):
        raise RolloutConfigError(
            "rollout.postprocess.bootstrap.type must be 'standard' (bootstrap on "
            f"truncation) or 'done' (bootstrap on any done), got {bootstrap_type!r}."
        )


@dataclass
class RolloutConfig:
    """Derived, validated view over a rollout ``DictConfig``.

    Both the env worker and the HF rollout worker build this from the same config so
    batch sizes and chunk-step counts cannot drift apart.
    """

    raw: DictConfig
    mode: str
    decoupled: bool
    stage_num: int
    enable_train: bool
    enable_eval: bool
    rollout_epoch: int
    eval_rollout_epoch: int
    total_num_train_envs: int
    total_num_eval_envs: int
    train_batch_size: int
    eval_batch_size: int
    n_train_chunk_steps: int
    n_eval_chunk_steps: int
    num_action_chunks: int
    action_dim: int
    sink_num_shards: Optional[int] = None
    plugin_flags: dict[str, bool] = field(default_factory=dict)

    @classmethod
    def from_dictconfig(
        cls, cfg: DictConfig, *, validate: bool = True
    ) -> "RolloutConfig":
        """Build a :class:`RolloutConfig` from a (already merged) ``DictConfig``."""
        if validate:
            validate_rollout_config(cfg)

        mode = cfg.rollout.mode
        stage_num = int(cfg.rollout.pipeline_stage_num)
        train_cfg = OmegaConf.select(cfg, "env.train")
        eval_cfg = OmegaConf.select(cfg, "env.eval")
        enable_train = mode == RolloutMode.COLLECT and train_cfg is not None
        enable_eval = eval_cfg is not None
        num_action_chunks = int(cfg.policy.model.num_action_chunks)

        total_train = int(train_cfg.total_num_envs) if enable_train else 0
        total_eval = int(eval_cfg.total_num_envs) if enable_eval else 0

        return cls(
            raw=cfg,
            mode=mode,
            decoupled=bool(cfg.rollout.decoupled),
            stage_num=stage_num,
            enable_train=enable_train,
            enable_eval=enable_eval,
            rollout_epoch=int(train_cfg.get("rollout_epoch", 1)) if enable_train else 1,
            eval_rollout_epoch=(
                int(eval_cfg.get("rollout_epoch", 1)) if enable_eval else 1
            ),
            total_num_train_envs=total_train,
            total_num_eval_envs=total_eval,
            train_batch_size=total_train // stage_num,
            eval_batch_size=total_eval // stage_num,
            n_train_chunk_steps=(
                int(train_cfg.max_steps_per_rollout_epoch) // num_action_chunks
                if enable_train
                else 0
            ),
            n_eval_chunk_steps=(
                int(eval_cfg.max_steps_per_rollout_epoch) // num_action_chunks
                if enable_eval
                else 0
            ),
            num_action_chunks=num_action_chunks,
            action_dim=int(cfg.policy.model.action_dim),
            sink_num_shards=OmegaConf.select(cfg, "sink.num_shards"),
            plugin_flags={
                "dagger": bool(
                    OmegaConf.select(
                        cfg, "rollout.plugins.dagger.enabled", default=False
                    )
                ),
                "rlt": bool(
                    OmegaConf.select(cfg, "rollout.plugins.rlt.enabled", default=False)
                ),
                "opd": bool(
                    OmegaConf.select(cfg, "rollout.plugins.opd.enabled", default=False)
                ),
                "online_lerobot": bool(
                    OmegaConf.select(
                        cfg,
                        "rollout.plugins.dagger.online_lerobot.enabled",
                        default=False,
                    )
                ),
            },
        )

    @property
    def policy_model_cfg(self) -> DictConfig:
        """Config of the policy being rolled out (shape / action space source)."""
        return self.raw.policy.model

    @property
    def rollout_model_cfg(self) -> DictConfig:
        """Rollout-side model overrides (precision, weights path)."""
        return self.raw.rollout.model

    @property
    def env_group_name(self) -> str:
        """Worker-group name of the env workers."""
        return self.raw.env.group_name

    @property
    def rollout_group_name(self) -> str:
        """Worker-group name of the rollout workers."""
        return self.raw.rollout.group_name

    def plugin_enabled(self, name: str) -> bool:
        """Whether optional plugin ``name`` is enabled."""
        return self.plugin_flags.get(name, False)
