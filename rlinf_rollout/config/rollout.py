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

"""Self-contained configuration for the standalone rollout system.

The training repo's env / rollout workers read ``cfg.actor.*``, ``cfg.algorithm.*``
and ``cfg.runner.*``. Those sections belong to a trainer, not to a rollout service,
so this module defines the rollout system's own schema. Two chains share it,
selected by ``rollout.kind``: ``embodied`` (Phase 2) and ``llm`` (Phase 3).

Embodied key mapping:

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

LLM key mapping:

=========================================  ==========================================
training-repo key                          rollout-system key
=========================================  ==========================================
``algorithm.sampling_params``              ``rollout.sampling_params``
``algorithm.group_size``                   ``rollout.group_size``
``data.rollout_batch_size``                ``rollout.batch_size``
``runner.seq_length``                      ``rollout.max_model_len``
``runner.resume_dir is not None``          ``rollout.validate_weight_first_sync: false``
``actor.tokenizer.trust_remote_code``      ``rollout.model.trust_remote_code``
``actor.group_name``                       ``rollout.weight_sync.source.group_name``
``actor.training_backend != "fsdp"``       ``rollout.weight_sync.source.presharded``
``placement.actor_{tp,pp}_size``           ``rollout.weight_sync.source.parallel_sizes``
``placement.actor_world_size``             ``rollout.weight_sync.source.world_size``
placement mode inferred from actor GPUs    ``rollout.placement_mode``
``rollout_server.online_router``           ``rollout.online_router``
``rollout_server.tracking_rollout``        ``rollout.tracking_server``
=========================================  ==========================================

Advantage / return computation is intentionally absent: it is a trainer concern and
is reachable only through the optional ``rollout.postprocess.trajectory_postprocessor``
hook (see :mod:`rlinf_rollout.postprocess`).
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

__all__ = [
    "DEFAULT_LLM_ROLLOUT_CONFIG",
    "DEFAULT_ROLLOUT_CONFIG",
    "LLMRolloutConfig",
    "RolloutConfig",
    "RolloutConfigError",
    "RolloutKind",
    "RolloutMode",
    "SUPPORTED_LLM_ROLLOUT_BACKENDS",
    "build_rollout_config",
    "validate_rollout_config",
]

#: Config sections that belong to a trainer and must not leak into the rollout system.
FORBIDDEN_SECTIONS: tuple[str, ...] = ("actor", "algorithm", "critic", "runner")

#: LLM engine backends the rollout system can drive.
SUPPORTED_LLM_ROLLOUT_BACKENDS: tuple[str, ...] = ("sglang", "vllm")


class RolloutKind:
    """Which rollout chain a config describes."""

    EMBODIED = "embodied"
    """Env workers + a HuggingFace policy serving action chunks."""

    LLM = "llm"
    """SGLang / vLLM engines serving token sequences."""

    ALL = (EMBODIED, LLM)


class RolloutMode:
    """Rollout service mode."""

    COLLECT = "collect"
    """Collect training trajectories and emit them through a ``TrajectorySink``."""

    EVAL = "eval"
    """Evaluate a fixed policy; no weight sync and no trajectory output."""

    ALL = (COLLECT, EVAL)


class RolloutConfigError(ValueError):
    """Raised when a rollout config is missing required keys or is inconsistent."""


#: Defaults merged under an embodied user config by :func:`build_rollout_config`.
DEFAULT_ROLLOUT_CONFIG: dict[str, Any] = {
    "policy": {
        "ckpt_path": None,
    },
    "rollout": {
        "kind": RolloutKind.EMBODIED,
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


#: Defaults merged under an LLM user config by :func:`build_rollout_config`.
#:
#: Only rollout-owned knobs live here. The engine-specific ``sglang`` / ``vllm``
#: blocks mirror the training repo's ``rollout.{sglang,vllm}`` sections verbatim so
#: existing YAML can be lifted across unchanged.
DEFAULT_LLM_ROLLOUT_CONFIG: dict[str, Any] = {
    "rollout": {
        "kind": RolloutKind.LLM,
        "group_name": "rollout",
        "mode": RolloutMode.COLLECT,
        "placement_mode": "disaggregated",
        "rollout_backend": "sglang",
        # Engine sizing.
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "gpu_memory_utilization": 0.6,
        "max_running_requests": 64,
        "cuda_graph_max_bs": 128,
        "enforce_eager": False,
        "disable_log_stats": False,
        "detokenize": False,
        "return_logprobs": False,
        # Sequence budget (was ``runner.seq_length``).
        "max_model_len": None,
        # Task shape (was ``algorithm.group_size`` / ``data.rollout_batch_size``).
        "group_size": 1,
        "batch_size": 1,
        # Was ``algorithm.sampling_params``.
        "sampling_params": {
            "do_sample": True,
            "temperature": 1.0,
            "top_k": 1000000,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "max_new_tokens": None,
        },
        # Weight validation. ``validate_weight_first_sync`` must be false whenever
        # the trainer resumes from a checkpoint (the engine's HF weights then cannot
        # match the trainer's); the rollout system cannot see that on its own.
        "validate_weight": False,
        "validate_weight_first_sync": False,
        "collect_meta_stats": False,
        "staleness_threshold": None,
        "weight_sync": {
            "no_wait": False,
            "source": {
                "group_name": None,
                "src_rank": 0,
                "world_size": 0,
                "presharded": False,
                "parallel_sizes": {"tp": 1, "pp": 1},
                "rank_map": {},
            },
        },
        "sglang": {
            "attention_backend": "triton",
            "decode_log_interval": 500000,
            "use_torch_compile": False,
            "torch_compile_max_bs": 128,
            "tool_call_parser": None,
            "serving_mode": None,
            "server": {"host": "0.0.0.0", "port": 8020},
        },
        "vllm": {
            "attention_backend": "FLASH_ATTN",
            "enable_chunked_prefill": True,
            "enable_prefix_caching": True,
            "enable_flash_infer_sampler": True,
            "max_num_batched_tokens": None,
            "torch_profiler_dir": None,
        },
        # OpenAI-compatible fan-in server (was ``rollout_server.online_router``).
        "online_router": {"host": "0.0.0.0", "port": 8081},
        # Feedback-ingest server (was ``rollout_server.tracking_rollout``).
        "tracking_server": {
            "host": "0.0.0.0",
            "port": 8082,
            "enable_dummy_data": False,
            "storage": None,
        },
    },
    "sink": {
        "num_shards": None,
        "enabled": True,
    },
}


def rollout_kind(cfg: DictConfig) -> str:
    """Return the chain ``cfg`` describes, defaulting to ``embodied``."""
    return str(OmegaConf.select(cfg, "rollout.kind", default=RolloutKind.EMBODIED))


def build_rollout_config(cfg: DictConfig) -> DictConfig:
    """Merge the defaults for ``cfg``'s chain under it and validate the result.

    The chain is selected by ``rollout.kind`` (``embodied`` by default). Delegates to
    :func:`validate_rollout_config`, which raises :class:`RolloutConfigError` when the
    config is invalid.

    Args:
        cfg: User-provided rollout config (typically loaded from YAML by Hydra
            or ``OmegaConf.load``).

    Returns:
        A new ``DictConfig`` with defaults filled in.
    """
    kind = rollout_kind(cfg)
    if kind not in RolloutKind.ALL:
        raise RolloutConfigError(
            f"rollout.kind must be one of {RolloutKind.ALL}, got {kind!r}."
        )
    defaults = (
        DEFAULT_LLM_ROLLOUT_CONFIG
        if kind == RolloutKind.LLM
        else DEFAULT_ROLLOUT_CONFIG
    )
    merged = OmegaConf.merge(OmegaConf.create(defaults), cfg)
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

    kind = rollout_kind(cfg)
    if kind not in RolloutKind.ALL:
        raise RolloutConfigError(
            f"rollout.kind must be one of {RolloutKind.ALL}, got {kind!r}."
        )

    num_shards = OmegaConf.select(cfg, "sink.num_shards")
    if num_shards is not None and (not isinstance(num_shards, int) or num_shards < 1):
        raise RolloutConfigError(
            f"sink.num_shards must be a positive int or null, got {num_shards!r}."
        )

    if kind == RolloutKind.LLM:
        _validate_llm_rollout_config(cfg, mode)
    else:
        _validate_embodied_rollout_config(cfg, mode)


def _validate_weight_sync_config(cfg: DictConfig, mode: str) -> None:
    """Check the weight-sync block, which both chains share."""
    if mode != RolloutMode.COLLECT:
        return
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


def _validate_llm_rollout_config(cfg: DictConfig, mode: str) -> None:
    """Validate the ``rollout.kind=llm`` schema."""
    backend = OmegaConf.select(cfg, "rollout.rollout_backend")
    if backend not in SUPPORTED_LLM_ROLLOUT_BACKENDS:
        raise RolloutConfigError(
            f"rollout.rollout_backend must be one of "
            f"{SUPPORTED_LLM_ROLLOUT_BACKENDS}, got {backend!r}."
        )

    if OmegaConf.select(cfg, "rollout.model.model_path") is None:
        raise RolloutConfigError("rollout.model.model_path must be provided.")

    for key in ("tensor_parallel_size", "pipeline_parallel_size"):
        value = OmegaConf.select(cfg, f"rollout.{key}")
        if not isinstance(value, int) or value < 1:
            raise RolloutConfigError(
                f"rollout.{key} must be a positive int, got {value!r}."
            )

    for key in ("group_size", "batch_size", "max_running_requests"):
        value = OmegaConf.select(cfg, f"rollout.{key}")
        if not isinstance(value, int) or value < 1:
            raise RolloutConfigError(
                f"rollout.{key} must be a positive int, got {value!r}."
            )

    max_model_len = OmegaConf.select(cfg, "rollout.max_model_len")
    if max_model_len is not None and (
        not isinstance(max_model_len, int) or max_model_len < 1
    ):
        raise RolloutConfigError(
            f"rollout.max_model_len must be a positive int or null, got "
            f"{max_model_len!r}."
        )

    sampling_params = OmegaConf.select(cfg, "rollout.sampling_params")
    if sampling_params is None:
        raise RolloutConfigError(
            "rollout.sampling_params must be provided for LLM rollout (it replaces "
            "the trainer's algorithm.sampling_params)."
        )
    max_new_tokens = OmegaConf.select(sampling_params, "max_new_tokens")
    if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise RolloutConfigError(
            f"rollout.sampling_params.max_new_tokens must be a positive int, got "
            f"{max_new_tokens!r}."
        )

    placement_mode = OmegaConf.select(cfg, "rollout.placement_mode")
    if placement_mode not in ("collocated", "disaggregated", "auto"):
        raise RolloutConfigError(
            "rollout.placement_mode must be 'collocated', 'disaggregated' or 'auto', "
            f"got {placement_mode!r}."
        )

    serving_mode = OmegaConf.select(cfg, "rollout.sglang.serving_mode", default=None)
    if serving_mode is not None and serving_mode != "worker_http":
        raise RolloutConfigError(
            f"rollout.sglang.serving_mode must be null or 'worker_http', got "
            f"{serving_mode!r}."
        )

    _validate_weight_sync_config(cfg, mode)
    if mode == RolloutMode.COLLECT:
        source_world_size = OmegaConf.select(
            cfg, "rollout.weight_sync.source.world_size", default=0
        )
        if not isinstance(source_world_size, int) or source_world_size < 1:
            raise RolloutConfigError(
                "rollout.weight_sync.source.world_size must be a positive int: the "
                "sender's rank layout cannot be derived without it (it replaces "
                "placement.actor_world_size)."
            )
        source_tp = OmegaConf.select(
            cfg, "rollout.weight_sync.source.parallel_sizes.tp", default=1
        )
        if not isinstance(source_tp, int) or source_tp < 1:
            raise RolloutConfigError(
                "rollout.weight_sync.source.parallel_sizes.tp must be a positive int "
                "(it replaces placement.actor_tp_size)."
            )
        if source_world_size % source_tp != 0:
            raise RolloutConfigError(
                f"rollout.weight_sync.source.world_size ({source_world_size}) must be "
                f"divisible by parallel_sizes.tp ({source_tp})."
            )


def _validate_embodied_rollout_config(cfg: DictConfig, mode: str) -> None:
    """Validate the ``rollout.kind=embodied`` schema."""
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

    _validate_weight_sync_config(cfg, mode)

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


@dataclass
class LLMRolloutConfig:
    """Derived, validated view over an ``rollout.kind=llm`` ``DictConfig``.

    Both engine workers and the HTTP server/router layer build this from the same
    config, so sampling params, sequence budgets and the weight-source topology
    cannot drift apart between them.
    """

    raw: DictConfig
    mode: str
    backend: str
    placement_mode: str
    tensor_parallel_size: int
    pipeline_parallel_size: int
    group_size: int
    batch_size: int
    max_running_requests: int
    return_logprobs: bool
    detokenize: bool
    enforce_eager: bool
    max_model_len: Optional[int] = None
    validate_weight: bool = False
    validate_weight_first_sync: bool = False
    collect_meta_stats: bool = False
    staleness_threshold: Optional[int] = None
    sink_num_shards: Optional[int] = None

    @classmethod
    def from_dictconfig(
        cls, cfg: DictConfig, *, validate: bool = True
    ) -> "LLMRolloutConfig":
        """Build an :class:`LLMRolloutConfig` from a (already merged) ``DictConfig``."""
        if validate:
            validate_rollout_config(cfg)
        if rollout_kind(cfg) != RolloutKind.LLM:
            raise RolloutConfigError(
                f"LLMRolloutConfig requires rollout.kind={RolloutKind.LLM!r}, got "
                f"{rollout_kind(cfg)!r}."
            )

        rollout = cfg.rollout
        return cls(
            raw=cfg,
            mode=str(rollout.mode),
            backend=str(rollout.rollout_backend),
            placement_mode=str(rollout.placement_mode),
            tensor_parallel_size=int(rollout.tensor_parallel_size),
            pipeline_parallel_size=int(rollout.pipeline_parallel_size),
            group_size=int(rollout.group_size),
            batch_size=int(rollout.batch_size),
            max_running_requests=int(rollout.max_running_requests),
            return_logprobs=bool(rollout.return_logprobs),
            detokenize=bool(rollout.detokenize),
            enforce_eager=bool(rollout.enforce_eager),
            max_model_len=OmegaConf.select(cfg, "rollout.max_model_len"),
            validate_weight=bool(
                OmegaConf.select(cfg, "rollout.validate_weight", default=False)
            ),
            validate_weight_first_sync=bool(
                OmegaConf.select(
                    cfg, "rollout.validate_weight_first_sync", default=False
                )
            ),
            collect_meta_stats=bool(
                OmegaConf.select(cfg, "rollout.collect_meta_stats", default=False)
            ),
            staleness_threshold=OmegaConf.select(cfg, "rollout.staleness_threshold"),
            sink_num_shards=OmegaConf.select(cfg, "sink.num_shards"),
        )

    @property
    def only_eval(self) -> bool:
        """Whether the service serves a fixed policy and never syncs weights."""
        return self.mode == RolloutMode.EVAL

    @property
    def model_cfg(self) -> DictConfig:
        """Engine model config (path, precision, ``trust_remote_code``)."""
        return self.raw.rollout.model

    @property
    def model_path(self) -> str:
        """Local path or hub id of the served model."""
        return str(self.raw.rollout.model.model_path)

    @property
    def trust_remote_code(self) -> bool:
        """Whether the tokenizer / model may execute repo code."""
        return bool(
            OmegaConf.select(self.raw, "rollout.model.trust_remote_code", default=False)
        )

    @property
    def rollout_group_name(self) -> str:
        """Worker-group name of the engine workers."""
        return str(self.raw.rollout.group_name)

    @property
    def sampling_params_cfg(self) -> DictConfig:
        """Rollout-owned sampling params (was ``algorithm.sampling_params``)."""
        return self.raw.rollout.sampling_params

    @property
    def total_tasks(self) -> int:
        """Sequences one rollout round produces (``batch_size * group_size``)."""
        return self.batch_size * self.group_size

    @property
    def num_gpus_per_engine(self) -> int:
        """Accelerators one engine process occupies."""
        return self.tensor_parallel_size * self.pipeline_parallel_size

    @property
    def source_presharded(self) -> bool:
        """Whether the sender ships per-TP-rank shards rather than full tensors."""
        return bool(
            OmegaConf.select(
                self.raw, "rollout.weight_sync.source.presharded", default=False
            )
        )

    def weight_source_topology(self):
        """Build the api/v1 ``SourceTopology`` describing the weight sender.

        Returns:
            :class:`rlinf_rollout.api.v1.SourceTopology`.

        Raises:
            RolloutConfigError: When called on an eval-only config, which has no
                weight sender.
        """
        from rlinf_rollout.api.v1 import SourceTopology

        if self.only_eval:
            raise RolloutConfigError(
                "weight sync is disabled (rollout.mode=eval); there is no source "
                "topology to build."
            )
        source_cfg = self.raw.rollout.weight_sync.source
        parallel_sizes = OmegaConf.select(source_cfg, "parallel_sizes", default=None)
        rank_map = OmegaConf.select(source_cfg, "rank_map", default=None)
        return SourceTopology(
            group_name=str(source_cfg.group_name),
            src_ranks=(int(OmegaConf.select(source_cfg, "src_rank", default=0)),),
            world_size=int(OmegaConf.select(source_cfg, "world_size", default=0)),
            parallel_sizes=(
                {
                    str(k): int(v)
                    for k, v in OmegaConf.to_container(parallel_sizes).items()
                }
                if parallel_sizes is not None
                else {}
            ),
            rank_map=(
                {str(k): int(v) for k, v in OmegaConf.to_container(rank_map).items()}
                if rank_map is not None
                else {}
            ),
        )
