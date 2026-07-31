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

"""Guard the Phase 2 embodied chain (envs, models, workers, config, plugins).

Static wherever possible so the suite runs without Ray, a GPU or the heavy
simulator SDKs: AST scans prove the workers carry no trainer coupling and expose
only the async surface, while behavioural tests cover the pure-Python pieces
(config schema, bootstrap shaping, schema conversion, sinks, weight receiver).
"""

import ast
from pathlib import Path

import pytest
from omegaconf import OmegaConf

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

#: Env subpackages that must be vendored.
VENDORED_ENV_PACKAGES = (
    "maniskill",
    "libero",
    "robotwin",
    "isaaclab",
    "metaworld",
    "behavior",
    "calvin",
    "robocasa",
    "robocasa365",
    "realworld",
    "frankasim",
    "habitat",
    "world_model",
    "genesis",
    "embodichain",
    "roboverse",
    "d4rl",
    "polaris",
    "venv",
    "wrappers",
)

#: Model subpackages that must be vendored.
VENDORED_MODEL_PACKAGES = (
    "openvla",
    "openvla_oft",
    "openpi",
    "openpi_pytorch",
    "openpi_cfg",
    "starvla",
    "mlp_policy",
    "cnn_policy",
    "flow_policy",
    "gr00t",
    "dexbotic_pi",
    "dexbotic_dm0",
    "dreamzero",
    "lingbotvla",
    "abot_m0",
    "prismatic",
    "modules",
    "reward",
    "value_model",
)

#: Phase 2 deliverables outside envs/models.
PHASE2_MODULES = (
    "config",
    "config.models",
    "config.rollout",
    "postprocess",
    "postprocess.base",
    "postprocess.bootstrap",
    "plugins",
    "plugins.expert",
    "plugins.rlt",
    "sinks",
    "sinks.channel",
    "data.convert",
    "weight_sync.receiver",
    "workers.env.env_worker",
    "workers.env.history_manager",
    "workers.rollout.hf.huggingface_worker",
)

#: Config prefixes owned by a trainer; the rollout system must never read them.
FORBIDDEN_CONFIG_PREFIXES = ("cfg.actor.", "cfg.algorithm.", "cfg.runner.")

#: Dotted config selectors that must not appear in ``OmegaConf.select`` calls.
FORBIDDEN_SELECTORS = ("actor.", "algorithm.", "runner.")

WORKER_SOURCES = (
    PACKAGE_ROOT / "workers" / "env" / "env_worker.py",
    PACKAGE_ROOT / "workers" / "rollout" / "hf" / "huggingface_worker.py",
)


def _module_path(dotted: str) -> Path:
    base = PACKAGE_ROOT / Path(*dotted.split("."))
    return base if base.is_dir() else base.with_suffix(".py")


def _docstring_lines(tree: ast.AST) -> set[int]:
    """Line numbers occupied by docstrings / bare string expressions."""
    spans: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            spans.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return spans


def _code_lines(path: Path) -> list[tuple[int, str]]:
    """Return ``(lineno, code)`` for executable lines, minus docstrings/comments.

    The migration notes in module docstrings legitimately mention the old
    ``cfg.actor.*`` keys, so the coupling checks must look at code only.
    """
    source = path.read_text(encoding="utf-8")
    skip = _docstring_lines(ast.parse(source))
    return [
        (lineno, line.split("#", 1)[0])
        for lineno, line in enumerate(source.splitlines(), 1)
        if lineno not in skip
    ]


# ---------------------------------------------------------------------------
# Vendoring completeness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", VENDORED_ENV_PACKAGES)
def test_env_subpackage_is_vendored(name):
    assert (PACKAGE_ROOT / "envs" / name / "__init__.py").is_file()


@pytest.mark.parametrize("name", VENDORED_MODEL_PACKAGES)
def test_model_subpackage_is_vendored(name):
    assert (PACKAGE_ROOT / "models" / "embodiment" / name / "__init__.py").is_file()


@pytest.mark.parametrize("dotted", PHASE2_MODULES)
def test_phase2_module_exists(dotted):
    path = _module_path(dotted)
    assert path.exists(), f"missing Phase 2 module: {dotted}"


def test_env_registry_covers_every_supported_env_type():
    """``get_env_cls`` must branch on every member of ``SupportedEnvType``."""
    source = (PACKAGE_ROOT / "envs" / "__init__.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    enum_members = {
        node.targets[0].id
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == "SupportedEnvType"
        for node in cls.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    assert enum_members, "SupportedEnvType has no members"
    handled = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "SupportedEnvType"
    }
    assert enum_members <= handled, (
        f"unhandled env types: {sorted(enum_members - handled)}"
    )


def test_model_registry_registers_the_embodied_policies():
    source = (PACKAGE_ROOT / "models" / "__init__.py").read_text(encoding="utf-8")
    for name in (
        "OPENVLA",
        "OPENVLA_OFT",
        "OPENPI",
        "OPENPI_PYTORCH",
        "MLP_POLICY",
        "CNN_POLICY",
        "FLOW_POLICY",
        "GR00T",
        "GR00T_N1D6",
        "GR00T_N1D7",
        "DREAMZERO",
        "LINGBOTVLA",
        "ABOT_M0",
        "STARVLA",
        "CFG_MODEL",
    ):
        assert f"SupportedModel.{name}.value" in source, f"{name} is not registered"


# ---------------------------------------------------------------------------
# Trainer decoupling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", WORKER_SOURCES, ids=lambda p: p.name)
def test_workers_never_read_trainer_config_sections(path):
    offenders = [
        f"{path.name}:{lineno}: {code.strip()}"
        for lineno, code in _code_lines(path)
        for prefix in FORBIDDEN_CONFIG_PREFIXES
        if prefix in code
    ]
    assert not offenders, "trainer config access in rollout worker:\n" + "\n".join(
        offenders
    )


@pytest.mark.parametrize("path", WORKER_SOURCES, ids=lambda p: p.name)
def test_workers_never_select_trainer_config_keys(path):
    """``OmegaConf.select(cfg, "actor.…")`` is as coupled as ``cfg.actor.…``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        for selector in FORBIDDEN_SELECTORS:
            if node.value.startswith(selector):
                offenders.append(f"{path.name}:{node.lineno}: {node.value!r}")
    assert not offenders, "trainer config selector in rollout worker:\n" + "\n".join(
        offenders
    )


@pytest.mark.parametrize("path", WORKER_SOURCES, ids=lambda p: p.name)
def test_workers_do_not_hardcode_a_trainer_group_name(path):
    """The weight sender's group name must come from ``SourceTopology``."""
    code = "\n".join(line for _, line in _code_lines(path))
    assert "actor_group_name" not in code
    assert '"actor"' not in code
    assert "actor_channel" not in code


def test_env_worker_drops_the_trainer_pipeline_and_advantage_paths():
    tree = ast.parse(WORKER_SOURCES[0].read_text(encoding="utf-8"))
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for removed in (
        "compute_advantages_and_returns",
        "prepare_pipeline_batch",
        "pack_pipeline_micro_batches",
        "send_rollout_trajectories_pipeline",
        "get_actor_split_num",
        "_init_pipeline_params",
    ):
        assert removed not in defined, f"{removed} should have been dropped"


def test_rollout_worker_replaces_actor_weight_sync_with_the_receiver_api():
    tree = ast.parse(WORKER_SOURCES[1].read_text(encoding="utf-8"))
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "setup_weight_sync" not in defined
    assert "sync_model_from_actor" not in defined
    assert {"sync_weights", "build_weight_update_request"} <= defined


@pytest.mark.parametrize(
    ("path", "entrypoints"),
    (
        (WORKER_SOURCES[0], ("interact", "_interact", "_run_interact_once")),
        (WORKER_SOURCES[1], ("generate", "_generate", "generate_one_epoch")),
    ),
    ids=("env_worker", "rollout_worker"),
)
def test_worker_entrypoints_are_async_only(path, entrypoints):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    async_defs = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef)
    }
    sync_defs = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    for name in entrypoints:
        assert name in async_defs, f"{name} must be a coroutine"
        assert name not in sync_defs, f"{name} also exists as a sync def"


def test_env_worker_publishes_through_the_trajectory_sink():
    code = "\n".join(line for _, line in _code_lines(WORKER_SOURCES[0]))
    assert "TrajectorySink" in code
    assert "trajectory_to_api" in code
    assert "publish_trajectories" in code


# ---------------------------------------------------------------------------
# RolloutConfig schema
# ---------------------------------------------------------------------------


def _minimal_cfg(**overrides):
    cfg = OmegaConf.create(
        {
            "policy": {
                "model": {
                    "model_type": "openvla",
                    "num_action_chunks": 2,
                    "action_dim": 7,
                    "precision": "bf16",
                    "model_path": "/tmp/policy",
                }
            },
            "rollout": {
                "group_name": "rollout",
                "pipeline_stage_num": 2,
                "model": {"precision": "bf16", "model_path": "/tmp/policy"},
                "weight_sync": {
                    "type": "bucket",
                    "bucket": {"bucket_size": 1024, "bucket_dtype": "bfloat16"},
                    "source": {"group_name": "trainer", "src_rank": 0, "world_size": 4},
                },
            },
            "env": {
                "group_name": "env",
                "train": {
                    "env_type": "maniskill",
                    "total_num_envs": 8,
                    "max_steps_per_rollout_epoch": 8,
                    "rollout_epoch": 3,
                },
                "eval": {
                    "env_type": "maniskill",
                    "total_num_envs": 4,
                    "max_steps_per_rollout_epoch": 4,
                    "rollout_epoch": 1,
                },
            },
        }
    )
    return OmegaConf.merge(cfg, OmegaConf.create(overrides))


def test_build_rollout_config_fills_defaults():
    from rlinf_rollout.config import RolloutMode, build_rollout_config

    cfg = build_rollout_config(_minimal_cfg())
    assert cfg.rollout.mode == RolloutMode.COLLECT
    assert cfg.rollout.decoupled is False
    assert cfg.rollout.collect_prev_infos is True
    assert cfg.rollout.postprocess.bootstrap.type == "standard"
    assert cfg.rollout.plugins.dagger.enabled is False
    assert cfg.rollout.plugins.rlt.schedule.enable is False
    assert cfg.sink.num_shards is None


def test_rollout_config_derives_batch_and_chunk_counts():
    from rlinf_rollout.config import RolloutConfig, build_rollout_config

    rollout_cfg = RolloutConfig.from_dictconfig(build_rollout_config(_minimal_cfg()))
    assert rollout_cfg.stage_num == 2
    assert rollout_cfg.train_batch_size == 4
    assert rollout_cfg.eval_batch_size == 2
    assert rollout_cfg.n_train_chunk_steps == 4
    assert rollout_cfg.n_eval_chunk_steps == 2
    assert rollout_cfg.rollout_epoch == 3
    assert rollout_cfg.num_action_chunks == 2
    assert rollout_cfg.action_dim == 7
    assert rollout_cfg.enable_train and rollout_cfg.enable_eval
    assert rollout_cfg.env_group_name == "env"
    assert rollout_cfg.rollout_group_name == "rollout"


def test_eval_mode_disables_training_collection():
    from rlinf_rollout.config import RolloutConfig, RolloutMode, build_rollout_config

    cfg = build_rollout_config(_minimal_cfg(rollout={"mode": RolloutMode.EVAL}))
    rollout_cfg = RolloutConfig.from_dictconfig(cfg)
    assert not rollout_cfg.enable_train
    assert rollout_cfg.enable_eval
    assert rollout_cfg.train_batch_size == 0
    assert rollout_cfg.n_train_chunk_steps == 0


@pytest.mark.parametrize("section", ("actor", "algorithm", "runner", "critic"))
def test_trainer_sections_are_rejected(section):
    from rlinf_rollout.config import RolloutConfigError, build_rollout_config

    cfg = _minimal_cfg(**{section: {"model": {"model_type": "openvla"}}})
    with pytest.raises(RolloutConfigError, match=section):
        build_rollout_config(cfg)


@pytest.mark.parametrize(
    ("overrides", "match"),
    (
        ({"rollout": {"mode": "train"}}, "rollout.mode"),
        ({"rollout": {"pipeline_stage_num": 0}}, "pipeline_stage_num"),
        ({"env": {"train": {"total_num_envs": 7}}}, "divisible"),
        ({"env": {"train": {"max_steps_per_rollout_epoch": 7}}}, "divisible"),
        ({"rollout": {"postprocess": {"bootstrap": {"type": "wrong"}}}}, "bootstrap"),
        ({"sink": {"num_shards": 0}}, "num_shards"),
        ({"rollout": {"weight_sync": {"source": {"group_name": None}}}}, "group_name"),
    ),
)
def test_invalid_configs_are_rejected(overrides, match):
    from rlinf_rollout.config import RolloutConfigError, build_rollout_config

    with pytest.raises(RolloutConfigError, match=match):
        build_rollout_config(_minimal_cfg(**overrides))


def test_missing_policy_model_field_is_rejected():
    from rlinf_rollout.config import (
        RolloutConfigError,
        build_rollout_config,
        validate_rollout_config,
    )

    cfg = build_rollout_config(_minimal_cfg())
    del cfg.policy.model.action_dim
    with pytest.raises(RolloutConfigError, match="action_dim"):
        validate_rollout_config(cfg)


def test_supported_model_registry_is_open():
    from rlinf_rollout.config import EMBODIED_MODEL, SupportedModel

    assert SupportedModel("openvla") is SupportedModel.OPENVLA
    assert SupportedModel.OPENPI in EMBODIED_MODEL
    registered = SupportedModel.register("my_custom_policy", force=True)
    assert SupportedModel("my_custom_policy") is registered
    with pytest.raises(NotImplementedError, match="not supported"):
        SupportedModel("definitely_not_registered")


@pytest.mark.parametrize(
    ("precision", "expected"),
    (("bf16", "torch.bfloat16"), ("fp16", "torch.float16"), ("fp32", "torch.float32")),
)
def test_precision_parsing(precision, expected):
    from rlinf_rollout.config import torch_dtype_from_precision

    assert str(torch_dtype_from_precision(precision)) == expected
    assert torch_dtype_from_precision(None) is None


# ---------------------------------------------------------------------------
# Runtime smoke test (needs Ray + a torch build with DTensor + gym deps)
# ---------------------------------------------------------------------------


def _runtime_available() -> str:
    try:
        import ray  # noqa: F401
        import torch  # noqa: F401
        from torch.distributed.tensor import DTensor  # noqa: F401
    except ImportError as exc:
        return str(exc)
    return ""


@pytest.mark.parametrize(
    "dotted",
    (
        "rlinf_rollout.workers.env",
        "rlinf_rollout.workers.rollout.hf",
        "rlinf_rollout.weight_sync.receiver",
        "rlinf_rollout.data.convert",
        "rlinf_rollout.sinks",
        "rlinf_rollout.postprocess",
        "rlinf_rollout.plugins.rlt",
        "rlinf_rollout.config",
    ),
)
def test_phase2_module_imports(dotted):
    import importlib

    reason = _runtime_available()
    if reason and dotted.startswith(
        ("rlinf_rollout.workers", "rlinf_rollout.weight_sync")
    ):
        pytest.skip(f"runtime dependency unavailable: {reason}")
    importlib.import_module(dotted)


def test_worker_classes_expose_the_async_service_surface():
    import importlib
    import inspect

    reason = _runtime_available()
    if reason:
        pytest.skip(f"runtime dependency unavailable: {reason}")

    env_worker = importlib.import_module("rlinf_rollout.workers.env").AsyncEnvWorker
    rollout_worker = importlib.import_module(
        "rlinf_rollout.workers.rollout.hf"
    ).AsyncMultiStepRolloutWorker

    assert inspect.iscoroutinefunction(env_worker.interact)
    assert inspect.iscoroutinefunction(env_worker.stop)
    assert inspect.iscoroutinefunction(env_worker.publish_trajectories)
    assert inspect.iscoroutinefunction(rollout_worker.generate)
    assert inspect.iscoroutinefunction(rollout_worker.sync_weights)
    assert inspect.iscoroutinefunction(rollout_worker.evaluate)
    assert not inspect.iscoroutinefunction(rollout_worker.stop)
