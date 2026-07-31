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

"""Guard the Phase 3 LLM chain (engines, engine workers, HTTP server/router).

Static wherever possible so the suite runs without Ray, a GPU, SGLang or vLLM: AST
scans prove the vendored workers and engine wrappers carry no trainer coupling, while
behavioural tests cover the pure-Python seams (LLM config schema, source-topology rank
mapping, rollout-only placement helpers).
"""

import ast
from pathlib import Path

import pytest
from omegaconf import OmegaConf

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

#: Engine wrappers vendored from ``rlinf/hybrid_engines/``.
ENGINE_MODULES = (
    "engines",
    "engines.sglang",
    "engines.sglang.common",
    "engines.sglang.common.io_struct",
    "engines.sglang.common.sgl_engine",
    "engines.sglang.common.sgl_scheduler",
    "engines.sglang.common.tokenizer_manager",
    "engines.sglang.common.detokenizer_manager",
    "engines.vllm",
    "engines.vllm.vllm_0_8_5",
    "engines.vllm.vllm_0_8_5.executor",
    "engines.vllm.vllm_0_8_5.worker",
    "engines.vllm.vllm_0_8_5.weight_loader",
)

#: Workers and seams vendored from ``rlinf/workers/rollout/``.
WORKER_MODULES = (
    "workers.rollout.utils",
    "workers.rollout.sglang.sglang_worker",
    "workers.rollout.sglang.sglang_worker_server",
    "workers.rollout.vllm.vllm_worker",
    "workers.rollout.sglang_server.launcher",
    "workers.rollout.sglang_server.server_worker",
    "workers.rollout.sglang_server.router_worker",
    "workers.rollout.server.server_rollout_worker",
    "workers.rollout.server.online_router_worker",
    "weight_sync.llm",
    "utils.placement_modes",
    "utils.rollout_placement",
)

#: Config prefixes owned by a trainer; the rollout system must never read them.
FORBIDDEN_CONFIG_PREFIXES = (
    "cfg.actor.",
    "cfg.algorithm.",
    "cfg.runner.",
    "cfg.data.",
    "rollout_server.",
)

#: Dotted config selectors that must not appear as string literals.
FORBIDDEN_SELECTORS = ("actor.", "algorithm.", "runner.", "rollout_server.")

#: Every Phase 3 source file that reads config or talks to a weight sender.
PHASE3_SOURCES = tuple(
    PACKAGE_ROOT / Path(*dotted.split(".")).with_suffix(".py")
    for dotted in (
        "engines.sglang.common.sgl_scheduler",
        "engines.vllm.vllm_0_8_5.executor",
        "engines.vllm.vllm_0_8_5.worker",
        "workers.rollout.utils",
        "workers.rollout.sglang.sglang_worker",
        "workers.rollout.sglang.sglang_worker_server",
        "workers.rollout.vllm.vllm_worker",
        "workers.rollout.server.server_rollout_worker",
        "workers.rollout.server.online_router_worker",
        "workers.rollout.sglang_server.launcher",
        "workers.rollout.sglang_server.server_worker",
        "workers.rollout.sglang_server.router_worker",
        "weight_sync.llm",
        "utils.rollout_placement",
    )
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

    The migration notes in module docstrings legitimately spell out the old
    ``cfg.actor.*`` keys, so the coupling checks must look at code only.
    """
    source = path.read_text(encoding="utf-8")
    skip = _docstring_lines(ast.parse(source))
    return [
        (lineno, line.split("#", 1)[0])
        for lineno, line in enumerate(source.splitlines(), 1)
        if lineno not in skip
    ]


def _defined_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


# ---------------------------------------------------------------------------
# Vendoring completeness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dotted", ENGINE_MODULES + WORKER_MODULES)
def test_phase3_module_exists(dotted):
    assert _module_path(dotted).exists(), f"missing Phase 3 module: {dotted}"


def test_engines_replace_the_training_side_hybrid_engines_package():
    """The engine wrappers moved out of ``hybrid_engines`` into ``engines``."""
    assert not (PACKAGE_ROOT / "hybrid_engines").exists()
    assert (PACKAGE_ROOT / "engines" / "__init__.py").is_file()


@pytest.mark.parametrize("backend", ("sglang", "vllm"))
def test_engine_version_dispatch_lives_next_to_the_engine(backend):
    """Version gating moved from the worker layer into ``engines.<backend>``."""
    source = (PACKAGE_ROOT / "engines" / backend / "__init__.py").read_text(
        encoding="utf-8"
    )
    assert "importlib.metadata" in source
    assert "not supported" in source


def test_engine_patch_targets_point_at_the_vendored_modules():
    """``Patcher.add_patch`` targets are strings, so the rewrite must cover them."""
    source = (
        PACKAGE_ROOT / "engines" / "sglang" / "common" / "sgl_engine.py"
    ).read_text(encoding="utf-8")
    targets = [
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith(("rlinf", "sglang"))
    ]
    vendored = [t for t in targets if t.startswith("rlinf")]
    assert vendored, "no patch replacement targets found"
    for target in vendored:
        assert target.startswith("rlinf_rollout.engines.sglang.common."), target


def test_vllm_worker_class_path_is_the_vendored_one():
    from importlib import util

    source = (
        PACKAGE_ROOT / "workers" / "rollout" / "vllm" / "vllm_worker.py"
    ).read_text(encoding="utf-8")
    assert (
        'VLLM_WORKER_CLS = "rlinf_rollout.engines.vllm.vllm_0_8_5.worker.VLLMWorker"'
        in source
    )
    # The dotted path must actually resolve to a file in the tree.
    assert util.find_spec is not None
    assert (PACKAGE_ROOT / "engines" / "vllm" / "vllm_0_8_5" / "worker.py").is_file()


# ---------------------------------------------------------------------------
# Trainer decoupling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", PHASE3_SOURCES, ids=lambda p: p.name)
def test_llm_chain_never_reads_trainer_config_sections(path):
    offenders = [
        f"{path.name}:{lineno}: {code.strip()}"
        for lineno, code in _code_lines(path)
        for prefix in FORBIDDEN_CONFIG_PREFIXES
        if prefix in code
    ]
    assert not offenders, "trainer config access in the LLM chain:\n" + "\n".join(
        offenders
    )


@pytest.mark.parametrize("path", PHASE3_SOURCES, ids=lambda p: p.name)
def test_llm_chain_never_selects_trainer_config_keys(path):
    """``OmegaConf.select(cfg, "algorithm.…")`` is as coupled as ``cfg.algorithm.…``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        for selector in FORBIDDEN_SELECTORS:
            if node.value.startswith(selector):
                offenders.append(f"{path.name}:{node.lineno}: {node.value!r}")
    assert not offenders, "trainer config selector in the LLM chain:\n" + "\n".join(
        offenders
    )


@pytest.mark.parametrize("path", PHASE3_SOURCES, ids=lambda p: p.name)
def test_llm_chain_does_not_hardcode_a_trainer_group_name(path):
    code = "\n".join(line for _, line in _code_lines(path))
    assert "_actor_group_name" not in code
    assert "actor_weight_rank" not in code
    assert '"actor"' not in code
    assert "'actor'" not in code


@pytest.mark.parametrize("path", PHASE3_SOURCES, ids=lambda p: p.name)
def test_llm_chain_never_uses_the_actor_aware_placement(path):
    """``ModelParallelComponentPlacement`` requires actor GPUs, which we do not have."""
    code = "\n".join(line for _, line in _code_lines(path))
    assert "ModelParallelComponentPlacement" not in code
    assert "actor_tp_size" not in code
    assert "actor_world_size" not in code


def test_rank_mapping_moved_off_the_placement_object():
    """``RankMapper`` must take a ``SourceLayout``, not a placement."""
    path = PACKAGE_ROOT / "weight_sync" / "llm.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mapper = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "RankMapper"
    )
    public = {
        node.name
        for node in mapper.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    }
    assert public == {
        "source_rank_to_rollout_rank_map",
        "rollout_rank_to_source_rank_map",
    }
    for node in ast.walk(mapper):
        if isinstance(node, ast.FunctionDef):
            args = {a.arg for a in node.args.args} | {
                a.arg for a in node.args.kwonlyargs
            }
            assert "placement" not in args, f"{node.name} still takes a placement"


def test_engine_workers_expose_the_receiver_style_weight_api():
    """``sync_model_from_actor`` is replaced by ``sync_weights`` + a setup builder."""
    for dotted in (
        "workers.rollout.sglang.sglang_worker",
        "workers.rollout.vllm.vllm_worker",
    ):
        defined = _defined_names(_module_path(dotted))
        assert "sync_model_from_actor" not in defined, dotted
        assert {"sync_weights", "build_weight_sync_setup"} <= defined, dotted


def test_sglang_scheduler_takes_a_weight_sync_setup():
    path = _module_path("engines.sglang.common.sgl_scheduler")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    init = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "init_rlinf_worker"
    )
    args = [a.arg for a in init.args.args]
    assert "weight_sync_setup" in args
    assert "placement" not in args
    assert "config" not in args


def _init_rlinf_worker_params() -> list[str]:
    tree = ast.parse(
        _module_path("engines.sglang.common.sgl_scheduler").read_text("utf-8")
    )
    init = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "init_rlinf_worker"
    )
    return [a.arg for a in init.args.args]


def test_sglang_worker_matches_the_scheduler_handshake_arity():
    """The worker calls ``init_rlinf_worker`` positionally across a process boundary.

    A signature/call-site drift here would only surface at engine startup, so pin the
    contract statically.
    """
    params = _init_rlinf_worker_params()[1:]  # drop `self`
    tree = ast.parse(
        _module_path("workers.rollout.sglang.sglang_worker").read_text("utf-8")
    )
    arg_counts = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.attr
            if isinstance(func, ast.Attribute)
            else func.id
            if isinstance(func, ast.Name)
            else None
        )
        if name != "TaskMethodInput":
            continue
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        method = kwargs.get("method_name")
        if not (
            isinstance(method, ast.Constant) and method.value == "init_rlinf_worker"
        ):
            continue
        args_tuple = kwargs.get("args")
        assert isinstance(args_tuple, ast.Tuple), "args must be a literal tuple"
        arg_counts.append(len(args_tuple.elts))

    assert arg_counts, "no init_rlinf_worker handshake found in the sglang worker"
    required = sum(
        1
        for i, _ in enumerate(params)
        if i < len(params) - _num_defaults_of_init_rlinf_worker()
    )
    for count in arg_counts:
        assert required <= count <= len(params), (
            f"init_rlinf_worker takes {required}..{len(params)} args, call passes {count}"
        )


def _num_defaults_of_init_rlinf_worker() -> int:
    tree = ast.parse(
        _module_path("engines.sglang.common.sgl_scheduler").read_text("utf-8")
    )
    init = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "init_rlinf_worker"
    )
    return len(init.args.defaults)


def test_vllm_executor_passes_only_kwargs_the_worker_accepts():
    """The executor builds the worker's kwargs dict in a spawned process."""
    worker_tree = ast.parse(
        _module_path("engines.vllm.vllm_0_8_5.worker").read_text("utf-8")
    )
    worker_cls = next(
        node
        for node in ast.walk(worker_tree)
        if isinstance(node, ast.ClassDef) and node.name == "VLLMWorker"
    )
    worker_init = next(
        node
        for node in worker_cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    accepted = {a.arg for a in worker_init.args.args} - {"self"}

    exec_tree = ast.parse(
        _module_path("engines.vllm.vllm_0_8_5.executor").read_text("utf-8")
    )
    # `all_kwargs[rank] = {...}` is the dict handed to WorkerWrapperBase.init_worker.
    passed: set[str] = set()
    for node in ast.walk(exec_tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Dict)
            and isinstance(node.targets[0], ast.Subscript)
        ):
            passed = {
                key.value
                for key in node.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
            break
    assert passed, "could not find the worker kwargs dict in the executor"
    assert passed <= accepted, f"unaccepted worker kwargs: {sorted(passed - accepted)}"
    assert {"weight_sync_setup", "enforce_eager", "parent_address"} <= passed


def test_offload_and_scaling_hooks_are_preserved():
    sglang = _module_path("workers.rollout.sglang.sglang_worker").read_text("utf-8")
    assert "ReleaseMemoryOccupationReqInput" in sglang
    assert "ResumeMemoryOccupationReqInput" in sglang
    assert "RolloutScalingScheduler" in sglang

    vllm = _module_path("workers.rollout.vllm.vllm_worker").read_text("utf-8")
    assert "enable_sleep_mode=True" in vllm
    assert "offload_model_weights" in vllm
    assert "RolloutScalingScheduler" in vllm


def test_openai_compatible_http_surface_is_preserved():
    server = _module_path("workers.rollout.sglang.sglang_worker_server").read_text(
        "utf-8"
    )
    assert "/v1/chat/completions" in server
    router = _module_path("workers.rollout.server.online_router_worker").read_text(
        "utf-8"
    )
    assert "/v1/completions" in router
    launcher = _module_path("workers.rollout.sglang_server.launcher").read_text("utf-8")
    assert "launch_sglang_router_and_server" in launcher


# ---------------------------------------------------------------------------
# LLM config schema
# ---------------------------------------------------------------------------


def _minimal_llm_cfg(**overrides):
    cfg = OmegaConf.create(
        {
            "rollout": {
                "kind": "llm",
                "group_name": "rollout",
                "rollout_backend": "sglang",
                "tensor_parallel_size": 2,
                "group_size": 8,
                "batch_size": 16,
                "max_model_len": 4096,
                "max_running_requests": 64,
                "model": {
                    "model_path": "/tmp/model",
                    "precision": "bf16",
                    "trust_remote_code": True,
                },
                "sampling_params": {"max_new_tokens": 512},
                "weight_sync": {
                    "type": "bucket",
                    "bucket": {"bucket_size": 1024, "bucket_dtype": "bfloat16"},
                    "source": {
                        "group_name": "trainer",
                        "src_rank": 0,
                        "world_size": 8,
                        "presharded": True,
                        "parallel_sizes": {"tp": 4, "pp": 1},
                    },
                },
            }
        }
    )
    return OmegaConf.merge(cfg, OmegaConf.create(overrides))


def test_llm_defaults_are_merged_and_do_not_leak_embodied_keys():
    from rlinf_rollout.config import RolloutMode, build_rollout_config

    cfg = build_rollout_config(_minimal_llm_cfg())
    assert cfg.rollout.mode == RolloutMode.COLLECT
    assert cfg.rollout.placement_mode == "disaggregated"
    assert cfg.rollout.sglang.attention_backend == "triton"
    assert cfg.rollout.vllm.enable_prefix_caching is True
    assert cfg.rollout.online_router.port == 8081
    assert cfg.rollout.tracking_server.port == 8082
    # Embodied-only sections must not appear for an LLM config.
    assert "policy" not in cfg
    assert "env" not in cfg
    assert "plugins" not in cfg.rollout
    assert "postprocess" not in cfg.rollout


def test_embodied_config_still_defaults_to_the_embodied_chain():
    from rlinf_rollout.config import RolloutKind, rollout_kind

    assert rollout_kind(OmegaConf.create({"rollout": {}})) == RolloutKind.EMBODIED
    assert rollout_kind(_minimal_llm_cfg()) == RolloutKind.LLM


def test_llm_rollout_config_derives_task_shape():
    from rlinf_rollout.config import LLMRolloutConfig, build_rollout_config

    llm_cfg = LLMRolloutConfig.from_dictconfig(build_rollout_config(_minimal_llm_cfg()))
    assert llm_cfg.backend == "sglang"
    assert llm_cfg.total_tasks == 128
    assert llm_cfg.num_gpus_per_engine == 2
    assert llm_cfg.max_model_len == 4096
    assert llm_cfg.trust_remote_code is True
    assert llm_cfg.source_presharded is True
    assert llm_cfg.only_eval is False
    assert llm_cfg.model_path == "/tmp/model"
    assert llm_cfg.rollout_group_name == "rollout"


def test_llm_config_builds_the_api_v1_source_topology():
    from rlinf_rollout.api.v1 import SourceTopology
    from rlinf_rollout.config import LLMRolloutConfig, build_rollout_config

    llm_cfg = LLMRolloutConfig.from_dictconfig(build_rollout_config(_minimal_llm_cfg()))
    source = llm_cfg.weight_source_topology()
    assert isinstance(source, SourceTopology)
    assert source.group_name == "trainer"
    assert source.world_size == 8
    assert source.parallel_sizes == {"tp": 4, "pp": 1}
    assert source.src_ranks == (0,)


def test_eval_mode_has_no_weight_source():
    from rlinf_rollout.config import (
        LLMRolloutConfig,
        RolloutConfigError,
        RolloutMode,
        build_rollout_config,
    )

    cfg = build_rollout_config(_minimal_llm_cfg(rollout={"mode": RolloutMode.EVAL}))
    llm_cfg = LLMRolloutConfig.from_dictconfig(cfg)
    assert llm_cfg.only_eval
    with pytest.raises(RolloutConfigError, match="eval"):
        llm_cfg.weight_source_topology()


@pytest.mark.parametrize("section", ("actor", "algorithm", "runner", "critic"))
def test_llm_config_rejects_trainer_sections(section):
    from rlinf_rollout.config import RolloutConfigError, build_rollout_config

    cfg = _minimal_llm_cfg(**{section: {"group_name": "actor"}})
    with pytest.raises(RolloutConfigError, match=section):
        build_rollout_config(cfg)


@pytest.mark.parametrize(
    ("overrides", "match"),
    (
        ({"rollout": {"kind": "not_a_chain"}}, "rollout.kind"),
        ({"rollout": {"rollout_backend": "tensorrt"}}, "rollout_backend"),
        ({"rollout": {"model": {"model_path": None}}}, "model_path"),
        ({"rollout": {"tensor_parallel_size": 0}}, "tensor_parallel_size"),
        ({"rollout": {"group_size": 0}}, "group_size"),
        ({"rollout": {"max_model_len": 0}}, "max_model_len"),
        ({"rollout": {"sampling_params": {"max_new_tokens": None}}}, "max_new_tokens"),
        ({"rollout": {"placement_mode": "hybrid"}}, "placement_mode"),
        ({"rollout": {"sglang": {"serving_mode": "grpc"}}}, "serving_mode"),
        (
            {"rollout": {"weight_sync": {"source": {"group_name": None}}}},
            "group_name",
        ),
        (
            {"rollout": {"weight_sync": {"source": {"world_size": 0}}}},
            "world_size",
        ),
        (
            {
                "rollout": {
                    "weight_sync": {
                        "source": {"world_size": 6, "parallel_sizes": {"tp": 4}}
                    }
                }
            },
            "divisible",
        ),
    ),
)
def test_invalid_llm_configs_are_rejected(overrides, match):
    from rlinf_rollout.config import RolloutConfigError, build_rollout_config

    with pytest.raises(RolloutConfigError, match=match):
        build_rollout_config(_minimal_llm_cfg(**overrides))


# ---------------------------------------------------------------------------
# Source-topology-driven rank mapping
# ---------------------------------------------------------------------------


def _layout(**kwargs):
    from rlinf_rollout.weight_sync.llm import SourceLayout

    defaults = {"group_name": "trainer", "world_size": 8, "tp_size": 4, "pp_size": 1}
    defaults.update(kwargs)
    return SourceLayout(**defaults)


def test_source_layout_is_projected_from_the_api_v1_topology():
    from rlinf_rollout.api.v1 import SourceTopology
    from rlinf_rollout.weight_sync.llm import SourceLayout

    layout = SourceLayout.from_topology(
        SourceTopology(
            group_name="trainer",
            src_ranks=(0,),
            world_size=8,
            parallel_sizes={"tp": 4, "pp": 2},
            rank_map={"1,0": 5},
        )
    )
    assert (layout.group_name, layout.world_size) == ("trainer", 8)
    assert (layout.tp_size, layout.pp_size) == (4, 2)
    assert layout.rank_map == {(1, 0): 5}


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"group_name": ""}, "group_name"),
        ({"world_size": 0}, "world_size"),
    ),
)
def test_source_layout_rejects_an_underspecified_topology(kwargs, match):
    from rlinf_rollout.api.v1 import SourceTopology
    from rlinf_rollout.weight_sync.llm import SourceLayout

    topology = SourceTopology(
        group_name=kwargs.get("group_name", "trainer"),
        src_ranks=(0,),
        world_size=kwargs.get("world_size", 8),
        parallel_sizes={"tp": 4},
    )
    with pytest.raises(ValueError, match=match):
        SourceLayout.from_topology(topology)


def test_rank_map_keys_round_trip():
    from rlinf_rollout.weight_sync.llm import format_rollout_rank, parse_rank_map

    assert format_rollout_rank(2, 3) == "2,3"
    assert parse_rank_map({"2,3": 7}) == {(2, 3): 7}
    assert parse_rank_map({}) == {}
    with pytest.raises(ValueError, match="dp_rank"):
        parse_rank_map({"2": 7})
    with pytest.raises(ValueError, match="dp_rank"):
        parse_rank_map({"a,b": 7})


@pytest.mark.parametrize("sync_mode_name", ("COLLOCATED", "DISAGGREGATED"))
def test_rank_mapping_pairs_every_rollout_rank_with_a_source_rank(sync_mode_name):
    from rlinf_rollout.utils.placement_modes import RolloutSyncMode
    from rlinf_rollout.weight_sync.llm import RankMapper

    sync_mode = RolloutSyncMode[sync_mode_name]
    rank_map = RankMapper.rollout_rank_to_source_rank_map(
        _layout(), rollout_tp_size=2, rollout_world_size=8, sync_mode=sync_mode
    )
    # 8 accelerators / tp=2 -> 4 engines, each with 2 tp ranks.
    assert set(rank_map) == {(dp, tp) for dp in range(4) for tp in range(2)}
    assert sorted(rank_map.values()) == list(range(8))


def test_tp_1_source_maps_ranks_row_major():
    from rlinf_rollout.utils.placement_modes import RolloutSyncMode
    from rlinf_rollout.weight_sync.llm import RankMapper

    rank_map = RankMapper.source_rank_to_rollout_rank_map(
        _layout(tp_size=1, world_size=4),
        rollout_tp_size=2,
        rollout_world_size=4,
        sync_mode=RolloutSyncMode.COLLOCATED,
    )
    assert rank_map == {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}


def test_an_explicit_rank_map_bypasses_the_derivation():
    from rlinf_rollout.utils.placement_modes import RolloutSyncMode
    from rlinf_rollout.weight_sync.llm import RankMapper

    rank_map = RankMapper.rollout_rank_to_source_rank_map(
        _layout(rank_map={(0, 0): 5}),
        rollout_tp_size=2,
        rollout_world_size=8,
        sync_mode=RolloutSyncMode.DISAGGREGATED,
    )
    assert rank_map == {(0, 0): 5}


def test_engine_weight_sync_setup_reports_its_pairing():
    from rlinf_rollout.utils.placement_modes import RolloutSyncMode
    from rlinf_rollout.weight_sync.llm import EngineWeightSyncSetup

    setup = EngineWeightSyncSetup(
        source=_layout(),
        rollout_tp_size=2,
        rollout_world_size=8,
        sync_mode=RolloutSyncMode.COLLOCATED,
        presharded_weights=True,
        validate_first_sync=True,
    )
    assert setup.is_collocated
    assert setup.source_group_name == "trainer"
    assert setup.source_rank_for(0, 0) == 0
    described = setup.describe()
    assert described["source_tp_size"] == 4
    assert described["sync_mode"] == "COLLOCATED"
    assert described["presharded_weights"] is True
    assert described["explicit_rank_map"] is False
    with pytest.raises(KeyError, match="no weight source"):
        setup.source_rank_for(99, 0)


def test_engine_weight_sync_setup_is_picklable():
    """It crosses a ``spawn`` boundary into the engine processes."""
    import pickle

    from rlinf_rollout.weight_sync.llm import EngineWeightSyncSetup

    setup = EngineWeightSyncSetup(
        source=_layout(), rollout_tp_size=2, rollout_world_size=8
    )
    assert pickle.loads(pickle.dumps(setup)) == setup


def test_weight_sync_package_exports_the_llm_seams():
    import rlinf_rollout.weight_sync as weight_sync

    for name in (
        "EngineWeightSyncSetup",
        "RankMapper",
        "SourceLayout",
        "CollocateRankMapper",
        "DisaggRankMapper",
    ):
        assert name in weight_sync.__all__
        assert getattr(weight_sync, name) is not None


# ---------------------------------------------------------------------------
# Rollout-only placement
# ---------------------------------------------------------------------------


def test_placement_modes_are_importable_without_ray():
    """The enums must not drag the vendored scheduler (and Ray) in."""
    source = (PACKAGE_ROOT / "utils" / "placement_modes.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert imported == {"enum"}, imported


def test_placement_module_still_re_exports_the_enums():
    """Vendored code spells them ``from rlinf_rollout.utils.placement import ...``."""
    source = (PACKAGE_ROOT / "utils" / "placement.py").read_text(encoding="utf-8")
    assert "from rlinf_rollout.utils.placement_modes import" in source
    for name in (
        "PlacementMode",
        "RolloutSyncMode",
        "placement_mode_to_rollout_sync_mode",
    ):
        assert name in source


def test_rollout_placement_requires_no_actor_component():
    """It must read only ``rollout`` (and optionally ``reward``) from placement."""
    source = (PACKAGE_ROOT / "utils" / "rollout_placement.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    components = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_get_component_hardware"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }
    assert components == {"rollout", "reward"}


def test_rollout_placement_declares_its_mode_and_derives_stride_from_the_source():
    defined = _defined_names(PACKAGE_ROOT / "utils" / "rollout_placement.py")
    assert {
        "RolloutComponentPlacement",
        "source_tp_size",
        "_collocated_stride",
    } <= defined
    source = (PACKAGE_ROOT / "utils" / "rollout_placement.py").read_text(
        encoding="utf-8"
    )
    assert "rollout.placement_mode" in source
    assert "rollout.weight_sync.source.parallel_sizes.tp" in source


def test_rollout_placement_modes_cover_the_config_values():
    from rlinf_rollout.utils.placement_modes import PlacementMode

    # Imported lazily: the module pulls in the vendored scheduler.
    source = (PACKAGE_ROOT / "utils" / "rollout_placement.py").read_text(
        encoding="utf-8"
    )
    for name in ("collocated", "disaggregated", "auto"):
        assert f'"{name}"' in source
    assert PlacementMode.HYBRID not in (
        PlacementMode.COLLOCATED,
        PlacementMode.DISAGGREGATED,
        PlacementMode.AUTO,
    )


# ---------------------------------------------------------------------------
# Runtime smoke tests (need the optional engine dependencies)
# ---------------------------------------------------------------------------


def _missing(*modules: str) -> str:
    import importlib.util

    for module in modules:
        if importlib.util.find_spec(module) is None:
            return module
    return ""


@pytest.mark.parametrize(
    ("dotted", "requires"),
    (
        ("rlinf_rollout.weight_sync.llm", ()),
        ("rlinf_rollout.utils.placement_modes", ()),
        ("rlinf_rollout.utils.rollout_placement", ("ray",)),
        ("rlinf_rollout.workers.rollout.utils", ("ray", "torch")),
        ("rlinf_rollout.engines.sglang", ("sglang",)),
        ("rlinf_rollout.engines.vllm", ("vllm",)),
        ("rlinf_rollout.workers.rollout.sglang", ("sglang", "ray")),
        ("rlinf_rollout.workers.rollout.vllm", ("vllm", "ray")),
        ("rlinf_rollout.workers.rollout.sglang_server", ("sglang", "sglang_router")),
        ("rlinf_rollout.workers.rollout.server", ("ray", "fastapi", "uvicorn")),
    ),
)
def test_phase3_module_imports(dotted, requires):
    import importlib

    reason = _missing(*requires)
    if reason:
        pytest.skip(f"optional dependency unavailable: {reason}")
    try:
        importlib.import_module(dotted)
    except (AssertionError, ImportError, ValueError) as exc:
        # Version gates raise on import: ray>=2.47 asserts, engines.{sglang,vllm}
        # raise ValueError, and an out-of-range engine shows up as a missing symbol
        # (e.g. vllm<0.8.5 has no ``vllm.config.VllmConfig``).
        pytest.skip(f"version gate: {type(exc).__name__}: {exc}")
