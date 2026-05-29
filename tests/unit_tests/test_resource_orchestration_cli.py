import json
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

import toolkits.resource_orchestration.run as run
from rlinf.scheduler.resource_pool.bindings import (
    CpuBinding,
    GpuBinding,
    WorkerResourceBinding,
)
from toolkits.resource_orchestration.config_loader import (
    load_base_bindings,
    load_plan_bindings,
)
from toolkits.resource_orchestration.run import parse_args


def _binding(component: str, rank: int) -> WorkerResourceBinding:
    return WorkerResourceBinding(
        component=component,
        rank=rank,
        cluster_node_rank=0,
        node_group_label="default",
        cpu=CpuBinding(process_cpu_cores=(rank,)),
        gpu=GpuBinding(
            mode="mps",
            sm_percent=50,
            visible_devices=("0",),
            parent_gpu=0,
        ),
    )


def _write_plan(path, bindings: list[WorkerResourceBinding]) -> None:
    path.write_text(
        json.dumps(
            {"bindings": [json.loads(binding.to_json()) for binding in bindings]}
        ),
        encoding="utf-8",
    )


def test_parse_args_accepts_required_cli_and_overrides() -> None:
    args = parse_args(
        [
            "--config-path",
            "examples/embodiment/config",
            "--config-name",
            "libero",
            "--output-dir",
            "outputs/orchestration",
            "--plan-output",
            "outputs/plan.json",
            "--override",
            "actor.micro_batch_size=2",
        ]
    )

    assert args.config_path == "examples/embodiment/config"
    assert args.config_name == "libero"
    assert args.output_dir == "outputs/orchestration"
    assert args.plan_output == "outputs/plan.json"
    assert args.override == ["actor.micro_batch_size=2"]
    assert args.candidate_pairs is None
    assert args.warmup_steps == 5
    assert args.measure_steps == 20
    assert args.base_plan is None
    assert args.selection_tolerance == 0.03


def test_main_wires_real_orchestrator_with_profiler(monkeypatch) -> None:
    cfg = OmegaConf.create(
        {
            "cluster": {
                "resource_pool": {
                    "gpu": {
                        "mode": "mps",
                    }
                }
            },
            "env": {
                "train": {
                    "total_num_envs": 4,
                    "max_steps_per_rollout_epoch": 8,
                }
            },
            "actor": {
                "global_batch_size": 4,
                "micro_batch_size": 2,
                "model": {
                    "num_action_chunks": 2,
                },
            },
            "algorithm": {
                "rollout_epoch": 1,
                "update_epoch": 1,
            },
            "rollout": {
                "pipeline_stage_num": 1,
            },
        }
    )
    base_bindings = {"actor": [], "rollout": []}
    profile_functions = Mock(name="profile_functions")
    profiler = Mock(name="profiler")
    load_hydra_config = Mock(return_value=cfg)
    load_base_bindings = Mock(return_value=base_bindings)
    default_profile_functions = Mock(return_value=profile_functions)
    toolkit_profiler = Mock(return_value=profiler)
    run_orchestration = Mock()
    monkeypatch.setattr(run, "load_hydra_config", load_hydra_config)
    monkeypatch.setattr(run, "load_base_bindings", load_base_bindings)
    monkeypatch.setattr(run, "default_profile_functions", default_profile_functions)
    monkeypatch.setattr(run, "ToolkitThroughputProfiler", toolkit_profiler)
    monkeypatch.setattr(run, "run_orchestration", run_orchestration)

    run.main(
        [
            "--config-path",
            "examples/embodiment/config",
            "--config-name",
            "libero",
            "--output-dir",
            "outputs/orchestration",
            "--plan-output",
            "outputs/plan.json",
            "--candidate-pairs",
            "20:80,30:70",
            "--warmup-steps",
            "3",
            "--measure-steps",
            "7",
            "--base-plan",
            "base-plan.json",
            "--selection-tolerance",
            "0.05",
        ]
    )

    load_hydra_config.assert_called_once_with(
        "examples/embodiment/config",
        "libero",
        overrides=(),
    )
    load_base_bindings.assert_called_once_with(cfg=cfg, base_plan="base-plan.json")
    default_profile_functions.assert_called_once_with()
    toolkit_profiler.assert_called_once()
    _, profiler_kwargs = toolkit_profiler.call_args
    assert profiler_kwargs == {
        "cfg": cfg,
        "summary": run.build_config_summary(cfg),
        "warmup_steps": 3,
        "measure_steps": 7,
        "functions": profile_functions,
    }
    run_orchestration.assert_called_once()
    _, orchestration_kwargs = run_orchestration.call_args
    assert orchestration_kwargs == {
        "config_summary": run.build_config_summary(cfg),
        "base_bindings": base_bindings,
        "candidates": run.parse_candidate_pairs("20:80,30:70"),
        "profiler": profiler,
        "output_dir": "outputs/orchestration",
        "plan_output": "outputs/plan.json",
        "selection_tolerance": 0.05,
    }


def test_load_plan_bindings_parses_plan_by_component_sorted_by_rank(tmp_path) -> None:
    plan_path = tmp_path / "plan.json"
    _write_plan(
        plan_path,
        [
            _binding("actor", 1),
            _binding("rollout", 0),
            _binding("actor", 0),
        ],
    )

    bindings = load_plan_bindings(plan_path)

    assert list(bindings) == ["actor", "rollout"]
    assert [binding.rank for binding in bindings["actor"]] == [0, 1]
    assert bindings["rollout"][0].component == "rollout"


def test_load_base_bindings_prefers_explicit_base_plan(tmp_path) -> None:
    explicit_plan = tmp_path / "explicit.json"
    cfg_plan = tmp_path / "cfg.json"
    _write_plan(explicit_plan, [_binding("actor", 0)])
    _write_plan(cfg_plan, [_binding("rollout", 0)])
    cfg = OmegaConf.create(
        {
            "cluster": {
                "resource_pool": {
                    "allocation_plan_path": str(cfg_plan),
                }
            }
        }
    )

    bindings = load_base_bindings(cfg, str(explicit_plan))

    assert list(bindings) == ["actor"]


def test_load_base_bindings_raises_when_no_plan_path_exists() -> None:
    cfg = OmegaConf.create({"cluster": {"resource_pool": {}}})

    with pytest.raises(ValueError, match="--base-plan.*allocation_plan_path"):
        load_base_bindings(cfg, None)
