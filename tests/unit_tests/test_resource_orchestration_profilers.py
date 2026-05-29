from __future__ import annotations

import builtins
import os
from types import SimpleNamespace
from typing import Any

import pytest

from toolkits.resource_orchestration.profilers import (
    ToolkitProfileFunctions,
    ToolkitThroughputProfiler,
    combine_profile_metrics,
    default_rollout_profile,
    default_training_profile,
)
from toolkits.resource_orchestration.types import CandidatePair, ConfigSummary


class ClosableAdapter:
    def __init__(self) -> None:
        self.close_count = 0

    @property
    def closed(self) -> bool:
        return self.close_count > 0

    def close(self) -> None:
        self.close_count += 1


def _summary() -> ConfigSummary:
    return ConfigSummary(
        total_num_envs=8,
        episode_env_steps=10,
        chunk_size=4,
        chunk_steps_per_env=5,
        rollout_epoch=1,
        rollout_chunk_count=40,
        update_epoch=2,
        actor_global_batch_size=16,
        actor_micro_batch_size=2,
        pipeline_stage_num=1,
        resource_pool_mode="mps",
    )


def _patch_rollout_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    build_process_env: Any | None = None,
    build_env_adapter: Any | None = None,
    build_model_adapter: Any | None = None,
    run_env_only_case: Any | None = None,
    run_model_only_case: Any | None = None,
) -> None:
    from toolkits.resource_orchestration import profilers
    from toolkits.rollout_eval import adapters
    from toolkits.rollout_eval.benchmark import resource_binding, single_runner

    patches = {
        "build_process_env": build_process_env,
        "build_env_adapter": build_env_adapter,
        "build_model_adapter": build_model_adapter,
        "run_env_only_case": run_env_only_case,
        "run_model_only_case": run_model_only_case,
    }
    module_targets = {
        "build_process_env": resource_binding,
        "build_env_adapter": adapters,
        "build_model_adapter": adapters,
        "run_env_only_case": single_runner,
        "run_model_only_case": single_runner,
    }

    for name, replacement in patches.items():
        if replacement is None:
            continue
        monkeypatch.setattr(profilers, name, replacement, raising=False)
        monkeypatch.setattr(module_targets[name], name, replacement)


def test_combine_profile_metrics_converts_env_steps_to_chunk_steps() -> None:
    throughput = combine_profile_metrics(
        env_steps_per_sec=80.0,
        model_infers_per_sec=12.5,
        actor_chunk_steps_per_sec=7.0,
        chunk_size=4,
        pipeline_samples_per_sec=3.0,
    )

    assert throughput.env_chunk_steps_per_sec == 20.0
    assert throughput.model_chunk_steps_per_sec == 12.5
    assert throughput.actor_chunk_steps_per_sec == 7.0
    assert throughput.pipeline_samples_per_sec == 3.0


def test_toolkit_throughput_profiler_calls_injected_functions() -> None:
    calls: list[tuple[str, Any, CandidatePair, Any, int, int]] = []
    cfg = SimpleNamespace(name="cfg")
    summary = _summary()
    candidate = CandidatePair(actor_sm=70, rollout_sm=30)

    def rollout_profile(
        profile_cfg: Any,
        profile_candidate: CandidatePair,
        warmup_steps: int,
        measure_steps: int,
    ) -> dict[str, float]:
        calls.append(
            (
                "rollout",
                profile_cfg,
                profile_candidate,
                None,
                warmup_steps,
                measure_steps,
            )
        )
        return {
            "env_steps_per_sec": 100.0,
            "model_infers_per_sec": 25.0,
            "pipeline_samples_per_sec": 9.0,
        }

    def training_profile(
        profile_cfg: Any,
        profile_candidate: CandidatePair,
        profile_summary: ConfigSummary,
        warmup_steps: int,
        measure_steps: int,
    ) -> dict[str, float]:
        calls.append(
            (
                "training",
                profile_cfg,
                profile_candidate,
                profile_summary,
                warmup_steps,
                measure_steps,
            )
        )
        return {"actor_chunk_steps_per_sec": 11.0}

    profiler = ToolkitThroughputProfiler(
        cfg=cfg,
        summary=summary,
        warmup_steps=1,
        measure_steps=3,
        functions=ToolkitProfileFunctions(
            rollout_profile=rollout_profile,
            training_profile=training_profile,
        ),
    )

    throughput = profiler.profile(candidate)

    assert calls == [
        ("rollout", cfg, candidate, None, 1, 3),
        ("training", cfg, candidate, summary, 1, 3),
    ]
    assert throughput.env_chunk_steps_per_sec == 25.0
    assert throughput.model_chunk_steps_per_sec == 25.0
    assert throughput.actor_chunk_steps_per_sec == 11.0
    assert throughput.pipeline_samples_per_sec == 9.0


def test_default_training_profile_raises_when_training_eval_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "toolkits.training_eval.run":
            raise ImportError("missing training_eval")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(RuntimeError, match="toolkits.training_eval.*required"):
        default_training_profile(
            cfg=SimpleNamespace(),
            candidate=CandidatePair(actor_sm=60, rollout_sm=40),
            summary=_summary(),
            warmup_steps=1,
            measure_steps=2,
        )


def test_default_rollout_profile_uses_mps_env_and_returns_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = SimpleNamespace()
    candidate = CandidatePair(actor_sm=65, rollout_sm=35)
    build_process_env_calls: list[int | None] = []
    built_env_adapters: list[Any] = []
    monkeypatch.setenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "99")

    class EnvAdapter:
        def __init__(self) -> None:
            self.closed = False

        def reset(self) -> tuple[dict[str, int], dict[str, int]]:
            return {"obs": 1}, {}

        def close(self) -> None:
            self.closed = True

    def build_process_env(
        *, base_env: Any, mps_active_thread_percentage: int | None
    ) -> dict[str, str]:
        build_process_env_calls.append(mps_active_thread_percentage)
        env = dict(base_env)
        env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(mps_active_thread_percentage)
        env["ROLL_OUT_TEST_MARKER"] = "active"
        return env

    def build_env_adapter(
        profile_cfg: Any, *, split: str, profile_output_dir: Any
    ) -> EnvAdapter:
        assert profile_cfg is cfg
        assert split == "eval"
        assert profile_output_dir is None
        assert os.environ["ROLL_OUT_TEST_MARKER"] == "active"
        adapter = EnvAdapter()
        built_env_adapters.append(adapter)
        return adapter

    def build_model_adapter(profile_cfg: Any, *, split_model_stages: bool) -> object:
        assert profile_cfg is cfg
        assert split_model_stages is False
        assert os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] == "35"
        return object()

    def run_env_only_case(
        *, env_adapter: Any, warmup_steps: int, measure_steps: int
    ) -> Any:
        assert env_adapter is built_env_adapters[0]
        assert warmup_steps == 1
        assert measure_steps == 2
        return SimpleNamespace(metrics=SimpleNamespace(env_steps_per_sec=123.0))

    def run_model_only_case(
        *,
        env_adapter: Any,
        model_adapter: Any,
        warmup_steps: int,
        measure_steps: int,
        obs_batch: Any,
    ) -> Any:
        assert env_adapter is None
        assert model_adapter is not None
        assert warmup_steps == 1
        assert measure_steps == 2
        assert obs_batch == {"obs": 1}
        return SimpleNamespace(
            metrics=SimpleNamespace(
                model_infers_per_sec=45.0,
                pipeline_samples_per_sec=6.0,
            )
        )

    _patch_rollout_dependencies(
        monkeypatch,
        build_process_env=build_process_env,
        build_env_adapter=build_env_adapter,
        build_model_adapter=build_model_adapter,
        run_env_only_case=run_env_only_case,
        run_model_only_case=run_model_only_case,
    )

    metrics = default_rollout_profile(
        cfg=cfg,
        candidate=candidate,
        warmup_steps=1,
        measure_steps=2,
    )

    assert build_process_env_calls == [35]
    assert metrics == {
        "env_steps_per_sec": 123.0,
        "model_infers_per_sec": 45.0,
        "pipeline_samples_per_sec": 6.0,
    }
    assert os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] == "99"
    assert built_env_adapters[1].closed is True


def test_default_rollout_profile_closes_first_env_when_env_only_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_adapter = ClosableAdapter()

    def build_process_env(
        *, base_env: Any, mps_active_thread_percentage: int | None
    ) -> dict[str, str]:
        return dict(base_env)

    def build_env_adapter(
        _cfg: Any, *, split: str, profile_output_dir: Any
    ) -> ClosableAdapter:
        return env_adapter

    def run_env_only_case(
        *, env_adapter: Any, warmup_steps: int, measure_steps: int
    ) -> Any:
        raise RuntimeError("env failed")

    _patch_rollout_dependencies(
        monkeypatch,
        build_process_env=build_process_env,
        build_env_adapter=build_env_adapter,
        run_env_only_case=run_env_only_case,
    )

    with pytest.raises(RuntimeError, match="env failed"):
        default_rollout_profile(
            cfg=SimpleNamespace(),
            candidate=CandidatePair(actor_sm=65, rollout_sm=35),
            warmup_steps=1,
            measure_steps=2,
        )

    assert env_adapter.closed is True


def test_default_rollout_profile_closes_template_env_when_reset_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_env = ClosableAdapter()

    class ResetFailingAdapter(ClosableAdapter):
        def reset(self) -> tuple[dict[str, int], dict[str, int]]:
            raise RuntimeError("reset failed")

    template_env = ResetFailingAdapter()
    env_adapters = [first_env, template_env]

    def build_process_env(
        *, base_env: Any, mps_active_thread_percentage: int | None
    ) -> dict[str, str]:
        return dict(base_env)

    def build_env_adapter(
        _cfg: Any, *, split: str, profile_output_dir: Any
    ) -> ClosableAdapter:
        return env_adapters.pop(0)

    def run_env_only_case(
        *, env_adapter: Any, warmup_steps: int, measure_steps: int
    ) -> Any:
        return SimpleNamespace(metrics=SimpleNamespace(env_steps_per_sec=123.0))

    _patch_rollout_dependencies(
        monkeypatch,
        build_process_env=build_process_env,
        build_env_adapter=build_env_adapter,
        run_env_only_case=run_env_only_case,
    )

    with pytest.raises(RuntimeError, match="reset failed"):
        default_rollout_profile(
            cfg=SimpleNamespace(),
            candidate=CandidatePair(actor_sm=65, rollout_sm=35),
            warmup_steps=1,
            measure_steps=2,
        )

    assert template_env.closed is True


def test_default_rollout_profile_closes_model_adapter_when_model_only_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TemplateAdapter(ClosableAdapter):
        def reset(self) -> tuple[dict[str, int], dict[str, int]]:
            return {"obs": 1}, {}

    model_adapter = ClosableAdapter()
    env_adapters = [ClosableAdapter(), TemplateAdapter()]

    def build_process_env(
        *, base_env: Any, mps_active_thread_percentage: int | None
    ) -> dict[str, str]:
        return dict(base_env)

    def build_env_adapter(
        _cfg: Any, *, split: str, profile_output_dir: Any
    ) -> ClosableAdapter:
        return env_adapters.pop(0)

    def build_model_adapter(_cfg: Any, *, split_model_stages: bool) -> ClosableAdapter:
        return model_adapter

    def run_env_only_case(
        *, env_adapter: Any, warmup_steps: int, measure_steps: int
    ) -> Any:
        return SimpleNamespace(metrics=SimpleNamespace(env_steps_per_sec=123.0))

    def run_model_only_case(
        *,
        env_adapter: Any,
        model_adapter: Any,
        warmup_steps: int,
        measure_steps: int,
        obs_batch: Any,
    ) -> Any:
        return SimpleNamespace(
            metrics=SimpleNamespace(
                model_infers_per_sec=45.0,
                pipeline_samples_per_sec=6.0,
            )
        )

    _patch_rollout_dependencies(
        monkeypatch,
        build_process_env=build_process_env,
        build_env_adapter=build_env_adapter,
        build_model_adapter=build_model_adapter,
        run_env_only_case=run_env_only_case,
        run_model_only_case=run_model_only_case,
    )

    default_rollout_profile(
        cfg=SimpleNamespace(),
        candidate=CandidatePair(actor_sm=65, rollout_sm=35),
        warmup_steps=1,
        measure_steps=2,
    )

    assert model_adapter.closed is True


def test_default_rollout_profile_closes_model_adapter_when_model_only_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TemplateAdapter(ClosableAdapter):
        def reset(self) -> tuple[dict[str, int], dict[str, int]]:
            return {"obs": 1}, {}

    model_adapter = ClosableAdapter()
    env_adapters = [ClosableAdapter(), TemplateAdapter()]

    def build_process_env(
        *, base_env: Any, mps_active_thread_percentage: int | None
    ) -> dict[str, str]:
        return dict(base_env)

    def build_env_adapter(
        _cfg: Any, *, split: str, profile_output_dir: Any
    ) -> ClosableAdapter:
        return env_adapters.pop(0)

    def build_model_adapter(_cfg: Any, *, split_model_stages: bool) -> ClosableAdapter:
        return model_adapter

    def run_env_only_case(
        *, env_adapter: Any, warmup_steps: int, measure_steps: int
    ) -> Any:
        return SimpleNamespace(metrics=SimpleNamespace(env_steps_per_sec=123.0))

    def run_model_only_case(
        *,
        env_adapter: Any,
        model_adapter: Any,
        warmup_steps: int,
        measure_steps: int,
        obs_batch: Any,
    ) -> Any:
        raise RuntimeError("model failed")

    _patch_rollout_dependencies(
        monkeypatch,
        build_process_env=build_process_env,
        build_env_adapter=build_env_adapter,
        build_model_adapter=build_model_adapter,
        run_env_only_case=run_env_only_case,
        run_model_only_case=run_model_only_case,
    )

    with pytest.raises(RuntimeError, match="model failed"):
        default_rollout_profile(
            cfg=SimpleNamespace(),
            candidate=CandidatePair(actor_sm=65, rollout_sm=35),
            warmup_steps=1,
            measure_steps=2,
        )

    assert model_adapter.closed is True
