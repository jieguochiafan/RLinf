from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from toolkits.resource_orchestration.types import (
    CandidatePair,
    ConfigSummary,
    StageThroughput,
)

RolloutProfileFn = Callable[[Any, CandidatePair, int, int], Mapping[str, float]]
TrainingProfileFn = Callable[
    [Any, CandidatePair, ConfigSummary, int, int], Mapping[str, float]
]


class ThroughputProfiler(Protocol):
    """Profiles measured stage throughput for a candidate allocation."""

    def profile(self, candidate: CandidatePair) -> StageThroughput:
        """Return measured throughput for a candidate allocation."""


@dataclass(frozen=True)
class ToolkitProfileFunctions:
    """Concrete profiling functions used by the toolkit adapter."""

    rollout_profile: RolloutProfileFn
    training_profile: TrainingProfileFn


def combine_profile_metrics(
    *,
    env_steps_per_sec: float,
    model_infers_per_sec: float,
    actor_chunk_steps_per_sec: float,
    chunk_size: int,
    total_num_envs: int,
    pipeline_samples_per_sec: float | None = None,
) -> StageThroughput:
    """Combine raw profiler metrics into orchestration stage throughput."""
    return StageThroughput(
        env_chunk_steps_per_sec=(
            float(env_steps_per_sec) * float(total_num_envs) / float(chunk_size)
        ),
        model_chunk_steps_per_sec=(
            float(model_infers_per_sec) * float(total_num_envs)
        ),
        actor_chunk_steps_per_sec=float(actor_chunk_steps_per_sec),
        pipeline_samples_per_sec=(
            None
            if pipeline_samples_per_sec is None
            else float(pipeline_samples_per_sec) * float(total_num_envs)
        ),
    )


@dataclass(frozen=True)
class ToolkitThroughputProfiler:
    """Throughput profiler backed by rollout_eval and training_eval toolkits."""

    cfg: Any
    summary: ConfigSummary
    warmup_steps: int
    measure_steps: int
    functions: ToolkitProfileFunctions

    def profile(self, candidate: CandidatePair) -> StageThroughput:
        """Return measured throughput for a candidate allocation."""
        rollout_metrics = self.functions.rollout_profile(
            self.cfg,
            candidate,
            self.warmup_steps,
            self.measure_steps,
        )
        training_metrics = self.functions.training_profile(
            self.cfg,
            candidate,
            self.summary,
            self.warmup_steps,
            self.measure_steps,
        )
        return combine_profile_metrics(
            env_steps_per_sec=rollout_metrics["env_steps_per_sec"],
            model_infers_per_sec=rollout_metrics["model_infers_per_sec"],
            actor_chunk_steps_per_sec=training_metrics["actor_chunk_steps_per_sec"],
            chunk_size=self.summary.chunk_size,
            total_num_envs=self.summary.total_num_envs,
            pipeline_samples_per_sec=rollout_metrics.get("pipeline_samples_per_sec"),
        )


def _metric_value(metrics: Any, name: str) -> float:
    if isinstance(metrics, Mapping):
        return float(metrics[name])
    return float(getattr(metrics, name))


def _replace_environ(env: Mapping[str, str]) -> None:
    os.environ.clear()
    os.environ.update({key: str(value) for key, value in env.items()})


def _close_if_present(adapter: Any) -> None:
    close = getattr(adapter, "close", None)
    if close is not None:
        close()


def default_rollout_profile(
    cfg: Any,
    candidate: CandidatePair,
    warmup_steps: int,
    measure_steps: int,
) -> dict[str, float]:
    """Profile rollout env/model stages using rollout_eval toolkit adapters."""
    from toolkits.rollout_eval.adapters import build_env_adapter, build_model_adapter
    from toolkits.rollout_eval.benchmark.resource_binding import build_process_env
    from toolkits.rollout_eval.benchmark.single_runner import (
        run_env_only_case,
        run_model_only_case,
    )

    original_env = dict(os.environ)
    process_env = build_process_env(
        base_env=os.environ,
        mps_active_thread_percentage=candidate.rollout_sm,
    )
    env_adapter = None
    template_env_adapter = None
    model_adapter = None

    try:
        _replace_environ(process_env)
        env_adapter = build_env_adapter(cfg, split="train", profile_output_dir=None)
        env_result = run_env_only_case(
            env_adapter=env_adapter,
            warmup_steps=warmup_steps,
            measure_steps=measure_steps,
        )

        template_env_adapter = build_env_adapter(
            cfg,
            split="train",
            profile_output_dir=None,
        )
        obs_batch, _ = template_env_adapter.reset()

        model_adapter = build_model_adapter(cfg, split_model_stages=False)
        model_result = run_model_only_case(
            env_adapter=None,
            model_adapter=model_adapter,
            warmup_steps=warmup_steps,
            measure_steps=measure_steps,
            obs_batch=obs_batch,
        )
    finally:
        _close_if_present(env_adapter)
        _close_if_present(template_env_adapter)
        _close_if_present(model_adapter)
        _replace_environ(original_env)

    return {
        "env_steps_per_sec": _metric_value(
            env_result.metrics,
            "env_steps_per_sec",
        ),
        "model_infers_per_sec": _metric_value(
            model_result.metrics,
            "model_infers_per_sec",
        ),
        "pipeline_samples_per_sec": _metric_value(
            model_result.metrics,
            "pipeline_samples_per_sec",
        ),
    }


def default_training_profile(
    cfg: Any,
    candidate: CandidatePair,
    summary: ConfigSummary,
    warmup_steps: int,
    measure_steps: int,
) -> Mapping[str, float]:
    """Profile actor training stage using training_eval toolkit."""
    try:
        from toolkits.training_eval.run import run_training_profile
    except ImportError as exc:
        raise RuntimeError(
            "toolkits.training_eval is required for training profiling"
        ) from exc

    return run_training_profile(
        cfg=cfg,
        actor_sm=candidate.actor_sm,
        warmup_steps=warmup_steps,
        measure_steps=measure_steps,
        rollout_chunk_count=summary.rollout_chunk_count,
    )


def default_profile_functions() -> ToolkitProfileFunctions:
    """Return default rollout/training toolkit profile functions."""
    return ToolkitProfileFunctions(
        rollout_profile=default_rollout_profile,
        training_profile=default_training_profile,
    )
