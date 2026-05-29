import json

import pytest

from rlinf.scheduler.resource_pool.bindings import (
    CpuBinding,
    GpuBinding,
    WorkerResourceBinding,
)
from toolkits.resource_orchestration.orchestrator import run_orchestration
from toolkits.resource_orchestration.types import (
    CandidatePair,
    ConfigSummary,
    StageThroughput,
)


class StubProfiler:
    def profile(self, candidate: CandidatePair) -> StageThroughput:
        if (candidate.actor_sm, candidate.rollout_sm) == (30, 70):
            return StageThroughput(
                env_chunk_steps_per_sec=10,
                model_chunk_steps_per_sec=10,
                actor_chunk_steps_per_sec=20,
            )
        if (candidate.actor_sm, candidate.rollout_sm) == (70, 30):
            return StageThroughput(
                env_chunk_steps_per_sec=10,
                model_chunk_steps_per_sec=10,
                actor_chunk_steps_per_sec=5,
            )
        raise RuntimeError("unexpected candidate")


class FailingProfiler:
    def profile(self, candidate: CandidatePair) -> StageThroughput:
        raise RuntimeError(f"profile failed for {candidate.candidate_id}")


def _summary() -> ConfigSummary:
    return ConfigSummary(
        total_num_envs=8,
        episode_env_steps=5,
        chunk_size=1,
        chunk_steps_per_env=1,
        rollout_epoch=1,
        rollout_chunk_count=20,
        update_epoch=1,
        actor_global_batch_size=8,
        actor_micro_batch_size=1,
        pipeline_stage_num=1,
        resource_pool_mode="mps",
    )


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


def _base_bindings() -> dict[str, list[WorkerResourceBinding]]:
    return {
        "actor": [_binding("actor", 0)],
        "rollout": [_binding("rollout", 1)],
    }


def test_run_orchestration_profiles_estimates_writes_plan_and_reports(
    tmp_path,
) -> None:
    plan_output = tmp_path / "plan.json"

    selection = run_orchestration(
        config_summary=_summary(),
        base_bindings=_base_bindings(),
        candidates=[
            CandidatePair(actor_sm=30, rollout_sm=70),
            CandidatePair(actor_sm=70, rollout_sm=30),
        ],
        profiler=StubProfiler(),
        output_dir=tmp_path,
        plan_output=plan_output,
    )

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    plan = json.loads(plan_output.read_text(encoding="utf-8"))

    assert selection.selected.candidate == CandidatePair(actor_sm=30, rollout_sm=70)
    assert summary["selected"]["candidate_id"] == "actor30_rollout70"
    assert summary["ranked_candidate_ids"] == [
        "actor30_rollout70",
        "actor70_rollout30",
    ]
    assert summary["failed_profiles"] == []
    assert plan["bindings"][0]["gpu"]["sm_percent"] == 30
    assert plan["bindings"][1]["gpu"]["sm_percent"] == 70


def test_run_orchestration_reports_failures_without_plan_when_all_candidates_fail(
    tmp_path,
) -> None:
    plan_output = tmp_path / "plan.json"

    with pytest.raises(RuntimeError, match="all .*candidates failed"):
        run_orchestration(
            config_summary=_summary(),
            base_bindings=_base_bindings(),
            candidates=[
                CandidatePair(actor_sm=30, rollout_sm=70),
                CandidatePair(actor_sm=70, rollout_sm=30),
            ],
            profiler=FailingProfiler(),
            output_dir=tmp_path,
            plan_output=plan_output,
        )

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))

    assert not plan_output.exists()
    assert summary["selected"] is None
    assert summary["failed_profiles"] == [
        {
            "candidate_id": "actor30_rollout70",
            "error": "profile failed for actor30_rollout70",
        },
        {
            "candidate_id": "actor70_rollout30",
            "error": "profile failed for actor70_rollout30",
        },
    ]
