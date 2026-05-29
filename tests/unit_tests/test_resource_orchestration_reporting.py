import json

from toolkits.resource_orchestration.reporting import write_reports
from toolkits.resource_orchestration.types import (
    CandidateEstimate,
    CandidatePair,
    SelectionResult,
    StageThroughput,
)


def _estimate() -> CandidateEstimate:
    return CandidateEstimate(
        candidate=CandidatePair(actor_sm=40, rollout_sm=60),
        throughput=StageThroughput(
            env_chunk_steps_per_sec=20,
            model_chunk_steps_per_sec=10,
            actor_chunk_steps_per_sec=30,
            pipeline_samples_per_sec=9,
        ),
        rollout_chunk_count=100,
        rollout_time_s=10,
        training_time_s=3.3333333333,
        epoch_time_s=10,
        balance_gap_s=6.6666666667,
        bottleneck_stage="model",
    )


def test_write_reports_writes_profiles_summary_json_and_markdown(tmp_path) -> None:
    estimate = _estimate()
    selection = SelectionResult(selected=estimate, ranked=(estimate,))

    outputs = write_reports(
        tmp_path,
        estimates=(estimate,),
        selection=selection,
        plan_output="plan.json",
        failed_profiles=[],
    )

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    profile = json.loads(
        (tmp_path / "profiles" / "actor40_rollout60.json").read_text(
            encoding="utf-8"
        )
    )
    summary_md = (tmp_path / "summary.md").read_text(encoding="utf-8")

    assert outputs == {
        "summary_json": tmp_path / "summary.json",
        "summary_md": tmp_path / "summary.md",
        "profiles_dir": tmp_path / "profiles",
    }
    assert summary["selected"]["candidate_id"] == "actor40_rollout60"
    assert summary["ranked_candidate_ids"] == ["actor40_rollout60"]
    assert summary["plan_output"] == "plan.json"
    assert profile["throughput"]["model_chunk_steps_per_sec"] == 10
    assert "actor40_rollout60" in summary_md


def test_write_reports_supports_all_failed_selection(tmp_path) -> None:
    write_reports(
        tmp_path,
        estimates=(),
        selection=None,
        plan_output="plan.json",
        failed_profiles=["actor40_rollout60"],
    )

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    summary_md = (tmp_path / "summary.md").read_text(encoding="utf-8")

    assert summary["selected"] is None
    assert summary["failed_profiles"] == ["actor40_rollout60"]
    assert "Selected candidate: none" in summary_md
