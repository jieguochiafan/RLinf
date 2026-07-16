import json
from pathlib import Path

from toolkits.rollout_eval.benchmark.openpi_stage_mps_profile import (
    StageProfileCase,
    _load_existing_passed_case,
    build_stage_cases,
    build_worker_env,
    write_summary,
)


class _Args:
    gpu = 2
    mps_sm = "20,60"
    num_envs_list = "1,4"
    mps_pipe_dir = "/tmp/rlinf-test-mps-pipe"
    mps_log_dir = "/tmp/rlinf-test-mps-log"


def test_build_stage_cases_expands_mps_and_batch_matrix() -> None:
    cases = build_stage_cases(_Args())

    assert cases == [
        StageProfileCase(case_id="mps-sm20-bs1", mps_sm=20, num_envs=1, gpu=2),
        StageProfileCase(case_id="mps-sm20-bs4", mps_sm=20, num_envs=4, gpu=2),
        StageProfileCase(case_id="mps-sm60-bs1", mps_sm=60, num_envs=1, gpu=2),
        StageProfileCase(case_id="mps-sm60-bs4", mps_sm=60, num_envs=4, gpu=2),
    ]


def test_build_worker_env_sets_single_gpu_mps_binding() -> None:
    case = StageProfileCase(case_id="mps-sm60-bs4", mps_sm=60, num_envs=4, gpu=2)

    env = build_worker_env(
        base_env={"CUDA_VISIBLE_DEVICES": "7", "KEEP": "value"},
        case=case,
        mps_pipe_dir="/tmp/pipe",
        mps_log_dir="/tmp/log",
    )

    assert env["KEEP"] == "value"
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] == "60"
    assert env["CUDA_MPS_PIPE_DIRECTORY"] == "/tmp/pipe"
    assert env["CUDA_MPS_LOG_DIRECTORY"] == "/tmp/log"


def test_build_worker_env_prefers_uuid_visible_device_for_mps() -> None:
    case = StageProfileCase(
        case_id="mps-sm60-bs4",
        mps_sm=60,
        num_envs=4,
        gpu=2,
        cuda_visible_device="GPU-test-uuid",
    )

    env = build_worker_env(
        base_env={},
        case=case,
        mps_pipe_dir="/tmp/pipe",
        mps_log_dir="/tmp/log",
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-test-uuid"


def test_write_summary_outputs_json_and_markdown(tmp_path: Path) -> None:
    records = [
        {
            "case_id": "mps-sm20-bs1",
            "status": "pass",
            "mps_sm": 20,
            "num_envs": 1,
            "gpu": 0,
            "metrics": {
                "total": {"avg_ms": 100.0, "p50_ms": 99.0, "p95_ms": 120.0},
                "vlm": {"avg_ms": 60.0, "p50_ms": 58.0, "p95_ms": 75.0},
                "action_head": {"avg_ms": 35.0, "p50_ms": 34.0, "p95_ms": 45.0},
                "other": {"avg_ms": 5.0, "p50_ms": 5.0, "p95_ms": 6.0},
            },
        }
    ]

    summary = write_summary(tmp_path, records)

    assert summary["counts"] == {"total": 1, "pass": 1, "failed": 0}
    assert json.loads((tmp_path / "summary.json").read_text()) == summary
    assert (
        "| mps-sm20-bs1 | pass | 20 | 1 | 100.000 | 60.000 | 35.000 | 5.000 | 0 |"
        in (tmp_path / "summary.md").read_text()
    )


def test_load_existing_passed_case_requires_pass_and_matching_case(
    tmp_path: Path,
) -> None:
    case = StageProfileCase(case_id="mps-sm20-bs1", mps_sm=20, num_envs=1, gpu=1)
    case_dir = tmp_path / "cases" / case.case_id
    case_dir.mkdir(parents=True)
    report = {
        "case_id": case.case_id,
        "status": "pass",
        "mps_sm": 20,
        "num_envs": 1,
        "gpu": 1,
        "metrics": {"total": {"avg_ms": 1.0}},
    }
    report_path = case_dir / "case_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    assert _load_existing_passed_case(tmp_path, case) == report

    report["status"] = "failed"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert _load_existing_passed_case(tmp_path, case) is None

    report["status"] = "pass"
    report["num_envs"] = 2
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert _load_existing_passed_case(tmp_path, case) is None
