from __future__ import annotations

import json

import pytest

from toolkits.rollout_eval.benchmark import openpi_sm_profile


def test_parse_args_defaults_to_single_gpu_mps100() -> None:
    args = openpi_sm_profile.parse_args(
        [
            "--config-path",
            "examples/embodiment/config",
            "--config-name",
            "libero_goal_ppo_openpi_pi05",
            "--gpu",
            "3",
        ]
    )

    assert args.gpu == "3"
    assert args.mps_sm == 100
    assert args.batch_sizes == "1,2,4,8,16,32"
    assert args.warmup_steps == 5
    assert args.measure_steps == 20


def test_build_cases_parses_batch_sizes_and_case_ids() -> None:
    args = openpi_sm_profile.parse_args(
        [
            "--config-path",
            "examples/embodiment/config",
            "--config-name",
            "libero_goal_ppo_openpi_pi05",
            "--gpu",
            "0",
            "--batch-sizes",
            "8,1,8,4",
        ]
    )

    cases = openpi_sm_profile.build_cases(args)

    assert [case.batch_size for case in cases] == [1, 4, 8]
    assert [case.case_id for case in cases] == [
        "openpi-sm100-bs1",
        "openpi-sm100-bs4",
        "openpi-sm100-bs8",
    ]


def test_build_cases_rejects_invalid_values() -> None:
    args = openpi_sm_profile.parse_args(
        [
            "--config-path",
            "examples/embodiment/config",
            "--config-name",
            "libero_goal_ppo_openpi_pi05",
            "--gpu",
            "0",
            "--mps-sm",
            "0",
        ]
    )

    with pytest.raises(ValueError, match="--mps-sm values must be in \\[1, 100\\]"):
        openpi_sm_profile.build_cases(args)


def test_summarize_chrome_trace_extracts_kernel_busy_and_occupancy(tmp_path) -> None:
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "ph": "X",
                        "cat": "cpu_op",
                        "name": "model.inference.openpi",
                        "ts": 0,
                        "dur": 1000,
                    },
                    {
                        "ph": "X",
                        "cat": "kernel",
                        "name": "gemm",
                        "ts": 100,
                        "dur": 200,
                        "args": {"est. achieved occupancy %": 50.0},
                    },
                    {
                        "ph": "X",
                        "cat": "Kernel",
                        "name": "attention",
                        "ts": 300,
                        "dur": 100,
                        "args": {"est. achieved occupancy %": "75"},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = openpi_sm_profile.summarize_chrome_trace(trace_path)

    assert summary.kernel_count == 2
    assert summary.kernel_time_us == pytest.approx(300.0)
    assert summary.trace_window_us == pytest.approx(1000.0)
    assert summary.kernel_busy_ratio == pytest.approx(0.3)
    assert summary.occupancy_available is True
    assert summary.occupancy_avg_pct == pytest.approx(62.5)
    assert summary.occupancy_max_pct == pytest.approx(75.0)


def test_summarize_chrome_trace_handles_missing_occupancy(tmp_path) -> None:
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "ph": "X",
                        "cat": "kernel",
                        "name": "conv",
                        "ts": 10,
                        "dur": 20,
                        "args": {},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = openpi_sm_profile.summarize_chrome_trace(trace_path)

    assert summary.kernel_count == 1
    assert summary.occupancy_available is False
    assert summary.occupancy_avg_pct is None
    assert summary.occupancy_max_pct is None
