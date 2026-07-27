from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from toolkits.rollout_eval.benchmark.orchestrator import _make_random_model_obs
from toolkits.rollout_eval.benchmark.types import BenchmarkCase
from toolkits.rollout_eval.profiling.roofline import (
    AggregatedRooflineRow,
    aggregate_roofline_rows,
    apply_stage_time_overrides,
    parse_chrome_trace_to_rows,
    plot_roofline,
    reduce_roofline_rows_by_config_stage,
    write_roofline_csv,
)


def _write_trace(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "ph": "X",
                        "cat": "user_annotation",
                        "name": "model.inference.openpi",
                        "ts": 0,
                        "dur": 1000,
                        "args": {"External id": 1},
                    },
                    {
                        "ph": "X",
                        "cat": "user_annotation",
                        "name": "model.backbone.openpi.paligemma_with_expert.forward",
                        "ts": 10,
                        "dur": 400,
                        "args": {"External id": 2},
                    },
                    {
                        "ph": "X",
                        "cat": "cpu_op",
                        "name": "aten::mm",
                        "ts": 20,
                        "dur": 100,
                        "args": {
                            "External id": 10,
                            "Input Dims": [[4, 8], [8, 16]],
                            "Input type": ["BFloat16", "BFloat16"],
                            "FLOPs": 1024,
                        },
                    },
                    {
                        "ph": "X",
                        "cat": "kernel",
                        "name": "ampere_bf16_gemm",
                        "ts": 30,
                        "dur": 5,
                        "args": {
                            "External id": 10,
                            "stream": 7,
                            "est. achieved occupancy %": 25,
                        },
                    },
                    {
                        "ph": "X",
                        "cat": "user_annotation",
                        "name": "model.action_head.openpi.action_out_proj.forward",
                        "ts": 500,
                        "dur": 400,
                        "args": {"External id": 3},
                    },
                    {
                        "ph": "X",
                        "cat": "cpu_op",
                        "name": "aten::silu",
                        "ts": 520,
                        "dur": 60,
                        "args": {
                            "External id": 20,
                            "Input Dims": [[2, 4, 8]],
                            "Input type": ["BFloat16"],
                        },
                    },
                    {
                        "ph": "X",
                        "cat": "kernel",
                        "name": "silu_kernel",
                        "ts": 530,
                        "dur": 10,
                        "args": {"External id": 20, "stream": 7},
                    },
                    {
                        "ph": "X",
                        "cat": "cpu_op",
                        "name": "aten::add",
                        "ts": 610,
                        "dur": 30,
                        "args": {
                            "External id": 30,
                            "Input Dims": [[2, 4, 8], [2, 4, 8]],
                            "Input type": ["Float", "Float"],
                        },
                    },
                    {
                        "ph": "X",
                        "cat": "cuda_runtime",
                        "name": "cudaLaunchKernel",
                        "ts": 612,
                        "dur": 5,
                        "args": {"External id": 30, "correlation": 99},
                    },
                    {
                        "ph": "X",
                        "cat": "kernel",
                        "name": "add_kernel_without_external_id",
                        "ts": 620,
                        "dur": 6,
                        "args": {"correlation": 99, "stream": 7},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def _write_nested_denoise_trace(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "ph": "X",
                        "cat": "user_annotation",
                        "name": "model.action_head.openpi.denoise_step",
                        "ts": 0,
                        "dur": 200,
                        "args": {"External id": 1},
                    },
                    {
                        "ph": "X",
                        "cat": "user_annotation",
                        "name": "model.backbone.openpi.paligemma_with_expert.forward",
                        "ts": 10,
                        "dur": 120,
                        "args": {"External id": 2},
                    },
                    {
                        "ph": "X",
                        "cat": "cpu_op",
                        "name": "aten::mm",
                        "ts": 20,
                        "dur": 80,
                        "args": {
                            "External id": 10,
                            "Input Dims": [[4, 8], [8, 16]],
                            "Input type": ["BFloat16", "BFloat16"],
                            "FLOPs": 1024,
                        },
                    },
                    {
                        "ph": "X",
                        "cat": "kernel",
                        "name": "nested_bf16_gemm",
                        "ts": 30,
                        "dur": 5,
                        "args": {"External id": 10, "stream": 7},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def test_parse_chrome_trace_to_rows_extracts_stage_op_flops_and_bytes(tmp_path) -> None:
    trace_path = tmp_path / "trace.json"
    _write_trace(trace_path)

    rows = parse_chrome_trace_to_rows(trace_path, config_name="openpi-bs4")

    assert [row.stage for row in rows] == [
        "vlm_prefill",
        "action_head_denoise",
        "action_head_denoise",
    ]
    assert [row.op_name for row in rows] == ["aten::mm", "aten::silu", "aten::add"]
    assert rows[0].flops == pytest.approx(1024.0)
    assert rows[0].bytes == pytest.approx((4 * 8 + 8 * 16 + 4 * 16) * 2)
    assert rows[1].flops == pytest.approx(4 * 2 * 4 * 8)
    assert rows[2].flops == pytest.approx(2 * 4 * 8)
    assert rows[2].kernel_name == "add_kernel_without_external_id"


def test_parse_chrome_trace_prioritizes_outer_denoise_stage(tmp_path) -> None:
    trace_path = tmp_path / "nested_denoise_trace.json"
    _write_nested_denoise_trace(trace_path)

    rows = parse_chrome_trace_to_rows(trace_path, config_name="openpi-bs4")

    assert len(rows) == 1
    assert rows[0].stage == "action_head_denoise"


def test_parse_chrome_trace_to_rows_accepts_control_chars(tmp_path) -> None:
    trace_path = tmp_path / "trace_with_control_char.json"
    _write_trace(trace_path)
    text = trace_path.read_text(encoding="utf-8")
    text = text.replace(
        '"traceEvents": [',
        '"traceEvents": ['
        '{"ph":"X","cat":"python_function","name":"'
        + chr(6)
        + '(0): <module>",'
        '"ts":1,"dur":1,"args":{}},',
        1,
    )
    trace_path.write_text(text, encoding="utf-8")

    rows = parse_chrome_trace_to_rows(trace_path, config_name="openpi-bs1")

    assert len(rows) == 3


def test_aggregate_roofline_rows_groups_and_marks_top_time() -> None:
    rows = [
        *parse_chrome_trace_to_rows(
            _synthetic_trace_path("case-a"), config_name="openpi-bs4"
        ),
        *parse_chrome_trace_to_rows(
            _synthetic_trace_path("case-b"), config_name="openpi-bs4"
        ),
    ]

    aggregated = aggregate_roofline_rows(rows, top_k_per_stage=1)

    gemm = next(row for row in aggregated if row.kernel_name == "ampere_bf16_gemm")
    assert gemm.config == "openpi-bs4"
    assert gemm.stage == "vlm_prefill"
    assert gemm.total_time_us == pytest.approx(10.0)
    assert gemm.call_count == 2
    assert gemm.annotation_rank == 1


def test_write_roofline_csv_and_plot_outputs(tmp_path) -> None:
    trace_path = tmp_path / "trace.json"
    _write_trace(trace_path)
    aggregated = aggregate_roofline_rows(
        parse_chrome_trace_to_rows(trace_path, config_name="openpi-bs4")
    )

    csv_path = tmp_path / "aggregated.csv"
    write_roofline_csv(aggregated, csv_path)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        loaded = list(csv.DictReader(handle))
    assert loaded
    assert {
        "config",
        "stage",
        "kernel_name",
        "op_name",
        "shape_bucket",
        "flops",
        "bytes",
        "AI",
        "achieved_tflops",
        "total_time_us",
        "dtype",
    }.issubset(loaded[0])

    pdf_path = tmp_path / "vla_roofline.pdf"
    svg_path = tmp_path / "vla_roofline.svg"
    scales = plot_roofline(
        aggregated,
        pdf_path=pdf_path,
        svg_path=svg_path,
        peak_tflops=312.0,
        peak_bw_gbs=1555.0,
    )

    assert pdf_path.stat().st_size > 0
    assert svg_path.stat().st_size > 0
    assert "bs4" in svg_path.read_text(encoding="utf-8")
    assert scales == ("linear", "linear")


def test_reduce_roofline_rows_by_config_stage_uses_time_weighted_means() -> None:
    rows = [
        AggregatedRooflineRow(
            config="openpi-bs4",
            stage="vlm_prefill",
            kernel_name="kernel_a",
            op_name="aten::mm",
            shape_bucket="a",
            flops=100,
            bytes=50,
            arithmetic_intensity=2,
            achieved_tflops=10,
            total_time_us=1,
            dtype="float32",
            call_count=1,
        ),
        AggregatedRooflineRow(
            config="openpi-bs4",
            stage="vlm_prefill",
            kernel_name="kernel_b",
            op_name="aten::mm",
            shape_bucket="b",
            flops=300,
            bytes=50,
            arithmetic_intensity=6,
            achieved_tflops=20,
            total_time_us=3,
            dtype="float32",
            call_count=2,
        ),
        AggregatedRooflineRow(
            config="openpi-bs8",
            stage="action_head_denoise",
            kernel_name="kernel_c",
            op_name="aten::add",
            shape_bucket="c",
            flops=20,
            bytes=80,
            arithmetic_intensity=0.25,
            achieved_tflops=1,
            total_time_us=5,
            dtype="float32",
            call_count=3,
        ),
    ]

    reduced = reduce_roofline_rows_by_config_stage(rows)

    assert len(reduced) == 2
    prefill = next(row for row in reduced if row.stage == "vlm_prefill")
    assert prefill.config == "openpi-bs4"
    assert prefill.kernel_name == "openpi-bs4/vlm_prefill"
    assert prefill.arithmetic_intensity == pytest.approx(5.0)
    assert prefill.achieved_tflops == pytest.approx(17.5)
    assert prefill.total_time_us == pytest.approx(4.0)
    assert prefill.call_count == 3


def test_reduce_roofline_rows_merges_vlm_decode_into_prefill() -> None:
    rows = [
        AggregatedRooflineRow(
            config="openpi-bs4",
            stage="vlm_prefill",
            kernel_name="prefill",
            op_name="aten::mm",
            shape_bucket="a",
            flops=100,
            bytes=50,
            arithmetic_intensity=10,
            achieved_tflops=20,
            total_time_us=1,
            dtype="float32",
            call_count=1,
        ),
        AggregatedRooflineRow(
            config="openpi-bs4",
            stage="vlm_decode",
            kernel_name="other",
            op_name="aten::addmm",
            shape_bucket="b",
            flops=200,
            bytes=100,
            arithmetic_intensity=2,
            achieved_tflops=4,
            total_time_us=3,
            dtype="float32",
            call_count=1,
        ),
        AggregatedRooflineRow(
            config="openpi-bs4",
            stage="action_head_denoise",
            kernel_name="action",
            op_name="aten::add",
            shape_bucket="c",
            flops=10,
            bytes=40,
            arithmetic_intensity=0.25,
            achieved_tflops=1,
            total_time_us=2,
            dtype="float32",
            call_count=1,
        ),
    ]

    reduced = reduce_roofline_rows_by_config_stage(rows)

    assert [row.stage for row in reduced] == ["action_head_denoise", "vlm_prefill"]
    vlm = next(row for row in reduced if row.stage == "vlm_prefill")
    assert vlm.kernel_name == "openpi-bs4/vlm_prefill"
    assert vlm.arithmetic_intensity == pytest.approx(4.0)
    assert vlm.achieved_tflops == pytest.approx(8.0)
    assert vlm.total_time_us == pytest.approx(4.0)


def test_apply_stage_time_overrides_updates_reduced_point_sizes() -> None:
    rows = [
        AggregatedRooflineRow(
            config="openpi-bs4",
            stage="vlm_prefill",
            kernel_name="openpi-bs4/vlm_prefill",
            op_name="vlm_prefill",
            shape_bucket="stage_mean",
            flops=100,
            bytes=50,
            arithmetic_intensity=2,
            achieved_tflops=10,
            total_time_us=10,
            dtype="mixed",
            call_count=1,
        ),
        AggregatedRooflineRow(
            config="openpi-bs4",
            stage="action_head_denoise",
            kernel_name="openpi-bs4/action_head_denoise",
            op_name="action_head_denoise",
            shape_bucket="stage_mean",
            flops=10,
            bytes=40,
            arithmetic_intensity=0.25,
            achieved_tflops=1,
            total_time_us=5,
            dtype="mixed",
            call_count=1,
        ),
    ]

    updated = apply_stage_time_overrides(
        rows,
        {
            ("openpi-bs4", "vlm_prefill"): 200,
            ("openpi-bs4", "action_head_denoise"): 100,
        },
    )

    assert [row.total_time_us for row in updated] == [200, 100]
    assert updated[0].arithmetic_intensity == rows[0].arithmetic_intensity
    assert updated[1].achieved_tflops == rows[1].achieved_tflops


def test_plot_roofline_reduces_points_by_config_and_stage(tmp_path) -> None:
    rows = [
        AggregatedRooflineRow(
            config="openpi-bs4",
            stage="vlm_prefill",
            kernel_name=f"kernel_{idx}",
            op_name="aten::mm",
            shape_bucket=str(idx),
            flops=100,
            bytes=50,
            arithmetic_intensity=2 + idx,
            achieved_tflops=10 + idx,
            total_time_us=10,
            dtype="float32",
            call_count=1,
        )
        for idx in range(3)
    ]

    svg_path = tmp_path / "vla_roofline.svg"
    plot_roofline(
        rows,
        pdf_path=tmp_path / "vla_roofline.pdf",
        svg_path=svg_path,
        peak_tflops=312.0,
        peak_bw_gbs=1555.0,
    )

    assert svg_path.read_text(encoding="utf-8").count("bs4") == 1


def test_random_model_obs_supports_libero_gr00t() -> None:
    cfg = OmegaConf.create(
        {
            "env": {"eval": {"total_num_envs": 3}},
            "actor": {"model": {"state_dim": 8}},
        }
    )
    case = BenchmarkCase(
        case_id="libero-gr00t-bs3",
        scenario="model_only_mps",
        preset_name="libero_gr00t",
        env_type="libero",
        model_type="gr00t",
        num_envs=3,
    )

    obs = _make_random_model_obs(cfg, case)

    assert obs["main_images"].shape == (3, 256, 256, 3)
    assert obs["wrist_images"].shape == (3, 256, 256, 3)
    assert obs["states"].shape == (3, 8)
    assert obs["task_descriptions"] == ["do something"] * 3


def _synthetic_trace_path(case_id: str) -> Path:
    path = Path("/tmp") / f"rollout_eval_roofline_{case_id}.json"
    _write_trace(path)
    return path
