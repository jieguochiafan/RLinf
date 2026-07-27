"""Model-only OpenPI/GR00T roofline profiler for rollout evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from toolkits.rollout_eval.benchmark.metrics import aggregate_case_metrics
from toolkits.rollout_eval.benchmark.orchestrator import _make_random_model_obs
from toolkits.rollout_eval.benchmark.types import BenchmarkCase
from toolkits.rollout_eval.profiling.roofline import (
    RooflineKernelRow,
    aggregate_roofline_rows,
    parse_chrome_trace_to_rows,
    plot_roofline,
    summarize_roofline_rows,
    write_roofline_csv,
)


@dataclass(frozen=True)
class RooflineProfileCase:
    """One roofline profile case."""

    case_id: str
    model_type: str
    batch_size: int
    gpu: str


@dataclass(frozen=True)
class RooflineOutputPaths:
    """Output layout for one roofline profile run."""

    root: Path
    traces_dir: Path
    out_dir: Path
    fig_dir: Path
    aggregated_csv: Path
    figure_pdf: Path
    figure_svg: Path
    readme: Path


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_int_csv(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in _parse_csv(value))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse roofline profiler CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Profile OpenPI/GR00T model-only CUDA kernels for roofline plots"
    )
    parser.add_argument("--config-path", required=True, help="Hydra config directory")
    parser.add_argument("--config-name", required=True, help="Hydra config name")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override, repeatable.",
    )
    parser.add_argument(
        "--output-dir",
        default="./rollout_eval_output/roofline_profile",
        help="Output root containing traces/, out/, figs/, and README.md.",
    )
    parser.add_argument(
        "--gpu",
        required=True,
        help=(
            "Physical GPU id or CUDA_VISIBLE_DEVICES token to expose to this run. "
            "The profiler process sees it as cuda:0."
        ),
    )
    parser.add_argument(
        "--model-type",
        default="openpi",
        choices=["openpi", "gr00t"],
        help="Model family to profile.",
    )
    parser.add_argument(
        "--batch-sizes",
        default="1,8,32,128",
        help="Comma-separated synthetic LIBERO model batch sizes.",
    )
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument(
        "--record-shapes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable torch profiler shape recording for FLOP/byte estimates.",
    )
    parser.add_argument(
        "--profile-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable torch profiler memory recording.",
    )
    parser.add_argument(
        "--with-stack",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record Python stacks in profiler traces.",
    )
    parser.add_argument(
        "--peak-tflops",
        type=float,
        default=312.0,
        help="Hardware peak compute roof in TFLOP/s.",
    )
    parser.add_argument(
        "--peak-bw-gbs",
        type=float,
        default=1555.0,
        help="Hardware memory bandwidth roof in GB/s.",
    )
    parser.add_argument(
        "--skip-validate-cfg",
        action="store_true",
        help="Kept for CLI symmetry; this script does not call validate_cfg.",
    )
    return parser.parse_args(argv)


def build_cases(args: argparse.Namespace) -> list[RooflineProfileCase]:
    """Build the deterministic batch-size matrix for a roofline run."""
    batch_sizes = sorted(set(_parse_int_csv(args.batch_sizes)))
    if not batch_sizes or any(batch_size <= 0 for batch_size in batch_sizes):
        raise ValueError("--batch-sizes values must be positive integers")
    return [
        RooflineProfileCase(
            case_id=f"{args.model_type}-bs{batch_size}",
            model_type=str(args.model_type),
            batch_size=batch_size,
            gpu=str(args.gpu),
        )
        for batch_size in batch_sizes
    ]


def output_paths(output_dir: Path | str) -> RooflineOutputPaths:
    """Return the standard output paths for a roofline profile run."""
    root = Path(output_dir)
    return RooflineOutputPaths(
        root=root,
        traces_dir=root / "traces",
        out_dir=root / "out",
        fig_dir=root / "figs",
        aggregated_csv=root / "out" / "aggregated.csv",
        figure_pdf=root / "figs" / "vla_roofline.pdf",
        figure_svg=root / "figs" / "vla_roofline.svg",
        readme=root / "README.md",
    )


def _load_cfg(args: argparse.Namespace, case: RooflineProfileCase):
    abs_config_path = str(Path(args.config_path).resolve())
    with initialize_config_dir(version_base="1.1", config_dir=abs_config_path):
        cfg = compose(config_name=args.config_name, overrides=list(args.override))

    with open_dict(cfg):
        cfg.env.eval.env_type = "libero"
        cfg.env.eval.total_num_envs = int(case.batch_size)
        cfg.actor.model.model_type = case.model_type
        cfg.rollout.model.model_type = case.model_type
    return cfg


def _resolve_model_path(cfg) -> str:
    if "rollout" in cfg and "model" in cfg.rollout and "model_path" in cfg.rollout.model:
        return str(cfg.rollout.model.model_path)
    return str(cfg.actor.model.model_path)


def _build_model(cfg, model_type: str):
    from rlinf.models import get_model
    from toolkits.rollout_eval.adapters.model_adapter import (
        _validate_model_path_or_raise,
    )

    model_path = _resolve_model_path(cfg)
    _validate_model_path_or_raise(model_path, model_type=model_type)
    model_cfg = cfg.actor.model.copy()
    with open_dict(model_cfg):
        if "openpi_data" in cfg:
            model_cfg.openpi_data = cfg.openpi_data
        if "rollout" in cfg and "model" in cfg.rollout:
            if "precision" in cfg.rollout.model:
                model_cfg.precision = cfg.rollout.model.precision
            if "model_path" in cfg.rollout.model:
                model_cfg.model_path = cfg.rollout.model.model_path
    model = get_model(model_cfg)
    model.eval()
    return model


def _sampling_defaults(cfg) -> dict[str, Any]:
    sampling_cfg = cfg.algorithm.get("sampling_params", {})
    temp_eval = float(sampling_cfg.get("temperature_eval", -1))
    do_sample = bool(temp_eval > 0)
    return {
        "do_sample": do_sample,
        "temperature": temp_eval if do_sample else 1.0,
        "top_k": int(sampling_cfg.get("top_k", 0)),
    }


def _case_benchmark(case: RooflineProfileCase) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=case.case_id,
        scenario="roofline_model_only",
        preset_name=f"libero_{case.model_type}",
        env_type="libero",
        model_type=case.model_type,
        num_envs=case.batch_size,
    )


def _run_case(
    args: argparse.Namespace,
    case: RooflineProfileCase,
    paths: RooflineOutputPaths,
) -> tuple[dict[str, Any], list[RooflineKernelRow]]:
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function

    from toolkits.rollout_eval.adapters.model_adapter import GenericModelAdapter

    case_started = time.perf_counter()
    trace_path = paths.traces_dir / f"{case.case_id}.json"
    per_kernel_csv = paths.out_dir / f"per_kernel_{case.case_id}.csv"
    paths.traces_dir.mkdir(parents=True, exist_ok=True)
    paths.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for roofline profiling")
        torch.cuda.set_device(0)

        cfg = _load_cfg(args, case)
        model = _build_model(cfg, case.model_type)
        adapter = GenericModelAdapter(
            model=model,
            model_type=case.model_type,
            split_model_stages=True,
            sampling_defaults=_sampling_defaults(cfg),
        )
        obs_batch = _make_random_model_obs(cfg, _case_benchmark(case))

        for _ in range(int(args.warmup_steps)):
            with torch.inference_mode():
                adapter.infer(obs_batch=obs_batch, mode="eval")
            torch.cuda.synchronize()

        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        latencies_ms: list[float] = []
        gpu_times_ms: list[float] = []
        measured_seconds = 0.0
        prof = profile(
            activities=activities,
            record_shapes=bool(args.record_shapes),
            profile_memory=bool(args.profile_memory),
            with_stack=bool(args.with_stack),
            with_flops=True,
        )
        with prof:
            for _ in range(int(args.measure_steps)):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                started = time.perf_counter()
                with record_function(f"model.inference.{case.model_type}"):
                    with torch.inference_mode():
                        adapter.infer(obs_batch=obs_batch, mode="eval")
                end_event.record()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                prof.step()
                measured_seconds += elapsed
                latencies_ms.append(elapsed * 1000.0)
                gpu_times_ms.append(float(start_event.elapsed_time(end_event)))

        prof.export_chrome_trace(str(trace_path))
        rows = parse_chrome_trace_to_rows(trace_path, config_name=case.case_id)
        _write_per_kernel_csv(rows, per_kernel_csv)
        metrics = aggregate_case_metrics(
            model_infer_count=int(args.measure_steps),
            model_infer_seconds=measured_seconds,
            model_infer_latency_ms=latencies_ms,
            model_infer_gpu_time_ms=gpu_times_ms,
        )
        summary = summarize_roofline_rows(rows)
        record = {
            "case_id": case.case_id,
            "status": "pass",
            "model_type": case.model_type,
            "batch_size": case.batch_size,
            "gpu": case.gpu,
            "trace_path": str(trace_path),
            "per_kernel_csv": str(per_kernel_csv),
            "elapsed_s": time.perf_counter() - case_started,
            "metrics": asdict(metrics),
            "roofline": asdict(summary),
            "error_message": None,
        }
    except Exception as exc:  # noqa: BLE001
        rows = []
        record = {
            "case_id": case.case_id,
            "status": "failed",
            "model_type": case.model_type,
            "batch_size": case.batch_size,
            "gpu": case.gpu,
            "trace_path": str(trace_path),
            "per_kernel_csv": str(per_kernel_csv),
            "elapsed_s": time.perf_counter() - case_started,
            "metrics": None,
            "roofline": None,
            "error_message": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }
    _write_json(record, paths.out_dir / f"{case.case_id}.json")
    return record, rows


def _write_per_kernel_csv(rows: list[RooflineKernelRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "config",
        "stage",
        "kernel_name",
        "op_name",
        "shape_bucket",
        "flops",
        "bytes",
        "AI",
        "achieved_tflops",
        "dur_us",
        "dtype",
        "stream",
    ]
    import csv

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "config": row.config,
                    "stage": row.stage,
                    "kernel_name": row.kernel_name,
                    "op_name": row.op_name,
                    "shape_bucket": row.shape_bucket,
                    "flops": f"{row.flops:.10g}",
                    "bytes": f"{row.bytes:.10g}",
                    "AI": f"{row.arithmetic_intensity:.10g}",
                    "achieved_tflops": f"{row.achieved_tflops:.10g}",
                    "dur_us": f"{row.dur_us:.10g}",
                    "dtype": row.dtype,
                    "stream": row.stream,
                }
            )


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_readme(
    args: argparse.Namespace,
    paths: RooflineOutputPaths,
    records: list[dict[str, Any]],
) -> None:
    import torch

    gpu_name = "unknown"
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001
            gpu_name = "unknown"

    lines = [
        "# VLA Roofline Profile",
        "",
        f"- Model type: `{args.model_type}`",
        f"- Config: `{args.config_path}/{args.config_name}`",
        f"- GPU: `{gpu_name}` exposed from `{args.gpu}`",
        f"- Torch: `{torch.__version__}`",
        f"- CUDA: `{torch.version.cuda}`",
        f"- Peak compute roof: `{float(args.peak_tflops):.3f}` TFLOP/s",
        f"- Peak memory roof: `{float(args.peak_bw_gbs):.3f}` GB/s",
        f"- Warmup steps: `{int(args.warmup_steps)}`",
        f"- Measure steps: `{int(args.measure_steps)}`",
        "",
        "## Outputs",
        "",
        f"- Aggregated CSV: `{paths.aggregated_csv}`",
        f"- PDF: `{paths.figure_pdf}`",
        f"- SVG: `{paths.figure_svg}`",
        f"- Raw traces: `{paths.traces_dir}`",
        "",
        "## Assumptions",
        "",
        "- FLOPs use torch profiler values when present, with shape-based fallbacks "
        "for GEMM, BMM, layer norm, GELU/SiLU, and common elementwise ops.",
        "- DRAM bytes are shape-based lower-bound estimates; Nsight Compute "
        "calibration is not applied by this script.",
        "",
        "## Cases",
        "",
        "| case_id | status | batch | plotted kernels | FLOPs coverage | trace |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for record in records:
        roofline = record.get("roofline") or {}
        lines.append(
            "| {case_id} | {status} | {batch} | {kernels} | {coverage:.3f} | {trace} |".format(
                case_id=record.get("case_id", ""),
                status=record.get("status", ""),
                batch=record.get("batch_size", ""),
                kernels=int(roofline.get("plotted_kernel_count", 0) or 0),
                coverage=float(roofline.get("flops_coverage", 0.0) or 0.0),
                trace=record.get("trace_path", ""),
            )
        )
    paths.readme.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(
    paths: RooflineOutputPaths,
    records: list[dict[str, Any]],
    rows: list[RooflineKernelRow],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Write summary, aggregated CSV, roofline figure, and README."""
    counts = {
        "total": len(records),
        "pass": sum(1 for record in records if record.get("status") == "pass"),
        "failed": sum(1 for record in records if record.get("status") == "failed"),
    }
    aggregated = aggregate_roofline_rows(rows, top_k_per_stage=3)
    write_roofline_csv(aggregated, paths.aggregated_csv)
    if aggregated:
        plot_roofline(
            aggregated,
            pdf_path=paths.figure_pdf,
            svg_path=paths.figure_svg,
            peak_tflops=float(args.peak_tflops),
            peak_bw_gbs=float(args.peak_bw_gbs),
        )

    summary = {
        "counts": counts,
        "aggregated_csv": str(paths.aggregated_csv),
        "figure_pdf": str(paths.figure_pdf),
        "figure_svg": str(paths.figure_svg),
        "cases": records,
    }
    _write_json(summary, paths.root / "summary.json")
    _write_readme(args, paths, records)
    return summary


def main(argv: list[str] | None = None) -> None:
    """Run the model-only roofline profiler."""
    args = parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    paths = output_paths(args.output_dir)
    records: list[dict[str, Any]] = []
    all_rows: list[RooflineKernelRow] = []
    for case in build_cases(args):
        record, rows = _run_case(args, case, paths)
        records.append(record)
        all_rows.extend(rows)
        write_summary(paths, records, all_rows, args)
    print(json.dumps(write_summary(paths, records, all_rows, args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1:])
