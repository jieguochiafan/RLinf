"""Single-GPU OpenPI torch-profiler SM/occupancy batch sweep."""

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
from toolkits.rollout_eval.benchmark.types import BenchmarkCase


@dataclass(frozen=True)
class OpenPISMProfileCase:
    """One OpenPI single-GPU profile case."""

    case_id: str
    batch_size: int
    mps_sm: int
    gpu: str


@dataclass(frozen=True)
class TraceSMSummary:
    """CUDA-kernel summary extracted from a torch profiler Chrome trace."""

    trace_path: str
    kernel_count: int
    kernel_time_us: float
    trace_window_us: float
    kernel_busy_ratio: float
    occupancy_available: bool
    occupancy_avg_pct: float | None
    occupancy_max_pct: float | None


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_int_csv(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in _parse_csv(value))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for single-GPU OpenPI SM profiling."""
    parser = argparse.ArgumentParser(
        description="Profile OpenPI model-only inference SM usage with torch.profiler"
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
        default="./rollout_eval_output/openpi_sm_profile",
        help="Output directory for profile traces and reports.",
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
        "--mps-sm",
        type=int,
        default=100,
        help="CUDA_MPS_ACTIVE_THREAD_PERCENTAGE for the profiled process.",
    )
    parser.add_argument(
        "--batch-sizes",
        default="1,2,4,8,16,32",
        help="Comma-separated OpenPI inference batch sizes.",
    )
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=20)
    parser.add_argument(
        "--record-shapes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable torch profiler shape recording.",
    )
    parser.add_argument(
        "--profile-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable torch profiler memory recording.",
    )
    parser.add_argument(
        "--with-stack",
        action="store_true",
        help="Record Python stacks in profiler traces.",
    )
    parser.add_argument(
        "--skip-validate-cfg",
        action="store_true",
        help="Kept for CLI symmetry; this script does not call validate_cfg.",
    )
    return parser.parse_args(argv)


def build_cases(args: argparse.Namespace) -> list[OpenPISMProfileCase]:
    """Build deterministic batch-size cases for one GPU and one SM quota."""
    if args.mps_sm <= 0 or args.mps_sm > 100:
        raise ValueError("--mps-sm values must be in [1, 100]")
    batch_sizes = sorted(set(_parse_int_csv(args.batch_sizes)))
    if not batch_sizes or any(batch_size <= 0 for batch_size in batch_sizes):
        raise ValueError("--batch-sizes values must be positive integers")
    return [
        OpenPISMProfileCase(
            case_id=f"openpi-sm{args.mps_sm}-bs{batch_size}",
            batch_size=batch_size,
            mps_sm=int(args.mps_sm),
            gpu=str(args.gpu),
        )
        for batch_size in batch_sizes
    ]


def _load_cfg(args: argparse.Namespace, batch_size: int):
    abs_config_path = str(Path(args.config_path).resolve())
    with initialize_config_dir(version_base="1.1", config_dir=abs_config_path):
        cfg = compose(config_name=args.config_name, overrides=list(args.override))

    with open_dict(cfg):
        cfg.env.eval.env_type = "libero"
        cfg.env.eval.total_num_envs = int(batch_size)
        cfg.actor.model.model_type = "openpi"
        cfg.rollout.model.model_type = "openpi"
    return cfg


def _resolve_model_path(cfg) -> str:
    if "rollout" in cfg and "model" in cfg.rollout and "model_path" in cfg.rollout.model:
        return str(cfg.rollout.model.model_path)
    return str(cfg.actor.model.model_path)


def _build_openpi_model(cfg):
    from rlinf.models import get_model
    from toolkits.rollout_eval.adapters.model_adapter import (
        _validate_model_path_or_raise,
    )

    model_path = _resolve_model_path(cfg)
    _validate_model_path_or_raise(model_path, model_type="openpi")
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


def _trace_events(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        events = payload.get("traceEvents", [])
        return events if isinstance(events, list) else []
    if isinstance(payload, list):
        return payload
    return []


def _event_ts_dur(event: dict[str, Any]) -> tuple[float | None, float]:
    try:
        ts = float(event.get("ts"))
    except (TypeError, ValueError):
        ts = None
    try:
        dur = float(event.get("dur", 0.0) or 0.0)
    except (TypeError, ValueError):
        dur = 0.0
    return ts, max(0.0, dur)


def _is_kernel_event(event: dict[str, Any]) -> bool:
    cat = str(event.get("cat", "")).lower()
    name = str(event.get("name", "")).lower()
    args = event.get("args", {})
    if "kernel" in cat:
        return True
    if isinstance(args, dict) and any("occupancy" in str(key).lower() for key in args):
        return True
    return name.startswith("void ") or "cuda kernel" in name


def _extract_occupancy_pct(event: dict[str, Any]) -> float | None:
    args = event.get("args", {})
    if not isinstance(args, dict):
        return None
    for key, value in args.items():
        lowered = str(key).lower()
        if "occupancy" not in lowered and "sm efficiency" not in lowered:
            continue
        try:
            occupancy = float(str(value).rstrip("%"))
        except (TypeError, ValueError):
            continue
        if occupancy <= 1.0 and "%" not in str(value) and "pct" not in lowered:
            occupancy *= 100.0
        return occupancy
    return None


def summarize_chrome_trace(trace_path: Path) -> TraceSMSummary:
    """Summarize CUDA kernel duration and occupancy fields from a Chrome trace."""
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    events = _trace_events(payload)

    min_ts: float | None = None
    max_ts: float | None = None
    kernel_time_us = 0.0
    kernel_count = 0
    occupancies: list[float] = []

    for event in events:
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        ts, dur = _event_ts_dur(event)
        if ts is not None:
            min_ts = ts if min_ts is None else min(min_ts, ts)
            max_ts = ts + dur if max_ts is None else max(max_ts, ts + dur)
        if not _is_kernel_event(event):
            continue
        kernel_count += 1
        kernel_time_us += dur
        occupancy = _extract_occupancy_pct(event)
        if occupancy is not None:
            occupancies.append(occupancy)

    trace_window_us = max(0.0, (max_ts or 0.0) - (min_ts or 0.0))
    kernel_busy_ratio = kernel_time_us / trace_window_us if trace_window_us > 0 else 0.0
    kernel_busy_ratio = min(1.0, kernel_busy_ratio)
    return TraceSMSummary(
        trace_path=str(trace_path),
        kernel_count=kernel_count,
        kernel_time_us=kernel_time_us,
        trace_window_us=trace_window_us,
        kernel_busy_ratio=kernel_busy_ratio,
        occupancy_available=bool(occupancies),
        occupancy_avg_pct=(
            sum(occupancies) / len(occupancies) if occupancies else None
        ),
        occupancy_max_pct=max(occupancies) if occupancies else None,
    )


def _case_dir(output_dir: Path, case: OpenPISMProfileCase) -> Path:
    return output_dir / "cases" / case.case_id


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _run_case(args: argparse.Namespace, case: OpenPISMProfileCase) -> dict[str, Any]:
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function

    from toolkits.rollout_eval.adapters.model_adapter import GenericModelAdapter
    from toolkits.rollout_eval.benchmark.orchestrator import _make_random_model_obs

    case_dir = _case_dir(Path(args.output_dir), case)
    case_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for OpenPI SM profiling")
        torch.cuda.set_device(0)

        cfg = _load_cfg(args, case.batch_size)
        model = _build_openpi_model(cfg)
        adapter = GenericModelAdapter(
            model=model,
            model_type="openpi",
            split_model_stages=True,
            sampling_defaults=_sampling_defaults(cfg),
        )
        obs_batch = _make_random_model_obs(
            cfg,
            BenchmarkCase(
                case_id=case.case_id,
                scenario="openpi_sm_profile",
                preset_name="libero_openpi",
                env_type="libero",
                model_type="openpi",
                num_envs=case.batch_size,
                mps_sm=case.mps_sm,
            ),
        )

        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        latencies_ms: list[float] = []
        gpu_times_ms: list[float] = []
        measured_seconds = 0.0
        trace_path = case_dir / "torch_trace.json"

        prof = profile(
            activities=activities,
            record_shapes=bool(args.record_shapes),
            profile_memory=bool(args.profile_memory),
            with_stack=bool(args.with_stack),
        )
        with prof:
            for step_idx in range(int(args.warmup_steps) + int(args.measure_steps)):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                start = time.perf_counter()
                with record_function("model.inference.openpi"):
                    adapter.infer(obs_batch=obs_batch, mode="eval")
                end_event.record()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                prof.step()

                if step_idx >= int(args.warmup_steps):
                    measured_seconds += elapsed
                    latencies_ms.append(elapsed * 1000.0)
                    gpu_times_ms.append(float(start_event.elapsed_time(end_event)))

        prof.export_chrome_trace(str(trace_path))

        metrics = aggregate_case_metrics(
            model_infer_count=int(args.measure_steps),
            model_infer_seconds=measured_seconds,
            model_infer_latency_ms=latencies_ms,
            model_infer_gpu_time_ms=gpu_times_ms,
        )
        sm_summary = summarize_chrome_trace(trace_path)
        record = {
            "case_id": case.case_id,
            "status": "pass",
            "batch_size": case.batch_size,
            "mps_sm": case.mps_sm,
            "gpu": case.gpu,
            "elapsed_s": time.perf_counter() - started,
            "metrics": asdict(metrics),
            "sm_profile": asdict(sm_summary),
            "error_message": None,
        }
    except Exception as exc:  # noqa: BLE001
        record = {
            "case_id": case.case_id,
            "status": "failed",
            "batch_size": case.batch_size,
            "mps_sm": case.mps_sm,
            "gpu": case.gpu,
            "elapsed_s": time.perf_counter() - started,
            "metrics": None,
            "sm_profile": None,
            "error_message": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }

    _write_json(record, case_dir / "case_report.json")
    return record


def write_summary(output_dir: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Write JSON and Markdown summaries for OpenPI SM profile records."""
    counts = {
        "total": len(records),
        "pass": sum(1 for record in records if record.get("status") == "pass"),
        "failed": sum(1 for record in records if record.get("status") == "failed"),
    }
    summary = {"counts": counts, "cases": records}
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(summary, output_dir / "summary.json")

    lines = [
        "# OpenPI Single-GPU SM Profile Summary",
        "",
        f"- Total: {counts['total']}",
        f"- Pass: {counts['pass']}",
        f"- Failed: {counts['failed']}",
        "",
        "| case_id | status | batch | mps_sm | infer/s | samples/s | gpu_ms(avg) | kernel_busy | occupancy_avg_pct | occupancy_max_pct | trace |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for record in records:
        metrics = record.get("metrics") or {}
        sm_profile = record.get("sm_profile") or {}
        infer_per_sec = float(metrics.get("model_infers_per_sec", 0.0))
        batch_size = int(record.get("batch_size") or 0)
        gpu_avg = float((metrics.get("model_infer_gpu_time_ms") or {}).get("avg_ms", 0.0))
        occupancy_avg = sm_profile.get("occupancy_avg_pct")
        occupancy_max = sm_profile.get("occupancy_max_pct")
        lines.append(
            "| {case_id} | {status} | {batch} | {mps_sm} | {infer:.6f} | "
            "{samples:.6f} | {gpu_avg:.3f} | {busy:.6f} | {occ_avg} | "
            "{occ_max} | {trace} |".format(
                case_id=record.get("case_id", ""),
                status=record.get("status", ""),
                batch=batch_size or "",
                mps_sm=record.get("mps_sm", ""),
                infer=infer_per_sec,
                samples=infer_per_sec * float(batch_size),
                gpu_avg=gpu_avg,
                busy=float(sm_profile.get("kernel_busy_ratio", 0.0)),
                occ_avg=(
                    f"{float(occupancy_avg):.3f}"
                    if occupancy_avg is not None
                    else "n/a"
                ),
                occ_max=(
                    f"{float(occupancy_max):.3f}"
                    if occupancy_max is not None
                    else "n/a"
                ),
                trace=sm_profile.get("trace_path", ""),
            )
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    """Run OpenPI single-GPU torch profiler batch sweep."""
    args = parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(args.mps_sm)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    output_dir = Path(args.output_dir)
    records = []
    for case in build_cases(args):
        records.append(_run_case(args, case))
        write_summary(output_dir, records)
    print(json.dumps(write_summary(output_dir, records), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1:])
