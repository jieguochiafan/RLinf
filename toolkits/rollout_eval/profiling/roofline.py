"""Roofline trace parsing, aggregation, and plotting for rollout eval."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RooflineKernelRow:
    """One CUDA kernel event with estimated roofline inputs."""

    config: str
    stage: str
    kernel_name: str
    op_name: str
    shape_bucket: str
    flops: float
    bytes: float
    dur_us: float
    dtype: str
    stream: str

    @property
    def arithmetic_intensity(self) -> float:
        return self.flops / self.bytes if self.flops > 0 and self.bytes > 0 else math.nan

    @property
    def achieved_tflops(self) -> float:
        if self.flops <= 0 or self.dur_us <= 0:
            return math.nan
        return self.flops / (self.dur_us * 1e-6) / 1e12


@dataclass(frozen=True)
class AggregatedRooflineRow:
    """Aggregated roofline point for one kernel/op/shape group."""

    config: str
    stage: str
    kernel_name: str
    op_name: str
    shape_bucket: str
    flops: float
    bytes: float
    arithmetic_intensity: float
    achieved_tflops: float
    total_time_us: float
    dtype: str
    call_count: int
    annotation_rank: int | None = None


@dataclass(frozen=True)
class RooflineSummary:
    """Sanity metrics for a parsed trace."""

    total_kernel_time_us: float
    plotted_kernel_time_us: float
    flops_coverage: float
    kernel_count: int
    plotted_kernel_count: int


_STAGE_ALIASES = (
    ("model.action_head", "action_head_denoise"),
    ("action_head_denoise", "action_head_denoise"),
    ("model.backbone", "vlm_prefill"),
    ("vlm_prefill", "vlm_prefill"),
    ("vlm_decode", "vlm_decode"),
    ("train_bwd", "train_bwd"),
    ("train_fwd", "train_fwd"),
    ("model.inference", "vlm_decode"),
)

_DTYPE_SIZES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "half": 2,
    "float16": 2,
    "bfloat16": 2,
    "c10::half": 2,
    "c10::bfloat16": 2,
    "short": 2,
    "int16": 2,
    "float": 4,
    "float32": 4,
    "int": 4,
    "int32": 4,
    "long": 8,
    "int64": 8,
    "double": 8,
    "float64": 8,
}


def parse_chrome_trace_to_rows(
    trace_path: Path | str,
    *,
    config_name: str,
) -> list[RooflineKernelRow]:
    """Parse torch profiler Chrome trace JSON into per-kernel roofline rows."""
    trace_path = Path(trace_path)
    payload = json.loads(
        trace_path.read_text(encoding="utf-8", errors="replace"),
        strict=False,
    )
    events = _trace_events(payload)

    external_to_op: dict[int, dict[str, Any]] = {}
    correlation_to_external: dict[int, int] = {}
    cpu_ops: list[dict[str, Any]] = []
    stage_events: list[dict[str, Any]] = []
    kernels: list[dict[str, Any]] = []

    for event in events:
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        cat = str(event.get("cat", "")).lower()
        args = _event_args(event)
        external_id = _coerce_int(args.get("External id"))
        if cat == "cuda_runtime":
            correlation = _coerce_int(args.get("correlation"))
            if correlation is not None and external_id is not None:
                correlation_to_external[correlation] = external_id
            continue
        if _is_kernel_event(event):
            kernels.append(event)
            continue
        if cat in {"cpu_op", "user_annotation"}:
            name = str(event.get("name", ""))
            op_record = _event_record(event)
            if _is_stage_name(name):
                stage_events.append(op_record)
            elif cat == "cpu_op":
                cpu_ops.append(op_record)
                if external_id is not None:
                    external_to_op[external_id] = event

    rows: list[RooflineKernelRow] = []
    for kernel in kernels:
        args = _event_args(kernel)
        external_id = _coerce_int(args.get("External id"))
        if external_id is None:
            correlation = _coerce_int(args.get("correlation"))
            if correlation is not None:
                external_id = correlation_to_external.get(correlation)
        op_event = external_to_op.get(external_id) if external_id is not None else None
        if op_event is None:
            op_event = _find_enclosing_op(kernel, cpu_ops)

        stage_name = _resolve_stage_name(kernel, op_event, stage_events)
        if stage_name is None:
            continue
        op_name = str(op_event.get("name", "unknown_op")) if op_event else "unknown_op"
        op_args = _event_args(op_event) if op_event else {}
        shapes = _extract_shapes(op_args)
        dtypes = _extract_dtypes(op_args)
        dtype = _first_dtype(dtypes, kernel_name=str(kernel.get("name", "")))
        flops = _extract_flops(op_args)
        if flops is None:
            flops = _estimate_flops(op_name, shapes)
        bytes_estimate = _estimate_bytes(op_name, shapes, dtypes or [dtype])
        dur_us = _event_duration_us(kernel)

        if flops is None or flops <= 0 or bytes_estimate <= 0 or dur_us <= 0:
            continue
        rows.append(
            RooflineKernelRow(
                config=config_name,
                stage=stage_name,
                kernel_name=str(kernel.get("name", "")),
                op_name=op_name,
                shape_bucket=_shape_bucket(shapes),
                flops=float(flops),
                bytes=float(bytes_estimate),
                dur_us=dur_us,
                dtype=dtype,
                stream=str(args.get("stream", "")),
            )
        )
    return rows


def aggregate_roofline_rows(
    rows: list[RooflineKernelRow],
    *,
    top_k_per_stage: int = 3,
) -> list[AggregatedRooflineRow]:
    """Aggregate per-kernel rows into roofline scatter points."""
    grouped: dict[tuple[str, str, str, str, str, str], list[RooflineKernelRow]] = defaultdict(list)
    for row in rows:
        grouped[
            (row.config, row.stage, row.kernel_name, row.op_name, row.shape_bucket, row.dtype)
        ].append(row)

    aggregated: list[AggregatedRooflineRow] = []
    for (config, stage, kernel_name, op_name, shape_bucket, dtype), items in grouped.items():
        total_time_us = sum(item.dur_us for item in items)
        total_flops = sum(item.flops for item in items)
        total_bytes = sum(item.bytes for item in items)
        achieved = total_flops / (total_time_us * 1e-6) / 1e12 if total_time_us > 0 else math.nan
        ai = total_flops / total_bytes if total_bytes > 0 else math.nan
        aggregated.append(
            AggregatedRooflineRow(
                config=config,
                stage=stage,
                kernel_name=kernel_name,
                op_name=op_name,
                shape_bucket=shape_bucket,
                flops=total_flops / len(items),
                bytes=total_bytes / len(items),
                arithmetic_intensity=ai,
                achieved_tflops=achieved,
                total_time_us=total_time_us,
                dtype=dtype,
                call_count=len(items),
            )
        )

    rank_by_key: dict[tuple[str, str, str, str, str, str], int] = {}
    by_stage: dict[str, list[AggregatedRooflineRow]] = defaultdict(list)
    for row in aggregated:
        by_stage[row.stage].append(row)
    for stage_rows in by_stage.values():
        for rank, row in enumerate(
            sorted(stage_rows, key=lambda item: item.total_time_us, reverse=True)[
                :top_k_per_stage
            ],
            start=1,
        ):
            rank_by_key[_aggregate_key(row)] = rank

    return [
        AggregatedRooflineRow(
            **{
                **row.__dict__,
                "annotation_rank": rank_by_key.get(_aggregate_key(row)),
            }
        )
        for row in sorted(
            aggregated,
            key=lambda item: (item.config, item.stage, -item.total_time_us, item.kernel_name),
        )
    ]


def summarize_roofline_rows(
    rows: list[RooflineKernelRow], *, total_kernel_time_us: float | None = None
) -> RooflineSummary:
    """Return basic sanity metrics for plotted roofline rows."""
    plotted_time = sum(row.dur_us for row in rows)
    total_time = float(total_kernel_time_us if total_kernel_time_us is not None else plotted_time)
    return RooflineSummary(
        total_kernel_time_us=total_time,
        plotted_kernel_time_us=plotted_time,
        flops_coverage=plotted_time / total_time if total_time > 0 else 0.0,
        kernel_count=len(rows),
        plotted_kernel_count=len(rows),
    )


def write_roofline_csv(rows: list[AggregatedRooflineRow], path: Path | str) -> None:
    """Write aggregated roofline rows to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
        "call_count",
        "annotation_rank",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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
                    "total_time_us": f"{row.total_time_us:.10g}",
                    "dtype": row.dtype,
                    "call_count": row.call_count,
                    "annotation_rank": row.annotation_rank or "",
                }
            )


def load_roofline_csv(path: Path | str) -> list[AggregatedRooflineRow]:
    """Load aggregated roofline rows from CSV."""
    rows: list[AggregatedRooflineRow] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            rows.append(
                AggregatedRooflineRow(
                    config=record["config"],
                    stage=record["stage"],
                    kernel_name=record["kernel_name"],
                    op_name=record["op_name"],
                    shape_bucket=record["shape_bucket"],
                    flops=float(record["flops"]),
                    bytes=float(record["bytes"]),
                    arithmetic_intensity=float(record["AI"]),
                    achieved_tflops=float(record["achieved_tflops"]),
                    total_time_us=float(record["total_time_us"]),
                    dtype=record["dtype"],
                    call_count=int(record.get("call_count") or 1),
                    annotation_rank=(
                        int(record["annotation_rank"])
                        if record.get("annotation_rank")
                        else None
                    ),
                )
            )
    return rows


def reduce_roofline_rows_by_config_stage(
    rows: list[AggregatedRooflineRow],
) -> list[AggregatedRooflineRow]:
    """Reduce kernel-level roofline rows to one point per config/stage."""
    grouped: dict[tuple[str, str], list[AggregatedRooflineRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.config, _normalized_plot_stage(row.stage))].append(row)

    reduced: list[AggregatedRooflineRow] = []
    for (config, stage), items in grouped.items():
        total_time_us = sum(item.total_time_us for item in items)
        if total_time_us <= 0:
            continue
        total_flops = sum(item.flops * item.call_count for item in items)
        total_bytes = sum(item.bytes * item.call_count for item in items)
        call_count = sum(item.call_count for item in items)
        reduced.append(
            AggregatedRooflineRow(
                config=config,
                stage=stage,
                kernel_name=f"{config}/{stage}",
                op_name=stage,
                shape_bucket="stage_mean",
                flops=total_flops / call_count if call_count > 0 else 0.0,
                bytes=total_bytes / call_count if call_count > 0 else 0.0,
                arithmetic_intensity=_weighted_mean(
                    [item.arithmetic_intensity for item in items],
                    [item.total_time_us for item in items],
                ),
                achieved_tflops=_weighted_mean(
                    [item.achieved_tflops for item in items],
                    [item.total_time_us for item in items],
                ),
                total_time_us=total_time_us,
                dtype="mixed",
                call_count=call_count,
            )
        )
    return sorted(reduced, key=lambda item: (_batch_sort_key(item.config), item.stage))


def apply_stage_time_overrides(
    rows: list[AggregatedRooflineRow],
    stage_time_overrides: dict[tuple[str, str], float],
) -> list[AggregatedRooflineRow]:
    """Replace point-size time with externally measured stage wall time."""
    updated: list[AggregatedRooflineRow] = []
    for row in rows:
        time_us = stage_time_overrides.get((row.config, row.stage))
        if time_us is None:
            updated.append(row)
            continue
        updated.append(
            AggregatedRooflineRow(
                **{
                    **row.__dict__,
                    "total_time_us": float(time_us),
                }
            )
        )
    return updated


def plot_roofline(
    rows: list[AggregatedRooflineRow],
    *,
    pdf_path: Path | str,
    svg_path: Path | str,
    peak_tflops: float,
    peak_bw_gbs: float,
    title: str | None = None,
    reduce_by_config_stage: bool = True,
    stage_time_overrides: dict[tuple[str, str], float] | None = None,
) -> tuple[str, str]:
    """Draw and save a roofline scatter plot."""
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import ScalarFormatter

    plot_rows = reduce_roofline_rows_by_config_stage(rows) if reduce_by_config_stage else rows
    if stage_time_overrides:
        plot_rows = apply_stage_time_overrides(plot_rows, stage_time_overrides)
    finite_rows = [
        row
        for row in plot_rows
        if row.arithmetic_intensity > 0
        and row.achieved_tflops > 0
        and math.isfinite(row.arithmetic_intensity)
        and math.isfinite(row.achieved_tflops)
    ]
    if not finite_rows:
        raise ValueError("No finite roofline points to plot")

    colors = {
        "vlm_prefill": "#2D4059",
        "vlm_decode": "#9DBDFF",
        "action_head_denoise": "#EA5455",
        "train_fwd": "#76BA99",
        "train_bwd": "#16A3A6",
        "llm_decode_ref": "#FFD460",
    }
    labels = {
        "vlm_prefill": "VLM",
        "vlm_decode": "VLM decode",
        "action_head_denoise": "Action denoise",
        "train_fwd": "Train fwd",
        "train_bwd": "Train bwd",
        "llm_decode_ref": "LLM decode ref",
    }

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    max_time = max(row.total_time_us for row in finite_rows)
    for stage in sorted({row.stage for row in finite_rows}):
        stage_rows = [row for row in finite_rows if row.stage == stage]
        xs = np.asarray([row.arithmetic_intensity for row in stage_rows], dtype=float)
        ys = np.asarray([row.achieved_tflops for row in stage_rows], dtype=float)
        sizes = np.asarray(
            [36.0 + 110.0 * math.sqrt(row.total_time_us / max_time) for row in stage_rows],
            dtype=float,
        )
        ax.scatter(
            xs,
            ys,
            s=sizes,
            color=colors.get(stage, "darkgray"),
            edgecolor="black",
            linewidth=0.6,
            alpha=0.82,
            label=labels.get(stage, stage),
            zorder=10,
        )

    for row in finite_rows:
        ax.annotate(
            _batch_label(row.config),
            (row.arithmetic_intensity, row.achieved_tflops),
            textcoords="offset points",
            xytext=(2, -7),
            fontsize=5.5,
            color="#404040",
            alpha=0.85,
            zorder=25,
        )

    ridge = peak_tflops / (peak_bw_gbs / 1000.0)
    x_max = max(max(row.arithmetic_intensity for row in finite_rows) * 1.15, ridge * 1.15, 1.0)
    y_max = max(max(row.achieved_tflops for row in finite_rows) * 1.25, peak_tflops * 1.08)
    roof_xs = np.linspace(0.0, x_max, 256)
    memory_roof = (peak_bw_gbs / 1000.0) * roof_xs
    ax.plot(
        roof_xs,
        np.minimum(memory_roof, peak_tflops),
        color="black",
        linewidth=1.4,
        label="Roofline",
        zorder=8,
    )
    ax.axhline(peak_tflops, color="black", linestyle="--", linewidth=1.0, zorder=6)
    ax.scatter([ridge], [peak_tflops], marker="x", color="black", s=45, zorder=20)
    ax.annotate(
        "ridge",
        (ridge, peak_tflops),
        textcoords="offset points",
        xytext=(5, -12),
        fontsize=8,
    )

    for row in finite_rows:
        if row.annotation_rank is None:
            continue
        ax.annotate(
            _short_kernel_label(row.kernel_name, row.op_name),
            (row.arithmetic_intensity, row.achieved_tflops),
            textcoords="offset points",
            xytext=(5, 7),
            fontsize=6.2,
            zorder=30,
        )

    ax.set_xlim(left=0.0, right=x_max)
    ax.set_ylim(bottom=0.0, top=y_max)
    ax.set_xlabel("Arithmetic Intensity (FLOP/Byte)", fontdict=_font(size=10))
    ax.set_ylabel("Achieved Performance (TFLOP/s)", fontdict=_font(size=10))
    if title:
        ax.set_title(title, fontdict=_font(size=10))
    ax.tick_params(axis="both", labelsize=8)
    for axis in (ax.xaxis, ax.yaxis):
        formatter = ScalarFormatter(useMathText=False)
        formatter.set_scientific(False)
        formatter.set_useOffset(False)
        axis.set_major_formatter(formatter)
    ax.grid(True, linestyle="--", linewidth=0.5, color="gray", alpha=0.5)
    ax.legend(
        ncol=2,
        fontsize=7,
        handletextpad=0.2,
        handlelength=1.0,
        columnspacing=0.7,
        loc="lower right",
    )
    fig.tight_layout()
    pdf_path = Path(pdf_path)
    svg_path = Path(svg_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    scales = (ax.get_xscale(), ax.get_yscale())
    plt.close(fig)
    return scales


def _trace_events(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        events = payload.get("traceEvents", [])
        return events if isinstance(events, list) else []
    return payload if isinstance(payload, list) else []


def _batch_label(config: str) -> str:
    match = re.search(r"(?:^|[-_])bs(\d+)(?:$|[-_])", config)
    if match:
        return f"bs{match.group(1)}"
    return config


def _normalized_plot_stage(stage: str) -> str:
    if stage == "vlm_decode":
        return "vlm_prefill"
    return stage


def _batch_sort_key(config: str) -> tuple[int, str]:
    match = re.search(r"(?:^|[-_])bs(\d+)(?:$|[-_])", config)
    if match:
        return int(match.group(1)), config
    return 10**9, config


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    total_weight = sum(weights)
    if total_weight <= 0:
        return math.nan
    return sum(value * weight for value, weight in zip(values, weights, strict=True)) / total_weight


def _event_args(event: dict[str, Any] | None) -> dict[str, Any]:
    if not event:
        return {}
    args = event.get("args", {})
    return args if isinstance(args, dict) else {}


def _event_record(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(event.get("name", "")),
        "ts": _event_ts_us(event),
        "end": _event_ts_us(event) + _event_duration_us(event),
        "tid": event.get("tid"),
        "event": event,
    }


def _event_ts_us(event: dict[str, Any]) -> float:
    try:
        return float(event.get("ts", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _event_duration_us(event: dict[str, Any]) -> float:
    try:
        return max(float(event.get("dur", 0.0) or 0.0), 0.0)
    except (TypeError, ValueError):
        return 0.0


def _is_kernel_event(event: dict[str, Any]) -> bool:
    cat = str(event.get("cat", "")).lower()
    name = str(event.get("name", "")).lower()
    return "kernel" in cat or name.startswith("void ") or "cuda kernel" in name


def _is_stage_name(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token, _alias in _STAGE_ALIASES)


def _resolve_stage_name(
    kernel: dict[str, Any],
    op_event: dict[str, Any] | None,
    stage_events: list[dict[str, Any]],
) -> str | None:
    priority_stage = _find_enclosing_priority_stage(op_event or kernel, stage_events)
    if priority_stage is not None:
        return _stage_alias(priority_stage["name"])
    candidates = [op_event] if op_event else []
    for event in candidates:
        stage = _stage_alias(str(event.get("name", "")))
        if stage:
            return stage
    enclosing = _find_enclosing_stage(op_event or kernel, stage_events)
    if enclosing is not None:
        return _stage_alias(enclosing["name"])
    return None


def _find_enclosing_priority_stage(
    event: dict[str, Any], stage_events: list[dict[str, Any]]
) -> dict[str, Any] | None:
    matches = [
        record
        for record in stage_events
        if _stage_alias(record["name"]) == "action_head_denoise"
    ]
    return _find_enclosing_stage(event, matches)


def _stage_alias(name: str) -> str | None:
    lowered = name.lower()
    for token, alias in _STAGE_ALIASES:
        if token in lowered:
            return alias
    return None


def _find_enclosing_op(
    kernel: dict[str, Any], cpu_ops: list[dict[str, Any]]
) -> dict[str, Any] | None:
    ts = _event_ts_us(kernel)
    matches = [
        record
        for record in cpu_ops
        if record["ts"] <= ts <= record["end"] and _op_can_estimate(record["name"])
    ]
    if matches:
        return min(matches, key=lambda record: record["end"] - record["ts"])["event"]
    return None


def _find_enclosing_stage(
    event: dict[str, Any], stage_events: list[dict[str, Any]]
) -> dict[str, Any] | None:
    ts = _event_ts_us(event)
    end = ts + _event_duration_us(event)
    matches = [
        record
        for record in stage_events
        if record["ts"] <= ts and end <= record["end"]
    ]
    if not matches:
        matches = [record for record in stage_events if record["ts"] <= ts <= record["end"]]
    if not matches:
        return None
    return min(matches, key=lambda record: record["end"] - record["ts"])


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_shapes(args: dict[str, Any]) -> list[list[int]]:
    for key in ("Input Dims", "Input dims", "input_dims", "Input Shapes", "input_shapes"):
        raw = args.get(key)
        shapes = _normalize_shapes(raw)
        if shapes:
            return shapes
    return []


def _normalize_shapes(raw: Any) -> list[list[int]]:
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return _parse_shape_string(raw)
    if not isinstance(raw, list):
        return []
    shapes: list[list[int]] = []
    for item in raw:
        if isinstance(item, (list, tuple)):
            dims = []
            for dim in item:
                try:
                    dims.append(int(dim))
                except (TypeError, ValueError):
                    pass
            if dims:
                shapes.append(dims)
    return shapes


def _parse_shape_string(value: str) -> list[list[int]]:
    shapes: list[list[int]] = []
    for match in re.finditer(r"\[([0-9,\s-]+)\]", value):
        dims = [int(dim) for dim in match.group(1).split(",") if dim.strip()]
        if dims:
            shapes.append(dims)
    return shapes


def _extract_dtypes(args: dict[str, Any]) -> list[str]:
    for key in ("Input type", "Input Types", "input_types", "dtypes"):
        raw = args.get(key)
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return [raw]
        if isinstance(raw, list):
            return [str(item) for item in raw if str(item)]
    return []


def _first_dtype(dtypes: list[str], *, kernel_name: str) -> str:
    if dtypes:
        return _normalize_dtype_name(dtypes[0])
    lowered = kernel_name.lower()
    if "bf16" in lowered or "bfloat16" in lowered:
        return "bfloat16"
    if "fp16" in lowered or "half" in lowered or "f16" in lowered:
        return "float16"
    return "float32"


def _normalize_dtype_name(dtype: str) -> str:
    dtype = dtype.replace("torch.", "").replace("c10::", "").strip().lower()
    if dtype in {"bfloat16", "bfloat", "bf16"}:
        return "bfloat16"
    if dtype in {"half", "float16", "fp16"}:
        return "float16"
    if dtype in {"float", "float32", "fp32"}:
        return "float32"
    return dtype


def _dtype_size(dtype: str) -> int:
    normalized = _normalize_dtype_name(dtype)
    return _DTYPE_SIZES.get(normalized, 4)


def _extract_flops(args: dict[str, Any]) -> float | None:
    for key in ("FLOPs", "flops", "Flops", "FLOPS"):
        value = args.get(key)
        try:
            flops = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(flops) and flops > 0:
            return flops
    return None


def _estimate_flops(op_name: str, shapes: list[list[int]]) -> float | None:
    lowered = op_name.lower()
    if any(token in lowered for token in ("aten::mm", "aten::matmul", "aten::addmm")) and len(shapes) >= 2:
        m, k = _matrix_mk(shapes[0])
        k2, n = _matrix_kn(shapes[1])
        if m and n and (k or k2):
            return float(2 * m * n * (k or k2))
    if "aten::bmm" in lowered and len(shapes) >= 2:
        if len(shapes[0]) >= 3 and len(shapes[1]) >= 3:
            b, m, k = shapes[0][-3:]
            _b2, _k2, n = shapes[1][-3:]
            return float(2 * b * m * n * k)
    if "layer_norm" in lowered or "native_layer_norm" in lowered:
        return float(8 * _numel(shapes[0])) if shapes else None
    if any(token in lowered for token in ("silu", "gelu")):
        return float(4 * _numel(shapes[0])) if shapes else None
    if any(token in lowered for token in ("aten::add", "aten::sub", "aten::mul", "aten::div")):
        return float(max((_numel(shape) for shape in shapes), default=0))
    if any(token in lowered for token in ("aten::neg", "aten::rsqrt", "aten::pow")):
        return float(_numel(shapes[0])) if shapes else None
    return None


def _matrix_mk(shape: list[int]) -> tuple[int | None, int | None]:
    if len(shape) < 2:
        return None, None
    return _numel(shape[:-1]), shape[-1]


def _matrix_kn(shape: list[int]) -> tuple[int | None, int | None]:
    if len(shape) < 2:
        return None, None
    return shape[-2], shape[-1]


def _estimate_bytes(op_name: str, shapes: list[list[int]], dtypes: list[str]) -> float:
    if not shapes:
        return 0.0
    lowered = op_name.lower()
    dtype_sizes = [_dtype_size(dtype) for dtype in dtypes] or [4]
    total = 0.0
    for idx, shape in enumerate(shapes):
        total += _numel(shape) * dtype_sizes[min(idx, len(dtype_sizes) - 1)]
    output_size = dtype_sizes[0]
    output_numel = _estimate_output_numel(lowered, shapes)
    if output_numel > 0:
        total += output_numel * output_size
    return total


def _estimate_output_numel(op_name: str, shapes: list[list[int]]) -> int:
    if any(token in op_name for token in ("aten::mm", "aten::matmul", "aten::addmm")) and len(shapes) >= 2:
        m, _k = _matrix_mk(shapes[0])
        _k2, n = _matrix_kn(shapes[1])
        if m and n:
            return m * n
    if "aten::bmm" in op_name and len(shapes) >= 2 and len(shapes[0]) >= 3:
        b, m, _k = shapes[0][-3:]
        n = shapes[1][-1]
        return b * m * n
    return max((_numel(shape) for shape in shapes), default=0)


def _numel(shape: list[int]) -> int:
    total = 1
    for dim in shape:
        total *= max(int(dim), 1)
    return total


def _shape_bucket(shapes: list[list[int]]) -> str:
    if not shapes:
        return ""
    buckets = []
    for shape in shapes[:4]:
        if not shape:
            continue
        canonical = ["B", *(str(dim) for dim in shape[1:])] if len(shape) > 1 else [str(shape[0])]
        buckets.append("x".join(canonical))
    return ";".join(buckets)


def _op_can_estimate(op_name: str) -> bool:
    lowered = op_name.lower()
    return any(
        token in lowered
        for token in (
            "aten::mm",
            "aten::matmul",
            "aten::addmm",
            "aten::bmm",
            "layer_norm",
            "silu",
            "gelu",
            "aten::add",
            "aten::sub",
            "aten::mul",
            "aten::div",
            "aten::neg",
            "aten::rsqrt",
            "aten::pow",
        )
    )


def _aggregate_key(row: AggregatedRooflineRow) -> tuple[str, str, str, str, str, str]:
    return (
        row.config,
        row.stage,
        row.kernel_name,
        row.op_name,
        row.shape_bucket,
        row.dtype,
    )


def _short_kernel_label(kernel_name: str, op_name: str) -> str:
    if op_name and op_name != "unknown_op":
        return op_name.replace("aten::", "")
    cleaned = kernel_name.replace("void ", "")
    cleaned = cleaned.split("<", maxsplit=1)[0]
    cleaned = cleaned.split("(", maxsplit=1)[0]
    return cleaned[-32:] if len(cleaned) > 32 else cleaned


def _font(size: int) -> dict[str, Any]:
    return {"family": "Arial", "weight": "normal", "size": size}
