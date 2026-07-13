#!/usr/bin/env python3
"""Plot async GIPO CPU, GPU, and worker resource utilization."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.figure import Figure
from matplotlib.patches import Patch

CPU_FILE = "cpu_core_1s.csv"
GPU_DEVICE_FILE = "gpu_device_1s.csv"
GPU_WORKER_FILE = "gpu_worker_1s.csv"
PHASE_FILE = "phase_windows.csv"
COVERAGE_FILE = "coverage.json"
METADATA_FILE = "metadata.json"
GPU_COLORS = ("#EA5455", "#2D4059", "#16A3A6", "#FFD460")
COMPONENT_ORDER = {"training": 0, "generation": 1, "env": 2}
COMPONENT_LABELS = {"training": "actor", "generation": "rollout", "env": "env"}

_AVAILABLE_FONTS = {font.name for font in font_manager.fontManager.ttflist}
_FONT_FAMILY = "Arial" if "Arial" in _AVAILABLE_FONTS else "DejaVu Sans"
plt.rcParams.update(
    {
        "font.family": _FONT_FAMILY,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    }
)


@dataclass(frozen=True)
class PhaseWindow:
    """One phase window measured in seconds from the profile origin."""

    phase: str
    step: int | str | None
    component: str
    rank: int | str | None
    start: float
    end: float
    source: str


@dataclass(frozen=True)
class ProfileData:
    """Normalized data needed by the three-panel resource figure."""

    origin: float
    bin_s: float
    cpu_times: np.ndarray
    cpu_matrix: np.ndarray
    gpu_times: np.ndarray
    gpu_labels: list[str]
    gpu_matrix: np.ndarray
    worker_times: np.ndarray
    worker_labels: list[str]
    worker_matrix: np.ndarray
    phases: list[PhaseWindow]
    coverage_warnings: list[str]
    metadata: dict[str, Any]


def _finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {field}: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"Invalid {field}: {value!r}")
    return result


def _optional_number(value: Any) -> int | str | None:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return int(number)
    return text


def _natural_key(value: Any) -> tuple[tuple[int, Any], ...]:
    parts = re.split(r"(\d+)", str(value))
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in parts
        if part
    )


def _infer_bin_s(timestamps: Iterable[float]) -> float:
    unique = sorted(set(timestamps))
    differences = [
        right - left
        for left, right in zip(unique, unique[1:], strict=False)
        if right > left
    ]
    if not differences:
        return 1.0
    bin_s = min(differences)
    if not math.isfinite(bin_s) or bin_s <= 0:
        raise ValueError("Could not infer a positive bin size from timestamps")
    return bin_s


def _resolve_bin_s(rows: Sequence[dict[str, Any]], bin_s: float | None) -> float:
    if bin_s is not None:
        resolved = _finite_float(bin_s, "bin_s")
        if resolved <= 0:
            raise ValueError("bin_s must be positive")
        return resolved
    return _infer_bin_s(
        _finite_float(row.get("timestamp"), "timestamp") for row in rows
    )


def _complete_timeline(
    rows: Sequence[dict[str, Any]], bin_s: float | None
) -> tuple[np.ndarray, float, dict[float, int]]:
    if not rows:
        raise ValueError("At least one data row is required")
    resolved_bin_s = _resolve_bin_s(rows, bin_s)
    timestamps = [_finite_float(row.get("timestamp"), "timestamp") for row in rows]
    start = min(timestamps)
    end = max(timestamps)
    bin_count = int(round((end - start) / resolved_bin_s)) + 1
    times = start + np.arange(bin_count, dtype=float) * resolved_bin_s
    indexes: dict[float, int] = {}
    tolerance = max(1e-7, resolved_bin_s * 1e-6)
    for timestamp in sorted(set(timestamps)):
        index = int(round((timestamp - start) / resolved_bin_s))
        if index < 0 or index >= bin_count or abs(times[index] - timestamp) > tolerance:
            raise ValueError(
                f"Timestamp {timestamp} is not aligned to bin size {resolved_bin_s}"
            )
        indexes[timestamp] = index
    return times, resolved_bin_s, indexes


def build_cpu_matrix(
    rows: Iterable[dict[str, Any]],
    num_cpus: int,
    bin_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a logical-CPU by time utilization matrix.

    A timestamp represented by any row is a covered bin, so unreported logical CPUs
    in that bin are zero. Entirely absent bins remain ``NaN``.
    """

    if isinstance(num_cpus, bool) or not isinstance(num_cpus, int) or num_cpus <= 0:
        raise ValueError("num_cpus must be a positive integer")
    materialized = list(rows)
    times, _, time_indexes = _complete_timeline(materialized, bin_s)
    matrix = np.full((num_cpus, len(times)), np.nan, dtype=float)
    for timestamp in {
        _finite_float(row.get("timestamp"), "timestamp") for row in materialized
    }:
        matrix[:, time_indexes[timestamp]] = 0.0

    samples: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in materialized:
        timestamp = _finite_float(row.get("timestamp"), "timestamp")
        raw_cpu = _finite_float(row.get("cpu"), "logical CPU")
        if not raw_cpu.is_integer():
            raise ValueError(f"Invalid logical CPU index: {row.get('cpu')!r}")
        cpu = int(raw_cpu)
        if cpu < 0 or cpu >= num_cpus:
            raise ValueError(
                f"Invalid logical CPU index {cpu}; expected 0 <= cpu < {num_cpus}"
            )
        utilization = _finite_float(row.get("util_pct"), "CPU utilization")
        samples[(cpu, time_indexes[timestamp])].append(utilization)
    for (cpu, time_index), values in samples.items():
        matrix[cpu, time_index] = float(np.mean(values))
    return times, matrix


def build_gpu_device_series(
    rows: Iterable[dict[str, Any]],
    bin_s: float | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Build sorted physical-GPU busy series on a complete time axis."""

    materialized = list(rows)
    times, _, time_indexes = _complete_timeline(materialized, bin_s)
    device_keys = sorted(
        {
            (str(row.get("gpu_label", "")), str(row.get("gpu_uuid", "")))
            for row in materialized
        },
        key=lambda item: (_natural_key(item[0]), item[1]),
    )
    key_indexes = {key: index for index, key in enumerate(device_keys)}
    labels = [key[0] for key in device_keys]
    matrix = np.full((len(device_keys), len(times)), np.nan, dtype=float)

    samples: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in materialized:
        timestamp = _finite_float(row.get("timestamp"), "timestamp")
        key = (str(row.get("gpu_label", "")), str(row.get("gpu_uuid", "")))
        busy = _finite_float(row.get("kernel_busy_pct"), "GPU busy percentage")
        if busy < 0 or busy > 100:
            raise ValueError(f"GPU busy percentage must be in 0..100, got {busy}")
        samples[(key_indexes[key], time_indexes[timestamp])].append(busy)
    for (device_index, time_index), values in samples.items():
        matrix[device_index, time_index] = float(np.mean(values))
    return times, labels, matrix


def _worker_rank_key(rank: int | str | None) -> tuple[int, Any]:
    if isinstance(rank, int):
        return (0, rank)
    if rank is None:
        return (2, "")
    return (1, str(rank))


def build_worker_matrix(
    rows: Iterable[dict[str, Any]],
    bin_s: float | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Build a component/rank-sorted worker occupancy matrix."""

    materialized = list(rows)
    times, _, time_indexes = _complete_timeline(materialized, bin_s)
    worker_keys = {
        (
            str(row.get("worker", "")),
            str(row.get("component", "")),
            _optional_number(row.get("rank")),
            str(row.get("gpu_uuid", "")),
            str(row.get("gpu_label", "")),
        )
        for row in materialized
    }
    ordered_keys = sorted(
        worker_keys,
        key=lambda item: (
            COMPONENT_ORDER.get(item[1], len(COMPONENT_ORDER)),
            _worker_rank_key(item[2]),
            item[0],
            _natural_key(item[4]),
            item[3],
        ),
    )
    device_counts: dict[tuple[str, str, int | str | None], set[tuple[str, str]]] = (
        defaultdict(set)
    )
    for worker, component, rank, gpu_uuid, gpu_label in worker_keys:
        device_counts[(worker, component, rank)].add((gpu_uuid, gpu_label))

    labels: list[str] = []
    for worker, component, rank, _, gpu_label in ordered_keys:
        role = COMPONENT_LABELS.get(component, component or worker or "worker")
        rank_text = "?" if rank is None else str(rank)
        label = f"{role} r{rank_text}"
        if len(device_counts[(worker, component, rank)]) > 1:
            label += f" (GPU {gpu_label})"
        labels.append(label)

    key_indexes = {key: index for index, key in enumerate(ordered_keys)}
    matrix = np.full((len(ordered_keys), len(times)), np.nan, dtype=float)
    samples: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in materialized:
        timestamp = _finite_float(row.get("timestamp"), "timestamp")
        key = (
            str(row.get("worker", "")),
            str(row.get("component", "")),
            _optional_number(row.get("rank")),
            str(row.get("gpu_uuid", "")),
            str(row.get("gpu_label", "")),
        )
        raw_occupancy = row.get("est_sm_occupancy_pct")
        try:
            occupancy = float(raw_occupancy)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid worker occupancy: {raw_occupancy!r}") from error
        if math.isfinite(occupancy) and not 0 <= occupancy <= 100:
            raise ValueError(f"Worker occupancy must be in 0..100, got {occupancy}")
        samples[(key_indexes[key], time_indexes[timestamp])].append(occupancy)
    for (worker_index, time_index), values in samples.items():
        finite_values = [value for value in values if math.isfinite(value)]
        matrix[worker_index, time_index] = (
            float(np.mean(finite_values)) if finite_values else math.nan
        )
    return times, labels, matrix


def read_phase_windows(path: Path, origin: float) -> list[PhaseWindow]:
    """Read epoch-second phase windows and translate them to relative seconds."""

    windows: list[PhaseWindow] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            start = _finite_float(row.get("start"), "phase start")
            end = _finite_float(row.get("end"), "phase end")
            if end < start:
                raise ValueError(f"Phase end {end} precedes start {start}")
            windows.append(
                PhaseWindow(
                    phase=str(row.get("phase", "")),
                    step=_optional_number(row.get("step")),
                    component=str(row.get("component", "")),
                    rank=_optional_number(row.get("rank")),
                    start=start - origin,
                    end=end - origin,
                    source=str(row.get("source", path)),
                )
            )
    return windows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _metadata_path(derived_dir: Path) -> Path:
    candidates = (derived_dir / METADATA_FILE, derived_dir.parent / METADATA_FILE)
    return next((path for path in candidates if path.is_file()), candidates[-1])


def _coverage_path(derived_dir: Path) -> Path:
    candidates = (derived_dir / COVERAGE_FILE, derived_dir.parent / COVERAGE_FILE)
    return next((path for path in candidates if path.is_file()), candidates[0])


def _required_paths(derived_dir: Path) -> list[Path]:
    return [
        derived_dir / CPU_FILE,
        derived_dir / GPU_DEVICE_FILE,
        derived_dir / GPU_WORKER_FILE,
        derived_dir / PHASE_FILE,
        _coverage_path(derived_dir),
        _metadata_path(derived_dir),
    ]


def _coverage_warnings(value: Any) -> list[str]:
    warnings: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "warning" and isinstance(item, str):
                warnings.append(item)
            elif key == "warnings" and isinstance(item, list):
                warnings.extend(str(entry) for entry in item)
            else:
                warnings.extend(_coverage_warnings(item))
    elif isinstance(value, list):
        for item in value:
            warnings.extend(_coverage_warnings(item))
    return list(dict.fromkeys(warnings))


def _earliest_timestamp(
    data_rows: Sequence[Sequence[dict[str, Any]]],
    phase_rows: Sequence[dict[str, Any]],
) -> float:
    timestamps: list[float] = []
    for rows in data_rows:
        for row in rows:
            try:
                timestamp = float(row.get("timestamp"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(timestamp):
                timestamps.append(timestamp)
    for row in phase_rows:
        for field in ("start", "end"):
            try:
                timestamp = float(row.get(field))
            except (TypeError, ValueError):
                continue
            if math.isfinite(timestamp):
                timestamps.append(timestamp)
    if not timestamps:
        raise ValueError("No valid timestamp found in derived inputs")
    return min(timestamps)


def load_profile(derived_dir: str | Path, num_cpus: int) -> ProfileData:
    """Load and normalize one generated async GIPO derived directory."""

    derived_dir = Path(derived_dir).resolve()
    required_paths = _required_paths(derived_dir)
    missing = [path for path in required_paths if not path.is_file()]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required input files: {names}")

    cpu_rows = _read_csv(derived_dir / CPU_FILE)
    gpu_rows = _read_csv(derived_dir / GPU_DEVICE_FILE)
    worker_rows = _read_csv(derived_dir / GPU_WORKER_FILE)
    phase_rows = _read_csv(derived_dir / PHASE_FILE)
    metadata = json.loads(_metadata_path(derived_dir).read_text())
    coverage = json.loads(_coverage_path(derived_dir).read_text())
    metadata_bin_s = metadata.get("bin_s")
    bin_s = (
        _finite_float(metadata_bin_s, "metadata bin_s")
        if metadata_bin_s is not None
        else _infer_bin_s(
            _finite_float(row.get("timestamp"), "timestamp") for row in cpu_rows
        )
    )
    if bin_s <= 0:
        raise ValueError("metadata bin_s must be positive")
    origin = _earliest_timestamp((cpu_rows, gpu_rows, worker_rows), phase_rows)
    cpu_times, cpu_matrix = build_cpu_matrix(cpu_rows, num_cpus, bin_s)
    gpu_times, gpu_labels, gpu_matrix = build_gpu_device_series(gpu_rows, bin_s)
    worker_times, worker_labels, worker_matrix = build_worker_matrix(worker_rows, bin_s)
    phases = read_phase_windows(derived_dir / PHASE_FILE, origin)
    return ProfileData(
        origin=origin,
        bin_s=bin_s,
        cpu_times=cpu_times,
        cpu_matrix=cpu_matrix,
        gpu_times=gpu_times,
        gpu_labels=gpu_labels,
        gpu_matrix=gpu_matrix,
        worker_times=worker_times,
        worker_labels=worker_labels,
        worker_matrix=worker_matrix,
        phases=phases,
        coverage_warnings=_coverage_warnings(coverage),
        metadata=metadata,
    )


def _heatmap_extent(
    times: np.ndarray, origin: float, bin_s: float, rows: int
) -> list[float]:
    return [
        float(times[0] - origin),
        float(times[-1] - origin + bin_s),
        -0.5,
        rows - 0.5,
    ]


def _add_phase_annotations(axes: Sequence[Any], phases: Sequence[PhaseWindow]) -> None:
    actor_windows = [phase for phase in phases if phase.phase == "actor_training"]
    for phase in actor_windows:
        for axis in axes:
            axis.axvspan(
                phase.start,
                phase.end,
                color="#E15759",
                alpha=0.08,
                linewidth=0,
                zorder=1,
            )

    seen_steps: set[tuple[float, int | str | None]] = set()
    for phase in sorted(phases, key=lambda item: (item.start, str(item.step))):
        if phase.phase != "step":
            continue
        key = (phase.start, phase.step)
        if key in seen_steps:
            continue
        seen_steps.add(key)
        for axis in axes:
            axis.axvline(
                phase.start,
                color="black",
                linewidth=0.7,
                alpha=0.45,
                zorder=20,
            )
        step_text = "?" if phase.step is None else phase.step
        axes[0].text(
            phase.start,
            1.01,
            f"Step {step_text}",
            transform=axes[0].get_xaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=9,
            color="black",
            clip_on=False,
        )


def draw_profile(profile: ProfileData) -> Figure:
    """Draw a three-panel paper-style resource utilization figure."""

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(12, 8),
        sharex=True,
        height_ratios=(3.2, 1.4, 2.0),
        layout="constrained",
    )
    cpu_axis, gpu_axis, worker_axis = axes
    color_map = plt.get_cmap("viridis").with_extremes(bad="white")

    cpu_image = cpu_axis.imshow(
        profile.cpu_matrix,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=_heatmap_extent(
            profile.cpu_times,
            profile.origin,
            profile.bin_s,
            profile.cpu_matrix.shape[0],
        ),
        cmap=color_map,
        vmin=0,
        vmax=100,
        zorder=0,
    )
    cpu_axis.set_ylabel("RLinf CPU utilization per logical core (%)", fontsize=8)
    cpu_ticks = (
        np.arange(profile.cpu_matrix.shape[0])
        if profile.cpu_matrix.shape[0] <= 16
        else np.arange(0, profile.cpu_matrix.shape[0], 16)
    )
    cpu_axis.set_yticks(cpu_ticks)
    for boundary in range(16, profile.cpu_matrix.shape[0], 16):
        cpu_axis.axhline(
            boundary - 0.5, color="black", linewidth=0.45, alpha=0.45, zorder=3
        )
    figure.colorbar(cpu_image, ax=cpu_axis, pad=0.01, shrink=0.92)

    gpu_x = profile.gpu_times - profile.origin + profile.bin_s / 2
    gpu_handles = []
    for index, (label, values) in enumerate(
        zip(profile.gpu_labels, profile.gpu_matrix, strict=True)
    ):
        gpu_handles.append(
            gpu_axis.plot(
                gpu_x,
                values,
                color=GPU_COLORS[index % len(GPU_COLORS)],
                linewidth=1.5,
                label=f"GPU {label}",
                zorder=10,
            )[0]
        )
    gpu_axis.set_ylim(0, 100)
    gpu_axis.set_ylabel("CUDA kernel busy by physical GPU (%)", fontsize=8)
    gpu_axis.grid(
        axis="y", linestyle="--", linewidth=0.5, color="gray", alpha=0.55, zorder=-1
    )
    legend_handles: list[Any] = list(gpu_handles)
    if any(phase.phase == "actor_training" for phase in profile.phases):
        legend_handles.append(
            Patch(
                facecolor="#E15759",
                alpha=0.15,
                edgecolor="none",
                label="Actor training",
            )
        )
    if legend_handles:
        gpu_axis.legend(
            handles=legend_handles,
            ncol=min(5, len(legend_handles)),
            fontsize=8,
            handlelength=1.4,
            handletextpad=0.35,
            columnspacing=0.8,
            loc="upper right",
            frameon=True,
            edgecolor="black",
        )

    worker_image = worker_axis.imshow(
        profile.worker_matrix,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=_heatmap_extent(
            profile.worker_times,
            profile.origin,
            profile.bin_s,
            profile.worker_matrix.shape[0],
        ),
        cmap=color_map,
        vmin=0,
        vmax=100,
        zorder=0,
    )
    worker_axis.set_yticks(np.arange(len(profile.worker_labels)))
    worker_axis.set_yticklabels(profile.worker_labels, fontsize=8)
    worker_axis.set_ylabel("Estimated SM occupancy by worker (%)", fontsize=8)
    worker_axis.set_xlabel("Time from first derived timestamp (s)", fontsize=10)
    figure.colorbar(worker_image, ax=worker_axis, pad=0.01, shrink=0.92)

    _add_phase_annotations(axes, profile.phases)
    data_end = max(
        float(profile.cpu_times[-1] - profile.origin + profile.bin_s),
        float(profile.gpu_times[-1] - profile.origin + profile.bin_s),
        float(profile.worker_times[-1] - profile.origin + profile.bin_s),
        max((phase.end for phase in profile.phases), default=0.0),
    )
    axes[-1].set_xlim(0, data_end)
    for axis in axes:
        axis.tick_params(axis="both", labelsize=8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    return figure


def create_figure(derived_dir: str | Path, num_cpus: int = 112) -> Figure:
    """Load a derived directory and create its resource figure."""

    return draw_profile(load_profile(derived_dir, num_cpus))


def save_figure(figure: Figure, output_prefix: str | Path) -> tuple[Path, Path]:
    """Save PDF and 200-DPI PNG outputs for a resource figure."""

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_prefix.with_suffix(".pdf")
    png_path = output_prefix.with_suffix(".png")
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, bbox_inches="tight", dpi=200)
    return pdf_path, png_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("derived_dir", type=Path, help="Generated derived directory")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="Output path prefix for the PDF and PNG",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=112,
        help="Number of logical CPUs shown in the CPU heatmap",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        profile = load_profile(args.derived_dir, args.num_cpus)
        for warning in profile.coverage_warnings:
            print(f"WARNING: {warning}", file=sys.stderr)
        figure = draw_profile(profile)
        pdf_path, png_path = save_figure(figure, args.output_prefix)
        plt.close(figure)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"WROTE={pdf_path}")
    print(f"WROTE={png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
