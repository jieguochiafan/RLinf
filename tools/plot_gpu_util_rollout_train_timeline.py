"""Plot continuous GPU utilization with rollout/train phase markers.

The input GPU utilization comes from ``nvidia-smi --query-gpu`` sampling,
so it covers the full wall-clock timeline. Rollout boundaries are taken
from worker timestamp JSONL files. Training boundaries are inferred from
the post-rollout GPU-active region and ``actor/run_training`` in train.log
unless explicit times are provided.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib import font_manager

GPU_TIME_FORMAT = "%Y/%m/%d %H:%M:%S.%f"
ISO_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
AVAILABLE_FONTS = {font.name for font in font_manager.fontManager.ttflist}
FONT_FAMILY = "Arial" if "Arial" in AVAILABLE_FONTS else "DejaVu Sans"


@dataclass(frozen=True)
class PhaseWindow:
    name: str
    start: datetime
    end: datetime
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot full-timeline GPU utilization for rollout and train."
    )
    parser.add_argument("run_dir", type=Path, help="Profile run directory.")
    parser.add_argument(
        "--gpu-csv",
        type=Path,
        default=None,
        help=(
            "GPU utilization CSV. Defaults to <run_dir>/gpu/gpu_util_10hz.csv, "
            "or <run_dir>/resource_profile/nvidia_smi/gpu_util_500ms.csv when present."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Output prefix. Defaults to <run_dir>/gpu/gpu_util_rollout_train_timeline.",
    )
    parser.add_argument(
        "--active-threshold",
        type=float,
        default=20.0,
        help="Mean GPU utilization threshold used to infer train start.",
    )
    parser.add_argument(
        "--min-active-duration",
        type=float,
        default=5.0,
        help="Minimum seconds for a post-rollout active region to count as train.",
    )
    parser.add_argument(
        "--train-start",
        type=str,
        default=None,
        help=f"Override train start time, format '{ISO_FORMAT}'.",
    )
    parser.add_argument(
        "--train-end",
        type=str,
        default=None,
        help=f"Override train end time, format '{ISO_FORMAT}'.",
    )
    parser.add_argument(
        "--x-relative",
        action="store_true",
        help="Use seconds from first GPU sample on the x-axis.",
    )
    parser.add_argument(
        "--relative-origin",
        choices=("sample", "rollout"),
        default="sample",
        help="Origin for --x-relative. Use rollout to set rollout start to 0s.",
    )
    parser.add_argument(
        "--align-phase-ticks",
        action="store_true",
        help="Use rollout/train boundaries as major x-axis ticks in relative-time plots.",
    )
    parser.add_argument(
        "--smooth-window-s",
        type=float,
        default=0.0,
        help="Centered moving-average window in seconds. 0 disables smoothing.",
    )
    parser.add_argument(
        "--smooth-mode",
        choices=("mean", "quantile", "max"),
        default="mean",
        help="Smoothing mode. Use quantile/max to preserve high utilization values.",
    )
    parser.add_argument(
        "--smooth-quantile",
        type=float,
        default=0.9,
        help="Quantile used when --smooth-mode=quantile.",
    )
    parser.add_argument(
        "--x-pad-s",
        type=float,
        default=60.0,
        help="Extra x-axis padding in seconds on both sides.",
    )
    return parser.parse_args()


def parse_datetime(value: str) -> datetime:
    return datetime.strptime(value, ISO_FORMAT)


def parse_gpu_datetime(value: str) -> datetime:
    value = value.strip()
    for fmt in (GPU_TIME_FORMAT, ISO_FORMAT, "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported GPU timestamp: {value}")


def parse_gpu_util(value: str) -> float:
    return float(value.strip().rstrip("%"))


def _is_gpu_header(row: list[str]) -> bool:
    if len(row) < 3:
        return False
    try:
        parse_gpu_datetime(row[0])
        int(row[1].strip())
        parse_gpu_util(row[2])
    except ValueError:
        return True
    return False


def read_gpu_samples(path: Path) -> tuple[list[datetime], list[int], np.ndarray]:
    by_time: dict[datetime, dict[int, float]] = defaultdict(dict)
    gpus: set[int] = set()
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        first_row = next(reader, None)
        if first_row is None:
            return [], [], np.empty((0, 0), dtype=float)

        if _is_gpu_header(first_row):
            header = [cell.strip() for cell in first_row]
            header_index = {name: idx for idx, name in enumerate(header)}
            timestamp_idx = header_index.get("timestamp", 0)
            gpu_idx = header_index.get("index", header_index.get("gpu", 1))
            util_idx = header_index.get(
                "utilization.gpu [%]",
                header_index.get("utilization.gpu", header_index.get("util", 2)),
            )
            rows = reader
        else:
            timestamp_idx, gpu_idx, util_idx = 0, 1, 2
            rows = [first_row, *reader]

        for row in rows:
            if len(row) <= max(timestamp_idx, gpu_idx, util_idx):
                continue
            timestamp = parse_gpu_datetime(row[timestamp_idx])
            gpu = int(row[gpu_idx].strip())
            util = parse_gpu_util(row[util_idx])
            by_time[timestamp][gpu] = util
            gpus.add(gpu)

    ordered_times = sorted(by_time)
    ordered_gpus = sorted(gpus)
    gpu_to_col = {gpu: idx for idx, gpu in enumerate(ordered_gpus)}
    util = np.full((len(ordered_times), len(ordered_gpus)), np.nan)
    for row_idx, timestamp in enumerate(ordered_times):
        for gpu, value in by_time[timestamp].items():
            util[row_idx, gpu_to_col[gpu]] = value
    return ordered_times, ordered_gpus, util


def resolve_gpu_csv(run_dir: Path, gpu_csv: Path | None) -> Path:
    if gpu_csv is not None:
        return gpu_csv
    candidates = [
        run_dir / "gpu" / "gpu_util_10hz.csv",
        run_dir / "resource_profile" / "nvidia_smi" / "gpu_util_500ms.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def smooth_utilization(
    times: list[datetime],
    util: np.ndarray,
    window_s: float,
    mode: str,
    quantile: float,
) -> np.ndarray:
    if window_s <= 0:
        return util
    if len(times) < 2:
        return util

    intervals = np.diff([time.timestamp() for time in times])
    sample_interval_s = float(np.nanmedian(intervals))
    if sample_interval_s <= 0:
        return util

    window = max(1, int(round(window_s / sample_interval_s)))
    if window <= 1:
        return util
    if window % 2 == 0:
        window += 1

    smoothed = np.empty_like(util, dtype=float)
    if mode == "mean":
        kernel = np.ones(window, dtype=float)
        for idx in range(util.shape[1]):
            values = util[:, idx]
            valid = np.isfinite(values).astype(float)
            filled = np.nan_to_num(values, nan=0.0)
            numerator = np.convolve(filled, kernel, mode="same")
            denominator = np.convolve(valid, kernel, mode="same")
            smoothed[:, idx] = np.divide(
                numerator,
                denominator,
                out=np.full_like(numerator, np.nan, dtype=float),
                where=denominator > 0,
            )
        return smoothed

    half_window = window // 2
    quantile = min(max(quantile, 0.0), 1.0)
    for idx in range(util.shape[1]):
        values = util[:, idx]
        for row_idx in range(len(values)):
            start = max(0, row_idx - half_window)
            end = min(len(values), row_idx + half_window + 1)
            window_values = values[start:end]
            if np.all(np.isnan(window_values)):
                smoothed[row_idx, idx] = np.nan
            elif mode == "max":
                smoothed[row_idx, idx] = np.nanmax(window_values)
            else:
                smoothed[row_idx, idx] = np.nanquantile(window_values, quantile)
    return smoothed


def iter_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def ns_to_datetime(wall_ns: int) -> datetime:
    return datetime.fromtimestamp(wall_ns / 1_000_000_000)


def read_rollout_window(run_dir: Path) -> PhaseWindow:
    timestamp_dirs = [
        run_dir / "logs" / "rollout_generation_timestamps",
        run_dir / "logs" / "env_sim_timestamps",
    ]
    starts: list[datetime] = []
    ends: list[datetime] = []

    for timestamp_dir in timestamp_dirs:
        for path in sorted(timestamp_dir.glob("*.jsonl")):
            for record in iter_jsonl(path):
                if "wall_ns" not in record:
                    continue
                dt = ns_to_datetime(int(record["wall_ns"]))
                event = record.get("event")
                if event == "start":
                    starts.append(dt)
                elif event == "end":
                    ends.append(dt)

    if not starts or not ends:
        raise ValueError(f"Could not infer rollout window under {run_dir}/logs")

    return PhaseWindow(
        name="Rollout",
        start=min(starts),
        end=max(ends),
        source="logs/rollout_generation_timestamps + logs/env_sim_timestamps",
    )


def read_actor_training_seconds(train_log: Path) -> float | None:
    if not train_log.exists():
        return None
    text = train_log.read_text(errors="ignore")
    match = re.search(r"actor/run_training=([0-9.]+)", text)
    if not match:
        return None
    return float(match.group(1))


def infer_train_window(
    times: list[datetime],
    util: np.ndarray,
    rollout_end: datetime,
    training_seconds: float | None,
    active_threshold: float,
    min_active_duration: float,
) -> PhaseWindow | None:
    mean_util = np.nanmean(util, axis=1)
    active = mean_util >= active_threshold
    segments: list[tuple[datetime, datetime]] = []
    current_start: datetime | None = None
    current_end: datetime | None = None

    for timestamp, is_active in zip(times, active, strict=True):
        if timestamp <= rollout_end:
            continue
        if is_active:
            if current_start is None:
                current_start = timestamp
            current_end = timestamp
        elif current_start is not None and current_end is not None:
            if (current_end - current_start).total_seconds() >= min_active_duration:
                segments.append((current_start, current_end))
            current_start = None
            current_end = None

    if current_start is not None and current_end is not None:
        if (current_end - current_start).total_seconds() >= min_active_duration:
            segments.append((current_start, current_end))

    if not segments:
        return None

    train_start = segments[0][0]
    if training_seconds is None:
        train_end = segments[-1][1]
        source = (
            f"post-rollout mean GPU util >= {active_threshold:g}% "
            f"for >= {min_active_duration:g}s"
        )
    else:
        train_end = train_start + timedelta(seconds=training_seconds)
        source = (
            f"train start from post-rollout mean GPU util >= {active_threshold:g}%; "
            f"duration from train.log actor/run_training={training_seconds:.3f}s"
        )

    return PhaseWindow(name="Train", start=train_start, end=train_end, source=source)


def write_phase_csv(path: Path, phases: list[PhaseWindow]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["phase", "start", "end", "duration_s", "source"]
        )
        writer.writeheader()
        for phase in phases:
            writer.writerow(
                {
                    "phase": phase.name,
                    "start": phase.start.strftime(ISO_FORMAT)[:-3],
                    "end": phase.end.strftime(ISO_FORMAT)[:-3],
                    "duration_s": f"{(phase.end - phase.start).total_seconds():.3f}",
                    "source": phase.source,
                }
            )


def draw_phase_markers(
    ax: plt.Axes,
    phases: list[PhaseWindow],
    x_relative: bool,
    origin: datetime,
) -> None:
    line_colors = {"Rollout": "#2D4059", "Train": "#EA5455"}
    for phase in phases:
        if x_relative:
            start = (phase.start - origin).total_seconds()
            end = (phase.end - origin).total_seconds()
        else:
            start = phase.start
            end = phase.end
        ax.axvline(
            start,
            color=line_colors.get(phase.name, "black"),
            linestyle="--",
            linewidth=1.0,
            alpha=0.9,
        )
        ax.axvline(
            end,
            color=line_colors.get(phase.name, "black"),
            linestyle="--",
            linewidth=1.0,
            alpha=0.9,
        )
        center = start + (end - start) / 2
        ax.text(
            center,
            104,
            phase.name,
            ha="center",
            va="bottom",
            fontsize=13,
            color=line_colors.get(phase.name, "black"),
        )


def draw_rollout_note(
    ax: plt.Axes,
    phases: list[PhaseWindow],
    x_relative: bool,
    origin: datetime,
) -> None:
    rollout = next((phase for phase in phases if phase.name == "Rollout"), None)
    if rollout is None:
        return

    if x_relative:
        start = (rollout.start - origin).total_seconds()
        end = (rollout.end - origin).total_seconds()
    else:
        start = rollout.start
        end = rollout.end
    center = start + (end - start) / 2

    ax.text(
        center,
        70,
        "Rollout is more than 90%.\nGPU Utl. is below 50%, \nmainly generation.",
        ha="center",
        va="center",
        fontsize=8,
        color="#EA5455",
        zorder=20,
    )


def plot_gpu_util(
    times: list[datetime],
    gpus: list[int],
    util: np.ndarray,
    phases: list[PhaseWindow],
    output_prefix: Path,
    x_relative: bool,
    relative_origin: str,
    align_phase_ticks: bool,
    smooth_window_s: float,
    smooth_mode: str,
    smooth_quantile: float,
    x_pad_s: float,
) -> None:
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": 12,
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(4, 1.5))
    origin = phases[0].start if x_relative and relative_origin == "rollout" else times[0]
    if x_relative:
        x_values = np.array([(time - origin).total_seconds() for time in times])
    else:
        x_values = np.array(times)

    colors = [
        "#2D4059",
        "#EA5455",
        "#76BA99",
        "#FFD460",
        "#16A3A6",
        "#9DBDFF",
        "#8E7DBE",
        "#F08A5D",
    ]
    plot_util = smooth_utilization(
        times,
        util,
        smooth_window_s,
        mode=smooth_mode,
        quantile=smooth_quantile,
    )
    mean_util = np.nanmean(plot_util, axis=1)
    ax.fill_between(x_values, mean_util, color="#EA5455", alpha=0.16, zorder=1)
    for idx, gpu in enumerate(gpus):
        ax.plot(
            x_values,
            plot_util[:, idx],
            color=colors[idx % len(colors)],
            linewidth=0.9 if smooth_window_s > 0 else 0.65,
            alpha=0.72 if smooth_window_s > 0 else 0.55,
            zorder=3,
        )

    draw_phase_markers(ax, phases, x_relative=x_relative, origin=origin)
    draw_rollout_note(ax, phases, x_relative=x_relative, origin=origin)
    ax.set_ylim(0, 110)
    ax.set_ylabel("GPU Util. (%)")
    ax.set_xlabel("Time (s)" if x_relative else "Wall-clock Time")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, color="gray", alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if x_relative:
        phase_tick_values = [
            (phase.start - origin).total_seconds() for phase in phases
        ] + [(phase.end - origin).total_seconds() for phase in phases]
        max_x = max(x_values[-1], max(phase_tick_values))
        left_x = min(0.0, x_values[0])
        ax.set_xlim(left_x - x_pad_s, max_x + x_pad_s)
        if align_phase_ticks:
            major_ticks = sorted(
                {
                    round(value, 1)
                    for value in phase_tick_values
                    if left_x <= value <= max_x
                }
            )
            tick_labels = [f"{tick:.1f}" if abs(tick) < 1 else f"{tick:.0f}" for tick in major_ticks]
            ax.xaxis.set_major_locator(mticker.FixedLocator(major_ticks))
            ax.xaxis.set_major_formatter(mticker.FixedFormatter(tick_labels))
            ax.tick_params(axis="x", which="major", length=5, width=1.1, labelsize=10)
            for label in ax.get_xticklabels():
                label.set_rotation(35)
                label.set_ha("right")
        else:
            major_ticks = np.arange(0, np.ceil(max_x / 200) * 200 + 1, 200)
            ax.xaxis.set_major_locator(mticker.FixedLocator(major_ticks))
            ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
    else:
        x_pad = timedelta(seconds=x_pad_s)
        ax.set_xlim(times[0] - x_pad, times[-1] + x_pad)
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        fig.autofmt_xdate(rotation=0)
    ax.tick_params(axis="both", which="both", length=0)

    fig.tight_layout()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    output_prefix = (
        args.output_prefix
        if args.output_prefix is not None
        else run_dir / "gpu" / "gpu_util_rollout_train_timeline"
    )

    times, gpus, util = read_gpu_samples(resolve_gpu_csv(run_dir, args.gpu_csv))
    rollout = read_rollout_window(run_dir)

    if args.train_start or args.train_end:
        if not args.train_start or not args.train_end:
            raise ValueError("--train-start and --train-end must be provided together")
        train = PhaseWindow(
            name="Train",
            start=parse_datetime(args.train_start),
            end=parse_datetime(args.train_end),
            source="manual override",
        )
    else:
        train = infer_train_window(
            times=times,
            util=util,
            rollout_end=rollout.end,
            training_seconds=read_actor_training_seconds(run_dir / "train.log"),
            active_threshold=args.active_threshold,
            min_active_duration=args.min_active_duration,
        )

    phases = [rollout]
    if train is not None:
        phases.append(train)

    plot_gpu_util(
        times=times,
        gpus=gpus,
        util=util,
        phases=phases,
        output_prefix=output_prefix,
        x_relative=args.x_relative,
        relative_origin=args.relative_origin,
        align_phase_ticks=args.align_phase_ticks,
        smooth_window_s=args.smooth_window_s,
        smooth_mode=args.smooth_mode,
        smooth_quantile=args.smooth_quantile,
        x_pad_s=args.x_pad_s,
    )
    write_phase_csv(output_prefix.with_name(output_prefix.name + "_phases.csv"), phases)
    print(f"Wrote {output_prefix.with_suffix('.pdf')}")
    print(f"Wrote {output_prefix.with_suffix('.png')}")
    print(f"Wrote {output_prefix.with_name(output_prefix.name + '_phases.csv')}")


if __name__ == "__main__":
    main()
