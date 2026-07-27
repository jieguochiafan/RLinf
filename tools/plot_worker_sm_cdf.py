#!/usr/bin/env python3
"""Plot per-worker torch-profiler estimated SM occupancy CDFs."""

from __future__ import annotations

import argparse
import csv
import json
import pprint
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

SM_OCCUPANCY_KEY = "est. achieved occupancy %"
OUTLIER_MAX_OCCUPANCY_PCT = 95.0


@dataclass(frozen=True)
class TraceEvent:
    name: str
    category: str
    ts_us: float
    dur_us: float
    args: dict[str, Any]

    @property
    def end_us(self) -> float:
        return self.ts_us + self.dur_us


def load_trace_events(path: Path) -> list[TraceEvent]:
    with path.open(encoding="utf-8") as f:
        trace = json.load(f)

    events: list[TraceEvent] = []
    for raw in trace.get("traceEvents", []):
        if raw.get("ph") != "X" or "ts" not in raw or "dur" not in raw:
            continue
        events.append(
            TraceEvent(
                name=str(raw.get("name", "")),
                category=str(raw.get("cat", "")),
                ts_us=float(raw["ts"]),
                dur_us=float(raw["dur"]),
                args=dict(raw.get("args", {})),
            )
        )
    return events


def phase_names_for_worker(worker: str) -> list[str]:
    if worker.startswith("rollout_rank"):
        return ["generation"]
    if worker.startswith("rank"):
        return ["fwd", "loss", "backward", "optimizer.step"]
    return []


def overlap_us(left: TraceEvent, right: TraceEvent) -> float:
    return max(0.0, min(left.end_us, right.end_us) - max(left.ts_us, right.ts_us))


def collect_worker_samples(trace_dir: Path) -> list[dict[str, Any]]:
    trace_path = next(trace_dir.glob("*.pt.trace.json"))
    trace_events = load_trace_events(trace_path)
    target_phases = set(phase_names_for_worker(trace_dir.name))
    phases = [
        event
        for event in trace_events
        if event.name in target_phases
    ]
    if not phases:
        return []

    kernels = [
        event
        for event in trace_events
        if event.category == "kernel" and SM_OCCUPANCY_KEY in event.args
    ]

    rows: list[dict[str, Any]] = []
    for phase in phases:
        for kernel in kernels:
            overlap = overlap_us(phase, kernel)
            if overlap <= 0:
                continue
            rows.append(
                {
                    "worker_type": "generation"
                    if trace_dir.name.startswith("rollout_rank")
                    else "training",
                    "worker": trace_dir.name,
                    "phase": phase.name,
                    "occupancy_pct": float(kernel.args[SM_OCCUPANCY_KEY]),
                    "duration_us": overlap,
                }
            )
    return rows


def collect_samples(torch_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace_dir in sorted(torch_dir.iterdir()):
        if not trace_dir.is_dir() or not phase_names_for_worker(trace_dir.name):
            continue
        if not list(trace_dir.glob("*.pt.trace.json")):
            continue
        rows.extend(collect_worker_samples(trace_dir))
    return rows


def cdf_points(samples: list[tuple[float, float]]) -> tuple[list[float], list[float]]:
    by_value: dict[float, float] = {}
    for value, weight in samples:
        if weight <= 0:
            continue
        by_value[value] = by_value.get(value, 0.0) + weight

    total = sum(by_value.values())
    if total <= 0:
        return [], []

    xs: list[float] = []
    ys: list[float] = []
    cumulative = 0.0
    for value, weight in sorted(by_value.items()):
        cumulative += weight
        xs.append(value)
        ys.append(cumulative / total)
    return xs, ys


def write_samples(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["worker_type", "worker", "phase", "occupancy_pct", "duration_us"]
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fieldnames} for row in rows)


def write_summary(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "worker_type",
        "worker",
        "sample_count",
        "weighted_mean_occupancy_pct",
        "p50_occupancy_pct",
        "p90_occupancy_pct",
        "total_kernel_overlap_us",
    ]
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for worker_type, worker, samples in iter_worker_samples(rows):
            samples = filter_occupancy_outliers(samples)
            xs, ys = cdf_points(samples)
            total_weight = sum(weight for _, weight in samples)
            weighted_mean = (
                sum(value * weight for value, weight in samples) / total_weight
                if total_weight > 0
                else 0.0
            )
            writer.writerow(
                {
                    "worker_type": worker_type,
                    "worker": worker,
                    "sample_count": len(samples),
                    "weighted_mean_occupancy_pct": weighted_mean,
                    "p50_occupancy_pct": percentile_from_cdf(xs, ys, 0.50),
                    "p90_occupancy_pct": percentile_from_cdf(xs, ys, 0.90),
                    "total_kernel_overlap_us": total_weight,
                }
            )


def build_cdf_data(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    cdf_data: dict[str, list[dict[str, Any]]] = {"generation": [], "training": []}
    for worker_type, worker, samples in iter_worker_samples(rows):
        samples = filter_occupancy_outliers(samples)
        xs, ys = cdf_points(samples)
        if not xs:
            continue
        cdf_data.setdefault(worker_type, []).append(
            {
                "worker": worker,
                "x": xs,
                "y_pct": cdf_y_percent(ys),
            }
        )
    return cdf_data


def build_combined_cdf_data(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[tuple[float, float]]] = {"generation": [], "training": []}
    for row in rows:
        worker_type = str(row["worker_type"])
        if worker_type not in grouped:
            continue
        grouped[worker_type].append(
            (float(row["occupancy_pct"]), float(row["duration_us"]))
        )

    combined: dict[str, dict[str, Any]] = {}
    for worker_type, samples in grouped.items():
        samples = filter_occupancy_outliers(samples)
        xs, ys = cdf_points(samples)
        combined[worker_type] = {
            "x": xs,
            "y_pct": cdf_y_percent(ys),
        }
    return combined


def write_cdf_points(cdf_data: dict[str, list[dict[str, Any]]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["worker_type", "worker", "point_index", "occupancy_pct", "cdf_pct"],
        )
        writer.writeheader()
        for worker_type, curves in cdf_data.items():
            for curve in curves:
                for point_index, (x_value, y_value) in enumerate(
                    zip(curve["x"], curve["y_pct"], strict=True)
                ):
                    writer.writerow(
                        {
                            "worker_type": worker_type,
                            "worker": curve["worker"],
                            "point_index": point_index,
                            "occupancy_pct": x_value,
                            "cdf_pct": y_value,
                        }
                    )


def write_combined_cdf_points(
    cdf_data: dict[str, dict[str, Any]],
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["worker_type", "point_index", "occupancy_pct", "cdf_pct"],
        )
        writer.writeheader()
        for worker_type, curve in cdf_data.items():
            for point_index, (x_value, y_value) in enumerate(
                zip(curve["x"], curve["y_pct"], strict=True)
            ):
                writer.writerow(
                    {
                        "worker_type": worker_type,
                        "point_index": point_index,
                        "occupancy_pct": x_value,
                        "cdf_pct": y_value,
                    }
                )


def write_static_plot_script(
    cdf_data: dict[str, list[dict[str, Any]]],
    combined_cdf_data: dict[str, dict[str, Any]],
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    script = f'''#!/usr/bin/env python3
"""Plot SM occupancy CDFs from embedded data."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


CDF_DATA = {pprint.pformat(cdf_data, width=100)}
COMBINED_CDF_DATA = {pprint.pformat(combined_cdf_data, width=100)}

COLORS = [
    "#EA5455",
    "#2D4059",
    "#16A3A6",
    "#FFD460",
    "#76BA99",
    "#9DBDFF",
    "#7D5A50",
    "#6C5CE7",
]


def x_axis_limit(curves: list[dict]) -> float:
    values = [value for curve in curves for value in curve["x"]]
    return max(values) if values else 1.0


def interpolate_cdf_y(xs: list[float], ys: list[float], x_value: float) -> float:
    if x_value <= xs[0]:
        return ys[0]
    for idx in range(1, len(xs)):
        if x_value <= xs[idx]:
            left_x = xs[idx - 1]
            right_x = xs[idx]
            left_y = ys[idx - 1]
            right_y = ys[idx]
            if right_x == left_x:
                return right_y
            ratio = (x_value - left_x) / (right_x - left_x)
            return left_y + ratio * (right_y - left_y)
    return ys[-1]


def smooth_cdf_curve(
    xs: list[float],
    ys: list[float],
    point_count: int = 180,
    window_size: int = 11,
) -> tuple[list[float], list[float]]:
    if len(xs) < 3 or point_count <= len(xs):
        return xs, ys

    min_x = xs[0]
    max_x = xs[-1]
    if max_x <= min_x:
        return xs, ys

    dense_x = [
        min_x + (max_x - min_x) * idx / (point_count - 1)
        for idx in range(point_count)
    ]
    dense_y = [interpolate_cdf_y(xs, ys, x_value) for x_value in dense_x]

    window_size = max(3, min(window_size, point_count))
    if window_size % 2 == 0:
        window_size += 1
    half_window = window_size // 2
    smoothed_y: list[float] = []
    for idx in range(point_count):
        left = max(0, idx - half_window)
        right = min(point_count, idx + half_window + 1)
        smoothed_y.append(sum(dense_y[left:right]) / (right - left))

    smoothed_y[0] = ys[0]
    smoothed_y[-1] = ys[-1]
    cumulative_max = smoothed_y[0]
    for idx, y_value in enumerate(smoothed_y):
        cumulative_max = max(cumulative_max, min(100.0, y_value))
        smoothed_y[idx] = cumulative_max
    smoothed_y[-1] = ys[-1]
    return dense_x, smoothed_y


def y_axis_limit_pct(curves: list[dict]) -> tuple[float, float]:
    values = [value for curve in curves for value in curve["y_pct"]]
    if not values:
        return (0.0, 100.0)
    lower = max(0.0, (min(values) // 10.0) * 10.0)
    return (round(lower, 1), 100.0)


def plot_cdf(worker_type: str, output_prefix: Path) -> None:
    plt.rcParams.update(
        {{
            "font.family": "Arial",
            "font.size": 11,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }}
    )
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    curves = CDF_DATA[worker_type]
    for idx, curve in enumerate(curves):
        ax.step(
            curve["x"],
            curve["y_pct"],
            where="post",
            color=COLORS[idx % len(COLORS)],
            linewidth=1.4,
            label=curve["worker"].replace("_rank", " "),
        )

    ax.set_xlabel("SM occupancy (%)")
    ax.set_ylabel("CDF (%)")
    ax.set_xlim(0, x_axis_limit(curves))
    ax.set_ylim(*y_axis_limit_pct(curves))
    ax.grid(axis="both", linestyle="--", linewidth=0.5, color="gray", alpha=0.45)
    ax.legend(
        ncol=2,
        fontsize=8,
        handletextpad=0.3,
        handlelength=1.4,
        columnspacing=0.6,
        loc="lower right",
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_combined_cdf(output_prefix: Path) -> None:
    plt.rcParams.update(
        {{
            "font.family": "Arial",
            "font.size": 10,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }}
    )
    fig, ax = plt.subplots(figsize=(3.0, 2.5))
    styles = {{
        "generation": {{"label": "Generation", "color": "#EA5455", "marker": "o"}},
        "training": {{"label": "Training", "color": "#2D4059", "marker": "s"}},
    }}

    x_values = []
    for worker_type in ("generation", "training"):
        curve = COMBINED_CDF_DATA[worker_type]
        if not curve["x"]:
            continue
        x_values.extend(curve["x"])
        style = styles[worker_type]
        plot_x, plot_y = smooth_cdf_curve(curve["x"], curve["y_pct"])
        ax.plot(
            plot_x,
            plot_y,
            color=style["color"],
            linewidth=1.5,
            label=style["label"],
            zorder=10,
        )

    if not x_values:
        raise RuntimeError("No combined CDF samples found")

    ax.set_xlabel("SM occupancy (%)")
    ax.set_ylabel("CDF (%)")
    ax.set_xlim(0, x_axis_limit([{{"x": x_values}}]))
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, color="gray", alpha=0.7, zorder=-1)
    ax.legend(
        ncol=1,
        fontsize=8,
        handletextpad=0.25,
        handlelength=1.1,
        columnspacing=0.5,
        loc="lower right",
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    base = Path(__file__).resolve().parent
    for worker_type in ("generation", "training"):
        if CDF_DATA.get(worker_type):
            plot_cdf(worker_type, base / f"{{worker_type}}_worker_sm_occupancy_cdf")
        else:
            print(f"SKIPPED={{worker_type}}_worker_sm_occupancy_cdf (no samples)")
    plot_combined_cdf(base / "combined_worker_sm_occupancy_cdf")


if __name__ == "__main__":
    main()
'''
    output.write_text(script, encoding="utf-8")


def iter_worker_samples(
    rows: list[dict[str, Any]],
) -> list[tuple[str, str, list[tuple[float, float]]]]:
    grouped: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for row in rows:
        key = (str(row["worker_type"]), str(row["worker"]))
        grouped.setdefault(key, []).append(
            (float(row["occupancy_pct"]), float(row["duration_us"]))
        )
    return [
        (worker_type, worker, samples)
        for (worker_type, worker), samples in sorted(grouped.items())
    ]


def percentile_from_cdf(xs: list[float], ys: list[float], percentile: float) -> float:
    for x_value, y_value in zip(xs, ys, strict=True):
        if y_value >= percentile:
            return x_value
    return xs[-1] if xs else 0.0


def x_axis_limit(values: list[float]) -> float:
    return max(values) if values else 1.0


def interpolate_cdf_y(xs: list[float], ys: list[float], x_value: float) -> float:
    if x_value <= xs[0]:
        return ys[0]
    for idx in range(1, len(xs)):
        if x_value <= xs[idx]:
            left_x = xs[idx - 1]
            right_x = xs[idx]
            left_y = ys[idx - 1]
            right_y = ys[idx]
            if right_x == left_x:
                return right_y
            ratio = (x_value - left_x) / (right_x - left_x)
            return left_y + ratio * (right_y - left_y)
    return ys[-1]


def smooth_cdf_curve(
    xs: list[float],
    ys: list[float],
    point_count: int = 180,
    window_size: int = 11,
) -> tuple[list[float], list[float]]:
    if len(xs) < 3 or point_count <= len(xs):
        return xs, ys

    min_x = xs[0]
    max_x = xs[-1]
    if max_x <= min_x:
        return xs, ys

    dense_x = [
        min_x + (max_x - min_x) * idx / (point_count - 1)
        for idx in range(point_count)
    ]
    dense_y = [interpolate_cdf_y(xs, ys, x_value) for x_value in dense_x]

    window_size = max(3, min(window_size, point_count))
    if window_size % 2 == 0:
        window_size += 1
    half_window = window_size // 2
    smoothed_y: list[float] = []
    for idx in range(point_count):
        left = max(0, idx - half_window)
        right = min(point_count, idx + half_window + 1)
        smoothed_y.append(sum(dense_y[left:right]) / (right - left))

    smoothed_y[0] = ys[0]
    smoothed_y[-1] = ys[-1]
    cumulative_max = smoothed_y[0]
    for idx, y_value in enumerate(smoothed_y):
        cumulative_max = max(cumulative_max, min(100.0, y_value))
        smoothed_y[idx] = cumulative_max
    smoothed_y[-1] = ys[-1]
    return dense_x, smoothed_y


def filter_occupancy_outliers(
    samples: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    return [
        (value, weight)
        for value, weight in samples
        if 0.0 < value < OUTLIER_MAX_OCCUPANCY_PCT
    ]


def cdf_y_axis_limit(ys: list[float]) -> tuple[float, float]:
    if not ys:
        return (0.0, 1.0)
    lower = max(0.0, (min(ys) // 0.05) * 0.05)
    return (round(lower, 2), 1.0)


def cdf_y_percent(ys: list[float]) -> list[float]:
    return [100.0 * value for value in ys]


def cdf_y_axis_limit_pct(ys: list[float]) -> tuple[float, float]:
    if not ys:
        return (0.0, 100.0)
    lower = max(0.0, (min(ys) // 10.0) * 10.0)
    return (round(lower, 1), 100.0)


def plot_combined_cdf(
    cdf_data: dict[str, dict[str, Any]],
    output_prefix: Path,
) -> bool:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(3.0, 2.5))
    styles = {
        "generation": {"label": "Generation", "color": "#EA5455", "marker": "o"},
        "training": {"label": "Training", "color": "#2D4059", "marker": "s"},
    }

    x_values: list[float] = []
    for worker_type in ("generation", "training"):
        curve = cdf_data[worker_type]
        if not curve["x"]:
            continue
        x_values.extend(curve["x"])
        style = styles[worker_type]
        plot_x, plot_y = smooth_cdf_curve(curve["x"], curve["y_pct"])
        ax.plot(
            plot_x,
            plot_y,
            color=style["color"],
            linewidth=1.5,
            label=style["label"],
            zorder=10,
        )

    if not x_values:
        return False

    ax.set_xlabel("SM occupancy (%)")
    ax.set_ylabel("CDF (%)")
    ax.set_xlim(0, x_axis_limit(x_values))
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, color="gray", alpha=0.7, zorder=-1)
    ax.legend(
        ncol=1,
        fontsize=8,
        handletextpad=0.25,
        handlelength=1.1,
        columnspacing=0.5,
        loc="lower right",
        frameon=True,
    )
    fig.tight_layout()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_cdf(
    cdf_data: dict[str, list[dict[str, Any]]],
    worker_type: str,
    output_prefix: Path,
) -> bool:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 11,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    colors = [
        "#EA5455",
        "#2D4059",
        "#16A3A6",
        "#FFD460",
        "#76BA99",
        "#9DBDFF",
        "#7D5A50",
        "#6C5CE7",
    ]

    x_values: list[float] = []
    curves = cdf_data.get(worker_type, [])
    for idx, curve in enumerate(curves):
        x_values.extend(curve["x"])
        ax.step(
            curve["x"],
            curve["y_pct"],
            where="post",
            color=colors[idx % len(colors)],
            linewidth=1.4,
            label=curve["worker"].replace("_rank", " "),
        )

    if not curves:
        plt.close(fig)
        return False

    ax.set_xlabel("SM occupancy (%)")
    ax.set_ylabel("CDF (%)")
    ax.set_xlim(0, x_axis_limit(x_values))
    y_values = [value for curve in curves for value in curve["y_pct"]]
    ax.set_ylim(*cdf_y_axis_limit_pct(y_values))
    ax.grid(axis="both", linestyle="--", linewidth=0.5, color="gray", alpha=0.45)
    ax.legend(
        ncol=2,
        fontsize=8,
        handletextpad=0.3,
        handlelength=1.4,
        columnspacing=0.6,
        loc="lower right",
        frameon=True,
    )
    fig.tight_layout()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("torch_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or args.torch_dir
    rows = collect_samples(args.torch_dir)
    if not rows:
        raise RuntimeError(f"No generation/training SM occupancy samples under {args.torch_dir}")

    write_samples(rows, output_dir / "worker_sm_occupancy_samples.csv")
    write_summary(rows, output_dir / "worker_sm_occupancy_cdf_summary.csv")
    cdf_data = build_cdf_data(rows)
    combined_cdf_data = build_combined_cdf_data(rows)
    write_cdf_points(cdf_data, output_dir / "worker_sm_occupancy_cdf_points.csv")
    write_combined_cdf_points(
        combined_cdf_data,
        output_dir / "combined_worker_sm_occupancy_cdf_points.csv",
    )
    write_static_plot_script(
        cdf_data,
        combined_cdf_data,
        output_dir / "plot_worker_sm_cdf_static.py",
    )
    wrote_any = False
    for worker_type in ("generation", "training"):
        if plot_cdf(
            cdf_data,
            worker_type,
            output_dir / f"{worker_type}_worker_sm_occupancy_cdf",
        ):
            wrote_any = True
        else:
            print(f"SKIPPED={worker_type}_worker_sm_occupancy_cdf (no samples)")

    if plot_combined_cdf(
        combined_cdf_data,
        output_dir / "combined_worker_sm_occupancy_cdf",
    ):
        wrote_any = True
    else:
        print("SKIPPED=combined_worker_sm_occupancy_cdf (no samples)")

    if not wrote_any:
        raise RuntimeError(f"No plottable SM occupancy samples under {args.torch_dir}")


if __name__ == "__main__":
    main()
