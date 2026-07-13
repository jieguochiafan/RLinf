from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from PIL import Image

matplotlib.use("Agg")

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "plot_async_gipo_resource_utilization.py"
)
SPEC = importlib.util.spec_from_file_location(
    "plot_async_gipo_resource_utilization", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
PLOT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLOT
SPEC.loader.exec_module(PLOT)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_profile(
    tmp_path: Path,
    *,
    timestamps: tuple[float, ...] = (1000.0, 1001.0, 1002.0),
    bin_s: float = 1.0,
    long_gpu_labels: bool = False,
) -> Path:
    profile_dir = tmp_path / "resource_profile"
    derived_dir = profile_dir / "derived"
    gpu_labels = (
        ("physical-GPU-label-4-with-a-long-suffix", "physical-GPU-label-5-long")
        if long_gpu_labels
        else ("4", "5")
    )
    _write_csv(
        derived_dir / "cpu_core_1s.csv",
        ["timestamp", "datetime", "cpu", "util_pct", "thread_count", "migration_count"],
        [
            {
                "timestamp": timestamp,
                "datetime": "",
                "cpu": cpu,
                "util_pct": (cpu + index * 7) % 100,
                "thread_count": 1,
                "migration_count": 0,
            }
            for index, timestamp in enumerate(timestamps)
            for cpu in range(4)
        ],
    )
    _write_csv(
        derived_dir / "gpu_device_1s.csv",
        ["timestamp", "datetime", "gpu_uuid", "gpu_label", "kernel_busy_pct"],
        [
            {
                "timestamp": timestamp,
                "datetime": "",
                "gpu_uuid": f"uuid-{gpu_label}",
                "gpu_label": gpu_label,
                "kernel_busy_pct": 20 + 15 * device + index * 5,
            }
            for index, timestamp in enumerate(timestamps)
            for device, gpu_label in enumerate(gpu_labels)
        ],
    )
    worker_rows = []
    for index, timestamp in enumerate(timestamps):
        for component, rank, worker, gpu_label, occupancy in (
            ("training", 0, "actor-rank-0", gpu_labels[0], 65 + index),
            ("generation", 1, "rollout-rank-1", gpu_labels[1], 45 + index),
            ("env", 2, "env-rank-2", gpu_labels[0], 25 + index),
        ):
            worker_rows.append(
                {
                    "timestamp": timestamp,
                    "datetime": "",
                    "worker": worker,
                    "component": component,
                    "rank": rank,
                    "gpu_uuid": f"uuid-{gpu_label}",
                    "gpu_label": gpu_label,
                    "kernel_busy_pct": 30,
                    "est_sm_occupancy_pct": occupancy,
                    "kernel_count": 1,
                }
            )
    if long_gpu_labels:
        worker_rows.extend(
            {
                "timestamp": timestamp,
                "datetime": "",
                "worker": "actor-rank-0",
                "component": "training",
                "rank": 0,
                "gpu_uuid": "uuid-second-device",
                "gpu_label": gpu_labels[1],
                "kernel_busy_pct": 30,
                "est_sm_occupancy_pct": 55,
                "kernel_count": 1,
            }
            for timestamp in timestamps
        )
    _write_csv(
        derived_dir / "gpu_worker_1s.csv",
        [
            "timestamp",
            "datetime",
            "worker",
            "component",
            "rank",
            "gpu_uuid",
            "gpu_label",
            "kernel_busy_pct",
            "est_sm_occupancy_pct",
            "kernel_count",
        ],
        worker_rows,
    )
    _write_csv(
        derived_dir / "phase_windows.csv",
        ["phase", "step", "component", "rank", "start", "end", "source"],
        [
            {
                "phase": "step",
                "step": 3,
                "component": "runner",
                "rank": "",
                "start": timestamps[0],
                "end": timestamps[-1] + bin_s,
                "source": "test",
            },
            {
                "phase": "actor_training",
                "step": 3,
                "component": "training",
                "rank": 0,
                "start": timestamps[1],
                "end": timestamps[-1] + bin_s,
                "source": "test",
            },
        ],
    )
    (derived_dir / "coverage.json").write_text(
        json.dumps({"warnings": ["synthetic coverage warning"]})
    )
    (profile_dir / "metadata.json").write_text(
        json.dumps({"bin_s": bin_s, "num_cpus": 4})
    )
    return derived_dir


def test_build_cpu_matrix_reconstructs_zero_and_missing_bins() -> None:
    rows = [
        {"timestamp": "100", "cpu": "0", "util_pct": "10"},
        {"timestamp": "100", "cpu": "2", "util_pct": "30"},
        {"timestamp": "102", "cpu": "1", "util_pct": "40"},
    ]

    times, matrix = PLOT.build_cpu_matrix(rows, num_cpus=4, bin_s=1.0)

    np.testing.assert_allclose(times, [100, 101, 102])
    np.testing.assert_allclose(matrix[:, 0], [10, 0, 30, 0])
    assert np.isnan(matrix[:, 1]).all()
    np.testing.assert_allclose(matrix[:, 2], [0, 40, 0, 0])
    assert matrix.shape == (4, 3)


def test_build_cpu_matrix_rejects_invalid_logical_core() -> None:
    with pytest.raises(ValueError, match="logical CPU"):
        PLOT.build_cpu_matrix(
            [{"timestamp": "100", "cpu": "4", "util_pct": "10"}],
            num_cpus=4,
            bin_s=1.0,
        )


def test_build_worker_matrix_sorts_labels_and_preserves_nan() -> None:
    rows = [
        {
            "timestamp": "100",
            "worker": "rollout-one",
            "component": "generation",
            "rank": "1",
            "gpu_uuid": "g4",
            "gpu_label": "4",
            "est_sm_occupancy_pct": "nan",
        },
        {
            "timestamp": "100",
            "worker": "actor-zero",
            "component": "training",
            "rank": "0",
            "gpu_uuid": "g7",
            "gpu_label": "7",
            "est_sm_occupancy_pct": "70",
        },
        {
            "timestamp": "100",
            "worker": "actor-zero",
            "component": "training",
            "rank": "0",
            "gpu_uuid": "g6",
            "gpu_label": "6",
            "est_sm_occupancy_pct": "60",
        },
        {
            "timestamp": "100",
            "worker": "env-two",
            "component": "env",
            "rank": "2",
            "gpu_uuid": "g5",
            "gpu_label": "5",
            "est_sm_occupancy_pct": "20",
        },
    ]

    times, labels, matrix = PLOT.build_worker_matrix(rows, bin_s=1.0)

    np.testing.assert_allclose(times, [100])
    assert labels == [
        "actor r0 (GPU 6)",
        "actor r0 (GPU 7)",
        "rollout r1",
        "env r2",
    ]
    assert np.isnan(matrix[2, 0])


def test_build_gpu_device_series_sorts_and_marks_missing_bin() -> None:
    rows = [
        {
            "timestamp": "100",
            "gpu_uuid": "g7",
            "gpu_label": "7",
            "kernel_busy_pct": "20",
        },
        {
            "timestamp": "100",
            "gpu_uuid": "g4",
            "gpu_label": "4",
            "kernel_busy_pct": "10",
        },
        {
            "timestamp": "102",
            "gpu_uuid": "g7",
            "gpu_label": "7",
            "kernel_busy_pct": "40",
        },
        {
            "timestamp": "102",
            "gpu_uuid": "g4",
            "gpu_label": "4",
            "kernel_busy_pct": "30",
        },
    ]

    times, labels, matrix = PLOT.build_gpu_device_series(rows, bin_s=1.0)

    np.testing.assert_allclose(times, [100, 101, 102])
    assert labels == ["4", "7"]
    np.testing.assert_allclose(matrix[:, 0], [10, 20])
    assert np.isnan(matrix[:, 1]).all()
    np.testing.assert_allclose(matrix[:, 2], [30, 40])


def test_build_gpu_device_series_keeps_per_device_missing_sample_nan() -> None:
    rows = [
        {
            "timestamp": "100",
            "gpu_uuid": "g4",
            "gpu_label": "4",
            "kernel_busy_pct": "10",
        },
        {
            "timestamp": "100",
            "gpu_uuid": "g7",
            "gpu_label": "7",
            "kernel_busy_pct": "20",
        },
        {
            "timestamp": "101",
            "gpu_uuid": "g7",
            "gpu_label": "7",
            "kernel_busy_pct": "30",
        },
    ]

    times, labels, matrix = PLOT.build_gpu_device_series(rows, bin_s=1.0)

    np.testing.assert_allclose(times, [100, 101])
    assert labels == ["4", "7"]
    np.testing.assert_allclose(matrix[0], [10, np.nan], equal_nan=True)
    np.testing.assert_allclose(matrix[1], [20, 30])


def test_build_gpu_device_series_validates_busy_percentage() -> None:
    with pytest.raises(ValueError, match="0..100"):
        PLOT.build_gpu_device_series(
            [
                {
                    "timestamp": "100",
                    "gpu_uuid": "g4",
                    "gpu_label": "4",
                    "kernel_busy_pct": "101",
                }
            ],
            bin_s=1.0,
        )


def test_read_phase_windows_uses_epoch_seconds_relative_to_origin(
    tmp_path: Path,
) -> None:
    path = tmp_path / "phase_windows.csv"
    _write_csv(
        path,
        ["phase", "step", "component", "rank", "start", "end", "source"],
        [
            {
                "phase": "actor_training",
                "step": 4,
                "component": "training",
                "rank": 0,
                "start": 1000.0,
                "end": 1002.5,
                "source": "test",
            }
        ],
    )

    phases = PLOT.read_phase_windows(path, origin=999.5)

    assert phases[0].start == pytest.approx(0.5)
    assert phases[0].end == pytest.approx(3.0)
    assert phases[0].step == 4


def test_plot_writes_nonempty_png_and_pdf(tmp_path: Path) -> None:
    derived_dir = _write_profile(tmp_path)
    output_prefix = tmp_path / "figures" / "async_gipo_util"

    figure = PLOT.create_figure(derived_dir, num_cpus=4)
    outputs = PLOT.save_figure(figure, output_prefix)

    assert outputs == (
        output_prefix.with_suffix(".pdf"),
        output_prefix.with_suffix(".png"),
    )
    assert all(path.stat().st_size > 1024 for path in outputs)
    with Image.open(outputs[1]) as image:
        pixels = np.asarray(image.convert("RGB"))
        assert image.width >= 1500
        assert image.height >= 900
        assert pixels.std() > 5
    PLOT.plt.close(figure)


def test_axes_use_required_labels_without_sm_active(tmp_path: Path) -> None:
    figure = PLOT.create_figure(_write_profile(tmp_path), num_cpus=4)
    try:
        text = "\n".join(
            [axis.get_xlabel() for axis in figure.axes]
            + [axis.get_ylabel() for axis in figure.axes]
            + [item.get_text() for axis in figure.axes for item in axis.texts]
        )
        assert "RLinf CPU utilization per logical core (%)" in text
        assert "CUDA kernel busy by physical GPU (%)" in text
        assert "Estimated SM occupancy by worker (%)" in text
        assert "SM_ACTIVE" not in text
    finally:
        PLOT.plt.close(figure)


def test_figure_uses_required_size_ratios_and_pdf_fonttype(tmp_path: Path) -> None:
    figure = PLOT.create_figure(_write_profile(tmp_path), num_cpus=4)
    try:
        np.testing.assert_allclose(figure.get_size_inches(), [12, 8])
        main_axes = [
            axis
            for axis in figure.axes
            if axis.get_ylabel()
            in {
                "RLinf CPU utilization per logical core (%)",
                "CUDA kernel busy by physical GPU (%)",
                "Estimated SM occupancy by worker (%)",
            }
        ]
        assert len(main_axes) == 3
        grid_spec = main_axes[0].get_subplotspec().get_gridspec()
        np.testing.assert_allclose(grid_spec.get_height_ratios(), [3.2, 1.4, 2.0])
        assert matplotlib.rcParams["pdf.fonttype"] == 42
    finally:
        PLOT.plt.close(figure)


def test_phase_styles_are_shared_across_three_main_axes(tmp_path: Path) -> None:
    figure = PLOT.create_figure(_write_profile(tmp_path), num_cpus=4)
    try:
        main_axes = [
            axis
            for axis in figure.axes
            if axis.get_ylabel()
            in {
                "RLinf CPU utilization per logical core (%)",
                "CUDA kernel busy by physical GPU (%)",
                "Estimated SM occupancy by worker (%)",
            }
        ]
        expected_span_color = matplotlib.colors.to_rgba("#E15759", alpha=0.08)
        for axis in main_axes:
            assert len(axis.patches) == 1
            span = axis.patches[0]
            assert span.get_facecolor() == pytest.approx(expected_span_color)
            assert span.get_alpha() == pytest.approx(0.08)
            step_lines = [line for line in axis.lines if line.get_color() == "black"]
            assert len(step_lines) == 1
            assert step_lines[0].get_linewidth() == pytest.approx(0.7)
            assert step_lines[0].get_alpha() == pytest.approx(0.45)
    finally:
        PLOT.plt.close(figure)


def test_cli_missing_input_is_nonzero_and_clear(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            str(tmp_path),
            "--output-prefix",
            str(tmp_path / "missing"),
            "--num-cpus",
            "112",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Missing required input" in result.stderr
    assert "cpu_core_1s.csv" in result.stderr


def test_cli_prints_coverage_warnings_to_stderr(tmp_path: Path) -> None:
    derived_dir = _write_profile(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            str(derived_dir),
            "--output-prefix",
            str(tmp_path / "warning-figure"),
            "--num-cpus",
            "4",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "WARNING: synthetic coverage warning" in result.stderr


def test_longest_worker_label_is_not_clipped(tmp_path: Path) -> None:
    figure = PLOT.create_figure(
        _write_profile(tmp_path, long_gpu_labels=True), num_cpus=4
    )
    try:
        figure.canvas.draw()
        worker_axis = next(
            axis
            for axis in figure.axes
            if axis.get_ylabel() == "Estimated SM occupancy by worker (%)"
        )
        renderer = figure.canvas.get_renderer()
        figure_box = figure.bbox
        label_boxes = [
            label.get_window_extent(renderer) for label in worker_axis.get_yticklabels()
        ]
        assert label_boxes
        assert min(box.x0 for box in label_boxes) >= figure_box.x0
    finally:
        PLOT.plt.close(figure)


def test_metadata_bin_size_takes_priority_over_timestamp_inference(
    tmp_path: Path,
) -> None:
    derived_dir = _write_profile(tmp_path, timestamps=(100.0, 102.0), bin_s=1.0)

    profile = PLOT.load_profile(derived_dir, num_cpus=4)

    assert profile.bin_s == pytest.approx(1.0)
    np.testing.assert_allclose(profile.cpu_times, [100, 101, 102])
    assert np.isnan(profile.cpu_matrix[:, 1]).all()
