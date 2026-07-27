from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import matplotlib.pyplot as plt  # noqa: E402

from tools import plot_cpu_gpu_util_timeline as timeline  # noqa: E402
from tools.plot_cpu_core_util_timeline import PhaseWindow  # noqa: E402


class _FakeFrame:
    def set_edgecolor(self, color: str) -> None:
        self.edgecolor = color

    def set_linewidth(self, linewidth: float) -> None:
        self.linewidth = linewidth


class _FakeLegend:
    def __init__(self) -> None:
        self.frame = _FakeFrame()

    def get_frame(self) -> _FakeFrame:
        return self.frame


class _FakeAxes:
    def __init__(self) -> None:
        self.fills = []
        self.plot_labels = []
        self.text_values = []
        self.vline_colors = []
        self.text_colors = []

    def fill_between(self, *args, **kwargs) -> None:
        self.fills.append((args, kwargs))

    def plot(self, *args, **kwargs) -> None:
        self.plot_labels.append(kwargs.get("label"))

    def axvline(self, *args, **kwargs) -> None:
        self.vline_colors.append(kwargs.get("color"))

    def text(self, *args, **kwargs) -> None:
        self.text_values.append(args[2])
        self.text_colors.append(kwargs.get("color"))

    def set_ylabel(self, *args, **kwargs) -> None:
        pass

    def legend(self, *args, **kwargs) -> _FakeLegend:
        return _FakeLegend()


def test_draw_cpu_panel_does_not_draw_red_background() -> None:
    axes = _FakeAxes()
    times = np.array([1_700_000_000.0, 1_700_000_001.0], dtype=float)

    timeline.draw_cpu_panel(
        ax=axes,
        times=times,
        mean_util=np.array([10.0, 20.0], dtype=float),
        p90_util=np.array([40.0, 50.0], dtype=float),
        max_util=np.array([90.0, 100.0], dtype=float),
        phases=[],
        smooth_window_s=0.0,
        detail_windows=[],
    )

    assert axes.fills == []
    assert axes.plot_labels == ["Mean"]


def test_draw_cpu_panel_can_skip_inset() -> None:
    start = datetime(2026, 6, 7, 10, 0, 0)
    times = np.array([start.timestamp() + idx for idx in range(4)], dtype=float)
    process_cpu_series = {
        "Generation": (
            np.array([start.timestamp() + 0.5]),
            np.array([50.0]),
        ),
        "Simulator": (
            np.array([start.timestamp() + 1.5]),
            np.array([1500.0]),
        ),
    }
    fig, ax = plt.subplots()

    try:
        inset_window = timeline.draw_cpu_panel(
            ax=ax,
            times=times,
            mean_util=np.array([10.0, 20.0, 30.0, 40.0], dtype=float),
            p90_util=np.array([10.0, 20.0, 30.0, 40.0], dtype=float),
            max_util=np.array([10.0, 20.0, 30.0, 40.0], dtype=float),
            phases=[],
            smooth_window_s=0.0,
            detail_windows=[],
            process_cpu_series=process_cpu_series,
            draw_inset=False,
        )

        assert inset_window is None
        assert ax.child_axes == []
    finally:
        plt.close(fig)


def test_draw_phase_markers_uses_black_for_rollout_and_train() -> None:
    axes = _FakeAxes()
    start = datetime(2026, 6, 7, 10, 0, 0)

    timeline.draw_phase_markers(
        axes,
        [
            PhaseWindow(
                name="Rollout",
                start=start,
                end=start + timedelta(seconds=30),
                source="test",
            ),
            PhaseWindow(
                name="Train",
                start=start + timedelta(seconds=40),
                end=start + timedelta(seconds=50),
                source="test",
            ),
        ],
        label=True,
    )

    assert axes.vline_colors == ["black", "black", "black", "black"]
    assert axes.text_colors == ["black", "black"]


def test_select_cpu_inset_window_prefers_high_variation_rollout_segment() -> None:
    start = datetime(2026, 6, 7, 10, 0, 0)
    times = np.array([start.timestamp() + idx for idx in range(100)], dtype=float)
    mean_util = np.full(100, 35.0, dtype=float)
    mean_util[45:56] = np.array(
        [25.0, 34.0, 48.0, 31.0, 58.0, 37.0, 62.0, 41.0, 55.0, 36.0, 49.0],
        dtype=float,
    )
    mean_util[80:91] = np.array(
        [5.0, 90.0, 8.0, 95.0, 7.0, 92.0, 6.0, 94.0, 9.0, 91.0, 10.0],
        dtype=float,
    )

    window = timeline.select_cpu_inset_window(
        times,
        mean_util,
        [
            PhaseWindow(
                name="Rollout",
                start=start + timedelta(seconds=40),
                end=start + timedelta(seconds=60),
                source="test",
            )
        ],
        window_s=10.0,
    )

    assert window is not None
    assert start + timedelta(seconds=45) <= window[0] <= start + timedelta(seconds=50)
    assert window[1] - window[0] == timedelta(seconds=10)


def test_read_rollout_detail_windows_labels_generation_and_simulator(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path
    generation_dir = run_dir / "logs" / "rollout_generation_timestamps"
    simulator_dir = run_dir / "logs" / "env_sim_timestamps"
    generation_dir.mkdir(parents=True)
    simulator_dir.mkdir(parents=True)
    base_ns = 1_780_000_000_000_000_000
    (generation_dir / "rollout_rank_0.jsonl").write_text(
        "\n".join(
            [
                f'{{"event": "start", "wall_ns": {base_ns}}}',
                f'{{"event": "end", "wall_ns": {base_ns + 200_000_000}}}',
            ]
        )
    )
    (simulator_dir / "env_rank_0.jsonl").write_text(
        "\n".join(
            [
                f'{{"event": "start", "wall_ns": {base_ns + 300_000_000}}}',
                f'{{"event": "end", "wall_ns": {base_ns + 800_000_000}}}',
            ]
        )
    )

    windows = timeline.read_rollout_detail_windows(run_dir)

    assert [window.name for window in windows] == ["Generation", "Simulator"]
    assert windows[0].end - windows[0].start == timedelta(seconds=0.2)
    assert windows[1].end - windows[1].start == timedelta(seconds=0.5)


def test_read_process_detail_windows_includes_simulator_children(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path
    generation_dir = run_dir / "logs" / "rollout_generation_timestamps"
    simulator_dir = run_dir / "logs" / "env_sim_timestamps"
    generation_dir.mkdir(parents=True)
    simulator_dir.mkdir(parents=True)
    base_ns = 1_780_000_000_000_000_000
    (generation_dir / "rollout_rank_0.jsonl").write_text(
        "\n".join(
            [
                (
                    f'{{"event": "start", "wall_ns": {base_ns}, '
                    '"pid": 10, "rank": 0, "epoch": 0, "chunk_step": 0, '
                    '"stage": 0, "phase": "action_generation"}'
                ),
                (
                    f'{{"event": "end", "wall_ns": {base_ns + 200_000_000}, '
                    '"pid": 10, "rank": 0, "epoch": 0, "chunk_step": 0, '
                    '"stage": 0, "phase": "action_generation"}'
                ),
            ]
        )
    )
    (simulator_dir / "env_rank_0.jsonl").write_text(
        "\n".join(
            [
                (
                    f'{{"event": "start", "wall_ns": {base_ns + 300_000_000}, '
                    '"pid": 20, "rank": 0, "epoch": 0, "chunk_step": 0, '
                    '"stage": 0}'
                ),
                (
                    f'{{"event": "subenv_start", "wall_ns": {base_ns + 350_000_000}, '
                    '"child_pid": 30, "rank": 0, "epoch": 0, "chunk_step": 0, '
                    '"stage": 0, "vector_step": 0, "global_env": 0, '
                    '"local_env": 0, "operation": "step"}'
                ),
                (
                    f'{{"event": "subenv_end", "wall_ns": {base_ns + 650_000_000}, '
                    '"child_pid": 30, "rank": 0, "epoch": 0, "chunk_step": 0, '
                    '"stage": 0, "vector_step": 0, "global_env": 0, '
                    '"local_env": 0, "operation": "step"}'
                ),
                (
                    f'{{"event": "end", "wall_ns": {base_ns + 800_000_000}, '
                    '"pid": 20, "rank": 0, "epoch": 0, "chunk_step": 0, '
                    '"stage": 0}'
                ),
            ]
        )
    )

    windows = timeline.read_process_detail_windows(run_dir)

    assert sorted((window.phase, window.pid) for window in windows) == [
        ("Generation", 10),
        ("Simulator", 20),
        ("Simulator", 30),
    ]


def test_merge_detail_windows_collapses_overlapping_rank_windows() -> None:
    start = datetime(2026, 6, 7, 10, 0, 0)
    windows = [
        PhaseWindow(
            name="Generation",
            start=start,
            end=start + timedelta(seconds=0.3),
            source="rank0",
        ),
        PhaseWindow(
            name="Generation",
            start=start + timedelta(seconds=0.05),
            end=start + timedelta(seconds=0.4),
            source="rank1",
        ),
        PhaseWindow(
            name="Simulator",
            start=start + timedelta(seconds=1.0),
            end=start + timedelta(seconds=1.2),
            source="rank0",
        ),
        PhaseWindow(
            name="Simulator",
            start=start + timedelta(seconds=1.22),
            end=start + timedelta(seconds=1.4),
            source="rank1",
        ),
    ]

    merged = timeline.merge_detail_windows(windows, max_gap_s=0.05)

    assert [(window.name, window.start, window.end) for window in merged] == [
        ("Generation", start, start + timedelta(seconds=0.4)),
        ("Simulator", start + timedelta(seconds=1.0), start + timedelta(seconds=1.4)),
    ]


def test_summarize_detail_cpu_util_uses_window_overlap_weights() -> None:
    start = datetime(2026, 6, 7, 10, 0, 0)
    times = np.array([start.timestamp() + idx for idx in range(6)], dtype=float)
    mean_util = np.array([10.0, 20.0, 80.0, 90.0, 95.0, 30.0], dtype=float)
    detail_windows = [
        PhaseWindow(
            name="Generation",
            start=start + timedelta(seconds=0.75),
            end=start + timedelta(seconds=1.25),
            source="test",
        ),
        PhaseWindow(
            name="Simulator",
            start=start + timedelta(seconds=2.0),
            end=start + timedelta(seconds=4.0),
            source="test",
        ),
    ]

    stats = timeline.summarize_detail_cpu_util(
        times,
        mean_util,
        detail_windows,
        (start, start + timedelta(seconds=5)),
    )

    assert stats["Generation"] == 20.0
    assert np.isclose(stats["Simulator"], 88.75)


def test_summarize_detail_cpu_load_preserves_phase_duration() -> None:
    start = datetime(2026, 6, 7, 10, 0, 0)
    times = np.array([start.timestamp() + idx for idx in range(6)], dtype=float)
    mean_util = np.array([10.0, 20.0, 80.0, 90.0, 95.0, 30.0], dtype=float)
    detail_windows = [
        PhaseWindow(
            name="Generation",
            start=start + timedelta(seconds=1.0),
            end=start + timedelta(seconds=2.0),
            source="test",
        ),
        PhaseWindow(
            name="Simulator",
            start=start + timedelta(seconds=2.0),
            end=start + timedelta(seconds=4.0),
            source="test",
        ),
    ]

    loads = timeline.summarize_detail_cpu_load(
        times,
        mean_util,
        detail_windows,
        (start, start + timedelta(seconds=5)),
    )

    assert loads["Generation"] == 50.0
    assert np.isclose(loads["Simulator"], 177.5)


def test_compute_detail_points_returns_one_point_per_phase() -> None:
    start = datetime(2026, 6, 7, 10, 0, 0)
    times = np.array([start.timestamp() + idx for idx in range(6)], dtype=float)
    mean_util = np.array([10.0, 20.0, 80.0, 90.0, 95.0, 30.0], dtype=float)
    detail_windows = [
        PhaseWindow(
            name="Generation",
            start=start + timedelta(seconds=1.0),
            end=start + timedelta(seconds=2.0),
            source="test",
        ),
        PhaseWindow(
            name="Simulator",
            start=start + timedelta(seconds=2.0),
            end=start + timedelta(seconds=4.0),
            source="test",
        ),
    ]

    points = timeline.compute_detail_points(
        times,
        mean_util,
        detail_windows,
        (start, start + timedelta(seconds=5)),
    )

    assert len(points) == 2
    assert points[0][0] == "Generation"
    assert points[0][1] == start + timedelta(seconds=1.5)
    assert points[0][2] == 50.0
    assert points[1][0] == "Simulator"
    assert points[1][1] == start + timedelta(seconds=3.0)
    assert np.isclose(points[1][2], 88.75)


def test_compute_process_phase_cpu_series_attributes_pid_windows(
    tmp_path: Path,
) -> None:
    raw_cpu = tmp_path / "worker_cpu_core_util.csv"
    raw_cpu.write_text(
        "\n".join(
            [
                "timestamp,pid,worker_label,cpu,active_pct,proc_delta_jiffies,core_busy_delta_jiffies,core_total_delta_jiffies",
                "0.0,10,gen,0,0,0,0,100",
                "0.0,20,sim,0,0,0,0,100",
                "1.0,10,gen,0,0,100,0,100",
                "1.0,10,gen,1,0,100,0,100",
                "1.0,20,sim,0,0,0,0,100",
                "2.0,10,gen,0,0,50,0,100",
                "2.0,20,sim,0,0,200,0,100",
                "3.0,20,sim,0,0,200,0,100",
            ]
        )
    )
    windows = [
        timeline.ProcessWindow(
            phase="Generation",
            pid=10,
            start=datetime.fromtimestamp(0.0),
            end=datetime.fromtimestamp(2.0),
        ),
        timeline.ProcessWindow(
            phase="Simulator",
            pid=20,
            start=datetime.fromtimestamp(1.0),
            end=datetime.fromtimestamp(3.0),
        ),
    ]

    series = timeline.compute_process_phase_cpu_series(
        raw_cpu,
        windows,
        bin_s=1.0,
        clock_ticks_per_second=100,
    )

    np.testing.assert_allclose(series["Generation"][0], np.array([0.5, 1.5]))
    np.testing.assert_allclose(series["Generation"][1], np.array([100.0, 50.0]))
    np.testing.assert_allclose(series["Simulator"][0], np.array([1.5, 2.5]))
    np.testing.assert_allclose(series["Simulator"][1], np.array([200.0, 200.0]))


def test_compute_process_phase_cpu_series_sums_concurrent_children(
    tmp_path: Path,
) -> None:
    raw_cpu = tmp_path / "worker_cpu_core_util.csv"
    raw_cpu.write_text(
        "\n".join(
            [
                "timestamp,pid,worker_label,cpu,active_pct,proc_delta_jiffies,core_busy_delta_jiffies,core_total_delta_jiffies",
                "0.0,20,sim,0,0,0,0,100",
                "0.0,30,sim,0,0,0,0,100",
                "1.0,20,sim,0,0,100,0,100",
                "1.0,30,sim,0,0,200,0,100",
            ]
        )
    )
    windows = [
        timeline.ProcessWindow(
            phase="Simulator",
            pid=20,
            start=datetime.fromtimestamp(0.0),
            end=datetime.fromtimestamp(1.0),
        ),
        timeline.ProcessWindow(
            phase="Simulator",
            pid=30,
            start=datetime.fromtimestamp(0.0),
            end=datetime.fromtimestamp(1.0),
        ),
    ]

    series = timeline.compute_process_phase_cpu_series(
        raw_cpu,
        windows,
        bin_s=1.0,
        clock_ticks_per_second=100,
    )

    np.testing.assert_allclose(series["Simulator"][0], np.array([0.5]))
    np.testing.assert_allclose(series["Simulator"][1], np.array([300.0]))


def test_build_process_phase_timeline_points_orders_points_by_time() -> None:
    start = datetime(2026, 6, 7, 10, 0, 0)
    process_cpu_series = {
        "Generation": (
            np.array([start.timestamp() + second for second in (0.5, 2.5)]),
            np.array([80.0, 40.0]),
        ),
        "Simulator": (
            np.array([start.timestamp() + second for second in (1.5, 3.5)]),
            np.array([1800.0, 1600.0]),
        ),
    }

    points = timeline.build_process_phase_timeline_points(
        process_cpu_series,
        (start, start + timedelta(seconds=4)),
    )

    assert [(phase, value) for phase, _time, value in points] == [
        ("Generation", 80.0),
        ("Simulator", 1800.0),
        ("Generation", 40.0),
        ("Simulator", 1600.0),
    ]


def test_draw_cpu_inset_uses_two_worker_lines_with_cpu_legend() -> None:
    start = datetime(2026, 6, 7, 10, 0, 0)
    process_cpu_series = {
        "Generation": (
            np.array([start.timestamp() + second for second in (0.5, 1.5, 2.5)]),
            np.array([80.0, 120.0, 40.0]),
        ),
        "Simulator": (
            np.array([start.timestamp() + second for second in (0.5, 1.5, 2.5)]),
            np.array([1400.0, 1800.0, 1600.0]),
        ),
    }
    detail_windows = [
        PhaseWindow(
            name="Generation",
            start=start,
            end=start + timedelta(seconds=3.0),
            source="test",
        ),
        PhaseWindow(
            name="Simulator",
            start=start,
            end=start + timedelta(seconds=3.0),
            source="test",
        ),
    ]
    fig, ax = plt.subplots()

    try:
        timeline.draw_cpu_inset(
            ax,
            (start, start + timedelta(seconds=3)),
            detail_windows,
            process_cpu_series,
        )

        inset_axes = ax.child_axes
        assert len(inset_axes) == 1
        inset_ax = inset_axes[0]
        assert [line.get_label() for line in inset_ax.lines] == [
            "Gen: 80%",
            "Sim: 1600%",
        ]
        assert len({line.get_color() for line in inset_ax.lines}) == 2
        assert len(inset_ax.patches) == 0
        legend = inset_ax.get_legend()
        assert legend is not None
        assert [text.get_text() for text in legend.get_texts()] == [
            "Gen: 80%",
            "Sim: 1600%",
        ]
    finally:
        plt.close(fig)


def test_save_cpu_inset_svg_writes_standalone_svg(tmp_path: Path) -> None:
    start = datetime(2026, 6, 7, 10, 0, 0)
    process_cpu_series = {
        "Generation": (
            np.array([start.timestamp() + second for second in (0.5, 1.5)]),
            np.array([80.0, 120.0]),
        ),
        "Simulator": (
            np.array([start.timestamp() + second for second in (0.5, 1.5)]),
            np.array([1400.0, 1800.0]),
        ),
    }
    output_path = tmp_path / "cpu_inset.svg"

    written_path = timeline.save_cpu_inset_svg(
        output_path,
        (start, start + timedelta(seconds=2)),
        process_cpu_series,
    )

    assert written_path == output_path
    assert output_path.exists()
    assert "<svg" in output_path.read_text()


def test_select_cpu_inset_window_prefers_nearby_gen_sim_pair() -> None:
    start = datetime(2026, 6, 7, 10, 0, 0)
    times = np.array([start.timestamp() + idx for idx in range(90)], dtype=float)
    mean_util = np.full(90, 30.0, dtype=float)
    mean_util[10:13] = 15.0
    mean_util[15:23] = 85.0
    mean_util[70:73] = 5.0
    mean_util[80:88] = 98.0
    detail_windows = [
        PhaseWindow(
            name="Generation",
            start=start + timedelta(seconds=10),
            end=start + timedelta(seconds=12),
            source="test",
        ),
        PhaseWindow(
            name="Simulator",
            start=start + timedelta(seconds=15),
            end=start + timedelta(seconds=22),
            source="test",
        ),
        PhaseWindow(
            name="Generation",
            start=start + timedelta(seconds=70),
            end=start + timedelta(seconds=72),
            source="test",
        ),
        PhaseWindow(
            name="Simulator",
            start=start + timedelta(seconds=80),
            end=start + timedelta(seconds=87),
            source="test",
        ),
    ]

    window = timeline.select_cpu_inset_window(
        times,
        mean_util,
        phases=[],
        detail_windows=detail_windows,
        window_s=15.0,
    )

    assert window is not None
    assert window[0] <= start + timedelta(seconds=10)
    assert window[1] >= start + timedelta(seconds=22)


def test_select_cpu_inset_window_checks_detail_pairs_linearly(monkeypatch) -> None:
    start = datetime(2026, 6, 7, 10, 0, 0)
    times = np.array([start.timestamp() + idx for idx in range(400)], dtype=float)
    mean_util = np.full(400, 30.0, dtype=float)
    detail_windows = []
    for idx in range(60):
        offset_s = idx * 5
        detail_windows.extend(
            [
                PhaseWindow(
                    name="Generation",
                    start=start + timedelta(seconds=offset_s),
                    end=start + timedelta(seconds=offset_s + 1),
                    source="test",
                ),
                PhaseWindow(
                    name="Simulator",
                    start=start + timedelta(seconds=offset_s + 2),
                    end=start + timedelta(seconds=offset_s + 4),
                    source="test",
                ),
            ]
        )
    calls = []

    def fake_summarize(times_arg, mean_arg, detail_windows_arg, window_arg):
        calls.append((times_arg, mean_arg, detail_windows_arg, window_arg))
        return {"Generation": 10.0, "Simulator": 80.0}

    monkeypatch.setattr(timeline, "summarize_detail_cpu_util", fake_summarize)

    window = timeline.select_cpu_inset_window(
        times,
        mean_util,
        phases=[],
        detail_windows=detail_windows,
        window_s=5.0,
    )

    assert window is not None
    assert len(calls) <= 120


def test_select_cpu_inset_window_merges_duplicate_detail_windows(monkeypatch) -> None:
    start = datetime(2026, 6, 7, 10, 0, 0)
    times = np.array([start.timestamp() + idx for idx in range(20)], dtype=float)
    mean_util = np.full(20, 30.0, dtype=float)
    detail_windows = []
    for _ in range(60):
        detail_windows.extend(
            [
                PhaseWindow(
                    name="Generation",
                    start=start,
                    end=start + timedelta(seconds=1),
                    source="test",
                ),
                PhaseWindow(
                    name="Simulator",
                    start=start + timedelta(seconds=2),
                    end=start + timedelta(seconds=4),
                    source="test",
                ),
            ]
        )
    calls = []

    def fake_summarize(times_arg, mean_arg, detail_windows_arg, window_arg):
        calls.append((times_arg, mean_arg, detail_windows_arg, window_arg))
        return {"Generation": 10.0, "Simulator": 80.0}

    monkeypatch.setattr(timeline, "summarize_detail_cpu_util", fake_summarize)

    window = timeline.select_cpu_inset_window(
        times,
        mean_util,
        phases=[],
        detail_windows=detail_windows,
        window_s=5.0,
    )

    assert window is not None
    assert len(calls) == 1


def test_draw_rollout_note_does_not_add_text() -> None:
    axes = _FakeAxes()
    start = datetime(2026, 6, 7, 10, 0, 0)

    timeline.draw_rollout_note(
        axes,
        [
            PhaseWindow(
                name="Rollout",
                start=start,
                end=start + timedelta(seconds=90),
                source="test",
            )
        ],
    )

    assert axes.text_values == []
    assert axes.text_colors == []
