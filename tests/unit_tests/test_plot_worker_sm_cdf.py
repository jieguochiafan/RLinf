from __future__ import annotations

from tools.plot_worker_sm_cdf import (
    build_combined_cdf_data,
    cdf_points,
    cdf_y_axis_limit,
    cdf_y_axis_limit_pct,
    cdf_y_percent,
    filter_occupancy_outliers,
    phase_names_for_worker,
    plot_cdf,
    smooth_cdf_curve,
    x_axis_limit,
)


def test_cdf_points_are_duration_weighted() -> None:
    x, y = cdf_points([(30.0, 1.0), (10.0, 2.0), (30.0, 1.0)])

    assert x == [10.0, 30.0]
    assert y == [0.5, 1.0]


def test_phase_names_for_generation_and_training_workers() -> None:
    assert phase_names_for_worker("rollout_rank3") == ["generation"]
    assert phase_names_for_worker("rank7") == ["fwd", "loss", "backward", "optimizer.step"]
    assert phase_names_for_worker("env_rank0") == []


def test_x_axis_limit_uses_observed_maximum_sm_occupancy() -> None:
    assert x_axis_limit([5.0, 42.0, 17.0]) == 42.0


def test_filter_occupancy_outliers_removes_zero_and_saturated_high_values() -> None:
    assert filter_occupancy_outliers(
        [(0.0, 1.0), (1.0, 1.0), (94.0, 1.0), (100.0, 1.0)]
    ) == [
        (1.0, 1.0),
        (94.0, 1.0),
    ]


def test_cdf_y_axis_limit_focuses_on_filtered_curve_band() -> None:
    assert cdf_y_axis_limit([0.839, 0.91, 1.0]) == (0.8, 1.0)
    assert cdf_y_axis_limit([0.923, 1.0]) == (0.9, 1.0)


def test_cdf_y_percent_and_axis_limit_use_0_to_100() -> None:
    assert cdf_y_percent([0.25, 1.0]) == [25.0, 100.0]
    assert cdf_y_axis_limit_pct([83.9, 91.0, 100.0]) == (80.0, 100.0)
    assert cdf_y_axis_limit_pct([92.3, 100.0]) == (90.0, 100.0)


def test_build_combined_cdf_data_aggregates_all_ranks_by_worker_type() -> None:
    rows = [
        {"worker_type": "generation", "worker": "rollout_rank0", "occupancy_pct": 1.0, "duration_us": 1.0},
        {"worker_type": "generation", "worker": "rollout_rank1", "occupancy_pct": 2.0, "duration_us": 3.0},
        {"worker_type": "generation", "worker": "rollout_rank1", "occupancy_pct": 0.0, "duration_us": 99.0},
        {"worker_type": "training", "worker": "rank0", "occupancy_pct": 4.0, "duration_us": 1.0},
        {"worker_type": "training", "worker": "rank1", "occupancy_pct": 100.0, "duration_us": 99.0},
    ]

    combined = build_combined_cdf_data(rows)

    assert combined["generation"]["x"] == [1.0, 2.0]
    assert combined["generation"]["y_pct"] == [25.0, 100.0]
    assert combined["training"]["x"] == [4.0]
    assert combined["training"]["y_pct"] == [100.0]


def test_smooth_cdf_curve_keeps_curve_monotonic() -> None:
    x, y = smooth_cdf_curve([1.0, 3.0, 8.0, 10.0], [10.0, 50.0, 60.0, 100.0])

    assert len(x) == len(y)
    assert y[0] == 10.0
    assert y[-1] == 100.0
    assert all(left <= right for left, right in zip(y, y[1:], strict=False))


def test_plot_cdf_skips_missing_worker_type(tmp_path) -> None:
    wrote = plot_cdf(
        {"training": [{"worker": "rank0", "x": [10.0], "y_pct": [100.0]}]},
        "generation",
        tmp_path / "generation_worker_sm_occupancy_cdf",
    )

    assert wrote is False
    assert not (tmp_path / "generation_worker_sm_occupancy_cdf.pdf").exists()
