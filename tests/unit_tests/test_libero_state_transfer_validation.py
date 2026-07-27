import pytest

from toolkits.validate_libero_state_transfer import (
    compute_timing_ratio,
    flat_state_filename,
    record_step_timing,
    sample_action,
    summarize_comparisons,
    summarize_timings,
)


def test_summarize_comparisons_counts_exact_matches_and_max_diff():
    summary = summarize_comparisons(
        [
            {
                "step": 10,
                "images": {
                    "agentview_image": {"equal": True, "max_abs_diff": 0},
                    "robot0_eye_in_hand_image": {"equal": True, "max_abs_diff": 0},
                },
            },
            {
                "step": 20,
                "images": {
                    "agentview_image": {"equal": False, "max_abs_diff": 3},
                    "robot0_eye_in_hand_image": {"equal": True, "max_abs_diff": 0},
                },
            },
        ]
    )

    assert summary["num_syncs"] == 2
    assert summary["all_images_equal"] is False
    assert summary["per_camera"]["agentview_image"]["num_equal"] == 1
    assert summary["per_camera"]["agentview_image"]["max_abs_diff"] == 3
    assert summary["per_camera"]["robot0_eye_in_hand_image"]["num_equal"] == 2


def test_summarize_timings_reports_basic_percentiles():
    summary = summarize_timings([0.001, 0.002, 0.003])

    assert summary["count"] == 3
    assert summary["mean_s"] == pytest.approx(0.002)
    assert summary["min_s"] == pytest.approx(0.001)
    assert summary["max_s"] == pytest.approx(0.003)
    assert summary["median_s"] == pytest.approx(0.002)


def test_summarize_timings_handles_empty_input():
    summary = summarize_timings([])

    assert summary["count"] == 0
    assert summary["mean_s"] is None


def test_compute_timing_ratio_handles_missing_means():
    assert compute_timing_ratio({"mean_s": 4.0}, {"mean_s": 2.0}) == 2.0
    assert compute_timing_ratio({"mean_s": None}, {"mean_s": 2.0}) is None
    assert compute_timing_ratio({"mean_s": 1.0}, {"mean_s": 0.0}) is None


def test_sample_action_is_seeded_and_within_bounds():
    import numpy as np

    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    low = np.array([-1.0, 0.0, 10.0])
    high = np.array([1.0, 2.0, 20.0])

    action_a = sample_action(rng_a, low, high)
    action_b = sample_action(rng_b, low, high)

    assert action_a == action_b
    assert all(lo <= value <= hi for value, lo, hi in zip(action_a, low, high))


def test_flat_state_filename_is_zero_padded_by_step():
    assert flat_state_filename(step_idx=10) == "flat_state_step_000010.npy"


def test_record_step_timing_appends_step_and_latency():
    records = []

    record_step_timing(records, step_idx=3, latency_s=0.123)

    assert records == [{"step": 3, "latency_s": 0.123}]
