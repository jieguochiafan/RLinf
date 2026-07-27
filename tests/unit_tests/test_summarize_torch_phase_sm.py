from __future__ import annotations

from tools.summarize_torch_phase_sm import TraceEvent, summarize_phase


def test_summarize_phase_computes_busy_and_weighted_sm_occupancy() -> None:
    events = [
        TraceEvent("generation", "user_annotation", 100.0, 100.0, {}),
        TraceEvent("kernel_a", "kernel", 110.0, 20.0, {"est. achieved occupancy %": 50}),
        TraceEvent("kernel_b", "kernel", 180.0, 40.0, {"est. achieved occupancy %": 100}),
    ]

    summary = summarize_phase(events, "generation")

    assert summary["phase_count"] == 1.0
    assert summary["phase_wall_us"] == 100.0
    assert summary["kernel_overlap_us"] == 40.0
    assert summary["gpu_busy_pct"] == 40.0
    assert summary["est_sm_occupancy_pct"] == 75.0


def test_summarize_phase_returns_zero_without_phase_window() -> None:
    events = [
        TraceEvent("kernel_a", "kernel", 110.0, 20.0, {"est. achieved occupancy %": 50}),
    ]

    summary = summarize_phase(events, "env.step")

    assert summary["phase_count"] == 0.0
    assert summary["gpu_busy_pct"] == 0.0
    assert summary["est_sm_occupancy_pct"] == 0.0
