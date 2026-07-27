"""Profiling helpers for rollout eval."""

from toolkits.rollout_eval.profiling.collector import LatencyCollector
from toolkits.rollout_eval.profiling.roofline import (
    AggregatedRooflineRow,
    RooflineKernelRow,
    aggregate_roofline_rows,
    apply_stage_time_overrides,
    load_roofline_csv,
    parse_chrome_trace_to_rows,
    plot_roofline,
    reduce_roofline_rows_by_config_stage,
    write_roofline_csv,
)
from toolkits.rollout_eval.profiling.torch_profiler import (
    TARGET_SPLIT_MODELS,
    RolloutTorchProfiler,
    aggregate_profile_metrics,
)

__all__ = [
    "LatencyCollector",
    "RolloutTorchProfiler",
    "aggregate_profile_metrics",
    "TARGET_SPLIT_MODELS",
    "RooflineKernelRow",
    "AggregatedRooflineRow",
    "apply_stage_time_overrides",
    "parse_chrome_trace_to_rows",
    "aggregate_roofline_rows",
    "write_roofline_csv",
    "load_roofline_csv",
    "reduce_roofline_rows_by_config_stage",
    "plot_roofline",
]
