import pytest

from toolkits.rollout_eval.profiling.gr00t_generation_profile import (
    _summary,
    parse_args,
)


def test_summary_interpolates_percentiles() -> None:
    assert _summary([1.0, 2.0, 3.0, 4.0]) == {
        "avg_ms": 2.5,
        "p50_ms": 2.5,
        "p95_ms": pytest.approx(3.85),
    }


def test_parse_args_uses_rollout_matching_defaults() -> None:
    args = parse_args([])

    assert args.config_name == "libero_object_ppo_gr00t"
    assert args.batch_size == 12
    assert args.mps_sm == 80
    assert args.manage_mps is True


def test_parse_args_rejects_two_cupti_profilers() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--nsys", "--torch-profiler"])
