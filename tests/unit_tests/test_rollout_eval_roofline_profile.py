from __future__ import annotations

from pathlib import Path

from toolkits.rollout_eval.benchmark import roofline_profile


def test_roofline_profile_parse_args_defaults() -> None:
    args = roofline_profile.parse_args(
        [
            "--config-path",
            "examples/embodiment/config",
            "--config-name",
            "libero_goal_ppo_openpi_pi05",
            "--gpu",
            "0",
        ]
    )

    assert args.model_type == "openpi"
    assert args.batch_sizes == "1,8,32,128"
    assert args.warmup_steps == 3
    assert args.measure_steps == 10
    assert args.peak_tflops == 312.0
    assert args.peak_bw_gbs == 1555.0


def test_roofline_profile_build_cases_for_gr00t() -> None:
    args = roofline_profile.parse_args(
        [
            "--config-path",
            "examples/embodiment/config",
            "--config-name",
            "libero_spatial_ppo_gr00t",
            "--gpu",
            "2",
            "--model-type",
            "gr00t",
            "--batch-sizes",
            "8,1,8",
        ]
    )

    cases = roofline_profile.build_cases(args)

    assert [case.case_id for case in cases] == ["gr00t-bs1", "gr00t-bs8"]
    assert all(case.model_type == "gr00t" for case in cases)
    assert all(case.gpu == "2" for case in cases)


def test_roofline_profile_output_paths(tmp_path: Path) -> None:
    paths = roofline_profile.output_paths(tmp_path)

    assert paths.traces_dir == tmp_path / "traces"
    assert paths.out_dir == tmp_path / "out"
    assert paths.fig_dir == tmp_path / "figs"
    assert paths.aggregated_csv == tmp_path / "out" / "aggregated.csv"
    assert paths.figure_pdf == tmp_path / "figs" / "vla_roofline.pdf"
    assert paths.figure_svg == tmp_path / "figs" / "vla_roofline.svg"
