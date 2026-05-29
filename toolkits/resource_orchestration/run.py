from __future__ import annotations

import argparse

from toolkits.resource_orchestration.candidates import parse_candidate_pairs
from toolkits.resource_orchestration.config_loader import (
    build_config_summary,
    load_base_bindings,
    load_hydra_config,
)
from toolkits.resource_orchestration.orchestrator import run_orchestration
from toolkits.resource_orchestration.profilers import (
    ToolkitThroughputProfiler,
    default_profile_functions,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse resource orchestration CLI arguments."""
    parser = argparse.ArgumentParser(description="Run resource orchestration.")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--plan-output", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--candidate-pairs", default=None)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=20)
    parser.add_argument("--base-plan", default=None)
    parser.add_argument("--selection-tolerance", type=float, default=0.03)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run resource orchestration from CLI inputs."""
    args = parse_args(argv)
    cfg = load_hydra_config(
        args.config_path,
        args.config_name,
        overrides=tuple(args.override),
    )
    summary = build_config_summary(cfg)
    candidates = parse_candidate_pairs(args.candidate_pairs)
    base_bindings = load_base_bindings(cfg=cfg, base_plan=args.base_plan)
    profiler = ToolkitThroughputProfiler(
        cfg=cfg,
        summary=summary,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        functions=default_profile_functions(),
    )
    run_orchestration(
        config_summary=summary,
        base_bindings=base_bindings,
        candidates=candidates,
        profiler=profiler,
        output_dir=args.output_dir,
        plan_output=args.plan_output,
        selection_tolerance=args.selection_tolerance,
    )


if __name__ == "__main__":
    main()
