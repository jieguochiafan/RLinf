from __future__ import annotations

from pathlib import Path

from toolkits.resource_orchestration.estimator import estimate_candidate
from toolkits.resource_orchestration.plan_writer import (
    ComponentBindings,
    write_mps_plan,
)
from toolkits.resource_orchestration.profilers import ThroughputProfiler
from toolkits.resource_orchestration.reporting import write_reports
from toolkits.resource_orchestration.selector import select_best_candidate
from toolkits.resource_orchestration.types import (
    CandidateEstimate,
    CandidatePair,
    ConfigSummary,
    SelectionResult,
)


def run_orchestration(
    config_summary: ConfigSummary,
    base_bindings: ComponentBindings,
    candidates: list[CandidatePair] | tuple[CandidatePair, ...],
    profiler: ThroughputProfiler,
    output_dir: str | Path,
    plan_output: str | Path,
    selection_tolerance: float = 0.03,
) -> SelectionResult:
    """Profile, estimate, select, and write a resource orchestration plan."""
    estimates: list[CandidateEstimate] = []
    failed_profiles: list[dict[str, str]] = []

    for candidate in candidates:
        try:
            throughput = profiler.profile(candidate)
            estimates.append(estimate_candidate(candidate, config_summary, throughput))
        except Exception as exc:
            failed_profiles.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    plan_output_str = str(plan_output)
    if not estimates:
        write_reports(
            output_dir,
            estimates=(),
            selection=None,
            plan_output=plan_output_str,
            failed_profiles=failed_profiles,
        )
        raise RuntimeError("all resource orchestration candidates failed")

    selection = select_best_candidate(estimates, tolerance=selection_tolerance)
    write_mps_plan(
        output_path=plan_output,
        base_bindings=base_bindings,
        candidate=selection.selected.candidate,
    )
    write_reports(
        output_dir,
        estimates=tuple(estimates),
        selection=selection,
        plan_output=plan_output_str,
        failed_profiles=failed_profiles,
    )
    return selection
