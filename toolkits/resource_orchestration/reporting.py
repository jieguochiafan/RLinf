from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from toolkits.resource_orchestration.types import CandidateEstimate, SelectionResult


def write_reports(
    output_dir: str | Path,
    estimates: tuple[CandidateEstimate, ...],
    selection: SelectionResult | None,
    plan_output: str,
    failed_profiles: list[str] | list[dict[str, str]],
) -> dict[str, Path]:
    """Write resource orchestration profile and summary reports."""
    output_path = Path(output_dir)
    profiles_dir = output_path / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    candidate_records = [_candidate_record(estimate) for estimate in estimates]
    for record in candidate_records:
        profile_path = profiles_dir / f"{record['candidate_id']}.json"
        _write_json(profile_path, record)

    ranked = selection.ranked if selection is not None else ()
    selected_record = (
        _candidate_record(selection.selected) if selection is not None else None
    )
    summary = {
        "selected": selected_record,
        "ranked_candidate_ids": [
            estimate.candidate.candidate_id for estimate in ranked
        ],
        "plan_output": plan_output,
        "failed_profiles": list(failed_profiles),
        "candidates": candidate_records,
    }

    summary_json = output_path / "summary.json"
    summary_md = output_path / "summary.md"
    _write_json(summary_json, summary)
    summary_md.write_text(_summary_markdown(summary), encoding="utf-8")

    return {
        "summary_json": summary_json,
        "summary_md": summary_md,
        "profiles_dir": profiles_dir,
    }


def _candidate_record(estimate: CandidateEstimate) -> dict:
    record = asdict(estimate)
    record["candidate_id"] = estimate.candidate.candidate_id
    return record


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _summary_markdown(summary: dict) -> str:
    selected = summary["selected"]
    selected_id = selected["candidate_id"] if selected is not None else "none"
    lines = [
        "# Resource orchestration summary",
        "",
        f"Selected candidate: {selected_id}",
        f"Plan output: {summary['plan_output']}",
        "",
        "| candidate | epoch_s | rollout_s | training_s | bottleneck | env_chunk_steps_per_sec | model_chunk_steps_per_sec | actor_chunk_steps_per_sec |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]

    for candidate in summary["candidates"]:
        throughput = candidate["throughput"]
        lines.append(
            "| "
            f"{candidate['candidate_id']} | "
            f"{candidate['epoch_time_s']} | "
            f"{candidate['rollout_time_s']} | "
            f"{candidate['training_time_s']} | "
            f"{candidate['bottleneck_stage']} | "
            f"{throughput['env_chunk_steps_per_sec']} | "
            f"{throughput['model_chunk_steps_per_sec']} | "
            f"{throughput['actor_chunk_steps_per_sec']} |"
        )
    lines.append("")
    return "\n".join(lines)
