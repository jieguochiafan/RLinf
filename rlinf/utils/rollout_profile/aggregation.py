from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RolloutProfileSummaryOutputs:
    summary_path: Path
    timeline_path: Path
    report_path: Path


def _iter_records(profile_dir: Path):
    for path in sorted(profile_dir.glob("*_rank_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def summarize_rollout_profile(profile_dir: str | Path) -> RolloutProfileSummaryOutputs:
    profile_path = Path(profile_dir)
    profile_path.mkdir(parents=True, exist_ok=True)
    records = list(_iter_records(profile_path))
    event_counts = Counter(record["event"] for record in records)
    duration_s_by_event: dict[str, float] = defaultdict(float)
    metric_sums: dict[str, float] = defaultdict(float)
    wall_times = [int(record["wall_ns"]) for record in records if "wall_ns" in record]
    t0 = min(wall_times) if wall_times else 0

    timeline_rows = []
    for record in sorted(records, key=lambda item: int(item.get("wall_ns", 0))):
        event = str(record.get("event", ""))
        duration = record.get("duration_s")
        if isinstance(duration, int | float):
            duration_s_by_event[event] += float(duration)
        metrics = record.get("metrics")
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if isinstance(value, int | float):
                    metric_sums[f"{event}.{key}"] += float(value)
        wall_ns = int(record.get("wall_ns", t0))
        relative_start_s = (wall_ns - t0) / 1_000_000_000 if t0 else 0.0
        relative_end_s = (
            relative_start_s + float(duration)
            if isinstance(duration, int | float)
            else relative_start_s
        )
        timeline_rows.append(
            {
                "relative_start_s": f"{relative_start_s:.6f}",
                "relative_end_s": f"{relative_end_s:.6f}",
                "event": event,
                "component": record.get("component", ""),
                "rank": record.get("rank", ""),
                "pid": record.get("pid", ""),
                "epoch": record.get("epoch", ""),
                "chunk_step": record.get("chunk_step", ""),
                "duration_s": "" if duration is None else f"{float(duration):.6f}",
            }
        )

    summary = {
        "record_count": len(records),
        "event_counts": dict(sorted(event_counts.items())),
        "duration_s_by_event": {
            key: float(value) for key, value in sorted(duration_s_by_event.items())
        },
        "metric_sums": {key: float(value) for key, value in sorted(metric_sums.items())},
    }

    summary_path = profile_path / "summary.json"
    timeline_path = profile_path / "timeline.csv"
    report_path = profile_path / "bottleneck_report.md"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with timeline_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "relative_start_s",
            "relative_end_s",
            "event",
            "component",
            "rank",
            "pid",
            "epoch",
            "chunk_step",
            "duration_s",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(timeline_rows)

    top_durations = sorted(
        duration_s_by_event.items(), key=lambda item: item[1], reverse=True
    )[:10]
    report_lines = ["# Rollout Profile Bottleneck Report", "", "## Top Durations", ""]
    for event, duration_s in top_durations:
        report_lines.append(f"- `{event}`: {duration_s:.6f}s")
    report_path.write_text("\n".join(report_lines) + "\n")
    return RolloutProfileSummaryOutputs(summary_path, timeline_path, report_path)
