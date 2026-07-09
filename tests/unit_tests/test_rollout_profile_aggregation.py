import csv
import json

from rlinf.utils.rollout_profile import summarize_rollout_profile


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_summarize_rollout_profile_writes_summary_timeline_and_report(tmp_path):
    profile_dir = tmp_path / "profile"
    _write_jsonl(
        profile_dir / "env_rank_0.jsonl",
        [
            {
                "event": "env.recv_rollout_results.start",
                "component": "env",
                "rank": 0,
                "pid": 10,
                "wall_ns": 1_000_000_000,
                "epoch": 0,
                "chunk_step": 0,
            },
            {
                "event": "env.recv_rollout_results.end",
                "component": "env",
                "rank": 0,
                "pid": 10,
                "wall_ns": 1_200_000_000,
                "duration_s": 0.2,
                "epoch": 0,
                "chunk_step": 0,
            },
            {
                "event": "env.chunk_profile",
                "component": "env",
                "rank": 0,
                "pid": 10,
                "wall_ns": 1_300_000_000,
                "metrics": {"wait_recv_s": 0.5, "stack_s": 0.1},
            },
        ],
    )
    _write_jsonl(
        profile_dir / "rollout_rank_0.jsonl",
        [
            {
                "event": "rollout.predict.end",
                "component": "rollout",
                "rank": 0,
                "pid": 20,
                "wall_ns": 1_500_000_000,
                "duration_s": 0.3,
            }
        ],
    )

    outputs = summarize_rollout_profile(profile_dir)

    assert outputs.summary_path.exists()
    assert outputs.timeline_path.exists()
    assert outputs.report_path.exists()
    summary = json.loads(outputs.summary_path.read_text())
    assert summary["event_counts"]["env.recv_rollout_results.end"] == 1
    assert summary["duration_s_by_event"]["rollout.predict.end"] == 0.3
    assert summary["metric_sums"]["env.chunk_profile.wait_recv_s"] == 0.5

    with outputs.timeline_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["event"] == "env.recv_rollout_results.start"
    assert rows[0]["relative_start_s"] == "0.000000"
    assert "rollout.predict.end" in outputs.report_path.read_text()
