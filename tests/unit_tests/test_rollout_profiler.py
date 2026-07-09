import json
import time

from omegaconf import OmegaConf

from rlinf.utils.rollout_profile import (
    NoopRolloutProfiler,
    RolloutProfilerConfig,
    make_rollout_profiler,
)


def test_rollout_profiler_config_defaults_to_disabled(tmp_path):
    cfg = OmegaConf.create({"runner": {"logger": {"log_path": str(tmp_path)}}})

    profile_cfg = RolloutProfilerConfig.from_cfg(cfg, component="env", rank=2)

    assert profile_cfg.enabled is False
    assert profile_cfg.component == "env"
    assert profile_cfg.rank == 2
    assert profile_cfg.output_dir == str(tmp_path / "rollout_profile")
    assert profile_cfg.record_child_steps is True
    assert profile_cfg.child_step_sample_interval == 1


def test_make_rollout_profiler_returns_noop_when_disabled(tmp_path):
    cfg = OmegaConf.create({"runner": {"logger": {"log_path": str(tmp_path)}}})

    profiler = make_rollout_profiler(cfg, component="env", rank=0)

    assert isinstance(profiler, NoopRolloutProfiler)
    with profiler.span("env.recv_rollout_results", epoch=0):
        time.sleep(0)
    profiler.event("env.event", value=1)
    profiler.flush()
    assert not (tmp_path / "rollout_profile").exists()


def test_rollout_profiler_writes_event_and_span(tmp_path):
    cfg = OmegaConf.create(
        {
            "runner": {"logger": {"log_path": str(tmp_path)}},
            "profiling": {
                "rollout": {
                    "enabled": True,
                    "output_dir": str(tmp_path / "profile"),
                }
            },
        }
    )

    profiler = make_rollout_profiler(cfg, component="rollout", rank=3)
    profiler.event("rollout.recv_env_output", mode="train", chunk_step=0)
    with profiler.span("rollout.predict", mode="train", batch_size=8):
        time.sleep(0)
    profiler.flush()

    path = tmp_path / "profile" / "rollout_rank_3.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["event"] for row in rows] == [
        "rollout.recv_env_output",
        "rollout.predict.start",
        "rollout.predict.end",
    ]
    assert rows[0]["component"] == "rollout"
    assert rows[0]["rank"] == 3
    assert rows[0]["mode"] == "train"
    assert isinstance(rows[0]["wall_ns"], int)
    assert rows[2]["duration_s"] >= 0.0


def test_rollout_profiler_env_override_enables_profile(tmp_path, monkeypatch):
    cfg = OmegaConf.create({"runner": {"logger": {"log_path": str(tmp_path)}}})
    monkeypatch.setenv("RLINF_ROLLOUT_PROFILE", "1")

    profiler = make_rollout_profiler(cfg, component="env", rank=1)
    profiler.event("env.enabled_by_env")
    profiler.flush()

    assert (tmp_path / "rollout_profile" / "env_rank_1.jsonl").exists()
