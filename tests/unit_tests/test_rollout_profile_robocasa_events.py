import json

from rlinf.envs.robocasa.venv import RobocasaSubprocEnv
from rlinf.envs.venv import BaseVectorEnv


def test_robocasa_child_step_timing_uses_rollout_profile_context(tmp_path):
    env = object.__new__(RobocasaSubprocEnv)
    env.is_closed = False
    env._sim_timestamp_context = {
        "output_dir": str(tmp_path / "legacy"),
        "rollout_profile_output_dir": str(tmp_path / "profile"),
        "rank": 2,
        "pid": 123,
        "epoch": 0,
        "chunk_step": 1,
        "stage": 0,
        "stage_num": 1,
        "local_envs": 1,
        "record_child_steps": True,
        "child_step_sample_interval": 1,
    }
    env._sim_timestamp_file = None
    env._sim_vector_step_index = 0
    env._sim_async_step_starts = {}
    env._last_chunk_profile = None
    env.env_num = 1

    env.record_robocasa_step_timing_events(
        [
            {
                "robocasa_step_timings": [
                    {
                        "local_env": 0,
                        "duration_s": 0.12,
                        "wall_start_ns": 10,
                        "wall_end_ns": 20,
                        "chunk_action_index": 0,
                        "repeat_index": 0,
                    }
                ]
            }
        ],
        vector_step=3,
    )

    profile_path = tmp_path / "profile" / "env_rank_2.jsonl"
    rows = [json.loads(line) for line in profile_path.read_text().splitlines()]
    assert rows[0]["event"] == "robocasa.child_step"
    assert rows[0]["rank"] == 2
    assert rows[0]["local_env"] == 0
    assert rows[0]["duration_s"] == 0.12


def test_robocasa_child_step_profile_respects_sample_interval(tmp_path):
    env = object.__new__(RobocasaSubprocEnv)
    env.is_closed = False
    env._sim_timestamp_context = {
        "output_dir": str(tmp_path / "legacy"),
        "rollout_profile_output_dir": str(tmp_path / "profile"),
        "rank": 2,
        "pid": 123,
        "epoch": 0,
        "chunk_step": 1,
        "stage": 0,
        "stage_num": 1,
        "local_envs": 1,
        "record_child_steps": True,
        "child_step_sample_interval": 4,
    }
    env._sim_timestamp_file = None
    env._sim_vector_step_index = 0
    env._sim_async_step_starts = {}
    env._last_chunk_profile = None
    env.env_num = 1

    env.record_robocasa_step_timing_events(
        [
            {
                "robocasa_step_timings": [
                    {
                        "local_env": 0,
                        "duration_s": 0.12,
                        "wall_start_ns": 10,
                        "wall_end_ns": 20,
                    }
                ]
            }
        ],
        vector_step=3,
    )

    assert not (tmp_path / "profile" / "env_rank_2.jsonl").exists()


def test_profile_only_subenv_context_does_not_require_legacy_output_dir(tmp_path):
    env = object.__new__(BaseVectorEnv)
    env._sim_timestamp_context = {
        "rollout_profile_output_dir": str(tmp_path / "profile"),
        "rank": 2,
        "pid": 123,
    }
    env._sim_timestamp_file = None

    assert env._get_sim_timestamp_file() is None
