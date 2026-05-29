from omegaconf import OmegaConf

from toolkits.resource_orchestration.config_loader import build_config_summary


def test_build_config_summary_extracts_rollout_chunk_count() -> None:
    cfg = OmegaConf.create(
        {
            "cluster": {
                "component_placement": {"actor": "0-1", "rollout": "0-1", "env": "0-1"},
                "resource_pool": {
                    "enabled": True,
                    "gpu": {"enabled": True, "mode": "mps"},
                },
            },
            "env": {
                "train": {
                    "total_num_envs": 8,
                    "max_steps_per_rollout_epoch": 40,
                    "max_episode_steps": 40,
                }
            },
            "actor": {
                "global_batch_size": 16,
                "micro_batch_size": 4,
                "model": {"num_action_chunks": 5},
            },
            "algorithm": {"rollout_epoch": 2, "update_epoch": 4},
            "rollout": {"pipeline_stage_num": 2},
        }
    )

    summary = build_config_summary(cfg)

    assert summary.total_num_envs == 8
    assert summary.chunk_size == 5
    assert summary.chunk_steps_per_env == 8
    assert summary.rollout_chunk_count == 128
    assert summary.actor_global_batch_size == 16
    assert summary.update_epoch == 4
    assert summary.resource_pool_mode == "mps"


def test_build_config_summary_rejects_non_mps_resource_pool() -> None:
    cfg = OmegaConf.create(
        {
            "cluster": {
                "resource_pool": {
                    "enabled": True,
                    "gpu": {"enabled": True, "mode": "mig"},
                }
            },
            "env": {"train": {"total_num_envs": 1, "max_steps_per_rollout_epoch": 5}},
            "actor": {
                "global_batch_size": 1,
                "micro_batch_size": 1,
                "model": {"num_action_chunks": 5},
            },
            "algorithm": {"rollout_epoch": 1, "update_epoch": 1},
            "rollout": {"pipeline_stage_num": 1},
        }
    )

    try:
        build_config_summary(cfg)
    except ValueError as exc:
        assert "MPS" in str(exc)
    else:
        raise AssertionError("expected non-MPS config to be rejected")
