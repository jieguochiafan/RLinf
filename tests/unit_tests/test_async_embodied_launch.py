from pathlib import Path


def test_train_async_selects_gipo_runner_for_gipo_loss():
    source = Path("examples/embodiment/train_async.py").read_text()

    assert "gipo_actor_critic" in source
    assert "AsyncGIPOEmbodiedRunner" in source
    assert "AsyncGIPOEmbodiedFSDPActor" in source


def test_async_gipo_example_config_contains_required_sections():
    path = Path("examples/embodiment/config/libero_spatial_async_gipo_openpi_pi05.yaml")
    text = path.read_text()

    assert "loss_type: gipo_actor_critic" in text
    assert "replay_buffer:" in text
    assert "gipo:" in text
    assert "target_batch_size:" in text
    assert "max_wait_time_s:" in text
