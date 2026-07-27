from pathlib import Path


def test_profile_wrapper_shifts_output_dir_before_hydra_overrides() -> None:
    script = Path("tools/run_robocasa_async_pipeline2_profile.sh").read_text()

    assert "shift" in script.split('CPU_INTERVAL=${CPU_INTERVAL:-0.1}', maxsplit=1)[0]
