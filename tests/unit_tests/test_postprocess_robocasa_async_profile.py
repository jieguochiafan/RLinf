import importlib.util
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "tools" / "postprocess_robocasa_async_profile.py"
SPEC = importlib.util.spec_from_file_location("postprocess_robocasa_async_profile", SCRIPT_PATH)
assert SPEC is not None
postprocess = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = postprocess
SPEC.loader.exec_module(postprocess)


def test_aggregate_cpu_gpu_accepts_spaced_nvidia_smi_headers(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "derived"
    run_dir.mkdir()
    (run_dir / "cpu_highres.csv").write_text(
        "\n".join(
            [
                "timestamp,time_s,elapsed_s,total_cores,total_util_pct,EnvWorker_cores,RolloutWorker_cores,ActorWorker_cores,CompileWorker_cores,Main_cores,raylet_cores,gcs_server_cores,Other_cores,tracked_items,other_items",
                "1000.0,0.0,1.0,2.0,1.8,1.0,0.5,0.25,0.0,0.0,0.0,0.0,0.25,3,0",
            ]
        )
        + "\n"
    )
    (run_dir / "gpu_highres.csv").write_text(
        "\n".join(
            [
                "timestamp, index, utilization.gpu [%], memory.used [MiB], power.draw [W]",
                f"{datetime.fromtimestamp(1000.0).strftime('%Y/%m/%d %H:%M:%S.%f')[:-3]}, 0, 25, 1024, 60",
            ]
        )
        + "\n"
    )

    postprocess.aggregate_cpu_gpu(
        run_dir,
        out_dir,
        t0=1000.0,
        bin_s=1.0,
        active_end_s=10.0,
        num_cpus=112.0,
    )

    output = out_dir / "active_cpu_gpu_roles.csv"
    text = output.read_text()
    assert "gpu_util_avg_pct" in text
    assert "25.000000" in text


def test_normalize_role_prefers_worker_cmdline_over_idle_comm() -> None:
    role = postprocess.normalize_role(
        "Other",
        "ray::IDLE",
        "ray::AsyncEnvWorker.init_worker",
    )

    assert role == "EnvWorker"
