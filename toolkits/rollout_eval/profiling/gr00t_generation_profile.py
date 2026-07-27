"""Profile GR00T generation phases with CUDA events, NVTX, and PyTorch.

The benchmark follows the rollout inference path while exposing phase boundaries
that are otherwise hidden inside ``predict_action_batch``.  It is intentionally
model-only: random LIBERO-shaped observations remove environment timing noise
while preserving the rollout batch size and model execution shapes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from statistics import mean
from typing import Any

PHASE_ORDER = (
    "generation",
    "preprocess",
    "prepare_input",
    "vlm_backbone",
    "action_head",
    "action_head.state_encoder",
    "action_head.denoise_0",
    "action_head.denoise_1",
    "action_head.denoise_2",
    "action_head.denoise_3",
    "action_head.value",
    "postprocess",
)


def _summary(values: list[float]) -> dict[str, float]:
    """Return stable latency summary statistics."""
    if not values:
        return {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        rank = (len(ordered) - 1) * fraction
        lower = int(rank)
        upper = min(lower + 1, len(ordered) - 1)
        weight = rank - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "avg_ms": float(mean(ordered)),
        "p50_ms": float(percentile(0.50)),
        "p95_ms": float(percentile(0.95)),
    }


class PhaseRecorder:
    """Record nested phase wall time and CUDA-stream span.

    CUDA event spans include launch gaps between the two events.  They therefore
    measure stage latency on the CUDA stream, not the sum of kernel busy time.
    PyTorch Profiler or Nsight Systems should be used for kernel busy time.
    """

    def __init__(self) -> None:
        import torch

        self._torch = torch
        self._wall_samples: dict[str, list[float]] = defaultdict(list)
        self._cuda_event_samples: dict[str, list[tuple[Any, Any]]] = defaultdict(list)
        self._denoise_index = 0
        self._record_samples = False

    def begin_iteration(self, *, record_samples: bool) -> None:
        self._denoise_index = 0
        self._record_samples = record_samples

    def next_denoise_phase(self) -> str:
        phase = f"action_head.denoise_{self._denoise_index}"
        self._denoise_index += 1
        return phase

    @contextmanager
    def phase(self, name: str, *, cuda_span: bool) -> Any:
        torch = self._torch
        start_event = None
        end_event = None
        if cuda_span:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        wall_start = time.perf_counter()
        torch.cuda.nvtx.range_push(f"gr00t.{name}")
        try:
            with torch.profiler.record_function(f"gr00t.{name}"):
                yield
        finally:
            torch.cuda.nvtx.range_pop()
            wall_ms = (time.perf_counter() - wall_start) * 1000.0
            if end_event is not None:
                end_event.record()
            if self._record_samples:
                self._wall_samples[name].append(wall_ms)
                if start_event is not None and end_event is not None:
                    self._cuda_event_samples[name].append((start_event, end_event))

    def summaries(self) -> dict[str, dict[str, dict[str, float]]]:
        torch = self._torch
        torch.cuda.synchronize()
        result: dict[str, dict[str, dict[str, float]]] = {}
        phase_names = set(self._wall_samples) | set(self._cuda_event_samples)
        for phase in sorted(phase_names):
            cuda_values = [
                float(start.elapsed_time(end))
                for start, end in self._cuda_event_samples.get(phase, [])
            ]
            result[phase] = {
                "wall": _summary(self._wall_samples.get(phase, [])),
                "cuda_stream_span": _summary(cuda_values),
            }
        return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the GR00T stage profiler CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Profile GR00T VLM/action-head generation phases with representative "
            "LIBERO inputs"
        )
    )
    parser.add_argument(
        "--config-path",
        default="examples/embodiment/config",
        help="Hydra config directory",
    )
    parser.add_argument(
        "--config-name",
        default="libero_object_ppo_gr00t",
        help="Hydra config name",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override, repeatable",
    )
    parser.add_argument("--gpu", type=int, default=0, help="Physical GPU id")
    parser.add_argument(
        "--mps-sm",
        type=int,
        default=80,
        help="CUDA_MPS_ACTIVE_THREAD_PERCENTAGE for the worker",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=12,
        help="Per-stage rollout batch size; the 192-env/8-GPU/2-stage run uses 12",
    )
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument(
        "--task-description",
        default="pick up the object and place it in the target container",
    )
    parser.add_argument(
        "--output-dir",
        default="./rollout_eval_output/gr00t_generation_profile",
    )
    parser.add_argument(
        "--torch-profiler",
        action="store_true",
        help="Export a PyTorch CPU/CUDA trace and operator table",
    )
    parser.add_argument(
        "--nsys",
        action="store_true",
        help="Run the measured region under Nsight Systems CUDA/NVTX/GPU metrics",
    )
    parser.add_argument(
        "--gpu-metrics-frequency",
        type=int,
        default=1000,
        help="Nsight Systems GPU metric sampling rate",
    )
    parser.add_argument(
        "--no-manage-mps",
        action="store_false",
        dest="manage_mps",
        help="Use an already-running MPS daemon instead of an isolated daemon",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--cuda-profiler-capture", action="store_true", help=argparse.SUPPRESS
    )
    parser.set_defaults(manage_mps=True)
    args = parser.parse_args(argv)
    if not 1 <= args.mps_sm <= 100:
        parser.error("--mps-sm must be in [1, 100]")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.warmup_steps < 0 or args.measure_steps <= 0:
        parser.error("warmup must be non-negative and measure steps must be positive")
    if args.nsys and args.torch_profiler:
        parser.error("--nsys and --torch-profiler cannot be combined (both use CUPTI)")
    return args


def _load_cfg(args: argparse.Namespace) -> Any:
    from hydra import compose, initialize_config_dir

    config_dir = str(Path(args.config_path).resolve())
    with initialize_config_dir(version_base="1.1", config_dir=config_dir):
        return compose(config_name=args.config_name, overrides=list(args.override))


def _build_model(cfg: Any) -> Any:
    import copy

    from omegaconf import open_dict

    from rlinf.models import get_model

    model_cfg = copy.deepcopy(cfg.actor.model)
    with open_dict(model_cfg):
        if "rollout" in cfg and "model" in cfg.rollout:
            if "precision" in cfg.rollout.model:
                model_cfg.precision = cfg.rollout.model.precision
            if "model_path" in cfg.rollout.model:
                model_cfg.model_path = cfg.rollout.model.model_path
    model = get_model(model_cfg)
    # GR00T overrides ``train``/``eval`` without returning ``self``, unlike the
    # default ``torch.nn.Module`` implementation, so these calls cannot chain.
    model.eval()
    model.cuda()
    if hasattr(model, "set_global_step"):
        model.set_global_step(0)
    return model


def _make_env_obs(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(1234)
    image_shape = (
        int(args.batch_size),
        int(args.image_size),
        int(args.image_size),
        3,
    )
    return {
        "main_images": torch.randint(
            0, 256, image_shape, dtype=torch.uint8, generator=generator
        ),
        "wrist_images": torch.randint(
            0, 256, image_shape, dtype=torch.uint8, generator=generator
        ),
        "states": torch.rand(
            int(args.batch_size),
            int(args.state_dim),
            dtype=torch.float32,
            generator=generator,
        ),
        "task_descriptions": [str(args.task_description)] * int(args.batch_size),
    }


def _install_action_head_ranges(model: Any, recorder: PhaseRecorder) -> None:
    """Wrap action-head subroutines without changing their computation."""
    action_head = model.action_head

    original_state_forward = action_head.state_encoder.forward

    def state_forward(*args: Any, **kwargs: Any) -> Any:
        with recorder.phase("action_head.state_encoder", cuda_span=True):
            return original_state_forward(*args, **kwargs)

    action_head.state_encoder.forward = state_forward

    original_sample = action_head.sample_mean_var_val

    def sample_mean_var_val(*args: Any, **kwargs: Any) -> Any:
        phase = recorder.next_denoise_phase()
        with recorder.phase(phase, cuda_span=True):
            return original_sample(*args, **kwargs)

    action_head.sample_mean_var_val = sample_mean_var_val

    original_get_value = action_head.get_value

    def get_value(*args: Any, **kwargs: Any) -> Any:
        with recorder.phase("action_head.value", cuda_span=True):
            return original_get_value(*args, **kwargs)

    action_head.get_value = get_value


def _profile_generation_once(
    model: Any,
    env_obs: dict[str, Any],
    recorder: PhaseRecorder,
) -> tuple[Any, dict[str, Any]]:
    """Run the production GR00T inference path with explicit phase boundaries."""
    import numpy as np
    import torch

    from rlinf.models.embodiment.gr00t.utils import (
        squeeze_dict_values,
        unsqueeze_dict_values,
    )

    with recorder.phase("generation", cuda_span=True):
        with recorder.phase("preprocess", cuda_span=False):
            env_obs["states"] = env_obs["states"].to(torch.bfloat16)
            env_obs["states"] = env_obs["states"].cpu().float()
            observations = model.obs_convert_fn(env_obs)
            obs_copy = observations.copy()
            is_batch = model._check_state_is_batched(obs_copy)
            if not is_batch:
                obs_copy = unsqueeze_dict_values(obs_copy)
            for key, value in obs_copy.items():
                if not isinstance(value, np.ndarray):
                    obs_copy[key] = np.array(value)
            normalized_input = model.apply_transforms(obs_copy)
            for key in normalized_input:
                if normalized_input[key].dtype == torch.float32:
                    normalized_input[key] = normalized_input[key].to(torch.bfloat16)
            normalized_input["eagle_input_ids"] = torch.nn.functional.pad(
                normalized_input["eagle_input_ids"],
                pad=(
                    0,
                    model.padding_value - normalized_input["eagle_input_ids"].shape[-1],
                ),
                mode="constant",
                value=0,
            )
            normalized_input["eagle_attention_mask"] = torch.nn.functional.pad(
                normalized_input["eagle_attention_mask"],
                pad=(
                    0,
                    model.padding_value
                    - normalized_input["eagle_attention_mask"].shape[-1],
                ),
                mode="constant",
                value=0,
            )

        with recorder.phase("prepare_input", cuda_span=True):
            backbone_inputs, action_inputs = model.prepare_input(normalized_input)

        with recorder.phase("vlm_backbone", cuda_span=True):
            backbone_outputs = model.backbone(backbone_inputs)

        with recorder.phase("action_head", cuda_span=True):
            action_head_outputs, rlinf_outputs = model.action_head.get_rl_action(
                backbone_outputs, action_inputs, mode="train"
            )

        model.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        normalized_action = rlinf_outputs["actions"].float()
        forward_inputs = {
            "chains": rlinf_outputs["chains"],
            "denoise_inds": rlinf_outputs["denoise_inds"],
            **normalized_input,
        }
        batch_size = normalized_input["state"].shape[0]
        forward_inputs["eagle_pixel_values"] = normalized_input[
            "eagle_pixel_values"
        ].reshape(
            batch_size,
            model.image_nums,
            *normalized_input["eagle_pixel_values"].shape[1:],
        )
        forward_inputs["eagle_image_sizes"] = normalized_input[
            "eagle_image_sizes"
        ].reshape(
            batch_size,
            model.image_nums,
            *normalized_input["eagle_image_sizes"].shape[1:],
        )
        result = {
            "prev_logprobs": rlinf_outputs["prev_logprobs"],
            "prev_values": rlinf_outputs["prev_values"],
            "forward_inputs": forward_inputs,
        }

        with recorder.phase("postprocess", cuda_span=False):
            unnormalized_action = model._get_unnormalized_action(normalized_action)
            if not is_batch:
                unnormalized_action = squeeze_dict_values(unnormalized_action)
            raw_action = model.action_convert_fn(
                unnormalized_action, chunk_size=model.output_action_chunks
            )
            actions = torch.from_numpy(raw_action)
    return actions, result


def _write_torch_profile(profiler: Any, output_dir: Path) -> dict[str, Any]:
    trace_path = output_dir / "torch_trace.json"
    operator_path = output_dir / "torch_operators.txt"
    profiler.export_chrome_trace(str(trace_path))
    events = profiler.key_averages()
    operator_path.write_text(
        events.table(sort_by="self_cuda_time_total", row_limit=80),
        encoding="utf-8",
    )

    phases: dict[str, dict[str, float]] = {}
    for event in events:
        key = str(getattr(event, "key", ""))
        if not key.startswith("gr00t."):
            continue
        phases[key] = {
            "cpu_time_total_us": float(getattr(event, "cpu_time_total", 0.0)),
            "self_cpu_time_total_us": float(getattr(event, "self_cpu_time_total", 0.0)),
            "cuda_time_total_us": float(
                getattr(
                    event,
                    "cuda_time_total",
                    getattr(event, "device_time_total", 0.0),
                )
            ),
            "self_cuda_time_total_us": float(
                getattr(
                    event,
                    "self_cuda_time_total",
                    getattr(event, "self_device_time_total", 0.0),
                )
            ),
            "count": int(getattr(event, "count", 0)),
        }
    return {
        "trace": str(trace_path),
        "operator_table": str(operator_path),
        "phase_events": phases,
    }


def _run_worker(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        import torch
        from torch.profiler import ProfilerActivity, profile

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.cuda.set_device(0)
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)

        cfg = _load_cfg(args)
        model = _build_model(cfg)
        env_obs = _make_env_obs(args)
        recorder = PhaseRecorder()
        _install_action_head_ranges(model, recorder)

        with torch.no_grad():
            for _ in range(int(args.warmup_steps)):
                recorder.begin_iteration(record_samples=False)
                _profile_generation_once(model, env_obs, recorder)
            torch.cuda.synchronize()

            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
            profiler = (
                profile(
                    activities=activities,
                    record_shapes=True,
                    profile_memory=False,
                    with_stack=False,
                )
                if args.torch_profiler
                else None
            )

            if args.cuda_profiler_capture:
                torch.cuda.profiler.start()
            try:
                profile_context = profiler if profiler is not None else _NullContext()
                with profile_context:
                    for _ in range(int(args.measure_steps)):
                        recorder.begin_iteration(record_samples=True)
                        _profile_generation_once(model, env_obs, recorder)
                        if profiler is not None:
                            profiler.step()
                torch.cuda.synchronize()
            finally:
                if args.cuda_profiler_capture:
                    torch.cuda.profiler.stop()

        profile_output = (
            _write_torch_profile(profiler, output_dir) if profiler is not None else None
        )
        model_config = model.config.to_dict() if hasattr(model, "config") else {}
        payload = {
            "status": "pass",
            "gpu": int(args.gpu),
            "mps_sm": int(args.mps_sm),
            "batch_size": int(args.batch_size),
            "warmup_steps": int(args.warmup_steps),
            "measure_steps": int(args.measure_steps),
            "elapsed_s": time.perf_counter() - started,
            "environment": {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": os.environ.get(
                    "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
                ),
                "device_name": torch.cuda.get_device_name(0),
                "reported_sm_count": torch.cuda.get_device_properties(
                    0
                ).multi_processor_count,
            },
            "model": {
                "model_path": str(cfg.rollout.model.model_path),
                "num_inference_timesteps": int(
                    model.action_head.num_inference_timesteps
                ),
                "action_horizon": int(model.action_head.config.action_horizon),
                "hidden_size": int(model.action_head.hidden_size),
                "config_model_type": model_config.get("model_type"),
            },
            "phase_latency": recorder.summaries(),
            "torch_profiler": profile_output,
            "notes": {
                "cuda_stream_span": (
                    "Elapsed time between CUDA events; includes launch gaps inside "
                    "a phase and is not kernel busy time."
                ),
                "kernel_busy_time": (
                    "Use torch_profiler phase_events or the Nsight Systems report."
                ),
            },
        }
        (output_dir / "summary.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _write_markdown_summary(output_dir / "summary.md", payload)
    except Exception as exc:  # noqa: BLE001
        payload = {
            "status": "failed",
            "error_message": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }
        (output_dir / "summary.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        raise


def _write_markdown_summary(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# GR00T Generation Profile",
        "",
        f"- GPU: {payload['environment']['device_name']}",
        f"- MPS active thread percentage: {payload['mps_sm']}",
        f"- Batch size: {payload['batch_size']}",
        f"- Measured iterations: {payload['measure_steps']}",
        "",
        "| phase | wall avg (ms) | CUDA stream span avg (ms) |",
        "| --- | ---: | ---: |",
    ]
    phase_latency = payload["phase_latency"]
    for phase in PHASE_ORDER:
        if phase not in phase_latency:
            continue
        metrics = phase_latency[phase]
        lines.append(
            f"| {phase} | {metrics['wall']['avg_ms']:.3f} | "
            f"{metrics['cuda_stream_span']['avg_ms']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False


def _resolve_gpu_uuid(gpu: int) -> str:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=uuid",
            "--format=csv,noheader",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    uuid = completed.stdout.strip().splitlines()[0].strip()
    if not uuid:
        raise RuntimeError(f"Could not resolve UUID for GPU {gpu}")
    return uuid


def _start_mps_daemon(cuda_visible_device: str, pipe_dir: Path, log_dir: Path) -> None:
    pipe_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": cuda_visible_device,
            "CUDA_MPS_PIPE_DIRECTORY": str(pipe_dir),
            "CUDA_MPS_LOG_DIRECTORY": str(log_dir),
        }
    )
    subprocess.run(["nvidia-cuda-mps-control", "-d"], env=env, check=True)


def _stop_mps_daemon(pipe_dir: Path, log_dir: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_MPS_PIPE_DIRECTORY": str(pipe_dir),
            "CUDA_MPS_LOG_DIRECTORY": str(log_dir),
        }
    )
    subprocess.run(
        ["nvidia-cuda-mps-control"],
        input="quit\n",
        text=True,
        env=env,
        check=False,
        capture_output=True,
    )


def _worker_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "toolkits.rollout_eval.profiling.gr00t_generation_profile",
        "--worker",
        "--config-path",
        str(args.config_path),
        "--config-name",
        str(args.config_name),
        "--gpu",
        str(args.gpu),
        "--mps-sm",
        str(args.mps_sm),
        "--batch-size",
        str(args.batch_size),
        "--warmup-steps",
        str(args.warmup_steps),
        "--measure-steps",
        str(args.measure_steps),
        "--image-size",
        str(args.image_size),
        "--state-dim",
        str(args.state_dim),
        "--task-description",
        str(args.task_description),
        "--output-dir",
        str(Path(args.output_dir).resolve()),
    ]
    if args.torch_profiler:
        cmd.append("--torch-profiler")
    if args.nsys:
        cmd.append("--cuda-profiler-capture")
    for override in args.override:
        cmd.extend(["--override", str(override)])
    return cmd


def _run_parent(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{os.getuid()}-{os.getpid()}"
    pipe_dir = Path(f"/tmp/rlinf-gr00t-profile-{run_id}/pipe")
    log_dir = Path(f"/tmp/rlinf-gr00t-profile-{run_id}/log")
    mps_started = False

    env = os.environ.copy()
    config_parent = str(Path(args.config_path).resolve().parent)
    env.setdefault("EMBODIED_PATH", config_parent)
    try:
        if args.manage_mps:
            visible_device = _resolve_gpu_uuid(int(args.gpu))
            _start_mps_daemon(visible_device, pipe_dir, log_dir)
            mps_started = True
            env["CUDA_VISIBLE_DEVICES"] = visible_device
            env["CUDA_MPS_PIPE_DIRECTORY"] = str(pipe_dir)
            env["CUDA_MPS_LOG_DIRECTORY"] = str(log_dir)
        else:
            env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(args.mps_sm)

        worker_cmd = _worker_command(args)
        if args.nsys:
            report_base = output_dir / "gr00t_generation"
            worker_cmd = [
                "nsys",
                "profile",
                "--force-overwrite=true",
                "--trace=cuda,nvtx,osrt",
                "--capture-range=cudaProfilerApi",
                "--capture-range-end=stop",
                f"--gpu-metrics-devices={args.gpu}",
                f"--gpu-metrics-frequency={args.gpu_metrics_frequency}",
                f"--output={report_base}",
                *worker_cmd,
            ]

        completed = subprocess.run(
            worker_cmd,
            cwd=Path.cwd(),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        (output_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        (output_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        return int(completed.returncode)
    finally:
        if mps_started:
            _stop_mps_daemon(pipe_dir, log_dir)


def main(argv: list[str] | None = None) -> None:
    """Run one isolated GR00T stage profile."""
    args = parse_args(argv)
    if args.worker:
        _run_worker(args)
        return
    raise SystemExit(_run_parent(args))


if __name__ == "__main__":
    main()
