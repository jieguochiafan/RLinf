"""OpenPI stage-level benchmark under single-GPU MPS SM quotas.

The parent process optionally starts one MPS control daemon for one physical
GPU, then runs one worker process per MPS percentage and batch size.  The worker
loads OpenPI once per case and reports CUDA-event latency for:

* ``vlm``: prefix/VLM cache construction via ``_build_prefix_cache``.
* ``action_head``: denoising steps via ``sample_mean_var_val``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class StageProfileCase:
    """One OpenPI stage benchmark case."""

    case_id: str
    mps_sm: int
    num_envs: int
    gpu: int
    cuda_visible_device: str | None = None


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_int_csv(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in _parse_csv(value))


def _validate_mps_percentage(value: int) -> int:
    if value < 1 or value > 100:
        raise ValueError(f"MPS SM percentage must be in [1, 100], got {value}")
    return value


def build_stage_cases(args: argparse.Namespace) -> list[StageProfileCase]:
    """Build the deterministic MPS percentage and batch-size matrix."""
    cases: list[StageProfileCase] = []
    cuda_visible_device = getattr(args, "cuda_visible_device", None)
    for mps_sm in sorted(set(_parse_int_csv(args.mps_sm))):
        _validate_mps_percentage(mps_sm)
        for num_envs in sorted(set(_parse_int_csv(args.num_envs_list))):
            if num_envs <= 0:
                raise ValueError("--num-envs-list values must be positive")
            cases.append(
                StageProfileCase(
                    case_id=f"mps-sm{mps_sm}-bs{num_envs}",
                    mps_sm=mps_sm,
                    num_envs=num_envs,
                    gpu=int(args.gpu),
                    cuda_visible_device=cuda_visible_device,
                )
            )
    return cases


def build_worker_env(
    *,
    base_env: dict[str, str],
    case: StageProfileCase,
    mps_pipe_dir: str,
    mps_log_dir: str,
) -> dict[str, str]:
    """Build child env for one physical GPU and one MPS SM percentage."""
    env = dict(base_env)
    env["CUDA_VISIBLE_DEVICES"] = str(case.cuda_visible_device or case.gpu)
    env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(
        _validate_mps_percentage(case.mps_sm)
    )
    env["CUDA_MPS_PIPE_DIRECTORY"] = str(mps_pipe_dir)
    env["CUDA_MPS_LOG_DIRECTORY"] = str(mps_log_dir)
    return env


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    ordered = sorted(values)

    def _percentile(pct: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        rank = (len(ordered) - 1) * pct
        lower = int(rank)
        upper = min(lower + 1, len(ordered) - 1)
        weight = rank - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "avg_ms": float(mean(ordered)),
        "p50_ms": float(_percentile(0.50)),
        "p95_ms": float(_percentile(0.95)),
    }


def _metric_avg(metrics: dict[str, Any] | None, key: str) -> float:
    if not metrics:
        return 0.0
    value = metrics.get(key) or {}
    return float(value.get("avg_ms", 0.0))


def write_summary(output_dir: Path, records: list[dict]) -> dict:
    """Write JSON and Markdown summaries for all stage profile cases."""
    counts = {
        "total": len(records),
        "pass": sum(1 for record in records if record.get("status") == "pass"),
        "failed": sum(1 for record in records if record.get("status") == "failed"),
    }
    summary = {"counts": counts, "cases": records}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# OpenPI Stage MPS Profile Summary",
        "",
        f"- Total: {counts['total']}",
        f"- Pass: {counts['pass']}",
        f"- Failed: {counts['failed']}",
        "",
        "| case_id | status | mps_sm | num_envs | total_ms(avg) | "
        "vlm_ms(avg) | action_head_ms(avg) | other_ms(avg) | gpu |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        metrics = record.get("metrics")
        lines.append(
            "| {case_id} | {status} | {mps_sm} | {num_envs} | {total:.3f} | "
            "{vlm:.3f} | {action_head:.3f} | {other:.3f} | {gpu} |".format(
                case_id=record.get("case_id", ""),
                status=record.get("status", ""),
                mps_sm=record.get("mps_sm", ""),
                num_envs=record.get("num_envs", ""),
                total=_metric_avg(metrics, "total"),
                vlm=_metric_avg(metrics, "vlm"),
                action_head=_metric_avg(metrics, "action_head"),
                other=_metric_avg(metrics, "other"),
                gpu=record.get("gpu", ""),
            )
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _load_existing_passed_case(output_dir: Path, case: StageProfileCase) -> dict | None:
    """Load a previous passing case report for resumable matrix runs."""
    report_path = output_dir / "cases" / case.case_id / "case_report.json"
    if not report_path.exists():
        return None

    record = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        record.get("status") != "pass"
        or int(record.get("mps_sm", -1)) != case.mps_sm
        or int(record.get("num_envs", -1)) != case.num_envs
        or int(record.get("gpu", -1)) != case.gpu
    ):
        return None
    return record


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse OpenPI stage MPS benchmark arguments."""
    parser = argparse.ArgumentParser(
        description="Profile OpenPI VLM and action-head CUDA time under MPS"
    )
    parser.add_argument("--config-path", required=True, help="Hydra config directory")
    parser.add_argument("--config-name", required=True, help="Hydra config name")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override, repeatable.",
    )
    parser.add_argument(
        "--output-dir",
        default="./rollout_eval_output/openpi_stage_mps_profile",
        help="Output directory for reports",
    )
    parser.add_argument("--gpu", type=int, default=0, help="Physical GPU id to use")
    parser.add_argument(
        "--mps-sm",
        default="20,40,60,80,100",
        help="Comma-separated CUDA_MPS_ACTIVE_THREAD_PERCENTAGE values",
    )
    parser.add_argument(
        "--num-envs-list",
        default="1,4,8,16,32",
        help="Comma-separated random observation batch sizes",
    )
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measure-steps", type=int, default=50)
    parser.add_argument(
        "--state-dim",
        type=int,
        default=None,
        help="Override random observation state dimension",
    )
    parser.add_argument(
        "--case-timeout-s",
        type=float,
        default=None,
        help="Optional timeout per case",
    )
    parser.add_argument(
        "--mps-pipe-dir",
        default=None,
        help="CUDA MPS pipe directory. Defaults to a run-specific /tmp path.",
    )
    parser.add_argument(
        "--mps-log-dir",
        default=None,
        help="CUDA MPS log directory. Defaults to a run-specific /tmp path.",
    )
    parser.add_argument(
        "--no-manage-mps",
        action="store_false",
        dest="manage_mps",
        help="Do not start/stop an MPS daemon; assume one is already running.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing passing case reports and run only missing/failed cases.",
    )
    parser.set_defaults(manage_mps=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--mps-sm-current", type=int, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument("--num-envs", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def _load_cfg(args: argparse.Namespace):
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict

    abs_config_path = str(Path(args.config_path).resolve())
    with initialize_config_dir(version_base="1.1", config_dir=abs_config_path):
        cfg = compose(config_name=args.config_name, overrides=list(args.override))
    with open_dict(cfg):
        cfg.env.eval.total_num_envs = int(args.num_envs)
        cfg.actor.model.model_type = "openpi"
        if "rollout" in cfg and "model" in cfg.rollout:
            cfg.rollout.model.model_type = "openpi"
    return cfg


def _resolve_model_path(cfg) -> str:
    if (
        "rollout" in cfg
        and "model" in cfg.rollout
        and "model_path" in cfg.rollout.model
    ):
        return str(cfg.rollout.model.model_path)
    return str(cfg.actor.model.model_path)


def _build_openpi_model(cfg):
    from omegaconf import open_dict

    from rlinf.models import get_model
    from toolkits.rollout_eval.adapters.model_adapter import (
        _validate_model_path_or_raise,
    )

    model_path = _resolve_model_path(cfg)
    _validate_model_path_or_raise(model_path, model_type="openpi")
    model_cfg = cfg.actor.model.copy()
    with open_dict(model_cfg):
        if "openpi_data" in cfg:
            model_cfg.openpi_data = cfg.openpi_data
        if "rollout" in cfg and "model" in cfg.rollout:
            if "precision" in cfg.rollout.model:
                model_cfg.precision = cfg.rollout.model.precision
            if "model_path" in cfg.rollout.model:
                model_cfg.model_path = cfg.rollout.model.model_path
    model = get_model(model_cfg)
    model.eval()
    return model


def _infer_state_dim(cfg, override: int | None) -> int:
    if override is not None:
        return int(override)
    actor_model = cfg.actor.model
    if "state_dim" in actor_model:
        return int(actor_model.state_dim)
    openpi_cfg = actor_model.get("openpi", {})
    config_name = str(openpi_cfg.get("config_name", ""))
    if "robocasa" in config_name:
        return 25
    if "calvin" in config_name:
        return 7
    return 8


def _make_random_env_obs(cfg, num_envs: int, state_dim_override: int | None):
    import torch

    state_dim = _infer_state_dim(cfg, state_dim_override)
    image_shape = (num_envs, 224, 224, 3)
    return {
        "main_images": torch.randint(0, 256, image_shape, dtype=torch.uint8),
        "wrist_images": torch.randint(0, 256, image_shape, dtype=torch.uint8),
        "extra_view_images": None,
        "states": torch.rand(num_envs, state_dim, dtype=torch.float32),
        "task_descriptions": ["do something"] * num_envs,
    }


def _prepare_model_inputs(model, env_obs: dict[str, Any]):
    from openpi.models import model as _model

    to_process_obs = model.obs_processor(env_obs)
    processed_obs = model.input_transform(to_process_obs, transpose=False)
    processed_obs = model.precision_processor(processed_obs)
    observation = _model.Observation.from_dict(processed_obs)
    return model._preprocess_observation(observation, train=False)


def _cuda_event_ms(start, end) -> float:
    return float(start.elapsed_time(end))


def _profile_once(model, prepared_inputs) -> dict[str, float]:
    import torch

    images, img_masks, lang_tokens, lang_masks, state = prepared_inputs
    bsize = int(state.shape[0])
    device = state.device
    actions_shape = (bsize, model.config.action_horizon, model.config.action_dim)

    total_start = torch.cuda.Event(enable_timing=True)
    total_end = torch.cuda.Event(enable_timing=True)
    vlm_start = torch.cuda.Event(enable_timing=True)
    vlm_end = torch.cuda.Event(enable_timing=True)
    action_events = []

    total_start.record()
    vlm_start.record()
    prefix_output, prefix_pad_masks, past_key_values = model._build_prefix_cache(
        images, img_masks, lang_tokens, lang_masks
    )
    vlm_end.record()

    if model.use_vlm_value:
        model.get_value_from_vlm(prefix_output)

    x_t = model.sample_noise(actions_shape, device)
    num_steps = int(model.config.num_steps)
    denoise_inds = torch.tensor([-1] * num_steps, device=device)[None].repeat(bsize, 1)
    for idx in range(num_steps):
        if idx == int(denoise_inds[0][idx]):
            sample_method = model.config.noise_method
        else:
            sample_method = "flow_ode"
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        x_t_mean, x_t_std, _value_t, _v_t = model.sample_mean_var_val(
            x_t,
            idx,
            state,
            prefix_pad_masks,
            past_key_values,
            sample_method,
            num_steps,
            compute_values=True,
        )
        end.record()
        action_events.append((start, end))
        x_t = x_t_mean + model.sample_noise(x_t.shape, device) * x_t_std

    total_end.record()
    torch.cuda.synchronize()

    vlm_ms = _cuda_event_ms(vlm_start, vlm_end)
    action_ms = sum(_cuda_event_ms(start, end) for start, end in action_events)
    total_ms = _cuda_event_ms(total_start, total_end)
    return {
        "total": total_ms,
        "vlm": vlm_ms,
        "action_head": action_ms,
        "other": max(total_ms - vlm_ms - action_ms, 0.0),
    }


def _worker_result_path(output_dir: Path, case_id: str) -> Path:
    return output_dir / "cases" / case_id / "result.json"


def _write_worker_result(output_dir: Path, case_id: str, payload: dict) -> None:
    path = _worker_result_path(output_dir, case_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _run_worker(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    output_dir = Path(args.output_dir)
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in the worker process")
        torch.cuda.set_device(0)

        cfg = _load_cfg(args)
        model = _build_openpi_model(cfg).cuda().eval()
        env_obs = _make_random_env_obs(cfg, int(args.num_envs), args.state_dim)
        prepared_inputs = _prepare_model_inputs(model, env_obs)

        total_steps = int(args.warmup_steps) + int(args.measure_steps)
        samples = {"total": [], "vlm": [], "action_head": [], "other": []}
        with torch.inference_mode():
            for step_idx in range(total_steps):
                result = _profile_once(model, prepared_inputs)
                if step_idx >= int(args.warmup_steps):
                    for key, value in result.items():
                        samples[key].append(value)

        payload = {
            "status": "pass",
            "case_id": str(args.case_id),
            "mps_sm": int(args.mps_sm_current),
            "num_envs": int(args.num_envs),
            "gpu": int(args.gpu),
            "warmup_steps": int(args.warmup_steps),
            "measure_steps": int(args.measure_steps),
            "elapsed_s": time.perf_counter() - started,
            "metrics": {key: _summary(values) for key, values in samples.items()},
            "model": {
                "config_name": str(cfg.actor.model.openpi.config_name),
                "model_path": _resolve_model_path(cfg),
                "num_steps": int(model.config.num_steps),
                "action_horizon": int(model.config.action_horizon),
                "action_dim": int(model.config.action_dim),
            },
        }
        _write_worker_result(output_dir, str(args.case_id), payload)
    except Exception as exc:  # noqa: BLE001
        _write_worker_result(
            output_dir,
            str(args.case_id),
            {
                "status": "failed",
                "case_id": str(args.case_id),
                "mps_sm": args.mps_sm_current,
                "num_envs": args.num_envs,
                "error_message": (
                    f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                ),
            },
        )
        raise


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
        raise RuntimeError(f"Failed to resolve UUID for GPU {gpu}")
    return uuid


def _start_mps_daemon(cuda_visible_device: str, pipe_dir: str, log_dir: str) -> None:
    Path(pipe_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": cuda_visible_device,
            "CUDA_MPS_PIPE_DIRECTORY": pipe_dir,
            "CUDA_MPS_LOG_DIRECTORY": log_dir,
        }
    )
    subprocess.run(["nvidia-cuda-mps-control", "-d"], env=env, check=True)


def _stop_mps_daemon(pipe_dir: str, log_dir: str) -> None:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_MPS_PIPE_DIRECTORY": pipe_dir,
            "CUDA_MPS_LOG_DIRECTORY": log_dir,
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


def _run_case(args: argparse.Namespace, case: StageProfileCase) -> dict:
    output_dir = Path(args.output_dir)
    case_dir = output_dir / "cases" / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    base_env = os.environ.copy()
    config_parent = str(Path(args.config_path).resolve().parent)
    base_env.setdefault("EMBODIED_PATH", config_parent)
    env = build_worker_env(
        base_env=base_env,
        case=case,
        mps_pipe_dir=str(args.mps_pipe_dir),
        mps_log_dir=str(args.mps_log_dir),
    )

    cmd = [
        sys.executable,
        "-m",
        "toolkits.rollout_eval.benchmark.openpi_stage_mps_profile",
        "--worker",
        "--config-path",
        args.config_path,
        "--config-name",
        args.config_name,
        "--output-dir",
        args.output_dir,
        "--gpu",
        str(case.gpu),
        "--case-id",
        case.case_id,
        "--mps-sm-current",
        str(case.mps_sm),
        "--num-envs",
        str(case.num_envs),
        "--warmup-steps",
        str(args.warmup_steps),
        "--measure-steps",
        str(args.measure_steps),
    ]
    if args.state_dim is not None:
        cmd.extend(["--state-dim", str(args.state_dim)])
    for override in args.override:
        cmd.extend(["--override", override])

    started = time.perf_counter()
    completed = subprocess.run(
        cmd,
        env=env,
        cwd=Path.cwd(),
        check=False,
        text=True,
        capture_output=True,
        timeout=args.case_timeout_s,
    )
    (case_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (case_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")

    result_path = _worker_result_path(output_dir, case.case_id)
    payload = (
        json.loads(result_path.read_text(encoding="utf-8"))
        if result_path.exists()
        else {}
    )
    status = (
        "pass"
        if completed.returncode == 0 and payload.get("status") == "pass"
        else "failed"
    )
    record = {
        "case_id": case.case_id,
        "status": status,
        "mps_sm": case.mps_sm,
        "num_envs": case.num_envs,
        "gpu": case.gpu,
        "elapsed_s": time.perf_counter() - started,
        "returncode": completed.returncode,
        "metrics": payload.get("metrics"),
        "model": payload.get("model"),
        "error_message": None,
    }
    if status != "pass":
        record["error_message"] = (
            payload.get("error_message") or completed.stderr[-4000:]
        )
    (case_dir / "case_report.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return record


def _resolve_mps_dirs(args: argparse.Namespace) -> None:
    if args.mps_pipe_dir is None:
        args.mps_pipe_dir = (
            f"/tmp/rlinf-openpi-stage-mps-{os.getuid()}-{os.getpid()}/pipe"
        )
    if args.mps_log_dir is None:
        args.mps_log_dir = (
            f"/tmp/rlinf-openpi-stage-mps-{os.getuid()}-{os.getpid()}/log"
        )


def main(argv: list[str] | None = None) -> None:
    """Run the OpenPI stage benchmark matrix or one worker."""
    args = parse_args(argv)
    if args.worker:
        _run_worker(args)
        return

    _resolve_mps_dirs(args)
    output_dir = Path(args.output_dir)
    records: list[dict] = []
    mps_started = False
    try:
        if args.manage_mps:
            args.cuda_visible_device = _resolve_gpu_uuid(int(args.gpu))
            _start_mps_daemon(
                str(args.cuda_visible_device),
                str(args.mps_pipe_dir),
                str(args.mps_log_dir),
            )
            mps_started = True
        else:
            args.cuda_visible_device = str(args.gpu)
        for case in build_stage_cases(args):
            record = None
            if args.skip_existing:
                record = _load_existing_passed_case(output_dir, case)
            if record is None:
                record = _run_case(args, case)
            records.append(record)
            write_summary(output_dir, records)
    finally:
        if mps_started:
            _stop_mps_daemon(str(args.mps_pipe_dir), str(args.mps_log_dir))

    print(json.dumps(write_summary(output_dir, records), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
