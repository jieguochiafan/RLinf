"""Validate LIBERO flat-state render transfer and MuJoCo call overhead."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from toolkits.inspect_libero_mujoco_render_state import (
    DEFAULT_CAMERA_NAMES,
    compare_observation_images,
    ensure_runtime_dirs,
    forced_observation_from_state,
    import_libero_modules,
    make_env,
    parse_dummy_action,
)


def summarize_timings(samples_s: list[float]) -> dict[str, float | int | None]:
    """Return basic timing statistics in seconds."""

    if not samples_s:
        return {
            "count": 0,
            "mean_s": None,
            "median_s": None,
            "min_s": None,
            "max_s": None,
            "p95_s": None,
            "p99_s": None,
        }
    sorted_samples = sorted(float(sample) for sample in samples_s)
    return {
        "count": len(sorted_samples),
        "mean_s": statistics.fmean(sorted_samples),
        "median_s": statistics.median(sorted_samples),
        "min_s": sorted_samples[0],
        "max_s": sorted_samples[-1],
        "p95_s": float(np.percentile(sorted_samples, 95)),
        "p99_s": float(np.percentile(sorted_samples, 99)),
    }


def compute_timing_ratio(
    numerator: dict[str, float | int | None],
    denominator: dict[str, float | int | None],
) -> float | None:
    """Return the ratio between two timing means."""

    numerator_mean = numerator.get("mean_s")
    denominator_mean = denominator.get("mean_s")
    if not numerator_mean or not denominator_mean:
        return None
    return float(numerator_mean) / float(denominator_mean)


def summarize_comparisons(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize per-sync image comparisons."""

    per_camera: dict[str, dict[str, Any]] = {}
    all_images_equal = True
    for comparison in comparisons:
        for image_key, image_result in comparison["images"].items():
            camera_summary = per_camera.setdefault(
                image_key,
                {
                    "num_equal": 0,
                    "num_compared": 0,
                    "max_abs_diff": 0,
                },
            )
            camera_summary["num_compared"] += 1
            if image_result["equal"]:
                camera_summary["num_equal"] += 1
            else:
                all_images_equal = False
            camera_summary["max_abs_diff"] = max(
                camera_summary["max_abs_diff"],
                int(image_result["max_abs_diff"]),
            )
    return {
        "num_syncs": len(comparisons),
        "all_images_equal": all_images_equal,
        "per_camera": per_camera,
    }


def static_body_pose_delta(source_env: Any, target_env: Any) -> dict[str, float]:
    """Return max static model body pose deltas between two envs."""

    return {
        "body_pos": float(
            np.max(np.abs(source_env.sim.model.body_pos - target_env.sim.model.body_pos))
        ),
        "body_quat": float(
            np.max(
                np.abs(source_env.sim.model.body_quat - target_env.sim.model.body_quat)
            )
        ),
    }


def sample_action(
    rng: np.random.Generator,
    low: np.ndarray,
    high: np.ndarray,
) -> list[float]:
    """Sample one continuous action uniformly within action bounds."""

    action = rng.uniform(low=low, high=high)
    return [float(value) for value in action.tolist()]


def flat_state_filename(*, step_idx: int) -> str:
    """Return the saved flat-state filename for a transfer step."""

    return f"flat_state_step_{step_idx:06d}.npy"


def record_step_timing(
    records: list[dict[str, float | int]],
    *,
    step_idx: int,
    latency_s: float,
) -> None:
    """Append one source env step timing record."""

    records.append({"step": int(step_idx), "latency_s": float(latency_s)})


def save_flat_state(
    output_dir: Path | None,
    flat_state: np.ndarray,
    *,
    step_idx: int,
) -> str | None:
    """Save a transferred flat state and return its path."""

    if output_dir is None:
        return None
    state_dir = output_dir / "flat_states"
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / flat_state_filename(step_idx=step_idx)
    np.save(path, flat_state)
    return str(path)


def action_for_step(
    *,
    random_action: bool,
    rng: np.random.Generator,
    action_low: np.ndarray,
    action_high: np.ndarray,
    fixed_action: list[float],
) -> list[float]:
    """Return either a seeded random action or the fixed action."""

    if random_action:
        return sample_action(rng, action_low, action_high)
    return list(fixed_action)


def _comparison_to_dict(comparison: Any) -> dict[str, Any]:
    """Convert an ImageComparison dataclass to a plain dictionary."""

    return asdict(comparison)


def _get_benchmark_instance(benchmark: Any, suite: str) -> Any:
    if hasattr(benchmark, "get_benchmark"):
        return benchmark.get_benchmark(suite)()
    return benchmark.get_benchmark_dict()[suite]()


def _bddl_path(get_libero_path: Any, task: Any) -> str:
    return str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)


def validate_state_transfer(args: argparse.Namespace) -> dict[str, Any]:
    """Run two same-seed envs and compare rendered frames after state transfer."""

    benchmark, get_libero_path, OffScreenRenderEnv = import_libero_modules(
        args.libero_type
    )
    bench = _get_benchmark_instance(benchmark, args.suite)
    task = bench.get_task(args.task_id)
    init_states = bench.get_task_init_states(args.task_id)
    if args.trial_id < 0 or args.trial_id >= len(init_states):
        raise ValueError(
            f"trial_id {args.trial_id} out of range [0, {len(init_states)})"
        )
    bddl_file = _bddl_path(get_libero_path, task)
    camera_names = list(args.camera_names)
    fixed_action = parse_dummy_action(args.dummy_action)
    action_rng = np.random.default_rng(args.action_seed)
    output_dir = Path(args.output_dir) if args.output_dir else None

    env_source = make_env(
        OffScreenRenderEnv,
        bddl_file=bddl_file,
        camera_names=camera_names,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        seed=args.seed,
    )
    env_target = make_env(
        OffScreenRenderEnv,
        bddl_file=bddl_file,
        camera_names=camera_names,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        seed=args.seed,
    )
    try:
        env_source.reset()
        env_source.set_init_state(init_states[args.trial_id])
        if args.reseed_target_before_reset:
            env_target.seed(args.seed)
        env_target.reset()
        env_target.set_init_state(init_states[args.trial_id])
        static_delta_after_reset = static_body_pose_delta(env_source, env_target)

        comparisons: list[dict[str, Any]] = []
        target_rerender_samples = []
        actions: list[dict[str, Any]] = []
        source_step_timing_records: list[dict[str, float | int]] = []
        for step_idx in range(1, args.total_steps + 1):
            action = action_for_step(
                random_action=args.random_action,
                rng=action_rng,
                action_low=np.asarray(env_source.env.action_spec[0], dtype=np.float64),
                action_high=np.asarray(env_source.env.action_spec[1], dtype=np.float64),
                fixed_action=fixed_action,
            )
            actions.append({"step": step_idx, "action": action})
            start = time.perf_counter()
            env_source.step(action)
            record_step_timing(
                source_step_timing_records,
                step_idx=step_idx,
                latency_s=time.perf_counter() - start,
            )
            if step_idx % args.transfer_interval != 0:
                continue
            flat_state = env_source.get_sim_state().copy()
            flat_state_path = save_flat_state(output_dir, flat_state, step_idx=step_idx)
            source_obs = forced_observation_from_state(env_source, flat_state)
            start = time.perf_counter()
            target_obs = forced_observation_from_state(env_target, flat_state)
            target_rerender_samples.append(time.perf_counter() - start)
            image_comparisons = compare_observation_images(
                source_obs,
                target_obs,
                camera_names,
            )
            comparisons.append(
                {
                    "step": step_idx,
                    "action": action,
                    "flat_state_path": flat_state_path,
                    "flat_state_shape": list(flat_state.shape),
                    "flat_state_dtype": str(flat_state.dtype),
                    "images": {
                        image_key: _comparison_to_dict(result)
                        for image_key, result in image_comparisons.items()
                    },
                }
            )

        return {
            "suite": args.suite,
            "task_id": args.task_id,
            "trial_id": args.trial_id,
            "seed": args.seed,
            "task_language": str(task.language),
            "bddl_file": bddl_file,
            "camera_names": camera_names,
            "camera_height": args.camera_height,
            "camera_width": args.camera_width,
            "total_steps": args.total_steps,
            "transfer_interval": args.transfer_interval,
            "random_action": bool(args.random_action),
            "action_seed": int(args.action_seed),
            "actions": actions,
            "reseed_target_before_reset": bool(args.reseed_target_before_reset),
            "static_model_delta_after_reset": static_delta_after_reset,
            "source_step_timings": source_step_timing_records,
            "source_step_timing": summarize_timings(
                [float(record["latency_s"]) for record in source_step_timing_records]
            ),
            "comparisons": comparisons,
            "summary": summarize_comparisons(comparisons),
            "target_rerender_from_flat_state_timing": summarize_timings(
                target_rerender_samples
            ),
        }
    finally:
        env_target.close()
        env_source.close()


def benchmark_forward_and_step(args: argparse.Namespace) -> dict[str, Any]:
    """Benchmark forward, raw sim.step, and high-level env.step(action)."""

    benchmark, get_libero_path, OffScreenRenderEnv = import_libero_modules(
        args.libero_type
    )
    bench = _get_benchmark_instance(benchmark, args.suite)
    task = bench.get_task(args.task_id)
    init_state = bench.get_task_init_states(args.task_id)[args.trial_id]
    bddl_file = _bddl_path(get_libero_path, task)
    env = make_env(
        OffScreenRenderEnv,
        bddl_file=bddl_file,
        camera_names=list(args.camera_names),
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        seed=args.seed,
    )
    try:
        env.reset()
        env.set_init_state(init_state)
        fixed_action = parse_dummy_action(args.dummy_action)
        action_rng = np.random.default_rng(args.action_seed)
        flat_state = env.get_sim_state().copy()

        for _ in range(args.benchmark_warmup):
            env.sim.forward()
        forward_samples = []
        for _ in range(args.benchmark_iters):
            env.set_state(flat_state)
            start = time.perf_counter()
            env.sim.forward()
            forward_samples.append(time.perf_counter() - start)

        env.set_init_state(init_state)
        flat_state = env.get_sim_state().copy()
        for _ in range(args.benchmark_warmup):
            env.set_state(flat_state)
            env.sim.step()
        step_samples = []
        for _ in range(args.benchmark_iters):
            env.set_state(flat_state)
            start = time.perf_counter()
            env.sim.step()
            step_samples.append(time.perf_counter() - start)

        env.set_init_state(init_state)
        flat_state = env.get_sim_state().copy()
        env_step_action = action_for_step(
            random_action=args.random_action,
            rng=action_rng,
            action_low=np.asarray(env.env.action_spec[0], dtype=np.float64),
            action_high=np.asarray(env.env.action_spec[1], dtype=np.float64),
            fixed_action=fixed_action,
        )
        for _ in range(args.benchmark_warmup):
            env.set_state(flat_state)
            env.sim.forward()
            env.step(env_step_action)
        env_step_action_samples = []
        for _ in range(args.benchmark_iters):
            env.set_state(flat_state)
            env.sim.forward()
            start = time.perf_counter()
            env.step(env_step_action)
            env_step_action_samples.append(time.perf_counter() - start)

        forward_summary = summarize_timings(forward_samples)
        step_summary = summarize_timings(step_samples)
        env_step_action_summary = summarize_timings(env_step_action_samples)
        return {
            "benchmark_iters": args.benchmark_iters,
            "benchmark_warmup": args.benchmark_warmup,
            "random_action": bool(args.random_action),
            "env_step_action_sample": env_step_action,
            "forward": forward_summary,
            "sim_step": step_summary,
            "env_step_action": env_step_action_summary,
            "sim_step_to_forward_mean_ratio": compute_timing_ratio(
                step_summary,
                forward_summary,
            ),
            "env_step_action_to_forward_mean_ratio": compute_timing_ratio(
                env_step_action_summary,
                forward_summary,
            ),
        }
    finally:
        env.close()


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    """Run both requested validations."""

    ensure_runtime_dirs(Path(args.output_dir) if args.output_dir else None)
    result = {
        "state_transfer": validate_state_transfer(args),
        "timing": benchmark_forward_and_step(args),
    }
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "libero_state_transfer_validation.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--trial-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--camera-names", nargs="+", default=DEFAULT_CAMERA_NAMES)
    parser.add_argument("--total-steps", type=int, default=50)
    parser.add_argument("--transfer-interval", type=int, default=10)
    parser.add_argument(
        "--reseed-target-before-reset",
        action="store_true",
        help=(
            "Call target.seed(seed) immediately before target.reset(). "
            "LIBERO seed() uses NumPy's global RNG, so this is needed when two "
            "envs are reset sequentially in one process."
        ),
    )
    parser.add_argument("--dummy-action", default=None)
    parser.add_argument(
        "--random-action",
        action="store_true",
        help="Use seeded uniformly random actions from the environment action bounds.",
    )
    parser.add_argument("--action-seed", type=int, default=123)
    parser.add_argument("--benchmark-iters", type=int, default=200)
    parser.add_argument("--benchmark-warmup", type=int, default=10)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--libero-type", choices=["standard", "pro", "plus"], default="standard")
    return parser


def main() -> None:
    """Run CLI."""

    args = build_arg_parser().parse_args()
    result = run_validation(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
