from __future__ import annotations

import argparse
import json
import os
import statistics as stats
import time
from multiprocessing import Pipe, get_context
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import numpy as np


def _parse_core_counts(value: str) -> list[int]:
    counts = [int(item) for item in value.split(",") if item.strip()]
    if not counts or any(count <= 0 for count in counts):
        raise argparse.ArgumentTypeError("core counts must be positive integers")
    return counts


def _parse_core_pool(value: str | None) -> list[int]:
    if not value:
        return sorted(os.sched_getaffinity(0))
    cores: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(item) for item in part.split("-", 1)]
            cores.extend(range(start, end + 1))
        else:
            cores.append(int(part))
    return sorted(dict.fromkeys(cores))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100.0
    lo = int(index)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "mean_ms": stats.mean(values) * 1000.0,
        "median_ms": stats.median(values) * 1000.0,
        "p90_ms": _percentile(values, 90.0) * 1000.0,
        "p95_ms": _percentile(values, 95.0) * 1000.0,
        "min_ms": min(values) * 1000.0,
        "max_ms": max(values) * 1000.0,
    }


def _make_env(
    *,
    task_name: str,
    robot_name: str,
    camera_width: int,
    camera_height: int,
    seed: int,
):
    import robocasa  # noqa: F401 RoboCasa registers tasks at import time.
    import robosuite
    from robosuite.controllers import load_composite_controller_config

    controller_config = load_composite_controller_config(
        controller=None,
        robot=robot_name,
    )
    return robosuite.make(
        env_name=task_name,
        robots=robot_name,
        controller_configs=controller_config,
        camera_names=[
            "robot0_agentview_left",
            "robot0_eye_in_hand",
            "robot0_agentview_right",
        ],
        camera_widths=camera_width,
        camera_heights=camera_height,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_depths=False,
        seed=seed,
        translucent_robot=False,
        render_camera="robot0_agentview_center",
    )


def _action_dim(env: Any) -> int:
    if hasattr(env, "action_dim"):
        return int(env.action_dim)
    low, high = env.action_spec
    del high
    return int(np.asarray(low).shape[0])


def _build_actions(env: Any, steps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    low, high = env.action_spec
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    actions = rng.uniform(low=low, high=high, size=(steps, _action_dim(env))).astype(
        np.float32
    )
    gripper_index = actions.shape[1] - 1
    actions[:, gripper_index] = np.sign(actions[:, gripper_index])
    return actions


def _worker(conn: Connection, args: argparse.Namespace, cores: tuple[int, ...]) -> None:
    os.sched_setaffinity(0, set(cores))
    result: dict[str, Any] = {
        "requested_cores": list(cores),
        "actual_affinity": sorted(os.sched_getaffinity(0)),
    }
    env = None
    try:
        env_start = time.perf_counter()
        env = _make_env(
            task_name=args.task_name,
            robot_name=args.robot_name,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            seed=args.seed,
        )
        result["make_env_s"] = time.perf_counter() - env_start

        reset_times: list[float] = []
        step_times: list[float] = []
        chunk_times: list[float] = []

        actions = _build_actions(
            env,
            args.warmup_steps + args.measure_steps,
            seed=args.seed + 1009,
        )

        for reset_index in range(args.resets):
            reset_start = time.perf_counter()
            env.reset()
            reset_times.append(time.perf_counter() - reset_start)

            for action in actions[: args.warmup_steps]:
                env.step(action)

            measured_actions = actions[args.warmup_steps :]
            for chunk_start in range(0, len(measured_actions), args.chunk_size):
                chunk_actions = measured_actions[
                    chunk_start : chunk_start + args.chunk_size
                ]
                if len(chunk_actions) < args.chunk_size:
                    break
                chunk_timer = time.perf_counter()
                for action in chunk_actions:
                    step_timer = time.perf_counter()
                    env.step(action)
                    step_times.append(time.perf_counter() - step_timer)
                chunk_times.append(time.perf_counter() - chunk_timer)

        result.update(
            {
                "reset": _summary(reset_times),
                "step": _summary(step_times),
                "chunk": _summary(chunk_times),
                "reset_samples_s": reset_times,
                "step_samples_s": step_times,
                "chunk_samples_s": chunk_times,
                "sample_counts": {
                    "reset": len(reset_times),
                    "step": len(step_times),
                    "chunk": len(chunk_times),
                },
                "status": "ok",
            }
        )
    except BaseException as exc:
        result.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        conn.send(result)
        conn.close()


def run_one(args: argparse.Namespace, cores: tuple[int, ...]) -> dict[str, Any]:
    ctx = get_context("spawn")
    parent, child = Pipe(duplex=False)
    process = ctx.Process(target=_worker, args=(child, args, cores))
    process.start()
    child.close()
    if not parent.poll(args.timeout_s):
        process.terminate()
        process.join(timeout=10)
        return {
            "requested_cores": list(cores),
            "status": "timeout",
            "timeout_s": args.timeout_s,
        }
    result = parent.recv()
    process.join(timeout=10)
    result["exitcode"] = process.exitcode
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a single RoboCasa/robosuite environment under different "
            "process CPU affinity core counts. This does not start Ray."
        )
    )
    parser.add_argument("--task-name", default="PnPCounterToCab")
    parser.add_argument("--robot-name", default="PandaOmron")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--camera-width", type=int, default=224)
    parser.add_argument("--camera-height", type=int, default=224)
    parser.add_argument("--core-counts", type=_parse_core_counts, default="1,2,4,8")
    parser.add_argument("--core-pool", default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--resets", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "logs/latency_balance_eval/robocasa_single_env_affinity_benchmark.json"
        ),
    )
    args = parser.parse_args()

    core_pool = _parse_core_pool(args.core_pool)
    max_core_count = max(args.core_counts)
    if len(core_pool) < max_core_count:
        raise ValueError(
            f"core pool has {len(core_pool)} cores, but {max_core_count} required"
        )

    results = []
    for core_count in args.core_counts:
        cores = tuple(core_pool[:core_count])
        for repeat in range(args.repeats):
            run_args = argparse.Namespace(**vars(args))
            run_args.seed = args.seed + repeat
            result = run_one(run_args, cores)
            result["core_count"] = core_count
            result["repeat"] = repeat
            results.append(result)
            status = result.get("status")
            if status == "ok":
                print(
                    "core_count={core_count} repeat={repeat} "
                    "step_mean={step:.2f}ms chunk_mean={chunk:.2f}ms "
                    "reset_mean={reset:.2f}ms affinity={affinity}".format(
                        core_count=core_count,
                        repeat=repeat,
                        step=result["step"]["mean_ms"],
                        chunk=result["chunk"]["mean_ms"],
                        reset=result["reset"]["mean_ms"],
                        affinity=result["actual_affinity"],
                    ),
                    flush=True,
                )
            else:
                print(
                    f"core_count={core_count} repeat={repeat} status={status} "
                    f"error={result.get('error', '')}",
                    flush=True,
                )

    grouped: dict[int, list[dict[str, Any]]] = {}
    for result in results:
        if result.get("status") == "ok":
            grouped.setdefault(int(result["core_count"]), []).append(result)

    summary = {}
    for core_count, items in sorted(grouped.items()):
        summary[core_count] = {
            "step_mean_ms_mean": stats.mean(
                item["step"]["mean_ms"] for item in items
            ),
            "step_mean_ms_stdev": stats.stdev(
                [item["step"]["mean_ms"] for item in items]
            )
            if len(items) > 1
            else 0.0,
            "chunk_mean_ms_mean": stats.mean(
                item["chunk"]["mean_ms"] for item in items
            ),
            "chunk_mean_ms_stdev": stats.stdev(
                [item["chunk"]["mean_ms"] for item in items]
            )
            if len(items) > 1
            else 0.0,
            "reset_mean_ms_mean": stats.mean(
                item["reset"]["mean_ms"] for item in items
            ),
            "ok_repeats": len(items),
        }

    payload = {
        "config": {
            "task_name": args.task_name,
            "robot_name": args.robot_name,
            "camera_width": args.camera_width,
            "camera_height": args.camera_height,
            "core_counts": args.core_counts,
            "core_pool": core_pool,
            "repeats": args.repeats,
            "resets": args.resets,
            "warmup_steps": args.warmup_steps,
            "measure_steps": args.measure_steps,
            "chunk_size": args.chunk_size,
        },
        "summary": summary,
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(args.output_json)


if __name__ == "__main__":
    main()
