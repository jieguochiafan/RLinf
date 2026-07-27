"""Inspect LIBERO MuJoCo state needed to reproduce rendered observations.

This tool creates two LIBERO offscreen environments for the same task, captures
the MuJoCo flat state from one environment, restores it into another environment,
and compares the camera observations.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_CAMERA_NAMES = ["agentview", "robot0_eye_in_hand"]
DEFAULT_DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
IMAGE_KEYS_BY_CAMERA = {
    "agentview": "agentview_image",
    "robot0_eye_in_hand": "robot0_eye_in_hand_image",
}


@dataclass(frozen=True)
class ArrayInfo:
    """JSON-serializable description of an array."""

    dtype: str
    shape: list[int]
    nbytes: int


@dataclass(frozen=True)
class ImageComparison:
    """Pixel comparison summary for one camera image."""

    image_key: str
    equal: bool
    max_abs_diff: int
    mean_abs_diff: float
    source: ArrayInfo
    target: ArrayInfo


@dataclass(frozen=True)
class RenderStateSummary:
    """Summary emitted by the inspection run."""

    suite: str
    task_id: int
    trial_id: int
    source_seed: int
    target_seed: int
    task_language: str
    bddl_file: str
    camera_names: list[str]
    camera_height: int
    camera_width: int
    model_sizes: dict[str, int]
    flat_state: ArrayInfo
    qpos: ArrayInfo
    qvel: ArrayInfo
    static_body_pos: ArrayInfo
    static_body_quat: ArrayInfo
    source_images: dict[str, ArrayInfo]
    same_seed_restored_images: dict[str, ImageComparison]
    different_seed_restored_images: dict[str, ImageComparison]
    different_seed_static_synced_images: dict[str, ImageComparison]
    static_model_delta_before_sync: dict[str, float]
    static_model_delta_after_sync: dict[str, float]
    conclusion: str


def array_info(array: np.ndarray) -> ArrayInfo:
    """Return dtype, shape, and size metadata for a numpy array."""

    return ArrayInfo(
        dtype=str(array.dtype),
        shape=[int(dim) for dim in array.shape],
        nbytes=int(array.nbytes),
    )


def compare_images(
    source: np.ndarray,
    target: np.ndarray,
    *,
    image_key: str,
) -> ImageComparison:
    """Compare two uint8 image arrays with shape and dtype metadata."""

    if source.shape != target.shape:
        raise ValueError(
            f"{image_key} shape mismatch: {source.shape} != {target.shape}"
        )
    diff = np.abs(source.astype(np.int16) - target.astype(np.int16))
    return ImageComparison(
        image_key=image_key,
        equal=bool(np.array_equal(source, target)),
        max_abs_diff=int(diff.max()) if diff.size else 0,
        mean_abs_diff=float(diff.mean()) if diff.size else 0.0,
        source=array_info(source),
        target=array_info(target),
    )


def parse_dummy_action(value: str | None) -> list[float]:
    """Parse a comma-separated action vector."""

    if value is None:
        return list(DEFAULT_DUMMY_ACTION)
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError(f"invalid action vector: {value!r}")
    return [float(part) for part in parts]


def ensure_runtime_dirs(output_dir: Path | None) -> None:
    """Route libraries that write caches/configs to writable locations."""

    base_dir = output_dir or Path("/tmp/rlinf_libero_render_state")
    base_dir.mkdir(parents=True, exist_ok=True)
    defaults = {
        "LIBERO_CONFIG_PATH": base_dir / "libero_config",
        "MPLCONFIGDIR": base_dir / "matplotlib",
        "XDG_CACHE_HOME": base_dir / "cache",
        "MESA_SHADER_CACHE_DIR": base_dir / "mesa_shader_cache",
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, str(value))
    for path_key in [
        "LIBERO_CONFIG_PATH",
        "MPLCONFIGDIR",
        "XDG_CACHE_HOME",
        "MESA_SHADER_CACHE_DIR",
    ]:
        Path(os.environ[path_key]).mkdir(parents=True, exist_ok=True)


def import_libero_modules(libero_type: str) -> tuple[Any, Any, Any]:
    """Import LIBERO variant modules lazily so tests can import this file."""

    os.environ["LIBERO_TYPE"] = libero_type
    if libero_type == "pro":
        from liberopro.liberopro import benchmark, get_libero_path
        from liberopro.liberopro.envs import OffScreenRenderEnv
    elif libero_type == "plus":
        from liberoplus.liberoplus import benchmark, get_libero_path
        from liberoplus.liberoplus.envs import OffScreenRenderEnv
    else:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    return benchmark, get_libero_path, OffScreenRenderEnv


def get_benchmark(benchmark: Any, suite: str) -> Any:
    """Return a LIBERO benchmark instance for the requested suite."""

    if hasattr(benchmark, "get_benchmark"):
        return benchmark.get_benchmark(suite)()
    return benchmark.get_benchmark_dict()[suite]()


def bddl_path_for_task(get_libero_path: Any, task: Any) -> str:
    """Resolve the absolute BDDL path for a benchmark task."""

    return str(
        Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )


def make_env(
    OffScreenRenderEnv: Any,
    *,
    bddl_file: str,
    camera_names: list[str],
    camera_height: int,
    camera_width: int,
    seed: int,
) -> Any:
    """Create one LIBERO offscreen environment."""

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_names=camera_names,
        camera_heights=camera_height,
        camera_widths=camera_width,
    )
    env.seed(seed)
    return env


def forced_observation_from_state(env: Any, flat_state: np.ndarray) -> dict[str, Any]:
    """Regenerate observations by writing a flat MuJoCo state and forcing sensors."""

    return env.regenerate_obs_from_state(flat_state)


def image_key_for_camera(camera_name: str) -> str:
    """Return LIBERO observation key for a camera name."""

    return IMAGE_KEYS_BY_CAMERA.get(camera_name, f"{camera_name}_image")


def copy_static_body_pose(source_env: Any, target_env: Any) -> None:
    """Copy static body transforms that LIBERO mutates during reset."""

    target_env.sim.model.body_pos[:] = source_env.sim.model.body_pos
    target_env.sim.model.body_quat[:] = source_env.sim.model.body_quat
    target_env.sim.forward()


def static_body_pose_delta(source_env: Any, target_env: Any) -> dict[str, float]:
    """Return max absolute differences for static body pose fields."""

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


def compare_observation_images(
    source_obs: dict[str, Any],
    target_obs: dict[str, Any],
    camera_names: list[str],
) -> dict[str, ImageComparison]:
    """Compare all configured camera images between two observations."""

    comparisons = {}
    for camera_name in camera_names:
        image_key = image_key_for_camera(camera_name)
        comparisons[image_key] = compare_images(
            np.asarray(source_obs[image_key]),
            np.asarray(target_obs[image_key]),
            image_key=image_key,
        )
    return comparisons


def collect_source_images(
    source_obs: dict[str, Any],
    camera_names: list[str],
) -> dict[str, ArrayInfo]:
    """Collect image metadata from an observation."""

    return {
        image_key_for_camera(camera_name): array_info(
            np.asarray(source_obs[image_key_for_camera(camera_name)])
        )
        for camera_name in camera_names
    }


def run_inspection(args: argparse.Namespace) -> RenderStateSummary:
    """Run the LIBERO render-state inspection experiment."""

    output_dir = Path(args.output_dir) if args.output_dir is not None else None
    ensure_runtime_dirs(output_dir)
    benchmark, get_libero_path, OffScreenRenderEnv = import_libero_modules(
        args.libero_type
    )
    bench = get_benchmark(benchmark, args.suite)
    task = bench.get_task(args.task_id)
    init_states = bench.get_task_init_states(args.task_id)
    if args.trial_id < 0 or args.trial_id >= len(init_states):
        raise ValueError(
            f"trial_id {args.trial_id} out of range [0, {len(init_states)})"
        )

    bddl_file = bddl_path_for_task(get_libero_path, task)
    camera_names = list(args.camera_names)
    dummy_action = parse_dummy_action(args.dummy_action)

    env_source = make_env(
        OffScreenRenderEnv,
        bddl_file=bddl_file,
        camera_names=camera_names,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        seed=args.source_seed,
    )
    env_same_seed = None
    env_different_seed = None
    env_static_synced = None
    try:
        env_source.reset()
        env_source.set_init_state(init_states[args.trial_id])
        for _ in range(args.step_count):
            env_source.step(dummy_action)
        flat_state = env_source.get_sim_state().copy()
        source_obs = forced_observation_from_state(env_source, flat_state)

        env_same_seed = make_env(
            OffScreenRenderEnv,
            bddl_file=bddl_file,
            camera_names=camera_names,
            camera_height=args.camera_height,
            camera_width=args.camera_width,
            seed=args.source_seed,
        )
        env_same_seed.reset()
        same_seed_obs = forced_observation_from_state(env_same_seed, flat_state)

        env_different_seed = make_env(
            OffScreenRenderEnv,
            bddl_file=bddl_file,
            camera_names=camera_names,
            camera_height=args.camera_height,
            camera_width=args.camera_width,
            seed=args.target_seed,
        )
        env_different_seed.reset()
        different_seed_obs = forced_observation_from_state(env_different_seed, flat_state)
        static_delta_before = static_body_pose_delta(env_source, env_different_seed)

        env_static_synced = make_env(
            OffScreenRenderEnv,
            bddl_file=bddl_file,
            camera_names=camera_names,
            camera_height=args.camera_height,
            camera_width=args.camera_width,
            seed=args.target_seed,
        )
        env_static_synced.reset()
        copy_static_body_pose(env_source, env_static_synced)
        static_synced_obs = forced_observation_from_state(env_static_synced, flat_state)
        static_delta_after = static_body_pose_delta(env_source, env_static_synced)

        model = env_source.sim.model
        data = env_source.sim.data
        model_sizes = {
            "nq": int(model.nq),
            "nv": int(model.nv),
            "na": int(model.na),
            "nu": int(model.nu),
            "nbody": int(model.nbody),
            "ngeom": int(model.ngeom),
            "njnt": int(model.njnt),
            "ncam": int(model.ncam),
        }
        summary = RenderStateSummary(
            suite=args.suite,
            task_id=int(args.task_id),
            trial_id=int(args.trial_id),
            source_seed=int(args.source_seed),
            target_seed=int(args.target_seed),
            task_language=str(task.language),
            bddl_file=bddl_file,
            camera_names=camera_names,
            camera_height=int(args.camera_height),
            camera_width=int(args.camera_width),
            model_sizes=model_sizes,
            flat_state=array_info(flat_state),
            qpos=array_info(np.asarray(data.qpos)),
            qvel=array_info(np.asarray(data.qvel)),
            static_body_pos=array_info(np.asarray(model.body_pos)),
            static_body_quat=array_info(np.asarray(model.body_quat)),
            source_images=collect_source_images(source_obs, camera_names),
            same_seed_restored_images=compare_observation_images(
                source_obs, same_seed_obs, camera_names
            ),
            different_seed_restored_images=compare_observation_images(
                source_obs, different_seed_obs, camera_names
            ),
            different_seed_static_synced_images=compare_observation_images(
                source_obs, static_synced_obs, camera_names
            ),
            static_model_delta_before_sync=static_delta_before,
            static_model_delta_after_sync=static_delta_after,
            conclusion=(
                "A LIBERO flat MuJoCo state is sufficient for pixel-identical "
                "rerendering only when the target simulator has the same task XML "
                "and static model body poses. Different reset seeds can mutate "
                "model.body_pos/body_quat for fixtures; save or reproduce those "
                "static fields as well if the target simulator is not seeded and "
                "reset identically."
            ),
        )
        if output_dir is not None:
            write_outputs(
                output_dir,
                summary,
                flat_state,
                np.asarray(model.body_pos).copy(),
                np.asarray(model.body_quat).copy(),
                source_obs,
                camera_names,
            )
        return summary
    finally:
        for env in [env_static_synced, env_different_seed, env_same_seed, env_source]:
            if env is not None:
                env.close()


def write_outputs(
    output_dir: Path,
    summary: RenderStateSummary,
    flat_state: np.ndarray,
    static_body_pos: np.ndarray,
    static_body_quat: np.ndarray,
    source_obs: dict[str, Any],
    camera_names: list[str],
) -> None:
    """Write summary JSON and captured arrays."""

    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = asdict(summary)
    (output_dir / "render_state_summary.json").write_text(
        json.dumps(serializable, indent=2, sort_keys=True) + "\n"
    )
    image_arrays = {
        f"source_{image_key_for_camera(camera_name)}": np.asarray(
            source_obs[image_key_for_camera(camera_name)]
        )
        for camera_name in camera_names
    }
    np.savez_compressed(
        output_dir / "render_state_capture.npz",
        flat_state=flat_state,
        static_body_pos=static_body_pos,
        static_body_quat=static_body_quat,
        **image_arrays,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--trial-id", type=int, default=0)
    parser.add_argument("--source-seed", type=int, default=0)
    parser.add_argument("--target-seed", type=int, default=1)
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--camera-names", nargs="+", default=DEFAULT_CAMERA_NAMES)
    parser.add_argument("--step-count", type=int, default=3)
    parser.add_argument("--dummy-action", default=None)
    parser.add_argument(
        "--libero-type",
        choices=["standard", "pro", "plus"],
        default=os.environ.get("LIBERO_TYPE", "standard").lower(),
    )
    parser.add_argument("--output-dir", default=None)
    return parser


def main() -> None:
    """Run the command-line inspection."""

    parser = build_arg_parser()
    args = parser.parse_args()
    summary = run_inspection(args)
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
