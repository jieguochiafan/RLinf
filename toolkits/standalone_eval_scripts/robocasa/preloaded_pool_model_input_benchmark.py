#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Capture exact RoboCasa model inputs and benchmark preload-pool memory.

The parent process launches isolated baseline and pooled cases. The baseline
preloads one physical object and one 512x512 texture; the pooled case preloads
four of each. Both use only the two cameras selected by the example's
``image_space: 2views`` configuration. Their RSS and per-PID NVML allocations
can therefore be compared without model-loading or debug-camera memory.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from PIL import Image, ImageDraw
from preloaded_reset_pool_smoke import (
    POOL_OBJECTS,
    _build_env_class,
    _model_identity,
    make_texture_pool,
)

MODEL_CAMERAS = ("robot0_agentview_left", "robot0_eye_in_hand")


def parse_args() -> argparse.Namespace:
    """Parse benchmark arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-resets", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--texture-size", type=int, default=512)
    parser.add_argument(
        "--case",
        choices=("baseline", "pooled"),
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def _valid_used_memory(value: Any) -> int | None:
    """Normalize NVML's unavailable-memory sentinel."""
    if value is None:
        return None
    value = int(value)
    if value < 0 or value >= 1 << 60:
        return None
    return value


def _nvml_snapshot(pid: int) -> dict[str, Any]:
    """Read global and current-PID GPU memory from NVML."""
    try:
        import pynvml

        pynvml.nvmlInit()
    except Exception as error:
        return {"available": False, "error": repr(error), "devices": []}

    devices = []
    try:
        for device_index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            allocations: dict[str, int] = {}
            process_errors = []
            for source, function_name in (
                ("graphics", "nvmlDeviceGetGraphicsRunningProcesses"),
                ("compute", "nvmlDeviceGetComputeRunningProcesses"),
            ):
                function = getattr(pynvml, function_name, None)
                if function is None:
                    continue
                try:
                    processes = function(handle)
                except Exception as error:
                    process_errors.append(f"{source}: {error!r}")
                    continue
                for process in processes:
                    if int(process.pid) != pid:
                        continue
                    used_memory = _valid_used_memory(process.usedGpuMemory)
                    if used_memory is not None:
                        allocations[source] = used_memory

            # Graphics and compute lists can report the same allocation. Use
            # the maximum instead of double-counting it.
            pid_used_bytes = max(allocations.values(), default=0)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            devices.append(
                {
                    "index": device_index,
                    "name": name,
                    "global_used_bytes": int(memory_info.used),
                    "pid_used_bytes": pid_used_bytes,
                    "pid_allocations": allocations,
                    "process_query_errors": process_errors,
                }
            )
    finally:
        pynvml.nvmlShutdown()
    return {"available": True, "devices": devices}


def _gpu_totals(snapshot: dict[str, Any]) -> dict[str, int | None]:
    """Summarize an NVML snapshot across devices."""
    if not snapshot["available"]:
        return {"pid_used_bytes": None, "global_used_bytes": None}
    return {
        "pid_used_bytes": sum(
            device["pid_used_bytes"] for device in snapshot["devices"]
        ),
        "global_used_bytes": sum(
            device["global_used_bytes"] for device in snapshot["devices"]
        ),
    }


def _state_16d(obs: dict[str, Any]) -> np.ndarray:
    """Construct the exact 16D state selected by the training config."""
    return np.concatenate(
        [
            obs["robot0_base_to_eef_pos"],
            obs["robot0_base_to_eef_quat"],
            obs["robot0_base_pos"],
            obs["robot0_base_quat"],
            obs["robot0_gripper_qpos"],
        ]
    ).astype(np.float32)


def _save_input_pair(
    output_dir: Path,
    reset_index: int,
    obs: dict[str, Any],
    state: np.ndarray,
    prompt: str,
) -> dict[str, Any]:
    """Save both input images and the corresponding RLinf input arrays."""
    main_image = np.ascontiguousarray(obs[f"{MODEL_CAMERAS[0]}_image"][::-1])
    wrist_image = np.ascontiguousarray(obs[f"{MODEL_CAMERAS[1]}_image"][::-1])
    reset_dir = output_dir / f"reset_{reset_index:02d}"
    reset_dir.mkdir(parents=True, exist_ok=True)
    main_path = reset_dir / "main_robot0_agentview_left.png"
    wrist_path = reset_dir / "wrist_robot0_eye_in_hand.png"
    Image.fromarray(main_image).save(main_path)
    Image.fromarray(wrist_image).save(wrist_path)

    np.savez_compressed(
        reset_dir / "rlinf_model_input.npz",
        main_images=main_image[None],
        wrist_images=wrist_image[None],
        states=state[None],
        task_descriptions=np.asarray([prompt]),
    )

    sheet = Image.new("RGB", (448, 252), "white")
    sheet.paste(Image.fromarray(main_image), (0, 28))
    sheet.paste(Image.fromarray(wrist_image), (224, 28))
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 7), "base_0_rgb: robot0_agentview_left", fill="black")
    draw.text((230, 7), "left_wrist_0_rgb: robot0_eye_in_hand", fill="black")
    pair_path = reset_dir / "model_input_pair.png"
    sheet.save(pair_path)

    return {
        "reset_index": reset_index,
        "main_image": os.fspath(main_path.relative_to(output_dir)),
        "wrist_image": os.fspath(wrist_path.relative_to(output_dir)),
        "input_pair": os.fspath(pair_path.relative_to(output_dir)),
        "npz": os.fspath((reset_dir / "rlinf_model_input.npz").relative_to(output_dir)),
        "main_shape": list(main_image.shape),
        "wrist_shape": list(wrist_image.shape),
        "state_shape": list(state.shape),
        "state_16d": state.tolist(),
        "prompt": prompt,
    }


def _create_env(
    texture_paths: list[Path],
    object_specs: tuple[tuple[str, str, float], ...],
    seed: int,
):
    """Create one env with exactly the configured model-input cameras."""
    from robosuite.controllers import load_composite_controller_config

    env_class = _build_env_class(include_debug_camera=False)
    controller_config = load_composite_controller_config(
        controller=None,
        robot="PandaOmron",
    )
    return env_class(
        robots="PandaOmron",
        controller_configs=controller_config,
        camera_names=list(MODEL_CAMERAS),
        camera_widths=224,
        camera_heights=224,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_depths=False,
        seed=seed,
        hard_reset=False,
        translucent_robot=False,
        texture_paths=texture_paths,
        object_specs=object_specs,
    )


def _model_counts(env: Any) -> dict[str, int]:
    """Return useful compiled-model asset counts."""
    model = env.sim.model
    return {
        name: int(getattr(model, name))
        for name in ("nbody", "ngeom", "nmesh", "nmeshvert", "ntex", "nmat")
    }


def run_case(args: argparse.Namespace) -> None:
    """Run one isolated memory-measurement case."""
    case_dir = args.output_dir / args.case
    case_dir.mkdir(parents=True, exist_ok=True)
    all_texture_paths = make_texture_pool(case_dir, args.texture_size)
    if args.case == "baseline":
        object_specs = POOL_OBJECTS[:1]
        texture_paths = all_texture_paths[:1]
        num_resets = 1
    else:
        object_specs = POOL_OBJECTS
        texture_paths = all_texture_paths
        num_resets = args.num_resets

    process = psutil.Process()
    gpu_before = _nvml_snapshot(os.getpid())
    rss_before = process.memory_info().rss
    initialize_start = time.perf_counter()
    env = _create_env(texture_paths, object_specs, args.seed)
    initialization_seconds = time.perf_counter() - initialize_start
    model_identity = _model_identity(env)
    input_records = []
    reset_seconds = []
    try:
        for reset_index in range(num_resets):
            reset_start = time.perf_counter()
            obs = env.reset()
            reset_seconds.append(time.perf_counter() - reset_start)
            if _model_identity(env) != model_identity:
                raise RuntimeError("Soft reset replaced the compiled MuJoCo model")
            state = _state_16d(obs)
            if state.shape != (16,):
                raise RuntimeError(f"Expected 16D state, got {state.shape}")
            prompt = str(env.get_ep_meta().get("lang", ""))
            input_records.append(
                _save_input_pair(case_dir, reset_index, obs, state, prompt)
            )

        gpu_after = _nvml_snapshot(os.getpid())
        rss_after = process.memory_info().rss
        gpu_before_totals = _gpu_totals(gpu_before)
        gpu_after_totals = _gpu_totals(gpu_after)
        pid_gpu_increment = None
        global_gpu_increment = None
        if gpu_after_totals["pid_used_bytes"] is not None:
            pid_gpu_increment = (
                gpu_after_totals["pid_used_bytes"] - gpu_before_totals["pid_used_bytes"]
            )
            global_gpu_increment = (
                gpu_after_totals["global_used_bytes"]
                - gpu_before_totals["global_used_bytes"]
            )
        metrics = {
            "case": args.case,
            "pid": os.getpid(),
            "seed": args.seed,
            "object_pool_size": len(object_specs),
            "texture_pool_size": len(texture_paths),
            "texture_size": args.texture_size,
            "camera_names": list(MODEL_CAMERAS),
            "image_space": "2views",
            "openpi_image_mapping": {
                "base_0_rgb": "robot0_agentview_left",
                "left_wrist_0_rgb": "robot0_eye_in_hand",
                "right_wrist_0_rgb": None,
            },
            "openpi_image_mask": {
                "base_0_rgb": True,
                "left_wrist_0_rgb": True,
                "right_wrist_0_rgb": False,
            },
            "initialization_seconds": initialization_seconds,
            "reset_seconds": reset_seconds,
            "compiled_model_identity": model_identity,
            "compiled_model_counts": _model_counts(env),
            "cpu_rss_before_env_bytes": rss_before,
            "cpu_rss_after_resets_bytes": rss_after,
            "cpu_rss_increment_bytes": rss_after - rss_before,
            "gpu_pid_used_before_bytes": gpu_before_totals["pid_used_bytes"],
            "gpu_pid_used_after_bytes": gpu_after_totals["pid_used_bytes"],
            "gpu_pid_increment_bytes": pid_gpu_increment,
            "gpu_global_used_before_bytes": gpu_before_totals["global_used_bytes"],
            "gpu_global_used_after_bytes": gpu_after_totals["global_used_bytes"],
            "gpu_global_increment_bytes": global_gpu_increment,
            "nvml_before": gpu_before,
            "nvml_after": gpu_after,
            "model_inputs": input_records,
        }
        metrics_path = case_dir / "case_metrics.json"
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(metrics, ensure_ascii=False))
    finally:
        env.close()


def _format_mib(value: int | None) -> float | None:
    """Convert bytes to MiB without hiding unavailable measurements."""
    return None if value is None else value / (1024**2)


def run_parent(args: argparse.Namespace) -> None:
    """Launch isolated cases and write their memory comparison."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    child_environment = os.environ.copy()
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        child_environment.setdefault(variable, "1")
    child_environment.setdefault("MUJOCO_GL", "egl")

    for case in ("baseline", "pooled"):
        command = [
            sys.executable,
            os.fspath(Path(__file__).resolve()),
            "--output-dir",
            os.fspath(args.output_dir.resolve()),
            "--num-resets",
            str(args.num_resets),
            "--seed",
            str(args.seed),
            "--texture-size",
            str(args.texture_size),
            "--case",
            case,
        ]
        subprocess.run(command, check=True, env=child_environment)

    baseline = json.loads(
        (args.output_dir / "baseline" / "case_metrics.json").read_text()
    )
    pooled = json.loads((args.output_dir / "pooled" / "case_metrics.json").read_text())
    cpu_extra = (
        pooled["cpu_rss_after_resets_bytes"] - baseline["cpu_rss_after_resets_bytes"]
    )
    gpu_extra = None
    if (
        pooled["gpu_pid_used_after_bytes"] is not None
        and baseline["gpu_pid_used_after_bytes"] is not None
    ):
        gpu_extra = (
            pooled["gpu_pid_used_after_bytes"] - baseline["gpu_pid_used_after_bytes"]
        )
    pooled_pair_paths = [
        args.output_dir / "pooled" / record["input_pair"]
        for record in pooled["model_inputs"]
    ]
    pair_images = [Image.open(path).convert("RGB") for path in pooled_pair_paths]
    contact_sheet = Image.new(
        "RGB",
        (896, ((len(pair_images) + 1) // 2) * 252),
        "white",
    )
    for index, pair_image in enumerate(pair_images):
        contact_sheet.paste(pair_image, ((index % 2) * 448, (index // 2) * 252))
    contact_sheet_path = args.output_dir / "pooled_model_inputs.png"
    contact_sheet.save(contact_sheet_path)
    for pair_image in pair_images:
        pair_image.close()
    comparison = {
        "comparison": "4-object/4-texture pool minus 1-object/1-texture pool",
        "extra_preloaded_objects": 3,
        "extra_preloaded_textures": 3,
        "cpu_rss_extra_bytes": cpu_extra,
        "cpu_rss_extra_mib": _format_mib(cpu_extra),
        "gpu_pid_extra_bytes": gpu_extra,
        "gpu_pid_extra_mib": _format_mib(gpu_extra),
        "pooled_model_input_contact_sheet": contact_sheet_path.name,
        "baseline": baseline,
        "pooled": pooled,
    }
    comparison_path = args.output_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


def main() -> None:
    """Run one child case or orchestrate the full benchmark."""
    args = parse_args()
    if args.num_resets < 1:
        raise ValueError("--num-resets must be positive")
    if args.case is None:
        run_parent(args)
    else:
        run_case(args)


if __name__ == "__main__":
    main()
