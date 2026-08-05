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

"""Integration-style unit test for LIBERO reset equivalence.

Run this test explicitly on a machine with LIBERO and MuJoCo rendering available:

.. code-block:: bash

    RLINF_RUN_LIBERO_RESET_EQUIVALENCE=1 \
      pytest -q tests/unit_tests/test_libero_reset_equivalence.py
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pytest

_RUN_EQUIVALENCE_TEST = os.environ.get("RLINF_RUN_LIBERO_RESET_EQUIVALENCE", "0") == "1"


@dataclass(frozen=True)
class _SceneSnapshot:
    """Observable and structural state captured after reset settling."""

    sim_state: np.ndarray
    images: dict[str, np.ndarray]
    model_xml: str
    model_shape: tuple[int, int, int, int]
    body_names: tuple[str, ...]
    geom_names: tuple[str, ...]
    object_names: tuple[str, ...]
    objects_of_interest: tuple[str, ...]


def _settle(env: Any, num_steps: int = 15) -> dict[str, np.ndarray]:
    """Match RLinf's post-reset settling loop and return its last observation."""
    observation = None
    for _ in range(num_steps):
        action = np.zeros(7, dtype=np.float64)
        action[-1] = -1.0
        observation, _, _, _ = env.step(action)
    assert observation is not None
    return observation


def _set_init_state_and_settle(
    env: Any, init_state: np.ndarray
) -> dict[str, np.ndarray]:
    """Install an official LIBERO state, then apply RLinf's settling steps."""
    env.set_init_state(init_state)
    return _settle(env)


def _accelerated_reset(env: Any) -> None:
    """Exercise the same temporary hard-reset override used by RLinf workers."""
    robosuite_env = env.env
    old_hard_reset = robosuite_env.hard_reset
    robosuite_env.hard_reset = False
    try:
        env.reset()
    finally:
        robosuite_env.hard_reset = old_hard_reset


def _snapshot(env: Any, observation: dict[str, np.ndarray]) -> _SceneSnapshot:
    """Capture enough state to detect physical or rendered scene differences."""
    model = env.sim.model
    images = {
        key: np.asarray(value).copy()
        for key, value in observation.items()
        if key.endswith("_image")
    }
    assert images, "LIBERO observation did not contain any camera images"
    return _SceneSnapshot(
        sim_state=np.asarray(env.get_sim_state()).copy(),
        images=images,
        model_xml=env.env.model.get_xml(),
        model_shape=(model.nq, model.nv, model.nbody, model.ngeom),
        body_names=tuple(model.body_names),
        geom_names=tuple(model.geom_names),
        object_names=tuple(sorted(env.env.objects_dict)),
        objects_of_interest=tuple(env.obj_of_interest),
    )


def _assert_same_scene(actual: _SceneSnapshot, expected: _SceneSnapshot) -> None:
    """Assert exact physical, structural, semantic, and rendered equivalence."""
    assert actual.model_xml == expected.model_xml
    assert actual.model_shape == expected.model_shape
    assert actual.body_names == expected.body_names
    assert actual.geom_names == expected.geom_names
    assert actual.object_names == expected.object_names
    assert actual.objects_of_interest == expected.objects_of_interest
    np.testing.assert_array_equal(actual.sim_state, expected.sim_state)
    assert actual.images.keys() == expected.images.keys()
    for key in actual.images:
        np.testing.assert_array_equal(actual.images[key], expected.images[key])


def _write_comparison_images(
    output_dir: Path,
    baseline: _SceneSnapshot,
    accelerated: _SceneSnapshot,
    hard: _SceneSnapshot,
) -> list[Path]:
    """Write individual reset images, a montage, and an exact pixel diff."""
    from PIL import Image, ImageDraw

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots = {
        "baseline": baseline,
        "accelerated": accelerated,
        "hard": hard,
    }
    output_paths = []
    for camera_key in baseline.images:
        policy_images = {
            label: np.flip(snapshot.images[camera_key], axis=(0, 1)).copy()
            for label, snapshot in snapshots.items()
        }
        for label, image in policy_images.items():
            output_path = output_dir / f"{label}_{camera_key}.png"
            Image.fromarray(image).save(output_path)
            output_paths.append(output_path)

        accelerated_image = policy_images["accelerated"]
        hard_image = policy_images["hard"]
        pixel_diff = np.abs(
            accelerated_image.astype(np.int16) - hard_image.astype(np.int16)
        ).astype(np.uint8)
        diff_path = output_dir / f"accelerated_vs_hard_diff_{camera_key}.png"
        Image.fromarray(pixel_diff).save(diff_path)
        output_paths.append(diff_path)

        height, width = accelerated_image.shape[:2]
        header_height = 24
        canvas = Image.new("RGB", (width * 4, height + header_height), "white")
        draw = ImageDraw.Draw(canvas)
        panels = [
            ("baseline", policy_images["baseline"]),
            ("accelerated", accelerated_image),
            ("hard", hard_image),
            (f"abs diff (max={int(pixel_diff.max())})", pixel_diff),
        ]
        for panel_index, (label, image) in enumerate(panels):
            x_offset = panel_index * width
            draw.text((x_offset + 4, 6), label, fill="black")
            canvas.paste(Image.fromarray(image), (x_offset, header_height))
        montage_path = output_dir / f"comparison_{camera_key}.png"
        canvas.save(montage_path)
        output_paths.append(montage_path)

        clear_canvas = Image.new("RGB", (width * 3, height + header_height), "white")
        clear_draw = ImageDraw.Draw(clear_canvas)
        clear_panels = [
            ("HARD RESET", hard_image),
            ("OPTIMIZED RESET", accelerated_image),
            (f"PIXEL DIFF (max={int(pixel_diff.max())})", pixel_diff),
        ]
        for panel_index, (label, image) in enumerate(clear_panels):
            x_offset = panel_index * width
            clear_draw.text((x_offset + 4, 6), label, fill="black")
            clear_canvas.paste(Image.fromarray(image), (x_offset, header_height))
        clear_path = output_dir / f"hard_vs_optimized_{camera_key}.png"
        clear_canvas.save(clear_path)
        output_paths.append(clear_path)

    return output_paths


def _observation_image(observation: dict[str, Any], key: str) -> np.ndarray:
    """Convert the first RLinf vector observation image to a NumPy array."""
    image = observation[key][0]
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    return np.asarray(image).copy()


def _write_rollout_reset_images(
    output_dir: Path,
    before_images: dict[str, np.ndarray],
    after_images: dict[str, np.ndarray],
    before_trial_id: int,
    after_trial_id: int,
) -> dict[str, dict[str, int]]:
    """Write the actual consecutive-rollout views and their pixel differences."""
    from PIL import Image, ImageDraw

    output_dir.mkdir(parents=True, exist_ok=True)
    difference_stats = {}
    for camera_name in before_images:
        before = before_images[camera_name]
        after = after_images[camera_name]
        difference = np.abs(before.astype(np.int16) - after.astype(np.int16)).astype(
            np.uint8
        )
        difference_stats[camera_name] = {
            "max": int(difference.max()),
            "nonzero_values": int(np.count_nonzero(difference)),
        }

        Image.fromarray(before).save(
            output_dir / f"rlinf_before_trial_{before_trial_id}_{camera_name}.png"
        )
        Image.fromarray(after).save(
            output_dir / f"rlinf_after_trial_{after_trial_id}_{camera_name}.png"
        )
        Image.fromarray(difference).save(
            output_dir
            / f"rlinf_trial_{before_trial_id}_vs_{after_trial_id}_diff_{camera_name}.png"
        )

        height, width = before.shape[:2]
        header_height = 24
        canvas = Image.new("RGB", (width * 3, height + header_height), "white")
        draw = ImageDraw.Draw(canvas)
        panels = [
            (f"before: trial {before_trial_id}", before),
            (f"after: trial {after_trial_id}", after),
            (f"abs diff (max={int(difference.max())})", difference),
        ]
        for panel_index, (label, image) in enumerate(panels):
            x_offset = panel_index * width
            draw.text((x_offset + 4, 6), label, fill="black")
            canvas.paste(Image.fromarray(image), (x_offset, header_height))
        canvas.save(output_dir / f"rlinf_rollout_reset_{camera_name}.png")

    return difference_stats


def _write_reset_round_images(
    output_dir: Path,
    round_index: int,
    trial_id: int,
    hard: _SceneSnapshot,
    optimized: _SceneSnapshot,
) -> tuple[dict[str, int], list[Path]]:
    """Write one clearly labelled hard-versus-optimized reset round."""
    from PIL import Image, ImageDraw

    output_dir.mkdir(parents=True, exist_ok=True)
    difference_stats = {}
    output_paths = []
    for camera_key in hard.images:
        hard_image = np.flip(hard.images[camera_key], axis=(0, 1)).copy()
        optimized_image = np.flip(optimized.images[camera_key], axis=(0, 1)).copy()
        pixel_diff = np.abs(
            hard_image.astype(np.int16) - optimized_image.astype(np.int16)
        ).astype(np.uint8)
        difference_stats[camera_key] = int(pixel_diff.max())

        height, width = hard_image.shape[:2]
        header_height = 24
        canvas = Image.new("RGB", (width * 3, height + header_height), "white")
        draw = ImageDraw.Draw(canvas)
        panels = [
            (f"ROUND {round_index}: HARD", hard_image),
            (f"ROUND {round_index}: OPTIMIZED", optimized_image),
            (f"PIXEL DIFF (max={int(pixel_diff.max())})", pixel_diff),
        ]
        for panel_index, (label, image) in enumerate(panels):
            x_offset = panel_index * width
            draw.text((x_offset + 4, 6), label, fill="black")
            canvas.paste(Image.fromarray(image), (x_offset, header_height))

        output_path = (
            output_dir
            / f"round_{round_index}_trial_{trial_id}_hard_vs_optimized_{camera_key}.png"
        )
        canvas.save(output_path)
        output_paths.append(output_path)

    return difference_stats, output_paths


def _write_two_reset_sequence_images(
    output_dir: Path,
    mode: str,
    reset_records: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    """Write reset 1, reset 2, and their difference for one reset mode."""
    from PIL import Image, ImageDraw

    output_dir.mkdir(parents=True, exist_ok=True)
    difference_stats = {}
    for camera_name in reset_records[0]["images"]:
        first = reset_records[0]["images"][camera_name]
        second = reset_records[1]["images"][camera_name]
        difference = np.abs(first.astype(np.int16) - second.astype(np.int16)).astype(
            np.uint8
        )
        difference_stats[camera_name] = {
            "max": int(difference.max()),
            "nonzero_values": int(np.count_nonzero(difference)),
        }

        height, width = first.shape[:2]
        header_height = 24
        canvas = Image.new("RGB", (width * 3, height + header_height), "white")
        draw = ImageDraw.Draw(canvas)
        panels = [
            (
                f"{mode}: RESET 1 (trial {reset_records[0]['trial_id']})",
                first,
            ),
            (
                f"{mode}: RESET 2 (trial {reset_records[1]['trial_id']})",
                second,
            ),
            (f"RESET 1 vs 2 (max={int(difference.max())})", difference),
        ]
        for panel_index, (label, image) in enumerate(panels):
            x_offset = panel_index * width
            draw.text((x_offset + 4, 6), label, fill="black")
            canvas.paste(Image.fromarray(image), (x_offset, header_height))
        canvas.save(output_dir / f"{mode.lower()}_two_resets_{camera_name}.png")

    return difference_stats


def _write_hard_fast_reset_pairs(
    output_dir: Path,
    hard_records: list[dict[str, Any]],
    optimized_records: list[dict[str, Any]],
) -> dict[str, list[int]]:
    """Write a two-row hard-versus-optimized comparison for both resets."""
    from PIL import Image, ImageDraw

    cross_mode_diffs = {}
    for camera_name in hard_records[0]["images"]:
        hard_images = [record["images"][camera_name] for record in hard_records]
        optimized_images = [
            record["images"][camera_name] for record in optimized_records
        ]
        differences = [
            np.abs(hard.astype(np.int16) - optimized.astype(np.int16)).astype(np.uint8)
            for hard, optimized in zip(hard_images, optimized_images, strict=True)
        ]
        cross_mode_diffs[camera_name] = [
            int(difference.max()) for difference in differences
        ]

        height, width = hard_images[0].shape[:2]
        header_height = 24
        canvas = Image.new("RGB", (width * 3, (height + header_height) * 2), "white")
        draw = ImageDraw.Draw(canvas)
        for reset_index in range(2):
            y_offset = reset_index * (height + header_height)
            panels = [
                (f"RESET {reset_index + 1}: HARD", hard_images[reset_index]),
                (
                    f"RESET {reset_index + 1}: OPTIMIZED",
                    optimized_images[reset_index],
                ),
                (
                    f"HARD vs OPT (max={int(differences[reset_index].max())})",
                    differences[reset_index],
                ),
            ]
            for panel_index, (label, image) in enumerate(panels):
                x_offset = panel_index * width
                draw.text((x_offset + 4, y_offset + 6), label, fill="black")
                canvas.paste(
                    Image.fromarray(image),
                    (x_offset, y_offset + header_height),
                )
        canvas.save(output_dir / f"hard_vs_optimized_both_resets_{camera_name}.png")

    return cross_mode_diffs


@pytest.mark.skipif(
    not _RUN_EQUIVALENCE_TEST,
    reason=(
        "requires a real LIBERO installation and MuJoCo renderer; set "
        "RLINF_RUN_LIBERO_RESET_EQUIVALENCE=1 to run"
    ),
)
def test_accelerated_reset_matches_hard_reset_for_same_init_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Fast and hard reset must produce the same scene for one fixed trial."""
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(tmp_path / "libero_config"))
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg_cache"))
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")
    monkeypatch.setenv("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))

    pytest.importorskip("libero.libero")
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_suite = benchmark.get_benchmark_dict()["libero_object"]()
    task = task_suite.get_task(0)
    bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )
    init_state = task_suite.get_task_init_states(0)[0]

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=128,
        camera_widths=128,
        hard_reset=True,
    )
    try:
        env.seed(17)

        env.reset()
        baseline_observation = _set_init_state_and_settle(env, init_state)
        baseline = _snapshot(env, baseline_observation)

        sim_before_fast_reset = env.sim
        raw_model_before_fast_reset = env.sim.model._model
        optimized_started_at = perf_counter()
        _accelerated_reset(env)
        fast_observation = _set_init_state_and_settle(env, init_state)
        optimized_seconds = perf_counter() - optimized_started_at
        accelerated = _snapshot(env, fast_observation)

        assert env.sim is sim_before_fast_reset
        assert env.sim.model._model is raw_model_before_fast_reset

        sim_before_hard_reset = env.sim
        raw_model_before_hard_reset = env.sim.model._model
        env.env.hard_reset = True
        hard_started_at = perf_counter()
        env.reset()
        hard_observation = _set_init_state_and_settle(env, init_state)
        hard_seconds = perf_counter() - hard_started_at
        hard = _snapshot(env, hard_observation)

        assert env.sim is not sim_before_hard_reset
        assert env.sim.model._model is not raw_model_before_hard_reset

        output_dir = Path(
            os.environ.get(
                "RLINF_LIBERO_RESET_IMAGE_DIR",
                str(tmp_path / "libero_reset_images"),
            )
        )
        output_paths = _write_comparison_images(
            output_dir,
            baseline,
            accelerated,
            hard,
        )
        print(f"LIBERO reset comparison images: {output_dir}")
        assert all(path.is_file() for path in output_paths)

        summary = {
            "task_id": 0,
            "init_state_id": 0,
            "optimized_reset_seconds": optimized_seconds,
            "hard_reset_seconds": hard_seconds,
            "optimized_reused_simulator": True,
            "hard_reset_recreated_simulator": True,
            "camera_pixel_max_diff": {
                key: int(
                    np.abs(
                        accelerated.images[key].astype(np.int16)
                        - hard.images[key].astype(np.int16)
                    ).max()
                )
                for key in accelerated.images
            },
            "sim_state_max_abs_diff": float(
                np.abs(accelerated.sim_state - hard.sim_state).max()
            ),
        }
        summary_path = output_dir / "hard_vs_optimized_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Hard vs optimized reset summary: {json.dumps(summary, indent=2)}")

        _assert_same_scene(accelerated, baseline)
        _assert_same_scene(hard, baseline)
    finally:
        env.close()


@pytest.mark.skipif(
    not _RUN_EQUIVALENCE_TEST,
    reason=(
        "requires a real LIBERO installation and MuJoCo renderer; set "
        "RLINF_RUN_LIBERO_RESET_EQUIVALENCE=1 to run"
    ),
)
def test_two_reset_rounds_compare_hard_and_optimized(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Compare hard and optimized reset across two different init states."""
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(tmp_path / "libero_config"))
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg_cache"))
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")
    monkeypatch.setenv("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))

    pytest.importorskip("libero.libero")
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_suite = benchmark.get_benchmark_dict()["libero_object"]()
    task = task_suite.get_task(0)
    bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )
    init_states = task_suite.get_task_init_states(0)
    output_dir = Path(
        os.environ.get(
            "RLINF_LIBERO_RESET_IMAGE_DIR",
            str(tmp_path / "libero_reset_images"),
        )
    )

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=128,
        camera_widths=128,
        hard_reset=True,
    )
    round_summaries = []
    try:
        env.seed(17)
        for round_index, trial_id in enumerate((0, 1), start=1):
            init_state = init_states[trial_id]

            sim_before_hard_reset = env.sim
            env.env.hard_reset = True
            hard_started_at = perf_counter()
            env.reset()
            hard_observation = _set_init_state_and_settle(env, init_state)
            hard_seconds = perf_counter() - hard_started_at
            hard = _snapshot(env, hard_observation)
            assert env.sim is not sim_before_hard_reset

            sim_before_optimized_reset = env.sim
            optimized_started_at = perf_counter()
            _accelerated_reset(env)
            optimized_observation = _set_init_state_and_settle(env, init_state)
            optimized_seconds = perf_counter() - optimized_started_at
            optimized = _snapshot(env, optimized_observation)
            assert env.sim is sim_before_optimized_reset

            _assert_same_scene(optimized, hard)
            camera_diffs, output_paths = _write_reset_round_images(
                output_dir,
                round_index,
                trial_id,
                hard,
                optimized,
            )
            assert all(path.is_file() for path in output_paths)
            round_summaries.append(
                {
                    "round": round_index,
                    "task_id": 0,
                    "init_state_id": trial_id,
                    "hard_reset_seconds": hard_seconds,
                    "optimized_reset_seconds": optimized_seconds,
                    "camera_pixel_max_diff": camera_diffs,
                    "sim_state_max_abs_diff": float(
                        np.abs(hard.sim_state - optimized.sim_state).max()
                    ),
                }
            )

        assert not np.array_equal(init_states[0], init_states[1])
        summary_path = output_dir / "two_reset_rounds_summary.json"
        summary_path.write_text(
            json.dumps({"rounds": round_summaries}, indent=2),
            encoding="utf-8",
        )
        print(f"Two reset rounds summary: {json.dumps(round_summaries, indent=2)}")
        print(f"Two reset rounds images: {output_dir}")
    finally:
        env.close()


@pytest.mark.skipif(
    not _RUN_EQUIVALENCE_TEST,
    reason=(
        "requires a real LIBERO installation and MuJoCo renderer; set "
        "RLINF_RUN_LIBERO_RESET_EQUIVALENCE=1 to run"
    ),
)
def test_rlinf_selected_reset_mode_runs_two_resets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Run two RLinf resets for one process-isolated reset mode."""
    reset_mode = os.environ.get("RLINF_LIBERO_RESET_COMPARISON_MODE", "").lower()
    if reset_mode not in {"hard", "optimized"}:
        pytest.skip("set RLINF_LIBERO_RESET_COMPARISON_MODE to 'hard' or 'optimized'")

    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(tmp_path / "libero_config"))
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg_cache"))
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")
    monkeypatch.setenv("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))

    pytest.importorskip("libero.libero")
    from omegaconf import OmegaConf

    from rlinf.envs.libero.libero_env import LiberoEnv

    repository_root = Path(__file__).parents[2]
    base_env_cfg = OmegaConf.load(
        repository_root / "examples/embodiment/config/env/libero_object.yaml"
    )
    experiment_cfg = OmegaConf.load(
        repository_root / "examples/embodiment/config/libero_object_ppo_gr00t.yaml"
    )
    common_cfg = OmegaConf.merge(base_env_cfg, experiment_cfg.env.train)
    common_cfg.total_num_envs = 1
    common_cfg.reward_coef = 1.0
    common_cfg.video_cfg.save_video = False
    common_cfg.init_params.camera_heights = 128
    common_cfg.init_params.camera_widths = 128

    if reset_mode == "hard":
        env_cfg = OmegaConf.merge(
            common_cfg,
            {"reset_optimization_enabled": False},
        )
    else:
        env_cfg = OmegaConf.merge(
            common_cfg,
            {
                "reset_optimization_enabled": True,
                "reset_mode": "task_aware",
            },
        )

    env = LiberoEnv(
        cfg=env_cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    records = []
    try:
        for reset_index in range(2):
            if reset_index > 0:
                # Exact RLinf rollout-boundary behavior from finish_rollout().
                env.update_reset_state_ids()
            # Exact RLinf bootstrap behavior before env.reset().
            env.is_start = True
            observation, info = env.reset()
            records.append(
                {
                    "reset_index": reset_index + 1,
                    "task_id": int(env.task_ids[0]),
                    "trial_id": int(env.trial_ids[0]),
                    "reset_metrics": info["reset_metrics"],
                    "images": {
                        "agentview": _observation_image(observation, "main_images"),
                        "wristview": _observation_image(observation, "wrist_images"),
                    },
                }
            )
    finally:
        env.close()

    reset_ids = [(record["task_id"], record["trial_id"]) for record in records]
    assert reset_ids[0] != reset_ids[1]
    for record in records:
        metrics = record["reset_metrics"]
        if reset_mode == "hard":
            assert metrics["full_count"] == 1
            assert metrics["state_count"] == 0
        else:
            assert metrics["full_count"] == 0
            assert metrics["state_count"] == 1

    output_dir = Path(
        os.environ.get(
            "RLINF_LIBERO_RESET_IMAGE_DIR",
            str(tmp_path / "libero_reset_images"),
        )
    )
    sequence_diffs = _write_two_reset_sequence_images(
        output_dir,
        reset_mode.upper(),
        records,
    )
    summary = {
        "mode": reset_mode,
        "resets": [
            {key: value for key, value in record.items() if key != "images"}
            for record in records
        ],
        "reset_1_vs_2_pixel_difference": sequence_diffs,
    }
    summary_path = output_dir / f"rlinf_{reset_mode}_two_resets_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"RLinf {reset_mode} two-reset summary: {json.dumps(summary, indent=2)}")
    print(f"RLinf {reset_mode} two-reset images: {output_dir}")


@pytest.mark.skipif(
    not _RUN_EQUIVALENCE_TEST,
    reason=(
        "requires a real LIBERO installation and MuJoCo renderer; set "
        "RLINF_RUN_LIBERO_RESET_EQUIVALENCE=1 to run"
    ),
)
def test_rlinf_rollout_reset_keeps_task_and_resamples_init_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Follow RLinf's rollout boundary reset using the real experiment config."""
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(tmp_path / "libero_config"))
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg_cache"))
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")
    monkeypatch.setenv("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))

    pytest.importorskip("libero.libero")
    from omegaconf import OmegaConf

    from rlinf.envs.libero.libero_env import LiberoEnv

    repository_root = Path(__file__).parents[2]
    base_env_cfg = OmegaConf.load(
        repository_root / "examples/embodiment/config/env/libero_object.yaml"
    )
    experiment_cfg = OmegaConf.load(
        repository_root / "examples/embodiment/config/libero_object_ppo_gr00t.yaml"
    )
    env_cfg = OmegaConf.merge(base_env_cfg, experiment_cfg.env.train)
    env_cfg.total_num_envs = 1
    env_cfg.reward_coef = 1.0
    env_cfg.video_cfg.save_video = False
    env_cfg.init_params.camera_heights = 128
    env_cfg.init_params.camera_widths = 128

    assert env_cfg.reset_optimization_enabled is True
    assert env_cfg.reset_mode == "task_aware"
    assert env_cfg.reset_sampling_strategy == "task_affine_fixed"
    assert env_cfg.use_fixed_reset_state_ids is True
    assert env_cfg.use_ordered_reset_state_ids is False
    assert env_cfg.auto_reset is False

    env = LiberoEnv(
        cfg=env_cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    try:
        # RLinf bootstrap_step(): mark the vector env as starting, then reset
        # with the state IDs selected for this rollout.
        env.is_start = True
        before_observation, before_info = env.reset()
        before_task_id = int(env.task_ids[0])
        before_trial_id = int(env.trial_ids[0])

        # RLinf finish_rollout() selects the next rollout's IDs. The following
        # bootstrap_step() marks is_start again and consumes those IDs.
        env.update_reset_state_ids()
        env.is_start = True
        after_observation, after_info = env.reset()
        after_task_id = int(env.task_ids[0])
        after_trial_id = int(env.trial_ids[0])

        before_images = {
            "agentview": _observation_image(before_observation, "main_images"),
            "wristview": _observation_image(before_observation, "wrist_images"),
        }
        after_images = {
            "agentview": _observation_image(after_observation, "main_images"),
            "wristview": _observation_image(after_observation, "wrist_images"),
        }

        output_dir = Path(
            os.environ.get(
                "RLINF_LIBERO_RESET_IMAGE_DIR",
                str(tmp_path / "libero_reset_images"),
            )
        )
        difference_stats = _write_rollout_reset_images(
            output_dir,
            before_images,
            after_images,
            before_trial_id,
            after_trial_id,
        )
        summary = {
            "task_id_before": before_task_id,
            "task_id_after": after_task_id,
            "trial_id_before": before_trial_id,
            "trial_id_after": after_trial_id,
            "before_reset_metrics": before_info["reset_metrics"],
            "after_reset_metrics": after_info["reset_metrics"],
            "pixel_difference": difference_stats,
        }
        summary_path = output_dir / "rlinf_rollout_reset_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"RLinf rollout reset summary: {json.dumps(summary, indent=2)}")
        print(f"RLinf rollout reset images: {output_dir}")

        assert before_task_id == after_task_id
        assert before_trial_id != after_trial_id
        assert before_info["reset_metrics"]["state_count"] == 1
        assert before_info["reset_metrics"]["full_count"] == 0
        assert after_info["reset_metrics"]["state_count"] == 1
        assert after_info["reset_metrics"]["full_count"] == 0
        assert any(
            not np.array_equal(before_images[key], after_images[key])
            for key in before_images
        )
    finally:
        env.env.close()
