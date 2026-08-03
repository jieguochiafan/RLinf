#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluate an RLinf OpenPI policy with RoboCasa's official environment."""

import argparse
import csv
import json
import os
import pathlib
import sys
import time

import imageio.v2 as imageio
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    """Parse standalone RoboCasa evaluation arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-name", default="robocasa_closedrawer_ppo_openpi_pi05")
    parser.add_argument("--task", default="CloseDrawer")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument(
        "--smoke", action="store_true", help="Run one action chunk only."
    )
    parser.add_argument(
        "--env-only",
        action="store_true",
        help="Create and reset the official environment, then exit.",
    )
    return parser.parse_args()


def load_model(args: argparse.Namespace):
    """Load an RLinf OpenPI model from a composed embodied config."""
    from rlinf.models.embodiment.openpi import get_model

    config_dir = str(REPO_ROOT / "examples" / "embodiment" / "config")
    with initialize_config_dir(version_base="1.1", config_dir=config_dir):
        cfg = compose(config_name=args.config_name)
    model_cfg = cfg.actor.model
    with open_dict(model_cfg):
        model_cfg.model_path = args.model_path
    model = get_model(model_cfg)
    model.eval().to(args.device)
    return model, OmegaConf.to_container(model_cfg, resolve=True)


def create_official_env(task: str, seed: int):
    """Create an environment using RoboCasa's public factory."""
    from robocasa.utils.env_utils import create_env

    return create_env(
        env_name=task,
        robots="PandaOmron",
        camera_names=["robot0_agentview_left", "robot0_eye_in_hand"],
        camera_widths=224,
        camera_heights=224,
        seed=seed,
        render_onscreen=False,
    )


def extract_policy_observation(obs: dict, prompt: str) -> dict:
    """Convert an official RoboCasa observation to RLinf's 16D format."""
    state = np.concatenate(
        [
            obs["robot0_base_to_eef_pos"],
            obs["robot0_base_to_eef_quat"],
            obs["robot0_base_pos"],
            obs["robot0_base_quat"],
            obs["robot0_gripper_qpos"],
        ]
    ).astype(np.float32)
    if state.shape != (16,):
        raise ValueError(f"Expected a 16D RoboCasa state, got {state.shape}.")
    return {
        "main_images": torch.from_numpy(
            np.ascontiguousarray(obs["robot0_agentview_left_image"][::-1][None])
        ),
        "wrist_images": torch.from_numpy(
            np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1][None])
        ),
        "extra_view_images": None,
        "states": torch.from_numpy(state[None]),
        "task_descriptions": [prompt],
    }


def write_results(output_dir: pathlib.Path, rows: list[dict], metadata: dict) -> None:
    """Persist incremental CSV results and a JSON summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "episodes.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        **metadata,
        "episodes_completed": len(rows),
        "successes": sum(int(row["success"]) for row in rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "episodes": rows,
    }
    (output_dir / "results.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    """Run standalone official-environment evaluation."""
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.env_only:
        import mujoco
        import robocasa

        env = create_official_env(args.task, args.seed)
        try:
            obs = env.reset()
            print(
                json.dumps(
                    {
                        "robocasa_path": robocasa.__file__,
                        "mujoco_version": mujoco.__version__,
                        "numpy_version": np.__version__,
                        "task": type(env).__name__,
                        "prompt": env.get_ep_meta()["lang"],
                        "image_shape": list(obs["robot0_agentview_left_image"].shape),
                        "initial_success": bool(env._check_success()),
                        "objects_path": os.environ.get("ROBOCASA_OBJECTS_PATH"),
                    }
                ),
                flush=True,
            )
        finally:
            env.close()
        return
    if not args.model_path:
        raise ValueError("--model-path is required unless --env-only is set")

    model, model_cfg = load_model(args)
    metadata = {
        "evaluator": "official robocasa.utils.env_utils.create_env",
        "success_check": "env._check_success()",
        "task": args.task,
        "model_path": os.path.realpath(args.model_path),
        "seed": args.seed,
        "max_steps": args.max_steps,
        "model_config": model_cfg,
    }
    rows = []
    for episode in range(args.episodes):
        episode_seed = args.seed + episode
        env = create_official_env(args.task, episode_seed)
        writer = None
        try:
            obs = env.reset()
            prompt = env.get_ep_meta()["lang"]
            success = bool(env._check_success())
            actions_executed = 0
            inference_calls = 0
            action_abs_max = 0.0
            if args.save_video:
                writer = imageio.get_writer(
                    output_dir / f"episode_{episode:04d}_seed_{episode_seed}.mp4",
                    fps=20,
                )
                writer.append_data(obs["robot0_agentview_left_image"][::-1])
            start = time.monotonic()
            while actions_executed < args.max_steps and not success:
                policy_obs = extract_policy_observation(obs, prompt)
                with torch.inference_mode():
                    action_chunk, _ = model.predict_action_batch(
                        policy_obs, mode="eval", compute_values=False
                    )
                action_chunk = action_chunk.detach().float().cpu().numpy()[0]
                if action_chunk.shape[1] != 12 or not np.isfinite(action_chunk).all():
                    raise ValueError(
                        f"Invalid action chunk shape/values: {action_chunk.shape}, "
                        f"finite={np.isfinite(action_chunk).all()}"
                    )
                inference_calls += 1
                action_abs_max = max(action_abs_max, float(np.abs(action_chunk).max()))
                for action in action_chunk:
                    obs, _, _, _ = env.step(action)
                    actions_executed += 1
                    success = bool(env._check_success())
                    if writer is not None:
                        writer.append_data(obs["robot0_agentview_left_image"][::-1])
                    if success or actions_executed >= args.max_steps:
                        break
                if args.smoke:
                    break
            row = {
                "episode": episode,
                "seed": episode_seed,
                "prompt": prompt,
                "success": int(success),
                "steps": actions_executed,
                "inference_calls": inference_calls,
                "action_abs_max": action_abs_max,
                "wall_time_sec": round(time.monotonic() - start, 3),
            }
            rows.append(row)
            write_results(output_dir, rows, metadata)
            print(json.dumps(row), flush=True)
        finally:
            if writer is not None:
                writer.close()
            env.close()
    print(json.dumps({"success_rate": float(np.mean([r["success"] for r in rows]))}))


if __name__ == "__main__":
    main()
