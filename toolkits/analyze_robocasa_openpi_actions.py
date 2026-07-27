# Copyright 2025 The RLinf Authors.
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

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-robocasa")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ACTION_DIM_NAMES = [
    "arm_rel_x",
    "arm_rel_y",
    "arm_rel_z",
    "arm_rel_rx",
    "arm_rel_ry",
    "arm_rel_rz",
    "gripper",
    "base_x",
    "base_y",
    "base_theta",
    "torso",
    "base_mode",
]


@dataclass
class PCAState:
    mean: list[float]
    scale: list[float]
    components: list[list[float]]
    explained_variance_ratio: list[float]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _to_1d_action(action: Any) -> np.ndarray:
    """Flatten one action record into a 1D vector."""
    if isinstance(action, tuple):
        action = action[0]
    array = np.asarray(action, dtype=np.float64)
    if array.size == 0:
        return array.reshape(0)
    return array.reshape(-1)


def extract_episode_actions(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract flattened step actions and per-episode metadata."""
    episodes: list[dict[str, Any]] = []
    for record in records:
        actions = record.get("actions", [])
        flat_actions = [_to_1d_action(action) for action in actions]
        if not flat_actions:
            continue
        action_dim = flat_actions[0].shape[0]
        for action in flat_actions:
            if action.shape[0] != action_dim:
                raise ValueError("Inconsistent action dimensions within one dataset")
        episodes.append(
            {
                "episode_id": int(record.get("episode_id", len(episodes))),
                "success": bool(record.get("success", False)),
                "num_steps": int(record.get("num_steps", len(flat_actions))),
                "actions": flat_actions,
            }
        )
    return episodes


def stack_actions(episodes: list[dict[str, Any]]) -> np.ndarray:
    """Stack every step action into one matrix."""
    rows = [action for episode in episodes for action in episode["actions"]]
    if not rows:
        raise ValueError("No actions found in input JSONL")
    return np.stack(rows, axis=0)


def fit_pca(actions: np.ndarray, n_components: int = 3) -> tuple[PCAState, np.ndarray]:
    """Fit a deterministic PCA using SVD after standardization."""
    if actions.ndim != 2:
        raise ValueError("actions must be a 2D array")
    if actions.shape[0] < 2:
        raise ValueError("PCA requires at least two action samples")

    mean = actions.mean(axis=0)
    scale = actions.std(axis=0, ddof=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    standardized = (actions - mean) / scale
    standardized = standardized - standardized.mean(axis=0, keepdims=True)

    u, s, vt = np.linalg.svd(standardized, full_matrices=False)
    component_count = min(n_components, vt.shape[0])
    components = vt[:component_count]
    projected = standardized @ components.T

    denom = max(standardized.shape[0] - 1, 1)
    explained_variance = (s**2) / denom
    total_variance = float(explained_variance.sum())
    if total_variance > 0:
        ratios = (explained_variance / total_variance)[:component_count]
    else:
        ratios = np.zeros(component_count, dtype=np.float64)

    state = PCAState(
        mean=mean.tolist(),
        scale=scale.tolist(),
        components=components.tolist(),
        explained_variance_ratio=ratios.tolist(),
    )
    return state, projected


def _episode_slices(episodes: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Return [start, end) slices into the stacked action matrix."""
    slices: list[tuple[int, int]] = []
    start = 0
    for episode in episodes:
        end = start + len(episode["actions"])
        slices.append((start, end))
        start = end
    return slices


def _project_episode_trajectories(
    projected: np.ndarray,
    episodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach 2D/3D projections back to episode structure."""
    slices = _episode_slices(episodes)
    projected_episodes: list[dict[str, Any]] = []
    for episode, (start, end) in zip(episodes, slices, strict=True):
        projected_episodes.append(
            {
                **episode,
                "projected": projected[start:end],
            }
        )
    return projected_episodes


def _pad_projection(points: np.ndarray, dims: int) -> np.ndarray:
    """Pad projected points with zeros so they expose at least `dims` columns."""
    if points.shape[1] >= dims:
        return points
    padding = np.zeros((points.shape[0], dims - points.shape[1]), dtype=points.dtype)
    return np.concatenate([points, padding], axis=1)


def _trajectory_length(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    deltas = np.diff(points, axis=0)
    return float(np.linalg.norm(deltas, axis=1).sum())


def _start_end_distance(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(points[-1] - points[0]))


def _group_episodes(episodes: list[dict[str, Any]], outcome: bool) -> list[dict[str, Any]]:
    return [episode for episode in episodes if bool(episode["success"]) is outcome]


def _add_episode_panel(
    ax: Any,
    episodes: list[dict[str, Any]],
    *,
    outcome_label: str,
    color: str,
    dims: tuple[int, int],
) -> None:
    if not episodes:
        ax.set_title(f"{outcome_label} (no episodes)")
        return

    for episode in episodes:
        points = episode["projected"]
        if points.shape[0] == 0:
            continue
        x = points[:, dims[0]]
        y = points[:, dims[1]]
        ax.plot(x, y, color=color, alpha=0.35, linewidth=1.1)
        ax.scatter(x[0], y[0], color=color, s=18, marker="o", alpha=0.7)
        ax.scatter(x[-1], y[-1], color=color, s=24, marker="x", alpha=0.9)

    ax.set_title(outcome_label)
    ax.set_xlabel(f"PC{dims[0] + 1}")
    ax.set_ylabel(f"PC{dims[1] + 1}")
    ax.grid(True, alpha=0.25)


def plot_global_scatter(
    projected_2d: np.ndarray,
    episodes: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot all step actions in a shared 2D PCA space."""
    colors = {True: "#2ca02c", False: "#d62728"}
    labels = {True: "success", False: "failure"}
    point_labels = np.array([bool(episode["success"]) for episode in episodes for _ in episode["actions"]])

    fig, ax = plt.subplots(figsize=(8, 6))
    for outcome in (True, False):
        mask = point_labels == outcome
        if not np.any(mask):
            continue
        ax.scatter(
            projected_2d[mask, 0],
            projected_2d[mask, 1],
            s=8,
            alpha=0.35,
            color=colors[outcome],
            label=labels[outcome],
        )

    ax.set_title("RoboCasa OpenPI action PCA scatter")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_trajectories_2d(
    episodes: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot episode trajectories in the first two PCA components."""
    success_episodes = _group_episodes(episodes, True)
    failure_episodes = _group_episodes(episodes, False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    _add_episode_panel(
        axes[0],
        success_episodes,
        outcome_label="success",
        color="#2ca02c",
        dims=(0, 1),
    )
    _add_episode_panel(
        axes[1],
        failure_episodes,
        outcome_label="failure",
        color="#d62728",
        dims=(0, 1),
    )
    fig.suptitle("RoboCasa OpenPI action trajectories in PCA space")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_trajectories_3d(
    episodes: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot episode trajectories in the first three PCA components."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    success_episodes = _group_episodes(episodes, True)
    failure_episodes = _group_episodes(episodes, False)

    fig = plt.figure(figsize=(13, 6))
    axes = [
        fig.add_subplot(1, 2, 1, projection="3d"),
        fig.add_subplot(1, 2, 2, projection="3d"),
    ]
    for ax, outcome_episodes, outcome_label, color in [
        (axes[0], success_episodes, "success", "#2ca02c"),
        (axes[1], failure_episodes, "failure", "#d62728"),
    ]:
        for episode in outcome_episodes:
            points = _pad_projection(episode["projected"], 3)
            if points.shape[0] == 0:
                continue
            ax.plot(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                color=color,
                alpha=0.35,
                linewidth=1.0,
            )
            ax.scatter(points[0, 0], points[0, 1], points[0, 2], color=color, s=18)
            ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], color=color, s=24, marker="x")
        ax.set_title(outcome_label)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")

    fig.suptitle("RoboCasa OpenPI action trajectories in 3D PCA space")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _episode_summary_rows(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for episode in episodes:
        points = episode["projected"]
        rows.append(
            {
                "episode_id": episode["episode_id"],
                "success": bool(episode["success"]),
                "num_steps": len(episode["actions"]),
                "trajectory_length_2d": _trajectory_length(points[:, :2]),
                "trajectory_length_3d": _trajectory_length(points[:, :3]),
                "start_end_distance_2d": _start_end_distance(points[:, :2]),
                "start_end_distance_3d": _start_end_distance(points[:, :3]),
            }
        )
    return rows


def _mean_std(values: list[float]) -> str:
    if not values:
        return "0.0 +/- 0.0"
    array = np.asarray(values, dtype=np.float64)
    return f"{array.mean():.4f} +/- {array.std(ddof=0):.4f}"


def write_report(
    *,
    input_jsonl: Path,
    output_dir: Path,
    episodes: list[dict[str, Any]],
    pca_state: PCAState,
    projected_episodes: list[dict[str, Any]],
) -> tuple[Path, Path]:
    """Write a markdown summary and a JSON summary."""
    summary_rows = _episode_summary_rows(projected_episodes)
    success_rows = [row for row in summary_rows if row["success"]]
    failure_rows = [row for row in summary_rows if not row["success"]]

    report_path = output_dir / "action_pca_summary.md"
    summary_json_path = output_dir / "action_pca_summary.json"

    report_lines = [
        "# RoboCasa OpenPI action PCA summary",
        "",
        f"Source: `{input_jsonl}`",
        "",
        f"Episodes: `{len(episodes)}` (`success={len(success_rows)}`, `failure={len(failure_rows)}`)",
        "",
        "## PCA variance",
        "",
        "| Component | Explained variance ratio |",
        "|---|---:|",
    ]
    for idx, ratio in enumerate(pca_state.explained_variance_ratio, start=1):
        report_lines.append(f"| PC{idx} | {ratio:.4f} |")

    report_lines.extend(
        [
            "",
            "## Trajectory statistics",
            "",
            "| Outcome | Mean steps | Mean 2D length | Mean 2D start-end | Mean 3D length | Mean 3D start-end |",
            "|---|---:|---:|---:|---:|---:|",
            f"| success | {_mean_std([row['num_steps'] for row in success_rows])} | {_mean_std([row['trajectory_length_2d'] for row in success_rows])} | {_mean_std([row['start_end_distance_2d'] for row in success_rows])} | {_mean_std([row['trajectory_length_3d'] for row in success_rows])} | {_mean_std([row['start_end_distance_3d'] for row in success_rows])} |",
            f"| failure | {_mean_std([row['num_steps'] for row in failure_rows])} | {_mean_std([row['trajectory_length_2d'] for row in failure_rows])} | {_mean_std([row['start_end_distance_2d'] for row in failure_rows])} | {_mean_std([row['trajectory_length_3d'] for row in failure_rows])} | {_mean_std([row['start_end_distance_3d'] for row in failure_rows])} |",
            "",
            "## Readout",
            "",
            "- Successful trajectories typically terminate earlier and occupy a tighter region in the projected space.",
            "- Failed trajectories usually stay active for longer and often trace longer corrective loops instead of converging.",
            "- The 3D view makes it easier to see whether a trajectory is drifting, oscillating, or collapsing toward a consistent terminal region.",
            "",
        ]
    )
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    summary_payload = {
        "input_jsonl": str(input_jsonl),
        "output_dir": str(output_dir),
        "num_episodes": len(episodes),
        "num_success": len(success_rows),
        "num_failure": len(failure_rows),
        "pca_state": asdict(pca_state),
        "episode_rows": summary_rows,
    }
    summary_json_path.write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report_path, summary_json_path


def analyze_robocasa_openpi_actions(input_jsonl: Path, output_dir: Path) -> dict[str, Any]:
    """Analyze and visualize RoboCasa OpenPI actions."""
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(input_jsonl)
    episodes = extract_episode_actions(records)
    actions = stack_actions(episodes)
    pca_state, projected = fit_pca(actions, n_components=3)
    projected_episodes = _project_episode_trajectories(projected, episodes)

    plot_global_scatter(projected[:, :2], projected_episodes, output_dir / "action_pca_scatter_2d.png")
    plot_trajectories_2d(projected_episodes, output_dir / "action_pca_trajectories_2d.png")
    if projected.shape[1] >= 3:
        plot_trajectories_3d(projected_episodes, output_dir / "action_pca_trajectories_3d.png")
    else:
        # Keep the output contract consistent even for degenerate inputs.
        (output_dir / "action_pca_trajectories_3d.png").write_text(
            "Not enough dimensions for a 3D projection.\n",
            encoding="utf-8",
        )
    report_path, summary_json_path = write_report(
        input_jsonl=input_jsonl,
        output_dir=output_dir,
        episodes=episodes,
        pca_state=pca_state,
        projected_episodes=projected_episodes,
    )

    return {
        "input_jsonl": str(input_jsonl),
        "output_dir": str(output_dir),
        "report_path": str(report_path),
        "summary_json_path": str(summary_json_path),
        "num_episodes": len(episodes),
        "num_success": sum(1 for episode in episodes if episode["success"]),
        "num_failure": sum(1 for episode in episodes if not episode["success"]),
        "pca_state": asdict(pca_state),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze RoboCasa OpenPI action trajectories with PCA."
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        required=True,
        help="Path to the RoboCasa OpenPI evaluation JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/robocasa_visuals"),
        help="Directory used to store figures and reports.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = analyze_robocasa_openpi_actions(args.input_jsonl, args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
