from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from toolkits.analyze_robocasa_openpi_actions import (
    _to_1d_action,
    analyze_robocasa_openpi_actions,
    fit_pca,
    read_jsonl,
)


def test_to_1d_action_flattens_nested_actions() -> None:
    action = [[[1.0, 2.0, 3.0]]]

    flattened = _to_1d_action(action)

    np.testing.assert_allclose(flattened, np.array([1.0, 2.0, 3.0]))


def test_fit_pca_returns_deterministic_projection_shape() -> None:
    actions = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.5, 2.0],
            [2.0, 1.0, 3.0],
            [3.0, 1.5, 4.0],
        ],
        dtype=np.float64,
    )

    state, projected = fit_pca(actions, n_components=3)

    assert projected.shape == (4, 3)
    assert len(state.explained_variance_ratio) == 3
    assert np.isfinite(projected).all()
    assert np.isclose(sum(state.explained_variance_ratio), 1.0)


def test_analyze_robocasa_openpi_actions_writes_outputs(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))

    input_jsonl = tmp_path / "robocasa.jsonl"
    output_dir = tmp_path / "out"
    records = [
        {
            "episode_id": 0,
            "success": True,
            "num_steps": 2,
            "actions": [
                [[[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0]]],
                [[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.9, 0.0, 0.0, 0.0, 0.0, -1.0]]],
            ],
        },
        {
            "episode_id": 1,
            "success": False,
            "num_steps": 2,
            "actions": [
                [[[1.0, 1.1, 1.2, 1.3, 1.4, 1.5, -1.0, 0.1, 0.0, 0.0, 0.0, -1.0]]],
                [[[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, -0.8, 0.1, 0.0, 0.0, 0.0, -1.0]]],
            ],
        },
    ]
    input_jsonl.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")

    result = analyze_robocasa_openpi_actions(input_jsonl, output_dir)
    parsed = read_jsonl(input_jsonl)

    assert len(parsed) == 2
    assert result["num_episodes"] == 2
    assert result["num_success"] == 1
    assert result["num_failure"] == 1
    assert (output_dir / "action_pca_scatter_2d.png").exists()
    assert (output_dir / "action_pca_trajectories_2d.png").exists()
    assert (output_dir / "action_pca_trajectories_3d.png").exists()
    assert (output_dir / "action_pca_summary.md").exists()
    assert (output_dir / "action_pca_summary.json").exists()

    summary = json.loads((output_dir / "action_pca_summary.json").read_text(encoding="utf-8"))
    assert summary["num_episodes"] == 2
    assert summary["num_success"] == 1
    assert summary["num_failure"] == 1
    assert len(summary["episode_rows"]) == 2
