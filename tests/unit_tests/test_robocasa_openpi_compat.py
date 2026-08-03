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

import pathlib

import numpy as np
import pytest

pytest.importorskip("openpi")

import openpi.models.pi0_config as pi0_config
import openpi.transforms as transforms
from omegaconf import OmegaConf

import rlinf.models.embodiment.openpi.dataconfig.robocasa_dataconfig as robocasa_data
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.models.embodiment.openpi.policies.robocasa_policy import extract_state_dict
from rlinf.utils.ckpt_convertor.convert_openpi_jax_to_python import (
    _resolve_assets_source,
)


def test_extract_state_dict_accepts_already_compact_16d_state() -> None:
    state = np.arange(16, dtype=np.float32)

    result = extract_state_dict({"observation/state": state}, "16d")

    np.testing.assert_array_equal(result["state"], state)


def test_extract_state_dict_selects_16d_state_from_canonical_25d() -> None:
    state = np.arange(25, dtype=np.float32)

    result = extract_state_dict({"observation/state": state}, "16d")

    expected = np.array(
        [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 7, 8],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(result["state"], expected)


@pytest.mark.parametrize(
    ("schema", "action_key", "image_key"),
    [
        ("public", "actions", "image_left"),
        (
            "unified",
            "action",
            "observation.images.robot0_agentview_left",
        ),
    ],
)
def test_robocasa_data_config_supports_dataset_schemas(
    monkeypatch,
    tmp_path: pathlib.Path,
    schema: str,
    action_key: str,
    image_key: str,
) -> None:
    monkeypatch.setattr(
        robocasa_data,
        "ModelTransformFactory",
        lambda: lambda _: transforms.Group(),
    )
    config = robocasa_data.LeRobotRobocasaDataConfig(
        repo_id="local/robocasa",
        dataset_schema=schema,
    )

    data_config = config.create(
        tmp_path,
        pi0_config.Pi0Config(pi05=True, action_horizon=5, discrete_state_input=False),
    )

    assert data_config.action_sequence_keys == (action_key,)
    assert data_config.repack_transforms.inputs[0].structure["observation/image"] == (
        image_key
    )


def test_pi05_robocasa_configs_are_registered() -> None:
    public = get_openpi_config("pi05_robocasa_human")
    unified = get_openpi_config("pi05_robocasa365_closedrawer")

    assert public.model.pi05 is True
    assert public.data.state_space == "16d"
    assert public.data.dataset_schema == "public"
    assert unified.model.pi05 is True
    assert unified.data.state_space == "16d"
    assert unified.data.dataset_schema == "unified"


def test_openpi_data_overrides_select_unified_robocasa_schema() -> None:
    data_kwargs = OmegaConf.create(
        {
            "dataset_schema": "unified",
            "state_space": "16d",
            "image_space": "2views",
            "action_space": "12d",
        }
    )

    config = get_openpi_config(
        "pi05_robocasa_human",
        data_kwargs=data_kwargs,
    )

    assert config.data.dataset_schema == "unified"
    assert config.data.state_space == "16d"


def test_openpi_data_accepts_explicit_norm_stats_path(
    tmp_path: pathlib.Path,
) -> None:
    norm_stats_dir = tmp_path / "assets" / "train" / "robocasa"
    norm_stats_dir.mkdir(parents=True)
    norm_stats_path = norm_stats_dir / "norm_stats.json"
    norm_stats_path.touch()

    config = get_openpi_config(
        "pi05_robocasa_human",
        model_path=str(tmp_path / "checkpoint"),
        data_kwargs=OmegaConf.create(
            {
                "action_space": "12d",
                "state_space": "16d",
                "image_space": "2views",
                "norm_stats_path": str(norm_stats_path),
            }
        ),
    )

    assert config.data.assets.assets_dir == str(norm_stats_dir.parent)
    assert config.data.assets.asset_id == norm_stats_dir.name
    assert "norm_stats_path" not in config.data.__dataclass_fields__


def test_checkpoint_assets_can_be_stored_inside_checkpoint(
    tmp_path: pathlib.Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    nested_assets = checkpoint / "assets"
    nested_assets.mkdir(parents=True)

    assert _resolve_assets_source(checkpoint) == nested_assets


def test_checkpoint_assets_prefer_adjacent_layout(tmp_path: pathlib.Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    adjacent_assets = tmp_path / "assets"
    adjacent_assets.mkdir()
    (checkpoint / "assets").mkdir()

    assert _resolve_assets_source(checkpoint) == adjacent_assets
