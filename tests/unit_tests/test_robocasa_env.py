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

from __future__ import annotations

import builtins
import importlib
import sys
from types import ModuleType

import numpy as np
from omegaconf import OmegaConf


def test_get_env_fns_imports_robocasa_before_robosuite_make(
    monkeypatch,
) -> None:
    fake_gymnasium = ModuleType("gymnasium")
    fake_gymnasium.Env = object
    fake_imageio = ModuleType("imageio")
    fake_pil = ModuleType("PIL")
    fake_pil_image = ModuleType("PIL.Image")
    fake_pil_image_draw = ModuleType("PIL.ImageDraw")
    fake_pil_image_font = ModuleType("PIL.ImageFont")
    fake_venv = ModuleType("rlinf.envs.robocasa.venv")
    fake_venv.RobocasaSubprocEnv = object
    monkeypatch.delitem(
        sys.modules,
        "rlinf.envs.robocasa.robocasa_env",
        raising=False,
    )
    monkeypatch.setitem(sys.modules, "gymnasium", fake_gymnasium)
    monkeypatch.setitem(sys.modules, "imageio", fake_imageio)
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_pil_image)
    monkeypatch.setitem(sys.modules, "PIL.ImageDraw", fake_pil_image_draw)
    monkeypatch.setitem(sys.modules, "PIL.ImageFont", fake_pil_image_font)
    monkeypatch.setitem(sys.modules, "rlinf.envs.robocasa.venv", fake_venv)

    module = importlib.import_module("rlinf.envs.robocasa.robocasa_env")
    RobocasaEnv = module.RobocasaEnv

    env = RobocasaEnv.__new__(RobocasaEnv)
    env.cfg = OmegaConf.create(
        {
            "init_params": {
                "camera_widths": 224,
                "camera_heights": 224,
            },
            "robot_name": "PandaOmron",
            "image_space": ["observation/image"],
        }
    )
    env.num_envs = 1
    env.task_ids = np.array([0])
    env.task_names = ["CloseDrawer"]
    env.env_seeds = np.array([7])

    import_state = {"robocasa_imported": False}
    fake_env = object()

    fake_robosuite = ModuleType("robosuite")

    def _fake_make(**kwargs):
        assert import_state["robocasa_imported"] is True
        assert kwargs["env_name"] == "CloseDrawer"
        assert kwargs["robots"] == "PandaOmron"
        return fake_env

    fake_robosuite.make = _fake_make

    fake_controllers = ModuleType("robosuite.controllers")
    fake_controllers.load_composite_controller_config = lambda controller, robot: {
        "controller": controller,
        "robot": robot,
    }

    fake_robocasa = ModuleType("robocasa")

    monkeypatch.setitem(sys.modules, "robosuite", fake_robosuite)
    monkeypatch.setitem(sys.modules, "robosuite.controllers", fake_controllers)
    monkeypatch.setitem(sys.modules, "robocasa", fake_robocasa)

    original_import = builtins.__import__

    def _tracking_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "robocasa":
            import_state["robocasa_imported"] = True
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _tracking_import)

    env_fn = env.get_env_fns()[0]

    assert env_fn() is fake_env


def test_robocasa_env_get_mujoco_diagnostics_delegates_to_vector_env() -> None:
    module = importlib.import_module("rlinf.envs.robocasa.robocasa_env")
    RobocasaEnv = module.RobocasaEnv
    env = RobocasaEnv.__new__(RobocasaEnv)

    class _VectorEnv:
        def get_mujoco_diagnostics(self, max_contacts, include_model_names):
            return [{"max_contacts": max_contacts, "names": include_model_names}]

    env.env = _VectorEnv()

    assert env.get_mujoco_diagnostics(max_contacts=7, include_model_names=True) == [
        {"max_contacts": 7, "names": True}
    ]


def _robocasa_observation() -> dict[str, np.ndarray]:
    return {
        "robot0_agentview_left_image": np.zeros((2, 2, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.ones((2, 2, 3), dtype=np.uint8),
        "robot0_agentview_right_image": np.full((2, 2, 3), 2, dtype=np.uint8),
        "robot0_eef_pos": np.arange(0, 3, dtype=np.float32),
        "robot0_eef_quat": np.arange(3, 7, dtype=np.float32),
        "robot0_gripper_qpos": np.arange(7, 9, dtype=np.float32),
        "robot0_gripper_qvel": np.arange(9, 11, dtype=np.float32),
        "robot0_base_to_eef_pos": np.arange(11, 14, dtype=np.float32),
        "robot0_base_to_eef_quat": np.arange(14, 18, dtype=np.float32),
        "robot0_base_pos": np.arange(18, 21, dtype=np.float32),
        "robot0_base_quat": np.arange(21, 25, dtype=np.float32),
    }


def test_extract_image_and_state_selects_configured_16d_state() -> None:
    module = importlib.import_module("rlinf.envs.robocasa.robocasa_env")
    env = module.RobocasaEnv.__new__(module.RobocasaEnv)
    env.cfg = OmegaConf.create({"state_space": "16d"})

    result = env._extract_image_and_state([_robocasa_observation()])

    expected = np.array(
        [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 7, 8],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(result["state"][0], expected)


def test_extract_image_and_state_preserves_configured_25d_state() -> None:
    module = importlib.import_module("rlinf.envs.robocasa.robocasa_env")
    env = module.RobocasaEnv.__new__(module.RobocasaEnv)
    env.cfg = OmegaConf.create({"state_space": "25d"})

    result = env._extract_image_and_state([_robocasa_observation()])

    np.testing.assert_array_equal(result["state"][0], np.arange(25, dtype=np.float32))
