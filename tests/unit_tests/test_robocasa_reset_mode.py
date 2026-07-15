from __future__ import annotations

import importlib
import sys
from types import ModuleType

import numpy as np
import pytest
from omegaconf import OmegaConf


@pytest.mark.parametrize(
    ("cfg_extra", "expected_hard_reset"),
    [
        ({"reset_mode": "full"}, True),
        ({"reset_mode": "state"}, True),
        ({"reset_mode": "state", "reset_optimization_enabled": True}, False),
        ({"reset_mode": "task_aware"}, True),
        ({"reset_mode": "task_aware", "reset_optimization_enabled": True}, False),
        ({"hard_reset": False}, False),
        ({}, True),
    ],
)
def test_robocasa_get_env_fns_maps_reset_mode_to_hard_reset(
    monkeypatch,
    cfg_extra,
    expected_hard_reset,
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
    monkeypatch.delitem(sys.modules, "rlinf.envs.robocasa.robocasa_env", raising=False)
    monkeypatch.setitem(sys.modules, "gymnasium", fake_gymnasium)
    monkeypatch.setitem(sys.modules, "imageio", fake_imageio)
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_pil_image)
    monkeypatch.setitem(sys.modules, "PIL.ImageDraw", fake_pil_image_draw)
    monkeypatch.setitem(sys.modules, "PIL.ImageFont", fake_pil_image_font)
    monkeypatch.setitem(sys.modules, "rlinf.envs.robocasa.venv", fake_venv)

    module = importlib.import_module("rlinf.envs.robocasa.robocasa_env")
    env = module.RobocasaEnv.__new__(module.RobocasaEnv)
    cfg_dict = {
        "init_params": {"camera_widths": 224, "camera_heights": 224},
        "robot_name": "PandaOmron",
        "image_space": ["observation/image"],
    }
    cfg_dict.update(cfg_extra)
    env.cfg = OmegaConf.create(cfg_dict)
    env.num_envs = 1
    env.task_ids = np.array([0])
    env.task_names = ["CloseDrawer"]
    env.env_seeds = np.array([7])

    captured_kwargs = {}
    fake_robosuite = ModuleType("robosuite")

    def _fake_make(**kwargs):
        captured_kwargs.update(kwargs)
        return object()

    fake_robosuite.make = _fake_make
    fake_controllers = ModuleType("robosuite.controllers")
    fake_controllers.load_composite_controller_config = (
        lambda controller, robot: {"controller": controller, "robot": robot}
    )
    monkeypatch.setitem(sys.modules, "robosuite", fake_robosuite)
    monkeypatch.setitem(sys.modules, "robosuite.controllers", fake_controllers)
    monkeypatch.setitem(sys.modules, "robocasa", ModuleType("robocasa"))

    env.get_env_fns()[0]()

    assert captured_kwargs["hard_reset"] is expected_hard_reset
