from __future__ import annotations

import importlib
import sys
from types import ModuleType

import numpy as np
import pytest
import torch
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


def test_robocasa_reset_wrapper_returns_same_obs_for_full_and_task_aware(
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

    make_calls = []

    class _FakeRobosuiteEnv:
        def __init__(self, hard_reset):
            self.hard_reset = hard_reset

        def reset(self):
            obs = {
                "robot0_agentview_left_image": np.array(
                    [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
                    dtype=np.uint8,
                ),
                "robot0_eye_in_hand_image": np.array(
                    [[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]],
                    dtype=np.uint8,
                ),
                "robot0_eef_pos": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "robot0_eef_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                "robot0_gripper_qpos": np.array([0.1, 0.2], dtype=np.float32),
                "robot0_gripper_qvel": np.array([0.3, 0.4], dtype=np.float32),
                "robot0_base_to_eef_pos": np.array(
                    [4.0, 5.0, 6.0], dtype=np.float32
                ),
                "robot0_base_to_eef_quat": np.array(
                    [1.0, 0.0, 0.0, 0.0], dtype=np.float32
                ),
                "robot0_base_pos": np.array([7.0, 8.0, 9.0], dtype=np.float32),
                "robot0_base_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            }
            info = {"ep_meta": {"lang": "CloseDrawer"}}
            return obs, info

    def _fake_make(**kwargs):
        make_calls.append(kwargs)
        return _FakeRobosuiteEnv(kwargs["hard_reset"])

    class _FakeVectorEnv:
        def __init__(self, env_fns):
            self.envs = [fn() for fn in env_fns]

        def reset(self, id=None):
            del id
            raw_obs_list = []
            info_list = []
            for env in self.envs:
                obs, info = env.reset()
                raw_obs_list.append(obs)
                info_list.append(info)
            return raw_obs_list, info_list

        def close(self):
            return None

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
    monkeypatch.setitem(sys.modules, "robocasa", ModuleType("robocasa"))

    fake_robosuite = ModuleType("robosuite")
    fake_robosuite.make = _fake_make
    fake_controllers = ModuleType("robosuite.controllers")
    fake_controllers.load_composite_controller_config = (
        lambda controller, robot: {"controller": controller, "robot": robot}
    )
    monkeypatch.setitem(sys.modules, "robosuite", fake_robosuite)
    monkeypatch.setitem(sys.modules, "robosuite.controllers", fake_controllers)
    fake_venv.RobocasaSubprocEnv = _FakeVectorEnv

    module = importlib.import_module("rlinf.envs.robocasa.robocasa_env")
    RobocasaEnv = module.RobocasaEnv

    base_cfg = {
        "init_params": {"camera_widths": 2, "camera_heights": 2},
        "robot_name": "PandaOmron",
        "image_space": "2views",
        "task_names": ["CloseDrawer"],
        "seed": 42,
        "group_size": 1,
        "ignore_terminations": False,
        "auto_reset": False,
        "use_rel_reward": False,
        "reward_coef": 1.0,
        "max_episode_steps": 100,
        "video_cfg": {
            "save_video": False,
            "info_on_video": True,
            "video_base_dir": "/tmp",
        },
    }

    env_full = RobocasaEnv(
        OmegaConf.create({**base_cfg, "reset_mode": "full"}),
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info={},
    )
    env_task_aware = RobocasaEnv(
        OmegaConf.create(
            {
                **base_cfg,
                "reset_mode": "task_aware",
                "reset_optimization_enabled": True,
            }
        ),
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info={},
    )

    obs_full, infos_full = env_full.reset()
    obs_task_aware, infos_task_aware = env_task_aware.reset()

    assert make_calls[0]["hard_reset"] is True
    assert make_calls[1]["hard_reset"] is False
    assert infos_full == infos_task_aware == {}
    assert obs_full["task_descriptions"] == obs_task_aware["task_descriptions"] == [
        "CloseDrawer"
    ]
    assert torch.equal(obs_full["states"], obs_task_aware["states"])
    assert torch.equal(obs_full["main_images"], obs_task_aware["main_images"])
    assert torch.equal(obs_full["wrist_images"], obs_task_aware["wrist_images"])
    assert obs_full["extra_view_images"] is None
    assert obs_task_aware["extra_view_images"] is None
