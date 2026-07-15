from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest
from omegaconf import OmegaConf


def _install_import_fakes(monkeypatch):
    fake_gym = ModuleType("gym")
    fake_gym.Env = object
    fake_gymnasium = ModuleType("gymnasium")
    fake_gymnasium.Env = object
    fake_imageio = ModuleType("imageio")
    fake_pil = ModuleType("PIL")
    fake_pil_image = ModuleType("PIL.Image")
    fake_pil_image_draw = ModuleType("PIL.ImageDraw")
    fake_pil_image_font = ModuleType("PIL.ImageFont")
    fake_libero_utils = ModuleType("rlinf.envs.libero.utils")
    fake_libero_utils.get_benchmark_overridden = lambda name: object
    fake_libero_utils.get_libero_image = lambda obs: obs["image"]
    fake_libero_utils.get_libero_type = lambda: "standard"
    fake_libero_utils.get_libero_wrist_image = lambda obs: obs["wrist"]
    fake_libero_utils.quat2axisangle = lambda quat: quat
    fake_libero_venv = ModuleType("rlinf.envs.libero.venv")
    fake_libero_venv.ReconfigureSubprocEnv = object

    monkeypatch.setitem(sys.modules, "gym", fake_gym)
    monkeypatch.setitem(sys.modules, "gymnasium", fake_gymnasium)
    monkeypatch.setitem(sys.modules, "imageio", fake_imageio)
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_pil_image)
    monkeypatch.setitem(sys.modules, "PIL.ImageDraw", fake_pil_image_draw)
    monkeypatch.setitem(sys.modules, "PIL.ImageFont", fake_pil_image_font)
    monkeypatch.setitem(sys.modules, "rlinf.envs.libero.utils", fake_libero_utils)
    monkeypatch.setitem(sys.modules, "rlinf.envs.libero.venv", fake_libero_venv)

    fake_libero = ModuleType("libero")
    fake_libero_core = ModuleType("libero.libero")
    fake_benchmark = ModuleType("libero.libero.benchmark")
    fake_benchmark.Benchmark = object
    fake_benchmark.get_benchmark_dict = lambda: {}
    fake_benchmark.get_benchmark = lambda name: object
    monkeypatch.setitem(sys.modules, "libero", fake_libero)
    monkeypatch.setitem(sys.modules, "libero.libero", fake_libero_core)
    monkeypatch.setitem(sys.modules, "libero.libero.benchmark", fake_benchmark)


class _FakeVectorEnv:
    def __init__(self):
        self.reconfigured_ids = []
        self.reset_ids = []
        self.soft_reset_ids = []
        self.set_init_state_ids = []
        self.set_init_state_obs = []

    def reconfigure_env_fns(self, env_fns, id=None):
        del env_fns
        self.reconfigured_ids.extend(list(id))

    def seed(self, seed):
        self.seed_value = seed

    def reset(self, id=None):
        self.reset_ids.extend(list(id))

    def soft_reset(self, id=None):
        self.soft_reset_ids.extend(list(id))

    def set_init_state(self, init_state, id=None):
        del init_state
        self.set_init_state_ids.extend(list(id))
        obs = [{"image": np.zeros((2, 2, 3), dtype=np.uint8), "wrist": np.zeros((2, 2, 3), dtype=np.uint8)} for _ in id]
        self.set_init_state_obs.extend(obs)
        return obs


def _make_libero_env(monkeypatch, *, reset_mode, fallback=True, is_eval=False):
    _install_import_fakes(monkeypatch)
    monkeypatch.delitem(sys.modules, "rlinf.envs.libero.libero_env", raising=False)
    from rlinf.envs.libero.libero_env import LiberoEnv

    env = LiberoEnv.__new__(LiberoEnv)
    env.cfg = OmegaConf.create(
        {
            "reset_mode": reset_mode,
            "reset_optimization_enabled": True,
            "reset_full_on_state_mismatch": fallback,
            "is_eval": is_eval,
        }
    )
    env.env = _FakeVectorEnv()
    env.num_envs = 2
    env.seed = 3
    env._is_start = False
    env.task_ids = np.array([0, 0])
    env.trial_ids = np.array([0, 1])
    env._get_task_and_trial_ids_from_reset_state_ids = lambda ids: (
        np.asarray(ids) // 10,
        np.asarray(ids) % 10,
    )
    env._get_reset_states = lambda env_idx: [f"state-{idx}" for idx in env_idx]
    env.get_env_fn_params = lambda env_idx: [f"fn-{idx}" for idx in env_idx]
    env._libero_bddl_may_change_without_task_change = lambda: False
    return env


class _FakeTaskSuite:
    def __init__(self, trial_bins):
        self.trial_bins = list(trial_bins)

    def get_num_tasks(self):
        return len(self.trial_bins)

    def get_task_init_states(self, task_id):
        return [f"state-{task_id}-{trial_id}" for trial_id in range(self.trial_bins[task_id])]


def _make_sampling_env(monkeypatch, *, strategy="random", num_envs=4, group_size=1):
    _install_import_fakes(monkeypatch)
    monkeypatch.delitem(sys.modules, "rlinf.envs.libero.libero_env", raising=False)
    from rlinf.envs.libero.libero_env import LiberoEnv

    env = LiberoEnv.__new__(LiberoEnv)
    env.cfg = OmegaConf.create(
        {
            "is_eval": False,
            "use_ordered_reset_state_ids": False,
            "reset_sampling_strategy": strategy,
        }
    )
    env.seed_offset = 0
    env.seed = 0
    env.num_envs = num_envs
    env.group_size = group_size
    env.num_group = num_envs // group_size
    env.total_num_processes = 1
    env.specific_reset_id = None
    env.task_id_filter = None
    env.task_ids_filter = None
    env._generator = np.random.default_rng(seed=0)
    env._generator_ordered = np.random.default_rng(seed=0)
    env.start_idx = 0
    env.task_suite = _FakeTaskSuite([3, 5, 7])
    env.reset_sampling_strategy = strategy
    env._compute_total_num_group_envs()
    return env


def test_task_affine_fixed_sampling_keeps_each_env_on_its_task(monkeypatch):
    env = _make_sampling_env(
        monkeypatch,
        strategy="task_affine_fixed",
        num_envs=4,
        group_size=1,
    )

    first = env._get_random_reset_state_ids(env.num_group)
    first_tasks, _ = env._get_task_and_trial_ids_from_reset_state_ids(first)
    later_tasks = []
    for _ in range(20):
        reset_state_ids = env._get_random_reset_state_ids(env.num_group)
        task_ids, _ = env._get_task_and_trial_ids_from_reset_state_ids(reset_state_ids)
        later_tasks.append(task_ids)

    assert first_tasks.tolist() == [0, 1, 2, 0]
    assert all(np.array_equal(task_ids, first_tasks) for task_ids in later_tasks)


def test_task_affine_fixed_sampling_repeats_by_group_size(monkeypatch):
    env = _make_sampling_env(
        monkeypatch,
        strategy="task_affine_fixed",
        num_envs=6,
        group_size=2,
    )

    reset_state_ids = env._get_random_reset_state_ids(env.num_group).repeat(env.group_size)
    task_ids, _ = env._get_task_and_trial_ids_from_reset_state_ids(reset_state_ids)

    assert task_ids.tolist() == [0, 0, 1, 1, 2, 2]


def test_task_aware_skips_reconfigure_when_task_is_unchanged(monkeypatch):
    env = _make_libero_env(monkeypatch, reset_mode="task_aware")

    metrics = env._reconfigure(np.array([1, 2]), np.array([0, 1]))

    assert env.env.reconfigured_ids == []
    assert env.env.reset_ids == []
    assert env.env.soft_reset_ids == [0, 1]
    assert env.env.set_init_state_ids == [0, 1]
    assert metrics["state_count"] == 2
    assert metrics["full_count"] == 0
    assert metrics["base_reset_time"] >= 0.0
    assert len(metrics["raw_obs"]) == 2


def test_task_aware_reconfigures_when_task_changes(monkeypatch):
    env = _make_libero_env(monkeypatch, reset_mode="task_aware")

    metrics = env._reconfigure(np.array([11, 2]), np.array([0, 1]))

    assert env.env.reconfigured_ids == [0]
    assert env.env.reset_ids == [0]
    assert env.env.soft_reset_ids == [1]
    assert env.env.set_init_state_ids == [0, 1]
    assert metrics["state_count"] == 1
    assert metrics["full_count"] == 1


def test_reset_optimization_disabled_forces_full_reset(monkeypatch):
    env = _make_libero_env(monkeypatch, reset_mode="task_aware")
    env.cfg.reset_optimization_enabled = False

    metrics = env._reconfigure(np.array([1, 2]), np.array([0, 1]))

    assert env.env.reconfigured_ids == [0, 1]
    assert env.env.reset_ids == [0, 1]
    assert env.env.soft_reset_ids == []
    assert env.env.set_init_state_ids == [0, 1]
    assert metrics["state_count"] == 0
    assert metrics["full_count"] == 2


def test_reset_optimization_disabled_forces_full_reset_in_eval(monkeypatch):
    env = _make_libero_env(monkeypatch, reset_mode="task_aware", is_eval=True)
    env.cfg.reset_optimization_enabled = False

    metrics = env._reconfigure(np.array([1, 2]), np.array([0, 1]))

    assert env.env.reconfigured_ids == [0, 1]
    assert env.env.reset_ids == [0, 1]
    assert env.env.soft_reset_ids == []
    assert metrics["state_count"] == 0
    assert metrics["full_count"] == 2


def test_state_mode_falls_back_to_full_on_task_change(monkeypatch):
    env = _make_libero_env(monkeypatch, reset_mode="state", fallback=True)

    metrics = env._reconfigure(np.array([11]), np.array([0]))

    assert env.env.reconfigured_ids == [0]
    assert metrics["fallback_count"] == 1


def test_state_mode_fail_fast_on_task_change(monkeypatch):
    env = _make_libero_env(monkeypatch, reset_mode="state", fallback=False)

    with pytest.raises(ValueError, match="reset_mode='state'"):
        env._reconfigure(np.array([11]), np.array([0]))


def test_libero_reset_returns_reset_metrics(monkeypatch):
    env = _make_libero_env(monkeypatch, reset_mode="task_aware")
    env.current_raw_obs = [
        {
            "image": np.zeros((2, 2, 3), dtype=np.uint8),
            "wrist": np.zeros((2, 2, 3), dtype=np.uint8),
        }
    ]
    env.num_envs = 1
    env.cfg.reset_gripper_open = True
    env._collecting = False
    env._reset_metrics = lambda env_idx: None
    env._wrap_obs = lambda raw_obs: {"raw_obs_len": len(raw_obs)}

    def _step(action, env_idx):
        del action
        raw_obs = [
            {
                "image": np.zeros((2, 2, 3), dtype=np.uint8),
                "wrist": np.zeros((2, 2, 3), dtype=np.uint8),
            }
        ]
        return raw_obs, np.zeros(len(env_idx)), np.zeros(len(env_idx), dtype=bool), [{}]

    env.env.step = _step

    _obs, infos = env.reset(env_idx=np.array([0]), reset_state_ids=np.array([0]))

    assert infos["reset_metrics"]["state_count"] == 1
    assert "settle_time" in infos["reset_metrics"]
    assert "total_time" in infos["reset_metrics"]
