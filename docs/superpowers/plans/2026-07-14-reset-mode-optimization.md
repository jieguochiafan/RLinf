# Reset Mode Optimization 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 为 embodied 环境实现统一 `reset_mode`，让 LIBERO 和 RoboCasa 在任务资产不变时使用轻量 state reset。

**架构：** 新增一个小型 reset mode helper，集中定义 `full/state/task_aware` 语义和环境映射。RoboCasa 通过 `hard_reset` 直接映射；LIBERO 在 reset 时按目标 task/BDDL 是否变化决定是否调用 `reconfigure_env_fns()`。默认行为保持 full reset，避免静默改变现有实验。

**技术栈：** Python、OmegaConf、pytest、Ruff、LIBERO/robosuite/RoboCasa 环境封装。

---

## 文件结构

- 创建：`rlinf/envs/reset_mode.py`
  - 负责 reset mode 常量、校验、默认值解析、RoboCasa `hard_reset` 映射、LIBERO full/state 决策。
  - 不导入 `rlinf.envs`、LIBERO、RoboCasa、robosuite 或 gym，保证 `rlinf/config.py` 可安全导入。
- 修改：`rlinf/config.py`
  - 在 embodied env train/eval 配置校验中验证 `reset_mode` 和 `reset_full_on_state_mismatch`。
- 修改：`rlinf/envs/robocasa/robocasa_env.py`
  - 将 `reset_mode` 或显式 `hard_reset` 解析为传给 `robosuite.make()` 的 `hard_reset` 参数。
- 修改：`rlinf/envs/libero/libero_env.py`
  - 将 `_reconfigure()` 改为 task-aware reset 决策，返回 reset timing metrics，并在 `reset()` 的 `infos` 中暴露。
- 修改：`rlinf/workers/env/env_worker.py`
  - 收集 bootstrap reset metrics，让 rollout 日志和 metric payload 能看见 full/state reset 次数与耗时。
- 修改：`tests/unit_tests/test_robocasa_env.py`
  - 覆盖 RoboCasa reset mode 到 `hard_reset` 的映射。
- 创建：`tests/unit_tests/test_env_reset_mode.py`
  - 覆盖通用 helper、配置校验 helper、LIBERO reset 决策。
- 创建：`tests/unit_tests/test_libero_reset_mode.py`
  - 使用 fake LIBERO env 验证 `_reconfigure()` 的 full/state 行为。
- 修改：`tests/unit_tests/test_overlap_env_bootstrap.py`
  - 覆盖 EnvWorker bootstrap reset metrics 收集。

## 任务 1：新增 reset mode helper 与配置校验

**文件：**
- 创建：`rlinf/envs/reset_mode.py`
- 修改：`rlinf/config.py`
- 测试：`tests/unit_tests/test_env_reset_mode.py`

- [ ] **步骤 1：编写失败的 helper 测试**

在 `tests/unit_tests/test_env_reset_mode.py` 写入以下测试：

```python
import pytest
from omegaconf import OmegaConf

from rlinf.envs.reset_mode import (
    RESET_MODE_FULL,
    RESET_MODE_STATE,
    RESET_MODE_TASK_AWARE,
    ResetModeError,
    get_reset_mode,
    libero_should_full_reset,
    reset_full_on_state_mismatch,
    robocasa_hard_reset_from_cfg,
    validate_env_reset_mode_cfg,
)


def test_get_reset_mode_defaults_to_full_without_mutating_cfg():
    cfg = OmegaConf.create({})

    assert get_reset_mode(cfg) == RESET_MODE_FULL
    assert "reset_mode" not in cfg


@pytest.mark.parametrize("mode", [RESET_MODE_FULL, RESET_MODE_STATE, RESET_MODE_TASK_AWARE])
def test_validate_env_reset_mode_accepts_known_modes(mode):
    cfg = OmegaConf.create({"reset_mode": mode})

    validate_env_reset_mode_cfg(cfg, "env.train")


def test_validate_env_reset_mode_rejects_unknown_mode():
    cfg = OmegaConf.create({"reset_mode": "fast"})

    with pytest.raises(ValueError, match="env.train.reset_mode"):
        validate_env_reset_mode_cfg(cfg, "env.train")


def test_reset_full_on_state_mismatch_defaults_true():
    assert reset_full_on_state_mismatch(OmegaConf.create({})) is True
    assert reset_full_on_state_mismatch(
        OmegaConf.create({"reset_full_on_state_mismatch": False})
    ) is False


@pytest.mark.parametrize(
    ("cfg_dict", "expected"),
    [
        ({"reset_mode": "full"}, True),
        ({"reset_mode": "state"}, False),
        ({"reset_mode": "task_aware"}, False),
        ({"hard_reset": False}, False),
        ({}, True),
    ],
)
def test_robocasa_hard_reset_from_cfg(cfg_dict, expected):
    assert robocasa_hard_reset_from_cfg(OmegaConf.create(cfg_dict)) is expected


def test_robocasa_reset_mode_overrides_hard_reset_compat_key():
    cfg = OmegaConf.create({"reset_mode": "state", "hard_reset": True})

    assert robocasa_hard_reset_from_cfg(cfg) is False


@pytest.mark.parametrize(
    ("mode", "task_changed", "is_eval", "bddl_may_change", "expected_full", "expected_fallback"),
    [
        ("full", False, False, False, True, False),
        ("full", False, True, False, False, False),
        ("task_aware", False, False, False, False, False),
        ("task_aware", True, False, False, True, False),
        ("task_aware", False, False, True, True, False),
        ("state", False, False, False, False, False),
        ("state", True, False, False, True, True),
    ],
)
def test_libero_should_full_reset_decision(
    mode,
    task_changed,
    is_eval,
    bddl_may_change,
    expected_full,
    expected_fallback,
):
    result = libero_should_full_reset(
        reset_mode=mode,
        task_changed=task_changed,
        is_eval=is_eval,
        bddl_may_change_without_task_change=bddl_may_change,
        fallback_on_state_mismatch=True,
    )

    assert result.full_reset is expected_full
    assert result.fallback_used is expected_fallback


def test_libero_state_mode_can_fail_fast_on_task_change():
    with pytest.raises(ResetModeError, match="requires state reset"):
        libero_should_full_reset(
            reset_mode="state",
            task_changed=True,
            is_eval=False,
            bddl_may_change_without_task_change=False,
            fallback_on_state_mismatch=False,
        )
```

- [ ] **步骤 2：运行 helper 测试验证失败**

运行：

```bash
pytest tests/unit_tests/test_env_reset_mode.py -q
```

预期：FAIL，报错包含 `ModuleNotFoundError: No module named 'rlinf.envs.reset_mode'`。

- [ ] **步骤 3：实现 `rlinf/envs/reset_mode.py`**

创建 `rlinf/envs/reset_mode.py`：

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


RESET_MODE_FULL = "full"
RESET_MODE_STATE = "state"
RESET_MODE_TASK_AWARE = "task_aware"
VALID_RESET_MODES = {
    RESET_MODE_FULL,
    RESET_MODE_STATE,
    RESET_MODE_TASK_AWARE,
}


class ResetModeError(ValueError):
    """Raised when a requested reset mode cannot be honored."""


@dataclass(frozen=True)
class LiberoResetDecision:
    full_reset: bool
    fallback_used: bool = False


def _cfg_contains(cfg: Any, key: str) -> bool:
    return hasattr(cfg, "__contains__") and key in cfg


def get_reset_mode(env_cfg: Any) -> str:
    mode = str(env_cfg.get("reset_mode", RESET_MODE_FULL)).lower()
    if mode not in VALID_RESET_MODES:
        raise ValueError(
            f"reset_mode must be one of {sorted(VALID_RESET_MODES)}, got {mode!r}"
        )
    return mode


def reset_full_on_state_mismatch(env_cfg: Any) -> bool:
    return bool(env_cfg.get("reset_full_on_state_mismatch", True))


def validate_env_reset_mode_cfg(env_cfg: Any, path: str) -> None:
    if _cfg_contains(env_cfg, "reset_mode"):
        try:
            get_reset_mode(env_cfg)
        except ValueError as error:
            raise ValueError(f"{path}.reset_mode is invalid: {error}") from error
    if _cfg_contains(env_cfg, "reset_full_on_state_mismatch"):
        value = env_cfg.get("reset_full_on_state_mismatch")
        if not isinstance(value, bool):
            raise ValueError(
                f"{path}.reset_full_on_state_mismatch must be a boolean, "
                f"got {type(value).__name__}"
            )


def robocasa_hard_reset_from_cfg(env_cfg: Any) -> bool:
    if _cfg_contains(env_cfg, "reset_mode"):
        return get_reset_mode(env_cfg) == RESET_MODE_FULL
    return bool(env_cfg.get("hard_reset", True))


def libero_should_full_reset(
    *,
    reset_mode: str,
    task_changed: bool,
    is_eval: bool,
    bddl_may_change_without_task_change: bool,
    fallback_on_state_mismatch: bool,
) -> LiberoResetDecision:
    mode = str(reset_mode).lower()
    if mode not in VALID_RESET_MODES:
        raise ValueError(
            f"reset_mode must be one of {sorted(VALID_RESET_MODES)}, got {mode!r}"
        )

    asset_mismatch = task_changed or bddl_may_change_without_task_change
    if mode == RESET_MODE_FULL:
        return LiberoResetDecision(full_reset=task_changed or not is_eval)
    if mode == RESET_MODE_TASK_AWARE:
        return LiberoResetDecision(full_reset=asset_mismatch)
    if asset_mismatch:
        if fallback_on_state_mismatch:
            return LiberoResetDecision(full_reset=True, fallback_used=True)
        raise ResetModeError(
            "reset_mode='state' requires state reset, but target task or BDDL "
            "differs from the current environment"
        )
    return LiberoResetDecision(full_reset=False)
```

- [ ] **步骤 4：将配置校验接入 `rlinf/config.py`**

在 import 区添加：

```python
from rlinf.envs.reset_mode import validate_env_reset_mode_cfg
```

在 `validate_chunk_step_cfg(cfg.env.eval, "env.eval")` 后添加：

```python
        validate_env_reset_mode_cfg(cfg.env.eval, "env.eval")
```

在 `validate_chunk_step_cfg(cfg.env.train, "env.train")` 后添加：

```python
        validate_env_reset_mode_cfg(cfg.env.train, "env.train")
```

- [ ] **步骤 5：运行 helper 测试验证通过**

运行：

```bash
pytest tests/unit_tests/test_env_reset_mode.py -q
```

预期：PASS。

- [ ] **步骤 6：Commit**

```bash
git add rlinf/envs/reset_mode.py rlinf/config.py tests/unit_tests/test_env_reset_mode.py
git commit -s -m "feat: add env reset mode helpers"
```

## 任务 2：实现 RoboCasa `reset_mode` 到 `hard_reset` 的映射

**文件：**
- 修改：`rlinf/envs/robocasa/robocasa_env.py`
- 修改：`tests/unit_tests/test_robocasa_env.py`

- [ ] **步骤 1：编写失败的 RoboCasa 映射测试**

在 `tests/unit_tests/test_robocasa_env.py` 顶部 import 区添加：

```python
import pytest
```

在同一文件中追加：

```python


@pytest.mark.parametrize(
    ("cfg_extra", "expected_hard_reset"),
    [
        ({"reset_mode": "full"}, True),
        ({"reset_mode": "state"}, False),
        ({"reset_mode": "task_aware"}, False),
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
```

- [ ] **步骤 2：运行 RoboCasa 映射测试验证失败**

运行：

```bash
pytest tests/unit_tests/test_robocasa_env.py::test_robocasa_get_env_fns_maps_reset_mode_to_hard_reset -q
```

预期：FAIL，报错为 `KeyError: 'hard_reset'` 或断言缺少 `hard_reset` 参数。

- [ ] **步骤 3：实现 RoboCasa 映射**

在 `rlinf/envs/robocasa/robocasa_env.py` import 区添加：

```python
from rlinf.envs.reset_mode import robocasa_hard_reset_from_cfg
```

在 `get_env_fns()` 中读取配置，并传入闭包：

```python
        hard_reset = robocasa_hard_reset_from_cfg(self.cfg)

        for env_id in range(self.num_envs):
            task_idx = self.task_ids[env_id]
            task_name = self.task_names[task_idx]
            env_seed = self.env_seeds[env_id]
            camera_widths = self.cfg.init_params.camera_widths
            camera_heights = self.cfg.init_params.camera_heights
            robot_name = self.cfg.robot_name

            def env_fn(
                task=task_name,
                seed=env_seed,
                width=camera_widths,
                height=camera_heights,
                robot=robot_name,
                hard_reset=hard_reset,
            ):
                import robocasa  # noqa: F401 RoboCasa must register envs per subprocess
                import robosuite
                from robosuite.controllers import load_composite_controller_config

                controller_config = load_composite_controller_config(
                    controller=None,
                    robot=robot,
                )
                env = robosuite.make(
                    env_name=task,
                    robots=robot,
                    controller_configs=controller_config,
                    camera_names=self.camera_names,
                    camera_widths=width,
                    camera_heights=height,
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    ignore_done=True,
                    use_object_obs=True,
                    use_camera_obs=True,
                    camera_depths=False,
                    seed=seed,
                    translucent_robot=False,
                    hard_reset=hard_reset,
                    render_camera="robot0_agentview_center",
                )
                return env
```

- [ ] **步骤 4：运行 RoboCasa tests**

运行：

```bash
pytest tests/unit_tests/test_robocasa_env.py -q
```

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add rlinf/envs/robocasa/robocasa_env.py tests/unit_tests/test_robocasa_env.py
git commit -s -m "feat: map robocasa reset mode to hard reset"
```

## 任务 3：用 fake LIBERO env 验证 reset 决策

**文件：**
- 创建：`tests/unit_tests/test_libero_reset_mode.py`
- 修改：`rlinf/envs/libero/libero_env.py`

- [ ] **步骤 1：编写失败的 LIBERO fake env 测试**

创建 `tests/unit_tests/test_libero_reset_mode.py`：

```python
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
        self.set_init_state_ids = []

    def reconfigure_env_fns(self, env_fns, id=None):
        del env_fns
        self.reconfigured_ids.extend(list(id))

    def seed(self, seed):
        self.seed_value = seed

    def reset(self, id=None):
        self.reset_ids.extend(list(id))

    def set_init_state(self, init_state, id=None):
        del init_state
        self.set_init_state_ids.extend(list(id))


def _make_libero_env(monkeypatch, *, reset_mode, fallback=True, is_eval=False):
    _install_import_fakes(monkeypatch)
    monkeypatch.delitem(sys.modules, "rlinf.envs.libero.libero_env", raising=False)
    from rlinf.envs.libero.libero_env import LiberoEnv

    env = LiberoEnv.__new__(LiberoEnv)
    env.cfg = OmegaConf.create(
        {
            "reset_mode": reset_mode,
            "reset_full_on_state_mismatch": fallback,
            "is_eval": is_eval,
        }
    )
    env.env = _FakeVectorEnv()
    env.num_envs = 2
    env.seed = 3
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


def test_task_aware_skips_reconfigure_when_task_is_unchanged(monkeypatch):
    env = _make_libero_env(monkeypatch, reset_mode="task_aware")

    metrics = env._reconfigure(np.array([1, 2]), np.array([0, 1]))

    assert env.env.reconfigured_ids == []
    assert env.env.reset_ids == [0, 1]
    assert env.env.set_init_state_ids == [0, 1]
    assert metrics["state_count"] == 2
    assert metrics["full_count"] == 0


def test_task_aware_reconfigures_when_task_changes(monkeypatch):
    env = _make_libero_env(monkeypatch, reset_mode="task_aware")

    metrics = env._reconfigure(np.array([11, 2]), np.array([0, 1]))

    assert env.env.reconfigured_ids == [0]
    assert metrics["state_count"] == 1
    assert metrics["full_count"] == 1


def test_state_mode_falls_back_to_full_on_task_change(monkeypatch):
    env = _make_libero_env(monkeypatch, reset_mode="state", fallback=True)

    metrics = env._reconfigure(np.array([11]), np.array([0]))

    assert env.env.reconfigured_ids == [0]
    assert metrics["fallback_count"] == 1


def test_state_mode_fail_fast_on_task_change(monkeypatch):
    env = _make_libero_env(monkeypatch, reset_mode="state", fallback=False)

    with pytest.raises(ValueError, match="reset_mode='state'"):
        env._reconfigure(np.array([11]), np.array([0]))
```

- [ ] **步骤 2：运行 LIBERO fake env 测试验证失败**

运行：

```bash
pytest tests/unit_tests/test_libero_reset_mode.py -q
```

预期：FAIL，报错包含 `_reconfigure` 不返回 metrics 或未跳过 `reconfigure_env_fns()`。

- [ ] **步骤 3：重构 LIBERO `_reconfigure()`**

在 `rlinf/envs/libero/libero_env.py` import 区添加：

```python
from rlinf.envs.reset_mode import (
    get_reset_mode,
    libero_should_full_reset,
    reset_full_on_state_mismatch,
)
```

在 `LiberoEnv` 中添加辅助方法：

```python
    def _libero_bddl_may_change_without_task_change(self) -> bool:
        variant = os.environ.get(
            "LIBERO_TYPE",
            self.cfg.get("libero_variant", "standard")
            if hasattr(self.cfg, "get")
            else "standard",
        )
        raw_suffix = os.environ.get(
            "LIBERO_SUFFIX",
            os.environ.get(
                "LIBERO_PERTURBATION",
                self.cfg.get("perturbation_suffix", None)
                if hasattr(self.cfg, "get")
                else None,
            ),
        )
        return (
            variant in {"pro", "plus"}
            and raw_suffix == "all"
            and not getattr(self.cfg, "is_eval", False)
        )
```

替换 `_reconfigure()` 的决策部分：

```python
    def _reconfigure(self, reset_state_ids, env_idx):
        import time as _time

        reset_mode = get_reset_mode(self.cfg)
        fallback = reset_full_on_state_mismatch(self.cfg)
        bddl_may_change = self._libero_bddl_may_change_without_task_change()
        reset_metrics = {
            "full_count": 0,
            "state_count": 0,
            "fallback_count": 0,
            "reconfigure_time": 0.0,
            "base_reset_time": 0.0,
            "set_init_state_time": 0.0,
        }
        reconfig_env_idx = []
        task_ids, trial_ids = self._get_task_and_trial_ids_from_reset_state_ids(
            reset_state_ids
        )
        for j, env_id in enumerate(env_idx):
            task_changed = self.task_ids[env_id] != task_ids[j]
            decision = libero_should_full_reset(
                reset_mode=reset_mode,
                task_changed=bool(task_changed),
                is_eval=bool(getattr(self.cfg, "is_eval", False)),
                bddl_may_change_without_task_change=bddl_may_change,
                fallback_on_state_mismatch=fallback,
            )
            self.task_ids[env_id] = task_ids[j]
            self.trial_ids[env_id] = trial_ids[j]
            if decision.full_reset:
                reconfig_env_idx.append(env_id)
                reset_metrics["full_count"] += 1
            else:
                reset_metrics["state_count"] += 1
            if decision.fallback_used:
                reset_metrics["fallback_count"] += 1

        if reconfig_env_idx:
            t0 = _time.perf_counter()
            env_fn_params = self.get_env_fn_params(reconfig_env_idx)
            self.env.reconfigure_env_fns(env_fn_params, reconfig_env_idx)
            reset_metrics["reconfigure_time"] += _time.perf_counter() - t0

        self.env.seed(self.seed * len(env_idx))
        t0 = _time.perf_counter()
        self.env.reset(id=env_idx)
        reset_metrics["base_reset_time"] += _time.perf_counter() - t0

        variant = os.environ.get(
            "LIBERO_TYPE",
            self.cfg.get("libero_variant", "standard")
            if hasattr(self.cfg, "get")
            else "standard",
        )
        if variant != "plus":
            init_state = self._get_reset_states(env_idx=env_idx)
            t0 = _time.perf_counter()
            self.env.set_init_state(init_state=init_state, id=env_idx)
            reset_metrics["set_init_state_time"] += _time.perf_counter() - t0
        return reset_metrics
```

- [ ] **步骤 4：运行 LIBERO fake env 测试验证通过**

运行：

```bash
pytest tests/unit_tests/test_libero_reset_mode.py tests/unit_tests/test_env_reset_mode.py -q
```

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add rlinf/envs/libero/libero_env.py tests/unit_tests/test_libero_reset_mode.py
git commit -s -m "feat: add task-aware libero reset decision"
```

## 任务 4：在 LIBERO `reset()` 中返回 reset metrics

**文件：**
- 修改：`rlinf/envs/libero/libero_env.py`
- 修改：`tests/unit_tests/test_libero_reset_mode.py`

- [ ] **步骤 1：编写失败的 metrics 测试**

在 `tests/unit_tests/test_libero_reset_mode.py` 中追加：

```python
def test_libero_reset_returns_reset_metrics(monkeypatch):
    env = _make_libero_env(monkeypatch, reset_mode="task_aware")
    env.current_raw_obs = [
        {"image": np.zeros((2, 2, 3), dtype=np.uint8), "wrist": np.zeros((2, 2, 3), dtype=np.uint8)}
    ]
    env.num_envs = 1
    env.cfg.reset_gripper_open = True
    env._collecting = False
    env._reset_metrics = lambda env_idx: None
    env._wrap_obs = lambda raw_obs: {"raw_obs_len": len(raw_obs)}

    def _step(action, env_idx):
        del action
        raw_obs = [
            {"image": np.zeros((2, 2, 3), dtype=np.uint8), "wrist": np.zeros((2, 2, 3), dtype=np.uint8)}
        ]
        return raw_obs, np.zeros(len(env_idx)), np.zeros(len(env_idx), dtype=bool), [{}]

    env.env.step = _step

    _obs, infos = env.reset(env_idx=np.array([0]), reset_state_ids=np.array([0]))

    assert infos["reset_metrics"]["state_count"] == 1
    assert "settle_time" in infos["reset_metrics"]
    assert "total_time" in infos["reset_metrics"]
```

- [ ] **步骤 2：运行 metrics 测试验证失败**

运行：

```bash
pytest tests/unit_tests/test_libero_reset_mode.py::test_libero_reset_returns_reset_metrics -q
```

预期：FAIL，报错为 `KeyError: 'reset_metrics'`。

- [ ] **步骤 3：实现 LIBERO reset metrics 返回**

修改 `LiberoEnv.reset()`：

```python
        import time as _time

        total_t0 = _time.perf_counter()
        reset_metrics = self._reconfigure(reset_state_ids, env_idx)
        settle_t0 = _time.perf_counter()
        for _ in range(15):
            zero_actions = np.zeros((len(env_idx), 7))
            if self.cfg.reset_gripper_open:
                zero_actions[:, -1] = -1
            raw_obs, _reward, terminations, info_lists = self.env.step(
                zero_actions, env_idx
            )
        reset_metrics["settle_time"] = _time.perf_counter() - settle_t0
        reset_metrics["total_time"] = _time.perf_counter() - total_t0
```

将末尾改为：

```python
        infos = {"reset_metrics": reset_metrics}
        return obs, infos
```

- [ ] **步骤 4：运行 LIBERO reset tests**

运行：

```bash
pytest tests/unit_tests/test_libero_reset_mode.py -q
```

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add rlinf/envs/libero/libero_env.py tests/unit_tests/test_libero_reset_mode.py
git commit -s -m "feat: report libero reset metrics"
```

## 任务 5：EnvWorker 收集 bootstrap reset metrics

**文件：**
- 修改：`rlinf/workers/env/env_worker.py`
- 修改：`tests/unit_tests/test_overlap_env_bootstrap.py`

- [ ] **步骤 1：编写失败的 EnvWorker metrics 测试**

在 `tests/unit_tests/test_overlap_env_bootstrap.py` 中追加：

```python
    def test_bootstrap_step_collects_reset_metrics(self):
        self.cfg.env.train.auto_reset = False
        mock_env = MagicMock()
        mock_env.reset.return_value = (
            {"main_images": torch.zeros(2, 3, 224, 224)},
            {
                "reset_metrics": {
                    "full_count": 1,
                    "state_count": 0,
                    "total_time": 1.25,
                }
            },
        )
        self.worker.env_list = [mock_env]
        self.worker._pending_reset_metrics = []

        self.worker.bootstrap_step()

        assert self.worker._pending_reset_metrics == [
            {"full_count": 1, "state_count": 0, "total_time": 1.25}
        ]
```

- [ ] **步骤 2：运行 EnvWorker metrics 测试验证失败**

运行：

```bash
pytest tests/unit_tests/test_overlap_env_bootstrap.py::TestOverlapEnvBootstrap::test_bootstrap_step_collects_reset_metrics -q
```

预期：FAIL，报错为 `_pending_reset_metrics` 未更新。

- [ ] **步骤 3：实现 EnvWorker metrics 收集**

在 `EnvWorker.__init__` 中添加：

```python
        self._pending_reset_metrics: list[dict[str, Any]] = []
```

在 `bootstrap_step()` 中每次 reset 后添加：

```python
                reset_metrics = infos.get("reset_metrics")
                if reset_metrics is not None:
                    self._pending_reset_metrics.append(reset_metrics)
```

在 `_run_interact_once()` 初始化 `env_metrics` 后清空待发送 metrics：

```python
        self._pending_reset_metrics = []
```

在每次 bootstrap 完成后，把 metrics 合并到 `env_metrics`：

```python
            while self._pending_reset_metrics:
                reset_metrics = self._pending_reset_metrics.pop(0)
                for key, value in reset_metrics.items():
                    env_metrics[f"reset/{key}"].append(
                        torch.tensor(float(value), dtype=torch.float32)
                    )
```

- [ ] **步骤 4：运行 EnvWorker metrics 测试**

运行：

```bash
pytest tests/unit_tests/test_overlap_env_bootstrap.py -q
```

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add rlinf/workers/env/env_worker.py tests/unit_tests/test_overlap_env_bootstrap.py
git commit -s -m "feat: collect env reset metrics"
```

## 任务 6：验证、Ruff 和本地 smoke 命令

**文件：**
- 修改：`examples/embodiment/config/env/robocasa_closedrawer.yaml`
- 可选修改：一个用于 profiling 的 LIBERO 配置，例如 `examples/embodiment/config/libero_spatial_async_gipo_openpi_pi05_verify.yaml`

- [ ] **步骤 1：为 profiling 配置显式启用 `task_aware`**

在要验证的配置中添加：

```yaml
reset_mode: task_aware
reset_full_on_state_mismatch: true
```

RoboCasa 默认 env 配置可添加：

```yaml
reset_mode: task_aware
```

- [ ] **步骤 2：运行相关单元测试**

运行：

```bash
pytest \
  tests/unit_tests/test_env_reset_mode.py \
  tests/unit_tests/test_robocasa_env.py \
  tests/unit_tests/test_libero_reset_mode.py \
  tests/unit_tests/test_overlap_env_bootstrap.py \
  -q
```

预期：PASS。

- [ ] **步骤 3：运行 Ruff**

运行：

```bash
ruff check \
  rlinf/envs/reset_mode.py \
  rlinf/config.py \
  rlinf/envs/robocasa/robocasa_env.py \
  rlinf/envs/libero/libero_env.py \
  rlinf/workers/env/env_worker.py \
  tests/unit_tests/test_env_reset_mode.py \
  tests/unit_tests/test_robocasa_env.py \
  tests/unit_tests/test_libero_reset_mode.py \
  tests/unit_tests/test_overlap_env_bootstrap.py
```

预期：PASS。

- [ ] **步骤 4：运行 LIBERO 本地 smoke**

运行：

```bash
/bin/bash -lc 'source /data1/miliang/RLinf/libero_openpi/bin/activate && \
export LIBERO_CONFIG_PATH=/tmp/libero_config_reset_mode_smoke && \
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBERO_TYPE=standard && \
PYTHONDONTWRITEBYTECODE=1 python - <<"PY"
import os
import time
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv

bench = get_benchmark("libero_10")()
task = bench.get_task(0)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
init_state = bench.get_task_init_states(0)[0]
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=64, camera_widths=64)
env.seed(0)
t0 = time.perf_counter()
env.reset()
full_like = time.perf_counter() - t0
t0 = time.perf_counter()
env.set_init_state(init_state)
state_only = time.perf_counter() - t0
print({"full_like_s": round(full_like, 6), "state_only_s": round(state_only, 6)})
env.close()
PY'
```

预期：命令退出码为 0，输出中 `state_only_s` 明显小于 `full_like_s`。

- [ ] **步骤 5：运行 RoboCasa 本地 smoke**

运行：

```bash
/bin/bash -lc 'source /data1/miliang/RLinf/robocasa_openpi/bin/activate && \
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MPLCONFIGDIR=/tmp/matplotlib-robocasa-reset-smoke && \
PYTHONDONTWRITEBYTECODE=1 python - <<"PY"
import time
import robocasa
import robosuite
from robosuite.controllers import load_composite_controller_config

controller_config = load_composite_controller_config(controller=None, robot="PandaOmron")
base = dict(
    env_name="CloseDrawer",
    robots="PandaOmron",
    controller_configs=controller_config,
    camera_names=["robot0_agentview_left", "robot0_eye_in_hand"],
    camera_widths=64,
    camera_heights=64,
    has_renderer=False,
    has_offscreen_renderer=True,
    ignore_done=True,
    use_object_obs=True,
    use_camera_obs=True,
    camera_depths=False,
    seed=42,
    translucent_robot=False,
    render_camera="robot0_agentview_center",
)
for hard_reset in (True, False):
    env = robosuite.make(**dict(base, hard_reset=hard_reset))
    t0 = time.perf_counter()
    env.reset()
    reset_s = time.perf_counter() - t0
    print({"hard_reset": hard_reset, "reset_s": round(reset_s, 6)})
    env.close()
PY'
```

预期：命令退出码为 0，`hard_reset=False` 的 `reset_s` 明显小于 `hard_reset=True`。

- [ ] **步骤 6：Commit**

```bash
git add \
  examples/embodiment/config/env/robocasa_closedrawer.yaml \
  examples/embodiment/config/libero_spatial_async_gipo_openpi_pi05_verify.yaml
git commit -s -m "chore: enable task-aware reset in profiling configs"
```

如果只修改了其中一个配置，只 `git add` 实际变更的文件。

## 最终验证

- [ ] **步骤 1：运行聚合测试**

```bash
pytest \
  tests/unit_tests/test_env_reset_mode.py \
  tests/unit_tests/test_robocasa_env.py \
  tests/unit_tests/test_libero_reset_mode.py \
  tests/unit_tests/test_overlap_env_bootstrap.py \
  -q
```

预期：PASS。

- [ ] **步骤 2：运行 Ruff 聚合检查**

```bash
ruff check \
  rlinf/envs/reset_mode.py \
  rlinf/config.py \
  rlinf/envs/robocasa/robocasa_env.py \
  rlinf/envs/libero/libero_env.py \
  rlinf/workers/env/env_worker.py \
  tests/unit_tests/test_env_reset_mode.py \
  tests/unit_tests/test_robocasa_env.py \
  tests/unit_tests/test_libero_reset_mode.py \
  tests/unit_tests/test_overlap_env_bootstrap.py
```

预期：PASS。

- [ ] **步骤 3：检查提交历史和工作区**

```bash
git status --short
git log --oneline -5
```

预期：只剩用户已有的无关改动；本计划产生的提交在最近提交中可见。
