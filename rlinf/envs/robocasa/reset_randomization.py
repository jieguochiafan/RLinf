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

from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
from typing import Any, Callable


def _drawer_open_range(config: Mapping[str, Any]) -> tuple[float, float] | None:
    raw_range = config.get("drawer_open_range")
    if raw_range is None:
        return None
    if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
        raise ValueError("reset_randomization.drawer_open_range must have two values")
    lower, upper = (float(value) for value in raw_range)
    if not 0.0 <= lower <= upper <= 1.0:
        raise ValueError(
            "reset_randomization.drawer_open_range must satisfy "
            f"0 <= min <= max <= 1, got [{lower}, {upper}]"
        )
    return lower, upper


def _resample_object_placements(env: Any, max_attempts: int) -> None:
    placement_initializer = getattr(env, "placement_initializer", None)
    current_placements = getattr(env, "object_placements", None)
    if placement_initializer is None or not current_placements:
        return

    last_error: Exception | None = None
    for _ in range(max_attempts):
        try:
            env.object_placements = placement_initializer.sample(
                placed_objects=getattr(env, "fxtr_placements", None)
            )
            return
        except Exception as error:  # RoboCasa raises RandomizationError.
            last_error = error
    raise RuntimeError(
        f"Failed to resample RoboCasa object placements after {max_attempts} attempts"
    ) from last_error


def _override_drawer_reset_range(
    drawer: Any,
    open_range: tuple[float, float],
) -> Callable[[], None]:
    had_instance_override = "set_door_state" in vars(drawer)
    previous_instance_value = vars(drawer).get("set_door_state")
    original = drawer.set_door_state
    lower, upper = open_range

    def set_door_state(*args: Any, **kwargs: Any) -> Any:
        if len(args) >= 2:
            args = (lower, upper, *args[2:])
        else:
            kwargs["min"] = lower
            kwargs["max"] = upper
        return original(*args, **kwargs)

    drawer.set_door_state = set_door_state

    def restore() -> None:
        if had_instance_override:
            drawer.set_door_state = previous_instance_value
        else:
            del drawer.set_door_state

    return restore


def install_reset_randomization(env: Any, config: Mapping[str, Any] | None) -> Any:
    """Install low-cost state randomization on a RoboCasa environment.

    The installed reset keeps the compiled MuJoCo model and render context. It
    can resample movable-object placements and override the task's drawer-open
    reset range, both of which are applied by RoboCasa's normal soft-reset path.

    Args:
        env: Constructed RoboCasa / robosuite environment.
        config: Reset-randomization configuration mapping.

    Returns:
        The same environment instance with its reset method wrapped when enabled.
    """
    if not config or not bool(config.get("enabled", False)):
        return env
    if getattr(env, "_rlinf_reset_randomization_installed", False):
        return env

    open_range = _drawer_open_range(config)
    resample_placements = bool(config.get("resample_object_placements", True))
    max_attempts = int(config.get("placement_sampling_attempts", 3))
    if max_attempts < 1:
        raise ValueError(
            "reset_randomization.placement_sampling_attempts must be at least 1"
        )

    original_reset = env.reset

    @wraps(original_reset)
    def reset(*args: Any, **kwargs: Any) -> Any:
        if resample_placements:
            _resample_object_placements(env, max_attempts)

        restore_drawer = None
        drawer = getattr(env, "drawer", None)
        if open_range is not None and drawer is not None:
            restore_drawer = _override_drawer_reset_range(drawer, open_range)
        try:
            return original_reset(*args, **kwargs)
        finally:
            if restore_drawer is not None:
                restore_drawer()

    env.reset = reset
    env._rlinf_reset_randomization_installed = True
    return env
