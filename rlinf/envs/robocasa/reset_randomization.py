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

import numpy as np

_DEFAULT_TEXTURE_GEOM_PATTERNS = (
    "wall_",
    "floor_",
    "counter_",
    "cab_",
    "stack_",
    "drawer_",
)


def _next_offset(rng: Any, size: int, previous: int) -> int:
    """Choose a cyclic offset that differs from the previous one."""
    if size < 2:
        return 0
    if hasattr(rng, "integers"):
        delta = int(rng.integers(1, size))
    else:
        delta = int(rng.randint(1, size))
    return (previous + delta) % size


class _CompiledModelVisualRandomizer:
    """Randomize already-compiled RoboCasa visual assets in-place."""

    _VISUAL_GEOM_FIELDS = (
        "geom_dataid",
        "geom_matid",
        "geom_pos",
        "geom_quat",
        "geom_rgba",
        "geom_size",
    )

    def __init__(self, env: Any, config: Mapping[str, Any]) -> None:
        self.env = env
        self.rng = getattr(env, "rng", np.random.default_rng())
        self.model = env.sim.model
        self.randomize_textures = bool(
            config.get("randomize_preloaded_textures", False)
        )
        self.shuffle_object_visuals = bool(config.get("shuffle_object_visuals", False))
        self.color_jitter = float(config.get("material_color_jitter", 0.0))
        if not 0.0 <= self.color_jitter <= 1.0:
            raise ValueError(
                "reset_randomization.material_color_jitter must be in [0, 1]"
            )

        raw_patterns = config.get(
            "texture_geom_patterns", _DEFAULT_TEXTURE_GEOM_PATTERNS
        )
        if not isinstance(raw_patterns, (list, tuple)) or not all(
            isinstance(pattern, str) and pattern for pattern in raw_patterns
        ):
            raise ValueError(
                "reset_randomization.texture_geom_patterns must be a list of "
                "non-empty strings"
            )
        self.texture_geom_patterns = tuple(pattern.lower() for pattern in raw_patterns)

        self._texture_material_ids = self._find_texture_material_ids()
        self._base_mat_texid = {
            mat_id: np.array(self.model.mat_texid[mat_id], copy=True)
            for mat_id in self._texture_material_ids
        }
        self._texture_ids_by_type = self._find_texture_ids_by_type()
        self._texture_offsets = dict.fromkeys(self._texture_ids_by_type, 0)

        self._object_visual_groups = self._snapshot_object_visual_groups()
        self._object_offsets = [0] * len(self._object_visual_groups)

        color_material_ids = set(self._texture_material_ids)
        for group in self._object_visual_groups:
            for visual in group:
                color_material_ids.update(
                    int(mat_id) for mat_id in visual["geom_matid"] if int(mat_id) >= 0
                )
        self._base_mat_rgba = {
            mat_id: np.array(self.model.mat_rgba[mat_id], copy=True)
            for mat_id in sorted(color_material_ids)
        }

    def _find_texture_material_ids(self) -> list[int]:
        if not self.randomize_textures and self.color_jitter == 0.0:
            return []
        material_ids = set()
        for geom_id, geom_name in enumerate(getattr(self.model, "geom_names", ())):
            if not any(
                pattern in geom_name.lower() for pattern in self.texture_geom_patterns
            ):
                continue
            mat_id = int(self.model.geom_matid[geom_id])
            if mat_id >= 0:
                material_ids.add(mat_id)
        return sorted(material_ids)

    def _find_texture_ids_by_type(self) -> dict[int, tuple[int, ...]]:
        if not self.randomize_textures:
            return {}
        texture_ids_by_type: dict[int, set[int]] = {}
        for row in self._base_mat_texid.values():
            for raw_texture_id in row:
                texture_id = int(raw_texture_id)
                if texture_id < 0:
                    continue
                texture_type = int(self.model.tex_type[texture_id])
                texture_ids_by_type.setdefault(texture_type, set()).add(texture_id)
        return {
            texture_type: tuple(sorted(texture_ids))
            for texture_type, texture_ids in texture_ids_by_type.items()
            if len(texture_ids) >= 2
        }

    def _snapshot_object_visual_groups(self) -> list[list[dict[str, np.ndarray]]]:
        if not self.shuffle_object_visuals:
            return []
        objects = getattr(self.env, "objects", None)
        if not isinstance(objects, Mapping):
            return []

        groups: dict[tuple[int, ...], list[dict[str, np.ndarray]]] = {}
        geom_names = set(getattr(self.model, "geom_names", ()))
        for obj in objects.values():
            visual_names = [
                name for name in getattr(obj, "visual_geoms", ()) if name in geom_names
            ]
            if not visual_names:
                continue
            geom_ids = np.asarray(
                [self.model.geom_name2id(name) for name in visual_names],
                dtype=np.int64,
            )
            signature = tuple(
                int(self.model.geom_type[geom_id]) for geom_id in geom_ids
            )
            visual = {"geom_ids": geom_ids}
            for field in self._VISUAL_GEOM_FIELDS:
                visual[field] = np.array(
                    getattr(self.model, field)[geom_ids], copy=True
                )
            groups.setdefault(signature, []).append(visual)
        return [group for group in groups.values() if len(group) >= 2]

    def _apply_texture_permutation(self) -> None:
        texture_maps: dict[int, dict[int, int]] = {}
        for texture_type, texture_ids in self._texture_ids_by_type.items():
            offset = _next_offset(
                self.rng,
                len(texture_ids),
                self._texture_offsets[texture_type],
            )
            self._texture_offsets[texture_type] = offset
            texture_maps[texture_type] = {
                texture_id: texture_ids[(index + offset) % len(texture_ids)]
                for index, texture_id in enumerate(texture_ids)
            }

        for mat_id, base_row in self._base_mat_texid.items():
            randomized_row = np.array(base_row, copy=True)
            for slot, raw_texture_id in enumerate(base_row):
                texture_id = int(raw_texture_id)
                if texture_id < 0:
                    continue
                texture_type = int(self.model.tex_type[texture_id])
                texture_map = texture_maps.get(texture_type)
                if texture_map is not None:
                    randomized_row[slot] = texture_map[texture_id]
            self.model.mat_texid[mat_id] = randomized_row

    def _apply_object_visual_permutation(self) -> None:
        for group_index, group in enumerate(self._object_visual_groups):
            offset = _next_offset(
                self.rng,
                len(group),
                self._object_offsets[group_index],
            )
            self._object_offsets[group_index] = offset
            for target_index, target in enumerate(group):
                source = group[(target_index + offset) % len(group)]
                geom_ids = target["geom_ids"]
                for field in self._VISUAL_GEOM_FIELDS:
                    getattr(self.model, field)[geom_ids] = source[field]

    def _apply_color_jitter(self) -> None:
        if self.color_jitter == 0.0:
            return
        lower = 1.0 - self.color_jitter
        upper = 1.0 + self.color_jitter
        for mat_id, base_rgba in self._base_mat_rgba.items():
            rgba = np.array(base_rgba, copy=True)
            rgba[:3] = np.clip(
                rgba[:3] * self.rng.uniform(lower, upper, size=3),
                0.0,
                1.0,
            )
            self.model.mat_rgba[mat_id] = rgba

    def apply(self) -> None:
        """Apply one visual variant without recompiling the MuJoCo model."""
        if self.randomize_textures:
            self._apply_texture_permutation()
        if self.shuffle_object_visuals:
            self._apply_object_visual_permutation()
        self._apply_color_jitter()


def _make_visual_randomizer(
    env: Any, config: Mapping[str, Any]
) -> _CompiledModelVisualRandomizer | None:
    requested = any(
        (
            bool(config.get("randomize_preloaded_textures", False)),
            bool(config.get("shuffle_object_visuals", False)),
            float(config.get("material_color_jitter", 0.0)) > 0.0,
        )
    )
    if not requested:
        return None
    if bool(getattr(env, "hard_reset", False)):
        raise ValueError(
            "Compiled-model visual randomization requires RoboCasa hard_reset=False"
        )
    if getattr(env, "sim", None) is None:
        return None
    return _CompiledModelVisualRandomizer(env, config)


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
    can resample movable-object placements, override the task's drawer-open
    reset range, permute already-loaded textures, and shuffle already-loaded
    object visuals. All changes use RoboCasa's normal soft-reset path.

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

    visual_randomizer = _make_visual_randomizer(env, config)

    original_reset = env.reset

    @wraps(original_reset)
    def reset(*args: Any, **kwargs: Any) -> Any:
        if resample_placements:
            _resample_object_placements(env, max_attempts)
        if visual_randomizer is not None:
            visual_randomizer.apply()

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
