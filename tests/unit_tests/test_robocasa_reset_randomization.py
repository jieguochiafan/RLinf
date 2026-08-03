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

import numpy as np
import pytest

from rlinf.envs.robocasa.reset_randomization import install_reset_randomization


class _FakeDrawer:
    def __init__(self) -> None:
        self.calls: list[tuple[float, float, float]] = []

    def set_door_state(self, min, max, env, rng) -> None:
        del env
        self.calls.append((min, max, float(rng.uniform(min, max))))


class _FakePlacementInitializer:
    def __init__(self) -> None:
        self.calls = 0

    def sample(self, placed_objects):
        self.calls += 1
        return {"object": (self.calls, placed_objects)}


class _FakeEnv:
    def __init__(self) -> None:
        self.rng = np.random.default_rng(7)
        self.drawer = _FakeDrawer()
        self.fxtr_placements = {"fixture": object()}
        self.object_placements = {"object": object()}
        self.placement_initializer = _FakePlacementInitializer()

    def reset(self):
        self.drawer.set_door_state(min=0.9, max=1.0, env=self, rng=self.rng)
        return {"drawer": self.drawer.calls[-1][2]}


class _FakeObject:
    def __init__(self, visual_geoms: list[str]) -> None:
        self.visual_geoms = visual_geoms


class _FakeModel:
    def __init__(self) -> None:
        self.geom_names = ["wall_a", "wall_b", "object_a", "object_b"]
        self.geom_type = np.array([6, 6, 7, 7], dtype=np.int32)
        self.geom_dataid = np.array([-1, -1, 10, 20], dtype=np.int32)
        self.geom_matid = np.array([0, 1, 2, 3], dtype=np.int32)
        self.geom_pos = np.zeros((4, 3), dtype=np.float64)
        self.geom_quat = np.tile([1.0, 0.0, 0.0, 0.0], (4, 1))
        self.geom_rgba = np.ones((4, 4), dtype=np.float32)
        self.geom_size = np.ones((4, 3), dtype=np.float64)
        self.mat_texid = np.array(
            [[0, -1], [1, -1], [-1, -1], [-1, -1]], dtype=np.int32
        )
        self.mat_rgba = np.ones((4, 4), dtype=np.float32)
        self.tex_type = np.array([0, 0], dtype=np.int32)

    def geom_name2id(self, name: str) -> int:
        return self.geom_names.index(name)


class _FakeVisualEnv(_FakeEnv):
    def __init__(self) -> None:
        super().__init__()
        self.hard_reset = False
        self.sim = type("FakeSim", (), {"model": _FakeModel()})()
        self.objects = {
            "object_a": _FakeObject(["object_a"]),
            "object_b": _FakeObject(["object_b"]),
        }


def test_reset_randomization_changes_soft_reset_state() -> None:
    env = _FakeEnv()
    original_set_door_state = env.drawer.set_door_state
    install_reset_randomization(
        env,
        {
            "enabled": True,
            "drawer_open_range": [0.65, 1.0],
            "resample_object_placements": True,
        },
    )

    first = env.reset()
    second = env.reset()

    assert first["drawer"] != second["drawer"]
    assert all(call[:2] == (0.65, 1.0) for call in env.drawer.calls)
    assert env.placement_initializer.calls == 2
    assert env.drawer.set_door_state == original_set_door_state


def test_reset_randomization_disabled_preserves_original_reset() -> None:
    env = _FakeEnv()
    original_reset = env.reset

    install_reset_randomization(env, {"enabled": False})

    assert env.reset == original_reset


def test_soft_reset_randomizes_preloaded_textures_and_object_visuals() -> None:
    env = _FakeVisualEnv()
    install_reset_randomization(
        env,
        {
            "enabled": True,
            "resample_object_placements": False,
            "randomize_preloaded_textures": True,
            "shuffle_object_visuals": True,
            "material_color_jitter": 0.1,
        },
    )

    env.reset()
    first_textures = env.sim.model.mat_texid.copy()
    first_object_meshes = env.sim.model.geom_dataid[2:].copy()
    first_colors = env.sim.model.mat_rgba.copy()
    env.reset()

    assert not np.array_equal(env.sim.model.mat_texid, first_textures)
    assert not np.array_equal(env.sim.model.geom_dataid[2:], first_object_meshes)
    assert not np.array_equal(env.sim.model.mat_rgba, first_colors)


def test_compiled_visual_randomization_rejects_hard_reset() -> None:
    env = _FakeVisualEnv()
    env.hard_reset = True

    with pytest.raises(ValueError, match="hard_reset=False"):
        install_reset_randomization(
            env,
            {
                "enabled": True,
                "randomize_preloaded_textures": True,
            },
        )


@pytest.mark.parametrize(
    "drawer_range",
    ([0.8], [-0.1, 1.0], [0.8, 1.1], [0.9, 0.8]),
)
def test_reset_randomization_rejects_invalid_drawer_range(drawer_range) -> None:
    with pytest.raises(ValueError, match="drawer_open_range"):
        install_reset_randomization(
            _FakeEnv(),
            {"enabled": True, "drawer_open_range": drawer_range},
        )
