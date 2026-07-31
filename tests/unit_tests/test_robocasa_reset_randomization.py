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
