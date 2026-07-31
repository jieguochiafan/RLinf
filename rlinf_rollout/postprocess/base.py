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

"""Optional trajectory post-processing hook.

The training repo computed advantages / returns inside the env worker (via its
``algorithms.registry.calculate_adv_and_returns``). That is a trainer concern:
it needs the loss type, ``gamma``/``gae_lambda``, group size and the actor's
micro-batch layout. The rollout system therefore emits raw trajectories and offers
this hook so a trainer-side plugin can pre-compute whatever it wants **before** the
trajectory reaches the sink.

Register a plugin either by passing an instance to the env worker or by setting
``rollout.postprocess.trajectory_postprocessor`` to a ``module:attr`` path that
resolves to a :class:`TrajectoryPostprocessor` subclass or a zero-arg factory.
"""

import importlib
from abc import ABC, abstractmethod
from typing import Any, Optional

__all__ = [
    "TrajectoryPostprocessor",
    "load_trajectory_postprocessor",
]


class TrajectoryPostprocessor(ABC):
    """Transforms a collected trajectory before it is handed to a ``TrajectorySink``.

    Implementations must be side-effect free with respect to rollout state: they only
    read the trajectory and return a (possibly enriched) trajectory.
    """

    @abstractmethod
    def process(self, trajectory: Any) -> Any:
        """Return the post-processed ``trajectory``.

        Args:
            trajectory: A ``rlinf_rollout.data.embodied_io_struct.Trajectory``.

        Returns:
            The trajectory to forward to the sink. Returning ``trajectory``
            unchanged is valid.
        """

    def __call__(self, trajectory: Any) -> Any:
        return self.process(trajectory)


def load_trajectory_postprocessor(
    spec: Optional[str],
) -> Optional[TrajectoryPostprocessor]:
    """Resolve ``module:attr`` into a :class:`TrajectoryPostprocessor` instance.

    A ``ValueError`` is raised when ``spec`` is malformed, the attribute is
    missing, or the resolved object is not a :class:`TrajectoryPostprocessor`.

    Args:
        spec: Dotted path such as ``my_pkg.adv:GAEPostprocessor``, or ``None``.

    Returns:
        An instance, or ``None`` when ``spec`` is ``None``/empty.
    """
    if not spec:
        return None
    if ":" not in spec:
        raise ValueError(
            "rollout.postprocess.trajectory_postprocessor must be 'module:attr', "
            f"got {spec!r}."
        )
    module_name, attr_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    try:
        attr = getattr(module, attr_name)
    except AttributeError as exc:
        raise ValueError(f"{module_name!r} has no attribute {attr_name!r}.") from exc

    obj = attr() if isinstance(attr, type) or callable(attr) else attr
    if not isinstance(obj, TrajectoryPostprocessor):
        raise ValueError(
            f"{spec!r} resolved to {type(obj).__name__}, which is not a "
            "TrajectoryPostprocessor."
        )
    return obj
