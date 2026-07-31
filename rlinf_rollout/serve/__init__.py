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

"""``rollout-serve``: the service side of the standalone rollout system.

Layout:

* :mod:`~rlinf_rollout.serve.protocol` — control-plane messages (no Ray, no torch
  beyond ``api/v1``), shared with the client SDK.
* :mod:`~rlinf_rollout.serve.control` — the control-plane state machine.
* :mod:`~rlinf_rollout.serve.control_worker` — its remotely addressable shell.
* :mod:`~rlinf_rollout.serve.task_source` — where work comes from.
* :mod:`~rlinf_rollout.serve.controller` — the resident driver loop.
* :mod:`~rlinf_rollout.serve.service` — config -> Ray -> workers -> controller.
* :mod:`~rlinf_rollout.serve.main` — the CLI.

Only the leaf modules are imported eagerly here; anything that needs Ray or an
engine is resolved on first attribute access (PEP 562), so
``import rlinf_rollout.serve`` stays cheap and works without a runtime.
"""

from typing import TYPE_CHECKING, Any

from rlinf_rollout.serve.control import ControlPlaneState
from rlinf_rollout.serve.protocol import (
    DEFAULT_SERVICE_NAME,
    ServiceEndpoints,
    ServiceState,
    ServiceStatus,
    TaskAck,
    WeightPushResult,
    WeightPushStatus,
    control_group_name,
)

if TYPE_CHECKING:
    from rlinf_rollout.serve.control_worker import ControlPlaneWorker  # noqa: F401
    from rlinf_rollout.serve.controller import (  # noqa: F401
        EmbodiedRolloutController,
        LLMRolloutController,
        RolloutController,
    )
    from rlinf_rollout.serve.service import (  # noqa: F401
        RolloutService,
        ServicePlan,
        load_service_config,
    )
    from rlinf_rollout.serve.task_source import (  # noqa: F401
        ControlPlaneTaskSource,
        JsonlPromptTaskSource,
        StaticTaskSource,
        episode_task,
    )

_LAZY_EXPORTS: dict[str, str] = {
    "ControlPlaneWorker": "rlinf_rollout.serve.control_worker",
    "EmbodiedRolloutController": "rlinf_rollout.serve.controller",
    "LLMRolloutController": "rlinf_rollout.serve.controller",
    "RolloutController": "rlinf_rollout.serve.controller",
    "RolloutService": "rlinf_rollout.serve.service",
    "ServicePlan": "rlinf_rollout.serve.service",
    "load_service_config": "rlinf_rollout.serve.service",
    "ControlPlaneTaskSource": "rlinf_rollout.serve.task_source",
    "JsonlPromptTaskSource": "rlinf_rollout.serve.task_source",
    "StaticTaskSource": "rlinf_rollout.serve.task_source",
    "episode_task": "rlinf_rollout.serve.task_source",
}

__all__ = [
    "ControlPlaneState",
    "DEFAULT_SERVICE_NAME",
    "ServiceEndpoints",
    "ServiceState",
    "ServiceStatus",
    "TaskAck",
    "WeightPushResult",
    "WeightPushStatus",
    "control_group_name",
    *sorted(_LAZY_EXPORTS),
]


def __getattr__(name: str) -> Any:
    """Import the submodule owning ``name`` on first access."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(__all__)
