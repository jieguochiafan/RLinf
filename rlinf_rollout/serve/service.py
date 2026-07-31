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

"""Assembly of a rollout service: config -> Ray -> workers -> controller.

:class:`RolloutService` is the object behind ``rollout-serve``. It performs the
four steps the training repo's entry scripts used to do inline, minus everything
trainer-shaped:

1. attach to (or start) a Ray cluster from ``cluster``;
2. launch the control plane and the component worker groups per placement;
3. build the matching :class:`~rlinf_rollout.serve.controller.RolloutController`
   and its :class:`~rlinf_rollout.api.v1.TaskSource`;
4. run the resident loop until a client (or a finished task source) stops it.

:meth:`RolloutService.plan` answers "what would you launch?" without touching
Ray, which is what ``rollout-serve --dry-run`` prints and what the tests assert.
"""

from __future__ import annotations

import asyncio
import importlib
from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

from rlinf_rollout.api.v1 import SamplingParams, TaskSource
from rlinf_rollout.config import (
    RolloutConfigError,
    RolloutKind,
    RolloutMode,
    build_rollout_config,
    rollout_kind,
    validate_rollout_config,
)
from rlinf_rollout.serve.controller import (
    EmbodiedRolloutController,
    LLMRolloutController,
    RolloutController,
)
from rlinf_rollout.serve.protocol import (
    DEFAULT_SERVICE_NAME,
    ServiceEndpoints,
    control_group_name,
)

__all__ = [
    "EMBODIED_WORKER_CLASS_PATHS",
    "LLM_WORKER_CLASS_PATHS",
    "RolloutService",
    "ServicePlan",
    "load_service_config",
    "resolve_worker_class_path",
]

#: Embodied component -> worker class, as an importable dotted path.
EMBODIED_WORKER_CLASS_PATHS: dict[str, str] = {
    "rollout": (
        "rlinf_rollout.workers.rollout.hf.huggingface_worker:"
        "AsyncMultiStepRolloutWorker"
    ),
    "env": "rlinf_rollout.workers.env.env_worker:AsyncEnvWorker",
}

#: ``(backend, serving_mode)`` -> engine worker class, as an importable dotted path.
#:
#: Mirrors :func:`rlinf_rollout.workers.rollout.utils.get_rollout_backend_worker`
#: without importing the engine modules, so a dry run works without SGLang / vLLM
#: installed. ``tests/test_phase4_serve.py`` asserts the two agree.
LLM_WORKER_CLASS_PATHS: dict[tuple[str, Optional[str]], str] = {
    ("sglang", None): "rlinf_rollout.workers.rollout.sglang.sglang_worker:SGLangWorker",
    ("sglang", "worker_http"): (
        "rlinf_rollout.workers.rollout.sglang.sglang_worker_server:"
        "SGLangWorkerWithHTTPServer"
    ),
    ("vllm", None): "rlinf_rollout.workers.rollout.vllm.vllm_worker:VLLMWorker",
}


def load_service_config(path: str, overrides: Optional[list[str]] = None) -> DictConfig:
    """Load a service config from YAML and apply ``key=value`` overrides.

    Args:
        path: Path to the YAML config.
        overrides: ``dotted.key=value`` strings, parsed as YAML scalars so that
            ``null``, ``true`` and numbers keep their types.

    Returns:
        The merged, defaults-filled and validated config.

    Raises:
        RolloutConfigError: If an override is malformed or the config is invalid.
    """
    cfg = OmegaConf.load(path)
    if overrides:
        dotlist = []
        for override in overrides:
            if "=" not in override:
                raise RolloutConfigError(
                    f"override {override!r} must be of the form key=value."
                )
            dotlist.append(override)
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    return build_rollout_config(cfg)


def resolve_worker_class_path(path: str) -> type:
    """Import a ``module:Class`` path and return the class.

    Args:
        path: Dotted module path and class name separated by ``:``.

    Returns:
        The imported class.
    """
    module_name, _, class_name = path.partition(":")
    if not class_name:
        raise ValueError(f"worker class path {path!r} must be 'module:Class'.")
    return getattr(importlib.import_module(module_name), class_name)


@dataclass
class ServicePlan:
    """What a service *would* launch, computed without touching Ray.

    Args:
        service_name: Logical service name.
        kind: Rollout chain (``embodied`` / ``llm``).
        mode: Service mode (``collect`` / ``eval``).
        endpoints: Names a client uses to reach the data plane.
        worker_groups: Group name -> worker class path, control plane included.
        placement: Component -> declared ``cluster.component_placement`` entry.
        num_nodes: ``cluster.num_nodes``.
        task_source: Description of the configured task source.
        accepted_task_kinds: Task kinds clients may submit.
        stop_when_done: Whether the service exits when the source is exhausted.
        warnings: Non-fatal notes (unsupported optional components, ...).
    """

    service_name: str
    kind: str
    mode: str
    endpoints: ServiceEndpoints
    worker_groups: dict[str, str]
    placement: dict[str, Any]
    num_nodes: int
    task_source: dict[str, Any]
    accepted_task_kinds: tuple[str, ...]
    stop_when_done: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the plan."""
        return {
            "service_name": self.service_name,
            "kind": self.kind,
            "mode": self.mode,
            "endpoints": {
                "control_group": self.endpoints.control_group,
                "output_channel": self.endpoints.output_channel,
                "num_output_partitions": self.endpoints.num_output_partitions,
                "rollout_group": self.endpoints.rollout_group,
                "env_group": self.endpoints.env_group,
                "http_endpoints": dict(self.endpoints.http_endpoints),
            },
            "worker_groups": dict(self.worker_groups),
            "placement": dict(self.placement),
            "num_nodes": self.num_nodes,
            "task_source": dict(self.task_source),
            "accepted_task_kinds": list(self.accepted_task_kinds),
            "stop_when_done": self.stop_when_done,
            "warnings": list(self.warnings),
        }


class RolloutService:
    """A standalone, long-running rollout service.

    Args:
        cfg: Rollout-system config. Validated on construction; pass the result of
            :func:`load_service_config` or :func:`build_rollout_config`.
        task_source: Explicit task source, overriding ``rollout.serve.task_source``.
        service_name: Overrides ``rollout.serve.name``.

    Raises:
        RolloutConfigError: If the config is invalid for a service.
        NotImplementedError: If the config asks for a component the standalone
            system does not vendor yet (e.g. an external reward model worker).
    """

    def __init__(
        self,
        cfg: DictConfig,
        *,
        task_source: Optional[TaskSource] = None,
        service_name: Optional[str] = None,
    ) -> None:
        validate_rollout_config(cfg)
        self.cfg = cfg
        self.kind = rollout_kind(cfg)
        self.mode = str(OmegaConf.select(cfg, "rollout.mode"))
        self.service_name = (
            service_name
            or OmegaConf.select(cfg, "rollout.serve.name", default=None)
            or DEFAULT_SERVICE_NAME
        )
        self._explicit_task_source = task_source
        self._cluster: Optional[Any] = None
        self._control_group: Optional[Any] = None
        self._worker_groups: dict[str, Any] = {}
        self._controller: Optional[RolloutController] = None
        self._warnings: list[str] = []
        self._check_unsupported_components()

    # ------------------------------------------------------------------ config

    def _check_unsupported_components(self) -> None:
        """Reject configs that need components this package does not vendor."""
        reward_cfg = OmegaConf.select(self.cfg, "reward", default=None)
        if reward_cfg is None:
            return
        if bool(reward_cfg.get("use_reward_model", False)) and not bool(
            reward_cfg.get("standalone_realworld", False)
        ):
            # TODO(agent): the reward worker is not vendored (see the Phase 2
            # forward-reference whitelist); wire it in when it lands.
            raise NotImplementedError(
                "reward.use_reward_model requires an external reward worker, which "
                "the standalone rollout system does not vendor yet. Set "
                "reward.use_reward_model=false, or run the reward model in the "
                "consumer and let it shape rewards there."
            )

    @property
    def endpoints(self) -> ServiceEndpoints:
        """Names a client uses to reach this service's data plane."""
        output_channel = (
            OmegaConf.select(self.cfg, "rollout.serve.output_channel", default=None)
            or f"{self.service_name}_output"
        )
        num_shards = OmegaConf.select(self.cfg, "sink.num_shards", default=None) or 1
        http_endpoints: dict[str, str] = {}
        if self.kind == RolloutKind.LLM:
            serving_mode = OmegaConf.select(
                self.cfg, "rollout.sglang.serving_mode", default=None
            )
            if serving_mode == "worker_http":
                host = OmegaConf.select(
                    self.cfg, "rollout.sglang.server.host", default="0.0.0.0"
                )
                port = OmegaConf.select(
                    self.cfg, "rollout.sglang.server.port", default=8020
                )
                http_endpoints["engine_openai"] = f"http://{host}:{port}"
        return ServiceEndpoints(
            control_group=control_group_name(self.service_name),
            output_channel=output_channel,
            num_output_partitions=int(num_shards),
            rollout_group=str(OmegaConf.select(self.cfg, "rollout.group_name")),
            env_group=(
                str(OmegaConf.select(self.cfg, "env.group_name", default="env"))
                if self.kind == RolloutKind.EMBODIED
                else None
            ),
            http_endpoints=http_endpoints,
        )

    @property
    def controller_cls(self) -> type[RolloutController]:
        """Controller class for this config's chain."""
        return (
            LLMRolloutController
            if self.kind == RolloutKind.LLM
            else EmbodiedRolloutController
        )

    def worker_class_paths(self) -> dict[str, str]:
        """Return component -> worker class path for this config."""
        if self.kind == RolloutKind.LLM:
            backend = str(OmegaConf.select(self.cfg, "rollout.rollout_backend"))
            serving_mode = OmegaConf.select(
                self.cfg, "rollout.sglang.serving_mode", default=None
            )
            key = (backend, serving_mode if backend == "sglang" else None)
            if key not in LLM_WORKER_CLASS_PATHS:
                raise RolloutConfigError(
                    f"no engine worker for backend={backend!r} "
                    f"serving_mode={serving_mode!r}."
                )
            return {"rollout": LLM_WORKER_CLASS_PATHS[key]}
        return dict(EMBODIED_WORKER_CLASS_PATHS)

    @property
    def stop_when_done(self) -> bool:
        """Whether the service exits once its task source is exhausted."""
        configured = OmegaConf.select(
            self.cfg, "rollout.serve.stop_when_done", default=None
        )
        if configured is not None:
            return bool(configured)
        source_type = OmegaConf.select(
            self.cfg, "rollout.serve.task_source.type", default="control_plane"
        )
        return source_type != "control_plane" or self._explicit_task_source is not None

    def plan(self) -> ServicePlan:
        """Describe what :meth:`start` would launch, without touching Ray."""
        endpoints = self.endpoints
        groups = {
            control_group_name(self.service_name): (
                "rlinf_rollout.serve.control_worker:ControlPlaneWorker"
            )
        }
        for component, class_path in self.worker_class_paths().items():
            group_name = (
                endpoints.rollout_group
                if component == "rollout"
                else endpoints.env_group
            )
            groups[str(group_name)] = class_path

        placement_cfg = OmegaConf.select(
            self.cfg, "cluster.component_placement", default=None
        )
        placement = (
            {
                str(key): value
                for key, value in OmegaConf.to_container(
                    placement_cfg, resolve=True
                ).items()
            }
            if placement_cfg is not None
            else {}
        )
        return ServicePlan(
            service_name=self.service_name,
            kind=self.kind,
            mode=self.mode,
            endpoints=endpoints,
            worker_groups=groups,
            placement=placement,
            num_nodes=int(OmegaConf.select(self.cfg, "cluster.num_nodes", default=1)),
            task_source=self._task_source_plan(),
            accepted_task_kinds=tuple(
                kind.value for kind in self.controller_cls.ACCEPTED_TASK_KINDS
            ),
            stop_when_done=self.stop_when_done,
            warnings=list(self._warnings),
        )

    def _task_source_plan(self) -> dict[str, Any]:
        """Describe the configured task source."""
        if self._explicit_task_source is not None:
            return {"type": type(self._explicit_task_source).__name__}
        source_cfg = OmegaConf.select(self.cfg, "rollout.serve.task_source")
        if source_cfg is None:
            return {"type": "control_plane"}
        return {
            str(key): value
            for key, value in OmegaConf.to_container(source_cfg, resolve=True).items()
            if value is not None
        }

    # ------------------------------------------------------------- task source

    def build_task_source(self) -> Optional[TaskSource]:
        """Build the task source described by ``rollout.serve.task_source``.

        Returns:
            The source, or ``None`` for ``control_plane`` (the controller then
            pulls straight from the control plane).
        """
        if self._explicit_task_source is not None:
            return self._explicit_task_source

        source_type = OmegaConf.select(
            self.cfg, "rollout.serve.task_source.type", default="control_plane"
        )
        if source_type == "control_plane":
            return None
        if source_type == "prompt_file":
            return self._build_prompt_file_source()
        if source_type == "eval_rounds":
            return self._build_eval_rounds_source()
        raise RolloutConfigError(f"unknown task source type {source_type!r}.")

    def _build_prompt_file_source(self) -> TaskSource:
        """Build a JSONL prompt source, tokenizing with the engine's tokenizer."""
        from rlinf_rollout.serve.task_source import JsonlPromptTaskSource

        source_cfg = OmegaConf.select(self.cfg, "rollout.serve.task_source")
        sampling_cfg = OmegaConf.select(self.cfg, "rollout.sampling_params")
        sampling = SamplingParams(
            n=int(OmegaConf.select(self.cfg, "rollout.group_size", default=1)),
            temperature=float(sampling_cfg.get("temperature", 1.0)),
            top_p=float(sampling_cfg.get("top_p", 1.0)),
            top_k=int(sampling_cfg.get("top_k", -1)),
            max_new_tokens=int(sampling_cfg.get("max_new_tokens")),
        )
        return JsonlPromptTaskSource(
            str(source_cfg.path),
            batch_size=int(OmegaConf.select(self.cfg, "rollout.batch_size", default=1)),
            group_size=sampling.n,
            encode=self._build_tokenizer_encode(),
            sampling=sampling,
            mode=("eval" if self.mode == RolloutMode.EVAL else "train"),
            max_prompts=int(source_cfg.get("max_prompts", 0) or 0),
            repeat=bool(source_cfg.get("repeat", False)),
        )

    def _build_tokenizer_encode(self):
        """Return an ``str -> list[int]`` encoder for the served model."""
        model_path = OmegaConf.select(self.cfg, "rollout.model.model_path")
        trust_remote_code = bool(
            OmegaConf.select(self.cfg, "rollout.model.trust_remote_code", default=False)
        )
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )

        def encode(text: str) -> list[int]:
            return list(tokenizer.encode(text))

        return encode

    def _build_eval_rounds_source(self) -> TaskSource:
        """Build a fixed number of embodied eval passes."""
        from rlinf_rollout.serve.task_source import StaticTaskSource, episode_task

        source_cfg = OmegaConf.select(self.cfg, "rollout.serve.task_source")
        eval_cfg = OmegaConf.select(self.cfg, "env.eval")
        num_rounds = int(source_cfg.get("num_rounds", 1) or 1)
        task = episode_task(
            env_type=str(eval_cfg.env_type),
            num_envs=int(eval_cfg.total_num_envs),
            mode="eval",
            group_size=int(eval_cfg.get("group_size", 1) or 1),
            max_episode_steps=int(eval_cfg.get("max_episode_steps", 0) or 0),
            auto_reset=bool(eval_cfg.get("auto_reset", True)),
        )
        return StaticTaskSource([task] * num_rounds)

    # --------------------------------------------------------------- lifecycle

    def start(self) -> RolloutController:
        """Start Ray, launch every worker group, and build the controller.

        Returns:
            The controller, already wired but not yet serving.
        """
        from rlinf_rollout.scheduler import Cluster, NodePlacementStrategy
        from rlinf_rollout.serve.control_worker import ControlPlaneWorker

        endpoints = self.endpoints
        self._cluster = Cluster(
            cluster_cfg=self.cfg.cluster,
            distributed_log_dir=OmegaConf.select(
                self.cfg, "rollout.serve.log_dir", default=None
            ),
        )
        controller_cls = self.controller_cls
        self._control_group = ControlPlaneWorker.create_group(
            self.service_name,
            [kind.value for kind in controller_cls.ACCEPTED_TASK_KINDS],
            int(
                OmegaConf.select(self.cfg, "rollout.serve.max_pending_tasks", default=0)
            ),
        ).launch(
            self._cluster,
            name=endpoints.control_group,
            placement_strategy=NodePlacementStrategy(node_ranks=[0]),
        )

        common = {
            "endpoints": endpoints,
            "control_plane": self._control_group,
            "task_source": self.build_task_source(),
            "service_name": self.service_name,
            "poll_interval": float(
                OmegaConf.select(
                    self.cfg, "rollout.serve.poll_interval_seconds", default=0.5
                )
            ),
            "status_interval": float(
                OmegaConf.select(
                    self.cfg, "rollout.serve.status_interval_seconds", default=5.0
                )
            ),
            "stop_when_task_source_exhausted": self.stop_when_done,
        }
        if self.kind == RolloutKind.LLM:
            self._controller = self._start_llm(**common)
        else:
            self._controller = self._start_embodied(**common)
        return self._controller

    def _start_embodied(self, **controller_kwargs: Any) -> RolloutController:
        """Launch the embodied groups and build their controller."""
        from rlinf_rollout.utils.placement import HybridComponentPlacement

        placement = HybridComponentPlacement(self.cfg, self._cluster)
        paths = self.worker_class_paths()
        rollout_group = (
            resolve_worker_class_path(paths["rollout"])
            .create_group(self.cfg)
            .launch(
                self._cluster,
                name=str(OmegaConf.select(self.cfg, "rollout.group_name")),
                placement_strategy=placement.get_strategy("rollout"),
            )
        )
        env_group = (
            resolve_worker_class_path(paths["env"])
            .create_group(self.cfg)
            .launch(
                self._cluster,
                name=str(OmegaConf.select(self.cfg, "env.group_name", default="env")),
                placement_strategy=placement.get_strategy("env"),
            )
        )
        self._worker_groups = {"rollout": rollout_group, "env": env_group}
        return EmbodiedRolloutController(
            self.cfg,
            rollout_group=rollout_group,
            env_group=env_group,
            **controller_kwargs,
        )

    def _start_llm(self, **controller_kwargs: Any) -> RolloutController:
        """Launch the engine group and build its controller."""
        from rlinf_rollout.utils.rollout_placement import RolloutComponentPlacement

        placement = RolloutComponentPlacement(self.cfg, self._cluster)
        worker_cls = resolve_worker_class_path(self.worker_class_paths()["rollout"])
        weight_reload = "sync" if self.mode == RolloutMode.COLLECT else None
        rollout_group = worker_cls.create_group(
            self.cfg, placement, weight_reload
        ).launch(
            self._cluster,
            name=str(OmegaConf.select(self.cfg, "rollout.group_name")),
            placement_strategy=placement.get_strategy("rollout"),
        )
        self._worker_groups = {"rollout": rollout_group}
        return LLMRolloutController(
            self.cfg,
            rollout_group=rollout_group,
            num_engines=placement.rollout_dp_size,
            **controller_kwargs,
        )

    def run(self) -> None:
        """Start the service and serve until stopped. Blocks the caller."""
        controller = self._controller or self.start()
        asyncio.run(self._serve(controller))

    @staticmethod
    async def _serve(controller: RolloutController) -> None:
        """Serve with ``controller`` and shut it down on any exit path."""
        try:
            await controller.serve_forever()
        finally:
            await controller.shutdown()

    @property
    def controller(self) -> Optional[RolloutController]:
        """The controller, once :meth:`start` has run."""
        return self._controller
