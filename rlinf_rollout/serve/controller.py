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

"""The resident driver loop of a rollout service.

A :class:`RolloutController` is what replaces the training repo's *runner*: it
owns the channels between worker groups, starts the resident coroutines, and then
sits in a poll loop doing three things only a driver can do:

1. execute client-pushed weight updates (:mod:`rlinf_rollout.serve.control`);
2. pull work from a :class:`~rlinf_rollout.api.v1.TaskSource` and dispatch it;
3. publish a :class:`~rlinf_rollout.serve.protocol.ServiceStatus` snapshot.

Everything trainer-shaped is gone: there is no optimizer step, no advantage
computation, no checkpoint saving, and no ``cfg.actor`` / ``cfg.algorithm`` read.

Worker groups arrive already launched and are only ever used through duck typing
(``group.method(...)`` returning something with ``wait()``), which keeps the
control flow testable without Ray.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Sequence

from omegaconf import DictConfig, OmegaConf

from rlinf_rollout.api.v1 import (
    ConsumerSpec,
    PartitionAxis,
    RolloutTask,
    TaskKind,
    TaskSource,
    TrajectorySink,
    WeightTransport,
    WeightUpdateAck,
    WeightUpdateRequest,
    WeightUpdateStatus,
)
from rlinf_rollout.config import RolloutMode
from rlinf_rollout.serve.protocol import (
    DEFAULT_SERVICE_NAME,
    ServiceEndpoints,
    ServiceState,
    ServiceStatus,
    WeightPushResult,
    WeightPushStatus,
)

__all__ = [
    "EmbodiedRolloutController",
    "LLMRolloutController",
    "RolloutController",
    "resolve_group_call",
]


def resolve_group_call(result: Any) -> list[Any]:
    """Normalize a worker-group call result into a per-rank list.

    The vendored scheduler returns a ``WorkerGroupFuncResult``; ``wait()`` yields
    one entry per rank. Plain values (from test stubs) are wrapped.

    Args:
        result: Whatever a worker-group method returned.

    Returns:
        One entry per rank.
    """
    wait = getattr(result, "wait", None)
    if wait is None:
        return [result]
    values = wait()
    if isinstance(values, (list, tuple)):
        return list(values)
    return [values]


class RolloutController(ABC):
    """Resident driver of one rollout service.

    Args:
        cfg: Validated rollout-system config.
        endpoints: Names a client uses to reach the data plane.
        control_plane: Control-plane handle (worker group or stub) or ``None`` to
            run head-less, driven only by ``task_source``.
        task_source: Where work comes from. Defaults to the control plane.
        service_name: Logical service name.
        channel_factory: ``name -> channel`` factory. Defaults to the vendored
            scheduler's ``Channel.create``; tests inject fakes.
        poll_interval: Seconds between service loop iterations.
        status_interval: Seconds between status publications.
        stop_when_task_source_exhausted: Whether the service shuts down once the
            task source reports exhaustion (the eval-run default).
    """

    #: Task kinds this controller can execute; used by the control plane to
    #: reject work the service cannot serve.
    ACCEPTED_TASK_KINDS: tuple[TaskKind, ...] = ()

    def __init__(
        self,
        cfg: DictConfig,
        *,
        endpoints: ServiceEndpoints,
        control_plane: Optional[Any] = None,
        task_source: Optional[TaskSource] = None,
        service_name: str = DEFAULT_SERVICE_NAME,
        channel_factory: Optional[Callable[..., Any]] = None,
        poll_interval: float = 0.5,
        status_interval: float = 5.0,
        stop_when_task_source_exhausted: bool = False,
    ) -> None:
        self.cfg = cfg
        self.endpoints = endpoints
        self.control_plane = control_plane
        self.service_name = service_name
        self.poll_interval = float(poll_interval)
        self.status_interval = float(status_interval)
        self.stop_when_task_source_exhausted = bool(stop_when_task_source_exhausted)

        self._channel_factory = channel_factory
        self._channels: dict[str, Any] = {}
        self._state = ServiceState.STARTING
        self._error: Optional[str] = None
        self._started_at = time.monotonic()
        self._stop_requested = False
        self._num_tasks_completed = 0
        self._num_outputs_published = 0
        self._num_weight_updates = 0
        self._served_version = -1
        self._metrics: dict[str, float] = {}
        self._last_status_at = 0.0

        if task_source is None and control_plane is not None:
            from rlinf_rollout.serve.task_source import ControlPlaneTaskSource

            task_source = ControlPlaneTaskSource(control_plane)
        self.task_source = task_source

    # ------------------------------------------------------------- properties

    @property
    def kind(self) -> str:
        """Rollout chain this controller drives (``embodied`` / ``llm``)."""
        return str(OmegaConf.select(self.cfg, "rollout.kind", default=""))

    @property
    def mode(self) -> str:
        """Service mode (``collect`` / ``eval``)."""
        return str(OmegaConf.select(self.cfg, "rollout.mode", default=""))

    @property
    def state(self) -> ServiceState:
        """Lifecycle state of the service."""
        return self._state

    # ---------------------------------------------------------------- channels

    def channel(self, name: str, **kwargs: Any) -> Any:
        """Return (and create on first use) the channel called ``name``."""
        if name not in self._channels:
            factory = self._channel_factory
            if factory is None:
                from rlinf_rollout.scheduler import Channel

                factory = Channel.create
            self._channels[name] = factory(name, **kwargs)
        return self._channels[name]

    # --------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        """Initialize workers and start the resident loops."""
        self._started_at = time.monotonic()
        await self._start_workers()
        self._state = ServiceState.RUNNING
        self._publish_status(force=True)

    async def serve_forever(self) -> None:
        """Run the service loop until a stop is requested or the source drains."""
        try:
            if self._state is ServiceState.STARTING:
                await self.start()
            while not await self._should_stop():
                await self._tick()
                self._publish_status()
                await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced through the status
            self._state = ServiceState.FAILED
            self._error = f"{type(exc).__name__}: {exc}"
            self._publish_status(force=True)
            raise
        finally:
            if self._state is not ServiceState.FAILED:
                self._state = ServiceState.DRAINING
                self._publish_status(force=True)

    async def shutdown(self) -> None:
        """Stop the worker groups and mark the service stopped."""
        try:
            await self._stop_workers()
        finally:
            self._state = ServiceState.STOPPED
            self._publish_status(force=True)

    def request_stop(self) -> None:
        """Ask the service loop to return after the current iteration."""
        self._stop_requested = True

    async def _should_stop(self) -> bool:
        """Whether the service loop must return."""
        if self._stop_requested:
            return True
        if self.control_plane is not None:
            requested = resolve_group_call(self.control_plane.stop_requested())
            if any(bool(value) for value in requested):
                self._stop_requested = True
                return True
        if (
            self.stop_when_task_source_exhausted
            and self.task_source is not None
            and await self.task_source.exhausted()
        ):
            return True
        return False

    # ------------------------------------------------------------------ status

    def status(self) -> ServiceStatus:
        """Build the current status snapshot."""
        num_pending = 0
        if self.control_plane is not None:
            pending = resolve_group_call(self.control_plane.num_pending_tasks())
            num_pending = int(pending[0] or 0) if pending else 0
        return ServiceStatus(
            service_name=self.service_name,
            kind=self.kind,
            mode=self.mode,
            state=self._state,
            endpoints=self.endpoints,
            served_version=self._served_version,
            num_tasks_pending=num_pending,
            num_tasks_completed=self._num_tasks_completed,
            num_outputs_published=self._num_outputs_published,
            num_weight_updates=self._num_weight_updates,
            uptime_seconds=time.monotonic() - self._started_at,
            metrics=dict(self._metrics),
            error=self._error,
        )

    def _publish_status(self, *, force: bool = False) -> None:
        """Publish a status snapshot to the control plane, rate-limited."""
        if self.control_plane is None:
            return
        now = time.monotonic()
        if not force and now - self._last_status_at < self.status_interval:
            return
        self._last_status_at = now
        resolve_group_call(self.control_plane.publish_status(self.status()))

    # ------------------------------------------------------------ weight sync

    async def _drain_weight_requests(self, max_updates: int = 4) -> int:
        """Execute queued client weight pushes; return how many ran."""
        if self.control_plane is None:
            return 0
        applied = 0
        for _ in range(max_updates):
            pending = resolve_group_call(self.control_plane.next_weight_request())
            request = pending[0] if pending else None
            if request is None:
                break
            started = time.perf_counter()
            try:
                result = await self.apply_weight_update(request)
            except Exception as exc:  # noqa: BLE001 - reported to the client
                result = WeightPushResult(
                    version=request.version,
                    status=WeightPushStatus.FAILED,
                    served_version=self._served_version,
                    elapsed_seconds=time.perf_counter() - started,
                    error=f"{type(exc).__name__}: {exc}",
                )
            resolve_group_call(self.control_plane.publish_weight_result(result))
            if result.status is not WeightPushStatus.FAILED:
                applied += 1
                self._num_weight_updates += 1
        return applied

    @staticmethod
    def _aggregate_acks(
        request: WeightUpdateRequest,
        acks: Sequence[WeightUpdateAck],
        elapsed_seconds: float,
    ) -> WeightPushResult:
        """Fold per-receiver acks into one :class:`WeightPushResult`."""
        failures = [ack for ack in acks if ack.status is WeightUpdateStatus.FAILED]
        served = min((ack.served_version for ack in acks), default=-1)
        if failures:
            return WeightPushResult(
                version=request.version,
                status=WeightPushStatus.FAILED,
                served_version=served,
                acks=tuple(acks),
                elapsed_seconds=elapsed_seconds,
                error="; ".join(
                    f"{ack.receiver_id or '?'}: {ack.error}" for ack in failures
                ),
            )
        return WeightPushResult(
            version=request.version,
            status=WeightPushStatus.APPLIED,
            served_version=served,
            acks=tuple(acks),
            elapsed_seconds=elapsed_seconds,
        )

    # -------------------------------------------------------------- task loop

    async def _next_tasks(self, max_num_tasks: int) -> list[RolloutTask]:
        """Pull tasks from the source, if any."""
        if self.task_source is None or max_num_tasks <= 0:
            return []
        return await self.task_source.next_batch(max_num_tasks)

    async def _report_task_done(
        self, task: RolloutTask, error: Optional[str] = None
    ) -> None:
        """Report task completion to the source and count it."""
        self._num_tasks_completed += 1
        if self.task_source is not None:
            await self.task_source.report_done(task.task_id, error)

    # ------------------------------------------------------------- extension

    @abstractmethod
    async def _start_workers(self) -> None:
        """Initialize worker groups and start the resident coroutines."""

    @abstractmethod
    async def _stop_workers(self) -> None:
        """Tell the worker groups to stop and wait for the resident loops."""

    @abstractmethod
    async def _tick(self) -> None:
        """Run one iteration of the service loop."""

    @abstractmethod
    async def apply_weight_update(
        self, request: WeightUpdateRequest
    ) -> WeightPushResult:
        """Apply one client-pushed weight update.

        Args:
            request: The client's update description.

        Returns:
            The aggregated outcome, one ack per receiver where available.
        """


class EmbodiedRolloutController(RolloutController):
    """Drives the embodied chain: env workers stepping against a HF policy.

    Replaces ``AsyncEmbodiedRunner`` / ``EmbodiedEvalRunner`` for a rollout-only
    deployment. The collect loop is *continuous and config-driven*: episode shape
    comes from ``env.train`` and the env workers publish trajectories through
    their :class:`~rlinf_rollout.api.v1.TrajectorySink` without being asked. The
    only task kind a client can submit is therefore
    :attr:`~rlinf_rollout.api.v1.TaskKind.EMBODIED_EVAL`, which runs one
    evaluation pass over ``env.eval``.

    Args:
        cfg: Validated ``rollout.kind: embodied`` config.
        rollout_group: Launched ``AsyncMultiStepRolloutWorker`` group.
        env_group: Launched ``AsyncEnvWorker`` group.
        reward_group: Optional reward worker group. Not vendored yet; passing one
            is only useful for a custom deployment.
        **kwargs: Forwarded to :class:`RolloutController`.
    """

    ACCEPTED_TASK_KINDS = (TaskKind.EMBODIED_EVAL,)

    def __init__(
        self,
        cfg: DictConfig,
        *,
        rollout_group: Any,
        env_group: Any,
        reward_group: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(cfg, **kwargs)
        self.rollout_group = rollout_group
        self.env_group = env_group
        self.reward_group = reward_group
        self._env_handle: Optional[Any] = None
        self._rollout_handle: Optional[Any] = None
        self._reward_handle: Optional[Any] = None
        self._num_eval_rounds = 0

    # ------------------------------------------------------------- channels

    @property
    def _env_channel(self) -> Any:
        """Rollout -> env channel (action chunks)."""
        return self.channel(f"{self.service_name}_env")

    @property
    def _rollout_channel(self) -> Any:
        """Env -> rollout channel (observations)."""
        return self.channel(f"{self.service_name}_rollout")

    @property
    def _trajectory_channel(self) -> Any:
        """Client-facing channel carrying ``api/v1`` trajectories."""
        return self.channel(self.endpoints.output_channel)

    @property
    def _env_metric_channel(self) -> Any:
        """Env-worker metric channel."""
        return self.channel(f"{self.service_name}_env_metric")

    @property
    def _rollout_metric_channel(self) -> Any:
        """Rollout-worker metric channel."""
        return self.channel(f"{self.service_name}_rollout_metric")

    @property
    def _collecting(self) -> bool:
        """Whether the service collects training trajectories."""
        return self.mode == RolloutMode.COLLECT

    # ------------------------------------------------------------- lifecycle

    async def _start_workers(self) -> None:
        """Initialize both groups and, in collect mode, start the resident loops."""
        resolve_group_call(self.rollout_group.init_worker())
        resolve_group_call(self.env_group.init_worker())

        if not self._collecting:
            return

        reward_channel = None
        if self.reward_group is not None:
            reward_channel = self.channel(f"{self.service_name}_reward")
            self._reward_handle = self.reward_group.compute_rewards_async(
                input_channel=reward_channel,
                output_channel=self._env_channel,
            )
        self._env_handle = self.env_group.interact(
            input_channel=self._env_channel,
            rollout_channel=self._rollout_channel,
            reward_channel=reward_channel,
            trajectory_channel=self._trajectory_channel,
            metric_channel=self._env_metric_channel,
        )
        self._rollout_handle = self.rollout_group.generate(
            input_channel=self._rollout_channel,
            output_channel=self._env_channel,
            metric_channel=self._rollout_metric_channel,
        )

    async def _stop_workers(self) -> None:
        """Cancel the resident loops and wait for both groups to return."""
        for group in (self.env_group, self.rollout_group, self.reward_group):
            if group is None:
                continue
            stop = getattr(group, "stop", None)
            if stop is not None:
                resolve_group_call(stop())
        for handle in (self._env_handle, self._rollout_handle, self._reward_handle):
            if handle is not None:
                resolve_group_call(handle)
        self._env_handle = self._rollout_handle = self._reward_handle = None

    # ------------------------------------------------------------ service loop

    async def _tick(self) -> None:
        """Apply weight pushes, harvest metrics, and run requested eval rounds."""
        await self._drain_weight_requests()
        self._harvest_metrics()
        for task in await self._next_tasks(1):
            await self._run_eval_round(task)

    def _harvest_metrics(self) -> None:
        """Drain the worker metric channels into the published status."""
        for channel, prefix in (
            (self._channels.get(f"{self.service_name}_env_metric"), "env"),
            (self._channels.get(f"{self.service_name}_rollout_metric"), "rollout"),
        ):
            if channel is None:
                continue
            for payload in _drain_channel(channel):
                self._metrics.update(_flatten_metrics(payload, prefix))
        traj_channel = self._channels.get(self.endpoints.output_channel)
        if traj_channel is not None:
            qsize = getattr(traj_channel, "qsize", None)
            if qsize is not None:
                self._metrics["output_channel_qsize"] = float(qsize())

    async def _run_eval_round(self, task: RolloutTask) -> None:
        """Run one evaluation pass and fold its metrics into the status.

        Args:
            task: The ``EMBODIED_EVAL`` task that requested the round.
        """
        from rlinf_rollout.utils.metric_utils import compute_evaluate_metrics

        error: Optional[str] = None
        try:
            env_handle = self.env_group.evaluate(
                input_channel=self._env_channel,
                rollout_channel=self._rollout_channel,
            )
            rollout_handle = None
            if not bool(OmegaConf.select(self.cfg, "rollout.decoupled", default=False)):
                rollout_handle = self.rollout_group.evaluate(
                    input_channel=self._rollout_channel,
                    output_channel=self._env_channel,
                )
            env_results = resolve_group_call(env_handle)
            if rollout_handle is not None:
                resolve_group_call(rollout_handle)
            metrics = compute_evaluate_metrics(
                [result for result in env_results if result is not None]
            )
            self._num_eval_rounds += 1
            self._metrics.update(
                {f"eval/{key}": float(value) for key, value in metrics.items()}
            )
            self._metrics["eval/num_rounds"] = float(self._num_eval_rounds)
        except Exception as exc:  # noqa: BLE001 - reported back to the client
            error = f"{type(exc).__name__}: {exc}"
        await self._report_task_done(task, error)

    # ------------------------------------------------------------ weight sync

    async def apply_weight_update(
        self, request: WeightUpdateRequest
    ) -> WeightPushResult:
        """Install new policy weights on the rollout workers.

        ``COLLECTIVE`` pushes are receiver-driven: this call blocks the service
        loop while the workers wait for the trainer's broadcast, so the client must
        run its sender concurrently (see
        :meth:`rlinf_rollout.client.RolloutClient.push_weights`). When
        ``rollout.weight_sync.no_wait`` is set the update is only *queued* on the
        workers and the result is
        :attr:`~rlinf_rollout.serve.protocol.WeightPushStatus.ACCEPTED`.
        """
        started = time.perf_counter()
        if not self._collecting:
            return WeightPushResult(
                version=request.version,
                status=WeightPushStatus.FAILED,
                served_version=self._served_version,
                elapsed_seconds=time.perf_counter() - started,
                error="weight sync is disabled in eval mode (rollout.mode=eval).",
            )

        no_wait = bool(
            OmegaConf.select(self.cfg, "rollout.weight_sync.no_wait", default=False)
        )
        if request.transport is WeightTransport.COLLECTIVE and no_wait:
            resolve_group_call(self.rollout_group.request_weight_sync())
            return WeightPushResult(
                version=request.version,
                status=WeightPushStatus.ACCEPTED,
                served_version=self._served_version,
                elapsed_seconds=time.perf_counter() - started,
            )

        acks = resolve_group_call(self.rollout_group.receive_weight_update(request))
        result = self._aggregate_acks(request, acks, time.perf_counter() - started)
        if result.status is WeightPushStatus.APPLIED:
            self._served_version = result.served_version
        return result


def _drain_channel(channel: Any, max_items: int = 256) -> list[Any]:
    """Pop up to ``max_items`` ready items from ``channel`` without blocking."""
    items: list[Any] = []
    for _ in range(max_items):
        try:
            items.append(channel.get_nowait())
        except asyncio.QueueEmpty:
            break
        except Exception:  # noqa: BLE001 - an empty fake channel may signal this way
            break
    return items


def _flatten_metrics(payload: Any, prefix: str) -> dict[str, float]:
    """Flatten a worker metric payload into scalar ``prefix/...`` entries."""
    flat: dict[str, float] = {}
    if not isinstance(payload, dict):
        return flat
    rank = payload.get("rank")
    for section in ("env", "time", "metrics"):
        values = payload.get(section)
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            scalar = _as_scalar(value)
            if scalar is None:
                continue
            name = key if "/" in key else f"{prefix}/{key}"
            if rank is not None:
                name = f"{name}/rank{rank}"
            flat[name] = scalar
    return flat


def _as_scalar(value: Any) -> Optional[float]:
    """Reduce a metric value (scalar, list or tensor) to a float mean."""
    if isinstance(value, (int, float)):
        return float(value)
    mean = getattr(value, "mean", None)
    if mean is None:
        return None
    try:
        return float(mean())
    except Exception:  # noqa: BLE001 - non-numeric payloads are simply skipped
        return None


class LLMRolloutController(RolloutController):
    """Drives the LLM chain: SGLang / vLLM engines generating token sequences.

    Unlike the embodied chain this one is *task-driven*: every
    :attr:`~rlinf_rollout.api.v1.TaskKind.LLM_GENERATION` task is split across the
    engine processes, generated, converted to
    :class:`~rlinf_rollout.api.v1.RolloutResult` and handed to the sink.

    Prompts must arrive tokenized (:attr:`PromptSpec.input_ids`): the driver holds
    no tokenizer, so text-only prompts are rejected with an explanatory error
    instead of being silently mis-encoded. The daemon's prompt-file source
    tokenizes up front (see :class:`~rlinf_rollout.serve.task_source.JsonlPromptTaskSource`).

    Args:
        cfg: Validated ``rollout.kind: llm`` config.
        rollout_group: Launched engine worker group.
        num_engines: Number of engine processes, i.e. the placement's
            ``rollout_dp_size``. One request is queued per engine, so this must
            match the launched group's size.
        sink: Destination for converted results; built from ``sink.num_shards``
            over the client-facing channel when omitted.
        **kwargs: Forwarded to :class:`RolloutController`.
    """

    ACCEPTED_TASK_KINDS = (TaskKind.LLM_GENERATION,)

    def __init__(
        self,
        cfg: DictConfig,
        *,
        rollout_group: Any,
        num_engines: int = 1,
        sink: Optional[TrajectorySink] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(cfg, **kwargs)
        if num_engines < 1:
            raise ValueError(f"num_engines must be positive, got {num_engines}.")
        self.rollout_group = rollout_group
        self.num_engines = int(num_engines)
        self.sink = sink
        self._group_size = int(
            OmegaConf.select(cfg, "rollout.group_size", default=1) or 1
        )
        timeout = OmegaConf.select(
            cfg, "rollout.serve.generation_timeout_seconds", default=None
        )
        self._generation_timeout = float(timeout) if timeout else None

    # -------------------------------------------------------------- channels

    @property
    def _engine_input_channel(self) -> Any:
        """Driver -> engine channel carrying ``RolloutRequest`` batches."""
        return self.channel(f"{self.service_name}_engine_in")

    @property
    def _engine_output_channel(self) -> Any:
        """Engine -> driver channel carrying internal rollout results."""
        return self.channel(f"{self.service_name}_engine_out")

    @property
    def _api_output_channel(self) -> Any:
        """Client-facing channel carrying ``api/v1`` rollout results."""
        return self.channel(self.endpoints.output_channel)

    # ------------------------------------------------------------- lifecycle

    async def _start_workers(self) -> None:
        """Bring the engines up and build the output sink."""
        resolve_group_call(self.rollout_group.init_worker())
        if self.sink is None:
            from rlinf_rollout.sinks import ChannelTrajectorySink

            num_shards = int(self.endpoints.num_output_partitions)
            self.sink = ChannelTrajectorySink(
                self._api_output_channel,
                ConsumerSpec(
                    num_partitions=num_shards,
                    axis=PartitionAxis.SEQUENCE
                    if num_shards > 1
                    else PartitionAxis.NONE,
                ),
            )

    async def _stop_workers(self) -> None:
        """Flush the sink and shut the engines down."""
        if self.sink is not None:
            await self.sink.close()
        for name in ("abort_generation", "shutdown", "stop"):
            method = getattr(self.rollout_group, name, None)
            if method is None:
                continue
            try:
                resolve_group_call(method())
            except Exception:  # noqa: BLE001 - best-effort teardown
                continue

    # ----------------------------------------------------------- service loop

    async def _tick(self) -> None:
        """Apply weight pushes, then run one generation task if available."""
        await self._drain_weight_requests()
        for task in await self._next_tasks(1):
            await self._run_generation(task)

    async def _run_generation(self, task: RolloutTask) -> None:
        """Generate one task's prompt batch and publish the results.

        Args:
            task: An ``LLM_GENERATION`` task carrying tokenized prompts.
        """
        error: Optional[str] = None
        try:
            num_published = await self._generate_and_publish(task)
            self._metrics["llm/last_task_outputs"] = float(num_published)
        except Exception as exc:  # noqa: BLE001 - reported back to the client
            error = f"{type(exc).__name__}: {exc}"
        await self._report_task_done(task, error)

    async def _generate_and_publish(self, task: RolloutTask) -> int:
        """Dispatch ``task`` to the engines and push converted results to the sink.

        A generation that exceeds ``rollout.serve.generation_timeout_seconds``
        surfaces as the timeout error raised by :meth:`_collect_results`.

        Returns:
            Number of ``api/v1`` payloads published.

        Raises:
            ValueError: If the task has no tokenized prompts.
        """
        from rlinf_rollout.data.convert import rollout_result_to_api
        from rlinf_rollout.data.io_struct import RolloutRequest
        from rlinf_rollout.utils.data_iter_utils import split_list

        prompts = task.prompts
        if prompts is None or not prompts.input_ids:
            raise ValueError(
                "LLM generation needs tokenized prompts: PromptSpec.input_ids is "
                "empty. The rollout service holds no tokenizer, so text-only "
                "prompts must be encoded by the submitter."
            )
        input_ids = [list(ids) for ids in prompts.input_ids]
        num_prompts = len(input_ids)
        num_samples = task.sampling.n if task.sampling is not None else self._group_size
        answers = list(prompts.answers or [None] * num_prompts)
        image_data = list(prompts.image_data or [None] * num_prompts)
        multi_modal = list(prompts.multi_modal_inputs or [None] * num_prompts)

        splits = [
            split_list(values, self.num_engines, enforce_divisible_batch=False)
            for values in (input_ids, answers, image_data, multi_modal)
        ]
        for ids_chunk, answer_chunk, image_chunk, mm_chunk in zip(*splits, strict=True):
            self._engine_input_channel.put(
                RolloutRequest(
                    n=num_samples,
                    input_ids=ids_chunk,
                    answers=answer_chunk,
                    image_data=image_chunk,
                    multi_modal_inputs=mm_chunk,
                ),
                async_op=True,
            )

        handle = self.rollout_group.rollout(
            input_channel=self._engine_input_channel,
            output_channel=self._engine_output_channel,
        )
        results = await self._collect_results(num_prompts)
        resolve_group_call(handle)

        published = 0
        for result in results:
            payload = rollout_result_to_api(
                result,
                request_ids=(task.task_id,) * int(result.num_sequence),
                versions=(max(self._served_version, 0),) * int(result.num_sequence),
                producer=f"{self.endpoints.rollout_group}",
                metadata={"task_id": task.task_id},
            )
            await self.sink.put(payload)
            published += 1
        self._num_outputs_published += published
        await self.sink.flush()
        return published

    async def _collect_results(self, expected: int) -> list[Any]:
        """Wait for ``expected`` sequence-group results from the engines.

        Args:
            expected: Number of prompts dispatched, i.e. sequence groups.

        Returns:
            The internal rollout results, in arrival order.

        Raises:
            TimeoutError: If fewer than ``expected`` results arrived in time.
        """
        results: list[Any] = []
        deadline = (
            None
            if self._generation_timeout is None
            else time.monotonic() + self._generation_timeout
        )
        while len(results) < expected:
            drained = _drain_channel(self._engine_output_channel, expected)
            results.extend(drained)
            if len(results) >= expected:
                break
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(
                    f"engines returned {len(results)}/{expected} sequence groups "
                    f"within {self._generation_timeout}s."
                )
            await asyncio.sleep(0.01)
        return results

    # ------------------------------------------------------------ weight sync

    async def apply_weight_update(
        self, request: WeightUpdateRequest
    ) -> WeightPushResult:
        """Have the engines receive one weight broadcast from the trainer.

        Only :attr:`~rlinf_rollout.api.v1.WeightTransport.COLLECTIVE` is
        supported. The engines apply whatever the sender described at init time
        (:class:`~rlinf_rollout.weight_sync.EngineWeightSyncSetup`) and do not
        report a version of their own, so the ack echoes the requested version.
        """
        started = time.perf_counter()
        if self.mode != RolloutMode.COLLECT:
            return WeightPushResult(
                version=request.version,
                status=WeightPushStatus.FAILED,
                served_version=self._served_version,
                elapsed_seconds=time.perf_counter() - started,
                error="weight sync is disabled in eval mode (rollout.mode=eval).",
            )
        if request.transport is not WeightTransport.COLLECTIVE:
            return WeightPushResult(
                version=request.version,
                status=WeightPushStatus.FAILED,
                served_version=self._served_version,
                elapsed_seconds=time.perf_counter() - started,
                error=(
                    f"transport {request.transport.value!r} is not supported by the "
                    "LLM chain; engines receive weights over the collective "
                    "configured at init time."
                ),
            )

        resolve_group_call(self.rollout_group.sync_weights())
        acks = tuple(
            WeightUpdateAck(
                version=request.version,
                status=WeightUpdateStatus.APPLIED,
                served_version=request.version,
                receiver_id=f"{self.endpoints.rollout_group}:{rank}",
            )
            for rank in range(self.num_engines)
        )
        result = self._aggregate_acks(request, acks, time.perf_counter() - started)
        self._served_version = result.served_version
        return result
