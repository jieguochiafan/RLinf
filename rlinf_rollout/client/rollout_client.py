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

"""Training-side SDK for a running rollout service.

The SDK is deliberately small and framework-agnostic — three verbs:

* :meth:`RolloutClient.push_weights` — install a new policy version, either by
  handing over a checkpoint path or by arming the rollout-side receivers while the
  trainer's own sender coroutine broadcasts.
* :meth:`RolloutClient.get_trajectories` — drain ``api/v1`` payloads
  (:class:`~rlinf_rollout.api.v1.Trajectory` for embodied,
  :class:`~rlinf_rollout.api.v1.RolloutResult` for LLM).
* :meth:`RolloutClient.submit_tasks` — hand work to the service.

Nothing here exposes a Ray object: the client resolves the service by name, and
all traffic goes through the control plane and the named output channel.
"""

from __future__ import annotations

import time
from typing import Any, Awaitable, Callable, Iterable, Optional, Sequence, Union

from rlinf_rollout.api.v1 import (
    PromptSpec,
    RolloutMode,
    RolloutResult,
    RolloutTask,
    SamplingParams,
    SourceTopology,
    TaskKind,
    Trajectory,
    WeightSyncMode,
    WeightTransport,
    WeightUpdateRequest,
)
from rlinf_rollout.serve.protocol import (
    DEFAULT_SERVICE_NAME,
    ServiceState,
    ServiceStatus,
    TaskAck,
    WeightPushResult,
    WeightPushStatus,
    control_group_name,
)

__all__ = ["RolloutClient", "RolloutClientError", "WeightSender", "make_prompt_tasks"]

#: Sender coroutine signature: ``await sender(request)`` performs the broadcast.
WeightSender = Callable[[WeightUpdateRequest], Union[None, Awaitable[None]]]


class RolloutClientError(RuntimeError):
    """Raised when the service cannot be reached or refuses an operation."""


class RolloutClient:
    """Client handle on a rollout service started by ``rollout-serve``.

    The client attaches to the service's Ray cluster lazily, on first use, and
    resolves the control plane by the well-known group name
    ``"<service_name>_control"``.

    Args:
        service_name: Name the service was started with
            (``rollout.serve.name`` / ``--name``).
        control_plane: Pre-resolved control-plane handle. Tests and in-process
            drivers pass one directly; otherwise it is resolved on demand.
        output_channel: Overrides the output channel name published by the
            service (only needed when connecting before the service publishes a
            status).
        connect_timeout: Seconds to wait for the service to appear.
        poll_interval: Seconds between polls while waiting on the service.
    """

    def __init__(
        self,
        service_name: str = DEFAULT_SERVICE_NAME,
        *,
        control_plane: Optional[Any] = None,
        output_channel: Optional[str] = None,
        connect_timeout: float = 300.0,
        poll_interval: float = 0.2,
    ) -> None:
        self.service_name = service_name
        self.connect_timeout = float(connect_timeout)
        self.poll_interval = float(poll_interval)
        self._control_plane = control_plane
        self._output_channel_name = output_channel
        self._output_channel: Optional[Any] = None

    # ------------------------------------------------------------- connection

    @property
    def control_group_name(self) -> str:
        """Worker-group name of the service's control plane."""
        return control_group_name(self.service_name)

    def _resolve_control_plane(self) -> Any:
        """Return the control-plane handle, attaching to the cluster if needed."""
        if self._control_plane is not None:
            return self._control_plane
        try:
            from rlinf_rollout.scheduler import Cluster
            from rlinf_rollout.scheduler.worker.worker_group import WorkerGroup
            from rlinf_rollout.serve.control_worker import ControlPlaneWorker

            Cluster()  # attach to the service's existing cluster
            self._control_plane = WorkerGroup.from_group_name(
                ControlPlaneWorker, self.control_group_name
            )
        except Exception as exc:  # noqa: BLE001 - wrapped for a clear message
            raise RolloutClientError(
                f"cannot reach rollout service {self.service_name!r} "
                f"(control group {self.control_group_name!r}): {exc}"
            ) from exc
        return self._control_plane

    @staticmethod
    def _call(result: Any) -> Any:
        """Unwrap a worker-group call result (single-rank control plane)."""
        wait = getattr(result, "wait", None)
        if wait is None:
            return result
        values = wait()
        if isinstance(values, (list, tuple)):
            return values[0] if values else None
        return values

    def close(self) -> None:
        """Drop cached handles. The service keeps running."""
        self._control_plane = None
        self._output_channel = None

    def __enter__(self) -> "RolloutClient":
        """Enter a context that closes the client on exit."""
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Close the client."""
        self.close()

    # ----------------------------------------------------------------- status

    def status(self) -> ServiceStatus:
        """Return the service's latest published status snapshot."""
        status = self._call(self._resolve_control_plane().get_status())
        if status is None:
            raise RolloutClientError(
                f"service {self.service_name!r} has not published a status yet."
            )
        return status

    def wait_until_ready(self, timeout: Optional[float] = None) -> ServiceStatus:
        """Block until the service reports :attr:`ServiceState.RUNNING`.

        Args:
            timeout: Seconds to wait; defaults to ``connect_timeout``.

        Returns:
            The first running status snapshot.

        Raises:
            RolloutClientError: On timeout or if the service failed.
        """
        deadline = time.monotonic() + (
            self.connect_timeout if timeout is None else float(timeout)
        )
        last: Optional[ServiceStatus] = None
        while time.monotonic() < deadline:
            try:
                last = self.status()
            except RolloutClientError:
                last = None
            if last is not None:
                if last.state is ServiceState.RUNNING:
                    return last
                if last.state is ServiceState.FAILED:
                    raise RolloutClientError(f"service failed: {last.error}")
            time.sleep(self.poll_interval)
        state = last.state.value if last is not None else "unreachable"
        raise RolloutClientError(
            f"service {self.service_name!r} did not become ready (state={state})."
        )

    # ------------------------------------------------------------- tasks out

    def submit_tasks(self, tasks: Sequence[RolloutTask]) -> TaskAck:
        """Hand work to the service.

        Args:
            tasks: Tasks to run. Kinds the service does not serve come back in
                :attr:`TaskAck.rejected` rather than raising.

        Returns:
            The acknowledgement.
        """
        ack = self._call(self._resolve_control_plane().submit_tasks(list(tasks)))
        if ack is None:
            raise RolloutClientError("control plane returned no acknowledgement.")
        return ack

    def submit_prompts(
        self,
        input_ids: Sequence[Sequence[int]],
        *,
        n: int = 1,
        answers: Optional[Sequence[Any]] = None,
        sampling: Optional[SamplingParams] = None,
        mode: Union[str, RolloutMode] = RolloutMode.TRAIN,
        priority: int = 0,
    ) -> TaskAck:
        """Submit one tokenized prompt batch as an LLM generation task.

        Args:
            input_ids: Tokenized prompts. The service holds no tokenizer, so text
                prompts must be encoded by the caller.
            n: Samples per prompt.
            answers: Reference answers echoed back with the results.
            sampling: Full sampling knobs; built from ``n`` when omitted.
            mode: ``train`` or ``eval``.
            priority: Scheduling priority; higher runs first.

        Returns:
            The acknowledgement.
        """
        task = RolloutTask(
            kind=TaskKind.LLM_GENERATION,
            mode=RolloutMode(mode),
            priority=priority,
            sampling=sampling or SamplingParams(n=n),
            prompts=PromptSpec(
                input_ids=[list(ids) for ids in input_ids],
                answers=list(answers) if answers is not None else None,
            ),
        )
        return self.submit_tasks([task])

    def request_eval(
        self,
        env_type: str = "configured",
        *,
        num_envs: int = 1,
        priority: int = 0,
    ) -> TaskAck:
        """Ask an embodied service to run one evaluation pass over ``env.eval``.

        The episode spec is informational: an embodied service builds its envs at
        startup from ``env.eval``, so the pass shape comes from the service's own
        config and only the *request* comes from the client.

        Args:
            env_type: Env type recorded in the task, for the service's logs.
            num_envs: Env count recorded in the task, for the service's logs.
            priority: Scheduling priority; higher runs first.

        Returns:
            The acknowledgement.
        """
        from rlinf_rollout.serve.task_source import episode_task

        task = episode_task(
            env_type=env_type,
            num_envs=num_envs,
            mode=RolloutMode.EVAL,
            priority=priority,
        )
        return self.submit_tasks([task])

    # ------------------------------------------------------------ weights out

    def push_weights(
        self,
        request: WeightUpdateRequest,
        *,
        sender: Optional[WeightSender] = None,
        wait: bool = True,
        timeout: Optional[float] = None,
    ) -> WeightPushResult:
        """Install a new policy version on the rollout side.

        Ordering matters for the collective transport, and this method encodes it:
        the request is queued first (so the service arms its receivers), then
        ``sender`` runs (the trainer's broadcast), then the result is polled.

        Args:
            request: The update description. Use
                :attr:`~rlinf_rollout.api.v1.WeightTransport.CHECKPOINT` with
                ``checkpoint_path`` for the loosely coupled path, or
                :attr:`~rlinf_rollout.api.v1.WeightTransport.COLLECTIVE` together
                with ``sender`` for the NCCL path.
            sender: Callable performing the trainer-side broadcast. May be a
                coroutine function; it is awaited on a private event loop.
            wait: Whether to block until the service reports the outcome.
            timeout: Seconds to wait for the outcome; defaults to
                ``connect_timeout``.

        Returns:
            The aggregated outcome. ``wait=False`` returns an
            :attr:`~rlinf_rollout.serve.protocol.WeightPushStatus.ACCEPTED`
            placeholder; with ``wait=True`` a missing outcome surfaces as
            :class:`RolloutClientError` from :meth:`wait_for_weights`.
        """
        control_plane = self._resolve_control_plane()
        self._call(control_plane.request_weight_update(request))

        if sender is not None:
            outcome = sender(request)
            if hasattr(outcome, "__await__"):
                import asyncio

                asyncio.run(_await(outcome))

        if not wait:
            return WeightPushResult(
                version=request.version, status=WeightPushStatus.ACCEPTED
            )
        return self.wait_for_weights(request.version, timeout=timeout)

    def push_checkpoint(
        self,
        checkpoint_path: str,
        version: int,
        *,
        wait: bool = True,
        timeout: Optional[float] = None,
    ) -> WeightPushResult:
        """Push weights by pointing the service at a checkpoint file.

        Args:
            checkpoint_path: Path readable by the rollout workers.
            version: Monotonically increasing weight version.
            wait: Whether to block until the service reports the outcome.
            timeout: Seconds to wait for the outcome.

        Returns:
            The aggregated outcome.
        """
        request = WeightUpdateRequest(
            version=version,
            transport=WeightTransport.CHECKPOINT,
            mode=WeightSyncMode.BUCKET,
            checkpoint_path=checkpoint_path,
            source=SourceTopology(group_name="client"),
        )
        return self.push_weights(request, wait=wait, timeout=timeout)

    def wait_for_weights(
        self, version: int, *, timeout: Optional[float] = None
    ) -> WeightPushResult:
        """Poll the service until it reports the outcome of ``version``.

        Args:
            version: Version passed to :meth:`push_weights`.
            timeout: Seconds to wait; defaults to ``connect_timeout``.

        Returns:
            The aggregated outcome.

        Raises:
            RolloutClientError: On timeout.
        """
        control_plane = self._resolve_control_plane()
        deadline = time.monotonic() + (
            self.connect_timeout if timeout is None else float(timeout)
        )
        while True:
            result = self._call(control_plane.get_weight_result(version))
            if result is not None:
                return result
            if time.monotonic() > deadline:
                raise RolloutClientError(
                    f"no weight-push result for version {version} within the timeout."
                )
            time.sleep(self.poll_interval)

    # -------------------------------------------------------------- data out

    def _resolve_output_channel(self) -> Any:
        """Connect to the service's output channel, discovering its name if needed."""
        if self._output_channel is not None:
            return self._output_channel
        name = self._output_channel_name
        if name is None:
            name = self.status().endpoints.output_channel
        if not name:
            raise RolloutClientError(
                "the service did not publish an output channel name."
            )
        try:
            from rlinf_rollout.scheduler import Channel, Cluster

            Cluster()
            self._output_channel = Channel.connect(name, current_worker=None)
        except Exception as exc:  # noqa: BLE001 - wrapped for a clear message
            raise RolloutClientError(
                f"cannot connect to output channel {name!r}: {exc}"
            ) from exc
        self._output_channel_name = name
        return self._output_channel

    def get_trajectories(
        self,
        max_items: int = 1,
        *,
        timeout: Optional[float] = None,
        key: Optional[Any] = None,
    ) -> list[Union[Trajectory, RolloutResult]]:
        """Drain up to ``max_items`` ``api/v1`` payloads from the output channel.

        Args:
            max_items: Upper bound on the number of payloads returned.
            timeout: Seconds to wait for the *first* payload; ``None`` waits
                forever, ``0`` polls once and may return an empty list.
            key: Channel key to read, for sharded sinks. Defaults to the
                channel's default key.

        Returns:
            The payloads received, oldest first.
        """
        import asyncio

        channel = self._resolve_output_channel()
        get_kwargs = {} if key is None else {"key": key}
        items: list[Union[Trajectory, RolloutResult]] = []
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        while len(items) < max_items:
            try:
                items.append(channel.get_nowait(**get_kwargs))
                continue
            except asyncio.QueueEmpty:
                pass
            if items:
                break  # do not wait for a full batch once something arrived
            if deadline is not None and time.monotonic() >= deadline:
                break
            time.sleep(self.poll_interval)
        return items

    def num_pending_outputs(self, *, key: Optional[Any] = None) -> int:
        """Return how many payloads are queued in the output channel."""
        channel = self._resolve_output_channel()
        return int(channel.qsize() if key is None else channel.qsize(key))

    # ------------------------------------------------------------------- stop

    def stop_service(self) -> None:
        """Ask the service to drain and shut down."""
        self._call(self._resolve_control_plane().request_stop())

    def describe(self) -> dict[str, Any]:
        """Return a compact, log-friendly view of the service state."""
        status = self.status()
        return {
            "service_name": status.service_name,
            "kind": status.kind,
            "mode": status.mode,
            "state": status.state.value,
            "served_version": status.served_version,
            "num_tasks_pending": status.num_tasks_pending,
            "num_tasks_completed": status.num_tasks_completed,
            "num_outputs_published": status.num_outputs_published,
            "output_channel": status.endpoints.output_channel,
        }


async def _await(awaitable: Any) -> None:
    """Await ``awaitable`` (helper so ``asyncio.run`` gets a coroutine)."""
    await awaitable


def make_prompt_tasks(
    batches: Iterable[Sequence[Sequence[int]]],
    *,
    n: int = 1,
    sampling: Optional[SamplingParams] = None,
    mode: Union[str, RolloutMode] = RolloutMode.TRAIN,
) -> list[RolloutTask]:
    """Build LLM generation tasks from batches of tokenized prompts.

    Args:
        batches: One entry per task; each is a list of tokenized prompts.
        n: Samples per prompt.
        sampling: Full sampling knobs; built from ``n`` when omitted.
        mode: ``train`` or ``eval``.

    Returns:
        The tasks, ready for :meth:`RolloutClient.submit_tasks`.
    """
    return [
        RolloutTask(
            kind=TaskKind.LLM_GENERATION,
            mode=RolloutMode(mode),
            sampling=sampling or SamplingParams(n=n),
            prompts=PromptSpec(input_ids=[list(ids) for ids in batch]),
        )
        for batch in batches
    ]
