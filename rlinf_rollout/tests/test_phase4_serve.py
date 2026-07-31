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

"""Guard Phase 4: the ``rollout-serve`` daemon and the client SDK.

The service is a Ray process tree, so the tests attack it from three angles that
all run on a laptop without Ray, a GPU, an engine or a simulator:

* the control-plane protocol and state machine are plain Python;
* the controllers only touch worker groups and channels through duck typing, so
  stubs drive the full control flow (start -> weight push -> task -> outputs ->
  stop);
* the service/CLI layer is exercised through ``--dry-run``, which validates the
  bundled example configs and prints the launch plan without starting Ray.

What genuinely needs hardware — bringing up SGLang/vLLM or a simulator and moving
real weights — is out of reach here and is called out in the plan document.
"""

import ast
import asyncio
import dataclasses
import inspect
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from rlinf_rollout.api.v1 import (
    EpisodeSpec,
    PromptSpec,
    RolloutMode,
    RolloutTask,
    SamplingParams,
    SchemaError,
    SourceTopology,
    TaskKind,
    WeightTransport,
    WeightUpdateAck,
    WeightUpdateRequest,
    WeightUpdateStatus,
)
from rlinf_rollout.serve.control import MAX_RETAINED_WEIGHT_RESULTS, ControlPlaneState
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

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PACKAGE_ROOT / "serve" / "configs"
EMBODIED_CONFIG = CONFIG_DIR / "embodied_eval_maniskill_mlp.yaml"
LLM_CONFIG = CONFIG_DIR / "llm_generate_sglang.yaml"
PROMPT_FILE = CONFIG_DIR / "example_prompts.jsonl"


def run_async(coro):
    """Run ``coro`` to completion on a private event loop."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Control-plane protocol
# ---------------------------------------------------------------------------

EXPECTED_FIELDS = {
    ServiceEndpoints: (
        "schema_version",
        "control_group",
        "output_channel",
        "num_output_partitions",
        "rollout_group",
        "env_group",
        "http_endpoints",
    ),
    ServiceStatus: (
        "schema_version",
        "service_name",
        "kind",
        "mode",
        "state",
        "endpoints",
        "served_version",
        "num_tasks_pending",
        "num_tasks_completed",
        "num_outputs_published",
        "num_weight_updates",
        "uptime_seconds",
        "metrics",
        "error",
        "metadata",
    ),
    TaskAck: (
        "schema_version",
        "num_accepted",
        "task_ids",
        "rejected",
        "num_pending",
        "metadata",
    ),
    WeightPushResult: (
        "schema_version",
        "version",
        "status",
        "served_version",
        "acks",
        "elapsed_seconds",
        "error",
        "metadata",
    ),
}


@pytest.mark.parametrize(
    ("message_cls", "expected"),
    tuple(EXPECTED_FIELDS.items()),
    ids=[cls.__name__ for cls in EXPECTED_FIELDS],
)
def test_control_plane_message_fields_are_locked(message_cls, expected):
    assert message_cls.field_names() == expected


@pytest.mark.parametrize(
    ("enum_cls", "expected"),
    (
        (ServiceState, ("starting", "running", "draining", "stopped", "failed")),
        (WeightPushStatus, ("accepted", "applied", "failed")),
    ),
    ids=("ServiceState", "WeightPushStatus"),
)
def test_control_plane_enum_values_are_locked(enum_cls, expected):
    assert tuple(member.value for member in enum_cls) == expected


@pytest.mark.parametrize(
    "message",
    (
        ServiceEndpoints(),
        ServiceStatus(),
        TaskAck(),
        WeightPushResult(version=0),
    ),
    ids=lambda message: type(message).__name__,
)
def test_control_plane_messages_carry_schema_version(message):
    assert dataclasses.is_dataclass(message)
    assert message.schema_version == "v1"
    assert "schema_version" in message.to_dict()


def test_control_plane_messages_reject_unknown_fields():
    payload = TaskAck().to_dict()
    payload["oops"] = 1
    with pytest.raises(SchemaError, match="Unknown field"):
        TaskAck.from_dict(payload)


def test_service_status_requires_an_error_when_failed():
    with pytest.raises(SchemaError, match="failed"):
        ServiceStatus(state=ServiceState.FAILED)
    assert ServiceStatus(state=ServiceState.FAILED, error="boom").error == "boom"


def test_service_status_is_serving_only_while_running():
    assert ServiceStatus(state=ServiceState.RUNNING).is_serving
    assert not ServiceStatus(state=ServiceState.DRAINING).is_serving


def test_task_ack_counts_must_match_ids():
    with pytest.raises(SchemaError, match="num_accepted"):
        TaskAck(num_accepted=2, task_ids=("a",))


def test_endpoints_reject_a_non_positive_partition_count():
    with pytest.raises(SchemaError, match="num_output_partitions"):
        ServiceEndpoints(num_output_partitions=0)


def test_weight_push_result_requires_an_error_when_failed():
    with pytest.raises(SchemaError, match="failed"):
        WeightPushResult(version=1, status=WeightPushStatus.FAILED)
    result = WeightPushResult(
        version=3,
        status=WeightPushStatus.APPLIED,
        served_version=3,
        acks=(WeightUpdateAck(version=3, served_version=3),),
        elapsed_seconds=1.25,
    )
    assert result.describe() == {
        "version": 3,
        "status": "applied",
        "served_version": 3,
        "num_acks": 1,
        "elapsed_seconds": 1.25,
        "error": None,
    }


def test_control_group_name_is_derived_from_the_service_name():
    assert control_group_name() == "rollout_control"
    assert control_group_name("trainer_side") == "trainer_side_control"
    assert DEFAULT_SERVICE_NAME == "rollout"


# ---------------------------------------------------------------------------
# Control-plane state machine
# ---------------------------------------------------------------------------


def _llm_task(priority: int = 0, task_id: str = "") -> RolloutTask:
    return RolloutTask(
        kind=TaskKind.LLM_GENERATION,
        task_id=task_id,
        priority=priority,
        prompts=PromptSpec(input_ids=[[1, 2, 3]]),
        sampling=SamplingParams(n=1),
    )


def _eval_task(task_id: str = "") -> RolloutTask:
    return RolloutTask(
        kind=TaskKind.EMBODIED_EVAL,
        task_id=task_id,
        mode=RolloutMode.EVAL,
        episode=EpisodeSpec(env_type="maniskill", num_envs=2),
    )


def test_control_plane_queues_and_dequeues_tasks():
    state = ControlPlaneState("svc")
    ack = state.submit_tasks([_llm_task(task_id="a"), _llm_task(task_id="b")])
    assert (ack.num_accepted, ack.task_ids, ack.num_pending) == (2, ("a", "b"), 2)
    assert state.num_tasks_submitted == 2

    first = state.next_tasks(1)
    assert [task.task_id for task in first] == ["a"]
    assert state.num_pending_tasks == 1
    assert [task.task_id for task in state.next_tasks(5)] == ["b"]
    assert state.next_tasks(5) == []


def test_control_plane_dequeues_by_priority_then_submission_order():
    state = ControlPlaneState("svc")
    state.submit_tasks(
        [
            _llm_task(priority=0, task_id="low"),
            _llm_task(priority=9, task_id="high"),
            _llm_task(priority=9, task_id="high2"),
        ]
    )
    assert [task.task_id for task in state.next_tasks(3)] == ["high", "high2", "low"]


def test_control_plane_rejects_task_kinds_it_does_not_serve():
    state = ControlPlaneState("svc", accepted_task_kinds=[TaskKind.EMBODIED_EVAL])
    ack = state.submit_tasks([_llm_task(task_id="llm"), _eval_task(task_id="eval")])
    assert ack.num_accepted == 1 and ack.task_ids == ("eval",)
    assert "llm_generation" in ack.rejected["llm"]
    assert state.accepted_task_kinds == {"embodied_eval"}


def test_control_plane_applies_back_pressure():
    state = ControlPlaneState("svc", max_pending_tasks=1)
    ack = state.submit_tasks([_llm_task(task_id="a"), _llm_task(task_id="b")])
    assert ack.num_accepted == 1
    assert "queue is full" in ack.rejected["b"]


def test_control_plane_rejects_tasks_after_a_stop_request():
    state = ControlPlaneState("svc")
    state.request_stop()
    ack = state.submit_tasks([_llm_task(task_id="a")])
    assert ack.num_accepted == 0
    assert ack.rejected["a"] == "service is shutting down"
    assert state.stop_requested


def test_control_plane_tracks_task_completion():
    state = ControlPlaneState("svc")
    state.report_task_done("a")
    state.report_task_done("b", "boom")
    assert state.num_tasks_completed == 2
    assert state.task_errors == {"b": "boom"}


def test_control_plane_queues_weight_requests_in_order():
    state = ControlPlaneState("svc")
    assert state.next_weight_request() is None
    for version in (1, 2):
        state.request_weight_update(WeightUpdateRequest(version=version))
    assert state.num_pending_weight_requests == 2
    assert state.next_weight_request().version == 1
    assert state.next_weight_request().version == 2
    assert state.next_weight_request() is None


def test_control_plane_retains_a_bounded_number_of_weight_results():
    state = ControlPlaneState("svc")
    total = MAX_RETAINED_WEIGHT_RESULTS + 5
    for version in range(total):
        state.publish_weight_result(WeightPushResult(version=version))
    assert state.get_weight_result(total - 1).version == total - 1
    assert state.get_weight_result(0) is None
    assert state.get_weight_result(total - MAX_RETAINED_WEIGHT_RESULTS) is not None


def test_control_plane_publishes_status_and_describes_itself():
    state = ControlPlaneState("svc")
    assert state.get_status().state is ServiceState.STARTING
    state.publish_status(ServiceStatus(service_name="svc", state=ServiceState.RUNNING))
    assert state.get_status().state is ServiceState.RUNNING

    state.submit_tasks([_llm_task()])
    state.request_weight_update(WeightUpdateRequest(version=7))
    described = state.describe()
    assert described["service_name"] == "svc"
    assert described["num_pending_tasks"] == 1
    assert described["num_pending_weight_requests"] == 1
    assert described["state"] == "running"
    assert described["stop_requested"] is False


# ---------------------------------------------------------------------------
# Task sources
# ---------------------------------------------------------------------------


def test_static_task_source_serves_once_then_reports_exhaustion():
    from rlinf_rollout.serve.task_source import StaticTaskSource

    source = StaticTaskSource([_eval_task("a"), _eval_task("b")])
    assert [task.task_id for task in run_async(source.next_batch(1))] == ["a"]
    assert not run_async(source.exhausted())
    assert [task.task_id for task in run_async(source.next_batch(5))] == ["b"]
    assert run_async(source.exhausted())
    assert run_async(source.next_batch(1)) == []
    assert source.num_served == 2


def test_static_task_source_can_repeat_forever():
    from rlinf_rollout.serve.task_source import StaticTaskSource

    source = StaticTaskSource([_eval_task("a")], repeat=True)
    assert len(run_async(source.next_batch(3))) == 3
    assert not run_async(source.exhausted())


def test_episode_task_maps_mode_to_kind():
    from rlinf_rollout.serve.task_source import episode_task

    train = episode_task(env_type="maniskill", num_envs=4, group_size=2)
    assert train.kind is TaskKind.EMBODIED_EPISODE
    assert train.episode.num_envs == 4 and train.episode.group_size == 2

    evaluation = episode_task(env_type="libero", num_envs=2, mode="eval")
    assert evaluation.kind is TaskKind.EMBODIED_EVAL
    assert evaluation.mode is RolloutMode.EVAL


class _StubHandle:
    """Mimics ``WorkerGroupFuncResult``: ``wait()`` returns one value per rank."""

    def __init__(self, values):
        self._values = values
        self.waited = False

    def wait(self):
        self.waited = True
        return self._values


class _StubControlPlane:
    """In-process stand-in for the control-plane worker group."""

    def __init__(self, state=None, *, wrap=True):
        self.state = state or ControlPlaneState("svc")
        self.wrap = wrap
        self.calls: list[str] = []

    def _return(self, value):
        self.calls.append(inspect.stack()[1].function)
        return _StubHandle([value]) if self.wrap else value

    def submit_tasks(self, tasks):
        return self._return(self.state.submit_tasks(tasks))

    def next_tasks(self, max_num_tasks):
        return self._return(self.state.next_tasks(max_num_tasks))

    def num_pending_tasks(self):
        return self._return(self.state.num_pending_tasks)

    def report_task_done(self, task_id, error=None):
        return self._return(self.state.report_task_done(task_id, error))

    def request_weight_update(self, request):
        return self._return(self.state.request_weight_update(request))

    def next_weight_request(self):
        return self._return(self.state.next_weight_request())

    def publish_weight_result(self, result):
        return self._return(self.state.publish_weight_result(result))

    def get_weight_result(self, version):
        return self._return(self.state.get_weight_result(version))

    def publish_status(self, status):
        return self._return(self.state.publish_status(status))

    def get_status(self):
        return self._return(self.state.get_status())

    def request_stop(self):
        return self._return(self.state.request_stop())

    def stop_requested(self):
        return self._return(self.state.stop_requested)


@pytest.mark.parametrize("wrap", (True, False), ids=("worker_group", "plain_value"))
def test_control_plane_task_source_drains_the_control_plane(wrap):
    from rlinf_rollout.serve.task_source import ControlPlaneTaskSource

    control_plane = _StubControlPlane(wrap=wrap)
    control_plane.state.submit_tasks([_eval_task("a"), _eval_task("b")])
    source = ControlPlaneTaskSource(control_plane)

    assert [task.task_id for task in run_async(source.next_batch(1))] == ["a"]
    assert run_async(source.next_batch(0)) == []
    run_async(source.report_done("a"))
    assert control_plane.state.num_tasks_completed == 1
    # A control-plane source is never exhausted: a client may submit any time.
    assert not run_async(source.exhausted())


def test_jsonl_prompt_source_uses_pretokenized_ids_and_batches(tmp_path):
    from rlinf_rollout.serve.task_source import JsonlPromptTaskSource

    path = tmp_path / "prompts.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"input_ids": [index, index + 1], "answer": str(index)})
            for index in range(5)
        ),
        encoding="utf-8",
    )
    source = JsonlPromptTaskSource(path, batch_size=2, group_size=3)
    assert source.num_prompts == 5

    tasks = run_async(source.next_batch(2))
    assert len(tasks) == 2
    assert tasks[0].kind is TaskKind.LLM_GENERATION
    assert tasks[0].prompts.input_ids == [[0, 1], [1, 2]]
    assert tasks[0].prompts.answers == ["0", "1"]
    assert tasks[0].sampling.n == 3
    assert not run_async(source.exhausted())
    remaining = run_async(source.next_batch(5))
    assert [len(task.prompts.input_ids) for task in remaining] == [1]
    assert run_async(source.exhausted())


def test_jsonl_prompt_source_tokenizes_text_prompts(tmp_path):
    from rlinf_rollout.serve.task_source import JsonlPromptTaskSource

    path = tmp_path / "prompts.jsonl"
    path.write_text('{"prompt": "abc"}\n{"text": "de"}\n', encoding="utf-8")
    source = JsonlPromptTaskSource(
        path, batch_size=2, encode=lambda text: list(map(ord, text))
    )
    task = run_async(source.next_batch(1))[0]
    assert task.prompts.input_ids == [[97, 98, 99], [100, 101]]
    assert task.prompts.prompt_texts == ["abc", "de"]


def test_jsonl_prompt_source_needs_an_encoder_for_text(tmp_path):
    from rlinf_rollout.serve.task_source import JsonlPromptTaskSource

    path = tmp_path / "prompts.jsonl"
    path.write_text('{"prompt": "abc"}\n', encoding="utf-8")
    source = JsonlPromptTaskSource(path)
    with pytest.raises(ValueError, match="encode"):
        run_async(source.next_batch(1))


@pytest.mark.parametrize(
    ("content", "match"),
    (
        ("", "no prompts"),
        ("not json\n", "invalid JSON"),
        ("[1, 2]\n", "JSON object"),
        ('{"nope": 1}\n', "input_ids"),
    ),
)
def test_jsonl_prompt_source_rejects_bad_files(tmp_path, content, match):
    from rlinf_rollout.serve.task_source import JsonlPromptTaskSource

    path = tmp_path / "prompts.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        source = JsonlPromptTaskSource(path)
        run_async(source.next_batch(1))


def test_jsonl_prompt_source_honors_max_prompts_and_repeat(tmp_path):
    from rlinf_rollout.serve.task_source import JsonlPromptTaskSource

    path = tmp_path / "prompts.jsonl"
    path.write_text(
        "\n".join(json.dumps({"input_ids": [index]}) for index in range(4)),
        encoding="utf-8",
    )
    limited = JsonlPromptTaskSource(path, max_prompts=2)
    assert limited.num_prompts == 2

    repeating = JsonlPromptTaskSource(path, batch_size=4, repeat=True)
    assert len(run_async(repeating.next_batch(3))) == 3
    assert not run_async(repeating.exhausted())


def test_jsonl_prompt_source_requires_an_existing_file(tmp_path):
    from rlinf_rollout.serve.task_source import JsonlPromptTaskSource

    with pytest.raises(FileNotFoundError):
        JsonlPromptTaskSource(tmp_path / "missing.jsonl")


def test_bundled_prompt_file_is_readable():
    from rlinf_rollout.serve.task_source import JsonlPromptTaskSource

    source = JsonlPromptTaskSource(
        PROMPT_FILE, batch_size=2, encode=lambda text: [len(text)]
    )
    assert source.num_prompts == 5
    tasks = run_async(source.next_batch(3))
    assert sum(len(task.prompts.input_ids) for task in tasks) == 5


# ---------------------------------------------------------------------------
# Controllers (driven by stub worker groups and channels)
# ---------------------------------------------------------------------------


class _FakeChannel:
    """Minimal stand-in for a vendored-scheduler ``Channel``."""

    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs
        self.items: list = []
        self.put_kwargs: list[dict] = []

    def put(self, item, **kwargs):
        self.items.append(item)
        self.put_kwargs.append(kwargs)
        return None

    def get_nowait(self, key=None):
        del key
        if not self.items:
            raise asyncio.QueueEmpty
        return self.items.pop(0)

    def qsize(self, key=None):
        del key
        return len(self.items)


class _FakeGroup:
    """Records worker-group calls and replays scripted per-rank results."""

    def __init__(self, name="group", world_size=1, **returns):
        self.name = name
        self.world_size = world_size
        self._returns = returns
        self.calls: list[tuple[str, tuple, dict]] = []

    def _record(self, method, args, kwargs):
        self.calls.append((method, args, kwargs))
        value = self._returns.get(method, None)
        if callable(value):
            value = value(*args, **kwargs)
        if not isinstance(value, list):
            value = [value] * self.world_size
        return _StubHandle(value)

    def called(self, method):
        """Return the kwargs of every recorded call to ``method``."""
        return [kwargs for name, _, kwargs in self.calls if name == method]

    def __getattr__(self, method):
        if method.startswith("_"):
            raise AttributeError(method)

        def call(*args, **kwargs):
            return self._record(method, args, kwargs)

        return call


def _embodied_cfg(**overrides):
    cfg = OmegaConf.create(
        {
            "rollout": {
                "kind": "embodied",
                "group_name": "RolloutGroup",
                "mode": "collect",
                "decoupled": False,
                "weight_sync": {"no_wait": False},
                "serve": {"name": "svc"},
            },
            "env": {"group_name": "EnvGroup"},
            "sink": {"num_shards": 1},
        }
    )
    return OmegaConf.merge(cfg, OmegaConf.create(overrides))


def _endpoints(**overrides):
    base = {
        "control_group": "svc_control",
        "output_channel": "svc_output",
        "num_output_partitions": 1,
        "rollout_group": "RolloutGroup",
        "env_group": "EnvGroup",
    }
    base.update(overrides)
    return ServiceEndpoints(**base)


def _embodied_controller(cfg=None, *, control_plane=None, **kwargs):
    from rlinf_rollout.serve.controller import EmbodiedRolloutController

    rollout_group = _FakeGroup("RolloutGroup")
    env_group = _FakeGroup("EnvGroup")
    controller = EmbodiedRolloutController(
        cfg if cfg is not None else _embodied_cfg(),
        rollout_group=rollout_group,
        env_group=env_group,
        endpoints=_endpoints(),
        control_plane=control_plane,
        service_name="svc",
        channel_factory=_FakeChannel,
        poll_interval=0.0,
        status_interval=0.0,
        **kwargs,
    )
    return controller, rollout_group, env_group


def test_embodied_controller_starts_the_resident_loops():
    control_plane = _StubControlPlane()
    controller, rollout_group, env_group = _embodied_controller(
        control_plane=control_plane
    )
    run_async(controller.start())

    assert rollout_group.called("init_worker") == [{}]
    assert env_group.called("init_worker") == [{}]
    interact = env_group.called("interact")[0]
    assert interact["input_channel"].name == "svc_env"
    assert interact["rollout_channel"].name == "svc_rollout"
    assert interact["trajectory_channel"].name == "svc_output"
    assert interact["reward_channel"] is None
    generate = rollout_group.called("generate")[0]
    assert generate["input_channel"].name == "svc_rollout"
    assert generate["output_channel"].name == "svc_env"

    status = control_plane.state.get_status()
    assert status.state is ServiceState.RUNNING
    assert status.kind == "embodied" and status.mode == "collect"
    assert status.endpoints.output_channel == "svc_output"


def test_embodied_controller_does_not_collect_in_eval_mode():
    controller, rollout_group, env_group = _embodied_controller(
        _embodied_cfg(rollout={"mode": "eval"})
    )
    run_async(controller.start())
    assert env_group.called("interact") == []
    assert rollout_group.called("generate") == []
    assert env_group.called("init_worker") == [{}]


def test_embodied_controller_applies_a_checkpoint_weight_push():
    control_plane = _StubControlPlane()
    ack = WeightUpdateAck(version=5, served_version=5, receiver_id="RolloutGroup:0")
    controller, rollout_group, _ = _embodied_controller(control_plane=control_plane)
    rollout_group._returns["receive_weight_update"] = ack
    run_async(controller.start())

    request = WeightUpdateRequest(
        version=5, transport=WeightTransport.CHECKPOINT, checkpoint_path="/tmp/w.pt"
    )
    control_plane.state.request_weight_update(request)
    run_async(controller._tick())

    result = control_plane.state.get_weight_result(5)
    assert result.status is WeightPushStatus.APPLIED
    assert result.served_version == 5
    assert result.acks[0].receiver_id == "RolloutGroup:0"
    assert controller.status().num_weight_updates == 1
    assert controller.status().served_version == 5


def test_embodied_controller_reports_a_failed_weight_push():
    control_plane = _StubControlPlane()
    controller, rollout_group, _ = _embodied_controller(control_plane=control_plane)
    rollout_group._returns["receive_weight_update"] = WeightUpdateAck(
        version=2,
        status=WeightUpdateStatus.FAILED,
        served_version=1,
        receiver_id="RolloutGroup:0",
        error="checkpoint missing",
    )
    run_async(controller.start())
    control_plane.state.request_weight_update(
        WeightUpdateRequest(
            version=2, transport=WeightTransport.CHECKPOINT, checkpoint_path="/nope"
        )
    )
    run_async(controller._tick())

    result = control_plane.state.get_weight_result(2)
    assert result.status is WeightPushStatus.FAILED
    assert "checkpoint missing" in result.error
    assert controller.status().num_weight_updates == 0


def test_embodied_controller_queues_a_background_collective_push():
    control_plane = _StubControlPlane()
    controller, rollout_group, _ = _embodied_controller(
        _embodied_cfg(rollout={"weight_sync": {"no_wait": True}}),
        control_plane=control_plane,
    )
    run_async(controller.start())
    control_plane.state.request_weight_update(
        WeightUpdateRequest(version=3, transport=WeightTransport.COLLECTIVE)
    )
    run_async(controller._tick())

    assert rollout_group.called("request_weight_sync") == [{}]
    assert rollout_group.called("receive_weight_update") == []
    result = control_plane.state.get_weight_result(3)
    assert result.status is WeightPushStatus.ACCEPTED


def test_embodied_controller_refuses_weight_pushes_in_eval_mode():
    control_plane = _StubControlPlane()
    controller, _, _ = _embodied_controller(
        _embodied_cfg(rollout={"mode": "eval"}), control_plane=control_plane
    )
    run_async(controller.start())
    control_plane.state.request_weight_update(WeightUpdateRequest(version=1))
    run_async(controller._tick())

    result = control_plane.state.get_weight_result(1)
    assert result.status is WeightPushStatus.FAILED
    assert "eval mode" in result.error


def test_embodied_controller_runs_a_requested_eval_round():
    import torch

    control_plane = _StubControlPlane(
        ControlPlaneState("svc", accepted_task_kinds=[TaskKind.EMBODIED_EVAL])
    )
    controller, rollout_group, env_group = _embodied_controller(
        _embodied_cfg(rollout={"mode": "eval"}), control_plane=control_plane
    )
    env_group._returns["evaluate"] = [{"success": torch.tensor([1.0, 0.0])}]
    run_async(controller.start())
    control_plane.state.submit_tasks([_eval_task("eval-1")])

    run_async(controller._tick())

    assert env_group.called("evaluate")[0]["input_channel"].name == "svc_env"
    assert rollout_group.called("evaluate")[0]["output_channel"].name == "svc_env"
    metrics = controller.status().metrics
    assert metrics["eval/num_rounds"] == 1.0
    assert metrics["eval/success"] == pytest.approx(0.5)
    assert control_plane.state.num_tasks_completed == 1
    assert control_plane.state.task_errors == {}


def test_embodied_controller_reports_an_eval_round_failure():
    control_plane = _StubControlPlane(
        ControlPlaneState("svc", accepted_task_kinds=[TaskKind.EMBODIED_EVAL])
    )
    controller, _, env_group = _embodied_controller(
        _embodied_cfg(rollout={"mode": "eval"}), control_plane=control_plane
    )

    def boom(**kwargs):
        raise RuntimeError("simulator died")

    env_group._returns["evaluate"] = boom
    run_async(controller.start())
    control_plane.state.submit_tasks([_eval_task("eval-1")])
    run_async(controller._tick())

    assert control_plane.state.task_errors == {"eval-1": "RuntimeError: simulator died"}


def test_embodied_controller_harvests_worker_metrics():
    import torch

    control_plane = _StubControlPlane()
    controller, _, _ = _embodied_controller(control_plane=control_plane)
    run_async(controller.start())
    controller.channel("svc_env_metric").items.append(
        {
            "rank": 0,
            "env": {"env/success": torch.tensor([1.0, 0.0])},
            "time": {"time/env/step": 2.0},
        }
    )
    controller.channel(controller.endpoints.output_channel).items.append("payload")

    run_async(controller._tick())
    metrics = controller.status().metrics
    assert metrics["env/success/rank0"] == pytest.approx(0.5)
    assert metrics["time/env/step/rank0"] == 2.0
    assert metrics["output_channel_qsize"] == 1.0


def test_embodied_controller_stops_both_groups_and_waits_for_the_loops():
    control_plane = _StubControlPlane()
    controller, rollout_group, env_group = _embodied_controller(
        control_plane=control_plane
    )
    run_async(controller.start())
    env_handle = env_group.called("interact") and controller._env_handle
    run_async(controller.shutdown())

    assert env_group.called("stop") == [{}]
    assert rollout_group.called("stop") == [{}]
    assert env_handle.waited
    assert controller.state is ServiceState.STOPPED
    assert control_plane.state.get_status().state is ServiceState.STOPPED


def test_controller_serve_loop_returns_when_the_client_stops_it():
    control_plane = _StubControlPlane()
    controller, _, _ = _embodied_controller(control_plane=control_plane)
    control_plane.state.request_stop()
    run_async(controller.serve_forever())
    assert controller.state is ServiceState.DRAINING


def test_controller_serve_loop_returns_when_the_task_source_drains():
    from rlinf_rollout.serve.task_source import StaticTaskSource

    control_plane = _StubControlPlane(
        ControlPlaneState("svc", accepted_task_kinds=[TaskKind.EMBODIED_EVAL])
    )
    source = StaticTaskSource([])
    controller, _, _ = _embodied_controller(
        _embodied_cfg(rollout={"mode": "eval"}),
        control_plane=control_plane,
        task_source=source,
        stop_when_task_source_exhausted=True,
    )
    run_async(controller.serve_forever())
    assert controller.state is ServiceState.DRAINING


def test_controller_serve_loop_marks_a_failure_in_the_status():
    control_plane = _StubControlPlane()
    controller, rollout_group, _ = _embodied_controller(control_plane=control_plane)

    def boom(**kwargs):
        raise RuntimeError("worker died")

    rollout_group._returns["init_worker"] = boom
    with pytest.raises(RuntimeError, match="worker died"):
        run_async(controller.serve_forever())
    status = control_plane.state.get_status()
    assert status.state is ServiceState.FAILED
    assert "worker died" in status.error


# ---------------------------------------------------------------------------
# LLM controller
# ---------------------------------------------------------------------------


def _llm_cfg(**overrides):
    cfg = OmegaConf.create(
        {
            "rollout": {
                "kind": "llm",
                "group_name": "RolloutGroup",
                "mode": "collect",
                "group_size": 2,
                "batch_size": 4,
                "serve": {"name": "svc", "generation_timeout_seconds": 5},
            },
            "sink": {"num_shards": 1},
        }
    )
    return OmegaConf.merge(cfg, OmegaConf.create(overrides))


class _RecordingSink:
    """Captures the payloads a controller publishes."""

    def __init__(self):
        from rlinf_rollout.api.v1 import ConsumerSpec

        self._spec = ConsumerSpec()
        self.items: list = []
        self.flushed = 0
        self.closed = False

    @property
    def consumer_spec(self):
        return self._spec

    async def put(self, item, partition=None):
        self.items.append((item, partition))

    async def flush(self):
        self.flushed += 1

    async def close(self):
        self.closed = True


#: Fields :func:`rlinf_rollout.data.convert.rollout_result_to_api` reads off the
#: engines' internal result struct. Locked against the real struct by
#: ``test_internal_llm_structs_match_what_the_controller_uses``.
INTERNAL_LLM_RESULT_FIELDS = (
    "num_sequence",
    "group_size",
    "prompt_lengths",
    "prompt_ids",
    "response_lengths",
    "response_ids",
    "is_end",
    "response_mask",
    "rollout_logprobs",
    "prompt_texts",
    "response_texts",
    "answers",
    "image_data",
    "multi_modal_inputs",
    "rewards",
)

#: Constructor keywords the controller passes to the engines' request struct.
INTERNAL_LLM_REQUEST_FIELDS = (
    "n",
    "input_ids",
    "image_data",
    "answers",
    "multi_modal_inputs",
)


@dataclasses.dataclass
class _StubInternalRolloutResult:
    """Ray-free stand-in for ``data.io_struct.RolloutResult``."""

    num_sequence: int
    group_size: int
    prompt_lengths: list
    prompt_ids: list
    response_lengths: list
    response_ids: list
    is_end: list
    response_mask: object = None
    rollout_logprobs: object = None
    prompt_texts: object = None
    response_texts: object = None
    answers: object = None
    image_data: object = None
    multi_modal_inputs: object = None
    rewards: object = None


@dataclasses.dataclass
class _StubRolloutRequest:
    """Ray-free stand-in for ``data.io_struct.RolloutRequest``."""

    n: int
    input_ids: list
    image_data: list
    answers: list
    multi_modal_inputs: list


@pytest.fixture
def io_struct_module(monkeypatch):
    """Yield ``data.io_struct``, standing in for it when Ray is too old.

    The controller imports ``RolloutRequest`` from there, and that module pulls in
    the Ray-backed scheduler. Where a matching Ray is installed the real module is
    used; otherwise a stub keeps the dispatch logic testable, and
    ``test_internal_llm_structs_match_what_the_controller_uses`` guarantees the
    stub cannot drift from the real struct unnoticed.
    """
    import types

    try:
        import rlinf_rollout.data.io_struct as module

        yield module
        return
    except Exception:
        pass

    stub = types.ModuleType("rlinf_rollout.data.io_struct")
    stub.RolloutRequest = _StubRolloutRequest
    stub.RolloutResult = _StubInternalRolloutResult
    monkeypatch.setitem(sys.modules, "rlinf_rollout.data.io_struct", stub)
    yield stub


def test_internal_llm_structs_match_what_the_controller_uses():
    try:
        from rlinf_rollout.data.io_struct import RolloutRequest, RolloutResult
    except Exception as exc:  # pragma: no cover - depends on the local runtime
        pytest.skip(f"data.io_struct needs the full runtime: {exc}")

    request_fields = {field.name for field in dataclasses.fields(RolloutRequest)}
    assert set(INTERNAL_LLM_REQUEST_FIELDS) <= request_fields
    assert set(INTERNAL_LLM_REQUEST_FIELDS) == {
        field.name for field in dataclasses.fields(_StubRolloutRequest)
    }

    result_fields = {field.name for field in dataclasses.fields(RolloutResult)}
    assert set(INTERNAL_LLM_RESULT_FIELDS) <= result_fields
    assert set(INTERNAL_LLM_RESULT_FIELDS) == {
        field.name for field in dataclasses.fields(_StubInternalRolloutResult)
    }


def _internal_llm_result(num_sequences=2, group_size=2):
    return _StubInternalRolloutResult(
        num_sequence=num_sequences,
        group_size=group_size,
        prompt_lengths=[3] * num_sequences,
        prompt_ids=[[1, 2, 3]] * num_sequences,
        response_lengths=[2] * num_sequences,
        response_ids=[[4, 5]] * num_sequences,
        is_end=([True, False] * num_sequences)[:num_sequences],
        answers=["42"] * num_sequences,
    )


def _llm_controller(cfg=None, *, control_plane=None, num_engines=2, **kwargs):
    from rlinf_rollout.serve.controller import LLMRolloutController

    rollout_group = _FakeGroup("RolloutGroup", world_size=num_engines)
    sink = _RecordingSink()
    controller = LLMRolloutController(
        cfg if cfg is not None else _llm_cfg(),
        rollout_group=rollout_group,
        num_engines=num_engines,
        sink=sink,
        endpoints=_endpoints(env_group=None),
        control_plane=control_plane,
        service_name="svc",
        channel_factory=_FakeChannel,
        poll_interval=0.0,
        status_interval=0.0,
        **kwargs,
    )
    return controller, rollout_group, sink


def test_llm_controller_requires_a_positive_engine_count():
    with pytest.raises(ValueError, match="num_engines"):
        _llm_controller(num_engines=0)


def test_llm_controller_splits_a_task_across_the_engines(io_struct_module):
    control_plane = _StubControlPlane(
        ControlPlaneState("svc", accepted_task_kinds=[TaskKind.LLM_GENERATION])
    )
    controller, rollout_group, sink = _llm_controller(control_plane=control_plane)
    run_async(controller.start())
    assert rollout_group.called("init_worker") == [{}]

    task = RolloutTask(
        kind=TaskKind.LLM_GENERATION,
        task_id="gen-1",
        sampling=SamplingParams(n=2),
        prompts=PromptSpec(
            input_ids=[[1, 2], [3, 4], [5, 6], [7, 8]], answers=["a", "b", "c", "d"]
        ),
    )
    control_plane.state.submit_tasks([task])
    # Four prompts -> four sequence groups; the engines answer with one result each.
    engine_output = controller.channel("svc_engine_out")
    engine_output.items.extend(_internal_llm_result() for _ in range(4))

    run_async(controller._tick())

    requests = controller.channel("svc_engine_in").items
    assert len(requests) == 2  # one per engine
    assert [len(request.input_ids) for request in requests] == [2, 2]
    assert requests[0].n == 2
    assert requests[0].answers == ["a", "b"]
    rollout_call = rollout_group.called("rollout")[0]
    assert rollout_call["input_channel"].name == "svc_engine_in"
    assert rollout_call["output_channel"].name == "svc_engine_out"

    assert len(sink.items) == 4
    payload, partition = sink.items[0]
    assert payload.schema_version == "v1"
    assert payload.num_sequences == 2 and payload.group_size == 2
    assert payload.request_ids == ("gen-1", "gen-1")
    assert payload.metadata.extra["task_id"] == "gen-1"
    assert partition is None
    assert sink.flushed == 1
    assert controller.status().num_outputs_published == 4
    assert control_plane.state.num_tasks_completed == 1
    assert control_plane.state.task_errors == {}


def test_llm_controller_rejects_untokenized_prompts(io_struct_module):
    control_plane = _StubControlPlane(
        ControlPlaneState("svc", accepted_task_kinds=[TaskKind.LLM_GENERATION])
    )
    controller, _, sink = _llm_controller(control_plane=control_plane)
    run_async(controller.start())
    control_plane.state.submit_tasks(
        [
            RolloutTask(
                kind=TaskKind.LLM_GENERATION,
                task_id="gen-text",
                prompts=PromptSpec(prompt_texts=["hello"]),
            )
        ]
    )
    run_async(controller._tick())

    assert sink.items == []
    error = control_plane.state.task_errors["gen-text"]
    assert "input_ids" in error and "tokenizer" in error


def test_llm_controller_times_out_a_stuck_generation(io_struct_module):
    control_plane = _StubControlPlane(
        ControlPlaneState("svc", accepted_task_kinds=[TaskKind.LLM_GENERATION])
    )
    controller, _, _ = _llm_controller(
        _llm_cfg(rollout={"serve": {"generation_timeout_seconds": 0.01}}),
        control_plane=control_plane,
        num_engines=1,
    )
    run_async(controller.start())
    control_plane.state.submit_tasks(
        [
            RolloutTask(
                kind=TaskKind.LLM_GENERATION,
                task_id="gen-stuck",
                prompts=PromptSpec(input_ids=[[1, 2]]),
            )
        ]
    )
    run_async(controller._tick())

    error = control_plane.state.task_errors["gen-stuck"]
    assert "0/1" in error or "sequence groups" in error


def test_llm_controller_applies_a_collective_weight_push():
    control_plane = _StubControlPlane()
    controller, rollout_group, _ = _llm_controller(control_plane=control_plane)
    run_async(controller.start())
    control_plane.state.request_weight_update(
        WeightUpdateRequest(
            version=7,
            transport=WeightTransport.COLLECTIVE,
            source=SourceTopology(group_name="trainer", world_size=8),
        )
    )
    run_async(controller._tick())

    assert rollout_group.called("sync_weights") == [{}]
    result = control_plane.state.get_weight_result(7)
    assert result.status is WeightPushStatus.APPLIED
    assert result.served_version == 7
    assert [ack.receiver_id for ack in result.acks] == [
        "RolloutGroup:0",
        "RolloutGroup:1",
    ]
    assert controller.status().served_version == 7


def test_llm_controller_refuses_a_checkpoint_weight_push():
    control_plane = _StubControlPlane()
    controller, rollout_group, _ = _llm_controller(control_plane=control_plane)
    run_async(controller.start())
    control_plane.state.request_weight_update(
        WeightUpdateRequest(
            version=1,
            transport=WeightTransport.CHECKPOINT,
            checkpoint_path="/tmp/w.pt",
        )
    )
    run_async(controller._tick())

    assert rollout_group.called("sync_weights") == []
    result = control_plane.state.get_weight_result(1)
    assert result.status is WeightPushStatus.FAILED
    assert "checkpoint" in result.error


def test_llm_controller_refuses_weight_pushes_in_eval_mode():
    control_plane = _StubControlPlane()
    controller, _, _ = _llm_controller(
        _llm_cfg(rollout={"mode": "eval"}), control_plane=control_plane
    )
    run_async(controller.start())
    control_plane.state.request_weight_update(
        WeightUpdateRequest(version=1, transport=WeightTransport.COLLECTIVE)
    )
    run_async(controller._tick())
    result = control_plane.state.get_weight_result(1)
    assert result.status is WeightPushStatus.FAILED
    assert "eval mode" in result.error


def test_llm_controller_builds_a_channel_sink_from_the_shard_count():
    from rlinf_rollout.api.v1 import PartitionAxis
    from rlinf_rollout.serve.controller import LLMRolloutController
    from rlinf_rollout.sinks import ChannelTrajectorySink

    controller = LLMRolloutController(
        _llm_cfg(),
        rollout_group=_FakeGroup("RolloutGroup"),
        num_engines=1,
        endpoints=_endpoints(env_group=None, num_output_partitions=3),
        service_name="svc",
        channel_factory=_FakeChannel,
    )
    run_async(controller.start())
    assert isinstance(controller.sink, ChannelTrajectorySink)
    spec = controller.sink.consumer_spec
    assert spec.num_partitions == 3
    assert spec.axis is PartitionAxis.SEQUENCE


def test_llm_controller_shutdown_closes_the_sink_and_aborts_generation():
    controller, rollout_group, sink = _llm_controller()
    run_async(controller.start())
    run_async(controller.shutdown())
    assert sink.closed
    assert rollout_group.called("abort_generation") == [{}]
    assert controller.state is ServiceState.STOPPED


# ---------------------------------------------------------------------------
# Service assembly and the rollout-serve CLI
# ---------------------------------------------------------------------------


def _service(config_path, overrides=None, **kwargs):
    from rlinf_rollout.serve.service import RolloutService, load_service_config

    return RolloutService(load_service_config(str(config_path), overrides), **kwargs)


def test_load_service_config_fills_the_serve_defaults():
    from rlinf_rollout.serve.service import load_service_config

    cfg = load_service_config(str(EMBODIED_CONFIG))
    assert cfg.rollout.mode == "eval"
    assert cfg.rollout.serve.name == "rollout"
    assert cfg.rollout.serve.status_interval_seconds == 5.0
    assert cfg.rollout.serve.task_source.type == "eval_rounds"
    assert cfg.rollout.serve.max_pending_tasks == 0


def test_load_service_config_applies_typed_overrides():
    from rlinf_rollout.serve.service import load_service_config

    cfg = load_service_config(
        str(EMBODIED_CONFIG),
        [
            "rollout.serve.max_pending_tasks=5",
            "rollout.serve.stop_when_done=false",
            "sink.num_shards=null",
        ],
    )
    assert cfg.rollout.serve.max_pending_tasks == 5
    assert cfg.rollout.serve.stop_when_done is False
    assert cfg.sink.num_shards is None


def test_load_service_config_rejects_a_malformed_override():
    from rlinf_rollout.config import RolloutConfigError
    from rlinf_rollout.serve.service import load_service_config

    with pytest.raises(RolloutConfigError, match="key=value"):
        load_service_config(str(EMBODIED_CONFIG), ["not-an-assignment"])


def test_embodied_service_plan():
    plan = _service(EMBODIED_CONFIG).plan().to_dict()
    assert plan["kind"] == "embodied" and plan["mode"] == "eval"
    assert plan["worker_groups"] == {
        "rollout_control": "rlinf_rollout.serve.control_worker:ControlPlaneWorker",
        "RolloutGroup": (
            "rlinf_rollout.workers.rollout.hf.huggingface_worker:"
            "AsyncMultiStepRolloutWorker"
        ),
        "EnvGroup": "rlinf_rollout.workers.env.env_worker:AsyncEnvWorker",
    }
    assert plan["endpoints"]["control_group"] == "rollout_control"
    assert plan["endpoints"]["output_channel"] == "rollout_output"
    assert plan["endpoints"]["env_group"] == "EnvGroup"
    assert plan["accepted_task_kinds"] == ["embodied_eval"]
    assert plan["task_source"]["type"] == "eval_rounds"
    assert plan["stop_when_done"] is True
    assert plan["num_nodes"] == 1
    assert plan["placement"] == {"env,rollout": 0}


def test_llm_service_plan():
    plan = _service(LLM_CONFIG).plan().to_dict()
    assert plan["kind"] == "llm" and plan["mode"] == "eval"
    assert plan["worker_groups"]["RolloutGroup"] == (
        "rlinf_rollout.workers.rollout.sglang.sglang_worker:SGLangWorker"
    )
    assert plan["endpoints"]["env_group"] is None
    assert plan["accepted_task_kinds"] == ["llm_generation"]
    assert plan["task_source"]["type"] == "prompt_file"
    assert plan["task_source"]["path"].endswith("example_prompts.jsonl")


def test_llm_service_plan_selects_the_http_engine_worker():
    plan = (
        _service(LLM_CONFIG, ["rollout.sglang.serving_mode=worker_http"])
        .plan()
        .to_dict()
    )
    assert plan["worker_groups"]["RolloutGroup"] == (
        "rlinf_rollout.workers.rollout.sglang.sglang_worker_server:"
        "SGLangWorkerWithHTTPServer"
    )
    assert plan["endpoints"]["http_endpoints"] == {
        "engine_openai": "http://0.0.0.0:8020"
    }


def test_llm_service_plan_selects_the_vllm_worker():
    plan = _service(LLM_CONFIG, ["rollout.rollout_backend=vllm"]).plan().to_dict()
    assert plan["worker_groups"]["RolloutGroup"] == (
        "rlinf_rollout.workers.rollout.vllm.vllm_worker:VLLMWorker"
    )


def test_service_name_override_renames_the_control_plane():
    service = _service(EMBODIED_CONFIG, service_name="trainer_a")
    plan = service.plan().to_dict()
    assert plan["service_name"] == "trainer_a"
    assert plan["endpoints"]["control_group"] == "trainer_a_control"
    assert plan["endpoints"]["output_channel"] == "trainer_a_output"
    assert "trainer_a_control" in plan["worker_groups"]


def test_service_rejects_an_external_reward_model():
    from rlinf_rollout.serve.service import RolloutService, load_service_config

    cfg = load_service_config(str(EMBODIED_CONFIG), ["reward.use_reward_model=true"])
    with pytest.raises(NotImplementedError, match="reward worker"):
        RolloutService(cfg)


def test_service_builds_the_eval_rounds_task_source():
    from rlinf_rollout.serve.task_source import StaticTaskSource

    service = _service(EMBODIED_CONFIG, ["rollout.serve.task_source.num_rounds=3"])
    source = service.build_task_source()
    assert isinstance(source, StaticTaskSource)
    tasks = run_async(source.next_batch(5))
    assert len(tasks) == 3
    assert all(task.kind is TaskKind.EMBODIED_EVAL for task in tasks)
    assert tasks[0].episode.env_type == "maniskill"
    assert tasks[0].episode.num_envs == 8
    assert run_async(source.exhausted())


def test_service_control_plane_source_is_the_default():
    service = _service(
        EMBODIED_CONFIG, ["rollout.serve.task_source.type=control_plane"]
    )
    assert service.build_task_source() is None
    assert service.stop_when_done is False
    assert service.plan().to_dict()["task_source"]["type"] == "control_plane"


def test_service_honors_an_explicit_task_source():
    from rlinf_rollout.serve.task_source import StaticTaskSource

    source = StaticTaskSource([_eval_task("a")])
    service = _service(
        EMBODIED_CONFIG,
        ["rollout.serve.task_source.type=control_plane"],
        task_source=source,
    )
    assert service.build_task_source() is source
    assert service.stop_when_done is True
    assert service.plan().to_dict()["task_source"] == {"type": "StaticTaskSource"}


@pytest.mark.parametrize(
    ("overrides", "match"),
    (
        (["rollout.serve.task_source.type=nope"], "task_source.type"),
        (["rollout.serve.poll_interval_seconds=0"], "poll_interval_seconds"),
        (["rollout.serve.status_interval_seconds=-1"], "status_interval_seconds"),
        (["rollout.serve.max_pending_tasks=-1"], "max_pending_tasks"),
        (["rollout.serve.task_source.num_rounds=0"], "num_rounds"),
        (["rollout.serve.task_source.type=prompt_file"], "only applies to"),
    ),
)
def test_invalid_serve_configs_are_rejected(overrides, match):
    from rlinf_rollout.config import RolloutConfigError
    from rlinf_rollout.serve.service import load_service_config

    with pytest.raises(RolloutConfigError, match=match):
        load_service_config(str(EMBODIED_CONFIG), overrides)


@pytest.mark.parametrize(
    ("overrides", "match"),
    (
        (["rollout.serve.task_source.type=eval_rounds"], "only applies to"),
        (["rollout.serve.task_source.path=null"], "task_source.path"),
    ),
)
def test_invalid_llm_serve_configs_are_rejected(overrides, match):
    from rlinf_rollout.config import RolloutConfigError
    from rlinf_rollout.serve.service import load_service_config

    with pytest.raises(RolloutConfigError, match=match):
        load_service_config(str(LLM_CONFIG), overrides)


def test_resolve_worker_class_path():
    from rlinf_rollout.serve.service import resolve_worker_class_path

    resolved = resolve_worker_class_path("rlinf_rollout.serve.protocol:ServiceStatus")
    assert resolved is ServiceStatus
    with pytest.raises(ValueError, match="module:Class"):
        resolve_worker_class_path("rlinf_rollout.serve.protocol")


def test_llm_worker_class_paths_agree_with_the_backend_dispatcher():
    from rlinf_rollout.serve.service import (
        LLM_WORKER_CLASS_PATHS,
        resolve_worker_class_path,
    )

    try:
        from rlinf_rollout.workers.rollout.utils import get_rollout_backend_worker
    except Exception as exc:  # pragma: no cover - depends on the local runtime
        pytest.skip(f"engine workers need the full runtime: {exc}")

    for (backend, serving_mode), class_path in LLM_WORKER_CLASS_PATHS.items():
        cfg = OmegaConf.create(
            {
                "rollout": {
                    "rollout_backend": backend,
                    "sglang": {"serving_mode": serving_mode},
                }
            }
        )
        assert get_rollout_backend_worker(cfg) is resolve_worker_class_path(class_path)


def test_cli_dry_run_prints_the_plan_for_both_bundled_configs():
    from rlinf_rollout.serve.main import run

    for config in (EMBODIED_CONFIG, LLM_CONFIG):
        assert run(["--config", str(config), "--dry-run"]) == 0


def test_cli_dry_run_emits_parseable_json():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "rlinf_rollout.serve.main",
            "--config",
            str(LLM_CONFIG),
            "--dry-run",
            "--set",
            "rollout.batch_size=2",
        ],
        cwd=PACKAGE_ROOT.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    plan = json.loads(completed.stdout)
    assert plan["kind"] == "llm"
    assert plan["service_name"] == "rollout"


def test_cli_reports_a_bad_config_without_a_traceback(capsys):
    from rlinf_rollout.serve.main import run

    assert run(["--config", str(EMBODIED_CONFIG), "--set", "oops"]) == 2
    assert "rollout-serve" in capsys.readouterr().err


def test_cli_can_print_the_resolved_config(capsys):
    from rlinf_rollout.serve.main import run

    assert run(["--config", str(EMBODIED_CONFIG), "--dry-run", "--print-config"]) == 0
    out = capsys.readouterr().out
    assert "policy:" in out and "serve:" in out


def test_cli_parser_exposes_the_documented_flags():
    from rlinf_rollout.serve.main import build_parser

    options = {
        option for action in build_parser()._actions for option in action.option_strings
    }
    assert {"--config", "--set", "--name", "--dry-run", "--print-config"} <= options


# ---------------------------------------------------------------------------
# Client SDK
# ---------------------------------------------------------------------------


def _client(control_plane=None, **kwargs):
    from rlinf_rollout.client import RolloutClient

    control_plane = control_plane or _StubControlPlane(
        ControlPlaneState("svc", accepted_task_kinds=[TaskKind.LLM_GENERATION])
    )
    return (
        RolloutClient("svc", control_plane=control_plane, poll_interval=0.0, **kwargs),
        control_plane,
    )


def test_client_reads_the_published_status():
    client, control_plane = _client()
    control_plane.state.publish_status(
        ServiceStatus(
            service_name="svc",
            kind="llm",
            mode="collect",
            state=ServiceState.RUNNING,
            endpoints=_endpoints(env_group=None),
            served_version=4,
            num_outputs_published=9,
        )
    )
    assert client.control_group_name == "svc_control"
    assert client.status().served_version == 4
    assert client.describe() == {
        "service_name": "svc",
        "kind": "llm",
        "mode": "collect",
        "state": "running",
        "served_version": 4,
        "num_tasks_pending": 0,
        "num_tasks_completed": 0,
        "num_outputs_published": 9,
        "output_channel": "svc_output",
    }


def test_client_waits_until_the_service_is_running():
    client, control_plane = _client()
    control_plane.state.publish_status(
        ServiceStatus(service_name="svc", state=ServiceState.RUNNING)
    )
    assert client.wait_until_ready(timeout=1).state is ServiceState.RUNNING


def test_client_surfaces_a_failed_service():
    from rlinf_rollout.client import RolloutClientError

    client, control_plane = _client()
    control_plane.state.publish_status(
        ServiceStatus(service_name="svc", state=ServiceState.FAILED, error="engine oom")
    )
    with pytest.raises(RolloutClientError, match="engine oom"):
        client.wait_until_ready(timeout=1)


def test_client_times_out_waiting_for_a_service_that_never_starts():
    from rlinf_rollout.client import RolloutClientError

    client, _ = _client()
    with pytest.raises(RolloutClientError, match="did not become ready"):
        client.wait_until_ready(timeout=0.05)


def test_client_submits_prompt_tasks():
    client, control_plane = _client()
    ack = client.submit_prompts([[1, 2], [3, 4]], n=3, answers=["a", "b"])
    assert ack.num_accepted == 1 and ack.num_pending == 1

    task = control_plane.state.next_tasks(1)[0]
    assert task.kind is TaskKind.LLM_GENERATION
    assert task.prompts.input_ids == [[1, 2], [3, 4]]
    assert task.prompts.answers == ["a", "b"]
    assert task.sampling.n == 3


def test_client_submit_tasks_reports_rejections_instead_of_raising():
    client, _ = _client()
    ack = client.submit_tasks([_eval_task("nope")])
    assert ack.num_accepted == 0
    assert "embodied_eval" in ack.rejected["nope"]


def test_client_requests_an_embodied_eval_round():
    control_plane = _StubControlPlane(
        ControlPlaneState("svc", accepted_task_kinds=[TaskKind.EMBODIED_EVAL])
    )
    client, _ = _client(control_plane)
    ack = client.request_eval(env_type="libero", num_envs=4)
    assert ack.num_accepted == 1
    task = control_plane.state.next_tasks(1)[0]
    assert task.kind is TaskKind.EMBODIED_EVAL and task.mode is RolloutMode.EVAL
    assert task.episode.env_type == "libero" and task.episode.num_envs == 4


def test_make_prompt_tasks_builds_one_task_per_batch():
    from rlinf_rollout.client import make_prompt_tasks

    tasks = make_prompt_tasks([[[1, 2]], [[3], [4]]], n=2)
    assert [len(task.prompts.input_ids) for task in tasks] == [1, 2]
    assert all(task.sampling.n == 2 for task in tasks)


def test_client_push_weights_queues_before_the_sender_broadcasts():
    """The ordering is the contract: arm the receivers, then broadcast."""
    client, control_plane = _client()
    observed: list[int] = []

    def sender(request):
        # The service has the request by now; emulate the controller applying it.
        observed.append(control_plane.state.num_pending_weight_requests)
        queued = control_plane.state.next_weight_request()
        control_plane.state.publish_weight_result(
            WeightPushResult(
                version=queued.version,
                status=WeightPushStatus.APPLIED,
                served_version=queued.version,
            )
        )

    request = WeightUpdateRequest(
        version=11,
        transport=WeightTransport.COLLECTIVE,
        source=SourceTopology(group_name="trainer", world_size=4),
    )
    result = client.push_weights(request, sender=sender, timeout=1)
    assert observed == [1]
    assert result.status is WeightPushStatus.APPLIED
    assert result.served_version == 11


def test_client_push_weights_awaits_an_async_sender():
    client, control_plane = _client()
    calls: list[int] = []

    async def sender(request):
        calls.append(request.version)
        control_plane.state.next_weight_request()
        control_plane.state.publish_weight_result(
            WeightPushResult(version=request.version, served_version=request.version)
        )

    result = client.push_weights(
        WeightUpdateRequest(version=2, transport=WeightTransport.COLLECTIVE),
        sender=sender,
        timeout=1,
    )
    assert calls == [2]
    assert result.status is WeightPushStatus.APPLIED


def test_client_push_weights_can_return_without_waiting():
    client, control_plane = _client()
    result = client.push_weights(WeightUpdateRequest(version=5), wait=False)
    assert result.status is WeightPushStatus.ACCEPTED
    assert control_plane.state.num_pending_weight_requests == 1


def test_client_push_checkpoint_builds_a_checkpoint_request():
    client, control_plane = _client()
    client.push_weights  # noqa: B018 - documents the delegation target
    result = client.push_checkpoint("/shared/w_v3.pt", version=3, wait=False)
    assert result.status is WeightPushStatus.ACCEPTED
    request = control_plane.state.next_weight_request()
    assert request.transport is WeightTransport.CHECKPOINT
    assert request.checkpoint_path == "/shared/w_v3.pt"
    assert request.version == 3


def test_client_wait_for_weights_times_out():
    from rlinf_rollout.client import RolloutClientError

    client, _ = _client()
    with pytest.raises(RolloutClientError, match="no weight-push result"):
        client.wait_for_weights(1, timeout=0.05)


def test_client_drains_trajectories_from_the_output_channel():
    client, control_plane = _client()
    control_plane.state.publish_status(
        ServiceStatus(
            service_name="svc",
            state=ServiceState.RUNNING,
            endpoints=_endpoints(env_group=None),
        )
    )
    channel = _FakeChannel("svc_output")
    channel.items.extend(["payload-1", "payload-2"])
    client._output_channel = channel

    assert client.num_pending_outputs() == 2
    assert client.get_trajectories(max_items=1) == ["payload-1"]
    assert client.get_trajectories(max_items=5, timeout=0) == ["payload-2"]
    assert client.get_trajectories(max_items=1, timeout=0) == []


def test_client_stop_service_and_close():
    client, control_plane = _client()
    client.stop_service()
    assert control_plane.state.stop_requested
    with client:
        pass
    assert client._control_plane is None


def test_checkpoint_weight_publisher_writes_versioned_files(tmp_path):
    from rlinf_rollout.client import CheckpointWeightPublisher

    saved: list[tuple[dict, Path]] = []

    def save_fn(payload, path):
        saved.append((payload, path))
        path.write_text("weights", encoding="utf-8")

    publisher = CheckpointWeightPublisher(
        tmp_path / "ckpt", prefix="policy", keep_last=2, save_fn=save_fn
    )
    requests = [publisher.publish({"w": version}, version) for version in range(3)]

    assert [request.version for request in requests] == [0, 1, 2]
    assert all(request.transport is WeightTransport.CHECKPOINT for request in requests)
    assert requests[-1].checkpoint_path.endswith("policy_v2.pt")
    assert requests[-1].source.group_name == "client"
    assert len(saved) == 3
    # keep_last=2 prunes the oldest file.
    assert not (tmp_path / "ckpt" / "policy_v0.pt").exists()
    assert {path.name for path in publisher.published_paths} == {
        "policy_v1.pt",
        "policy_v2.pt",
    }


def test_collective_sender_contract_is_documented():
    from rlinf_rollout.client import describe_collective_sender

    contract = describe_collective_sender()
    assert {
        "group_name",
        "src_rank",
        "world_size",
        "syncer",
        "ordering",
        "version",
    } <= set(contract)
    assert "receiver-driven" in contract["ordering"]


# ---------------------------------------------------------------------------
# Checkpoint weight transport
# ---------------------------------------------------------------------------


def _tiny_model():
    import torch

    return torch.nn.Linear(2, 1, bias=False)


def _checkpoint_request(path, version=1, transport=WeightTransport.CHECKPOINT):
    return WeightUpdateRequest(
        version=version, transport=transport, checkpoint_path=str(path)
    )


def test_checkpoint_receiver_applies_a_state_dict(tmp_path):
    import torch

    from rlinf_rollout.weight_sync.checkpoint import CheckpointWeightReceiver

    model = _tiny_model()
    new_weight = torch.tensor([[3.0, 4.0]])
    path = tmp_path / "w.pt"
    torch.save({"weight": new_weight}, path)

    receiver = CheckpointWeightReceiver(model=model, receiver_id="rollout:0")
    assert receiver.served_version == -1

    ack = run_async(receiver.recv(_checkpoint_request(path, version=7)))
    assert ack.status is WeightUpdateStatus.APPLIED
    assert ack.served_version == 7 and ack.version == 7
    assert ack.receiver_id == "rollout:0"
    assert ack.num_tensors_applied == 1
    assert ack.num_bytes_received == new_weight.numel() * new_weight.element_size()
    assert torch.equal(model.weight.detach(), new_weight)
    assert receiver.served_version == 7
    assert receiver.last_checkpoint_path == str(path)


def test_checkpoint_receiver_unwraps_a_nested_state_dict(tmp_path):
    import torch

    from rlinf_rollout.weight_sync.checkpoint import CheckpointWeightReceiver

    model = _tiny_model()
    path = tmp_path / "w.pt"
    torch.save({"state_dict": {"weight": torch.tensor([[1.0, 1.0]])}, "step": 3}, path)
    ack = run_async(
        CheckpointWeightReceiver(model=model).recv(_checkpoint_request(path))
    )
    assert ack.status is WeightUpdateStatus.APPLIED
    assert torch.equal(model.weight.detach(), torch.tensor([[1.0, 1.0]]))


def test_checkpoint_receiver_skips_a_stale_version(tmp_path):
    import torch

    from rlinf_rollout.weight_sync.checkpoint import CheckpointWeightReceiver

    path = tmp_path / "w.pt"
    torch.save({"weight": torch.tensor([[1.0, 2.0]])}, path)
    receiver = CheckpointWeightReceiver(model=_tiny_model())
    run_async(receiver.recv(_checkpoint_request(path, version=4)))

    ack = run_async(receiver.recv(_checkpoint_request(path, version=4)))
    assert ack.status is WeightUpdateStatus.SKIPPED
    assert ack.served_version == 4


def test_checkpoint_receiver_reports_a_missing_file(tmp_path):
    from rlinf_rollout.weight_sync.checkpoint import CheckpointWeightReceiver

    ack = run_async(
        CheckpointWeightReceiver(model=_tiny_model()).recv(
            _checkpoint_request(tmp_path / "missing.pt")
        )
    )
    assert ack.status is WeightUpdateStatus.FAILED
    assert "does not exist" in ack.error
    assert ack.served_version == -1


def test_checkpoint_receiver_rejects_another_transport(tmp_path):
    from rlinf_rollout.weight_sync.checkpoint import CheckpointWeightReceiver

    ack = run_async(
        CheckpointWeightReceiver(model=_tiny_model()).recv(
            WeightUpdateRequest(version=1, transport=WeightTransport.COLLECTIVE)
        )
    )
    assert ack.status is WeightUpdateStatus.FAILED
    assert "checkpoint" in ack.error


def test_checkpoint_receiver_accepts_a_custom_loader(tmp_path):
    import torch

    from rlinf_rollout.weight_sync.checkpoint import CheckpointWeightReceiver

    path = tmp_path / "weights.custom"
    path.write_text("ignored", encoding="utf-8")
    model = _tiny_model()
    receiver = CheckpointWeightReceiver(
        model=model,
        loader=lambda target: {"weight": torch.tensor([[9.0, 9.0]])},
    )
    ack = run_async(receiver.recv(_checkpoint_request(path, version=2)))
    assert ack.status is WeightUpdateStatus.APPLIED
    assert torch.equal(model.weight.detach(), torch.tensor([[9.0, 9.0]]))
    described = receiver.describe()
    assert described["transport"] == "checkpoint"
    assert described["served_version"] == 2
    assert described["last_checkpoint_path"] == str(path)


def test_checkpoint_receiver_is_exported_lazily():
    import rlinf_rollout.weight_sync as weight_sync

    assert "CheckpointWeightReceiver" in weight_sync.__all__
    from rlinf_rollout.weight_sync.checkpoint import CheckpointWeightReceiver

    assert weight_sync.CheckpointWeightReceiver is CheckpointWeightReceiver


def test_checkpoint_receiver_implements_the_v1_interface():
    from rlinf_rollout.api.v1 import WeightReceiver
    from rlinf_rollout.weight_sync.checkpoint import CheckpointWeightReceiver

    assert issubclass(CheckpointWeightReceiver, WeightReceiver)
    receiver = CheckpointWeightReceiver(model=_tiny_model())
    run_async(receiver.prepare(WeightUpdateRequest(version=0)))
    run_async(receiver.close())


# ---------------------------------------------------------------------------
# LLM result conversion (internal struct -> api/v1)
# ---------------------------------------------------------------------------


def test_rollout_result_to_api_maps_generation_facts():
    import torch

    from rlinf_rollout.api.v1 import FinishReason
    from rlinf_rollout.data.convert import rollout_result_to_api

    internal = _StubInternalRolloutResult(
        num_sequence=2,
        group_size=2,
        prompt_lengths=[3, 3],
        prompt_ids=[[1, 2, 3], [1, 2, 3]],
        response_lengths=[2, 4],
        response_ids=[[4, 5], [4, 5, 6, 7]],
        is_end=[True, False],
        answers=["42", "42"],
        rewards=torch.tensor([1.0, 0.0]),
        rollout_logprobs=[[-0.1, -0.2], [-0.3, -0.4, -0.5, -0.6]],
    )
    payload = rollout_result_to_api(
        internal,
        request_ids=("task-1", "task-1"),
        versions=(4, 4),
        producer="RolloutGroup",
        metadata={"task_id": "task-1"},
    )

    assert payload.schema_version == "v1"
    assert payload.num_sequences == 2 and payload.group_size == 2
    assert payload.num_prompts == 1
    assert payload.finish_reasons == [FinishReason.STOP, FinishReason.LENGTH]
    assert payload.is_end == [True, True]  # neither sequence was aborted
    assert payload.request_ids == ("task-1", "task-1")
    assert payload.versions == (4, 4)
    assert payload.rewards == [1.0, 0.0]
    assert payload.answers == ["42", "42"]
    assert payload.metadata.producer == "RolloutGroup"
    assert payload.metadata.extra == {"task_id": "task-1"}


def test_rollout_result_to_api_drops_trainer_only_fields():
    from rlinf_rollout.api.v1 import RolloutResult as ApiRolloutResult

    trainer_only = {
        "advantages",
        "returns",
        "values",
        "ref_logprobs",
        "recomputed_logprobs",
    }
    assert trainer_only.isdisjoint(ApiRolloutResult.field_names())


# ---------------------------------------------------------------------------
# Layout, packaging and decoupling
# ---------------------------------------------------------------------------

PHASE4_MODULES = (
    "serve.protocol",
    "serve.control",
    "serve.control_worker",
    "serve.task_source",
    "serve.controller",
    "serve.service",
    "serve.main",
    "client.rollout_client",
    "client.weight_sender",
    "weight_sync.checkpoint",
)


@pytest.mark.parametrize("dotted", PHASE4_MODULES)
def test_phase4_module_exists(dotted):
    base = PACKAGE_ROOT / Path(*dotted.split("."))
    assert base.with_suffix(".py").is_file(), f"missing module: {dotted}"


def test_bundled_service_configs_exist():
    assert EMBODIED_CONFIG.is_file()
    assert LLM_CONFIG.is_file()
    assert PROMPT_FILE.is_file()


def test_pyproject_registers_the_daemon_entry_point_and_packages():
    pyproject = tomllib.loads(
        (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert pyproject["project"]["scripts"] == {
        "rollout-serve": "rlinf_rollout.serve.main:main"
    }
    packages = set(pyproject["tool"]["setuptools"]["packages"])
    assert {"rlinf_rollout.serve", "rlinf_rollout.client"} <= packages


def test_service_configs_are_shipped_by_the_package_data_globs():
    pyproject = tomllib.loads(
        (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    globs = pyproject["tool"]["setuptools"]["package-data"]["rlinf_rollout"]
    assert "**/*.yaml" in globs, "service YAML configs would not ship in the wheel"
    assert "**/*.jsonl" in globs, "the example prompt file would not ship"


SERVE_AND_CLIENT_SOURCES = tuple(
    sorted(
        path
        for directory in ("serve", "client")
        for path in (PACKAGE_ROOT / directory).rglob("*.py")
        if "__pycache__" not in path.parts
    )
)


def _code_lines(path):
    """Yield the source lines of ``path`` with docstrings removed."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    docstring_lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            if isinstance(first.value.value, str):
                docstring_lines.update(
                    range(first.lineno, (first.end_lineno or first.lineno) + 1)
                )
    return [
        (lineno, line)
        for lineno, line in enumerate(source.splitlines(), start=1)
        if lineno not in docstring_lines
    ]


def test_serve_and_client_sources_were_discovered():
    names = {path.name for path in SERVE_AND_CLIENT_SOURCES}
    assert {"controller.py", "service.py", "main.py", "rollout_client.py"} <= names


@pytest.mark.parametrize(
    "path", SERVE_AND_CLIENT_SOURCES, ids=lambda path: f"{path.parent.name}/{path.name}"
)
def test_serve_and_client_never_read_trainer_config_sections(path):
    offenders = [
        f"{path.name}:{lineno}: {line.strip()}"
        for lineno, line in _code_lines(path)
        for section in ("cfg.actor.", "cfg.algorithm.", "cfg.runner.", "cfg.critic.")
        if section in line
    ]
    assert not offenders, (
        "trainer config sections leaked into the service:\n" + "\n".join(offenders)
    )


@pytest.mark.parametrize(
    "path", SERVE_AND_CLIENT_SOURCES, ids=lambda path: f"{path.parent.name}/{path.name}"
)
def test_serve_and_client_never_select_trainer_config_keys(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden = ("actor.", "algorithm.", "runner.", "critic.", "rollout_server.")
    offenders = [
        f"{path.name}:{node.lineno}: {node.value!r}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith(forbidden)
    ]
    assert not offenders, "trainer config selectors leaked in:\n" + "\n".join(offenders)


def test_serve_package_lazily_exports_its_heavy_symbols():
    import rlinf_rollout.serve as serve

    assert "RolloutService" in serve.__all__
    assert "ControlPlaneWorker" in serve.__all__
    # Leaf symbols resolve without a runtime.
    assert serve.ControlPlaneState is ControlPlaneState
    assert serve.RolloutService.__name__ == "RolloutService"
    with pytest.raises(AttributeError):
        serve.does_not_exist


def test_client_package_exports_the_three_verbs():
    import rlinf_rollout.client as client

    assert set(client.__all__) == {
        "CheckpointWeightPublisher",
        "RolloutClient",
        "RolloutClientError",
        "describe_collective_sender",
        "make_prompt_tasks",
    }
    for verb in ("push_weights", "get_trajectories", "submit_tasks"):
        assert callable(getattr(client.RolloutClient, verb))


def test_hf_rollout_worker_dispatches_weight_updates_by_transport():
    """The worker must expose the transport-dispatching entry the service calls."""
    source = (
        PACKAGE_ROOT / "workers" / "rollout" / "hf" / "huggingface_worker.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {"receive_weight_update", "sync_weights", "served_weight_version"} <= defined
    assert "CheckpointWeightReceiver" in source
    assert "WeightTransport.CHECKPOINT" in source
