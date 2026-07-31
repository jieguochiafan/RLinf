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

"""Lock the frozen ``v1`` rollout protocol.

The field sets and enum values below are the contract: changing them breaks
peers that were built against ``v1``. If a change is genuinely needed, add a new
``api/v2`` package instead of editing these expectations.
"""

import asyncio
import dataclasses
import inspect
from pathlib import Path

import pytest
import torch

from rlinf_rollout.api import v1
from rlinf_rollout.api.v1 import (
    SCHEMA_VERSION,
    ActionChunkResult,
    ConsumerSpec,
    EpisodeSpec,
    FinishReason,
    Metadata,
    PartitionAxis,
    PolicyInputs,
    PromptSpec,
    RolloutMode,
    RolloutResult,
    RolloutTask,
    SamplingParams,
    SchemaBase,
    SchemaError,
    SourceTopology,
    TaskKind,
    TaskSource,
    TensorLayout,
    TensorSpec,
    Trajectory,
    TrajectorySink,
    WeightReceiver,
    WeightSyncMode,
    WeightTransport,
    WeightUpdateAck,
    WeightUpdateRequest,
    WeightUpdateStatus,
)

# ---------------------------------------------------------------------------
# Frozen field sets
# ---------------------------------------------------------------------------

EXPECTED_FIELDS: dict[type, tuple[str, ...]] = {
    Metadata: ("schema_version", "producer", "created_at", "tags", "extra"),
    TensorSpec: (
        "schema_version",
        "name",
        "shape",
        "dtype",
        "shard_dim",
        "num_bytes",
    ),
    SourceTopology: (
        "schema_version",
        "group_name",
        "src_ranks",
        "world_size",
        "parallel_sizes",
        "rank_map",
        "endpoint",
    ),
    WeightUpdateRequest: (
        "schema_version",
        "version",
        "mode",
        "transport",
        "source",
        "tensors",
        "checkpoint_path",
        "compression",
        "bucket_size_bytes",
        "blocking",
        "metadata",
    ),
    WeightUpdateAck: (
        "schema_version",
        "version",
        "status",
        "served_version",
        "receiver_id",
        "num_tensors_applied",
        "num_bytes_received",
        "elapsed_seconds",
        "error",
        "metadata",
    ),
    PolicyInputs: (
        "schema_version",
        "producer",
        "layout",
        "tensors",
        "required_keys",
    ),
    ActionChunkResult: (
        "schema_version",
        "actions",
        "prev_logprobs",
        "prev_values",
        "intervene_flags",
        "versions",
        "policy_inputs",
        "metadata",
    ),
    Trajectory: (
        "schema_version",
        "trajectory_id",
        "layout",
        "num_steps",
        "num_envs",
        "max_episode_length",
        "model_weights_id",
        "actions",
        "rewards",
        "dones",
        "terminations",
        "truncations",
        "prev_logprobs",
        "prev_values",
        "intervene_flags",
        "versions",
        "curr_obs",
        "next_obs",
        "policy_inputs",
        "metadata",
    ),
    RolloutResult: (
        "schema_version",
        "num_sequences",
        "group_size",
        "request_ids",
        "prompt_ids",
        "prompt_lengths",
        "response_ids",
        "response_lengths",
        "finish_reasons",
        "response_masks",
        "rollout_logprobs",
        "prompt_texts",
        "response_texts",
        "answers",
        "image_data",
        "multi_modal_inputs",
        "versions",
        "rewards",
        "metadata",
    ),
    ConsumerSpec: (
        "schema_version",
        "num_partitions",
        "axis",
        "partition_sizes",
        "layout",
        "required_keys",
    ),
    SamplingParams: (
        "schema_version",
        "n",
        "temperature",
        "top_p",
        "top_k",
        "max_new_tokens",
        "min_new_tokens",
        "repetition_penalty",
        "stop",
        "stop_token_ids",
        "seed",
        "return_logprobs",
    ),
    PromptSpec: (
        "schema_version",
        "input_ids",
        "prompt_texts",
        "image_data",
        "multi_modal_inputs",
        "answers",
    ),
    EpisodeSpec: (
        "schema_version",
        "env_type",
        "env_ids",
        "num_envs",
        "group_size",
        "num_episodes",
        "num_chunk_steps",
        "max_episode_steps",
        "seeds",
        "auto_reset",
        "env_config",
    ),
    RolloutTask: (
        "schema_version",
        "kind",
        "task_id",
        "mode",
        "prompts",
        "sampling",
        "episode",
        "min_weight_version",
        "priority",
        "metadata",
    ),
}

EXPECTED_ENUM_VALUES: dict[type, tuple[str, ...]] = {
    WeightSyncMode: ("bucket", "patch"),
    WeightTransport: ("collective", "checkpoint", "http_push"),
    WeightUpdateStatus: ("applied", "skipped", "failed"),
    TensorLayout: ("time_major", "batch_major", "flat"),
    PartitionAxis: ("env", "sequence", "none"),
    FinishReason: ("stop", "length", "abort"),
    TaskKind: ("llm_generation", "embodied_episode", "embodied_eval"),
    RolloutMode: ("train", "eval"),
}


@pytest.mark.parametrize(
    ("message_cls", "expected"),
    list(EXPECTED_FIELDS.items()),
    ids=[cls.__name__ for cls in EXPECTED_FIELDS],
)
def test_field_set_is_frozen(message_cls: type, expected: tuple[str, ...]):
    assert message_cls.field_names() == expected


@pytest.mark.parametrize(
    ("enum_cls", "expected"),
    list(EXPECTED_ENUM_VALUES.items()),
    ids=[cls.__name__ for cls in EXPECTED_ENUM_VALUES],
)
def test_enum_values_are_frozen(enum_cls: type, expected: tuple[str, ...]):
    assert tuple(member.value for member in enum_cls) == expected


def test_every_exported_message_is_locked():
    """A newly exported message type must be added to ``EXPECTED_FIELDS``."""
    exported_messages = {
        getattr(v1, name)
        for name in v1.__all__
        if inspect.isclass(getattr(v1, name))
        and issubclass(getattr(v1, name), SchemaBase)
        and getattr(v1, name) is not SchemaBase
    }
    assert exported_messages == set(EXPECTED_FIELDS)


def test_every_exported_enum_is_locked():
    import enum as enum_module

    exported_enums = {
        getattr(v1, name)
        for name in v1.__all__
        if inspect.isclass(getattr(v1, name))
        and issubclass(getattr(v1, name), enum_module.Enum)
    }
    assert exported_enums == set(EXPECTED_ENUM_VALUES)


def test_all_exports_resolve():
    for name in v1.__all__:
        assert hasattr(v1, name), f"{name} listed in __all__ but not importable"


# ---------------------------------------------------------------------------
# schema_version semantics
# ---------------------------------------------------------------------------


def test_schema_version_is_v1():
    assert SCHEMA_VERSION == "v1"
    assert SchemaBase.SCHEMA_VERSION == "v1"


@pytest.mark.parametrize(
    "message",
    [
        Metadata(),
        TensorSpec(name="w"),
        SourceTopology(),
        WeightUpdateRequest(version=0),
        WeightUpdateAck(version=0),
        PolicyInputs(),
        ActionChunkResult(),
        Trajectory(),
        RolloutResult(num_sequences=0),
        ConsumerSpec(),
        SamplingParams(),
        PromptSpec(prompt_texts=["hi"]),
        EpisodeSpec(env_type="maniskill"),
        RolloutTask(kind=TaskKind.EMBODIED_EPISODE, episode=EpisodeSpec(env_type="x")),
    ],
    ids=lambda message: type(message).__name__,
)
def test_messages_carry_schema_version(message: SchemaBase):
    assert dataclasses.is_dataclass(message)
    assert message.schema_version == SCHEMA_VERSION
    assert "schema_version" in message.to_dict()


def test_wrong_schema_version_is_rejected():
    with pytest.raises(SchemaError, match="schema_version"):
        Metadata(schema_version="v0")


def test_unknown_field_is_rejected_on_decode():
    payload = SourceTopology(group_name="trainer").to_dict()
    payload["unexpected"] = 1
    with pytest.raises(SchemaError, match="Unknown field"):
        SourceTopology.from_dict(payload)


def test_to_dict_from_dict_roundtrip():
    request = WeightUpdateRequest(
        version=7,
        mode=WeightSyncMode.PATCH,
        source=SourceTopology(group_name="trainer", src_ranks=(0,), world_size=8),
        tensors=(TensorSpec(name="w", shape=(4, 4), dtype="bfloat16"),),
        compression="zstd",
    )
    restored = WeightUpdateRequest.from_dict(request.to_dict())
    assert restored == request
    assert restored.tensors[0].shape == (4, 4)


# ---------------------------------------------------------------------------
# Weight protocol behavior
# ---------------------------------------------------------------------------


def test_weight_request_rejects_negative_version():
    with pytest.raises(SchemaError, match="non-negative"):
        WeightUpdateRequest(version=-1)


def test_checkpoint_transport_requires_path():
    with pytest.raises(SchemaError, match="checkpoint_path"):
        WeightUpdateRequest(version=1, transport=WeightTransport.CHECKPOINT)
    request = WeightUpdateRequest(
        version=1,
        transport=WeightTransport.CHECKPOINT,
        checkpoint_path="/ckpt/global_step_1",
    )
    assert request.transport is WeightTransport.CHECKPOINT


def test_collective_transport_requires_src_ranks():
    with pytest.raises(SchemaError, match="src_ranks"):
        WeightUpdateRequest(version=1, source=SourceTopology(src_ranks=()))


def test_enum_values_accept_plain_strings():
    request = WeightUpdateRequest(version=1, mode="patch", transport="http_push")
    assert request.mode is WeightSyncMode.PATCH
    assert request.transport is WeightTransport.HTTP_PUSH


def test_failed_ack_requires_error():
    with pytest.raises(SchemaError, match="error"):
        WeightUpdateAck(version=1, status=WeightUpdateStatus.FAILED)
    ack = WeightUpdateAck(version=1, status="failed", error="nccl timeout")
    assert ack.status is WeightUpdateStatus.FAILED


def test_weight_receiver_is_abstract_and_implementable():
    with pytest.raises(TypeError):
        WeightReceiver()

    class DummyReceiver(WeightReceiver):
        def __init__(self):
            self._version = -1

        @property
        def served_version(self) -> int:
            return self._version

        async def recv(self, request: WeightUpdateRequest) -> WeightUpdateAck:
            if request.version <= self._version:
                return WeightUpdateAck(
                    version=request.version,
                    status=WeightUpdateStatus.SKIPPED,
                    served_version=self._version,
                )
            self._version = request.version
            return WeightUpdateAck(
                version=request.version,
                status=WeightUpdateStatus.APPLIED,
                served_version=self._version,
                receiver_id="dummy:0",
            )

    async def scenario():
        receiver = DummyReceiver()
        assert receiver.served_version == -1
        await receiver.prepare(WeightUpdateRequest(version=1))
        first = await receiver.recv(WeightUpdateRequest(version=1))
        second = await receiver.recv(WeightUpdateRequest(version=1))
        await receiver.close()
        return first, second, receiver

    first, second, receiver = asyncio.run(scenario())
    assert first.status is WeightUpdateStatus.APPLIED
    assert second.status is WeightUpdateStatus.SKIPPED
    assert receiver.describe()["served_version"] == 1


# ---------------------------------------------------------------------------
# Output protocol behavior
# ---------------------------------------------------------------------------


def _make_trajectory(num_steps: int = 3, num_envs: int = 2) -> Trajectory:
    return Trajectory(
        trajectory_id="traj-0",
        num_steps=num_steps,
        num_envs=num_envs,
        max_episode_length=50,
        actions=torch.zeros(num_steps, num_envs, 7),
        rewards=torch.zeros(num_steps, num_envs, 1),
        versions=torch.zeros(num_steps, num_envs, 1),
        policy_inputs=PolicyInputs(
            producer="openvla_oft",
            tensors={
                "input_ids": torch.zeros(num_steps, num_envs, 8, dtype=torch.long)
            },
            required_keys=("input_ids",),
        ),
    )


def test_trajectory_shape_validation():
    trajectory = _make_trajectory()
    trajectory.validate()
    assert trajectory.num_transitions == 6
    assert trajectory.layout is TensorLayout.TIME_MAJOR

    trajectory.actions = torch.zeros(3, 5, 7)
    with pytest.raises(SchemaError, match="leading dims"):
        trajectory.validate()


def test_trajectory_has_no_training_side_fields():
    """Advantages/returns stay on the trainer; bootstrap values in a plugin."""
    forbidden = {
        "advantages",
        "returns",
        "ref_logprobs",
        "bootstrap_values",
        "forward_inputs",
    }
    assert forbidden.isdisjoint(set(Trajectory.field_names()))
    assert forbidden.isdisjoint(set(RolloutResult.field_names()))
    assert "forward_inputs" not in ActionChunkResult.field_names()


def test_policy_inputs_requires_declared_keys():
    with pytest.raises(SchemaError, match="required_keys"):
        PolicyInputs(producer="p", tensors={}, required_keys=("input_ids",))

    inputs = PolicyInputs(
        producer="p", tensors={"x": torch.zeros(2, 3)}, required_keys=("x",)
    )
    assert inputs.describe() == {"x": {"shape": (2, 3), "dtype": "torch.float32"}}


def test_action_chunk_result_batch_size():
    assert ActionChunkResult().batch_size == 0
    assert ActionChunkResult(actions=torch.zeros(4, 14)).batch_size == 4


def test_rollout_result_group_consistency():
    with pytest.raises(SchemaError, match="divisible"):
        RolloutResult(num_sequences=5, group_size=2)

    result = RolloutResult(
        num_sequences=4,
        group_size=2,
        finish_reasons=["stop", "length", "abort", FinishReason.STOP],
        versions=(3, 3, 3, 3),
    )
    assert result.num_prompts == 2
    assert result.finish_reasons[0] is FinishReason.STOP
    assert result.is_end == [True, True, False, True]


def test_consumer_spec_declares_partitioning():
    spec = ConsumerSpec(
        num_partitions=4, axis=PartitionAxis.ENV, partition_sizes=(2, 2, 2, 2)
    )
    assert spec.axis is PartitionAxis.ENV

    with pytest.raises(SchemaError, match="partition_sizes"):
        ConsumerSpec(num_partitions=4, axis="env", partition_sizes=(2, 2))
    with pytest.raises(SchemaError, match="num_partitions must be 1"):
        ConsumerSpec(num_partitions=2, axis=PartitionAxis.NONE)
    with pytest.raises(SchemaError, match="num_partitions must be positive"):
        ConsumerSpec(num_partitions=0)


def test_trajectory_sink_is_abstract_and_implementable():
    with pytest.raises(TypeError):
        TrajectorySink()

    class DummySink(TrajectorySink):
        def __init__(self):
            self.items = []

        @property
        def consumer_spec(self) -> ConsumerSpec:
            return ConsumerSpec(num_partitions=2, axis=PartitionAxis.ENV)

        async def put(self, item):
            self.items.append(item)

    async def scenario():
        sink = DummySink()
        await sink.put(_make_trajectory())
        await sink.put(RolloutResult(num_sequences=2, group_size=2))
        await sink.flush()
        await sink.close()
        return sink

    sink = asyncio.run(scenario())
    assert sink.consumer_spec.num_partitions == 2
    assert isinstance(sink.items[0], Trajectory)
    assert isinstance(sink.items[1], RolloutResult)


# ---------------------------------------------------------------------------
# Task protocol behavior
# ---------------------------------------------------------------------------


def test_task_payload_is_exclusive_per_kind():
    with pytest.raises(SchemaError, match="require 'prompts'"):
        RolloutTask(kind=TaskKind.LLM_GENERATION)
    with pytest.raises(SchemaError, match="require 'episode'"):
        RolloutTask(kind=TaskKind.EMBODIED_EPISODE)
    with pytest.raises(SchemaError, match="must not carry 'episode'"):
        RolloutTask(
            kind=TaskKind.LLM_GENERATION,
            prompts=PromptSpec(input_ids=[[1, 2]]),
            episode=EpisodeSpec(env_type="maniskill"),
        )
    with pytest.raises(SchemaError, match="must not carry 'prompts'"):
        RolloutTask(
            kind=TaskKind.EMBODIED_EPISODE,
            episode=EpisodeSpec(env_type="maniskill"),
            prompts=PromptSpec(input_ids=[[1, 2]]),
        )


def test_eval_task_requires_eval_mode():
    with pytest.raises(SchemaError, match="mode='eval'"):
        RolloutTask(kind=TaskKind.EMBODIED_EVAL, episode=EpisodeSpec(env_type="libero"))
    task = RolloutTask(
        kind=TaskKind.EMBODIED_EVAL,
        mode=RolloutMode.EVAL,
        episode=EpisodeSpec(env_type="libero"),
    )
    assert task.mode is RolloutMode.EVAL


def test_task_id_is_auto_generated_and_unique():
    episode = EpisodeSpec(env_type="maniskill")
    first = RolloutTask(kind=TaskKind.EMBODIED_EPISODE, episode=episode)
    second = RolloutTask(kind=TaskKind.EMBODIED_EPISODE, episode=episode)
    assert first.task_id and second.task_id
    assert first.task_id != second.task_id

    explicit = RolloutTask(
        kind=TaskKind.EMBODIED_EPISODE, task_id="fixed", episode=episode
    )
    assert explicit.task_id == "fixed"


def test_prompt_spec_requires_prompts():
    with pytest.raises(SchemaError, match="input_ids or prompt_texts"):
        PromptSpec()
    assert PromptSpec(input_ids=[[1], [2], [3]]).num_prompts == 3
    assert PromptSpec(prompt_texts=["a", "b"]).num_prompts == 2


def test_episode_spec_validation():
    with pytest.raises(SchemaError, match="env_type"):
        EpisodeSpec(env_type="")
    with pytest.raises(SchemaError, match="num_envs must be positive"):
        EpisodeSpec(env_type="maniskill", num_envs=0)
    with pytest.raises(SchemaError, match="divisible by group_size"):
        EpisodeSpec(env_type="maniskill", num_envs=6, group_size=4)
    with pytest.raises(SchemaError, match="seeds has"):
        EpisodeSpec(env_type="maniskill", num_envs=4, seeds=(1, 2))
    spec = EpisodeSpec(
        env_type="maniskill", num_envs=8, group_size=4, seeds=tuple(range(8))
    )
    assert spec.num_envs == 8


def test_sampling_params_validation():
    with pytest.raises(SchemaError, match="n must be positive"):
        SamplingParams(n=0)
    with pytest.raises(SchemaError, match="temperature"):
        SamplingParams(temperature=-0.1)
    assert SamplingParams(n=8, temperature=0.0).top_k == -1


def test_task_source_is_abstract_and_implementable():
    with pytest.raises(TypeError):
        TaskSource()

    class DummySource(TaskSource):
        def __init__(self, tasks):
            self._tasks = list(tasks)
            self.done = []

        async def next_batch(self, max_num_tasks: int) -> list[RolloutTask]:
            batch = self._tasks[:max_num_tasks]
            self._tasks = self._tasks[max_num_tasks:]
            return batch

        async def exhausted(self) -> bool:
            return not self._tasks

        async def report_done(self, task_id, error=None):
            self.done.append((task_id, error))

    async def scenario():
        source = DummySource(
            RolloutTask(
                kind=TaskKind.LLM_GENERATION,
                prompts=PromptSpec(input_ids=[[1, 2, 3]]),
                sampling=SamplingParams(n=4),
            )
            for _ in range(3)
        )
        batch = await source.next_batch(2)
        await source.report_done(batch[0].task_id)
        exhausted_after_two = await source.exhausted()
        rest = await source.next_batch(8)
        exhausted_at_end = await source.exhausted()
        await source.close()
        return batch, rest, exhausted_after_two, exhausted_at_end, source

    batch, rest, mid, end, source = asyncio.run(scenario())
    assert len(batch) == 2 and len(rest) == 1
    assert mid is False and end is True
    assert source.done == [(batch[0].task_id, None)]


# ---------------------------------------------------------------------------
# Independence from the training-side `rlinf` package
# ---------------------------------------------------------------------------

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _python_sources() -> list[Path]:
    return [
        path for path in PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in path.parts
    ]


def test_no_imports_from_the_training_package():
    offenders = []
    for path in _python_sources():
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = line.strip()
            if stripped.startswith(("import rlinf.", "from rlinf.")) or stripped in {
                "import rlinf",
                "from rlinf import",
            }:
                offenders.append(
                    f"{path.relative_to(PACKAGE_ROOT)}:{lineno}: {stripped}"
                )
    assert not offenders, (
        "rlinf_rollout must not import the training-side rlinf package:\n"
        + "\n".join(offenders)
    )


def test_every_source_directory_is_a_package():
    missing = []
    for directory in PACKAGE_ROOT.rglob("*"):
        if not directory.is_dir() or "__pycache__" in directory.parts:
            continue
        has_modules = any(
            child.suffix == ".py" and child.name != "__init__.py"
            for child in directory.iterdir()
            if child.is_file()
        )
        if has_modules and not (directory / "__init__.py").exists():
            missing.append(str(directory.relative_to(PACKAGE_ROOT)))
    assert not missing, f"Missing __init__.py in: {sorted(missing)}"
