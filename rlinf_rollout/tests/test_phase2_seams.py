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

"""Behavioural tests for the Phase 2 decoupling seams.

Covers the pieces that replaced trainer-coupled code inside the workers:
bootstrap reward shaping, the trajectory post-processing hook, internal ->
``api/v1`` conversion, ``TrajectorySink`` backends and the collective
``WeightReceiver``. All of it is pure Python plus ``torch``, so no Ray, GPU or
simulator SDK is needed.
"""

import asyncio

import pytest
import torch
from omegaconf import OmegaConf

from rlinf_rollout.api.v1 import (
    ConsumerSpec,
    PartitionAxis,
    SourceTopology,
    TensorLayout,
    WeightSyncMode,
    WeightTransport,
    WeightUpdateRequest,
    WeightUpdateStatus,
)
from rlinf_rollout.postprocess import (
    BootstrapRewardShaper,
    TrajectoryPostprocessor,
    estimate_bootstrap_values,
    load_trajectory_postprocessor,
)

# ---------------------------------------------------------------------------
# Bootstrap reward shaping
# ---------------------------------------------------------------------------


def _rewards() -> torch.Tensor:
    return torch.tensor([[1.0, 2.0], [3.0, 4.0]])


def test_bootstrap_shaper_adds_discounted_value_on_truncation():
    shaper = BootstrapRewardShaper(gamma=0.9, bootstrap_type="standard")
    shaped = shaper.shape(
        rewards=_rewards(),
        dones=torch.tensor([[False, True], [False, False]]),
        truncations=torch.tensor([[False, True], [False, False]]),
        bootstrap_values=torch.tensor([[10.0], [20.0]]),
    )
    # Only env 0 was truncated: 2.0 + 0.9 * 10.0
    assert shaped.tolist() == [[1.0, 11.0], [3.0, 4.0]]


def test_bootstrap_shaper_done_mode_bootstraps_terminations_too():
    shaper = BootstrapRewardShaper(gamma=0.5, bootstrap_type="done")
    shaped = shaper.shape(
        rewards=_rewards(),
        dones=torch.tensor([[False, True], [False, True]]),
        truncations=torch.tensor([[False, False], [False, False]]),
        bootstrap_values=torch.tensor([[10.0], [20.0]]),
    )
    assert shaped.tolist() == [[1.0, 7.0], [3.0, 14.0]]


def test_bootstrap_shaper_is_a_no_op_when_disabled_or_without_auto_reset():
    values = torch.tensor([[10.0], [20.0]])
    dones = torch.tensor([[False, True], [False, True]])
    for shaper in (
        BootstrapRewardShaper(enabled=False, gamma=0.9),
        BootstrapRewardShaper(auto_reset=False, gamma=0.9),
    ):
        shaped = shaper.shape(
            rewards=_rewards(),
            dones=dones,
            truncations=dones,
            bootstrap_values=values,
        )
        assert shaped.tolist() == _rewards().tolist()


def test_bootstrap_shaper_blends_reward_model_output():
    shaper = BootstrapRewardShaper(
        enabled=False, reward_weight=2.0, env_reward_weight=0.5
    )
    shaped = shaper.shape(
        rewards=_rewards(),
        dones=None,
        truncations=None,
        bootstrap_values=None,
        reward_model_output=torch.ones_like(_rewards()),
    )
    assert shaped.tolist() == [[2.5, 3.0], [3.5, 4.0]]


def test_bootstrap_shaper_returns_none_without_rewards():
    shaper = BootstrapRewardShaper()
    assert (
        shaper.shape(rewards=None, dones=None, truncations=None, bootstrap_values=None)
        is None
    )


def test_bootstrap_shaper_reads_the_rollout_config_keys():
    cfg = OmegaConf.create(
        {
            "rollout": {
                "postprocess": {
                    "bootstrap": {"enabled": True, "type": "done", "gamma": 0.97}
                }
            },
            "reward": {"reward_weight": 3.0, "env_reward_weight": 0.25},
            "env": {"train": {"auto_reset": False}},
        }
    )
    shaper = BootstrapRewardShaper.from_config(cfg)
    assert (shaper.gamma, shaper.bootstrap_type) == (0.97, "done")
    assert shaper.auto_reset is False
    assert (shaper.reward_weight, shaper.env_reward_weight) == (3.0, 0.25)


# ---------------------------------------------------------------------------
# Bootstrap value estimation
# ---------------------------------------------------------------------------


class _NoValueHead:
    pass


class _WithValueHead:
    value_head = object()


def test_estimate_bootstrap_values_requires_final_obs_and_a_value_head():
    def predict(_obs):
        raise AssertionError("must not run")

    assert estimate_bootstrap_values(_WithValueHead(), predict, None) is None
    assert estimate_bootstrap_values(_NoValueHead(), predict, {"states": 1}) is None


def test_estimate_bootstrap_values_uses_prev_values():
    def predict(_obs):
        return torch.zeros(2, 4), {"prev_values": torch.tensor([[5.0], [6.0]])}

    values = estimate_bootstrap_values(_WithValueHead(), predict, {"states": 1})
    assert values.tolist() == [[5.0], [6.0]]
    assert values.device.type == "cpu"


def test_estimate_bootstrap_values_falls_back_to_zeros():
    def predict(_obs):
        return torch.ones(3, 4), {"prev_values": None}

    values = estimate_bootstrap_values(_WithValueHead(), predict, {"states": 1})
    assert values.tolist() == [[0.0], [0.0], [0.0]]


# ---------------------------------------------------------------------------
# Trajectory post-processing hook
# ---------------------------------------------------------------------------


class _TaggingPostprocessor(TrajectoryPostprocessor):
    def process(self, trajectory):
        trajectory.model_weights_id = "postprocessed"
        return trajectory


class _NotAPostprocessor:
    pass


def test_load_trajectory_postprocessor_accepts_none_and_a_valid_spec():
    assert load_trajectory_postprocessor(None) is None
    assert load_trajectory_postprocessor("") is None
    loaded = load_trajectory_postprocessor(f"{__name__}:_TaggingPostprocessor")
    assert isinstance(loaded, TrajectoryPostprocessor)


@pytest.mark.parametrize(
    ("spec", "match"),
    (
        ("no_colon_here", "module:attr"),
        (f"{__name__}:missing_attr", "has no attribute"),
        (f"{__name__}:_NotAPostprocessor", "not a"),
    ),
)
def test_load_trajectory_postprocessor_rejects_bad_specs(spec, match):
    with pytest.raises(ValueError, match=match):
        load_trajectory_postprocessor(spec)


# ---------------------------------------------------------------------------
# internal -> api/v1 conversion
# ---------------------------------------------------------------------------


def _internal_trajectory():
    from rlinf_rollout.data.embodied_io_struct import Trajectory as InternalTrajectory

    return InternalTrajectory(
        max_episode_length=16,
        model_weights_id="abc",
        actions=torch.zeros(4, 3, 14),
        rewards=torch.zeros(4, 3, 2),
        dones=torch.zeros(5, 3, 2, dtype=torch.bool),
        prev_logprobs=torch.zeros(4, 3, 2),
        versions=torch.zeros(4, 3, 1),
        forward_inputs={"action": torch.zeros(4, 3, 14), "mask": None},
        curr_obs={"states": torch.zeros(4, 3, 7)},
    )


def test_trajectory_to_api_maps_fields_and_wraps_forward_inputs():
    from rlinf_rollout.data.convert import trajectory_to_api

    payload = trajectory_to_api(
        _internal_trajectory(), producer="openvla", required_keys=("action",)
    )
    assert payload.schema_version == "v1"
    assert payload.layout is TensorLayout.TIME_MAJOR
    assert (payload.num_steps, payload.num_envs) == (4, 3)
    assert payload.max_episode_length == 16
    assert payload.model_weights_id == "abc"
    assert payload.policy_inputs.producer == "openvla"
    # ``None`` entries are dropped so the schema only carries real tensors.
    assert set(payload.policy_inputs.tensors) == {"action"}
    assert payload.policy_inputs.required_keys == ("action",)
    assert "states" in payload.curr_obs
    # Training-only quantities have no v1 counterpart.
    assert not hasattr(payload, "advantages")
    assert not hasattr(payload, "returns")


def test_trajectory_to_api_tolerates_uneven_step_counts_by_default():
    from rlinf_rollout.api.v1.common import SchemaError
    from rlinf_rollout.data.convert import trajectory_to_api

    trajectory = _internal_trajectory()
    trajectory.rewards = torch.zeros(5, 3, 2)  # terminal obs adds one reward row
    trajectory_to_api(trajectory, producer="openvla")  # no raise
    with pytest.raises(SchemaError):
        trajectory_to_api(trajectory, producer="openvla", validate=True)


def test_chunk_result_to_api_drops_bootstrap_values_into_metadata():
    from rlinf_rollout.data.convert import chunk_result_to_api
    from rlinf_rollout.data.embodied_io_struct import RolloutResult

    chunk = chunk_result_to_api(
        RolloutResult(
            actions=torch.zeros(3, 14),
            prev_logprobs=torch.zeros(3, 2),
            prev_values=torch.zeros(3, 1),
            bootstrap_values=torch.zeros(3, 1),
            versions=torch.zeros(3, 1),
            forward_inputs={"action": torch.zeros(3, 14)},
        ),
        producer="openpi",
        metadata={"has_bootstrap_values": True},
    )
    assert chunk.batch_size == 3
    assert chunk.policy_inputs.layout is TensorLayout.FLAT
    assert chunk.metadata.extra["has_bootstrap_values"] is True
    assert not hasattr(chunk, "bootstrap_values")


# ---------------------------------------------------------------------------
# Trajectory sinks
# ---------------------------------------------------------------------------


class _FakeWork:
    def __init__(self):
        self.waited = False

    async def async_wait(self):
        self.waited = True


class _FakeChannel:
    def __init__(self):
        self.puts: list[tuple] = []
        self.works: list[_FakeWork] = []

    def put(self, item, weight=0, key=None, async_op=False):
        del async_op
        self.puts.append((item, weight, key))
        work = _FakeWork()
        self.works.append(work)
        return work


def test_channel_sink_round_robins_over_declared_partitions():
    from rlinf_rollout.sinks import ChannelTrajectorySink

    channel = _FakeChannel()
    spec = ConsumerSpec(num_partitions=3, axis=PartitionAxis.ENV)
    sink = ChannelTrajectorySink(channel, spec, keys=["k0", "k1", "k2"])
    assert sink.consumer_spec.num_partitions == 3

    asyncio.run(_put_many(sink, ["a", "b", "c", "d"]))
    assert [key for _, _, key in channel.puts] == ["k0", "k1", "k2", "k0"]
    assert [item for item, _, _ in channel.puts] == ["a", "b", "c", "d"]


async def _put_many(sink, items):
    for item in items:
        await sink.put(item)


def test_channel_sink_honours_an_explicit_partition_and_flushes():
    from rlinf_rollout.sinks import ChannelTrajectorySink

    channel = _FakeChannel()
    sink = ChannelTrajectorySink(
        channel, ConsumerSpec(num_partitions=2, axis=PartitionAxis.ENV), keys=["a", "b"]
    )

    async def scenario():
        await sink.put("x", partition=1)
        with pytest.raises(IndexError):
            await sink.put("y", partition=5)
        await sink.close()

    asyncio.run(scenario())
    assert [key for _, _, key in channel.puts] == ["b"]
    assert all(work.waited for work in channel.works)


def test_channel_sink_rejects_a_key_count_mismatch():
    from rlinf_rollout.sinks import ChannelTrajectorySink

    with pytest.raises(ValueError, match="partitions"):
        ChannelTrajectorySink(
            _FakeChannel(),
            ConsumerSpec(num_partitions=3, axis=PartitionAxis.ENV),
            keys=["only-one"],
        )


def test_null_sink_defaults_to_a_single_unpartitioned_stream():
    from rlinf_rollout.sinks import NullTrajectorySink

    sink = NullTrajectorySink()
    assert sink.consumer_spec.num_partitions == 1
    assert sink.consumer_spec.axis is PartitionAxis.NONE
    asyncio.run(_put_many(sink, ["a", "b"]))
    assert sink.num_dropped == 2


# ---------------------------------------------------------------------------
# Collective weight receiver
# ---------------------------------------------------------------------------


class _FakeSyncer:
    """Minimal stand-in for ``WeightSyncer`` (no torch.distributed needed)."""

    comm_options = None

    def __init__(self, versions=(7,), fail=False):
        self._versions = list(versions)
        self._fail = fail
        self._receiver_initialized = False
        self.init_calls = 0
        self.apply_calls = 0

    def receiver_initialized(self):
        return self._receiver_initialized

    async def init_receiver(self, state_dict, recv, send=None):
        del state_dict, recv, send
        self.init_calls += 1
        self._receiver_initialized = True

    async def apply(self, model, recv):
        del model, recv
        self.apply_calls += 1
        if self._fail:
            raise RuntimeError("broadcast died")
        return self._versions[min(self.apply_calls - 1, len(self._versions) - 1)]


class _FakeModel:
    def __init__(self):
        self.global_step = None

    def state_dict(self):
        return {}

    def set_global_step(self, step):
        self.global_step = step


class _FakeWorker:
    torch_platform = None

    def broadcast(self, *args, **kwargs):
        raise AssertionError("the fake syncer never calls the transport")

    def send(self, *args, **kwargs):
        raise AssertionError("the fake syncer never calls the transport")


def _request(transport=WeightTransport.COLLECTIVE, **kwargs):
    return WeightUpdateRequest(
        version=3,
        mode=WeightSyncMode.BUCKET,
        transport=transport,
        source=SourceTopology(group_name="trainer", src_ranks=(0,), world_size=4),
        **kwargs,
    )


def _receiver(syncer, model=None):
    from rlinf_rollout.weight_sync.receiver import CollectiveWeightReceiver

    return CollectiveWeightReceiver(
        worker=_FakeWorker(),
        syncer=syncer,
        model=model or _FakeModel(),
        receiver_group_name="rollout",
        receiver_ranks=[0, 1],
        is_handshake_sender=True,
        receiver_id="rollout:0",
    )


def test_receiver_applies_weights_and_reports_the_served_version():
    syncer = _FakeSyncer(versions=(7, 8))
    model = _FakeModel()
    receiver = _receiver(syncer, model)
    assert receiver.served_version == -1

    ack = asyncio.run(receiver.recv(_request()))
    assert ack.status is WeightUpdateStatus.APPLIED
    assert ack.served_version == 7
    assert ack.version == 3  # the request's minimum-expected version
    assert ack.receiver_id == "rollout:0"
    assert receiver.served_version == 7
    assert model.global_step == 7
    # The handshake runs exactly once across repeated updates.
    assert asyncio.run(receiver.recv(_request())).served_version == 8
    assert syncer.init_calls == 1
    assert syncer.apply_calls == 2


def test_receiver_reports_failures_through_the_ack():
    receiver = _receiver(_FakeSyncer(fail=True))
    ack = asyncio.run(receiver.recv(_request()))
    assert ack.status is WeightUpdateStatus.FAILED
    assert "broadcast died" in ack.error
    assert ack.served_version == -1
    assert receiver.served_version == -1


def test_receiver_rejects_non_collective_transports():
    receiver = _receiver(_FakeSyncer())
    ack = asyncio.run(
        receiver.recv(
            _request(transport=WeightTransport.CHECKPOINT, checkpoint_path="/tmp/ckpt")
        )
    )
    assert ack.status is WeightUpdateStatus.FAILED
    assert "collective" in ack.error


def test_receiver_describe_reports_the_backend():
    receiver = _receiver(_FakeSyncer())
    info = receiver.describe()
    assert info["transport"] == WeightTransport.COLLECTIVE.value
    assert info["receiver"] == "CollectiveWeightReceiver"
    assert info["receiver_group_name"] == "rollout"
    assert info["receiver_ranks"] == (0, 1)
