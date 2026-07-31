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

"""Async HuggingFace policy rollout worker.

Ported from the training repo's ``rlinf/workers/rollout/hf/{huggingface_worker,
async_huggingface_worker}.py``, keeping only the async/service path and removing
every trainer coupling:

* ``cfg.actor.model`` -> ``cfg.policy.model``; ``cfg.runner.*`` -> ``cfg.rollout.*``.
* ``cfg.algorithm.{loss_type,adv_type,dagger,staleness_threshold}`` ->
  ``cfg.rollout.{plugins,staleness_threshold}``.
* ``setup_weight_sync``'s hardcoded actor group/rank ->
  :class:`~rlinf_rollout.weight_sync.CollectiveWeightReceiver` driven by a
  :class:`~rlinf_rollout.api.v1.WeightUpdateRequest`.
* ``get_bootstrap_values`` -> :func:`rlinf_rollout.postprocess.estimate_bootstrap_values`.
* DAgger expert / RLT helpers -> :mod:`rlinf_rollout.plugins`.
"""

import asyncio
import copy
import time
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from rlinf_rollout.api.v1 import (
    SourceTopology,
    WeightSyncMode,
    WeightTransport,
    WeightUpdateAck,
    WeightUpdateRequest,
    WeightUpdateStatus,
)
from rlinf_rollout.config import RolloutConfig, RolloutMode
from rlinf_rollout.data.embodied_io_struct import RolloutResult
from rlinf_rollout.models import get_model
from rlinf_rollout.models.embodiment.base_policy import BasePolicy
from rlinf_rollout.postprocess import estimate_bootstrap_values
from rlinf_rollout.scheduler import Channel, Cluster, Worker, split_channel_message
from rlinf_rollout.utils.placement import HybridComponentPlacement
from rlinf_rollout.weight_sync import CollectiveWeightReceiver, WeightSyncer

if TYPE_CHECKING:
    from rlinf_rollout.weight_sync import CheckpointWeightReceiver

__all__ = ["AsyncMultiStepRolloutWorker"]

#: Model types whose ``predict_action_batch`` takes ``mode`` instead of sampling kwargs.
_MODE_ONLY_MODELS = (
    "openpi",
    "openpi_pytorch",
    "mlp_policy",
    "gr00t",
    "gr00t_n1d6",
    "gr00t_n1d7",
    "abot_m0",
    "dreamzero",
    "cnn_policy",
    "cfg_model",
)

#: Model types that accept ``return_obs``.
_RETURN_OBS_MODELS = ("cnn_policy", "flow_policy", "mlp_policy")


class AsyncMultiStepRolloutWorker(Worker):
    """Serves action chunks to env workers and applies pushed policy weights.

    The worker runs as a resident coroutine: :meth:`generate` never returns until
    :meth:`stop` is called, and weight updates are interleaved either inline or in
    the background (``rollout.weight_sync.no_wait``).
    """

    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.rollout_cfg = RolloutConfig.from_dictconfig(cfg)
        self.should_stop = False

        self.only_eval = self.rollout_cfg.mode == RolloutMode.EVAL
        self.model_cfg = self.rollout_cfg.policy_model_cfg
        self.device = self.torch_platform.current_device()

        self.num_pipeline_stages = self.rollout_cfg.stage_num
        self.enable_offload = bool(cfg.rollout.enable_offload)
        self.enable_cuda_graph = bool(cfg.rollout.enable_cuda_graph)
        self.env_decoupled_mode = self.rollout_cfg.decoupled
        self.rollout_queue_size = int(cfg.rollout.rollout_queue_size)
        self.collect_prev_infos = bool(cfg.rollout.collect_prev_infos)
        self.collect_transitions = bool(cfg.rollout.collect_transitions)

        self.placement = HybridComponentPlacement(cfg, Cluster())
        rollout_world_size = self.placement.get_world_size("rollout")
        self._weight_sync_rollout_ranks = list(range(rollout_world_size))
        self._weight_sync_is_sender = self._rank == 0

        self.enable_train = self.rollout_cfg.enable_train
        self.enable_eval = self.rollout_cfg.enable_eval
        self.rollout_epoch = self.rollout_cfg.rollout_epoch
        self.eval_rollout_epoch = self.rollout_cfg.eval_rollout_epoch

        self.enable_dagger = self.rollout_cfg.plugin_enabled("dagger")
        self.enable_opd = self.rollout_cfg.plugin_enabled("opd")
        self.enable_rlt = self.rollout_cfg.plugin_enabled("rlt")
        self.expert_model = None
        self.rlt_feature_model = None
        self.rlt_route = None

        self.total_num_train_envs = self.rollout_cfg.total_num_train_envs
        self.total_num_eval_envs = self.rollout_cfg.total_num_eval_envs
        self.train_batch_size = self.rollout_cfg.train_batch_size
        self.eval_batch_size = self.rollout_cfg.eval_batch_size
        self.per_node_train_batch_size = self.train_batch_size // self._world_size
        self.per_node_eval_batch_size = self.eval_batch_size // self._world_size
        self.n_train_chunk_steps = self.rollout_cfg.n_train_chunk_steps
        self.n_eval_chunk_steps = self.rollout_cfg.n_eval_chunk_steps

        self.version = 0
        self.finished_episodes: Optional[int] = None
        self.hf_model: Optional[BasePolicy] = None
        self.weight_receiver: Optional[CollectiveWeightReceiver] = None
        self._checkpoint_receiver: Optional["CheckpointWeightReceiver"] = None
        self._weight_syncer: Optional[WeightSyncer] = None
        self._weight_source: Optional[SourceTopology] = None
        self._weight_sync_mode = WeightSyncMode.BUCKET

        if not self.only_eval:
            weight_sync_cfg = OmegaConf.select(cfg, "rollout.weight_sync")
            self._weight_syncer = WeightSyncer.create(weight_sync_cfg)
            self._weight_sync_mode = WeightSyncMode(weight_sync_cfg.type)
            source_cfg = weight_sync_cfg.source
            self._weight_source = SourceTopology(
                group_name=str(source_cfg.group_name),
                src_ranks=(int(source_cfg.get("src_rank", 0)),),
                world_size=int(source_cfg.get("world_size", 0)),
            )

        assert not (self.enable_offload and self.enable_train), (
            "rollout.enable_offload is not supported while collecting; the async "
            "rollout worker must stay resident on device."
        )

        if self.env_decoupled_mode:
            # In decoupled mode the rollout worker answers whichever env rank asked,
            # so it records the incoming routes per tag and replies along them.
            self.batch_router: dict[str, list] = {"rollout_results": []}

        # Async/service state.
        self._generate_task: Optional[asyncio.Task] = None
        self.staleness_threshold = OmegaConf.select(
            cfg, "rollout.staleness_threshold", default=None
        )
        self.sync_rollout_weight_time = max(
            1, self.num_pipeline_stages * self.n_train_chunk_steps * self.rollout_epoch
        )
        self._background_weight_sync_active = bool(
            OmegaConf.select(cfg, "rollout.weight_sync.no_wait", default=False)
        )
        self._weight_sync_requested = False
        self._weight_sync_work: Optional[asyncio.Task] = None
        self._weight_sync_apply_total = 0
        self._weight_sync_coalesced_total = 0
        self._weight_sync_request_total = 0

    # ------------------------------------------------------------------ setup

    def init_worker(self):
        """Build the policy (plus optional expert / RLT feature models)."""
        rollout_model_config = copy.deepcopy(self.model_cfg)
        with open_dict(rollout_model_config):
            rollout_model_config.precision = (
                self.rollout_cfg.rollout_model_cfg.precision
            )
            rollout_model_config.model_path = (
                self.rollout_cfg.rollout_model_cfg.model_path
            )

        self.hf_model: BasePolicy = get_model(rollout_model_config)

        ckpt_path = OmegaConf.select(self.cfg, "policy.ckpt_path", default=None)
        if ckpt_path:
            self.hf_model.load_state_dict(torch.load(ckpt_path))

        rlt_feature_model_config = OmegaConf.select(
            self.cfg, "rollout.rlt_feature_model", default=None
        )
        if rlt_feature_model_config is not None:
            from rlinf_rollout.plugins.rlt import build_rlt_route

            self.rlt_feature_model = get_model(copy.deepcopy(rlt_feature_model_config))
            self.rlt_feature_model.eval()
            self.rlt_feature_model.requires_grad_(False)
            self.rlt_route = build_rlt_route(self.cfg)

        if (
            OmegaConf.select(self.cfg, "rollout.expert_model", default=None)
            and not self.enable_opd
        ):
            from rlinf_rollout.plugins import build_expert_model_config

            expert_model_config = build_expert_model_config(
                self.cfg,
                self.model_cfg,
                rlt_feature_model_config=rlt_feature_model_config,
            )
            self.expert_model = get_model(expert_model_config)

            expert_ckpt_path = OmegaConf.select(
                self.cfg, "rollout.plugins.dagger.expert_ckpt_path", default=None
            )
            if expert_ckpt_path:
                self.expert_model.load_state_dict(torch.load(expert_ckpt_path))

        self.hf_model.eval()
        if self.expert_model is not None:
            self.expert_model.eval()
        if self.rlt_feature_model is not None:
            self.rlt_feature_model.eval()

        if bool(
            OmegaConf.select(self.cfg, "rollout.enable_torch_compile", default=False)
        ):
            self.hf_model.enable_torch_compile(
                mode=OmegaConf.select(
                    self.cfg,
                    "rollout.torch_compile_mode",
                    default="max-autotune-no-cudagraphs",
                )
            )
        if self.enable_cuda_graph and not self.enable_offload:
            self.hf_model.capture_cuda_graph(
                train_batch_size=self.per_node_train_batch_size,
                eval_batch_size=self.per_node_eval_batch_size,
            )

        self.setup_sample_params()
        if self._weight_syncer is not None:
            self.weight_receiver = CollectiveWeightReceiver(
                worker=self,
                syncer=self._weight_syncer,
                model=self.hf_model,
                receiver_group_name=self._group_name,
                receiver_ranks=self._weight_sync_rollout_ranks,
                is_handshake_sender=self._weight_sync_is_sender,
                receiver_id=f"{self._group_name}:{self._rank}",
            )
        if self.enable_offload:
            self.offload_model()

    def setup_sample_params(self):
        """Materialize train/eval sampling kwargs and the DAgger beta schedule."""
        sampling_params = OmegaConf.select(
            self.cfg, "rollout.sampling_params", default=None
        )
        if sampling_params is not None:
            sampling_params = OmegaConf.to_container(sampling_params, resolve=True)
            self._train_sampling_params = {
                "do_sample": sampling_params["do_sample"],
                "temperature": sampling_params["temperature_train"]
                if sampling_params["do_sample"]
                else 1.0,
                "top_k": sampling_params["top_k"],
                "top_p": sampling_params["top_p"],
                "max_new_tokens": sampling_params["max_new_tokens"],
            }
            self._eval_sampling_params = {
                "do_sample": sampling_params.get("temperature_eval", -1) > 0,
                "temperature": sampling_params["temperature_eval"],
                "top_k": sampling_params["top_k"],
                "top_p": sampling_params["top_p"],
                "max_new_tokens": sampling_params["max_new_tokens"],
            }
        else:
            self._train_sampling_params = {}
            self._eval_sampling_params = {}

        self._dagger_sampling_params: dict[str, Any] = {}
        if self.expert_model is not None and self.enable_dagger:
            dagger_cfg = OmegaConf.select(self.cfg, "rollout.plugins.dagger")
            self._dagger_sampling_params = {
                "beta": float(dagger_cfg.init_beta),
                "beta_schedule": str(dagger_cfg.beta_schedule),
                "beta_min": float(dagger_cfg.beta_min),
                "beta_decay": float(dagger_cfg.beta_decay),
            }

    # ------------------------------------------------------- decoupled routing

    async def recv_from_and_record_batch_routes_with_timeout(
        self,
        group_name: str,
        channel: Any | None,
        *,
        route_key: Any = None,
        tag: str | None = None,
        batch_size: int | None = None,
        merge_fn: Optional[Callable[[list[Any]], Any]] = None,
        infer_batch_size_fn: Optional[Callable[[Any], int]] = None,
        timeout_time: float = 0.02,
        recv_queue_size: int = 0,
    ):
        """Receive routed batch shards and record their return routes.

        Used in decoupled mode. Builds a receive plan for the source worker group,
        drains ``channel`` until the plan is satisfied or ``timeout_time`` elapses,
        then merges the shards. Each shard's ``batch_index`` is remembered in
        ``self.batch_router[tag]`` so :meth:`send_to_recorded_batch_routes` can
        answer the exact env rank that asked.

        Args:
            group_name: Source worker group name.
            channel: Channel used to receive routed batch shards.
            route_key: Optional key separating independent routed streams.
            tag: Routing tag used to build receive keys and index recorded routes.
            batch_size: Expected batch size of each planned receive entry.
            merge_fn: Custom merge function for received shards.
            infer_batch_size_fn: Function inferring a shard's batch size.
            timeout_time: Seconds to wait before finalizing partial results.
            recv_queue_size: Receive-queue depth used when building the plan.

        Returns:
            ``(merged_payload, split_sizes)``.
        """
        from rlinf_rollout.scheduler import (
            decoupled_build_recv_plan,
            get_batch_size,
            get_group_world_size,
            merge_batches,
        )

        world_size = get_group_world_size(self._manager_proxy, group_name)
        plan = decoupled_build_recv_plan(
            src_group_name=group_name,
            dst_group_name=self.worker_address.root_group_name,
            recv_rank=None,
            src_world_size=self._world_size,
            dst_world_size=world_size,
            tag=tag,
            route_key=route_key,
            batch_size=batch_size,
            recv_queue_size=recv_queue_size,
        )

        def _finalize(received_items: list[Any]):
            assert received_items, "received_items is empty"

            _, _, _, recv_tag = split_channel_message(received_items[0]["batch_index"])
            assert recv_tag in self.batch_router, (
                f"{recv_tag=} need to be already in the batch_router"
            )

            payloads = []
            for item in received_items:
                payloads.append(item["batch"])
                self.batch_router[recv_tag].append(item["batch_index"])

            split_sizes = [
                get_batch_size(item, infer_batch_size_fn) for item in payloads
            ]
            if merge_fn is not None:
                return merge_fn(payloads), split_sizes
            if len(payloads) == 1:
                return payloads[0], split_sizes
            return merge_batches(payloads), split_sizes

        deadline = timeout_time + time.time()
        get_items = None
        max_item_num = len(plan.entries)
        get_item_num = 0
        received_items: list[Any] = []
        while get_item_num < max_item_num:
            if get_items is None:
                get_items = channel.get(
                    key=plan.entries[get_item_num].key, async_op=True
                )
            else:
                await asyncio.sleep(0.0001)

            if get_items.done():
                received_items.append(await get_items.async_wait())
                get_items = None
                get_item_num += 1

            if time.time() >= deadline:
                max_item_num = get_item_num
                if get_items is not None:
                    received_items.append(await get_items.async_wait())
                    get_items = None
                    get_item_num += 1

        return _finalize(received_items)

    def send_to_recorded_batch_routes(
        self,
        group_name: str,
        channel: Any | None,
        data: Any,
        *,
        route_key: Any = None,
        tag: str | None = None,
        split_fn: Optional[Callable[[Any, list[int]], list[Any]]] = None,
        split_sizes: list[int],
    ):
        """Split ``data`` and send each shard back along a previously recorded route.

        Args:
            group_name: Destination worker group name.
            channel: Channel used to send the split payloads.
            data: Payload to split and send.
            route_key: Optional key separating independent routed streams.
            tag: Routing tag whose recorded batch indices are consumed.
            split_fn: Custom splitter; defaults to the scheduler's ``split_batch``.
            split_sizes: Per-shard batch sizes, same length as the recorded routes.

        Returns:
            ``AsyncRouteWork`` wrapping the async channel puts.
        """
        from rlinf_rollout.scheduler import build_send_key, split_batch
        from rlinf_rollout.scheduler.collective import AsyncRouteWork

        assert tag in self.batch_router, (
            f"{tag=} need to be already in the batch_router"
        )
        assert len(self.batch_router[tag]) > 0, f"{self.batch_router[tag]=} is empty"
        assert len(self.batch_router[tag]) == len(split_sizes), (
            f"{self.batch_router[tag]=} length should equal {split_sizes=} length"
        )

        payloads = (
            split_fn(data, split_sizes)
            if split_fn is not None
            else split_batch(data, split_sizes)
        )

        works = []
        for i, payload in enumerate(payloads):
            batch_index = self.batch_router[tag][i]
            send_rank, _, mode, _ = split_channel_message(batch_index)
            key = build_send_key(
                src_group_name=self.worker_address.root_group_name,
                dst_group_name=group_name,
                src_rank=None,
                dst_rank=send_rank,
                tag=tag if mode is None else f"{mode}_{tag}",
                route_key=route_key,
            )
            works.append(
                channel.put(
                    item={"batch_index": batch_index, "batch": payload},
                    key=key,
                    async_op=True,
                )
            )

        self.batch_router[tag] = []
        return AsyncRouteWork(works, lambda _: None)

    # ---------------------------------------------------------------- predict

    def update_dagger_beta(self):
        """Advance the DAgger expert-usage probability by one schedule step."""
        if self.expert_model is None or not self.enable_dagger:
            return

        schedule = self._dagger_sampling_params["beta_schedule"]
        if schedule != "exponential":
            raise NotImplementedError(f"Beta schedule {schedule} is not implemented")
        self._dagger_sampling_params["beta"] = max(
            self._dagger_sampling_params["beta_min"],
            self._dagger_sampling_params["beta"]
            * self._dagger_sampling_params["beta_decay"],
        )

    @Worker.timer("predict")
    def predict(
        self, env_obs: dict[str, Any], mode: Literal["train", "eval"] = "train"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run one policy forward pass, optionally routed through the expert."""
        kwargs = (
            self._train_sampling_params
            if mode == "train"
            else self._eval_sampling_params
        )

        model_type = str(self.model_cfg.model_type)
        if model_type in _MODE_ONLY_MODELS:
            kwargs = {"mode": "eval" if self.enable_dagger else mode}
        if model_type in _RETURN_OBS_MODELS:
            kwargs["return_obs"] = not hasattr(self.hf_model, "q_head")

        only_save_expert = bool(
            OmegaConf.select(
                self.cfg, "rollout.plugins.dagger.only_save_expert", default=True
            )
        )

        use_expert = (
            mode == "train"
            and self.expert_model is not None
            and self.enable_dagger
            and torch.rand(1).item() < self._dagger_sampling_params["beta"]
        )

        with torch.no_grad():
            expert_label_flag = False
            if use_expert:
                actions, result = self.expert_model.predict_action_batch(
                    env_obs=env_obs, **kwargs
                )
                expert_label_flag = True
            else:
                actions, result = self.hf_model.predict_action_batch(
                    env_obs=env_obs, **kwargs
                )

            if (
                not only_save_expert
                and not use_expert
                and self.expert_model is not None
                and self.enable_dagger
                and mode == "train"
            ):
                # Classic DAgger: keep acting with the student but relabel with the
                # expert so the trainer learns the expert's action.
                _, expert_result = self.expert_model.predict_action_batch(
                    env_obs=env_obs, **kwargs
                )
                expert_forward_inputs = expert_result["forward_inputs"]
                expert_target = expert_forward_inputs["model_action"]
                if expert_target is not None:
                    result["forward_inputs"]["action"] = expert_forward_inputs["action"]
                    result["forward_inputs"]["model_action"] = expert_target
                expert_label_flag = True

        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)

        result["expert_label_flag"] = bool(expert_label_flag)
        return actions, result

    def _predict_rollout_actions(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
        final_obs: dict[str, Any] | None = None,
        rlt_switch_flags: torch.Tensor | None = None,
        intervene_requested: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.rlt_feature_model is not None:
            from rlinf_rollout.plugins.rlt import predict_rlt_actions

            return predict_rlt_actions(
                policy_model=self.hf_model,
                feature_model=self.rlt_feature_model,
                rlt_route=self.rlt_route,
                env_obs=env_obs,
                final_obs=final_obs,
                mode=mode,
                version=self.version,
                rlt_switch_flags=rlt_switch_flags,
                intervene_requested=intervene_requested,
                expert_model=self.expert_model,
            )
        return self.predict(env_obs, mode=mode)

    def get_bootstrap_values(
        self, final_obs: dict[str, Any] | None
    ) -> torch.Tensor | None:
        """Estimate the value of a chunk's terminal observation, if available."""
        return estimate_bootstrap_values(
            self.hf_model, self._predict_rollout_actions, final_obs
        )

    def _build_rollout_result(
        self,
        actions: torch.Tensor,
        result: dict[str, Any],
        *,
        final_obs: dict[str, Any] | None = None,
    ) -> RolloutResult:
        intervene_flags = result.get("intervene_flags")
        if intervene_flags is None and result.get("expert_label_flag", False):
            intervene_flags = torch.full(
                (actions.shape[0], self.rollout_cfg.num_action_chunks),
                True,
                dtype=torch.bool,
                device=actions.device,
            )
        return RolloutResult(
            actions=actions,
            prev_logprobs=result["prev_logprobs"] if self.collect_prev_infos else None,
            prev_values=result["prev_values"] if self.collect_prev_infos else None,
            bootstrap_values=self.get_bootstrap_values(final_obs),
            intervene_flags=intervene_flags,
            forward_inputs=result["forward_inputs"],
            versions=torch.full_like(
                result["prev_logprobs"],
                float(self.version),
                dtype=torch.float32,
            ),
        )

    # ----------------------------------------------------------- weight sync

    def build_weight_update_request(self) -> WeightUpdateRequest:
        """Describe the update the trainer is expected to broadcast next.

        The collective transport is receiver-driven, so ``version`` is only the
        minimum expected version; the ack reports the version actually applied.
        """
        assert self._weight_source is not None, (
            "weight sync is disabled (rollout.mode=eval)."
        )
        return WeightUpdateRequest(
            version=max(self.version, 0),
            mode=self._weight_sync_mode,
            transport=WeightTransport.COLLECTIVE,
            source=self._weight_source,
            blocking=not self._background_weight_sync_active,
        )

    @Worker.timer("sync_weights")
    async def sync_weights(self) -> int:
        """Apply one pushed weight update and return the version now served."""
        ack = await self.receive_weight_update(self.build_weight_update_request())
        if ack.status is WeightUpdateStatus.FAILED:
            raise RuntimeError(f"Weight update failed: {ack.error}")
        return self.version

    @Worker.timer("receive_weight_update")
    async def receive_weight_update(
        self, request: WeightUpdateRequest
    ) -> WeightUpdateAck:
        """Apply one client-described weight update, dispatching on its transport.

        This is the entry point the service controller calls for a client-initiated
        push (see :mod:`rlinf_rollout.serve.controller`); :meth:`sync_weights` is
        the shorthand for "apply whatever the configured sender broadcasts next".

        Args:
            request: Update description. ``COLLECTIVE`` is served by the
                configured :class:`~rlinf_rollout.weight_sync.CollectiveWeightReceiver`
                (receiver-driven: the ack's ``served_version`` is authoritative);
                ``CHECKPOINT`` loads a state dict from a shared path.

        Returns:
            The acknowledgement, with ``FAILED`` (rather than an exception) for an
            unsupported transport or a receiver-side error.
        """
        if request.transport is WeightTransport.CHECKPOINT:
            receiver = self._checkpoint_weight_receiver()
        elif request.transport is WeightTransport.COLLECTIVE:
            if self.weight_receiver is None:
                return WeightUpdateAck(
                    version=request.version,
                    status=WeightUpdateStatus.FAILED,
                    served_version=self.version,
                    receiver_id=f"{self._group_name}:{self._rank}",
                    error=(
                        "collective weight sync is disabled on this worker "
                        "(rollout.mode=eval, or init_worker() has not run)."
                    ),
                )
            receiver = self.weight_receiver
        else:
            return WeightUpdateAck(
                version=request.version,
                status=WeightUpdateStatus.FAILED,
                served_version=self.version,
                receiver_id=f"{self._group_name}:{self._rank}",
                error=(
                    f"transport {request.transport.value!r} is not implemented by "
                    f"the HuggingFace rollout worker."
                ),
            )

        ack = await receiver.recv(request)
        if ack.status is WeightUpdateStatus.FAILED:
            return ack

        self.version = ack.served_version
        if self.finished_episodes is None:
            self.finished_episodes = (
                self.version * self.total_num_train_envs * self.rollout_epoch
            )
        return ack

    def _checkpoint_weight_receiver(self) -> "CheckpointWeightReceiver":
        """Return (and lazily build) the checkpoint-transport receiver."""
        if self._checkpoint_receiver is None:
            from rlinf_rollout.weight_sync import CheckpointWeightReceiver

            self._checkpoint_receiver = CheckpointWeightReceiver(
                model=self.hf_model,
                receiver_id=f"{self._group_name}:{self._rank}",
                strict=False,
            )
        return self._checkpoint_receiver

    def served_weight_version(self) -> int:
        """Weight version this worker currently serves (``0`` before any update)."""
        return self.version

    async def wait_if_stale(self) -> None:
        """Throttle collection when the trainer falls too far behind."""
        if self.staleness_threshold is None:
            return
        assert self.finished_episodes is not None, (
            "finished_episodes should be initialized by the first weight sync."
        )
        episodes_per_round = self.total_num_train_envs * self.rollout_epoch
        while True:
            capacity = (
                self.staleness_threshold + self.version + 1
            ) * episodes_per_round
            if self.finished_episodes + episodes_per_round <= capacity:
                break
            await asyncio.sleep(0.01)

    def _start_background_weight_sync_if_needed(self):
        if (
            not self._background_weight_sync_active
            or not self._weight_sync_requested
            or self._weight_sync_work is not None
        ):
            return
        self._weight_sync_requested = False
        self._weight_sync_work = asyncio.create_task(self.sync_weights())

    @Worker.timer("rollout/poll_weight_sync")
    async def _poll_background_weight_sync(self):
        self._start_background_weight_sync_if_needed()
        if self._weight_sync_work is None or not self._weight_sync_work.done():
            return

        await self._weight_sync_work
        self._weight_sync_work = None
        self._weight_sync_apply_total += 1
        self._start_background_weight_sync_if_needed()

    @Worker.timer("rollout/request_weight_sync")
    async def request_weight_sync(self):
        """Signal that a new weight version is available upstream."""
        self._weight_sync_request_total += 1
        if self._weight_sync_requested or self._weight_sync_work is not None:
            self._weight_sync_coalesced_total += 1
        self._weight_sync_requested = True
        self._start_background_weight_sync_if_needed()

    # -------------------------------------------------------------- generate

    @Worker.timer("rollout/generate")
    async def generate(
        self,
        input_channel: Channel,
        output_channel: Channel,
        metric_channel: Channel,
    ):
        """Resident generation loop; returns only when :meth:`stop` cancels it."""
        assert self._generate_task is None or self._generate_task.done(), (
            "generate task is still running while a new generate call is made."
        )
        self._generate_task = asyncio.create_task(
            self._generate(input_channel, output_channel, metric_channel)
        )
        try:
            await self._generate_task
        except asyncio.CancelledError:
            pass

    async def _generate(
        self,
        input_channel: Channel,
        output_channel: Channel,
        metric_channel: Channel,
    ):
        if self.env_decoupled_mode:
            await self.decoupled_generate(input_channel, output_channel)
            return

        while True:
            if self._background_weight_sync_active:
                await self._poll_background_weight_sync()

            for _ in range(self.rollout_epoch):
                await self.generate_one_epoch(input_channel, output_channel)
            if self.finished_episodes is not None:
                self.finished_episodes += self.total_num_train_envs * self.rollout_epoch
            rollout_metrics = {
                f"time/rollout/{k}": v for k, v in self.pop_execution_times().items()
            }
            metric_channel.put(
                {"rank": self._rank, "time": rollout_metrics}, async_op=True
            )

    @Worker.timer("generate_one_epoch")
    async def generate_one_epoch(self, input_channel: Channel, output_channel: Channel):
        """Serve one rollout epoch of action chunks to the env workers."""
        self.update_dagger_beta()
        for _ in range(self.n_train_chunk_steps):
            for stage_id in range(self.num_pipeline_stages):
                await self._serve_one_chunk_step(
                    input_channel, output_channel, stage_id, terminal=False
                )
        # One extra round serves the epoch's terminal observation. The env worker
        # records it but does not step the env, so logprobs/versions are omitted.
        for stage_id in range(self.num_pipeline_stages):
            await self._serve_one_chunk_step(
                input_channel, output_channel, stage_id, terminal=True
            )

    async def _serve_one_chunk_step(
        self,
        input_channel: Channel,
        output_channel: Channel,
        stage_id: int,
        *,
        terminal: bool,
    ) -> None:
        env_output = await self.recv_from(
            group_name=self.rollout_cfg.env_group_name,
            channel=input_channel,
            tag="train_rollout_results",
            route_key=stage_id,
            async_op=True,
            batch_size=self.train_batch_size,
            merge_fn=self._merge_obs_batches,
            infer_batch_size_fn=self._infer_env_batch_size,
        ).async_wait()
        final_obs = env_output.get("final_obs", None)
        actions, result = self._predict_rollout_actions(
            env_output["obs"],
            final_obs=final_obs,
            rlt_switch_flags=env_output.get("rlt_switch_flags", None),
            intervene_requested=env_output.get("intervene_flags", None),
        )
        if terminal and not self.enable_opd:
            # OPD keeps the full result on the terminal step to retain student
            # action tokens for post-rollout teacher logprobs.
            rollout_result = RolloutResult(
                actions=actions,
                prev_values=(
                    result["prev_values"] if self.collect_prev_infos else None
                ),
                bootstrap_values=self.get_bootstrap_values(final_obs),
                forward_inputs=(
                    result["forward_inputs"]
                    if self.rlt_feature_model is not None
                    else {}
                ),
            )
        else:
            rollout_result = self._build_rollout_result(
                actions, result, final_obs=final_obs
            )
        self.send_to(
            group_name=self.rollout_cfg.env_group_name,
            channel=output_channel,
            data=rollout_result,
            tag="train_rollout_results",
            route_key=stage_id,
            async_op=True,
            batch_size=self.train_batch_size,
            split_fn=self._split_rollout_result,
        )

    async def decoupled_generate(self, input_channel: Channel, output_channel: Channel):
        """Serve action chunks without a fixed env-rank pairing."""
        self.update_dagger_beta()
        served = 1
        while True:
            if served % self.sync_rollout_weight_time == 0:
                self.update_dagger_beta()
                if self._background_weight_sync_active:
                    await self._poll_background_weight_sync()
                await self.wait_if_stale()
            served += 1

            (
                env_output,
                split_sizes,
            ) = await self.recv_from_and_record_batch_routes_with_timeout(
                group_name=self.rollout_cfg.env_group_name,
                channel=input_channel,
                tag="rollout_results",
                batch_size=self.train_batch_size,
                merge_fn=self._merge_obs_batches,
                infer_batch_size_fn=self._infer_env_batch_size,
                timeout_time=0.02,
                recv_queue_size=self.rollout_queue_size,
            )
            actions, result = self._predict_rollout_actions(
                env_output["obs"],
                final_obs=env_output.get("final_obs", None),
                rlt_switch_flags=env_output.get("rlt_switch_flags", None),
                intervene_requested=env_output.get("intervene_flags", None),
            )
            rollout_result = self._build_rollout_result(
                actions, result, final_obs=env_output.get("final_obs", None)
            )
            self.send_to_recorded_batch_routes(
                group_name=self.rollout_cfg.env_group_name,
                channel=output_channel,
                data=rollout_result,
                tag="rollout_results",
                split_fn=self._split_rollout_result,
                split_sizes=split_sizes,
            )

    @Worker.timer("evaluate")
    async def evaluate(self, input_channel: Channel, output_channel: Channel):
        """Serve eval action chunks with the current weights."""
        if self.enable_offload:
            self.reload_model()
        try:
            if self.env_decoupled_mode:
                await self._decoupled_evaluate(input_channel, output_channel)
            else:
                await self._paired_evaluate(input_channel, output_channel)
        finally:
            if self.enable_offload:
                self.offload_model()

    async def _decoupled_evaluate(
        self, input_channel: Channel, output_channel: Channel
    ):
        while True:
            (
                env_output,
                split_sizes,
            ) = await self.recv_from_and_record_batch_routes_with_timeout(
                group_name=self.rollout_cfg.env_group_name,
                channel=input_channel,
                tag="rollout_results",
                batch_size=self.eval_batch_size,
                merge_fn=self._merge_obs_batches,
                infer_batch_size_fn=self._infer_env_batch_size,
                timeout_time=0.02,
                recv_queue_size=self.rollout_queue_size,
            )
            actions = self._predict_eval_actions(env_output)
            self.send_to_recorded_batch_routes(
                group_name=self.rollout_cfg.env_group_name,
                channel=output_channel,
                data=actions,
                tag="rollout_results",
                split_sizes=split_sizes,
            )

    async def _paired_evaluate(self, input_channel: Channel, output_channel: Channel):
        for _ in range(self.eval_rollout_epoch):
            for _ in range(self.n_eval_chunk_steps):
                for stage_id in range(self.num_pipeline_stages):
                    env_output = await self.recv_from(
                        group_name=self.rollout_cfg.env_group_name,
                        channel=input_channel,
                        tag="eval_rollout_results",
                        route_key=stage_id,
                        async_op=True,
                        batch_size=self.eval_batch_size,
                        merge_fn=self._merge_obs_batches,
                        infer_batch_size_fn=self._infer_env_batch_size,
                    ).async_wait()
                    actions = self._predict_eval_actions(env_output)
                    self.send_to(
                        group_name=self.rollout_cfg.env_group_name,
                        channel=output_channel,
                        data=actions,
                        tag="eval_rollout_results",
                        route_key=stage_id,
                        async_op=True,
                        batch_size=self.eval_batch_size,
                    )

    def _predict_eval_actions(self, env_output: dict[str, Any]) -> torch.Tensor:
        actions, _ = self._predict_rollout_actions(
            env_output["obs"],
            mode="eval",
            final_obs=env_output.get("final_obs", None),
            rlt_switch_flags=env_output.get("rlt_switch_flags", None),
            intervene_requested=env_output.get("intervene_flags", None),
        )
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().contiguous()
        return actions

    def stop(self):
        """Cancel the resident generation loop."""
        self.should_stop = True
        if self._generate_task is not None and not self._generate_task.done():
            self._generate_task.cancel()

    # ------------------------------------------------------------- offloading

    def offload_model(self):
        """Move policy (and helpers) to host memory and release device caches."""
        if self.enable_cuda_graph:
            self.hf_model.release_cuda_graph()
        self.hf_model.to("cpu")
        for model in (self.rlt_feature_model, self.expert_model):
            if model is not None:
                model.to("cpu")
        self.torch_platform.empty_cache()

    def reload_model(self):
        """Move policy (and helpers) back to the accelerator."""
        self.hf_model.to(self.device)
        for model in (self.rlt_feature_model, self.expert_model):
            if model is not None:
                model.to(self.device)
        if self.enable_cuda_graph:
            self.hf_model.capture_cuda_graph(
                train_batch_size=self.per_node_train_batch_size,
                eval_batch_size=self.per_node_eval_batch_size,
            )

    # ------------------------------------------------------- batch plumbing

    @staticmethod
    def _infer_env_batch_size(obs_batch: dict[str, Any]) -> int:
        obs = obs_batch["obs"] if "obs" in obs_batch else obs_batch
        for key in ("states", "main_images", "task_descriptions"):
            value = obs.get(key)
            if isinstance(value, torch.Tensor):
                return value.shape[0]
            if isinstance(value, list):
                return len(value)
        raise ValueError("Cannot infer batch size from env obs.")

    def _merge_optional_flag_tensors(
        self,
        obs_dicts: list[dict[str, Any]],
        flags_list: list[torch.Tensor | None],
    ) -> torch.Tensor | None:
        if not any(flags is not None for flags in flags_list):
            return None
        ref_flags = next(flags for flags in flags_list if flags is not None)
        filled_flags = []
        for obs_dict, flags in zip(obs_dicts, flags_list):
            if flags is None:
                batch_size = self._infer_env_batch_size(obs_dict)
                filled_flags.append(
                    torch.zeros(
                        (batch_size, *ref_flags.shape[1:]), dtype=ref_flags.dtype
                    )
                )
            else:
                filled_flags.append(flags)
        return torch.cat(filled_flags, dim=0)

    def _merge_obs_batches(self, obs_batches: list[dict[str, Any]]) -> dict[str, Any]:
        if not obs_batches:
            return {}
        obs_dicts = [
            obs_batch["obs"] if "obs" in obs_batch else obs_batch
            for obs_batch in obs_batches
        ]
        final_obs_list = [obs_batch.get("final_obs", None) for obs_batch in obs_batches]
        rlt_switch_flags_list = [
            obs_batch.get("rlt_switch_flags", None) for obs_batch in obs_batches
        ]
        intervene_flags_list = [
            obs_batch.get("intervene_flags", None) for obs_batch in obs_batches
        ]

        def _merge_obs_dicts(dicts: list[dict[str, Any]]) -> dict[str, Any]:
            merged: dict[str, Any] = {}
            for key in dicts[0].keys():
                values = [obs_dict[key] for obs_dict in dicts]
                first_non_none = next(
                    (value for value in values if value is not None), None
                )
                if first_non_none is None:
                    merged[key] = None
                elif isinstance(first_non_none, torch.Tensor):
                    merged[key] = torch.cat(values, dim=0)
                elif isinstance(first_non_none, list):
                    merged[key] = [item for sublist in values for item in sublist]
                else:
                    merged[key] = values
            return merged

        merged_final_obs = None
        if any(final_obs is not None for final_obs in final_obs_list):
            merged_final_obs = _merge_obs_dicts(
                [
                    final_obs if final_obs is not None else obs_dict
                    for obs_dict, final_obs in zip(obs_dicts, final_obs_list)
                ]
            )

        return {
            "obs": _merge_obs_dicts(obs_dicts),
            "final_obs": merged_final_obs,
            "rlt_switch_flags": self._merge_optional_flag_tensors(
                obs_dicts, rlt_switch_flags_list
            ),
            "intervene_flags": self._merge_optional_flag_tensors(
                obs_dicts, intervene_flags_list
            ),
        }

    def _split_rollout_result(
        self, rollout_result: RolloutResult, sizes: list[int]
    ) -> list[RolloutResult]:
        def _split_optional_tensor(
            tensor: torch.Tensor | None,
        ) -> tuple[torch.Tensor | None, ...]:
            if tensor is None:
                return tuple(None for _ in sizes)
            return tuple(torch.split(tensor, sizes, dim=0))

        split_actions = _split_optional_tensor(rollout_result.actions)
        split_prev_logprobs = _split_optional_tensor(rollout_result.prev_logprobs)
        split_prev_values = _split_optional_tensor(rollout_result.prev_values)
        split_bootstrap_values = _split_optional_tensor(rollout_result.bootstrap_values)
        split_intervene_flags = _split_optional_tensor(rollout_result.intervene_flags)
        split_versions = _split_optional_tensor(rollout_result.versions)
        split_forward_inputs = (
            [{} for _ in sizes]
            if not rollout_result.forward_inputs
            else [
                {
                    key: torch.split(value, sizes, dim=0)[idx]
                    for key, value in rollout_result.forward_inputs.items()
                    if value is not None
                }
                for idx in range(len(sizes))
            ]
        )

        return [
            RolloutResult(
                actions=split_actions[idx],
                prev_logprobs=split_prev_logprobs[idx],
                prev_values=split_prev_values[idx],
                bootstrap_values=split_bootstrap_values[idx],
                intervene_flags=split_intervene_flags[idx],
                forward_inputs=split_forward_inputs[idx],
                versions=split_versions[idx],
            )
            for idx in range(len(sizes))
        ]

    def set_global_step(self, global_step: int):
        """Forward the trainer's global step to the policy, when it tracks one."""
        if hasattr(self.hf_model, "set_global_step"):
            self.hf_model.set_global_step(global_step)
