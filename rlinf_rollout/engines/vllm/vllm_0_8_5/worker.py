# Copyright 2025 The RLinf Authors.
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

from typing import Any, Optional

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu_worker import Worker as _VllmInnerWorker

from rlinf_rollout.scheduler import Worker as _RLinfWorker
from rlinf_rollout.scheduler import WorkerAddress
from rlinf_rollout.weight_sync.llm import EngineWeightSyncSetup

from . import weight_loader  # noqa all

logger = init_logger(__name__)


class VLLMWorker(_VllmInnerWorker):
    """vLLM GPU worker that can receive weights from an external sender.

    Args:
        weight_sync_setup: Sender topology + rollout parallel sizes, or ``None`` for a
            fixed-weight (eval) engine. Replaces the training repo's
            ``(rlinf_config, placement)`` pair, which read ``cfg.actor.*``.
        enforce_eager: Mirror of ``rollout.enforce_eager``; weight sync asserts it is
            ``False``.
        parent_address: Address of the owning rollout ``Worker``.
        vllm_config: Upstream vLLM config.
        distributed_init_method: Upstream vLLM arg.
        local_rank: Upstream vLLM arg.
        rank: Upstream vLLM arg.
        is_driver_worker: Upstream vLLM arg.
    """

    def __init__(
        self,
        # rollout-system specific args
        weight_sync_setup: Optional[EngineWeightSyncSetup],
        parent_address: WorkerAddress,
        enforce_eager: bool = False,
        # vllm former args
        vllm_config: Optional[VllmConfig] = None,
        distributed_init_method: Optional[str] = None,
        local_rank: int = 0,
        rank: int = 0,
        is_driver_worker: bool = False,
    ):
        super().__init__(
            vllm_config, local_rank, rank, distributed_init_method, is_driver_worker
        )
        self._weight_setup = weight_sync_setup
        self._enforce_eager = bool(enforce_eager)
        self.using_sharded_weight = (
            weight_sync_setup.presharded_weights
            if weight_sync_setup is not None
            else False
        )
        self._rlinf_worker = _RLinfWorker(
            parent_address=parent_address,
            world_size=vllm_config.parallel_config.world_size,
            rank=rank,
        )
        if weight_sync_setup is not None:
            self._weight_source_group_name = weight_sync_setup.source_group_name
            self._weight_source_rank = weight_sync_setup.source_rank_for(
                self._rlinf_worker.get_parent_rank(), self.rank
            )
        else:
            self._weight_source_group_name = None
            self._weight_source_rank = None
        self.is_weight_offloaded = False

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config."""
        if isinstance(kv_cache_config, list):
            assert (
                len(kv_cache_config) == self.vllm_config.parallel_config.world_size
            ), (
                f"Got {len(kv_cache_config)} KVCacheConfig, expected {self.vllm_config.parallel_config.world_size}"
            )
            kv_cache_config = kv_cache_config[self.rank]
        super().initialize_from_config(kv_cache_config)

    def offload_model_weights(self) -> None:
        super().sleep(level=2)
        self.is_weight_offloaded = True

    def batch_load_hf_weight(self, state_dict: dict[str, Any]) -> Any:
        model = self.model_runner.model
        batch_weight = []
        if self._weight_setup is not None and self._weight_setup.is_collocated:
            for name, handle in state_dict.items():
                func, args = handle
                list_args = list(args)
                # NOTE: the key is to change device id to the current device id
                # in case two processes have different CUDA_VISIBLE_DEVICES
                list_args[6] = torch.cuda.current_device()
                new_weight = func(*list_args)
                batch_weight.append((name, new_weight))
            model.load_weights(batch_weight)
        else:
            # disaggregate mode, recv tensor directly
            model.load_weights(state_dict.items())

        for name, weight in batch_weight:
            del weight
        batch_weight.clear()

    def sync_hf_weight(self) -> None:
        assert self._weight_setup is not None, (
            "sync_hf_weight requires a weight_sync_setup; this engine was built "
            "fixed-weight (eval)."
        )
        assert not self._enforce_eager, (
            "vLLM weight sync requires cuda graph capture (enforce_eager=False)."
        )

        state_dict = self._rlinf_worker.recv(
            src_group_name=self._weight_source_group_name,
            src_rank=self._weight_source_rank,
        )

        bucket_length = state_dict.get("bucket_length", None)

        if bucket_length is None:
            # recv from the fsdp backend
            # fsdp just send a bucket and don't have the key bucket_length
            bucket_length = 1
        else:
            # recv from the Megatron backend
            # Megatron use weight bucket to sync weight, the bucket length in dict of bucket 0, bucket_length
            state_dict.pop("bucket_length")

        if self.is_weight_offloaded:
            super().wake_up()
            self.is_weight_offloaded = False

        assert bucket_length > 0, f"bucket_length {bucket_length} is invalid"

        self.batch_load_hf_weight(state_dict)
        if bucket_length > 1:
            recv_handle = self._rlinf_worker.recv(
                src_group_name=self._weight_source_group_name,
                src_rank=self._weight_source_rank,
                async_op=True,
            )

            for _ in range(bucket_length - 2):
                next_recv_handle = self._rlinf_worker.recv(
                    src_group_name=self._weight_source_group_name,
                    src_rank=self._weight_source_rank,
                    async_op=True,
                )
                state_dict = recv_handle.wait()
                self.batch_load_hf_weight(state_dict)
                recv_handle = next_recv_handle

            state_dict = recv_handle.wait()
            self.batch_load_hf_weight(state_dict)

        super().compile_or_warm_up_model()

    def use_sharded_weights(self) -> None:
        model = self.model_runner.model
        for _, param in model.named_parameters():
            setattr(param, "is_sharded_weight", self.using_sharded_weight)

    def get_dp_rank(self) -> int:
        return self._rlinf_worker.get_parent_rank()
