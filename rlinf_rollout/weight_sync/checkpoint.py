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

"""Checkpoint :class:`~rlinf_rollout.api.v1.WeightReceiver` backend.

The collective backend (:mod:`rlinf_rollout.weight_sync.receiver`) needs the
trainer to be alive, co-scheduled and holding a matching collective group. That
is the fast path, but it is also the tightest coupling a client can have. The
checkpoint transport is the loose one: the trainer writes a state dict somewhere
both sides can read and pushes only a path plus a version.

Unlike the collective transport this one is *sender-driven and versioned by the
sender*: :attr:`WeightUpdateRequest.version` is authoritative, and a request that
is not newer than what is already served is answered with
:attr:`~rlinf_rollout.api.v1.WeightUpdateStatus.SKIPPED` instead of reloading.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any, Callable, Optional

import torch

from rlinf_rollout.api.v1 import (
    WeightReceiver,
    WeightTransport,
    WeightUpdateAck,
    WeightUpdateRequest,
    WeightUpdateStatus,
)

__all__ = ["CheckpointWeightReceiver"]

#: Keys a checkpoint may nest its state dict under.
_STATE_DICT_KEYS = ("state_dict", "model", "model_state_dict", "module")


class CheckpointWeightReceiver(WeightReceiver):
    """Loads new policy weights from a filesystem path.

    Args:
        model: Live policy the weights are written into.
        receiver_id: Human-readable identity reported in the ack.
        map_location: ``torch.load`` device mapping; the default keeps the load
            on host memory and lets ``load_state_dict`` move tensors.
        strict: Whether ``load_state_dict`` requires an exact key match.
        loader: Optional ``path -> state_dict`` override, for checkpoint formats
            ``torch.load`` cannot read (sharded safetensors, custom bundles).
        empty_cache: Whether to drop cached accelerator memory after a load.
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        receiver_id: str = "",
        map_location: str = "cpu",
        strict: bool = True,
        loader: Optional[Callable[[Path], dict[str, Any]]] = None,
        empty_cache: bool = False,
    ) -> None:
        self._model = model
        self._receiver_id = receiver_id or type(model).__name__
        self._map_location = map_location
        self._strict = strict
        self._loader = loader
        self._empty_cache = empty_cache
        self._served_version = -1
        self._last_path: Optional[str] = None

    @property
    def served_version(self) -> int:
        """Weight version currently served, or ``-1`` before the first update."""
        return self._served_version

    @property
    def last_checkpoint_path(self) -> Optional[str]:
        """Path of the most recently applied checkpoint, if any."""
        return self._last_path

    def _load_state_dict(self, path: Path) -> dict[str, Any]:
        if self._loader is not None:
            return self._loader(path)
        payload = torch.load(path, map_location=self._map_location, weights_only=False)
        if isinstance(payload, dict):
            for key in _STATE_DICT_KEYS:
                nested = payload.get(key)
                if isinstance(nested, dict):
                    return nested
            return payload
        raise TypeError(
            f"checkpoint {path} decoded to {type(payload).__name__}; expected a dict "
            "or a nested state dict. Pass a custom 'loader' for this format."
        )

    async def recv(self, request: WeightUpdateRequest) -> WeightUpdateAck:
        """Load ``request.checkpoint_path`` into the live model.

        Args:
            request: Update description; must use
                :attr:`~rlinf_rollout.api.v1.WeightTransport.CHECKPOINT`.

        Returns:
            The acknowledgement. ``SKIPPED`` when ``request.version`` is not newer
            than the served one, ``FAILED`` when the checkpoint cannot be loaded.
        """
        started = time.perf_counter()
        if request.transport is not WeightTransport.CHECKPOINT:
            return WeightUpdateAck(
                version=request.version,
                status=WeightUpdateStatus.FAILED,
                served_version=self._served_version,
                receiver_id=self._receiver_id,
                elapsed_seconds=time.perf_counter() - started,
                error=(
                    f"{type(self).__name__} only supports "
                    f"{WeightTransport.CHECKPOINT.value} transport, got "
                    f"{request.transport.value}."
                ),
            )
        if request.version <= self._served_version:
            return WeightUpdateAck(
                version=request.version,
                status=WeightUpdateStatus.SKIPPED,
                served_version=self._served_version,
                receiver_id=self._receiver_id,
                elapsed_seconds=time.perf_counter() - started,
            )

        try:
            path = Path(str(request.checkpoint_path))
            if not path.exists():
                raise FileNotFoundError(f"checkpoint path does not exist: {path}")
            state_dict = self._load_state_dict(path)
            self._model.load_state_dict(state_dict, strict=self._strict)
        except Exception as exc:  # noqa: BLE001 - reported through the ack
            return WeightUpdateAck(
                version=request.version,
                status=WeightUpdateStatus.FAILED,
                served_version=self._served_version,
                receiver_id=self._receiver_id,
                elapsed_seconds=time.perf_counter() - started,
                error=f"{type(exc).__name__}: {exc}",
            )

        num_tensors = sum(
            1 for value in state_dict.values() if isinstance(value, torch.Tensor)
        )
        num_bytes = sum(
            value.numel() * value.element_size()
            for value in state_dict.values()
            if isinstance(value, torch.Tensor)
        )
        del state_dict
        self._served_version = int(request.version)
        self._last_path = str(request.checkpoint_path)
        if hasattr(self._model, "set_global_step"):
            self._model.set_global_step(self._served_version)
        gc.collect()
        if self._empty_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return WeightUpdateAck(
            version=request.version,
            status=WeightUpdateStatus.APPLIED,
            served_version=self._served_version,
            receiver_id=self._receiver_id,
            num_tensors_applied=num_tensors,
            num_bytes_received=num_bytes,
            elapsed_seconds=time.perf_counter() - started,
        )

    def describe(self) -> dict[str, Any]:
        """Return backend diagnostics."""
        info = super().describe()
        info.update(
            {
                "transport": WeightTransport.CHECKPOINT.value,
                "receiver_id": self._receiver_id,
                "strict": self._strict,
                "last_checkpoint_path": self._last_path,
            }
        )
        return info
