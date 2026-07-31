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

"""Weight synchronization: payload codecs plus the rollout-side receiver.

Exports are resolved lazily (PEP 562). The syncer codecs pull in
``torch.distributed.tensor.DTensor``, which only exists on newer torch builds;
the transport-level :class:`~rlinf_rollout.weight_sync.CollectiveWeightReceiver`
must stay importable without them.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Re-exported lazily at runtime through ``__getattr__``; listed here so type
    # checkers and IDEs still resolve ``rlinf_rollout.weight_sync.X``.
    from .base import WeightSyncer  # noqa: F401
    from .bucket_syncer import BucketWeightSyncer  # noqa: F401
    from .compressor import (  # noqa: F401
        IdentityCompressor,
        NVCompCompressor,
        PatchCompressor,
    )
    from .patch_syncer import (  # noqa: F401
        CompressedWeightPatch,
        PatchWeightSyncer,
        WeightPatch,
        WeightPatchTransport,
    )
    from .receiver import CollectiveWeightReceiver  # noqa: F401

_EXPORTS: dict[str, str] = {
    "WeightSyncer": ".base",
    "BucketWeightSyncer": ".bucket_syncer",
    "IdentityCompressor": ".compressor",
    "NVCompCompressor": ".compressor",
    "PatchCompressor": ".compressor",
    "CompressedWeightPatch": ".patch_syncer",
    "PatchWeightSyncer": ".patch_syncer",
    "WeightPatch": ".patch_syncer",
    "WeightPatchTransport": ".patch_syncer",
    "CollectiveWeightReceiver": ".receiver",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Import the submodule owning ``name`` on first access."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_name, __name__), name)


def __dir__() -> list[str]:
    return __all__
