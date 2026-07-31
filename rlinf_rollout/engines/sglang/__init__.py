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

"""Version-gated entry point for the patched SGLang engine.

Vendored from the training repo's ``rlinf/workers/rollout/sglang/__init__.py``;
the dispatch lives next to the engine code here so the worker layer stays free of
engine-version logic.
"""

from importlib.metadata import PackageNotFoundError, version

from packaging.version import parse

__all__ = ["Engine", "SGLANG_VERSION", "io_struct"]

MIN_SGLANG_VERSION = "0.4.4"
MAX_SGLANG_VERSION = "0.5.4"


def _installed_version(pkg: str):
    try:
        return parse(version(pkg))
    except PackageNotFoundError:
        return None


_package_version = _installed_version("sglang")

if _package_version is None:
    raise ValueError(
        "sglang is not installed; install rlinf-rollout with the [sglang] extra."
    )
elif _package_version >= parse(MIN_SGLANG_VERSION) and _package_version <= parse(
    MAX_SGLANG_VERSION
):
    #: Version of the SGLang the wrappers were dispatched for.
    SGLANG_VERSION = _package_version
    from rlinf_rollout.engines.sglang.common import io_struct
    from rlinf_rollout.engines.sglang.common.sgl_engine import Engine
else:
    raise ValueError(
        f"sglang version {_package_version} not supported "
        f"(need >={MIN_SGLANG_VERSION},<={MAX_SGLANG_VERSION})"
    )
