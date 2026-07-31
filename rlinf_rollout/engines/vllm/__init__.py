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

"""Version-gated entry point for the patched vLLM executor.

Vendored from the training repo's ``rlinf/workers/rollout/vllm/__init__.py``; the
dispatch lives next to the engine code here so the worker layer stays free of
engine-version logic.
"""

from importlib.metadata import PackageNotFoundError, version

from packaging.version import parse

__all__ = ["VLLMExecutor", "VLLM_VERSION"]

MIN_VLLM_VERSION = "0.8.5"
MAX_VLLM_VERSION = "0.9.0"


def _installed_version(pkg: str):
    try:
        return parse(version(pkg))
    except PackageNotFoundError:
        return None


_package_version = _installed_version("vllm")

if _package_version is None:
    raise ValueError(
        "vllm is not installed; install rlinf-rollout with the [vllm] extra."
    )
elif _package_version >= parse(MIN_VLLM_VERSION) and _package_version < parse(
    MAX_VLLM_VERSION
):
    #: Version of vLLM the wrappers were dispatched for.
    VLLM_VERSION = _package_version
    from rlinf_rollout.engines.vllm.vllm_0_8_5.executor import VLLMExecutor
else:
    raise ValueError(
        f"vllm version {_package_version} not supported "
        f"(need >={MIN_VLLM_VERSION},<{MAX_VLLM_VERSION})"
    )
