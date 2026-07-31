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

"""RLinf Rollout: a standalone rollout system for embodied and LLM policies.

This package is developed inside the RLinf repository but is intended to be
exported as an independent repository (``git subtree split``). It therefore must
never import the training-side ``rlinf`` package; every dependency it needs is
vendored under ``rlinf_rollout/``.

The stable, training-framework-agnostic surface lives in
:mod:`rlinf_rollout.api.v1`.
"""

__version__ = "0.1.0.dev0"
