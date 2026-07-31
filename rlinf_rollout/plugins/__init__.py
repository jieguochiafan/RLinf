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

"""Optional rollout plugins.

These are algorithm-flavoured extras that the core rollout loop does not need:

* :mod:`rlinf_rollout.plugins.expert` — build a teacher/expert policy config from
  ``rollout.expert_model`` overrides (DAgger, OPD).
* :mod:`rlinf_rollout.plugins.rlt` — RLT feature model routing and transition
  bookkeeping.

They are only imported when the corresponding ``rollout.plugins.*.enabled`` flag is
set, so the core package stays importable without them.
"""

from rlinf_rollout.plugins.expert import build_expert_model_config

__all__ = ["build_expert_model_config"]
