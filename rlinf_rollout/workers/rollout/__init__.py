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

"""Rollout workers.

* :mod:`~rlinf_rollout.workers.rollout.hf` — embodied HuggingFace policy (Phase 2).
* :mod:`~rlinf_rollout.workers.rollout.sglang` /
  :mod:`~rlinf_rollout.workers.rollout.vllm` — in-process LLM engines that receive
  weights over the vendored scheduler's collectives.
* :mod:`~rlinf_rollout.workers.rollout.sglang_server` — out-of-process ``sglang``
  HTTP servers behind an ``sglang_router``.
* :mod:`~rlinf_rollout.workers.rollout.server` — OpenAI-compatible fan-in router and
  the feedback-ingest server.

Nothing is re-exported here: each engine subpackage imports its (heavy, optional)
engine dependency at module import time, so callers import the one they need. Use
:func:`rlinf_rollout.workers.rollout.utils.get_rollout_backend_worker` to pick an LLM
engine worker from config.
"""
