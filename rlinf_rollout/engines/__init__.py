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

"""LLM inference-engine wrappers (vendored from ``rlinf/hybrid_engines/``).

Each subpackage patches one upstream engine so it can (a) receive weights from an
external trainer over the vendored scheduler's collectives and (b) release/resume
its device memory between rollout rounds:

* :mod:`rlinf_rollout.engines.sglang` — SGLang ``Engine`` / ``Scheduler`` /
  ``TokenizerManager`` / ``DetokenizerManager`` overrides, dispatched by the
  installed ``sglang`` version.
* :mod:`rlinf_rollout.engines.vllm` — vLLM ``MultiprocExecutor`` / ``Worker``
  overrides, dispatched by the installed ``vllm`` version.

Both are import-time version-gated, so importing them without the corresponding
extra installed raises immediately instead of failing later at engine startup.
Nothing here reads a trainer config section: the sender's topology arrives as an
:class:`rlinf_rollout.weight_sync.llm.EngineWeightSyncSetup`.
"""
