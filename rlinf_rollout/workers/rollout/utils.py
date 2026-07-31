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

"""Shared helpers for the LLM engine workers.

Vendored from the training repo's ``rlinf/workers/rollout/utils.py`` minus the rank
mapping, which moved to :mod:`rlinf_rollout.weight_sync.llm` because it now derives
the sender layout from an api/v1 ``SourceTopology`` instead of from
``ModelParallelComponentPlacement``.
"""

import asyncio
import json
import os
import time
import typing
from contextlib import contextmanager
from dataclasses import dataclass

from omegaconf import DictConfig, OmegaConf

from rlinf_rollout.config import SUPPORTED_LLM_ROLLOUT_BACKENDS
from rlinf_rollout.data.io_struct import SeqGroupInfo
from rlinf_rollout.scheduler.worker.worker import Worker

if typing.TYPE_CHECKING:
    from vllm.outputs import RequestOutput


__all__ = [
    "MetaInfoStatsCollector",
    "RolloutEngineStats",
    "RunningStatusManager",
    "get_rollout_backend_worker",
    "print_multi_outputs",
    "print_multi_sglang_outputs",
    "print_sglang_outputs",
    "print_vllm_outputs",
]

COLOR_END = "\033[0m"


def green(text: str):
    """Wrap ``text`` in an ANSI green escape."""
    return f"\033[32m{text}\033[0m"


@contextmanager
def sharp_cover(header_text: str, prelen: int = 30, color="\033[32m"):
    """Print a ``#``-framed banner around the wrapped block."""
    len(header_text)
    print("#" * prelen + f" {color}>>> {header_text}{COLOR_END} " + "#" * prelen)

    try:
        yield
    finally:
        print("#" * prelen + f" {color}>>> {header_text}{COLOR_END} " + "#" * prelen)


def print_vllm_outputs(outputs: list["RequestOutput"]):
    """Pretty-print vLLM request outputs (prompt, text, token ids)."""
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generated_ids = output.outputs[0].token_ids
        print(
            f"{green('Prompt')}         : {prompt!r}",
            f"{green('Generated text')} : {generated_text!r}",
            f"{green('Generated ids')}  : {generated_ids}",
            sep="\n",
        )


def print_multi_outputs(resps_all: list[list["RequestOutput"]]):
    """Pretty-print vLLM outputs grouped per engine (dp) rank."""
    for i, resps in enumerate(resps_all):
        with sharp_cover(f"vllm dp {i}"):
            print_vllm_outputs(resps)


def print_sglang_outputs(prompts, outputs: list[dict], tokenizer):
    """Pretty-print SGLang outputs, decoding token ids with ``tokenizer``."""
    output_ids = [output["output_ids"] for output in outputs]
    output_texts = tokenizer.batch_decode(output_ids)
    for p, t, ids in zip(prompts, output_texts, output_ids):
        print(
            f"{green('Prompt')}         : {p!r}",
            f"{green('Generated text')} : {t!r}",
            f"{green('Generated ids')}  : {ids}",
            sep="\n",
        )


def print_multi_sglang_outputs(prompts, outputs: list[list[dict]], tokenizer):
    """Pretty-print SGLang outputs grouped per engine (dp) rank."""
    for i, resps in enumerate(outputs):
        with sharp_cover(f"sglang dp {i}"):
            print_sglang_outputs(prompts, resps, tokenizer)


def get_rollout_backend_worker(cfg: DictConfig) -> type[Worker]:
    """Return the engine worker class selected by ``rollout.rollout_backend``.

    ``rollout.sglang.serving_mode: worker_http`` selects the variant that also exposes
    an OpenAI-compatible HTTP endpoint per engine.

    Args:
        cfg: Rollout-system config (``rollout.kind: llm``).

    Returns:
        The worker class to launch.

    Raises:
        ValueError: If the backend is missing or unsupported.
    """
    rollout_backend = OmegaConf.select(cfg, "rollout.rollout_backend", default=None)
    if rollout_backend is None:
        raise ValueError(
            f"rollout.rollout_backend must be specified in the config. "
            f"Support {', '.join(SUPPORTED_LLM_ROLLOUT_BACKENDS)}."
        )
    if rollout_backend not in SUPPORTED_LLM_ROLLOUT_BACKENDS:
        raise ValueError(
            f"rollout_backend {rollout_backend} is not supported. "
            f"Support {', '.join(SUPPORTED_LLM_ROLLOUT_BACKENDS)}."
        )

    if rollout_backend == "vllm":
        from rlinf_rollout.workers.rollout.vllm.vllm_worker import VLLMWorker

        return VLLMWorker

    serving_mode = OmegaConf.select(cfg, "rollout.sglang.serving_mode", default=None)
    if serving_mode is None:
        from rlinf_rollout.workers.rollout.sglang.sglang_worker import SGLangWorker

        return SGLangWorker
    assert serving_mode in ("worker_http",), (
        f"Got serving_mode={serving_mode!r}; only 'worker_http' is supported when set."
    )
    from rlinf_rollout.workers.rollout.sglang.sglang_worker_server import (
        SGLangWorkerWithHTTPServer,
    )

    return SGLangWorkerWithHTTPServer


class RunningStatusManager:
    """Tracks the running / done / aborted sequence groups of one engine."""

    def __init__(self):
        self._running_seq_group: dict[SeqGroupInfo, asyncio.Task] = {}
        self._aborted_seq_group: list[SeqGroupInfo] = []
        # SeqGroupInfo that have been completed and sent downstream
        # only retained for debugging
        self._done_seq_group: list[SeqGroupInfo] = []

        # asyncio Events
        # set by scheduler coroutine to prevent rollout coroutine from exiting before potential migrations
        self.exit_rollout_iter = asyncio.Event()

    def add_task(self, seq_group: SeqGroupInfo, task: asyncio.Task):
        """Register the generation task of ``seq_group``."""
        assert seq_group not in self._running_seq_group, (
            f"Task for sequence group {seq_group.id} is already running."
        )
        self._running_seq_group[seq_group] = task

    def mark_done(self, seq_group: SeqGroupInfo):
        """Move ``seq_group`` from running to done."""
        assert seq_group in self._running_seq_group, (
            f"Task for SeqGroup {seq_group.id} not found. "
            "Check whether it has been added correctly or already marked done."
        )
        assert seq_group not in self._done_seq_group
        self._running_seq_group.pop(seq_group)
        self._done_seq_group.append(seq_group)

    def mark_aborted(self, seq_group: SeqGroupInfo):
        """Move ``seq_group`` from running to aborted."""
        assert seq_group in self._running_seq_group, (
            f"Task for SeqGroup {seq_group.id} not found. "
            "Check whether it has been added correctly or already marked aborted."
        )
        assert seq_group not in self._aborted_seq_group
        self._running_seq_group.pop(seq_group)
        self._aborted_seq_group.append(seq_group)

    async def wait_notification(self):
        """
        Wait until the scheduler notifies that it is safe to continue.
        This is used to prevent the rollout coroutine from exiting before potential migrations.
        """
        await self.exit_rollout_iter.wait()
        self.exit_rollout_iter.clear()

    def notify(self):
        """
        Call by scheduler to notify the rollout to continue.
        This is used to prevent the rollout coroutine from exiting before potential migrations.
        """
        self.exit_rollout_iter.set()

    def clear(self):
        """Drop all tracked groups and reset the migration event."""
        self._running_seq_group.clear()
        self._aborted_seq_group.clear()
        self._done_seq_group.clear()
        self.exit_rollout_iter.clear()

    def empty(self) -> bool:
        """Whether nothing is running and nothing has completed."""
        return len(self._running_seq_group) == 0 and len(self._done_seq_group) == 0

    def get_running_seq_groups(self) -> list[SeqGroupInfo]:
        """Sequence groups still generating."""
        return list(self._running_seq_group.keys())

    def get_done_seq_groups(self) -> list[SeqGroupInfo]:
        """Sequence groups that completed."""
        return self._done_seq_group

    def get_aborted_seq_groups(self) -> list[SeqGroupInfo]:
        """Sequence groups aborted for migration."""
        return self._aborted_seq_group

    def get_running_tasks(self) -> list[asyncio.Task]:
        """Asyncio tasks of the running sequence groups."""
        return list(self._running_seq_group.values())

    @property
    def num_seq_group_running(self) -> int:
        """Number of running sequence groups."""
        return len(self._running_seq_group)

    @property
    def num_seq_group_done(self) -> int:
        """Number of completed sequence groups."""
        return len(self._done_seq_group)

    @property
    def num_seq_group_aborted(self) -> int:
        """Number of aborted sequence groups."""
        return len(self._aborted_seq_group)

    @property
    def num_seq_group(self) -> int:
        """Total tracked sequence groups."""
        return (
            self.num_seq_group_running
            + self.num_seq_group_done
            + self.num_seq_group_aborted
        )

    @property
    def num_seq_running(self) -> int:
        """Sequences still generating across all running groups."""
        return sum(sg.num_running for sg in self.get_running_seq_groups())

    @property
    def num_seq_returned(self) -> int:
        """Sequences that already produced their output."""
        return self.num_seq - self.num_seq_running

    @property
    def num_seq(self) -> int:
        """Total sequences across running and done groups."""
        return sum(sg.group_size for sg in self.get_running_seq_groups()) + sum(
            sg.group_size for sg in self.get_done_seq_groups()
        )


@dataclass
class RolloutEngineStats:
    """Snapshot of one engine's queue / KV-cache occupancy."""

    num_running_reqs: int = 0
    max_running_reqs: int = 0
    num_used_tokens: int = 0
    max_total_num_tokens: int = 0
    token_usage: float = 0.0
    gen_throughput: float = 0.0
    num_queue_reqs: int = 0


class MetaInfoStatsCollector:
    """Collector for SGLang ``meta_info`` statistics.

    Only constructed when ``rollout.collect_meta_stats`` is true. Records
    per-request prompt/completion token counts and latencies into a JSONL file
    named by ``rollout.async_meta_stats_file``.
    """

    def __init__(self, output_file: str):
        self.output_file = output_file
        self.stats_buffer = []
        self.buffer_size = 100  # Write to file every 100 records

        # Ensure output directory exists
        os.makedirs(
            os.path.dirname(self.output_file)
            if os.path.dirname(self.output_file)
            else ".",
            exist_ok=True,
        )

        # Initialize file with header if it doesn't exist
        if not os.path.exists(self.output_file):
            with open(self.output_file, "w") as f:
                f.write("")  # Create empty file

    def collect_batch_stats(self, outputs: list[dict], batch_id: int) -> None:
        """Collect statistics from a batch of SGLang outputs.

        Args:
            outputs: List of SGLang output dictionaries
            batch_id: Unique identifier for this batch
        """
        current_time = time.time()

        for req_idx, output in enumerate(outputs):
            try:
                # Extract meta_info
                meta_info = output.get("meta_info", {})

                # Extract the specific metrics you requested
                stats_record = {
                    "timestamp": current_time,
                    "batch_id": batch_id,
                    "request_id": f"batch_{batch_id}_req_{req_idx}",
                    "prompt_tokens": meta_info.get("prompt_tokens", None),
                    "completion_tokens": meta_info.get("completion_tokens", None),
                    "e2e_latency": meta_info.get("e2e_latency", None),
                    "ttft": meta_info.get("ttft", None),
                    # Additional useful meta_info fields (if available)
                    "finish_reason": meta_info.get("finish_reason", {}).get(
                        "type", None
                    ),
                    "total_tokens": (
                        meta_info.get("prompt_tokens", 0)
                        + meta_info.get("completion_tokens", 0)
                    )
                    if meta_info.get("prompt_tokens") is not None
                    and meta_info.get("completion_tokens") is not None
                    else None,
                    # Add any other meta_info fields that might be useful
                    "meta_info_keys": list(
                        meta_info.keys()
                    ),  # For debugging/inspection
                }

                self.stats_buffer.append(stats_record)

            except Exception as e:
                # Log error but continue processing
                error_record = {
                    "timestamp": current_time,
                    "batch_id": batch_id,
                    "request_id": f"batch_{batch_id}_req_{req_idx}",
                    "error": str(e),
                    "output_keys": list(output.keys())
                    if isinstance(output, dict)
                    else "not_dict",
                }
                self.stats_buffer.append(error_record)

        # Write to file if buffer is full
        if len(self.stats_buffer) >= self.buffer_size:
            self._flush_to_file()

    def _flush_to_file(self) -> None:
        """Write buffered statistics to file."""
        if not self.stats_buffer:
            return

        with open(self.output_file, "a") as f:
            for record in self.stats_buffer:
                f.write(json.dumps(record) + "\n")

        print(f"Written {len(self.stats_buffer)} records to {self.output_file}")
        self.stats_buffer = []

    def finalize(self) -> None:
        """Flush any remaining data and close."""
        self._flush_to_file()
        print(f"Finalized stats collection. Data saved to {self.output_file}")
