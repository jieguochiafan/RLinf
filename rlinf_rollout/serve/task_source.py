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

""":class:`~rlinf_rollout.api.v1.TaskSource` backends used by the daemon.

Three sources cover the Phase 4 entry point:

* :class:`ControlPlaneTaskSource` — the default. Work arrives from clients
  through the control plane, so the service is a passive server.
* :class:`StaticTaskSource` — a fixed list, used for embodied eval runs and
  tests.
* :class:`JsonlPromptTaskSource` — an LLM prompt file, used by the fixed-weight
  generation smoke run.

A source never touches Ray: :class:`ControlPlaneTaskSource` receives an
already-resolved control-plane handle and only calls ``next_tasks`` on it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence, Union

from rlinf_rollout.api.v1 import (
    EpisodeSpec,
    PromptSpec,
    RolloutMode,
    RolloutTask,
    SamplingParams,
    TaskKind,
    TaskSource,
)

__all__ = [
    "ControlPlaneTaskSource",
    "JsonlPromptTaskSource",
    "StaticTaskSource",
    "episode_task",
]


def episode_task(
    *,
    env_type: str,
    num_envs: int,
    mode: Union[str, RolloutMode] = RolloutMode.TRAIN,
    group_size: int = 1,
    num_chunk_steps: int = 0,
    max_episode_steps: int = 0,
    auto_reset: bool = True,
    env_config: Optional[dict[str, Any]] = None,
    priority: int = 0,
) -> RolloutTask:
    """Build one embodied :class:`~rlinf_rollout.api.v1.RolloutTask`.

    Args:
        env_type: Registered env type, e.g. ``maniskill``.
        num_envs: Parallel environments the task drives.
        mode: ``train`` (collect) or ``eval`` (measure).
        group_size: Episodes sharing an initial state.
        num_chunk_steps: Chunk steps per episode; ``0`` defers to the env.
        max_episode_steps: Episode-length cap; ``0`` defers to the env.
        auto_reset: Whether the env resets finished episodes itself.
        env_config: Extra env options passed through verbatim.
        priority: Scheduling priority; higher runs first.

    Returns:
        An ``EMBODIED_EVAL`` task in eval mode, else an ``EMBODIED_EPISODE`` task.
    """
    mode = RolloutMode(mode)
    kind = (
        TaskKind.EMBODIED_EVAL
        if mode is RolloutMode.EVAL
        else TaskKind.EMBODIED_EPISODE
    )
    return RolloutTask(
        kind=kind,
        mode=mode,
        priority=priority,
        episode=EpisodeSpec(
            env_type=env_type,
            num_envs=num_envs,
            group_size=group_size,
            num_chunk_steps=num_chunk_steps,
            max_episode_steps=max_episode_steps,
            auto_reset=auto_reset,
            env_config=dict(env_config or {}),
        ),
    )


class StaticTaskSource(TaskSource):
    """Serves a fixed list of tasks, then reports exhaustion.

    Args:
        tasks: Tasks to serve, in order.
        repeat: When ``True`` the list is served forever and the source never
            reports exhaustion — useful to keep a collect-mode service busy
            without a client.
    """

    def __init__(self, tasks: Iterable[RolloutTask], *, repeat: bool = False) -> None:
        self._tasks = list(tasks)
        self._repeat = repeat
        self._cursor = 0
        self.num_served = 0

    async def next_batch(self, max_num_tasks: int) -> list[RolloutTask]:
        """Return up to ``max_num_tasks`` tasks from the list."""
        if max_num_tasks <= 0 or not self._tasks:
            return []
        batch: list[RolloutTask] = []
        while len(batch) < max_num_tasks:
            if self._cursor >= len(self._tasks):
                if not self._repeat:
                    break
                self._cursor = 0
            batch.append(self._tasks[self._cursor])
            self._cursor += 1
        self.num_served += len(batch)
        return batch

    async def exhausted(self) -> bool:
        """Whether every task has been served (never ``True`` when repeating)."""
        return not self._repeat and self._cursor >= len(self._tasks)


class ControlPlaneTaskSource(TaskSource):
    """Pulls tasks submitted by clients through the control plane.

    Args:
        control_plane: Handle exposing ``next_tasks(max_num_tasks)`` and
            ``report_task_done(task_id, error)``. In the daemon this is the
            control-plane worker group; tests pass a stub. Calls may return a
            future-like object with ``wait()`` (the vendored scheduler's
            behaviour) or the value itself.
    """

    def __init__(self, control_plane: Any) -> None:
        self._control_plane = control_plane

    @staticmethod
    def _resolve(result: Any) -> Any:
        """Unwrap a worker-group call result into a plain value.

        The vendored scheduler returns a ``WorkerGroupFuncResult`` whose
        ``wait()`` yields one entry per rank; the control plane has a single rank.
        """
        wait = getattr(result, "wait", None)
        if wait is None:
            return result
        values = wait()
        if isinstance(values, (list, tuple)):
            return values[0] if values else None
        return values

    async def next_batch(self, max_num_tasks: int) -> list[RolloutTask]:
        """Drain up to ``max_num_tasks`` client-submitted tasks."""
        if max_num_tasks <= 0:
            return []
        tasks = self._resolve(self._control_plane.next_tasks(max_num_tasks))
        return list(tasks or [])

    async def report_done(self, task_id: str, error: Optional[str] = None) -> None:
        """Forward task completion to the control plane."""
        report = getattr(self._control_plane, "report_task_done", None)
        if report is None:
            return
        self._resolve(report(task_id, error))


class JsonlPromptTaskSource(TaskSource):
    """Turns a JSONL prompt file into LLM generation tasks.

    Each line is a JSON object. Recognized keys:

    * ``input_ids`` (list[int]) — pre-tokenized prompt, used as-is;
    * ``prompt`` / ``text`` / ``question`` (str) — raw text, tokenized with
      ``encode``;
    * ``answer`` / ``answers`` — reference answer echoed back in the output;
    * ``image_data`` — raw image payloads (bytes or URLs).

    Args:
        path: JSONL file to read.
        batch_size: Prompts per emitted task.
        group_size: Samples generated per prompt.
        encode: Tokenizer callable ``str -> list[int]``. Required when any line
            carries text instead of ``input_ids``.
        sampling: Sampling knobs attached to every task.
        mode: ``train`` or ``eval``.
        max_prompts: Stop after this many prompts; ``0`` reads the whole file.
        repeat: Restart from the top instead of reporting exhaustion.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a line has neither ``input_ids`` nor a text field, or a
            text field is present without ``encode``.
    """

    _TEXT_KEYS = ("prompt", "text", "question")

    def __init__(
        self,
        path: Union[str, Path],
        *,
        batch_size: int = 1,
        group_size: int = 1,
        encode: Optional[Callable[[str], Sequence[int]]] = None,
        sampling: Optional[SamplingParams] = None,
        mode: Union[str, RolloutMode] = RolloutMode.TRAIN,
        max_prompts: int = 0,
        repeat: bool = False,
    ) -> None:
        self._path = Path(path)
        if not self._path.is_file():
            raise FileNotFoundError(f"prompt file not found: {self._path}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        self._batch_size = batch_size
        self._group_size = group_size
        self._encode = encode
        self._sampling = sampling
        self._mode = RolloutMode(mode)
        self._repeat = repeat
        self._records = self._read(self._path, max_prompts)
        self._cursor = 0

    @staticmethod
    def _read(path: Path, max_prompts: int) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
                if not isinstance(record, dict):
                    raise ValueError(
                        f"{path}:{lineno}: expected a JSON object, got "
                        f"{type(record).__name__}."
                    )
                records.append(record)
                if 0 < max_prompts <= len(records):
                    break
        if not records:
            raise ValueError(f"{path} contains no prompts.")
        return records

    @property
    def num_prompts(self) -> int:
        """Number of prompt records read from the file."""
        return len(self._records)

    def _prompt_of(self, record: dict[str, Any]) -> tuple[list[int], Optional[str]]:
        input_ids = record.get("input_ids")
        if input_ids:
            text = next(
                (record[key] for key in self._TEXT_KEYS if record.get(key)), None
            )
            return [int(token) for token in input_ids], text
        text = next((record[key] for key in self._TEXT_KEYS if record.get(key)), None)
        if text is None:
            raise ValueError(
                f"prompt record needs 'input_ids' or one of {self._TEXT_KEYS}, got "
                f"keys {sorted(record)}."
            )
        if self._encode is None:
            raise ValueError(
                "JsonlPromptTaskSource needs an 'encode' callable to tokenize text "
                "prompts; pass one or provide 'input_ids' in the file."
            )
        return [int(token) for token in self._encode(text)], text

    async def next_batch(self, max_num_tasks: int) -> list[RolloutTask]:
        """Return up to ``max_num_tasks`` prompt-batch tasks."""
        tasks: list[RolloutTask] = []
        for _ in range(max(max_num_tasks, 0)):
            if self._cursor >= len(self._records):
                if not self._repeat:
                    break
                self._cursor = 0
            batch = self._records[self._cursor : self._cursor + self._batch_size]
            self._cursor += len(batch)
            input_ids: list[list[int]] = []
            prompt_texts: list[str] = []
            answers: list[Any] = []
            image_data: list[list[Any]] = []
            for record in batch:
                ids, text = self._prompt_of(record)
                input_ids.append(ids)
                prompt_texts.append(text or "")
                answers.append(record.get("answer", record.get("answers")))
                image_data.append(list(record.get("image_data") or []))
            tasks.append(
                RolloutTask(
                    kind=TaskKind.LLM_GENERATION,
                    mode=self._mode,
                    sampling=self._sampling or SamplingParams(n=self._group_size),
                    prompts=PromptSpec(
                        input_ids=input_ids,
                        prompt_texts=prompt_texts if any(prompt_texts) else None,
                        answers=answers,
                        image_data=image_data if any(image_data) else None,
                    ),
                )
            )
        return tasks

    async def exhausted(self) -> bool:
        """Whether the whole file has been served (never when repeating)."""
        return not self._repeat and self._cursor >= len(self._records)
