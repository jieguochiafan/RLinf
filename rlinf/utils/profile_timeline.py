# Copyright 2025 The RLinf Authors.
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

from __future__ import annotations

import json
import os
import socket
import threading
import time
from contextlib import contextmanager
from functools import wraps
from inspect import iscoroutinefunction, signature
from pathlib import Path
from typing import Any, Callable, Iterator


class TraceChunkHandler:
    """Export deterministic profiler chunks and append a JSONL manifest."""

    def __init__(self, output_dir: str | Path, *, component: str, rank: int) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.component = component
        self.rank = int(rank)
        self.session_id = f"{os.getpid()}_{time.time_ns()}"
        self.chunk_index = 0

    def __call__(self, profiler: Any) -> None:
        filename = (
            f"{self.session_id}_chunk_{self.chunk_index:06d}"
            f"_step_{int(profiler.step_num)}.pt.trace.json"
        )
        path = self.output_dir / filename
        profiler.export_chrome_trace(str(path))
        record = {
            "component": self.component,
            "rank": self.rank,
            "pid": os.getpid(),
            "session_id": self.session_id,
            "chunk_index": self.chunk_index,
            "step_num": int(profiler.step_num),
            "trace_file": filename,
            "exported_at_ns": time.time_ns(),
            "size_bytes": path.stat().st_size,
        }
        with (self.output_dir / "trace_manifest.jsonl").open(
            "a", encoding="utf-8", buffering=1
        ) as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        self.chunk_index += 1


def build_cuda_metadata(cuda: Any | None = None) -> dict[str, Any]:
    """Return CUDA device metadata for aligning profiler traces offline."""
    if cuda is None:
        import torch

        cuda = torch.cuda

    visible = [
        item.strip()
        for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if item.strip()
    ]
    result: dict[str, Any] = {
        "cuda_visible_devices": visible,
        "cuda_current_device": None,
        "cuda_devices": [],
    }
    if not cuda.is_available():
        return result

    result["cuda_current_device"] = int(cuda.current_device())
    for local_index in range(cuda.device_count()):
        props = cuda.get_device_properties(local_index)
        result["cuda_devices"].append(
            {
                "local_index": local_index,
                "visible_id": (
                    visible[local_index]
                    if local_index < len(visible)
                    else str(local_index)
                ),
                "name": str(props.name),
                "uuid": str(getattr(props, "uuid", "")),
            }
        )
    return result


def torch_profiler_schedule_kwargs() -> dict[str, int]:
    """Read torch profiler schedule settings from environment variables."""
    return {
        "wait": int(os.environ.get("RLINF_TORCH_PROFILE_WAIT", "5")),
        "warmup": int(os.environ.get("RLINF_TORCH_PROFILE_WARMUP", "3")),
        "active": int(os.environ.get("RLINF_TORCH_PROFILE_ACTIVE", "10")),
        "repeat": int(os.environ.get("RLINF_TORCH_PROFILE_REPEAT", "1")),
    }


def write_time_anchor(
    output_dir: str | Path,
    *,
    component: str,
    rank: int,
    include_cuda_metadata: bool = True,
) -> Path:
    """Write wall-clock and monotonic anchors for profiler trace alignment."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    record = {
        "component": component,
        "rank": int(rank),
        "pid": os.getpid(),
        "time_ns": time.time_ns(),
        "perf_counter_ns": time.perf_counter_ns(),
        "monotonic_ns": time.monotonic_ns(),
        "torch_profiler_schedule": torch_profiler_schedule_kwargs(),
    }
    if include_cuda_metadata:
        record.update(build_cuda_metadata())
    else:
        record["cuda_visible_devices"] = [
            item.strip()
            for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if item.strip()
        ]
    anchor_path = path / "time_anchor.json"
    anchor_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return anchor_path


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    """Read one key from a DictConfig, mapping, or object."""
    if config is None:
        return default
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def _json_safe(value: Any) -> Any:
    """Convert common metadata values to JSON-safe scalar values."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return str(value)


class TimelineRecorder:
    """Write complete runtime spans to a per-worker JSONL file.

    The recorder is intentionally a no-op when disabled so it can remain in hot
    paths. Each record contains both wall-clock timestamps for cross-process
    alignment and monotonic timestamps for reliable duration calculation.
    """

    def __init__(
        self,
        output_dir: str | Path | None,
        *,
        component: str,
        rank: int,
        enabled: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.component = component
        self.rank = int(rank)
        self._handle = None
        self._lock = threading.Lock()
        if not self.enabled:
            return

        if output_dir is None:
            raise ValueError(
                "output_dir is required when timeline recording is enabled"
            )
        process_name = (
            f"{component}_rank{self.rank}_pid{os.getpid()}_{time.time_ns()}"
        )
        self.output_dir = Path(output_dir) / process_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        write_time_anchor(
            self.output_dir,
            component=self.component,
            rank=self.rank,
            include_cuda_metadata=False,
        )
        self._handle = (self.output_dir / "events.jsonl").open(
            "a", encoding="utf-8", buffering=1
        )

    @classmethod
    def from_config(
        cls,
        cfg: Any,
        *,
        component: str,
        rank: int,
    ) -> "TimelineRecorder":
        """Build a recorder from ``runner.profile_timeline`` configuration."""
        runner_cfg = _config_get(cfg, "runner", None)
        timeline_cfg = _config_get(runner_cfg, "profile_timeline", None)
        enabled = bool(_config_get(timeline_cfg, "enabled", False))
        output_dir = _config_get(timeline_cfg, "output_dir", None)
        if enabled and not output_dir:
            logger_cfg = _config_get(runner_cfg, "logger", None)
            log_path = _config_get(logger_cfg, "log_path", "./results")
            output_dir = Path(str(log_path)) / "profile_timeline"
        return cls(
            output_dir,
            component=component,
            rank=rank,
            enabled=enabled,
        )

    @contextmanager
    def span(self, name: str, **metadata: Any) -> Iterator[None]:
        """Record a named interval, including failed intervals."""
        if not self.enabled:
            yield
            return

        start_wall_ns = time.time_ns()
        start_perf_ns = time.perf_counter_ns()
        status = "ok"
        try:
            yield
        except BaseException:
            status = "error"
            raise
        finally:
            end_perf_ns = time.perf_counter_ns()
            event = {
                "name": name,
                "component": self.component,
                "rank": self.rank,
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "start_wall_ns": start_wall_ns,
                "end_wall_ns": time.time_ns(),
                "start_perf_ns": start_perf_ns,
                "end_perf_ns": end_perf_ns,
                "duration_ns": end_perf_ns - start_perf_ns,
                "status": status,
                "args": _json_safe(metadata),
            }
            assert self._handle is not None
            with self._lock:
                self._handle.write(json.dumps(event, sort_keys=True) + "\n")

    def close(self) -> None:
        """Close the output stream if recording was enabled."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def timeline_span(
    name: str,
    *,
    include_args: tuple[str, ...] = (),
) -> Callable:
    """Decorate a worker method with a configurable timeline span."""

    def decorator(func: Callable) -> Callable:
        func_signature = signature(func)

        def get_metadata(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict:
            metadata = {}
            if include_args:
                bound = func_signature.bind_partial(*args, **kwargs)
                metadata.update(
                    {
                        key: bound.arguments[key]
                        for key in include_args
                        if key in bound.arguments
                    }
                )
            worker = args[0]
            if hasattr(worker, "global_step"):
                metadata["global_step"] = worker.global_step
            if hasattr(worker, "version"):
                metadata["policy_version"] = worker.version
            return metadata

        if iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(self, *args, **kwargs):
                recorder = getattr(self, "timeline", None)
                if recorder is None or not recorder.enabled:
                    return await func(self, *args, **kwargs)
                metadata = get_metadata((self, *args), kwargs)
                with recorder.span(name, **metadata):
                    return await func(self, *args, **kwargs)

            return async_wrapper

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            recorder = getattr(self, "timeline", None)
            if recorder is None or not recorder.enabled:
                return func(self, *args, **kwargs)
            metadata = get_metadata((self, *args), kwargs)
            with recorder.span(name, **metadata):
                return func(self, *args, **kwargs)

        return wrapper

    return decorator
