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

"""Shared timeline metadata for torch profiler traces."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _unlink_best_effort(path: Path) -> None:
    """Remove a temporary file without allowing cleanup errors to escape."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.warning("Unable to remove profiler temporary file %s", path)


def stop_profiler_safely(profiler: Any) -> Exception | None:
    """Stop a profiler and return any failure instead of raising it."""
    try:
        profiler.stop()
    except Exception as exc:
        return exc
    return None


def _atomic_write_text(path: Path, content: str) -> None:
    """Write text through a same-directory temporary file and atomic replace."""
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as temporary_file:
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    except Exception:
        _unlink_best_effort(temporary_path)
        raise


def _append_manifest_atomically(path: Path, line: str) -> None:
    """Append one complete manifest line under a stable sidecar file lock."""
    lock_path = path.with_suffix(".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        _atomic_write_text(path, existing + line)


class TraceChunkHandler:
    """Export profiler cycles as deterministic trace chunks with a manifest."""

    def __init__(
        self,
        output_dir: str | os.PathLike[str],
        component: str,
        rank: int,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.component = component
        self.rank = rank
        self.pid = os.getpid()
        self.session_id = f"{self.pid}_{time.time_ns()}"
        self.chunk_index = 0
        self._disabled = False
        session_metadata = build_time_anchor(
            component=self.component,
            rank=self.rank,
            pid=self.pid,
        )
        try:
            session_metadata.update(build_cuda_metadata())
        except Exception:
            logger.warning(
                "CUDA metadata unavailable for profiler session", exc_info=True
            )
        session_metadata["torch_profiler_schedule"] = torch_profiler_schedule_kwargs()
        session_metadata["session_id"] = self.session_id
        _atomic_write_text(
            self.output_dir / f"session_{self.session_id}.json",
            json.dumps(session_metadata, sort_keys=True) + "\n",
        )

    def __call__(self, profiler: Any) -> None:
        """Export one completed profiler cycle and append its manifest record."""
        if self._disabled:
            return

        temporary_path: Path | None = None
        try:
            step_num = int(profiler.step_num)
            trace_file = (
                f"{self.session_id}_chunk_{self.chunk_index:06d}_step_{step_num}"
                ".pt.trace.json"
            )
            trace_path = self.output_dir / trace_file
            descriptor, temporary_name = tempfile.mkstemp(
                dir=self.output_dir,
                prefix=f".{trace_file}.",
                suffix=".tmp",
            )
            os.close(descriptor)
            temporary_path = Path(temporary_name)
            profiler.export_chrome_trace(os.fspath(temporary_path))
            size_bytes = temporary_path.stat().st_size
            os.replace(temporary_path, trace_path)
            temporary_path = None
            manifest_entry = {
                "component": self.component,
                "rank": self.rank,
                "pid": self.pid,
                "session_id": self.session_id,
                "chunk_index": self.chunk_index,
                "step_num": step_num,
                "trace_file": trace_file,
                "exported_at_ns": time.time_ns(),
                "size_bytes": size_bytes,
            }
            manifest_path = self.output_dir / "trace_manifest.jsonl"
            _append_manifest_atomically(
                manifest_path,
                json.dumps(manifest_entry, sort_keys=True) + "\n",
            )
            self.chunk_index += 1
        except Exception:
            self._disabled = True
            logger.warning("Disabling torch trace export after failure", exc_info=True)
            if temporary_path is not None:
                _unlink_best_effort(temporary_path)


def build_time_anchor(
    component: str, rank: int, pid: int | None = None
) -> dict[str, Any]:
    """Build a wall/perf timestamp pair for aligning independent traces."""
    return {
        "component": component,
        "rank": rank,
        "pid": os.getpid() if pid is None else pid,
        "wall_time_ns": time.time_ns(),
        "perf_counter_ns": time.perf_counter_ns(),
        "clock": "time.time_ns/perf_counter_ns",
    }


def write_time_anchor(output_dir: str, component: str, rank: int) -> dict[str, Any]:
    """Write and return a time anchor in ``output_dir/time_anchor.json``."""
    os.makedirs(output_dir, exist_ok=True)
    anchor = build_time_anchor(component=component, rank=rank)
    try:
        anchor.update(build_cuda_metadata())
    except Exception:
        logger.warning("CUDA metadata unavailable for time anchor", exc_info=True)
    anchor["torch_profiler_schedule"] = torch_profiler_schedule_kwargs()
    path = Path(output_dir) / "time_anchor.json"
    _atomic_write_text(path, json.dumps(anchor, sort_keys=True) + "\n")
    return anchor


def build_cuda_metadata(cuda: Any = None) -> dict[str, Any]:
    """Build CUDA device metadata using process-local device numbering."""
    visible = [
        device.strip()
        for device in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if device.strip()
    ]
    result = {
        "cuda_visible_devices": visible,
        "cuda_current_device": None,
        "cuda_devices": [],
    }
    if cuda is None:
        import torch

        cuda = torch.cuda
    try:
        is_available = bool(cuda.is_available())
    except Exception:
        logger.warning("CUDA availability query failed", exc_info=True)
        return result
    if not is_available:
        return result

    try:
        result["cuda_current_device"] = int(cuda.current_device())
    except Exception:
        logger.warning("CUDA current device query failed", exc_info=True)
    try:
        device_count = int(cuda.device_count())
    except Exception:
        logger.warning("CUDA device count query failed", exc_info=True)
        return result

    for local_index in range(device_count):
        visible_id = (
            visible[local_index] if local_index < len(visible) else str(local_index)
        )
        try:
            properties = cuda.get_device_properties(local_index)
            name = str(getattr(properties, "name", ""))
            uuid = str(getattr(properties, "uuid", ""))
        except Exception:
            logger.warning(
                "CUDA device properties query failed for local device %s",
                local_index,
                exc_info=True,
            )
            name = ""
            uuid = ""
        result["cuda_devices"].append(
            {
                "local_index": local_index,
                "visible_id": visible_id,
                "name": name,
                "uuid": uuid,
            }
        )
    return result


def torch_profiler_schedule_kwargs() -> dict[str, int]:
    """Read shared torch profiler schedule settings from environment variables."""
    return {
        "wait": int(os.environ.get("RLINF_TORCH_PROFILE_WAIT", "5")),
        "warmup": int(os.environ.get("RLINF_TORCH_PROFILE_WARMUP", "3")),
        "active": int(os.environ.get("RLINF_TORCH_PROFILE_ACTIVE", "10")),
        "repeat": int(os.environ.get("RLINF_TORCH_PROFILE_REPEAT", "1")),
    }
