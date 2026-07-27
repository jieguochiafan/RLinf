from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from omegaconf import OmegaConf


def _cfg_get(cfg: Any, path: str, default: Any) -> Any:
    return OmegaConf.select(cfg, path, default=default)


@dataclass(frozen=True)
class RolloutProfilerConfig:
    enabled: bool
    output_dir: str
    component: str
    rank: int
    record_child_steps: bool = True
    child_step_sample_interval: int = 1
    record_channel_wait: bool = True
    record_chunk_profile: bool = True

    @classmethod
    def from_cfg(cls, cfg: Any, *, component: str, rank: int) -> "RolloutProfilerConfig":
        log_path = str(_cfg_get(cfg, "runner.logger.log_path", "logs"))
        enabled = bool(_cfg_get(cfg, "profiling.rollout.enabled", False))
        if os.environ.get("RLINF_ROLLOUT_PROFILE") == "1":
            enabled = True
        output_dir = str(
            _cfg_get(
                cfg,
                "profiling.rollout.output_dir",
                os.path.join(log_path, "rollout_profile"),
            )
        )
        return cls(
            enabled=enabled,
            output_dir=output_dir,
            component=component,
            rank=int(rank),
            record_child_steps=bool(
                _cfg_get(cfg, "profiling.rollout.record_child_steps", True)
            ),
            child_step_sample_interval=max(
                1,
                int(_cfg_get(cfg, "profiling.rollout.child_step_sample_interval", 1)),
            ),
            record_channel_wait=bool(
                _cfg_get(cfg, "profiling.rollout.record_channel_wait", True)
            ),
            record_chunk_profile=bool(
                _cfg_get(cfg, "profiling.rollout.record_chunk_profile", True)
            ),
        )


class TraceWriter(Protocol):
    def write(self, record: dict[str, Any]) -> None: ...

    def flush(self) -> None: ...


class JsonlTraceWriter:
    def __init__(self, output_dir: str, *, component: str, rank: int) -> None:
        self._path = Path(output_dir) / f"{component}_rank_{rank}.jsonl"
        self._handle = None

    @property
    def path(self) -> Path:
        return self._path

    def _ensure_open(self):
        if self._handle is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self._path.open("a", encoding="utf-8", buffering=1)
        return self._handle

    def write(self, record: dict[str, Any]) -> None:
        handle = self._ensure_open()
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    def flush(self) -> None:
        if self._handle is not None:
            self._handle.flush()


class NoopRolloutProfiler:
    enabled = False

    def event(self, event: str, **fields: Any) -> None:
        return None

    @contextmanager
    def span(self, event: str, **fields: Any) -> Iterator[None]:
        yield

    def record_metrics(self, event: str, metrics: dict[str, Any], **fields: Any) -> None:
        return None

    def flush(self) -> None:
        return None


class RolloutProfiler:
    enabled = True

    def __init__(
        self, config: RolloutProfilerConfig, writer: TraceWriter | None = None
    ) -> None:
        self.config = config
        self._writer = writer or JsonlTraceWriter(
            config.output_dir,
            component=config.component,
            rank=config.rank,
        )

    def _base_record(self, event: str, fields: dict[str, Any]) -> dict[str, Any]:
        return {
            "event": event,
            "component": self.config.component,
            "rank": self.config.rank,
            "pid": os.getpid(),
            "wall_ns": time.time_ns(),
            **fields,
        }

    def event(self, event: str, **fields: Any) -> None:
        self._writer.write(self._base_record(event, fields))

    @contextmanager
    def span(self, event: str, **fields: Any) -> Iterator[None]:
        start = time.perf_counter()
        self.event(f"{event}.start", **fields)
        try:
            yield
        finally:
            self.event(
                f"{event}.end",
                duration_s=max(time.perf_counter() - start, 0.0),
                **fields,
            )

    def record_metrics(self, event: str, metrics: dict[str, Any], **fields: Any) -> None:
        self.event(event, metrics=metrics, **fields)

    def flush(self) -> None:
        self._writer.flush()


def make_rollout_profiler(cfg: Any, *, component: str, rank: int):
    config = RolloutProfilerConfig.from_cfg(cfg, component=component, rank=rank)
    if not config.enabled:
        return NoopRolloutProfiler()
    return RolloutProfiler(config)


def write_profile_context_event(
    context: dict[str, Any] | None,
    record: dict[str, Any],
) -> None:
    if not context:
        return
    output_dir = context.get("rollout_profile_output_dir")
    if not output_dir:
        return
    rank = int(context.get("rank", 0))
    path = Path(str(output_dir)) / f"env_rank_{rank}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "component": "env",
        "rank": rank,
        "pid": context.get("pid"),
        "wall_ns": time.time_ns(),
        **record,
    }
    with path.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
