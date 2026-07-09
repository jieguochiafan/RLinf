# RoboCasa Rollout Profiler 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers-zh:executing-plans 逐任务实现此计划；实现功能或修复 bug 前使用 superpowers-zh:test-driven-development。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 实现一个默认关闭、可插拔的 rollout profile 模块，用于分析 RoboCasa Env CPU binding 端到端收益被抵消的位置。

**架构：** 新增 `rlinf/utils/rollout_profile/` 模块，集中管理 config、no-op profiler、JSONL writer、span/event schema 和后处理聚合。EnvWorker、RolloutWorker、RoboCasa vector env 只通过小接口记录边界事件，不直接处理文件路径或聚合逻辑。

**技术栈：** Python、OmegaConf、pytest、JSONL、CSV、现有 `Worker.timer` / timestamp 机制。

---

## 文件结构

- 创建：`rlinf/utils/rollout_profile/__init__.py`
  - 导出 profiler config、factory、writer、aggregator。
- 创建：`rlinf/utils/rollout_profile/profiler.py`
  - 定义 `RolloutProfilerConfig`、`RolloutProfiler`、`NoopRolloutProfiler`、`JsonlTraceWriter`、`make_rollout_profiler`。
- 创建：`rlinf/utils/rollout_profile/aggregation.py`
  - 读取 rollout profile JSONL，生成 `summary.json`、`timeline.csv`、`bottleneck_report.md`。
- 创建：`tools/summarize_rollout_profile.py`
  - CLI 包装 `aggregation.py`，便于 profile 运行结束后手动后处理。
- 修改：`rlinf/workers/env/env_worker.py`
  - 初始化 Env profiler；在 bootstrap send、rollout result wait、env chunk step、chunk profile、actor trajectory send 等边界调用 profiler。
- 修改：`rlinf/workers/rollout/hf/huggingface_worker.py`
  - 初始化 Rollout profiler；记录 env output wait、obs merge、predict、bootstrap value、split/send 等边界。
- 修改：`rlinf/envs/venv/venv.py`
  - 让 vector env 的子环境 start/end 事件可通过新 profile writer 输出，同时保留现有 `log_sim_timestamps` 行为。
- 修改：`rlinf/envs/robocasa/venv.py`
  - 将 RoboCasa child step timing 写入新 profile writer，而不是直接散落 JSONL 文件逻辑。
- 修改：`examples/embodiment/config/robocasa_baseline_profile_openpi.yaml`
  - 添加默认关闭的 `profiling.rollout` 示例字段，便于 override 启用。
- 修改：`examples/embodiment/config/robocasa_profile_pairing_dynamic_core_donation.yaml`
  - 添加默认关闭的 `profiling.rollout` 示例字段。
- 创建：`tests/unit_tests/test_rollout_profiler.py`
  - 覆盖 config 解析、no-op、span/event、writer schema。
- 创建：`tests/unit_tests/test_rollout_profile_aggregation.py`
  - 覆盖 JSONL 后处理 summary/timeline/report。
- 创建：`tests/unit_tests/test_rollout_profile_worker_integration.py`
  - 覆盖 EnvWorker/RolloutWorker 使用 fake profiler 的边界埋点。
- 创建：`tests/unit_tests/test_rollout_profile_robocasa_events.py`
  - 覆盖 RoboCasa child step timing 通过统一 writer 输出。

## 任务 1：核心可插拔 Profiler 模块

**文件：**
- 创建：`rlinf/utils/rollout_profile/__init__.py`
- 创建：`rlinf/utils/rollout_profile/profiler.py`
- 测试：`tests/unit_tests/test_rollout_profiler.py`

- [ ] **步骤 1：编写失败测试**

在 `tests/unit_tests/test_rollout_profiler.py` 写入：

```python
import json
import time

from omegaconf import OmegaConf

from rlinf.utils.rollout_profile import (
    NoopRolloutProfiler,
    RolloutProfilerConfig,
    make_rollout_profiler,
)


def test_rollout_profiler_config_defaults_to_disabled(tmp_path):
    cfg = OmegaConf.create({"runner": {"logger": {"log_path": str(tmp_path)}}})

    profile_cfg = RolloutProfilerConfig.from_cfg(cfg, component="env", rank=2)

    assert profile_cfg.enabled is False
    assert profile_cfg.component == "env"
    assert profile_cfg.rank == 2
    assert profile_cfg.output_dir == str(tmp_path / "rollout_profile")
    assert profile_cfg.record_child_steps is True
    assert profile_cfg.child_step_sample_interval == 1


def test_make_rollout_profiler_returns_noop_when_disabled(tmp_path):
    cfg = OmegaConf.create({"runner": {"logger": {"log_path": str(tmp_path)}}})

    profiler = make_rollout_profiler(cfg, component="env", rank=0)

    assert isinstance(profiler, NoopRolloutProfiler)
    with profiler.span("env.recv_rollout_results", epoch=0):
        time.sleep(0)
    profiler.event("env.event", value=1)
    profiler.flush()
    assert not (tmp_path / "rollout_profile").exists()


def test_rollout_profiler_writes_event_and_span(tmp_path):
    cfg = OmegaConf.create(
        {
            "runner": {"logger": {"log_path": str(tmp_path)}},
            "profiling": {
                "rollout": {
                    "enabled": True,
                    "output_dir": str(tmp_path / "profile"),
                }
            },
        }
    )

    profiler = make_rollout_profiler(cfg, component="rollout", rank=3)
    profiler.event("rollout.recv_env_output", mode="train", chunk_step=0)
    with profiler.span("rollout.predict", mode="train", batch_size=8):
        time.sleep(0)
    profiler.flush()

    path = tmp_path / "profile" / "rollout_rank_3.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["event"] for row in rows] == [
        "rollout.recv_env_output",
        "rollout.predict.start",
        "rollout.predict.end",
    ]
    assert rows[0]["component"] == "rollout"
    assert rows[0]["rank"] == 3
    assert rows[0]["mode"] == "train"
    assert isinstance(rows[0]["wall_ns"], int)
    assert rows[2]["duration_s"] >= 0.0


def test_rollout_profiler_env_override_enables_profile(tmp_path, monkeypatch):
    cfg = OmegaConf.create({"runner": {"logger": {"log_path": str(tmp_path)}}})
    monkeypatch.setenv("RLINF_ROLLOUT_PROFILE", "1")

    profiler = make_rollout_profiler(cfg, component="env", rank=1)
    profiler.event("env.enabled_by_env")
    profiler.flush()

    assert (tmp_path / "rollout_profile" / "env_rank_1.jsonl").exists()
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
pytest tests/unit_tests/test_rollout_profiler.py -q
```

预期：FAIL，报错包含 `ModuleNotFoundError: No module named 'rlinf.utils.rollout_profile'`。

- [ ] **步骤 3：实现最小核心模块**

在 `rlinf/utils/rollout_profile/profiler.py` 实现：

```python
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol

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
        rollout_cfg = _cfg_get(cfg, "profiling.rollout", {})
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
                1, int(_cfg_get(cfg, "profiling.rollout.child_step_sample_interval", 1))
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

    def __init__(self, config: RolloutProfilerConfig, writer: TraceWriter | None = None):
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
```

在 `rlinf/utils/rollout_profile/__init__.py` 导出：

```python
from .profiler import (
    JsonlTraceWriter,
    NoopRolloutProfiler,
    RolloutProfiler,
    RolloutProfilerConfig,
    make_rollout_profiler,
)

__all__ = [
    "JsonlTraceWriter",
    "NoopRolloutProfiler",
    "RolloutProfiler",
    "RolloutProfilerConfig",
    "make_rollout_profiler",
]
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
pytest tests/unit_tests/test_rollout_profiler.py -q
```

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add rlinf/utils/rollout_profile tests/unit_tests/test_rollout_profiler.py
git commit -s -m "feat: add pluggable rollout profiler"
```

## 任务 2：后处理聚合模块和 CLI

**文件：**
- 创建：`rlinf/utils/rollout_profile/aggregation.py`
- 创建：`tools/summarize_rollout_profile.py`
- 修改：`rlinf/utils/rollout_profile/__init__.py`
- 测试：`tests/unit_tests/test_rollout_profile_aggregation.py`

- [ ] **步骤 1：编写失败测试**

在 `tests/unit_tests/test_rollout_profile_aggregation.py` 写入：

```python
import csv
import json

from rlinf.utils.rollout_profile import summarize_rollout_profile


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_summarize_rollout_profile_writes_summary_timeline_and_report(tmp_path):
    profile_dir = tmp_path / "profile"
    _write_jsonl(
        profile_dir / "env_rank_0.jsonl",
        [
            {
                "event": "env.recv_rollout_results.start",
                "component": "env",
                "rank": 0,
                "pid": 10,
                "wall_ns": 1_000_000_000,
                "epoch": 0,
                "chunk_step": 0,
            },
            {
                "event": "env.recv_rollout_results.end",
                "component": "env",
                "rank": 0,
                "pid": 10,
                "wall_ns": 1_200_000_000,
                "duration_s": 0.2,
                "epoch": 0,
                "chunk_step": 0,
            },
            {
                "event": "env.chunk_profile",
                "component": "env",
                "rank": 0,
                "pid": 10,
                "wall_ns": 1_300_000_000,
                "metrics": {"wait_recv_s": 0.5, "stack_s": 0.1},
            },
        ],
    )
    _write_jsonl(
        profile_dir / "rollout_rank_0.jsonl",
        [
            {
                "event": "rollout.predict.end",
                "component": "rollout",
                "rank": 0,
                "pid": 20,
                "wall_ns": 1_500_000_000,
                "duration_s": 0.3,
            }
        ],
    )

    outputs = summarize_rollout_profile(profile_dir)

    assert outputs.summary_path.exists()
    assert outputs.timeline_path.exists()
    assert outputs.report_path.exists()
    summary = json.loads(outputs.summary_path.read_text())
    assert summary["event_counts"]["env.recv_rollout_results.end"] == 1
    assert summary["duration_s_by_event"]["rollout.predict.end"] == 0.3
    assert summary["metric_sums"]["env.chunk_profile.wait_recv_s"] == 0.5

    with outputs.timeline_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["event"] == "env.recv_rollout_results.start"
    assert rows[0]["relative_start_s"] == "0.000000"
    assert "rollout.predict.end" in outputs.report_path.read_text()
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
pytest tests/unit_tests/test_rollout_profile_aggregation.py -q
```

预期：FAIL，报错包含 `cannot import name 'summarize_rollout_profile'`。

- [ ] **步骤 3：实现聚合模块和 CLI**

在 `rlinf/utils/rollout_profile/aggregation.py` 实现：

```python
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RolloutProfileSummaryOutputs:
    summary_path: Path
    timeline_path: Path
    report_path: Path


def _iter_records(profile_dir: Path):
    for path in sorted(profile_dir.glob("*_rank_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def summarize_rollout_profile(profile_dir: str | Path) -> RolloutProfileSummaryOutputs:
    profile_path = Path(profile_dir)
    records = list(_iter_records(profile_path))
    event_counts = Counter(record["event"] for record in records)
    duration_s_by_event: dict[str, float] = defaultdict(float)
    metric_sums: dict[str, float] = defaultdict(float)
    wall_times = [int(record["wall_ns"]) for record in records if "wall_ns" in record]
    t0 = min(wall_times) if wall_times else 0

    timeline_rows = []
    for record in sorted(records, key=lambda item: int(item.get("wall_ns", 0))):
        event = str(record.get("event", ""))
        duration = record.get("duration_s")
        if isinstance(duration, (int, float)):
            duration_s_by_event[event] += float(duration)
        metrics = record.get("metrics")
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    metric_sums[f"{event}.{key}"] += float(value)
        wall_ns = int(record.get("wall_ns", t0))
        relative_start_s = (wall_ns - t0) / 1_000_000_000 if t0 else 0.0
        relative_end_s = (
            relative_start_s + float(duration)
            if isinstance(duration, (int, float))
            else relative_start_s
        )
        timeline_rows.append(
            {
                "relative_start_s": f"{relative_start_s:.6f}",
                "relative_end_s": f"{relative_end_s:.6f}",
                "event": event,
                "component": record.get("component", ""),
                "rank": record.get("rank", ""),
                "pid": record.get("pid", ""),
                "epoch": record.get("epoch", ""),
                "chunk_step": record.get("chunk_step", ""),
                "duration_s": "" if duration is None else f"{float(duration):.6f}",
            }
        )

    summary = {
        "record_count": len(records),
        "event_counts": dict(sorted(event_counts.items())),
        "duration_s_by_event": {
            key: float(value) for key, value in sorted(duration_s_by_event.items())
        },
        "metric_sums": {
            key: float(value) for key, value in sorted(metric_sums.items())
        },
    }

    summary_path = profile_path / "summary.json"
    timeline_path = profile_path / "timeline.csv"
    report_path = profile_path / "bottleneck_report.md"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with timeline_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "relative_start_s",
            "relative_end_s",
            "event",
            "component",
            "rank",
            "pid",
            "epoch",
            "chunk_step",
            "duration_s",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(timeline_rows)

    top_durations = sorted(
        duration_s_by_event.items(), key=lambda item: item[1], reverse=True
    )[:10]
    report_lines = ["# Rollout Profile Bottleneck Report", "", "## Top Durations", ""]
    for event, duration_s in top_durations:
        report_lines.append(f"- `{event}`: {duration_s:.6f}s")
    report_path.write_text("\n".join(report_lines) + "\n")
    return RolloutProfileSummaryOutputs(summary_path, timeline_path, report_path)
```

在 `rlinf/utils/rollout_profile/__init__.py` 增加导出：

```python
from .aggregation import RolloutProfileSummaryOutputs, summarize_rollout_profile
```

在 `tools/summarize_rollout_profile.py` 实现：

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from rlinf.utils.rollout_profile import summarize_rollout_profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile_dir", type=Path)
    args = parser.parse_args()
    outputs = summarize_rollout_profile(args.profile_dir)
    print(outputs.summary_path)
    print(outputs.timeline_path)
    print(outputs.report_path)


if __name__ == "__main__":
    main()
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
pytest tests/unit_tests/test_rollout_profile_aggregation.py -q
python tools/summarize_rollout_profile.py /tmp/nonexistent-profile-dir
```

预期：pytest PASS。CLI 对空或不存在目录应生成空 summary，若实现选择要求目录存在，则命令预期为带明确错误；保持测试以存在目录为准。

- [ ] **步骤 5：Commit**

```bash
git add rlinf/utils/rollout_profile tools/summarize_rollout_profile.py tests/unit_tests/test_rollout_profile_aggregation.py
git commit -s -m "feat: summarize rollout profile traces"
```

## 任务 3：EnvWorker 接入 Profiler

**文件：**
- 修改：`rlinf/workers/env/env_worker.py`
- 测试：`tests/unit_tests/test_rollout_profile_worker_integration.py`

- [ ] **步骤 1：编写失败测试**

在 `tests/unit_tests/test_rollout_profile_worker_integration.py` 写入 Env fake profiler 测试：

```python
from contextlib import contextmanager

import torch
from omegaconf import OmegaConf

from rlinf.data.embodied_io_struct import RolloutResult
from rlinf.workers.env.env_worker import EnvWorker


class FakeProfiler:
    def __init__(self):
        self.events = []

    def event(self, event, **fields):
        self.events.append((event, fields))

    @contextmanager
    def span(self, event, **fields):
        self.events.append((event + ".start", fields))
        yield
        self.events.append((event + ".end", fields))

    def record_metrics(self, event, metrics, **fields):
        self.events.append((event, {"metrics": metrics, **fields}))

    def flush(self):
        self.events.append(("flush", {}))


class FakeChannel:
    def __init__(self, item):
        self.item = item

    def get(self, key=None):
        return self.item


def _make_env_worker_for_recv():
    worker = object.__new__(EnvWorker)
    worker._rank = 0
    worker.src_rank_map = {"rollout_train": [(0, 2)]}
    worker.rollout_profiler = FakeProfiler()
    return worker


def test_env_worker_profiles_recv_rollout_results():
    worker = _make_env_worker_for_recv()
    result = RolloutResult(
        actions=torch.zeros(2, 5, 12),
        prev_logprobs=None,
        prev_values=None,
        bootstrap_values=None,
        forward_inputs={"action": torch.zeros(2, 5, 12)},
    )

    worker.recv_rollout_results(FakeChannel(result), mode="train")

    event_names = [name for name, _fields in worker.rollout_profiler.events]
    assert "env.recv_rollout_results.start" in event_names
    assert "env.recv_rollout_results.end" in event_names


def test_env_worker_initializes_rollout_profiler_from_cfg(monkeypatch, tmp_path):
    captured = {}

    def fake_factory(cfg, *, component, rank):
        captured["component"] = component
        captured["rank"] = rank
        return FakeProfiler()

    monkeypatch.setattr(
        "rlinf.workers.env.env_worker.make_rollout_profiler",
        fake_factory,
    )
    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create({"runner": {"logger": {"log_path": str(tmp_path)}}})
    worker._rank = 4

    worker._init_rollout_profiler()

    assert captured == {"component": "env", "rank": 4}
    assert isinstance(worker.rollout_profiler, FakeProfiler)
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
pytest tests/unit_tests/test_rollout_profile_worker_integration.py::test_env_worker_initializes_rollout_profiler_from_cfg -q
pytest tests/unit_tests/test_rollout_profile_worker_integration.py::test_env_worker_profiles_recv_rollout_results -q
```

预期：FAIL，分别报 `_init_rollout_profiler` 不存在或没有记录 span。

- [ ] **步骤 3：实现 EnvWorker 接入**

在 `rlinf/workers/env/env_worker.py` 顶部导入：

```python
from rlinf.utils.rollout_profile import make_rollout_profiler
```

在 `EnvWorker.__init__` 末尾或 `init_worker` 前可用位置增加：

```python
self._init_rollout_profiler()
```

如果单元测试用 `object.__new__`，新增方法：

```python
def _init_rollout_profiler(self) -> None:
    self.rollout_profiler = make_rollout_profiler(
        self.cfg,
        component="env",
        rank=self._rank,
    )
```

在 `recv_rollout_results` 中包裹 channel get 和 merge：

```python
with self.rollout_profiler.span("env.recv_rollout_results", mode=mode):
    ...
```

在 `_run_interact_once` 的关键边界补充：

```python
with self.rollout_profiler.span("env.bootstrap_step", epoch=epoch):
    env_outputs = self.bootstrap_step()
with self.rollout_profiler.span("env.send_env_batch", mode="train", epoch=epoch, stage=stage_id):
    self.send_env_batch(...)
with self.rollout_profiler.span("env.send_rollout_trajectories", stage=stage_id):
    await self.send_rollout_trajectories(...)
```

在 `env_interact_step` 中拆分：

```python
with self.rollout_profiler.span("env.prepare_actions", epoch=epoch, chunk_step=chunk_step_idx, stage=stage_id):
    chunk_actions = prepare_actions(...)
with self.rollout_profiler.span("env.chunk_step", epoch=epoch, chunk_step=chunk_step_idx, stage=stage_id):
    ...
if chunk_profile and self.rollout_profiler.enabled:
    self.rollout_profiler.record_metrics(
        "env.chunk_profile",
        chunk_profile,
        epoch=epoch,
        chunk_step=chunk_step_idx,
        stage=stage_id,
    )
```

保持现有 `log_sim_timestamps` 逻辑不变。

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
pytest tests/unit_tests/test_rollout_profile_worker_integration.py::test_env_worker_initializes_rollout_profiler_from_cfg -q
pytest tests/unit_tests/test_rollout_profile_worker_integration.py::test_env_worker_profiles_recv_rollout_results -q
```

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add rlinf/workers/env/env_worker.py tests/unit_tests/test_rollout_profile_worker_integration.py
git commit -s -m "feat: instrument env rollout profile spans"
```

## 任务 4：RolloutWorker 接入 Profiler

**文件：**
- 修改：`rlinf/workers/rollout/hf/huggingface_worker.py`
- 测试：`tests/unit_tests/test_rollout_profile_worker_integration.py`

- [ ] **步骤 1：编写失败测试**

在同一测试文件追加：

```python
import asyncio

from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class FakeAsyncWait:
    def __init__(self, item):
        self.item = item

    async def async_wait(self):
        return self.item


class FakeAsyncChannel:
    def __init__(self, item):
        self.item = item

    def get(self, key=None, async_op=False):
        assert async_op is True
        return FakeAsyncWait(self.item)


def _make_rollout_worker_for_recv():
    worker = object.__new__(MultiStepRolloutWorker)
    worker._rank = 0
    worker.src_ranks = {"train": [(0, 2)]}
    worker.rollout_profiler = FakeProfiler()
    return worker


def test_rollout_worker_profiles_recv_env_output():
    worker = _make_rollout_worker_for_recv()
    obs_batch = {
        "obs": {
            "states": torch.zeros(2, 4),
            "task_descriptions": ["a", "b"],
        },
        "final_obs": None,
    }

    asyncio.run(worker.recv_env_output(FakeAsyncChannel(obs_batch), mode="train"))

    event_names = [name for name, _fields in worker.rollout_profiler.events]
    assert "rollout.recv_env_output.start" in event_names
    assert "rollout.recv_env_output.end" in event_names


def test_rollout_worker_initializes_rollout_profiler_from_cfg(monkeypatch, tmp_path):
    captured = {}

    def fake_factory(cfg, *, component, rank):
        captured["component"] = component
        captured["rank"] = rank
        return FakeProfiler()

    monkeypatch.setattr(
        "rlinf.workers.rollout.hf.huggingface_worker.make_rollout_profiler",
        fake_factory,
    )
    worker = object.__new__(MultiStepRolloutWorker)
    worker.cfg = OmegaConf.create({"runner": {"logger": {"log_path": str(tmp_path)}}})
    worker._rank = 5

    worker._init_rollout_profiler()

    assert captured == {"component": "rollout", "rank": 5}
    assert isinstance(worker.rollout_profiler, FakeProfiler)
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
pytest tests/unit_tests/test_rollout_profile_worker_integration.py::test_rollout_worker_initializes_rollout_profiler_from_cfg -q
pytest tests/unit_tests/test_rollout_profile_worker_integration.py::test_rollout_worker_profiles_recv_env_output -q
```

预期：FAIL，报 `_init_rollout_profiler` 不存在或没有记录 span。

- [ ] **步骤 3：实现 RolloutWorker 接入**

在 `rlinf/workers/rollout/hf/huggingface_worker.py` 顶部导入：

```python
from rlinf.utils.rollout_profile import make_rollout_profiler
```

在 `MultiStepRolloutWorker.__init__` 初始化：

```python
self._init_rollout_profiler()
```

新增方法：

```python
def _init_rollout_profiler(self) -> None:
    self.rollout_profiler = make_rollout_profiler(
        self.cfg,
        component="rollout",
        rank=self._rank,
    )
```

在 `recv_env_output` 中包裹 channel wait 和 merge：

```python
with self.rollout_profiler.span("rollout.recv_env_output", mode=mode):
    ...
with self.rollout_profiler.span("rollout.merge_obs_batches", mode=mode, source_count=len(obs_batches)):
    return self._merge_obs_batches(obs_batches)
```

在 `predict` 中用现有 `profile_context` tags 记录：

```python
with self.rollout_profiler.span("rollout.predict", mode=mode, **(profile_context or {})):
    ...
```

在 `get_bootstrap_values`、`send_rollout_result` 中补充：

```python
with self.rollout_profiler.span("rollout.bootstrap_values", **(profile_context or {})):
    ...
with self.rollout_profiler.span("rollout.send_rollout_result", mode=mode):
    ...
```

保持现有 `log_generation_timestamps` 和 torch profiler 不变。

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
pytest tests/unit_tests/test_rollout_profile_worker_integration.py -q
```

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add rlinf/workers/rollout/hf/huggingface_worker.py tests/unit_tests/test_rollout_profile_worker_integration.py
git commit -s -m "feat: instrument rollout profile spans"
```

## 任务 5：RoboCasa 子进程事件统一写入

**文件：**
- 修改：`rlinf/envs/venv/venv.py`
- 修改：`rlinf/envs/robocasa/venv.py`
- 测试：`tests/unit_tests/test_rollout_profile_robocasa_events.py`

- [ ] **步骤 1：编写失败测试**

在 `tests/unit_tests/test_rollout_profile_robocasa_events.py` 写入：

```python
import json

from rlinf.envs.robocasa.venv import RobocasaSubprocEnv


def test_robocasa_child_step_timing_uses_rollout_profile_context(tmp_path):
    env = object.__new__(RobocasaSubprocEnv)
    env.is_closed = False
    env._sim_timestamp_context = {
        "output_dir": str(tmp_path / "legacy"),
        "rollout_profile_output_dir": str(tmp_path / "profile"),
        "rank": 2,
        "pid": 123,
        "epoch": 0,
        "chunk_step": 1,
        "stage": 0,
        "stage_num": 1,
        "local_envs": 1,
        "record_child_steps": True,
        "child_step_sample_interval": 1,
    }
    env._sim_timestamp_file = None
    env._sim_vector_step_index = 0
    env._sim_async_step_starts = {}
    env._last_chunk_profile = None
    env.env_num = 1

    env.record_robocasa_step_timing_events(
        [
            {
                "robocasa_step_timings": [
                    {
                        "local_env": 0,
                        "duration_s": 0.12,
                        "wall_start_ns": 10,
                        "wall_end_ns": 20,
                        "chunk_action_index": 0,
                        "repeat_index": 0,
                    }
                ]
            }
        ],
        vector_step=3,
    )

    profile_path = tmp_path / "profile" / "env_rank_2.jsonl"
    rows = [json.loads(line) for line in profile_path.read_text().splitlines()]
    assert rows[0]["event"] == "robocasa.child_step"
    assert rows[0]["rank"] == 2
    assert rows[0]["local_env"] == 0
    assert rows[0]["duration_s"] == 0.12
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
pytest tests/unit_tests/test_rollout_profile_robocasa_events.py -q
```

预期：FAIL，因为 profile JSONL 尚未写入。

- [ ] **步骤 3：实现统一 context 写入 helper**

在 `rlinf/utils/rollout_profile/profiler.py` 增加 helper：

```python
def write_profile_context_event(context: dict[str, Any] | None, record: dict[str, Any]) -> None:
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
```

在 `rlinf/envs/robocasa/venv.py` 导入并调用：

```python
from rlinf.utils.rollout_profile import write_profile_context_event
```

在 `record_robocasa_step_timing_events` 中保留 legacy writer，同时新增：

```python
if context.get("record_child_steps", False):
    interval = max(1, int(context.get("child_step_sample_interval", 1)))
    if int(vector_step) % interval == 0:
        write_profile_context_event(context, {"event": "robocasa.child_step", ...})
```

在 `EnvWorker.env_interact_step` 构造 `subenv_timestamp_context` 时，当 profiler enabled 也传入：

```python
"rollout_profile_output_dir": self.rollout_profiler.config.output_dir,
"record_child_steps": self.rollout_profiler.config.record_child_steps,
"child_step_sample_interval": self.rollout_profiler.config.child_step_sample_interval,
```

如果 profiler 是 no-op，使用 `getattr(self.rollout_profiler, "config", None)` 防止访问失败。

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
pytest tests/unit_tests/test_rollout_profile_robocasa_events.py -q
pytest tests/unit_tests/test_resource_pool_env_binding.py -q
```

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add rlinf/utils/rollout_profile rlinf/envs/robocasa/venv.py rlinf/envs/venv/venv.py rlinf/workers/env/env_worker.py tests/unit_tests/test_rollout_profile_robocasa_events.py
git commit -s -m "feat: route robocasa child step profile events"
```

## 任务 6：Profile 配置示例

**文件：**
- 修改：`examples/embodiment/config/robocasa_baseline_profile_openpi.yaml`
- 修改：`examples/embodiment/config/robocasa_profile_pairing_dynamic_core_donation.yaml`

- [ ] **步骤 1：添加默认关闭配置**

在两个配置文件顶层追加：

```yaml
profiling:
  rollout:
    enabled: false
    output_dir: ${runner.logger.log_path}/rollout_profile
    record_child_steps: true
    child_step_sample_interval: 1
    record_channel_wait: true
    record_chunk_profile: true
```

- [ ] **步骤 2：验证 Hydra 可解析**

运行：

```bash
source /data1/miliang/RLinf/robocasa_openpi/bin/activate
python - <<'PY'
from hydra import initialize_config_dir, compose
from pathlib import Path
config_dir = str(Path("examples/embodiment/config").resolve())
with initialize_config_dir(version_base="1.1", config_dir=config_dir):
    for name in [
        "robocasa_baseline_profile_openpi",
        "robocasa_profile_pairing_dynamic_core_donation",
    ]:
        cfg = compose(config_name=name)
        assert cfg.profiling.rollout.enabled is False
        assert str(cfg.profiling.rollout.output_dir).endswith("/rollout_profile")
print("ok")
PY
```

预期：输出 `ok`。

- [ ] **步骤 3：Commit**

```bash
git add examples/embodiment/config/robocasa_baseline_profile_openpi.yaml examples/embodiment/config/robocasa_profile_pairing_dynamic_core_donation.yaml
git commit -s -m "docs: add robocasa rollout profile config knobs"
```

## 任务 7：集成验证和修正

**文件：**
- 可能修改：前面任务涉及的实现文件

- [ ] **步骤 1：运行目标单测**

运行：

```bash
pytest \
  tests/unit_tests/test_rollout_profiler.py \
  tests/unit_tests/test_rollout_profile_aggregation.py \
  tests/unit_tests/test_rollout_profile_worker_integration.py \
  tests/unit_tests/test_rollout_profile_robocasa_events.py \
  tests/unit_tests/test_profile_timeline.py \
  tests/unit_tests/test_resource_pool_env_binding.py \
  -q
```

预期：PASS。

- [ ] **步骤 2：运行 Ruff**

运行：

```bash
ruff check \
  rlinf/utils/rollout_profile \
  rlinf/workers/env/env_worker.py \
  rlinf/workers/rollout/hf/huggingface_worker.py \
  rlinf/envs/robocasa/venv.py \
  tests/unit_tests/test_rollout_profiler.py \
  tests/unit_tests/test_rollout_profile_aggregation.py \
  tests/unit_tests/test_rollout_profile_worker_integration.py \
  tests/unit_tests/test_rollout_profile_robocasa_events.py \
  tools/summarize_rollout_profile.py
```

预期：PASS。

- [ ] **步骤 3：最小本地 smoke**

运行：

```bash
source /data1/miliang/RLinf/robocasa_openpi/bin/activate
python tools/summarize_rollout_profile.py /tmp/rlinf-empty-rollout-profile
```

如果目录不存在，实现应创建空 summary 或给出清晰错误。最终行为必须和任务 2 的测试一致。

- [ ] **步骤 4：可选 RoboCasa smoke**

资源允许时运行 1 step 小配置：

```bash
source /data1/miliang/RLinf/robocasa_openpi/bin/activate
export EMBODIED_PATH=/data1/miliang/RLinf/examples/embodiment
export ROBOT_PLATFORM=ROBOCASA
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
python examples/embodiment/train_embodied_agent.py \
  --config-path examples/embodiment/config \
  --config-name robocasa_baseline_profile_openpi \
  runner.max_steps=1 \
  runner.max_epochs=1 \
  runner.val_check_interval=-1 \
  runner.save_interval=-1 \
  runner.logger.logger_backends=[] \
  profiling.rollout.enabled=true \
  profiling.rollout.output_dir=/tmp/rlinf-robocasa-rollout-profile
```

预期：训练若环境和模型可用，应生成 `/tmp/rlinf-robocasa-rollout-profile/env_rank_*.jsonl` 和 `rollout_rank_*.jsonl`。如果模型或 RoboCasa 资源不可用，记录错误并保留单测作为验证证据。

- [ ] **步骤 5：最终 Commit**

若步骤 1 或 2 触发修正：

```bash
git add <fixed-files>
git commit -s -m "test: verify robocasa rollout profiler"
```

如果没有修正，不需要空提交。

## 自检

- 规格覆盖：计划覆盖可插拔模块、默认 no-op、Env/Rollout/RoboCasa 子进程事件、后处理 summary/timeline/report、配置示例和验证命令。
- 占位符扫描：计划不包含待定实现项；每个任务都有文件、测试、命令和预期结果。
- 类型一致性：统一使用 `RolloutProfilerConfig`、`RolloutProfiler`、`NoopRolloutProfiler`、`make_rollout_profiler`、`summarize_rollout_profile`。
