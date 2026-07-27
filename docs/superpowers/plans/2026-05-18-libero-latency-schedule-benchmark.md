# LIBERO Latency Schedule Benchmark 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 构建一个 toolkit benchmark，从 profiling CSV 抽样 LIBERO tasks，比较 task-id baseline、random baseline 和 latency-aware trapezoid pipeline 在真实并发 `env.step()` 下的吞吐与 idle/bubble。

**架构：** 新增一个独立 CLI 脚本 `toolkits/run_libero_latency_schedule_benchmark.py`，内部拆分为 CSV/task 数据、latency 估计、schedule plan、worker 运行、summary/report 写出几个小单元。默认单测使用 fake worker/env，不依赖 LIBERO；真实 benchmark 通过 CLI 手动运行。

**技术栈：** Python 标准库（`argparse`, `csv`, `json`, `multiprocessing`, `time`, `dataclasses`, `pathlib`）、`numpy`、`pytest`、现有 `toolkits.profile_libero_step_latency` 中的 LIBERO 导入和 task spec 构建 helper。

---

## 文件结构

- 创建：`toolkits/run_libero_latency_schedule_benchmark.py`
  - 负责 CLI、CSV 读取、任务抽样、latency 估计、schedule 构造、worker 协调、summary/report 输出。
  - 保持单文件是为了匹配现有 `toolkits/profile_libero_step_latency.py` 的 standalone toolkit 风格；内部用 dataclass 和纯函数维持边界。
- 创建：`tests/unit_tests/test_libero_latency_schedule_benchmark.py`
  - 覆盖不依赖 LIBERO 的纯函数、fake runner、CLI 输出路径。
- 修改：无现有生产代码修改。

## 任务 1：CSV 读取、抽样和 latency 估计

**文件：**
- 创建：`toolkits/run_libero_latency_schedule_benchmark.py`
- 测试：`tests/unit_tests/test_libero_latency_schedule_benchmark.py`

- [ ] **步骤 1：编写失败的 CSV/task 测试**

在 `tests/unit_tests/test_libero_latency_schedule_benchmark.py` 添加：

```python
import csv
import math
from pathlib import Path

import pytest

from toolkits.run_libero_latency_schedule_benchmark import (
    TaskRecord,
    estimate_latency_scores,
    load_task_records,
    sample_task_records,
)


def _write_task_csv(path: Path) -> None:
    rows = [
        {
            "task_id": "3",
            "task_name": "task_c",
            "mean_latency_ms": "13.0",
            "njnt": "12",
            "ngeom": "100",
            "scene_type": "study",
        },
        {
            "task_id": "1",
            "task_name": "task_a",
            "mean_latency_ms": "21.0",
            "njnt": "17",
            "ngeom": "200",
            "scene_type": "kitchen",
        },
        {
            "task_id": "2",
            "task_name": "task_b",
            "mean_latency_ms": "17.0",
            "njnt": "15",
            "ngeom": "150",
            "scene_type": "living_room",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_load_task_records_preserves_required_and_extra_fields(tmp_path: Path):
    csv_path = tmp_path / "tasks.csv"
    _write_task_csv(csv_path)

    records = load_task_records(csv_path)

    assert [record.task_id for record in records] == [3, 1, 2]
    assert records[1].task_name == "task_a"
    assert records[1].mean_latency_ms == 21.0
    assert records[1].njnt == 17
    assert records[1].ngeom == 200
    assert records[1].extra["scene_type"] == "kitchen"


def test_load_task_records_rejects_missing_required_columns(tmp_path: Path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("task_id,task_name,njnt,ngeom\n1,t,2,3\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        load_task_records(csv_path)


def test_sample_task_records_is_seeded_without_replacement(tmp_path: Path):
    csv_path = tmp_path / "tasks.csv"
    _write_task_csv(csv_path)
    records = load_task_records(csv_path)

    first = sample_task_records(records, num_envs=2, seed=123)
    second = sample_task_records(records, num_envs=2, seed=123)

    assert [record.task_id for record in first] == [record.task_id for record in second]
    assert len({record.task_id for record in first}) == 2


def test_sample_task_records_rejects_oversized_request(tmp_path: Path):
    csv_path = tmp_path / "tasks.csv"
    _write_task_csv(csv_path)
    records = load_task_records(csv_path)

    with pytest.raises(ValueError, match="num_envs"):
        sample_task_records(records, num_envs=4, seed=0)


def test_estimate_latency_scores_uses_z_scored_njnt_and_ngeom():
    records = [
        TaskRecord(task_id=0, task_name="low", mean_latency_ms=1.0, njnt=10, ngeom=100),
        TaskRecord(task_id=1, task_name="high", mean_latency_ms=2.0, njnt=20, ngeom=200),
    ]

    scored = estimate_latency_scores(records, weight_jnt=0.45, weight_geom=0.55)

    assert scored[1].estimated_latency_score > scored[0].estimated_latency_score
    assert math.isclose(
        sum(record.estimated_latency_score for record in scored),
        0.0,
        abs_tol=1e-12,
    )
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
pytest tests/unit_tests/test_libero_latency_schedule_benchmark.py -q
```

预期：FAIL，报错包含 `No module named 'toolkits.run_libero_latency_schedule_benchmark'`。

- [ ] **步骤 3：实现 CSV/task 基础代码**

创建 `toolkits/run_libero_latency_schedule_benchmark.py`，加入：

```python
"""Benchmark latency-aware LIBERO task scheduling with real env.step calls."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_TASK_COLUMNS = {"task_id", "task_name", "mean_latency_ms", "njnt", "ngeom"}


@dataclass(frozen=True)
class TaskRecord:
    task_id: int
    task_name: str
    mean_latency_ms: float
    njnt: int
    ngeom: int
    estimated_latency_score: float = 0.0
    extra: dict[str, str] = field(default_factory=dict)


def _parse_int(row: dict[str, str], key: str) -> int:
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer column {key!r}: {row.get(key)!r}") from exc


def _parse_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid float column {key!r}: {row.get(key)!r}") from exc


def load_task_records(path: str | Path) -> list[TaskRecord]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_TASK_COLUMNS - fieldnames)
        if missing:
            raise ValueError(f"missing required columns: {', '.join(missing)}")
        records = []
        for row in reader:
            extra = {
                key: value
                for key, value in row.items()
                if key not in REQUIRED_TASK_COLUMNS and value is not None
            }
            records.append(
                TaskRecord(
                    task_id=_parse_int(row, "task_id"),
                    task_name=row["task_name"],
                    mean_latency_ms=_parse_float(row, "mean_latency_ms"),
                    njnt=_parse_int(row, "njnt"),
                    ngeom=_parse_int(row, "ngeom"),
                    extra=extra,
                )
            )
    return records


def sample_task_records(
    records: list[TaskRecord],
    *,
    num_envs: int,
    seed: int,
) -> list[TaskRecord]:
    if num_envs < 1:
        raise ValueError("num_envs must be >= 1")
    if num_envs > len(records):
        raise ValueError(
            f"num_envs={num_envs} exceeds available task records={len(records)}"
        )
    rng = random.Random(seed)
    return rng.sample(records, num_envs)


def _z_scores(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    std = float(np.std(array))
    if std == 0.0:
        return [0.0 for _ in values]
    mean = float(np.mean(array))
    return [float((value - mean) / std) for value in array]


def estimate_latency_scores(
    records: list[TaskRecord],
    *,
    weight_jnt: float = 0.45,
    weight_geom: float = 0.55,
) -> list[TaskRecord]:
    jnt_scores = _z_scores([float(record.njnt) for record in records])
    geom_scores = _z_scores([float(record.ngeom) for record in records])
    scored = []
    for record, jnt_score, geom_score in zip(records, jnt_scores, geom_scores):
        score = weight_jnt * jnt_score + weight_geom * geom_score
        scored.append(replace(record, estimated_latency_score=float(score)))
    return scored
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
pytest tests/unit_tests/test_libero_latency_schedule_benchmark.py -q
```

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add toolkits/run_libero_latency_schedule_benchmark.py tests/unit_tests/test_libero_latency_schedule_benchmark.py
git commit -s -m "feat: add libero latency benchmark task loading"
```

## 任务 2：schedule plan 构造

**文件：**
- 修改：`toolkits/run_libero_latency_schedule_benchmark.py`
- 修改：`tests/unit_tests/test_libero_latency_schedule_benchmark.py`

- [ ] **步骤 1：编写失败的 schedule plan 测试**

在测试文件追加：

```python
from toolkits.run_libero_latency_schedule_benchmark import (
    ScheduleItem,
    build_random_baseline_plan,
    build_task_id_baseline_plan,
    build_trapezoid_pipeline_plan,
)


def _records_for_schedule() -> list[TaskRecord]:
    return [
        TaskRecord(task_id=4, task_name="t4", mean_latency_ms=4.0, njnt=14, ngeom=140, estimated_latency_score=4.0),
        TaskRecord(task_id=1, task_name="t1", mean_latency_ms=1.0, njnt=11, ngeom=110, estimated_latency_score=1.0),
        TaskRecord(task_id=3, task_name="t3", mean_latency_ms=3.0, njnt=13, ngeom=130, estimated_latency_score=3.0),
        TaskRecord(task_id=2, task_name="t2", mean_latency_ms=2.0, njnt=12, ngeom=120, estimated_latency_score=2.0),
    ]


def test_task_id_baseline_assigns_sorted_tasks_to_core_columns():
    plan = build_task_id_baseline_plan(_records_for_schedule(), cpu_ids=[10, 11])

    assert [item.task.task_id for item in plan] == [1, 2, 3, 4]
    assert [(item.core_index, item.cpu_id, item.layer_index) for item in plan] == [
        (0, 10, 0),
        (1, 11, 0),
        (0, 10, 1),
        (1, 11, 1),
    ]
    assert {item.schedule_name for item in plan} == {"task_id_baseline"}


def test_random_baseline_is_seeded_and_static():
    first = build_random_baseline_plan(_records_for_schedule(), cpu_ids=[0, 1], seed=7)
    second = build_random_baseline_plan(_records_for_schedule(), cpu_ids=[0, 1], seed=7)

    assert [item.task.task_id for item in first] == [item.task.task_id for item in second]
    assert {item.schedule_name for item in first} == {"random_baseline"}


def test_trapezoid_pipeline_keeps_long_short_pairs_on_same_core():
    plan = build_trapezoid_pipeline_plan(_records_for_schedule(), cpu_ids=[0, 1])

    long_items = [item for item in plan if item.side == "long"]
    short_items = [item for item in plan if item.side == "short"]
    assert [item.task.task_id for item in long_items] == [4, 3]
    assert [item.task.task_id for item in short_items] == [1, 2]
    assert [item.core_index for item in long_items] == [0, 1]
    assert [item.core_index for item in short_items] == [0, 1]
    assert {item.schedule_name for item in plan} == {"trapezoid_pipeline"}


def test_trapezoid_pipeline_rejects_odd_task_count():
    records = _records_for_schedule()[:3]

    with pytest.raises(ValueError, match="even"):
        build_trapezoid_pipeline_plan(records, cpu_ids=[0, 1])
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
pytest tests/unit_tests/test_libero_latency_schedule_benchmark.py -q
```

预期：FAIL，报错包含 `cannot import name 'ScheduleItem'`。

- [ ] **步骤 3：实现 schedule plan**

在 `toolkits/run_libero_latency_schedule_benchmark.py` 追加：

```python
TASK_ID_BASELINE = "task_id_baseline"
RANDOM_BASELINE = "random_baseline"
TRAPEZOID_PIPELINE = "trapezoid_pipeline"


@dataclass(frozen=True)
class ScheduleItem:
    schedule_name: str
    task: TaskRecord
    core_index: int
    cpu_id: int
    layer_index: int
    order_index: int
    side: str = "baseline"


def _require_cpu_ids(cpu_ids: list[int]) -> None:
    if not cpu_ids:
        raise ValueError("cpu_ids must not be empty")


def _layered_plan(
    records: list[TaskRecord],
    *,
    cpu_ids: list[int],
    schedule_name: str,
    side: str = "baseline",
    order_offset: int = 0,
) -> list[ScheduleItem]:
    _require_cpu_ids(cpu_ids)
    items = []
    for index, record in enumerate(records):
        core_index = index % len(cpu_ids)
        layer_index = index // len(cpu_ids)
        items.append(
            ScheduleItem(
                schedule_name=schedule_name,
                task=record,
                core_index=core_index,
                cpu_id=cpu_ids[core_index],
                layer_index=layer_index,
                order_index=order_offset + index,
                side=side,
            )
        )
    return items


def build_task_id_baseline_plan(
    records: list[TaskRecord],
    *,
    cpu_ids: list[int],
) -> list[ScheduleItem]:
    ordered = sorted(records, key=lambda record: record.task_id)
    return _layered_plan(
        ordered,
        cpu_ids=cpu_ids,
        schedule_name=TASK_ID_BASELINE,
    )


def build_random_baseline_plan(
    records: list[TaskRecord],
    *,
    cpu_ids: list[int],
    seed: int,
) -> list[ScheduleItem]:
    ordered = list(records)
    random.Random(seed).shuffle(ordered)
    return _layered_plan(
        ordered,
        cpu_ids=cpu_ids,
        schedule_name=RANDOM_BASELINE,
    )


def build_trapezoid_pipeline_plan(
    records: list[TaskRecord],
    *,
    cpu_ids: list[int],
) -> list[ScheduleItem]:
    _require_cpu_ids(cpu_ids)
    if len(records) % 2 != 0:
        raise ValueError("trapezoid_pipeline requires an even number of tasks")
    ordered = sorted(
        records,
        key=lambda record: (-record.estimated_latency_score, record.task_id),
    )
    split = len(ordered) // 2
    long_half = ordered[:split]
    short_half = list(reversed(ordered[split:]))
    long_items = _layered_plan(
        long_half,
        cpu_ids=cpu_ids,
        schedule_name=TRAPEZOID_PIPELINE,
        side="long",
    )
    short_items = _layered_plan(
        short_half,
        cpu_ids=cpu_ids,
        schedule_name=TRAPEZOID_PIPELINE,
        side="short",
        order_offset=len(long_items),
    )
    return long_items + short_items
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
pytest tests/unit_tests/test_libero_latency_schedule_benchmark.py -q
```

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add toolkits/run_libero_latency_schedule_benchmark.py tests/unit_tests/test_libero_latency_schedule_benchmark.py
git commit -s -m "feat: add libero latency schedule plans"
```

## 任务 3：summary 指标与 fake 并发 runner

**文件：**
- 修改：`toolkits/run_libero_latency_schedule_benchmark.py`
- 修改：`tests/unit_tests/test_libero_latency_schedule_benchmark.py`

- [ ] **步骤 1：编写失败的 runner/summary 测试**

在测试文件追加：

```python
from toolkits.run_libero_latency_schedule_benchmark import (
    StepEvent,
    compute_schedule_summary,
    run_schedule_with_step_function,
)


def test_run_schedule_with_step_function_completes_equal_steps_per_task():
    records = _records_for_schedule()
    plan = build_task_id_baseline_plan(records, cpu_ids=[0, 1])
    latency_by_task = {1: 0.01, 2: 0.02, 3: 0.03, 4: 0.04}

    events = run_schedule_with_step_function(
        plan,
        steps_per_env=2,
        step_fn=lambda item, step_index: latency_by_task[item.task.task_id],
    )

    assert len(events) == 8
    counts = {}
    for event in events:
        counts[event.task_id] = counts.get(event.task_id, 0) + 1
    assert counts == {1: 2, 2: 2, 3: 2, 4: 2}
    assert all(event.round_wall_time_s >= event.latency_s for event in events)


def test_compute_schedule_summary_reports_throughput_and_idle():
    events = [
        StepEvent(
            schedule_name="s",
            round_index=0,
            core_index=0,
            cpu_id=0,
            task_id=1,
            task_name="a",
            task_step_index=0,
            latency_s=0.01,
            round_wall_time_s=0.02,
            idle_time_s=0.01,
            cpu_affinity_applied=True,
        ),
        StepEvent(
            schedule_name="s",
            round_index=0,
            core_index=1,
            cpu_id=1,
            task_id=2,
            task_name="b",
            task_step_index=0,
            latency_s=0.02,
            round_wall_time_s=0.02,
            idle_time_s=0.0,
            cpu_affinity_applied=True,
        ),
    ]

    summary = compute_schedule_summary("s", events)

    assert summary["schedule_name"] == "s"
    assert summary["status"] == "completed"
    assert summary["total_steps"] == 2
    assert summary["makespan_s"] == 0.02
    assert summary["steps_per_second"] == 100.0
    assert summary["mean_core_idle_ratio"] == 0.25
    assert summary["cpu_affinity_success_rate"] == 1.0


def test_compute_schedule_summary_marks_degraded_affinity():
    event = StepEvent(
        schedule_name="s",
        round_index=0,
        core_index=0,
        cpu_id=0,
        task_id=1,
        task_name="a",
        task_step_index=0,
        latency_s=0.01,
        round_wall_time_s=0.01,
        idle_time_s=0.0,
        cpu_affinity_applied=False,
    )

    summary = compute_schedule_summary("s", [event])

    assert summary["status"] == "degraded"
    assert summary["cpu_affinity_success_rate"] == 0.0
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
pytest tests/unit_tests/test_libero_latency_schedule_benchmark.py -q
```

预期：FAIL，报错包含 `cannot import name 'StepEvent'`。

- [ ] **步骤 3：实现 fake runner 和 summary**

在 `toolkits/run_libero_latency_schedule_benchmark.py` 追加：

```python
@dataclass(frozen=True)
class StepEvent:
    schedule_name: str
    round_index: int
    core_index: int
    cpu_id: int
    task_id: int
    task_name: str
    task_step_index: int
    latency_s: float
    round_wall_time_s: float
    idle_time_s: float
    cpu_affinity_applied: bool


def _items_by_core(plan: list[ScheduleItem]) -> dict[int, list[ScheduleItem]]:
    grouped: dict[int, list[ScheduleItem]] = {}
    for item in sorted(plan, key=lambda value: (value.core_index, value.order_index)):
        grouped.setdefault(item.core_index, []).append(item)
    return grouped


def _next_item_for_core(
    items: list[ScheduleItem],
    per_task_counts: dict[int, int],
    *,
    steps_per_env: int,
    cursor: int,
) -> tuple[ScheduleItem | None, int]:
    if not items:
        return None, cursor
    for offset in range(len(items)):
        index = (cursor + offset) % len(items)
        item = items[index]
        if per_task_counts.get(item.task.task_id, 0) < steps_per_env:
            return item, (index + 1) % len(items)
    return None, cursor


def run_schedule_with_step_function(
    plan: list[ScheduleItem],
    *,
    steps_per_env: int,
    step_fn: Any,
    cpu_affinity_by_core: dict[int, bool] | None = None,
) -> list[StepEvent]:
    if steps_per_env < 1:
        raise ValueError("steps_per_env must be >= 1")
    grouped = _items_by_core(plan)
    per_task_counts = {item.task.task_id: 0 for item in plan}
    cursors = {core_index: 0 for core_index in grouped}
    events: list[StepEvent] = []
    round_index = 0
    while any(count < steps_per_env for count in per_task_counts.values()):
        round_results = []
        for core_index, items in grouped.items():
            item, next_cursor = _next_item_for_core(
                items,
                per_task_counts,
                steps_per_env=steps_per_env,
                cursor=cursors[core_index],
            )
            cursors[core_index] = next_cursor
            if item is None:
                continue
            task_step_index = per_task_counts[item.task.task_id]
            latency_s = float(step_fn(item, task_step_index))
            per_task_counts[item.task.task_id] = task_step_index + 1
            round_results.append((item, task_step_index, max(latency_s, 0.0)))
        if not round_results:
            break
        round_wall_time_s = max(latency for _, _, latency in round_results)
        for item, task_step_index, latency_s in round_results:
            affinity = True
            if cpu_affinity_by_core is not None:
                affinity = bool(cpu_affinity_by_core.get(item.core_index, True))
            events.append(
                StepEvent(
                    schedule_name=item.schedule_name,
                    round_index=round_index,
                    core_index=item.core_index,
                    cpu_id=item.cpu_id,
                    task_id=item.task.task_id,
                    task_name=item.task.task_name,
                    task_step_index=task_step_index,
                    latency_s=latency_s,
                    round_wall_time_s=round_wall_time_s,
                    idle_time_s=max(round_wall_time_s - latency_s, 0.0),
                    cpu_affinity_applied=affinity,
                )
            )
        round_index += 1
    return events


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def compute_schedule_summary(
    schedule_name: str,
    events: list[StepEvent],
    *,
    failed: bool = False,
) -> dict[str, Any]:
    if failed:
        status = "failed"
    elif any(not event.cpu_affinity_applied for event in events):
        status = "degraded"
    else:
        status = "completed"
    if not events:
        return {
            "schedule_name": schedule_name,
            "status": status,
            "total_steps": 0,
            "makespan_s": 0.0,
            "steps_per_second": 0.0,
            "mean_core_idle_ratio": None,
            "cpu_affinity_success_rate": None,
        }
    round_wall_times = {
        event.round_index: event.round_wall_time_s for event in events
    }
    makespan_s = float(sum(round_wall_times.values()))
    latencies = [event.latency_s for event in events]
    idle_ratios = [
        0.0
        if event.round_wall_time_s == 0.0
        else event.idle_time_s / event.round_wall_time_s
        for event in events
    ]
    affinity_rate = sum(1 for event in events if event.cpu_affinity_applied) / len(events)
    return {
        "schedule_name": schedule_name,
        "status": status,
        "total_steps": len(events),
        "makespan_s": makespan_s,
        "steps_per_second": 0.0 if makespan_s == 0.0 else len(events) / makespan_s,
        "mean_step_latency_s": float(np.mean(np.asarray(latencies))),
        "median_step_latency_s": float(np.median(np.asarray(latencies))),
        "p90_step_latency_s": _percentile(latencies, 90),
        "p95_step_latency_s": _percentile(latencies, 95),
        "p99_step_latency_s": _percentile(latencies, 99),
        "mean_core_idle_ratio": float(np.mean(np.asarray(idle_ratios))),
        "p90_round_idle_ratio": _percentile(idle_ratios, 90),
        "p99_round_idle_ratio": _percentile(idle_ratios, 99),
        "cpu_affinity_success_rate": float(affinity_rate),
    }
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
pytest tests/unit_tests/test_libero_latency_schedule_benchmark.py -q
```

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add toolkits/run_libero_latency_schedule_benchmark.py tests/unit_tests/test_libero_latency_schedule_benchmark.py
git commit -s -m "feat: add libero latency benchmark metrics"
```

## 任务 4：runner 结果对象和 fake 同步执行路径

**文件：**
- 修改：`toolkits/run_libero_latency_schedule_benchmark.py`
- 修改：`tests/unit_tests/test_libero_latency_schedule_benchmark.py`

- [ ] **步骤 1：编写失败的 BenchmarkRunner 测试**

在测试文件追加：

```python
from toolkits.run_libero_latency_schedule_benchmark import BenchmarkRunner


def test_benchmark_runner_uses_injected_step_function_for_unit_tests():
    records = _records_for_schedule()
    plan = build_task_id_baseline_plan(records, cpu_ids=[0, 1])
    runner = BenchmarkRunner(
        steps_per_env=1,
        step_fn=lambda item, step_index: 0.01 * item.task.task_id,
    )

    result = runner.run("task_id_baseline", plan)

    assert result.summary["schedule_name"] == "task_id_baseline"
    assert result.summary["total_steps"] == 4
    assert len(result.events) == 4
    assert result.errors == []


def test_benchmark_runner_reports_failed_schedule_from_step_exception():
    records = _records_for_schedule()
    plan = build_task_id_baseline_plan(records, cpu_ids=[0, 1])

    def fail_on_task(item: ScheduleItem, step_index: int) -> float:
        if item.task.task_id == 2:
            raise RuntimeError("step failed")
        return 0.01

    runner = BenchmarkRunner(steps_per_env=1, step_fn=fail_on_task)

    result = runner.run("task_id_baseline", plan)

    assert result.summary["status"] == "failed"
    assert result.errors
    assert "step failed" in result.errors[0]["error"]
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
pytest tests/unit_tests/test_libero_latency_schedule_benchmark.py -q
```

预期：FAIL，报错包含 `cannot import name 'BenchmarkRunner'`。

- [ ] **步骤 3：实现 BenchmarkRunner 注入路径**

在 `toolkits/run_libero_latency_schedule_benchmark.py` 追加：

```python
@dataclass(frozen=True)
class BenchmarkResult:
    schedule_name: str
    events: list[StepEvent]
    summary: dict[str, Any]
    errors: list[dict[str, Any]]


class BenchmarkRunner:
    def __init__(
        self,
        *,
        steps_per_env: int,
        step_fn: Any | None = None,
    ) -> None:
        self.steps_per_env = steps_per_env
        self.step_fn = step_fn

    def run(self, schedule_name: str, plan: list[ScheduleItem]) -> BenchmarkResult:
        if self.step_fn is None:
            raise RuntimeError("step_fn is required until process worker support is added in task 5")
        try:
            events = run_schedule_with_step_function(
                plan,
                steps_per_env=self.steps_per_env,
                step_fn=self.step_fn,
            )
            summary = compute_schedule_summary(schedule_name, events)
            return BenchmarkResult(
                schedule_name=schedule_name,
                events=events,
                summary=summary,
                errors=[],
            )
        except Exception as exc:
            summary = compute_schedule_summary(schedule_name, [], failed=True)
            return BenchmarkResult(
                schedule_name=schedule_name,
                events=[],
                summary=summary,
                errors=[
                    {
                        "event": "error",
                        "schedule_name": schedule_name,
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                    }
                ],
            )
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
pytest tests/unit_tests/test_libero_latency_schedule_benchmark.py -q
```

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add toolkits/run_libero_latency_schedule_benchmark.py tests/unit_tests/test_libero_latency_schedule_benchmark.py
git commit -s -m "feat: add libero latency benchmark runner"
```

## 任务 5：真实 multiprocessing worker 与主进程 barrier

**文件：**
- 修改：`toolkits/run_libero_latency_schedule_benchmark.py`
- 修改：`tests/unit_tests/test_libero_latency_schedule_benchmark.py`

- [ ] **步骤 1：编写失败的 multiprocessing worker 计划测试**

在测试文件追加：

```python
from toolkits.run_libero_latency_schedule_benchmark import (
    apply_cpu_affinity,
    build_worker_plans,
    run_schedule_with_process_workers,
)


class FakeProcessEnv:
    def __init__(self, latency_s: float):
        self.latency_s = latency_s

    def step(self, action):
        del action
        return {}, 0.0, False, {}


def fake_process_env_factory(item: ScheduleItem):
    return FakeProcessEnv(latency_s=item.task.mean_latency_ms / 1000.0)


def test_build_worker_plans_groups_items_by_core():
    records = _records_for_schedule()
    plan = build_task_id_baseline_plan(records, cpu_ids=[10, 11])

    worker_plans = build_worker_plans(plan)

    assert sorted(worker_plans) == [0, 1]
    assert [item.cpu_id for item in worker_plans[0]] == [10, 10]
    assert [item.cpu_id for item in worker_plans[1]] == [11, 11]


def test_run_schedule_with_process_workers_completes_equal_steps_per_task():
    records = _records_for_schedule()
    plan = build_task_id_baseline_plan(records, cpu_ids=[0, 1])

    result = run_schedule_with_process_workers(
        plan,
        steps_per_env=1,
        env_factory=fake_process_env_factory,
        dummy_action=[0.0] * 7,
        subprocess_timeout_s=10.0,
    )

    assert result.errors == []
    assert len(result.events) == 4
    assert {event.task_id for event in result.events} == {1, 2, 3, 4}
    assert all(event.round_wall_time_s >= event.latency_s for event in result.events)


def test_apply_cpu_affinity_returns_false_when_affinity_unavailable(monkeypatch):
    monkeypatch.delattr("os.sched_setaffinity", raising=False)

    assert apply_cpu_affinity(0) is False
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
pytest tests/unit_tests/test_libero_latency_schedule_benchmark.py -q
```

预期：FAIL，报错包含 `cannot import name 'run_schedule_with_process_workers'`。

- [ ] **步骤 3：实现 CPU affinity、worker plan 和 worker result 类型**

在 `toolkits/run_libero_latency_schedule_benchmark.py` 增加 imports：

```python
import multiprocessing as mp
import os
import time
```

追加：

```python
@dataclass(frozen=True)
class ProcessRunResult:
    events: list[StepEvent]
    errors: list[dict[str, Any]]


def apply_cpu_affinity(cpu_id: int | None) -> bool:
    if cpu_id is None or not hasattr(os, "sched_setaffinity"):
        return False
    try:
        os.sched_setaffinity(0, {cpu_id})
    except OSError:
        return False
    return True


def build_worker_plans(plan: list[ScheduleItem]) -> dict[int, list[ScheduleItem]]:
    return _items_by_core(plan)
```

- [ ] **步骤 4：实现 worker entry 和 process runner**

在 `toolkits/run_libero_latency_schedule_benchmark.py` 追加：

```python
def _worker_loop(
    *,
    core_index: int,
    items: list[ScheduleItem],
    steps_per_env: int,
    env_factory: Any,
    dummy_action: list[float],
    command_queue: Any,
    result_queue: Any,
) -> None:
    cpu_id = items[0].cpu_id if items else None
    affinity_applied = apply_cpu_affinity(cpu_id)
    envs: dict[int, Any] = {}
    task_counts = {item.task.task_id: 0 for item in items}
    try:
        for item in items:
            if item.task.task_id not in envs:
                envs[item.task.task_id] = env_factory(item)
        while True:
            command = command_queue.get()
            if command == "stop":
                break
            round_index, item = command
            env = envs[item.task.task_id]
            task_step_index = task_counts[item.task.task_id]
            start = time.perf_counter()
            env.step(np.asarray(dummy_action, dtype=np.float32))
            latency_s = max(float(time.perf_counter() - start), 0.0)
            task_counts[item.task.task_id] = task_step_index + 1
            result_queue.put(
                {
                    "event": "step",
                    "round_index": round_index,
                    "core_index": core_index,
                    "cpu_id": cpu_id,
                    "task_id": item.task.task_id,
                    "task_name": item.task.task_name,
                    "task_step_index": task_step_index,
                    "latency_s": latency_s,
                    "cpu_affinity_applied": affinity_applied,
                }
            )
    except Exception as exc:
        result_queue.put(
            {
                "event": "error",
                "core_index": core_index,
                "cpu_id": cpu_id,
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            }
        )
    finally:
        for env in envs.values():
            close = getattr(env, "close", None)
            if close is not None:
                close()


def _next_round_commands(
    worker_plans: dict[int, list[ScheduleItem]],
    per_task_counts: dict[int, int],
    cursors: dict[int, int],
    *,
    steps_per_env: int,
) -> list[tuple[int, ScheduleItem]]:
    commands = []
    for core_index, items in worker_plans.items():
        item, next_cursor = _next_item_for_core(
            items,
            per_task_counts,
            steps_per_env=steps_per_env,
            cursor=cursors[core_index],
        )
        cursors[core_index] = next_cursor
        if item is not None:
            commands.append((core_index, item))
    return commands


def run_schedule_with_process_workers(
    plan: list[ScheduleItem],
    *,
    steps_per_env: int,
    env_factory: Any,
    dummy_action: list[float],
    subprocess_timeout_s: float = 300.0,
) -> ProcessRunResult:
    worker_plans = build_worker_plans(plan)
    task_ids = [item.task.task_id for item in plan]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("schedule plan must contain each task_id once")
    per_task_counts = {item.task.task_id: 0 for item in plan}
    cursors = {core_index: 0 for core_index in worker_plans}
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    command_queues = {core_index: ctx.Queue() for core_index in worker_plans}
    processes = []
    for core_index, items in worker_plans.items():
        process = ctx.Process(
            target=_worker_loop,
            kwargs={
                "core_index": core_index,
                "items": items,
                "steps_per_env": steps_per_env,
                "env_factory": env_factory,
                "dummy_action": dummy_action,
                "command_queue": command_queues[core_index],
                "result_queue": result_queue,
            },
        )
        process.start()
        processes.append(process)
    events: list[StepEvent] = []
    errors: list[dict[str, Any]] = []
    try:
        round_index = 0
        while any(count < steps_per_env for count in per_task_counts.values()):
            commands = _next_round_commands(
                worker_plans,
                per_task_counts,
                cursors,
                steps_per_env=steps_per_env,
            )
            if not commands:
                break
            for core_index, item in commands:
                command_queues[core_index].put((round_index, item))
            raw_results = []
            deadline = time.monotonic() + subprocess_timeout_s
            while len(raw_results) < len(commands):
                remaining = max(deadline - time.monotonic(), 0.0)
                if remaining == 0.0:
                    errors.append(
                        {
                            "event": "error",
                            "schedule_name": plan[0].schedule_name,
                            "error_type": "TimeoutError",
                            "error": "worker round timed out",
                            "round_index": round_index,
                        }
                    )
                    return ProcessRunResult(events=events, errors=errors)
                result = result_queue.get(timeout=remaining)
                if result.get("event") == "error":
                    errors.append(result)
                    return ProcessRunResult(events=events, errors=errors)
                raw_results.append(result)
            round_wall_time_s = max(float(item["latency_s"]) for item in raw_results)
            for result in raw_results:
                per_task_counts[int(result["task_id"])] += 1
                latency_s = float(result["latency_s"])
                events.append(
                    StepEvent(
                        schedule_name=plan[0].schedule_name,
                        round_index=round_index,
                        core_index=int(result["core_index"]),
                        cpu_id=int(result["cpu_id"]),
                        task_id=int(result["task_id"]),
                        task_name=str(result["task_name"]),
                        task_step_index=int(result["task_step_index"]),
                        latency_s=latency_s,
                        round_wall_time_s=round_wall_time_s,
                        idle_time_s=max(round_wall_time_s - latency_s, 0.0),
                        cpu_affinity_applied=bool(result["cpu_affinity_applied"]),
                    )
                )
            round_index += 1
        return ProcessRunResult(events=events, errors=errors)
    finally:
        for queue in command_queues.values():
            queue.put("stop")
        for process in processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
```

- [ ] **步骤 5：运行 worker 测试验证通过**

运行：

```bash
pytest tests/unit_tests/test_libero_latency_schedule_benchmark.py::test_build_worker_plans_groups_items_by_core tests/unit_tests/test_libero_latency_schedule_benchmark.py::test_run_schedule_with_process_workers_completes_equal_steps_per_task -q
```

预期：PASS。

- [ ] **步骤 6：实现真实 LIBERO env factory**

在 `toolkits/run_libero_latency_schedule_benchmark.py` 追加：

```python
def make_libero_env_factory(
    *,
    suite: str,
    camera_height: int,
    camera_width: int,
    libero_type: str,
    seed: int,
    warmup_steps: int,
    dummy_action: list[float],
) -> Any:
    from toolkits.profile_libero_step_latency import (
        ProfileConfig,
        build_task_trial_specs,
        make_libero_env_factory as make_profile_env_factory,
    )

    task_spec_cache: dict[int, Any] = {}
    init_state_cache: dict[int, Any] = {}

    def env_factory(item: ScheduleItem) -> Any:
        if item.task.task_id not in task_spec_cache:
            config = ProfileConfig(
                suite=suite,
                task_ids=str(item.task.task_id),
                trials_per_task=1,
                specific_trial_ids=None,
                warmup_steps=warmup_steps,
                measure_steps=1,
                cpu_id=item.cpu_id,
                cpu_ids=None,
                camera_height=camera_height,
                camera_width=camera_width,
                libero_type=libero_type,
                seed=seed,
                output_dir=Path("."),
                dummy_action=dummy_action,
                stop_on_done=False,
                subprocess_timeout_s=None,
                mujoco_profiler=False,
            )
            specs, init_states = build_task_trial_specs(config)
            task_spec_cache[item.task.task_id] = (config, specs[0])
            init_state_cache[item.task.task_id] = init_states[0]
        config, spec = task_spec_cache[item.task.task_id]
        env = make_profile_env_factory(config, spec)()
        if hasattr(env, "seed"):
            env.seed(spec.seed)
        env.reset()
        init_state = init_state_cache[item.task.task_id]
        if init_state is not None and hasattr(env, "set_init_state"):
            env.set_init_state(init_state)
        for _ in range(warmup_steps):
            env.step(np.asarray(dummy_action, dtype=np.float32))
        env.reset()
        if init_state is not None and hasattr(env, "set_init_state"):
            env.set_init_state(init_state)
        return env

    return env_factory
```

- [ ] **步骤 7：修改 BenchmarkRunner，让真实模式使用 process workers**

替换 `BenchmarkRunner` 为：

```python
class BenchmarkRunner:
    def __init__(
        self,
        *,
        steps_per_env: int,
        step_fn: Any | None = None,
        env_factory: Any | None = None,
        dummy_action: list[float] | None = None,
        subprocess_timeout_s: float = 300.0,
    ) -> None:
        self.steps_per_env = steps_per_env
        self.step_fn = step_fn
        self.env_factory = env_factory
        self.dummy_action = list(dummy_action or [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
        self.subprocess_timeout_s = subprocess_timeout_s

    def run(self, schedule_name: str, plan: list[ScheduleItem]) -> BenchmarkResult:
        try:
            if self.step_fn is not None:
                events = run_schedule_with_step_function(
                    plan,
                    steps_per_env=self.steps_per_env,
                    step_fn=self.step_fn,
                )
                errors: list[dict[str, Any]] = []
            else:
                if self.env_factory is None:
                    raise RuntimeError("env_factory is required for real worker mode")
                process_result = run_schedule_with_process_workers(
                    plan,
                    steps_per_env=self.steps_per_env,
                    env_factory=self.env_factory,
                    dummy_action=self.dummy_action,
                    subprocess_timeout_s=self.subprocess_timeout_s,
                )
                events = process_result.events
                errors = process_result.errors
            summary = compute_schedule_summary(
                schedule_name,
                events,
                failed=bool(errors),
            )
            return BenchmarkResult(
                schedule_name=schedule_name,
                events=events,
                summary=summary,
                errors=errors,
            )
        except Exception as exc:
            summary = compute_schedule_summary(schedule_name, [], failed=True)
            return BenchmarkResult(
                schedule_name=schedule_name,
                events=[],
                summary=summary,
                errors=[
                    {
                        "event": "error",
                        "schedule_name": schedule_name,
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                    }
                ],
            )
```

- [ ] **步骤 8：运行测试验证通过**

运行：

```bash
pytest tests/unit_tests/test_libero_latency_schedule_benchmark.py -q
```

预期：PASS。

- [ ] **步骤 9：Commit**

```bash
git add toolkits/run_libero_latency_schedule_benchmark.py tests/unit_tests/test_libero_latency_schedule_benchmark.py
git commit -s -m "feat: add libero latency benchmark process workers"
```

## 任务 6：CLI、文件输出和 comparison report

**文件：**
- 修改：`toolkits/run_libero_latency_schedule_benchmark.py`
- 修改：`tests/unit_tests/test_libero_latency_schedule_benchmark.py`

- [ ] **步骤 1：编写失败的 CLI 输出测试**

在测试文件追加：

```python
import json

from toolkits.run_libero_latency_schedule_benchmark import main


def test_main_fake_mode_writes_outputs(tmp_path: Path):
    csv_path = tmp_path / "tasks.csv"
    _write_task_csv(csv_path)
    output_dir = tmp_path / "out"

    exit_code = main(
        [
            "--task-csv",
            str(csv_path),
            "--num-envs",
            "2",
            "--cpu-ids",
            "0,1",
            "--steps-per-env",
            "2",
            "--output-dir",
            str(output_dir),
            "--fake-latency-from-csv",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "run_config.json").exists()
    assert (output_dir / "selected_tasks.csv").exists()
    assert (output_dir / "schedule_plan_task_id_baseline.csv").exists()
    assert (output_dir / "step_events_task_id_baseline.jsonl").exists()
    assert (output_dir / "schedule_summary.csv").exists()
    assert (output_dir / "schedule_summary.json").exists()
    assert (output_dir / "comparison_report.md").exists()
    summaries = json.loads((output_dir / "schedule_summary.json").read_text())
    assert {item["schedule_name"] for item in summaries} >= {
        "task_id_baseline",
        "trapezoid_pipeline",
    }
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
pytest tests/unit_tests/test_libero_latency_schedule_benchmark.py::test_main_fake_mode_writes_outputs -q
```

预期：FAIL，报错包含 `cannot import name 'main'`。

- [ ] **步骤 3：实现 serialization helpers**

在 `toolkits/run_libero_latency_schedule_benchmark.py` 增加 imports：

```python
import argparse
import json
from dataclasses import asdict
```

追加：

```python
def parse_int_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("integer list must not be empty")
    return [int(item) for item in items]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def write_selected_tasks(path: Path, records: list[TaskRecord]) -> None:
    fieldnames = [
        "task_id",
        "task_name",
        "mean_latency_ms",
        "njnt",
        "ngeom",
        "estimated_latency_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: getattr(record, key) for key in fieldnames})


def write_schedule_plan(path: Path, plan: list[ScheduleItem]) -> None:
    fieldnames = [
        "schedule_name",
        "order_index",
        "core_index",
        "cpu_id",
        "layer_index",
        "side",
        "task_id",
        "task_name",
        "estimated_latency_score",
        "mean_latency_ms",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in plan:
            writer.writerow(
                {
                    "schedule_name": item.schedule_name,
                    "order_index": item.order_index,
                    "core_index": item.core_index,
                    "cpu_id": item.cpu_id,
                    "layer_index": item.layer_index,
                    "side": item.side,
                    "task_id": item.task.task_id,
                    "task_name": item.task.task_name,
                    "estimated_latency_score": item.task.estimated_latency_score,
                    "mean_latency_ms": item.task.mean_latency_ms,
                }
            )


def write_step_events(path: Path, events: list[StepEvent]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(asdict(event), sort_keys=True) + "\n")


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for summary in summaries for key in summary})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def write_comparison_report(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = ["# LIBERO Latency Schedule Benchmark", ""]
    lines.append("| schedule | status | steps/sec | idle ratio |")
    lines.append("|---|---:|---:|---:|")
    for summary in summaries:
        lines.append(
            "| {schedule} | {status} | {sps:.6f} | {idle} |".format(
                schedule=summary["schedule_name"],
                status=summary["status"],
                sps=float(summary.get("steps_per_second") or 0.0),
                idle=summary.get("mean_core_idle_ratio"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
```

- [ ] **步骤 4：实现 CLI main**

在 `toolkits/run_libero_latency_schedule_benchmark.py` 追加：

```python
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-csv", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu-ids", required=True)
    parser.add_argument("--steps-per-env", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-baseline-repeats", type=int, default=1)
    parser.add_argument("--suite", default="libero_90")
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--libero-type", choices=["standard", "pro", "plus"], default="standard")
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--subprocess-timeout-s", type=float, default=300.0)
    parser.add_argument("--dummy-action", default="0,0,0,0,0,0,-1")
    parser.add_argument(
        "--fake-latency-from-csv",
        action="store_true",
        help="Use CSV mean_latency_ms as fake latency for unit/local smoke tests.",
    )
    return parser


def _dummy_action_from_arg(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _fake_step_fn(item: ScheduleItem, step_index: int) -> float:
    del step_index
    return item.task.mean_latency_ms / 1000.0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cpu_ids = parse_int_list(args.cpu_ids)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    records = estimate_latency_scores(
        sample_task_records(
            load_task_records(args.task_csv),
            num_envs=args.num_envs,
            seed=args.seed,
        )
    )
    dummy_action = _dummy_action_from_arg(args.dummy_action)
    run_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    run_config["cpu_ids"] = cpu_ids
    write_json(output_dir / "run_config.json", run_config)
    write_selected_tasks(output_dir / "selected_tasks.csv", records)

    plans = [
        build_task_id_baseline_plan(records, cpu_ids=cpu_ids),
        build_trapezoid_pipeline_plan(records, cpu_ids=cpu_ids),
    ]
    for repeat in range(args.random_baseline_repeats):
        plans.append(
            build_random_baseline_plan(
                records,
                cpu_ids=cpu_ids,
                seed=args.seed + repeat + 1,
            )
        )

    if args.fake_latency_from_csv:
        runner = BenchmarkRunner(steps_per_env=args.steps_per_env, step_fn=_fake_step_fn)
    else:
        env_factory = make_libero_env_factory(
            suite=args.suite,
            camera_height=args.camera_height,
            camera_width=args.camera_width,
            libero_type=args.libero_type,
            seed=args.seed,
            warmup_steps=args.warmup_steps,
            dummy_action=dummy_action,
        )
        runner = BenchmarkRunner(
            steps_per_env=args.steps_per_env,
            env_factory=env_factory,
            dummy_action=dummy_action,
            subprocess_timeout_s=args.subprocess_timeout_s,
        )

    summaries = []
    errors = []
    for plan in plans:
        schedule_name = plan[0].schedule_name
        if schedule_name == RANDOM_BASELINE and args.random_baseline_repeats > 1:
            random_index = sum(1 for summary in summaries if summary["schedule_name"].startswith(RANDOM_BASELINE))
            schedule_name = f"{RANDOM_BASELINE}_{random_index}"
            plan = [replace(item, schedule_name=schedule_name) for item in plan]
        write_schedule_plan(output_dir / f"schedule_plan_{schedule_name}.csv", plan)
        result = runner.run(schedule_name, plan)
        write_step_events(output_dir / f"step_events_{schedule_name}.jsonl", result.events)
        summaries.append(result.summary)
        errors.extend(result.errors)

    write_summary_csv(output_dir / "schedule_summary.csv", summaries)
    write_json(output_dir / "schedule_summary.json", summaries)
    write_comparison_report(output_dir / "comparison_report.md", summaries)
    if errors:
        with (output_dir / "errors.jsonl").open("w", encoding="utf-8") as handle:
            for error in errors:
                handle.write(json.dumps(error, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **步骤 5：运行 CLI 输出测试验证通过**

运行：

```bash
pytest tests/unit_tests/test_libero_latency_schedule_benchmark.py::test_main_fake_mode_writes_outputs -q
```

预期：PASS。

- [ ] **步骤 6：运行完整新测试文件**

运行：

```bash
pytest tests/unit_tests/test_libero_latency_schedule_benchmark.py -q
```

预期：PASS。

- [ ] **步骤 7：Commit**

```bash
git add toolkits/run_libero_latency_schedule_benchmark.py tests/unit_tests/test_libero_latency_schedule_benchmark.py
git commit -s -m "feat: add libero latency benchmark cli"
```

## 任务 7：验证命令、文档化示例和最终清理

**文件：**
- 修改：`docs/superpowers/specs/2026-05-18-libero-latency-schedule-benchmark-design.md`
- 修改：`toolkits/run_libero_latency_schedule_benchmark.py`
- 修改：`tests/unit_tests/test_libero_latency_schedule_benchmark.py`

- [ ] **步骤 1：在规格文档追加运行示例**

在规格文档 `Outputs` 章节后追加：

```markdown
## Example Commands

Unit-test smoke mode:

```bash
python toolkits/run_libero_latency_schedule_benchmark.py \
  --task-csv results/libero90_step_latency_all_tasks_10steps/task_latency_ranked.csv \
  --num-envs 32 \
  --cpu-ids 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --steps-per-env 10 \
  --output-dir results/libero90_latency_schedule_smoke \
  --fake-latency-from-csv
```

Real LIBERO benchmark:

```bash
python toolkits/run_libero_latency_schedule_benchmark.py \
  --task-csv results/libero90_step_latency_all_tasks_10steps/task_latency_ranked.csv \
  --num-envs 64 \
  --cpu-ids 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 \
  --steps-per-env 100 \
  --warmup-steps 20 \
  --output-dir results/libero90_latency_schedule_real
```
```

- [ ] **步骤 2：运行 unit tests**

运行：

```bash
pytest tests/unit_tests/test_libero_latency_schedule_benchmark.py -q
```

预期：PASS。

- [ ] **步骤 3：运行 fake CLI smoke**

运行：

```bash
python toolkits/run_libero_latency_schedule_benchmark.py \
  --task-csv results/libero90_step_latency_all_tasks_10steps/task_latency_ranked.csv \
  --num-envs 32 \
  --cpu-ids 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --steps-per-env 2 \
  --output-dir /tmp/libero_latency_schedule_fake \
  --fake-latency-from-csv
```

预期：退出码 0，并生成 `/tmp/libero_latency_schedule_fake/schedule_summary.csv`。

- [ ] **步骤 4：运行格式检查**

运行：

```bash
python -m ruff check toolkits/run_libero_latency_schedule_benchmark.py tests/unit_tests/test_libero_latency_schedule_benchmark.py
```

预期：PASS。如果 ruff 不可用，记录实际错误并至少运行 pytest。

- [ ] **步骤 5：查看 git diff，确认没有误改无关文件**

运行：

```bash
git diff --stat
git status --short
```

预期：只包含本计划涉及的脚本、测试和规格文档变更，既有用户改动不被回退。

- [ ] **步骤 6：Commit**

```bash
git add toolkits/run_libero_latency_schedule_benchmark.py tests/unit_tests/test_libero_latency_schedule_benchmark.py docs/superpowers/specs/2026-05-18-libero-latency-schedule-benchmark-design.md
git commit -s -m "docs: add libero latency benchmark usage"
```

## 实现注意事项

- 不要修改 Ray、RL runner、训练 worker 或 placement 逻辑。
- 不要把 `results/` 下的大型 profiler 输出加入 commit。
- 真实 LIBERO benchmark 可能很慢，不放进默认 CI。
- 现有工作区有未提交改动；执行计划时只 add 本任务明确列出的文件。
- fake 注入路径只用于单元测试和 smoke mode；默认真实 CLI 路径必须通过 `run_schedule_with_process_workers()` 使用每 core 一个 worker 进程。
