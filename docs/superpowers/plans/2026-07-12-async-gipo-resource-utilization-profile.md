# Async GIPO 资源利用率采集实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 为完整 8-step Async GIPO 运行增加分块 `torch.profiler`、RLinf-only 逐核 CPU 采样、离线聚合和三面板资源利用率图。

**架构：** Worker 使用可循环的 profiler schedule 和确定性 trace handler 分块导出 CUDA/CPU trace，同时写入 GPU 与时钟元数据；独立进程从 `/proc` 采集 RLinf 线程 CPU jiffies 和当前逻辑核。训练结束后，离线工具将 trace、CPU 采样和 runner 阶段事件对齐到墙钟时间，生成 1 秒聚合 CSV、覆盖率报告和 PNG/PDF 图。

**技术栈：** Python 3、PyTorch `torch.profiler`、Ray worker 环境变量、Linux `/proc`、CSV/JSONL、NumPy、Matplotlib、pytest、Ruff、Bash。

---

## 文件结构

- 修改 `rlinf/utils/profile_timeline.py`
  - 扩展 profiler schedule 参数；
  - 记录 CUDA device identity；
  - 提供循环分块 trace handler 和 manifest。
- 修改 `rlinf/workers/actor/async_ppo_fsdp_worker.py`
  - actor profiler 使用统一 trace handler。
- 修改 `rlinf/workers/env/env_worker.py`
  - env profiler 使用统一 trace handler。
- 修改 `rlinf/workers/rollout/hf/huggingface_worker.py`
  - rollout profiler 使用统一 trace handler。
- 修改 `rlinf/runners/async_ppo_embodied_runner.py`
  - profiling 开启时写入 step 和 actor-training 墙钟事件。
- 创建 `tools/rlinf_cpu_core_monitor.py`
  - 发现 RLinf/Ray 进程树并采集线程逐核 CPU 时间。
- 创建 `tools/async_gipo_resource_profile.py`
  - 解析分块 trace、CPU CSV 和阶段事件，生成聚合数据与覆盖率报告。
- 创建 `tools/plot_async_gipo_resource_utilization.py`
  - 绘制 CPU 热力图、GPU kernel busy 曲线和 worker occupancy 热力图。
- 创建 `tools/run_async_gipo_resource_profile.sh`
  - 启动/停止采样器、运行完整 GIPO、保留退出码并执行后处理。
- 修改 `docs/async_gipo_128traj_execution_guide.md`
  - 增加执行命令、指标口径、产物和限制。
- 修改/创建对应 `tests/unit_tests/` 测试文件。

## 任务 1：支持循环 profiler trace 与设备元数据

**文件：**
- 修改：`rlinf/utils/profile_timeline.py:22`
- 修改：`rlinf/workers/actor/async_ppo_fsdp_worker.py:45`
- 修改：`rlinf/workers/env/env_worker.py:315`
- 修改：`rlinf/workers/rollout/hf/huggingface_worker.py:292`
- 修改：`tests/unit_tests/test_profile_timeline.py`
- 修改：`tests/unit_tests/test_async_ppo_actor_profiler.py`

- [ ] **步骤 1：为 schedule、CUDA 元数据和 trace manifest 编写失败测试**

在 `tests/unit_tests/test_profile_timeline.py` 增加：

```python
from pathlib import Path
from types import SimpleNamespace

from rlinf.utils.profile_timeline import (
    TraceChunkHandler,
    build_cuda_metadata,
)


def test_torch_profiler_schedule_kwargs_reads_repeat(monkeypatch) -> None:
    monkeypatch.setenv("RLINF_TORCH_PROFILE_WAIT", "0")
    monkeypatch.setenv("RLINF_TORCH_PROFILE_WARMUP", "0")
    monkeypatch.setenv("RLINF_TORCH_PROFILE_ACTIVE", "4")
    monkeypatch.setenv("RLINF_TORCH_PROFILE_REPEAT", "0")

    assert torch_profiler_schedule_kwargs() == {
        "wait": 0,
        "warmup": 0,
        "active": 4,
        "repeat": 0,
    }


def test_build_cuda_metadata_records_visible_and_local_devices(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        current_device=lambda: 0,
        get_device_properties=lambda index: SimpleNamespace(
            name=f"A800-{index}", uuid=f"GPU-uuid-{index}"
        ),
    )

    metadata = build_cuda_metadata(fake_cuda)

    assert metadata["cuda_visible_devices"] == ["4", "5"]
    assert metadata["cuda_current_device"] == 0
    assert metadata["cuda_devices"] == [
        {"local_index": 0, "visible_id": "4", "name": "A800-0", "uuid": "GPU-uuid-0"},
        {"local_index": 1, "visible_id": "5", "name": "A800-1", "uuid": "GPU-uuid-1"},
    ]


def test_trace_chunk_handler_exports_manifest_rows(tmp_path) -> None:
    exported = []

    class FakeProfiler:
        step_num = 12

        def export_chrome_trace(self, path: str) -> None:
            exported.append(path)
            Path(path).write_text('{"traceEvents": []}')

    handler = TraceChunkHandler(tmp_path, component="generation", rank=1)
    handler(FakeProfiler())
    handler(FakeProfiler())

    manifest = [
        json.loads(line)
        for line in (tmp_path / "trace_manifest.jsonl").read_text().splitlines()
    ]
    assert [row["chunk_index"] for row in manifest] == [0, 1]
    assert [row["step_num"] for row in manifest] == [12, 12]
    assert len(exported) == 2
    assert all(Path(path).exists() for path in exported)
```

在 `tests/unit_tests/test_async_ppo_actor_profiler.py` 增加 repeat 循环断言：

```python
def test_create_actor_torch_profiler_repeats_schedule(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RLINF_TORCH_PROFILE", "1")
    monkeypatch.setenv("RLINF_TORCH_PROFILE_DIR", str(tmp_path))
    monkeypatch.setenv("RLINF_TORCH_PROFILE_WAIT", "0")
    monkeypatch.setenv("RLINF_TORCH_PROFILE_WARMUP", "0")
    monkeypatch.setenv("RLINF_TORCH_PROFILE_ACTIVE", "2")
    monkeypatch.setenv("RLINF_TORCH_PROFILE_REPEAT", "0")

    profiler, _, _ = create_actor_torch_profiler(rank=2, start=False)

    assert profiler.schedule(0).name == "RECORD"
    assert profiler.schedule(1).name == "RECORD_AND_SAVE"
    assert profiler.schedule(2).name == "RECORD"
    assert profiler.schedule(3).name == "RECORD_AND_SAVE"
```

- [ ] **步骤 2：运行测试并确认新 API 尚不存在**

运行：

```bash
source /data1/miliang/RLinf/libero_openpi/bin/activate
pytest -q tests/unit_tests/test_profile_timeline.py tests/unit_tests/test_async_ppo_actor_profiler.py
```

预期：FAIL，错误包含 `cannot import name 'TraceChunkHandler'` 或 repeat 断言失败。

- [ ] **步骤 3：实现 schedule、CUDA 元数据和确定性 trace handler**

在 `rlinf/utils/profile_timeline.py` 增加以下接口，并让 `write_time_anchor()` 合并
`build_cuda_metadata()` 的返回值：

```python
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
                "visible_id": visible[local_index] if local_index < len(visible) else str(local_index),
                "name": str(props.name),
                "uuid": str(getattr(props, "uuid", "")),
            }
        )
    return result


def torch_profiler_schedule_kwargs() -> dict[str, int]:
    return {
        "wait": int(os.environ.get("RLINF_TORCH_PROFILE_WAIT", "5")),
        "warmup": int(os.environ.get("RLINF_TORCH_PROFILE_WARMUP", "3")),
        "active": int(os.environ.get("RLINF_TORCH_PROFILE_ACTIVE", "10")),
        "repeat": int(os.environ.get("RLINF_TORCH_PROFILE_REPEAT", "1")),
    }
```

`write_time_anchor()` 写出的 JSON 继续保留原有 clock 字段，并加入
`cuda_visible_devices`、`cuda_current_device`、`cuda_devices` 和 schedule 参数。

- [ ] **步骤 4：将 actor、env、rollout profiler 接到统一 handler**

三个 worker 删除 `tensorboard_trace_handler` 导入，改为从
`rlinf.utils.profile_timeline` 导入 `TraceChunkHandler`。每个 `profile(...)` 使用：

```python
on_trace_ready=TraceChunkHandler(
    output_dir,
    component="training",  # env 使用 "env"，rollout 使用 "generation"
    rank=resolved_rank,     # env/rollout 使用 self._rank
),
```

保留现有 `record_shapes=False`、`profile_memory=False`、`with_stack=False` 和
`profiler.stop()` 行为。不要改变 profiling 未开启时的路径。

- [ ] **步骤 5：运行 profiler 单元测试**

运行：

```bash
pytest -q \
  tests/unit_tests/test_profile_timeline.py \
  tests/unit_tests/test_async_ppo_actor_profiler.py \
  tests/unit_tests/test_rollout_profile_worker_integration.py
```

预期：全部 PASS；原有默认 schedule 仍为 `repeat=1`。

- [ ] **步骤 6：提交 profiler 基础设施**

```bash
git add \
  rlinf/utils/profile_timeline.py \
  rlinf/workers/actor/async_ppo_fsdp_worker.py \
  rlinf/workers/env/env_worker.py \
  rlinf/workers/rollout/hf/huggingface_worker.py \
  tests/unit_tests/test_profile_timeline.py \
  tests/unit_tests/test_async_ppo_actor_profiler.py
git commit -s -m "feat: add chunked torch profiler traces"
```

## 任务 2：记录 GIPO step 与 actor-training 阶段

**文件：**
- 修改：`rlinf/runners/async_ppo_embodied_runner.py:107`
- 修改：`tests/unit_tests/test_async_embodied_launch.py`

- [ ] **步骤 1：编写 runner 事件失败测试**

在 `tests/unit_tests/test_async_embodied_launch.py` 增加：

```python
import json


def test_runner_resource_profile_events_are_gated_and_flushed(monkeypatch, tmp_path) -> None:
    runner = object.__new__(AsyncPPOEmbodiedRunner)
    runner.cfg = OmegaConf.create({"runner": {"logger": {"log_path": str(tmp_path)}}})
    runner._resource_profile_event_file = None

    monkeypatch.delenv("RLINF_RESOURCE_PROFILE", raising=False)
    runner._write_resource_profile_event("step.start", step=1)
    assert not (tmp_path / "resource_profile" / "runner_events.jsonl").exists()

    monkeypatch.setenv("RLINF_RESOURCE_PROFILE", "1")
    runner._write_resource_profile_event("step.start", step=1)
    runner._write_resource_profile_event("actor_training.start", step=1)
    runner._close_resource_profile_events()

    rows = [
        json.loads(line)
        for line in (
            tmp_path / "resource_profile" / "runner_events.jsonl"
        ).read_text().splitlines()
    ]
    assert [row["event"] for row in rows] == [
        "step.start",
        "actor_training.start",
    ]
    assert all(row["step"] == 1 for row in rows)
    assert all(isinstance(row["wall_ns"], int) for row in rows)
```

- [ ] **步骤 2：运行测试并确认事件接口缺失**

运行：

```bash
pytest -q tests/unit_tests/test_async_embodied_launch.py::test_runner_resource_profile_events_are_gated_and_flushed
```

预期：FAIL，错误为 `_write_resource_profile_event` 不存在。

- [ ] **步骤 3：实现 runner JSONL writer**

在 `AsyncPPOEmbodiedRunner` 增加：

```python
def _write_resource_profile_event(self, event: str, *, step: int) -> None:
    if os.environ.get("RLINF_RESOURCE_PROFILE") != "1":
        return
    handle = getattr(self, "_resource_profile_event_file", None)
    if handle is None:
        output_dir = Path(str(self.cfg.runner.logger.log_path)) / "resource_profile"
        output_dir.mkdir(parents=True, exist_ok=True)
        handle = (output_dir / "runner_events.jsonl").open(
            "a", encoding="utf-8", buffering=1
        )
        self._resource_profile_event_file = handle
    handle.write(
        json.dumps(
            {"event": event, "step": int(step), "wall_ns": time.time_ns()},
            sort_keys=True,
        )
        + "\n"
    )


def _close_resource_profile_events(self) -> None:
    handle = getattr(self, "_resource_profile_event_file", None)
    if handle is not None:
        handle.close()
        self._resource_profile_event_file = None
```

补充 `json`、`os` 和 `Path` 导入。

- [ ] **步骤 4：在训练循环写入成对阶段事件**

在 `while self.global_step < self.max_steps` 的每次迭代中使用显示 step
`display_step = self.global_step + 1`，按以下顺序写入：

```python
self._write_resource_profile_event("step.start", step=display_step)
try:
    ...
    self._write_resource_profile_event("actor_training.start", step=display_step)
    try:
        actor_training_handle = self.actor.run_training()
        training_metrics = actor_training_handle.wait()
    finally:
        self._write_resource_profile_event("actor_training.end", step=display_step)
    ...
finally:
    self._write_resource_profile_event("step.end", step=display_step)
```

在 `run()` 的外层 `finally` 调用 `_close_resource_profile_events()`，确保异常退出也 flush。
保持现有训练异常继续向上传播。

- [ ] **步骤 5：运行 runner 测试**

运行：

```bash
pytest -q \
  tests/unit_tests/test_async_embodied_launch.py \
  tests/unit_tests/test_async_gipo_runner.py
```

预期：全部 PASS，profiling 未开启时不创建新文件。

- [ ] **步骤 6：提交阶段事件**

```bash
git add \
  rlinf/runners/async_ppo_embodied_runner.py \
  tests/unit_tests/test_async_embodied_launch.py
git commit -s -m "feat: record async training phase windows"
```

## 任务 3：实现 RLinf-only 线程逐核 CPU 采样器

**文件：**
- 创建：`tools/rlinf_cpu_core_monitor.py`
- 创建：`tests/unit_tests/test_rlinf_cpu_core_monitor.py`

- [ ] **步骤 1：编写 `/proc` 解析和归因失败测试**

创建 `tests/unit_tests/test_rlinf_cpu_core_monitor.py`：

```python
from tools.rlinf_cpu_core_monitor import (
    ThreadSnapshot,
    compute_thread_delta,
    parse_environ,
    parse_task_stat,
)


def test_parse_task_stat_reads_jiffies_and_processor() -> None:
    prefix = "42 (ray worker name) S "
    fields = [str(index) for index in range(1, 45)]
    snapshot = parse_task_stat(prefix + " ".join(fields))

    assert snapshot.jiffies == 23
    assert snapshot.processor == 36


def test_parse_environ_reads_worker_identity() -> None:
    env = parse_environ(
        b"GROUP_NAME=ActorGroup\x00RANK=1\x00CUDA_VISIBLE_DEVICES=6\x00"
    )
    assert env["GROUP_NAME"] == "ActorGroup"
    assert env["RANK"] == "1"


def test_compute_thread_delta_assigns_to_current_cpu_and_marks_migration() -> None:
    before = ThreadSnapshot(jiffies=100, processor=7)
    after = ThreadSnapshot(jiffies=125, processor=9)

    delta = compute_thread_delta(before, after, clk_tck=100)

    assert delta.cpu == 9
    assert delta.cpu_time_s == 0.25
    assert delta.migrated is True
```

- [ ] **步骤 2：运行测试并确认模块不存在**

运行：

```bash
pytest -q tests/unit_tests/test_rlinf_cpu_core_monitor.py
```

预期：FAIL，错误为 `ModuleNotFoundError: tools.rlinf_cpu_core_monitor`。

- [ ] **步骤 3：实现 stat/environ 解析与线程增量**

在 `tools/rlinf_cpu_core_monitor.py` 定义：

```python
@dataclass(frozen=True)
class ThreadSnapshot:
    jiffies: int
    processor: int


@dataclass(frozen=True)
class ThreadDelta:
    cpu: int
    cpu_time_s: float
    migrated: bool


def parse_task_stat(text: str) -> ThreadSnapshot:
    close = text.rfind(")")
    if close < 0:
        raise ValueError(f"Malformed task stat: {text!r}")
    fields = text[close + 2 :].split()
    return ThreadSnapshot(
        jiffies=int(fields[11]) + int(fields[12]),
        processor=int(fields[36]),
    )


def parse_environ(raw: bytes) -> dict[str, str]:
    result = {}
    for item in raw.split(b"\x00"):
        if b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        result[key.decode(errors="replace")] = value.decode(errors="replace")
    return result


def compute_thread_delta(
    before: ThreadSnapshot,
    after: ThreadSnapshot,
    *,
    clk_tck: int,
) -> ThreadDelta:
    delta_jiffies = max(after.jiffies - before.jiffies, 0)
    return ThreadDelta(
        cpu=after.processor,
        cpu_time_s=delta_jiffies / clk_tck,
        migrated=before.processor != after.processor,
    )
```

- [ ] **步骤 4：实现 RLinf 进程发现和 CSV 采样循环**

采样器读取每个进程 `/proc/<pid>/environ`。如果 `GROUP_NAME`、`WORKER_NAME`、
`RANK` 或 cmdline 表明它是训练主进程、ActorGroup、RolloutGroup、EnvGroup、Ray worker，
则跟踪该进程及其后代。CSV 固定字段：

```python
CSV_FIELDS = [
    "interval_start",
    "interval_end",
    "pid",
    "tid",
    "component",
    "rank",
    "cpu",
    "cpu_time_s",
    "migrated",
]
```

CLI 必须为：

```text
python tools/rlinf_cpu_core_monitor.py \
  --output <resource_profile>/cpu/thread_core_samples.csv \
  --stop-file <resource_profile>/cpu/monitor.stop \
  --interval 0.1
```

循环使用 `time.monotonic()` 控制间隔、`time.time()` 写墙钟；新线程在第二个样本后才写
delta；消失线程从缓存删除；输出文件使用 `buffering=1` 并每轮 flush。SIGTERM 和
stop file 都应正常退出。

- [ ] **步骤 5：增加进程消失和非 RLinf 排除测试**

使用 `tmp_path` 构造假的 proc 目录，并让采样函数接受 `proc_root: Path = Path("/proc")`。
测试：

```python
def test_discover_rlinf_processes_uses_worker_environment(fake_proc) -> None:
    fake_proc.add_process(
        pid=101,
        environ={"GROUP_NAME": "EnvGroup", "RANK": "3"},
        cmdline="ray::EnvWorker",
    )
    fake_proc.add_process(pid=202, environ={}, cmdline="unrelated_job")

    processes = discover_rlinf_processes(fake_proc.root)

    assert [(proc.pid, proc.component, proc.rank) for proc in processes] == [
        (101, "env", 3)
    ]
```

- [ ] **步骤 6：运行 CPU sampler 测试和 Ruff**

运行：

```bash
pytest -q tests/unit_tests/test_rlinf_cpu_core_monitor.py
ruff check tools/rlinf_cpu_core_monitor.py tests/unit_tests/test_rlinf_cpu_core_monitor.py
```

预期：全部 PASS，Ruff 无输出。

- [ ] **步骤 7：提交 CPU sampler**

```bash
git add \
  tools/rlinf_cpu_core_monitor.py \
  tests/unit_tests/test_rlinf_cpu_core_monitor.py
git commit -s -m "feat: sample RLinf CPU usage per core"
```

## 任务 4：实现 trace、CPU 和阶段数据聚合

**文件：**
- 创建：`tools/async_gipo_resource_profile.py`
- 创建：`tests/unit_tests/test_async_gipo_resource_profile.py`

- [ ] **步骤 1：编写区间并集和 occupancy 加权失败测试**

创建 `tests/unit_tests/test_async_gipo_resource_profile.py`：

```python
import math

from tools.async_gipo_resource_profile import (
    KernelInterval,
    aggregate_gpu_bins,
    interval_union_duration,
)


def test_interval_union_duration_deduplicates_overlapping_kernels() -> None:
    assert interval_union_duration([(0.0, 0.7), (0.4, 1.0)], 0.0, 1.0) == 1.0


def test_aggregate_gpu_bins_caps_busy_and_weights_occupancy() -> None:
    kernels = [
        KernelInterval("rollout_rank0", "generation", 0, "GPU-a", "4", 0.0, 0.75, 25.0),
        KernelInterval("rollout_rank0", "generation", 0, "GPU-a", "4", 0.25, 1.0, 75.0),
    ]

    device_rows, worker_rows = aggregate_gpu_bins(kernels, bin_s=1.0)

    assert device_rows[0]["kernel_busy_pct"] == 100.0
    assert worker_rows[0]["est_sm_occupancy_pct"] == 50.0


def test_aggregate_gpu_bins_uses_nan_when_occupancy_is_missing() -> None:
    kernels = [
        KernelInterval("env_rank0", "env", 0, "GPU-a", "4", 0.0, 0.5, None)
    ]

    _, worker_rows = aggregate_gpu_bins(kernels, bin_s=1.0)

    assert math.isnan(worker_rows[0]["est_sm_occupancy_pct"])
```

- [ ] **步骤 2：运行测试并确认聚合模块不存在**

运行：

```bash
pytest -q tests/unit_tests/test_async_gipo_resource_profile.py
```

预期：FAIL，错误为 `ModuleNotFoundError`。

- [ ] **步骤 3：实现核心数据类型和区间算法**

在 `tools/async_gipo_resource_profile.py` 增加：

```python
@dataclass(frozen=True)
class KernelInterval:
    worker: str
    component: str
    rank: int
    gpu_uuid: str
    gpu_label: str
    start_s: float
    end_s: float
    occupancy_pct: float | None


def interval_union_duration(
    intervals: list[tuple[float, float]],
    bin_start: float,
    bin_end: float,
) -> float:
    clipped = sorted(
        (max(start, bin_start), min(end, bin_end))
        for start, end in intervals
        if min(end, bin_end) > max(start, bin_start)
    )
    total = 0.0
    current_start = current_end = None
    for start, end in clipped:
        if current_start is None:
            current_start, current_end = start, end
        elif start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    if current_start is not None:
        total += current_end - current_start
    return total
```

`aggregate_gpu_bins()` 必须按物理 GPU 求所有 worker kernel 区间并集；worker occupancy
按 kernel 与桶交集时长加权。结果字段固定为：

```text
gpu_device_1s.csv:
timestamp,datetime,gpu_uuid,gpu_label,kernel_busy_pct

gpu_worker_1s.csv:
timestamp,datetime,worker,component,rank,gpu_uuid,gpu_label,
kernel_busy_pct,est_sm_occupancy_pct,kernel_count
```

- [ ] **步骤 4：实现多 trace 块解析和墙钟对齐**

对每个包含 `time_anchor.json` 和 `trace_manifest.jsonl` 的 worker 目录：

1. 校验 manifest 中引用的 trace 文件存在；
2. 读取所有 `*.pt.trace.json`；
3. 只保留 `cat == "kernel"` 且 `ph == "X"` 的事件；
4. 使用以下转换得到墙钟秒：

```python
def trace_us_to_wall_s(trace_us: float, anchor: dict[str, Any]) -> float:
    clock_offset_ns = int(anchor["wall_time_ns"]) - int(anchor["perf_counter_ns"])
    return (trace_us * 1_000.0 + clock_offset_ns) / 1_000_000_000.0
```

5. 从 kernel args 的 `device`、`Device Id` 或 `device_id` 解析 local device；
6. 使用 anchor 的 `cuda_devices` 映射 UUID 和 `visible_id`；
7. 从 `est. achieved occupancy %` 读取 occupancy，缺失时为 `None`。

单元测试构造两个 trace 文件，断言两个 chunk 都被读取，且 local device 0 映射到
`visible_id=4` 和 `GPU-a`。

- [ ] **步骤 5：实现 CPU 1 秒聚合**

读取 `thread_core_samples.csv`，将每个 `[interval_start, interval_end]` 的
`cpu_time_s` 按与 1 秒桶的重叠比例分配。输出每个时间桶、每个逻辑核一行：

```text
timestamp,datetime,cpu,util_pct,thread_count,migration_count
```

利用率公式为 `100 * cpu_time_s / bin_s`。原始值不裁剪；coverage 报告统计
`util_pct > 100` 的行数和迁核比例。

- [ ] **步骤 6：实现阶段窗口和 coverage 报告**

解析 `<run_dir>/resource_profile/runner_events.jsonl`，按 `(event prefix, step)` 配对：

```text
step.start              -> step.end
actor_training.start    -> actor_training.end
```

再读取现有 `rollout_generation_timestamps/*.jsonl` 和
`env_sim_timestamps/*.jsonl` 的 start/end 事件，生成：

```text
phase_windows.csv:
phase,step,component,rank,start,end,source
```

`coverage.json` 至少包含：

```json
{
  "cpu": {"start": 0.0, "end": 0.0, "migration_ratio": 0.0, "over_100_bins": 0},
  "torch": {
    "workers": [],
    "missing_trace_files": [],
    "unmapped_device_events": 0,
    "missing_occupancy_events": 0,
    "total_trace_bytes": 0
  },
  "phases": {"unmatched_starts": [], "unmatched_ends": []},
  "warnings": []
}
```

缺文件、无法映射 GPU、未配对阶段只写 warning；JSON/CSV 语法错误应抛出带路径的
`ValueError`，使正式采集不会静默生成错误图。

- [ ] **步骤 7：实现 CLI 和输出目录**

CLI：

```text
python tools/async_gipo_resource_profile.py \
  /tmp/rlinf_async_gipo_resource_YYYYMMDD_HHMMSS \
  --bin-s 1.0 \
  --num-cpus 112
```

输入 run dir 是 `runner.logger.log_path`。输出到
`<run_dir>/resource_profile/derived/`，同时写
`<run_dir>/resource_profile/metadata.json`。如果 CPU CSV 或 torch 目录不存在，命令
失败并明确列出缺少的路径。

- [ ] **步骤 8：运行聚合测试与 Ruff**

运行：

```bash
pytest -q tests/unit_tests/test_async_gipo_resource_profile.py
ruff check \
  tools/async_gipo_resource_profile.py \
  tests/unit_tests/test_async_gipo_resource_profile.py
```

预期：全部 PASS，Ruff 无输出。

- [ ] **步骤 9：提交离线聚合器**

```bash
git add \
  tools/async_gipo_resource_profile.py \
  tests/unit_tests/test_async_gipo_resource_profile.py
git commit -s -m "feat: aggregate async GIPO resource traces"
```

## 任务 5：绘制三面板资源利用率图

**文件：**
- 创建：`tools/plot_async_gipo_resource_utilization.py`
- 创建：`tests/unit_tests/test_plot_async_gipo_resource_utilization.py`

- [ ] **步骤 1：编写矩阵构建和绘图失败测试**

创建 `tests/unit_tests/test_plot_async_gipo_resource_utilization.py`：

```python
from tools.plot_async_gipo_resource_utilization import (
    build_cpu_matrix,
    plot_resource_utilization,
)


def test_build_cpu_matrix_keeps_all_logical_cores() -> None:
    rows = [
        {"timestamp": 10.0, "cpu": 0, "util_pct": 25.0},
        {"timestamp": 11.0, "cpu": 1, "util_pct": 50.0},
    ]

    times, matrix = build_cpu_matrix(rows, num_cpus=4)

    assert times.tolist() == [10.0, 11.0]
    assert matrix.shape == (4, 2)
    assert matrix[0, 0] == 25.0
    assert matrix[1, 1] == 50.0


def test_plot_resource_utilization_writes_png_and_pdf(tmp_path, derived_fixture) -> None:
    output_prefix = tmp_path / "resource_utilization"

    plot_resource_utilization(
        derived_dir=derived_fixture,
        output_prefix=output_prefix,
        num_cpus=4,
    )

    assert output_prefix.with_suffix(".png").stat().st_size > 1_000
    assert output_prefix.with_suffix(".pdf").stat().st_size > 1_000
```

`derived_fixture` 写入最小的 CPU、GPU device、GPU worker 和 phase CSV，包含一个
occupancy `NaN` 桶。

- [ ] **步骤 2：运行测试并确认绘图模块不存在**

运行：

```bash
MPLBACKEND=Agg pytest -q tests/unit_tests/test_plot_async_gipo_resource_utilization.py
```

预期：FAIL，错误为 `ModuleNotFoundError`。

- [ ] **步骤 3：实现 CSV 读取和稳定矩阵构建**

`build_cpu_matrix()` 将最早到最晚时间重建为完整的等间隔时间轴，并返回形状
`(num_cpus, len(times))` 的矩阵。一个已覆盖时间桶中未出现的核心填 0；整个采样桶
缺失时该列填 `NaN`，使图中显示空白而不是伪造空闲。

worker occupancy 矩阵按以下顺序排序：

```python
COMPONENT_ORDER = {"training": 0, "generation": 1, "env": 2}


def worker_sort_key(row: dict[str, object]) -> tuple[int, int]:
    return COMPONENT_ORDER[str(row["component"])], int(row["rank"])
```

occupancy 缺失保持 `np.nan`，绘图 colormap 的 bad color 设置为白色。

- [ ] **步骤 4：实现共享横轴三面板图**

图尺寸使用 `figsize=(12.0, 8.0)`，`GridSpec` 高度比为 `(3.2, 1.4, 2.0)`：

1. CPU panel：`imshow`，纵轴 0 到 `num_cpus - 1`，颜色范围 0-100%；
2. GPU panel：物理 GPU label 各一条 `kernel_busy_pct` 曲线，纵轴 0-100%；
3. occupancy panel：worker heatmap，纵轴标签为 `actor r0`、`rollout r0`、`env r0`。

阶段窗口使用：

```python
for phase in phases:
    if phase.phase == "step":
        for axis in axes:
            axis.axvline(phase.start, color="black", linewidth=0.7, alpha=0.45)
    elif phase.phase == "actor_training":
        for axis in axes:
            axis.axvspan(phase.start, phase.end, color="#E15759", alpha=0.08)
```

横轴转换为从第一个有效样本开始的秒数。标题和标签必须使用：

```text
RLinf CPU utilization per logical core (%)
CUDA kernel busy by physical GPU (%)
Estimated SM occupancy by worker (%)
```

不得出现 `SM_ACTIVE`。PNG 使用 200 DPI，PDF 使用可编辑字体
`pdf.fonttype = 42`。

- [ ] **步骤 5：运行绘图测试和现有相邻测试**

运行：

```bash
MPLBACKEND=Agg pytest -q \
  tests/unit_tests/test_plot_async_gipo_resource_utilization.py \
  tests/unit_tests/test_plot_worker_sm_timeline.py \
  tests/unit_tests/test_summarize_torch_phase_sm.py
ruff check \
  tools/plot_async_gipo_resource_utilization.py \
  tests/unit_tests/test_plot_async_gipo_resource_utilization.py
```

预期：全部 PASS，现有 profiler 图工具不回归。

- [ ] **步骤 6：提交绘图工具**

```bash
git add \
  tools/plot_async_gipo_resource_utilization.py \
  tests/unit_tests/test_plot_async_gipo_resource_utilization.py
git commit -s -m "feat: plot async GIPO resource utilization"
```

## 任务 6：增加一键执行脚本和执行指南

**文件：**
- 创建：`tools/run_async_gipo_resource_profile.sh`
- 修改：`docs/async_gipo_128traj_execution_guide.md`

- [ ] **步骤 1：编写 shell 脚本静态失败检查**

先创建只包含 shebang 的脚本，然后运行：

```bash
bash -n tools/run_async_gipo_resource_profile.sh
rg -n "RLINF_TORCH_PROFILE_REPEAT|rlinf_cpu_core_monitor|async_gipo_resource_profile|plot_async_gipo" tools/run_async_gipo_resource_profile.sh
```

预期：`bash -n` PASS，`rg` 返回 1，因为脚本尚未包含必需流程。

- [ ] **步骤 2：实现采样器启停和环境配置**

脚本开头固定：

```bash
#!/usr/bin/env bash
set -uo pipefail

RLINF=/data1/miliang/RLinf
RUN_DIR=${RUN_DIR:-/tmp/rlinf_async_gipo_resource_$(date +%Y%m%d_%H%M%S)}
CPU_INTERVAL=${CPU_INTERVAL:-0.1}
PROFILE_ACTIVE_STEPS=${PROFILE_ACTIVE_STEPS:-25}

mkdir -p "$RUN_DIR/resource_profile/cpu" "$RUN_DIR/resource_profile/torch"
source "$RLINF/libero_openpi/bin/activate"
cd "$RLINF"

export EMBODIED_PATH=$RLINF/examples/embodiment
export PYTHONPATH=$RLINF:${PYTHONPATH:-}
export LIBERO_TYPE=standard
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export ROBOT_PLATFORM=LIBERO
export RLINF_TRAINING_EVAL_LOCAL_RAY=1
export RLINF_RESOURCE_PROFILE=1
export RLINF_TORCH_PROFILE=1
export RLINF_TORCH_PROFILE_DIR=$RUN_DIR/resource_profile/torch
export RLINF_TORCH_PROFILE_WAIT=0
export RLINF_TORCH_PROFILE_WARMUP=0
export RLINF_TORCH_PROFILE_ACTIVE=$PROFILE_ACTIVE_STEPS
export RLINF_TORCH_PROFILE_REPEAT=0
export MPLCONFIGDIR=$RUN_DIR/matplotlib
```

启动 CPU monitor，并用 trap 保证 stop file 被创建：

```bash
STOP_FILE=$RUN_DIR/resource_profile/cpu/monitor.stop
rm -f "$STOP_FILE"
python tools/rlinf_cpu_core_monitor.py \
  --output "$RUN_DIR/resource_profile/cpu/thread_core_samples.csv" \
  --stop-file "$STOP_FILE" \
  --interval "$CPU_INTERVAL" \
  > "$RUN_DIR/resource_profile/cpu/monitor.log" 2>&1 &
CPU_MONITOR_PID=$!

cleanup() {
  touch "$STOP_FILE"
  kill "$CPU_MONITOR_PID" 2>/dev/null || true
  wait "$CPU_MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
```

- [ ] **步骤 3：加入完整 8-step GIPO 命令并保留退出码**

脚本使用执行指南中的同一配置和 override：

```bash
TRAIN_STATUS=0
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python examples/embodiment/train_async.py \
  --config-path "$RLINF/examples/embodiment/config" \
  --config-name libero_spatial_async_gipo_openpi_pi05_verify \
  runner.max_epochs=8 \
  runner.max_steps=8 \
  runner.val_check_interval=-1 \
  runner.save_interval=-1 \
  runner.logger.logger_backends=[] \
  runner.logger.log_path="$RUN_DIR" \
  +actor.recv_drain_max_trajectories=8 \
  "$@" \
  2>&1 | python tools/timestamp_stream.py | tee "$RUN_DIR/train.log" \
  || TRAIN_STATUS=${PIPESTATUS[0]}
```

训练结束后先停止 CPU monitor，再执行：

```bash
cleanup
trap - EXIT INT TERM

PROFILE_STATUS=0
python tools/async_gipo_resource_profile.py "$RUN_DIR" --bin-s 1.0 --num-cpus 112 \
  || PROFILE_STATUS=$?
python tools/plot_async_gipo_resource_utilization.py \
  "$RUN_DIR/resource_profile/derived" \
  --output-prefix "$RUN_DIR/resource_profile/resource_utilization" \
  --num-cpus 112 \
  || PROFILE_STATUS=$?

if [[ "$TRAIN_STATUS" -ne 0 ]]; then
  exit "$TRAIN_STATUS"
fi
exit "$PROFILE_STATUS"
```

只有训练成功时才要求聚合/绘图成功；训练失败时仍尝试生成已有数据，但最终必须返回原始
`TRAIN_STATUS`。

- [ ] **步骤 4：更新执行指南**

在 `docs/async_gipo_128traj_execution_guide.md` 的 Launch Command 后新增
“Resource Utilization Profiling”章节，包含：

```bash
RUN_DIR=/tmp/rlinf_async_gipo_resource_$(date +%Y%m%d_%H%M%S) \
bash tools/run_async_gipo_resource_profile.sh
```

文档必须明确：

- CPU 是 RLinf-only、100 ms 采样、按后一次观测核心近似归因；
- GPU `kernel_busy_pct` 是 kernel 时间区间并集；
- `est_sm_occupancy_pct` 来自 `torch.profiler` kernel launch 信息；
- 两者都不是真实硬件 `SM_ACTIVE`；
- 完整 profiler 会增加运行时间和磁盘占用；
- 正式分析前检查 `resource_profile/derived/coverage.json`；
- 最终图位于 `resource_profile/resource_utilization.png` 和 `.pdf`。

- [ ] **步骤 5：运行脚本和文档静态检查**

运行：

```bash
bash -n tools/run_async_gipo_resource_profile.sh
rg -n \
  "kernel_busy_pct|est_sm_occupancy_pct|SM_ACTIVE|coverage.json|resource_utilization" \
  docs/async_gipo_128traj_execution_guide.md
```

预期：shell 语法通过，文档包含全部指标和限制。

- [ ] **步骤 6：提交启动脚本和文档**

```bash
git add \
  tools/run_async_gipo_resource_profile.sh \
  docs/async_gipo_128traj_execution_guide.md
git commit -s -m "docs: add async GIPO resource profiling guide"
```

## 任务 7：完成全量验证和 GPU smoke test

**文件：**
- 验证全部本计划修改文件

- [ ] **步骤 1：运行定向单元测试**

```bash
source /data1/miliang/RLinf/libero_openpi/bin/activate
MPLBACKEND=Agg pytest -q \
  tests/unit_tests/test_profile_timeline.py \
  tests/unit_tests/test_async_ppo_actor_profiler.py \
  tests/unit_tests/test_async_embodied_launch.py \
  tests/unit_tests/test_async_gipo_runner.py \
  tests/unit_tests/test_rlinf_cpu_core_monitor.py \
  tests/unit_tests/test_async_gipo_resource_profile.py \
  tests/unit_tests/test_plot_async_gipo_resource_utilization.py \
  tests/unit_tests/test_plot_worker_sm_timeline.py \
  tests/unit_tests/test_summarize_torch_phase_sm.py
```

预期：全部 PASS。

- [ ] **步骤 2：运行 Ruff 检查与格式化验证**

```bash
ruff check \
  rlinf/utils/profile_timeline.py \
  rlinf/runners/async_ppo_embodied_runner.py \
  rlinf/workers/actor/async_ppo_fsdp_worker.py \
  rlinf/workers/env/env_worker.py \
  rlinf/workers/rollout/hf/huggingface_worker.py \
  tools/rlinf_cpu_core_monitor.py \
  tools/async_gipo_resource_profile.py \
  tools/plot_async_gipo_resource_utilization.py \
  tests/unit_tests/test_profile_timeline.py \
  tests/unit_tests/test_async_ppo_actor_profiler.py \
  tests/unit_tests/test_async_embodied_launch.py \
  tests/unit_tests/test_rlinf_cpu_core_monitor.py \
  tests/unit_tests/test_async_gipo_resource_profile.py \
  tests/unit_tests/test_plot_async_gipo_resource_utilization.py

ruff format --check \
  rlinf/utils/profile_timeline.py \
  rlinf/runners/async_ppo_embodied_runner.py \
  rlinf/workers/actor/async_ppo_fsdp_worker.py \
  rlinf/workers/env/env_worker.py \
  rlinf/workers/rollout/hf/huggingface_worker.py \
  tools/rlinf_cpu_core_monitor.py \
  tools/async_gipo_resource_profile.py \
  tools/plot_async_gipo_resource_utilization.py
```

预期：两个命令都退出 0。

- [ ] **步骤 3：运行缩短版 GPU smoke test**

使用完整脚本但覆盖为 1 step：

```bash
export RUN_DIR=/tmp/rlinf_async_gipo_resource_smoke_$(date +%Y%m%d_%H%M%S)
PROFILE_ACTIVE_STEPS=5 \
bash tools/run_async_gipo_resource_profile.sh \
  runner.max_epochs=1 \
  runner.max_steps=1
```

预期：训练退出码为 0，并存在：

```text
resource_profile/cpu/thread_core_samples.csv
resource_profile/torch/*/trace_manifest.jsonl
resource_profile/derived/cpu_core_1s.csv
resource_profile/derived/gpu_worker_1s.csv
resource_profile/derived/gpu_device_1s.csv
resource_profile/derived/phase_windows.csv
resource_profile/derived/coverage.json
resource_profile/resource_utilization.png
resource_profile/resource_utilization.pdf
```

- [ ] **步骤 4：检查 smoke coverage 和图片非空**

```bash
python -m json.tool "$RUN_DIR/resource_profile/derived/coverage.json"
test -s "$RUN_DIR/resource_profile/resource_utilization.png"
test -s "$RUN_DIR/resource_profile/resource_utilization.pdf"
```

预期：JSON 可解析，`missing_trace_files` 为空，PNG/PDF 非空。允许部分 env kernel 没有
occupancy 字段，但必须在 `missing_occupancy_events` 中计数。

- [ ] **步骤 5：检查最终 diff 和工作区边界**

```bash
git diff --check
git status --short
```

预期：`git diff --check` 无输出；`git status` 不包含本计划遗漏的未提交修改。不要清理或
回退任务开始前已经存在的用户修改。

- [ ] **步骤 6：提交 smoke test 后的必要修正**

仅当 smoke test 产生代码或文档修正时执行：

```bash
git add \
  rlinf/utils/profile_timeline.py \
  rlinf/runners/async_ppo_embodied_runner.py \
  rlinf/workers/actor/async_ppo_fsdp_worker.py \
  rlinf/workers/env/env_worker.py \
  rlinf/workers/rollout/hf/huggingface_worker.py \
  tools/rlinf_cpu_core_monitor.py \
  tools/async_gipo_resource_profile.py \
  tools/plot_async_gipo_resource_utilization.py \
  tools/run_async_gipo_resource_profile.sh \
  docs/async_gipo_128traj_execution_guide.md \
  tests/unit_tests/test_profile_timeline.py \
  tests/unit_tests/test_async_ppo_actor_profiler.py \
  tests/unit_tests/test_async_embodied_launch.py \
  tests/unit_tests/test_rlinf_cpu_core_monitor.py \
  tests/unit_tests/test_async_gipo_resource_profile.py \
  tests/unit_tests/test_plot_async_gipo_resource_utilization.py
git commit -s -m "fix: harden async GIPO resource profiling"
```

如果 smoke test 无需修正，则跳过此提交。
