# Async GIPO 全程资源利用率采集设计

## 背景

`docs/async_gipo_128traj_execution_guide.md` 记录了 Async GIPO LIBERO/OpenPI
128 trajectory 的完整 8-step 运行方式和时序统计，但目前缺少运行期间的 CPU、GPU
资源利用率数据。目标是在同一次完整运行中采集：

- RLinf 进程在每个逻辑 CPU 核上的利用率；
- GPU 4-7 上的 CUDA kernel busy ratio；
- actor、rollout、env worker 的估算 SM occupancy；
- 与 warmup、GIPO step、rollout、actor training 阶段对齐的资源时间线图。

用户选择使用 `torch.profiler`。因此 GPU 指标必须明确标记为
`kernel_busy_pct` 和 `est_sm_occupancy_pct`，不能称为真实硬件
`SM_ACTIVE`。真实 `SM_ACTIVE` 需要 Nsight Systems、DCGM 或 Nsight Compute
硬件计数器，不属于本设计范围。

## 目标与非目标

### 目标

1. 覆盖完整 8-step GIPO 运行，而不是只采代表性窗口。
2. 通过循环分块 trace 控制单个文件和 profiler 内存占用。
3. 将不同进程、不同 trace 块和 CPU 采样统一到同一个墙钟时间轴。
4. 输出可复算的原始数据、1 秒聚合数据、覆盖率报告以及 PNG/PDF 图。
5. 保持采集逻辑可关闭，默认不改变普通训练行为。

### 非目标

- 不采集真实 `SM_ACTIVE`、tensor core active 或 DRAM 硬件计数器。
- 不将 `nvidia-smi utilization.gpu` 当作 SM 利用率。
- 不精确重放线程在 100 ms 采样间隔内的每次 CPU 迁移。
- 不按 CUDA kernel 精确拆分同一 GPU 上多个进程的硬件 SM 配额。
- 不将完整 8-step GPU 运行加入自动化测试。

## 总体架构

采集由三个彼此独立的单元组成：

1. **Worker torch profiler**
   - actor、rollout、env worker 分别创建 profiler；
   - 使用 CPU 和 CUDA activities，以保留 `record_function` 阶段和 CUDA kernel；
   - 关闭 shape、memory、stack 等非必要采集；
   - 使用 `wait=0`、`warmup=0`、固定 `active` 长度和 `repeat=0` 循环分块导出；
   - 每个 trace 块独立落盘，避免整次运行只生成一个超大 trace。

2. **RLinf CPU core sampler**
   - 在训练进程外独立运行，每 100 ms 刷新一次；
   - 从训练日志和进程树识别 RLinf/Ray worker 及其子进程；
   - 遍历 `/proc/<pid>/task/<tid>/stat`，读取线程 `utime + stime` 和
     `processor`；
   - 将相邻采样间的 CPU jiffies 增量近似归入后一次采样观测到的逻辑核；
   - 保留 PID、TID、component、rank、CPU 和迁核标记，供质量检查和聚合。

3. **Offline postprocessor and plotter**
   - 读取全部 trace 块、CPU 原始采样和已有 GIPO 时间戳；
   - 对齐墙钟时间、校验覆盖范围、生成 1 秒聚合 CSV；
   - 输出共享时间轴的三面板资源利用率图。

三个单元通过文件接口通信，不要求训练进程内直接调用绘图或分析代码。

## Profiler 分块与时间对齐

### 分块策略

每个 worker 的 profiler 周期由固定数量的 `profiler.step()` 组成。一个周期结束时，
`on_trace_ready` 将当前块写到该 worker 的独立目录，然后 profiler 自动进入下一周期。
`repeat=0` 表示周期持续重复，直到 worker 正常停止 profiler。`warmup=0` 用于覆盖运行
开头，但首个 trace 块会包含 profiler/CUPTI 启动扰动，后处理和图中必须将其标明。

分块长度是环境变量控制的静态整数，所有 worker 使用同一默认值。由于 actor、rollout、
env 调用 `profiler.step()` 的频率不同，块边界不要求跨 worker 对齐；后处理只依赖墙钟
时间，不依赖块序号或训练 step 序号。

### Trace 元数据

每个 worker 目录写入元数据，至少包含：

- `component`、`rank`、`pid`；
- `wall_time_ns` 和 `perf_counter_ns` 锚点；
- `CUDA_VISIBLE_DEVICES`；
- `torch.cuda.current_device()`；
- GPU name 和 UUID；
- profiler schedule 参数；
- worker 启停时间。

物理 GPU 使用 UUID 作为稳定主键，GPU 4-7 只是本次运行的显示标签。后处理不得假设
worker 内部 CUDA device 0 等于物理 GPU 0。

### 跨块解析

后处理读取 worker 目录中的全部 `*.pt.trace.json`，而不是只读取第一个文件。每个块
保留原始 trace 时间戳，并结合时间锚点转换到墙钟时间。转换后必须满足：

- 同一 worker 的块按时间单调递增；
- 块内 kernel 区间不出现负时长；
- 物理 GPU UUID 可以解析；
- 块间重叠和缺口被记录到覆盖率报告。

## CPU 每核利用率口径

对于线程 `t` 的相邻采样 `i-1` 和 `i`：

```text
delta_cpu_s(t, i) =
  max(jiffies(t, i) - jiffies(t, i-1), 0) / CLK_TCK
```

该增量归入采样 `i` 中 `/proc/<pid>/task/<tid>/stat` 的 `processor` 字段。对 1 秒
时间桶 `b` 和逻辑核 `c`：

```text
cpu_core_util_pct(b, c) =
  100 * sum(delta_cpu_s assigned to c in b) / bin_duration_s
```

只统计识别为 RLinf 主进程、Ray actor/rollout/env worker 及其子进程的线程，排除机器上
其他任务。线程消失视为正常生命周期结束；新线程在取得两个连续样本后才产生利用率。

如果线程的 `processor` 在相邻样本间变化，则记录一次迁核。由于无法观察间隔内部的
调度过程，整个 jiffies 增量仍归入后一个核心。这是低开销逐核归因的明确近似。原始值
不裁剪；绘图颜色范围限制在 0-100%，超过 100% 的桶在质量报告中列出。

## GPU 指标口径

### CUDA kernel busy

对每个物理 GPU 和 1 秒桶，收集该 GPU 上所有 RLinf worker 的 CUDA kernel 区间，先
求区间并集，再计算：

```text
kernel_busy_pct = 100 * union(kernel_intervals).duration / bin_duration
```

使用区间并集可以正确处理同一 stream、不同 stream 和不同进程的重叠，结果范围为
0-100%。该指标表示至少一个 CUDA kernel 正在执行的时间比例，不表示平均活跃 SM
比例。

### 估算 SM occupancy

对 worker 在时间桶内与 kernel 区间的交集，使用 trace 字段
`est. achieved occupancy %` 做持续时间加权：

```text
est_sm_occupancy_pct =
  sum(kernel_overlap_duration * kernel_est_occupancy)
  / sum(kernel_overlap_duration)
```

该指标只在存在带 occupancy 字段的 CUDA kernel 时产生。无 kernel 或字段缺失的桶写为
`NaN`，不能写成 0%。它用于比较不同 worker、阶段和时间段的 kernel launch 资源特征，
不能解释为真实设备 `SM_ACTIVE`。

设备级 occupancy 可以作为派生估算数据输出，但正式图优先显示 worker 级热力图，避免
同一 GPU 上并发 worker 的加权结果被误读成硬件计数器。

## 阶段窗口

阶段窗口优先从现有数据生成：

- `rollout_generation_timestamps`；
- `env_sim_timestamps`；
- actor training profiler `record_function`；
- GIPO runner 或训练日志中的 global step 边界。

后处理输出 `phase_windows.csv`，字段包含 `phase`、`step`、`start`、`end` 和
`source`。如果某一类精细阶段缺失，仍生成资源图，但仅标注能够可靠推导的范围，并在
覆盖率报告中说明缺失来源。

## 文件布局

本次运行的所有资源数据位于：

```text
<runner.logger.log_path>/resource_profile/
├── metadata.json
├── cpu/
│   └── thread_core_samples.csv
├── torch/
│   ├── actor_rank_0/
│   ├── rollout_rank_0/
│   └── env_rank_0/
├── derived/
│   ├── cpu_core_1s.csv
│   ├── gpu_worker_1s.csv
│   ├── gpu_device_1s.csv
│   ├── phase_windows.csv
│   └── coverage.json
├── resource_utilization.png
└── resource_utilization.pdf
```

实际目录包含配置所启动的全部 rank。`metadata.json` 保存运行参数、采样间隔、聚合桶
大小、worker/GPU 映射和总 trace 大小。`coverage.json` 保存每个数据源的开始时间、
结束时间、缺口、警告和有效覆盖率。

## 图表设计

图表使用一个共享横轴和三个上下排列的面板：

1. **RLinf per-core CPU utilization**
   - 横轴为完整运行墙钟时间；
   - 纵轴为逻辑 CPU 0-111；
   - 颜色为 1 秒桶内 RLinf 线程利用率；
   - 根据已知 affinity/core group 添加水平分隔线。

2. **CUDA kernel busy by physical GPU**
   - GPU 4、5、6、7 各一条曲线；
   - 纵轴固定为 0-100%；
   - 数据来自所有 worker kernel 区间的设备级并集。

3. **Estimated SM occupancy by worker**
   - 每行一个 actor、rollout 或 env rank；
   - 颜色为持续时间加权 occupancy；
   - 无有效 kernel 的时间桶显示为空白。

三个面板共同绘制 warmup、GIPO step 1-8、rollout 和 actor training 边界。输出 PNG
用于快速检查，PDF 用于报告。图例和标题必须使用“CUDA kernel busy”和
“Estimated SM occupancy”，不得使用“SM_ACTIVE”。

## 启停与故障处理

- 采集功能通过显式环境变量或配置开启，默认关闭。
- CPU 采样器必须在训练启动前运行，并通过 stop file 或信号正常停止和 flush。
- profiler 停止时导出最后一个未满块；单个 worker 失败不阻止其他 worker 保存数据。
- PID/TID 正常消失不报错；无法解析的 stat 行计入跳过计数。
- trace 缺块、时间倒退、GPU 映射失败或 occupancy 字段缺失写入
  `coverage.json`。
- 图中对缺失区间留白，不插值或用 0 填充。
- 记录每个 trace 块和总 trace 大小；若磁盘写入失败，训练侧记录清晰警告并关闭该
  worker 后续 profile，不能让绘图错误掩盖训练退出状态。
- 启动脚本必须使用 trap 停止 CPU 采样器，并保留原始训练退出码。

## 性能与数据量控制

- `record_shapes=False`；
- `profile_memory=False`；
- `with_stack=False`；
- trace 使用固定 profiler-step 数量循环导出；
- CPU 原始采样采用逐行缓冲写入；
- 图表默认使用 1 秒聚合，原始 100 ms CPU 数据仍保留；
- 正式报告同时记录 profiling run 的墙钟时间和 trace 总大小，说明结果来自带
  profiler 开销的运行。

循环分块降低单文件和进程内内存压力，但不会减少完整 8-step 的总事件量。这一限制在
执行指南中必须明确说明。

## 测试策略

### 单元测试

1. CPU stat 解析、jiffies 增量、首次采样、线程退出和迁核标记。
2. Ray/RLinf 进程树识别与 component/rank 归类。
3. 多 trace 块读取、墙钟对齐、时间单调性和缺口检测。
4. 重叠 CUDA kernel 区间并集，验证 busy 不超过 100%。
5. occupancy 持续时间加权和缺失字段的 `NaN` 处理。
6. worker 到物理 GPU UUID 的映射。
7. CPU/GPU 聚合 CSV 和三面板绘图产物。

### GPU smoke test

使用缩短版 GIPO 配置运行少量 step，验证：

- actor、rollout、env 中有预期 trace 块；
- CPU 采样器发现预期 worker；
- 后处理生成全部 derived 文件；
- 图像非空且横轴覆盖训练窗口；
- 训练退出码保持不变。

完整 8-step 运行用于正式数据采集，不加入自动化测试。

## 文档更新

实现完成后更新 `docs/async_gipo_128traj_execution_guide.md`，新增：

- profiling 启动命令；
- 指标口径和限制；
- 输出目录说明；
- 后处理与绘图命令；
- coverage 检查方法；
- `torch.profiler` 结果不等于真实 `SM_ACTIVE` 的警告。

## 验收标准

1. 完整 8-step 运行可连续生成分块 trace，不依赖单个完整 trace 文件。
2. `cpu_core_1s.csv` 包含运行期间观测到的全部逻辑 CPU 核和明确的 RLinf-only
   利用率口径。
3. `gpu_device_1s.csv` 的 `kernel_busy_pct` 始终位于 0-100%。
4. `gpu_worker_1s.csv` 对缺少 occupancy 的桶使用 `NaN`。
5. `coverage.json` 能发现 trace 缺口、GPU 映射失败和 CPU 迁核比例异常。
6. PNG/PDF 图包含 CPU 热力图、GPU 4-7 busy 曲线和 worker occupancy 热力图。
7. 图中所有 GPU 指标名称与本设计定义一致，不出现误导性的真实
   `SM_ACTIVE` 表述。
8. profiling 默认关闭，现有非 profiling 训练行为和测试保持不变。
