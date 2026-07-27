# Async GIPO resource profile：`TRAJECTORIES_PER_TRAIN=32` 执行 runbook

本文档面向接手任务的 agent，说明如何执行：

```bash
TRAJECTORIES_PER_TRAIN=32 bash tools/run_async_gipo_resource_profile.sh
```

目标是运行 Async GIPO LIBERO/OpenPI 资源利用率 profile，采集 CPU sampler、分块
`torch.profiler` trace，生成后处理 CSV 和三面板 PDF/PNG 图。

## 一句话执行

推荐显式指定 `RUN_DIR`，避免把大 trace 写到 `/tmp`：

```bash
cd /data1/miliang/RLinf

RUN_DIR=/data1/miliang/rlinf_runs/rlinf_async_gipo_resource_32traj_$(date +%Y%m%d_%H%M%S) \
TRAJECTORIES_PER_TRAIN=32 \
bash tools/run_async_gipo_resource_profile.sh
```

如需保留日志：

```bash
cd /data1/miliang/RLinf

RUN_DIR=/data1/miliang/rlinf_runs/rlinf_async_gipo_resource_32traj_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_DIR"

TRAJECTORIES_PER_TRAIN=32 \
RUN_DIR="$RUN_DIR" \
bash tools/run_async_gipo_resource_profile.sh 2>&1 | tee "$RUN_DIR/launcher.log"
```

## 这个变量实际控制什么

`tools/run_async_gipo_resource_profile.sh` 中 `TRAJECTORIES_PER_TRAIN` 默认是 `16`。
设置为 `32` 后，会同步覆盖这些 Hydra 配置：

```text
env.train.v17_continuous_collect.target_trajectories=32
algorithm.replay_buffer.min_buffer_size=32
algorithm.replay_buffer.cache_size=32
algorithm.replay_buffer.sample_window_size=32
+actor.recv_drain_max_trajectories=32
```

含义：每收到约 32 条 rollout trajectory 后进入一次 actor 训练/采样窗口。这个值越小，
actor training 越早、更频繁地出现在 trace 中；值越大，rollout generation 阶段更长，
actor training 相对更稀疏。

## 脚本默认行为

脚本会自动做这些事：

1. 激活环境：`/data1/miliang/RLinf/libero_openpi/bin/activate`
2. 设置 LIBERO/MuJoCo/EGL 相关环境变量。
3. 开启资源 profile：
   - `RLINF_RESOURCE_PROFILE=1`
   - `RLINF_TORCH_PROFILE=1`
   - `RLINF_TORCH_PROFILE_DIR=$RUN_DIR/resource_profile/torch`
4. 启动 CPU sampler：
   - 输出到 `$RUN_DIR/resource_profile/cpu/thread_core_samples.csv`
   - 默认采样间隔 `CPU_INTERVAL=0.1` 秒
5. 启动训练：
   - `python examples/embodiment/train_async.py`
   - 配置：`libero_spatial_async_gipo_openpi_pi05_verify`
   - `runner.max_epochs=8`
   - `runner.max_steps=8`
   - `CUDA_VISIBLE_DEVICES=4,5,6,7`
6. 训练结束后自动运行后处理：
   - `tools/async_gipo_resource_profile.py`
   - `tools/plot_async_gipo_resource_utilization.py`
7. 保留训练退出码；如果训练失败，脚本最终返回训练失败码。

## 运行前检查

### 1. 确认在正确仓库

```bash
cd /data1/miliang/RLinf
pwd
git status --short
```

注意：当前仓库可能有大量用户已有修改。agent 不要 reset/clean，不要删除不属于本次
run 的文件。

### 2. 检查 GPU

脚本默认使用物理 GPU 4、5、6、7：

```bash
nvidia-smi
```

要求：

- GPU 4-7 有足够显存。
- 如果 MPS 是实验前提，确认 MPS 已经由用户或上游 agent 启好。
- 不要杀掉看起来不属于本次 RLinf run 的进程，例如其他用户的 vLLM、token compression
  等任务。

如果只能用两张 GPU，不要直接用这个脚本默认命令；需要额外 Hydra override
`cluster.component_placement.*`，并相应减少 env 数和 batch。此前可行的两卡模式是
actor 用 GPU 5、rollout 用 GPU 4、env 用 GPU 4-5，但这不是本 runbook 的默认路径。

### 3. 检查磁盘

完整 profiling 会产生大 trace。先查目标盘和 `/tmp`：

```bash
df -h /data1 /tmp
```

经验值：

- 256 条 trajectory 的两卡完整 rollout generation trace 曾产生约 `10.84 GiB` torch
  trace，actor training trace 约 `309 MiB`。
- 四卡、8 step、`TRAJECTORIES_PER_TRAIN=32` 的 trace 大小依赖运行时长和 worker 数，
  应预留至少几十 GiB；如果 `PROFILE_ACTIVE_STEPS` 较大或运行变长，继续放大预留。
- 不建议默认写到 `/tmp`，因为 `/tmp` 可能和根分区共用空间。

## 常用执行变体

### 正式 32 trajectory profile

```bash
cd /data1/miliang/RLinf

RUN_DIR=/data1/miliang/rlinf_runs/rlinf_async_gipo_resource_32traj_$(date +%Y%m%d_%H%M%S) \
TRAJECTORIES_PER_TRAIN=32 \
bash tools/run_async_gipo_resource_profile.sh
```

### 降低 trace 单块大小

默认 `PROFILE_ACTIVE_STEPS=25`。如果担心单个 trace 文件或 profiler 内存过大，可以降低：

```bash
cd /data1/miliang/RLinf

RUN_DIR=/data1/miliang/rlinf_runs/rlinf_async_gipo_resource_32traj_active10_$(date +%Y%m%d_%H%M%S) \
TRAJECTORIES_PER_TRAIN=32 \
PROFILE_ACTIVE_STEPS=10 \
bash tools/run_async_gipo_resource_profile.sh
```

说明：降低 `PROFILE_ACTIVE_STEPS` 主要减小单个 trace chunk，不一定显著减少总 trace
体积；完整运行事件量仍会落盘。

### 调整 CPU 采样间隔

默认 `CPU_INTERVAL=0.1` 秒。如果只需要粗粒度 CPU 图，可以放宽：

```bash
cd /data1/miliang/RLinf

RUN_DIR=/data1/miliang/rlinf_runs/rlinf_async_gipo_resource_32traj_cpu02_$(date +%Y%m%d_%H%M%S) \
TRAJECTORIES_PER_TRAIN=32 \
CPU_INTERVAL=0.2 \
bash tools/run_async_gipo_resource_profile.sh
```

### 只做 smoke test

用于验证环境、profile 代码和画图管线，不代表正式结果：

```bash
cd /data1/miliang/RLinf

RUN_DIR=/data1/miliang/rlinf_runs/rlinf_async_gipo_resource_32traj_smoke_$(date +%Y%m%d_%H%M%S) \
TRAJECTORIES_PER_TRAIN=32 \
PROFILE_ACTIVE_STEPS=5 \
bash tools/run_async_gipo_resource_profile.sh \
  runner.max_epochs=1 \
  runner.max_steps=1
```

## 运行中观察

### 训练日志

主训练日志：

```text
<RUN_DIR>/train.log
```

如果执行时用了 `tee "$RUN_DIR/launcher.log"`，还有：

```text
<RUN_DIR>/launcher.log
```

可用下面命令观察：

```bash
tail -f <RUN_DIR>/train.log
```

### GPU

```bash
nvidia-smi
```

预期：GPU 4-7 会出现 Python/Ray worker 进程，rollout generation 阶段 GPU busy 可能较高。

### trace 增长

```bash
du -sh <RUN_DIR>/resource_profile/torch
find <RUN_DIR>/resource_profile/torch -name '*.pt.trace.json' | wc -l
```

### CPU sampler

```bash
tail -20 <RUN_DIR>/resource_profile/cpu/monitor.log
ls -lh <RUN_DIR>/resource_profile/cpu/thread_core_samples.csv
```

## 结束后必须检查

### 1. 脚本退出码

如果 agent 直接运行命令，必须记录最终 exit code。`0` 才表示训练和后处理都没有返回失败。

### 2. 关键产物

```bash
RUN_DIR=<实际 run dir>

test -s "$RUN_DIR/train.log"
test -s "$RUN_DIR/resource_profile/cpu/thread_core_samples.csv"
test -s "$RUN_DIR/resource_profile/derived/coverage.json"
test -s "$RUN_DIR/resource_profile/resource_utilization.png"
test -s "$RUN_DIR/resource_profile/resource_utilization.pdf"
find "$RUN_DIR/resource_profile/torch" -name 'trace_manifest.jsonl' -print
```

主要产物布局：

```text
<RUN_DIR>/resource_profile/
├── cpu/thread_core_samples.csv
├── torch/*/trace_manifest.jsonl
├── derived/cpu_core_1s.csv
├── derived/gpu_device_1s.csv
├── derived/gpu_worker_1s.csv
├── derived/phase_windows.csv
├── derived/coverage.json
├── resource_utilization.png
└── resource_utilization.pdf
```

### 3. coverage 报告

```bash
python -m json.tool "$RUN_DIR/resource_profile/derived/coverage.json" | less
```

重点看：

- 是否有 missing traces。
- 是否有 unmapped GPU/device。
- 是否有 missing occupancy fields。
- CPU coverage 是否覆盖主要训练窗口。
- phase window 是否匹配到 rollout generation 和 actor training。

### 4. trace 大小

```bash
du -sh "$RUN_DIR/resource_profile/torch" "$RUN_DIR/resource_profile"
find "$RUN_DIR/resource_profile/torch" -name '*.pt.trace.json' -printf '%s\t%p\n' \
  | sort -n \
  | tail -20
```

如果 trace 已确认不再需要，可以只删除大 trace，保留 derived CSV 和图：

```bash
rm -rf "$RUN_DIR/resource_profile/torch"
```

删除前必须确认用户允许，且路径必须是本次 run 的精确路径。

## 指标口径和限制

- CPU 利用率是 RLinf-only。采样器读取 Ray/RLinf 相关线程的 `/proc` jiffies，并按后一次
  样本看到的 logical CPU 归因。
- GPU `kernel_busy_pct` 来自 torch profiler 中 CUDA kernel 区间的并集，不是硬件
  `SM_ACTIVE`。
- Worker `est_sm_occupancy_pct` 来自 profiler kernel metadata 的持续时间加权估算，
  不是真实 SM active counter。
- 若需要真实硬件 SM/DRAM/Tensor Core counter，应改用 Nsight Systems、Nsight Compute
  或 DCGM；本脚本不采这些硬件计数器。
- profiling 会显著增加运行时和磁盘占用。正式报告中要说明结果来自带 profiler 开销的运行。

## 常见问题处理

### 没有 rollout generation trace

检查：

```bash
find "$RUN_DIR/resource_profile/torch" -maxdepth 2 -type f | sort
rg -n "gipo_action_generation|generation|torch profiler|profile" "$RUN_DIR/train.log"
python -m json.tool "$RUN_DIR/resource_profile/derived/coverage.json"
```

如果 `torch` 目录已被删除，不能从该 run 重新生成 GPU trace 图；只能重新运行 profile。

### actor training 只有一小段

这通常不是错误：

- runner 级 actor training 时间可能包含等待 replay buffer 的时间。
- actor torch profiler 只在实际训练 compute loop 中 step。
- `TRAJECTORIES_PER_TRAIN=32` 会让训练比 16 更少、更晚一些；如果需要更密集的 actor
  compute，可改回 `16`。

### CPU 面板空白

检查 CPU sampler 是否启动并写出样本：

```bash
ls -lh "$RUN_DIR/resource_profile/cpu/thread_core_samples.csv"
tail -50 "$RUN_DIR/resource_profile/cpu/monitor.log"
```

如果手动跑训练而没有用 wrapper 脚本，通常不会有 CPU sampler 数据。

### 后处理或画图失败

可单独重跑：

```bash
cd /data1/miliang/RLinf
source /data1/miliang/RLinf/libero_openpi/bin/activate

python tools/async_gipo_resource_profile.py "$RUN_DIR" --bin-s 1.0 --num-cpus 112
python tools/plot_async_gipo_resource_utilization.py \
  "$RUN_DIR/resource_profile/derived" \
  --output-prefix "$RUN_DIR/resource_profile/resource_utilization" \
  --num-cpus 112
```

### `/tmp` 或目标盘空间紧张

先只读检查：

```bash
df -h /tmp /data1
du -xh --max-depth=1 /tmp 2>/dev/null | sort -h | tail -40
```

只删除明确属于本次 run 的目录或 trace。不要清理其他用户、其他实验、Ray 当前 session，
除非用户明确授权。

## 给 agent 的交付格式

完成后请向用户报告：

```text
run_dir: <RUN_DIR>
exit_code: <exit code>
trajectory setting: TRAJECTORIES_PER_TRAIN=32
torch trace size: <du -sh resource_profile/torch>
resource profile size: <du -sh resource_profile>
figure:
  <RUN_DIR>/resource_profile/resource_utilization.pdf
  <RUN_DIR>/resource_profile/resource_utilization.png
coverage:
  <RUN_DIR>/resource_profile/derived/coverage.json
notes:
  - 是否有 missing trace / missing occupancy / CPU coverage warning
  - GPU 使用情况和是否开启 MPS
  - 是否保留或删除了大 trace
```

不要只说“执行完成”；必须给出上述可核验路径和退出码。
