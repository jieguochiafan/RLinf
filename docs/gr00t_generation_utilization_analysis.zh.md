# GR00T rollout generation GPU 利用率分析与优化方案

## 1. 背景与问题

在 LIBERO + GR00T rollout 运行中，GPU 设置了 80% 的 MPS active thread
percentage。日志确认 `43–393 s` 只有 rollout 在运行，actor 仍在等待 rollout
数据；同时 rollout 开启了两个 env pipeline stage。直觉上，generation 持续执行时
SM 利用率应该长期接近 80%，但 Nsight Systems 曲线的前半段平均只有约 40%。

本次分析要回答三个问题：

1. 约 40% 的平均值是持续在 40%，还是高低利用率交替后的平均？
2. generation 内是 VLM backbone 还是 action head 利用率较低？
3. RLinf 当前执行路径中有哪些空洞可以通过 pipeline 或算子优化消除？

## 2. 数据与实验口径

### 2.1 生产运行的长时间 GPU Metrics

- 原始数据：
  [`nsys-sm-20260722-020345_sm_active_raw.csv`](../nsight_profile/libero_4080mps/nsys-sm-20260722-020345_sm_active_raw.csv)
- 指标：Nsight Systems `SMs Active [Throughput %]`
- 采样率：约 949 Hz
- 确认的 rollout-only 区间：`43 s <= time < 393 s`
- MPS active thread percentage：80%，该区间观测最大值也是 80%
- 所有占比均按相邻采样时间间隔进行 time weighting，而不是简单按样本数计数

另外选择该区间内“一秒平均 SM Active 在 35%–45%”的秒，共 280 秒，专门分析
“平均约 40%”时内部由哪些利用率区间组成。已有派生数据为：

- `nsys-sm-20260722-020345_around40_summary.csv`
- `nsys-sm-20260722-020345_around40_bin_share.csv`

### 2.2 独立 GR00T generation 实验

独立 profiler：
[`gr00t_generation_profile.py`](../toolkits/rollout_eval/profiling/gr00t_generation_profile.py)。

实验配置：

| 项目 | 设置 |
| --- | --- |
| GPU | NVIDIA A800-SXM4-80GB |
| 模型 | `/data1/gaobowen/model/RLinf-Gr00t-SFT-Object` |
| Python | `/data1/gaobowen/RLinf/.venv-libero-gr00t` |
| batch size | 12（对应实验时 192 env / 8 rollout workers / 2 stages） |
| action horizon | 16 |
| denoise steps | 4 |
| warmup / measure | 5 / 10 |
| GPU Metrics 频率 | 1000 Hz |
| MPS 限制 | 未施加 80% 限制；active percentage 设为 100% |

测试使用确定性的随机 LIBERO 形状 observation，复现生产
`GR00TActionModel.predict_action_batch()` 的完整模型路径，但去掉 simulator、Ray
通信和 actor 干扰。脚本为以下范围添加 CUDA Event、NVTX 和 PyTorch record
function：

- CPU observation preprocessing；
- CPU 到 GPU 的 `prepare_input`；
- VLM backbone；
- action head、四次 denoise、state encoder 和 value head；
- action postprocessing。

Nsight GPU 阶段统计不能直接使用 host NVTX 起止时间，因为 CUDA launch 是异步的。
本报告通过 CUDA runtime `correlationId` 将 kernel 归属到其 launch 时所在的 NVTX
范围，再用该组 kernel 的真实 GPU 起止区间对齐 1 kHz hardware metrics。这样可以避免
把 VLM 队列尾部误算到随后开始的 action-head host range。

复现实验命令：

```bash
cd /data1/gaobowen/RLinf_new

sudo -E /data1/gaobowen/RLinf/.venv-libero-gr00t/bin/python \
  toolkits/rollout_eval/profiling/gr00t_generation_profile.py \
  --gpu 1 \
  --no-manage-mps \
  --mps-sm 100 \
  --batch-size 12 \
  --warmup-steps 5 \
  --measure-steps 10 \
  --nsys \
  --gpu-metrics-frequency 1000 \
  --output-dir /tmp/gr00t-profile-b12-no-mps-nsys
```

GPU hardware counters 在该机器上需要 root 权限。命令结束前，父脚本会缓存一部分
子进程输出；模型加载和 Nsight `Processing events` 阶段终端短时间没有输出不表示
卡死。

### 2.3 实验限制

- 独立实验保留了机器上已有的系统 MPS daemon，但 client active percentage 为 100%，
  因而没有复现生产运行的 80% client cap；它不是完全关闭 MPS daemon 的实验。
- 随机 observation 保留模型 tensor shape，但没有包含 simulator、Ray Channel、env output
  到达抖动以及其他 rollout worker 的 CPU 竞争。
- Nsight hardware metrics 的分辨率是 1 ms。VLM、整个 action head 和每步 denoise 有足够
  样本；state encoder、value head 等亚毫秒范围的 SM Active 仅用于定性参考。
- Nsight 注入和 GPU Metrics 采集会影响绝对 wall time。因此阶段归因和利用率结论比单次
  wall latency 更可靠；实施优化后应使用相同工具、相同采集配置做 A/B 对比。

## 3. 实验结果

### 3.1 rollout-only 区间 `43–393 s`

完整 350 秒区间的 time-weighted 结果：

| 指标 | 结果 |
| --- | ---: |
| SM Active 平均值 | 38.01% |
| 最大值 | 80% |
| 精确为 0 的时间占比 | 29.30% |
| 低于 10% 的时间占比 | 33.64% |
| 60%–80% 的时间占比 | 40.88% |

完整分布如下：

| SM Active 区间 | 时间占比 |
| --- | ---: |
| 0–<10% | 33.64% |
| 10–<20% | 6.79% |
| 20–<30% | 5.35% |
| 30–<40% | 3.04% |
| 40–<50% | 3.97% |
| 50–<60% | 6.32% |
| 60–<70% | 12.99% |
| 70–<80% | 27.89% |
| 80–<90% | 0.02% |
| 90%–100% | 0% |

这说明 38% 并不是 GPU 稳定运行在约 40%，而是大量接近空闲的区间与大量接近 MPS
上限的区间交替出现。

### 3.2 “一秒均值约 40%”的 280 秒子集

选择一秒平均值在 35%–45% 的 280 秒后，其 time-weighted 平均值为 40.29%，精确
为 0 的时间仍占 25.72%。

| SM Active 区间 | 时间 | 时间占比 |
| --- | ---: | ---: |
| 0–<10% | 83.96 s | 29.99% |
| 10–<20% | 19.26 s | 6.88% |
| 20–<30% | 15.35 s | 5.48% |
| 30–<40% | 9.13 s | 3.26% |
| 40–<50% | 11.90 s | 4.25% |
| 50–<60% | 18.72 s | 6.69% |
| 60–<70% | 38.53 s | 13.76% |
| 70–<80% | 83.10 s | 29.68% |
| 80–<90% | 0.06 s | 0.02% |
| 90%–100% | 0 s | 0% |

其中 60%–80% 合计 43.44%，而低于 10% 占 29.99%。因此“平均约 40%”本质上
仍然是明显的双峰混合，不是 MPS 只给到了 40%。MPS 的 80% 是可用 active thread
上限，并不保证 workload 能持续使用到 80%。

### 3.3 独立 generation 的阶段延迟

Nsight Systems 运行中，每次 generation 的平均阶段时间为：

| 阶段 | 平均时间 |
| --- | ---: |
| generation 总计 | 290.37 ms |
| CPU preprocess | 69.97 ms |
| prepare input / H2D | 2.92 ms |
| VLM backbone CUDA span | 128.54 ms |
| action head CUDA span | 87.88 ms |
| CPU postprocess | 0.75 ms |

CPU preprocess 占单次 generation wall time 的约 24.1%，该阶段没有模型 kernel。
它主要包含：

1. observation 的 Torch、NumPy 和 GR00T modality 格式转换；
2. 两个相机视角的 crop、resize、Tensor/NumPy/PIL 转换和 Eagle image processor；
3. task description 的 chat template、vision-info 处理和 tokenization；
4. state BF16 信息损失复现、归一化、padding 和 mask；
5. 将 batch 拆成 12 个元素后，Python `apply_single()` 串行处理；
6. input ID 和 attention mask padding。

### 3.4 VLM 与 action head 的 GPU 利用率

按 kernel launch 归属及真实 GPU 区间对齐后的数据：

整个 host generation window 的 SM Active 为 54.21%，CPU preprocess 范围内为
0%。模型 GPU 阶段的细分如下：

| GPU 阶段 | Kernel 数/次 | GPU span/次 | Kernel busy | SM Active | SM Issue | Tensor Active |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VLM backbone | 1271 | 128.35 ms | 92.28% | 89.57% | 26.10% | 48.03% |
| Action head | 1636 | 87.67 ms | 60.54% | 47.86% | 14.03% | 31.03% |
| state encoder | 9 | 0.20 ms | 94.92% | 21.00% | 13.00% | 1.00% |
| denoise 0 | 368 | 18.32 ms | 54.27% | 39.66% | 11.40% | 24.99% |
| denoise 1 | 367 | 15.32 ms | 63.97% | 45.95% | 13.47% | 28.59% |
| denoise 2 | 371 | 16.98 ms | 57.21% | 40.90% | 11.85% | 25.77% |
| denoise 3 | 370 | 17.80 ms | 54.29% | 39.52% | 11.67% | 24.45% |
| value head | 16 | 0.62 ms | 34.17% | 4.00% | 2.17% | 1.00% |

结论：

- VLM backbone 已经能较充分使用 GPU，SM Active 接近 90%，kernel 时间线也较连续；
- action head 是模型内部的主要低利用率阶段，SM Active 只有约 48%；
- action head 的绝大部分工作是四步 diffusion denoise；每步包含大量短 GEMM、attention
  和 elementwise kernel，kernel 数多但单个规模小；
- action head 约 39.5% 的 GPU span 没有 kernel 在执行，这些空洞可能来自 kernel launch、
  依赖、allocation 或同步；
- generation 开始处另有约 70 ms CPU preprocessing 空洞。

## 4. RLinf 当前为什么不能隐藏 preprocessing

当前 `MultiStepRolloutWorker.generate_one_epoch()` 对每个 chunk step 和 pipeline
stage 顺序执行，参见
[`huggingface_worker.py`](../rlinf/workers/rollout/hf/huggingface_worker.py#L567)：

```text
await recv_env_output(stage N)
    -> predict(stage N)
    -> send_rollout_result(stage N)
    -> await recv_env_output(stage N+1)
```

`predict()` 又同步调用完整的
[`GR00TActionModel.predict_action_batch()`](../rlinf/models/embodiment/gr00t/gr00t_action_model.py#L564)，
后者内部包含：

```text
CPU preprocess
    -> H2D
    -> VLM
    -> action head
    -> D2H / CPU postprocess
```

因此，即使下一 stage 的 env output 已经在 Channel 中等待，同一个 rollout worker 的
Python/asyncio 执行线程仍被当前同步 `predict()` 占用，无法提前开始下一 batch 的 CPU
preprocess。

`pipeline_stage_num=2` 当前解决的是 env simulation 与 rollout generation 的 stage
交错，不会自动拆开一次 generation 内部的 CPU preprocessing 和 GPU inference。不同
rollout worker 进程可以彼此并行，但单个 worker 内仍是上述串行路径。

## 5. 改进方案

### 5.1 P0：先补齐生产路径观测

独立实验隔离了根因，但最终优化必须用真实 RLinf workload 验证。建议先把相同 NVTX
范围接入生产 `predict_action_batch()`，记录每个 rank、stage 和 batch 的：

- queue/recv wait；
- preprocess；
- H2D；
- VLM；
- action head 及每步 denoise；
- D2H/postprocess；
- batch size、p50/p95 latency、kernel busy 和 SM Active。

生产 trace 应至少覆盖数十秒稳定 rollout，并单独标记 bootstrap-value generation，避免
把它和正常 action generation 混合统计。历史实验的 batch size 是 12；如果配置中的 env
数量或 worker 数已经变化，需要按当前实际 batch 重测。

### 5.2 P1：降低 CPU preprocessing 成本

这是风险较低、可以独立验证的一组优化：

1. **缓存静态语言处理结果。** LIBERO task description 在一个 episode 内通常不变，按
   instruction 缓存 chat template 和 tokenization，避免每一步为 12 个样本重复执行。
2. **批量化 image preprocessing。** 避免 `apply_batch()` 拆成 12 个 Python 元素串行
   `apply_single()`；尽量一次处理 `[B, V, H, W, C]`。
3. **减少数据格式往返。** 合并 Torch -> NumPy -> PIL -> NumPy -> Torch 转换；如果
   Eagle processor 接受 Tensor，则直接走 batched Tensor 路径。
4. **避免不必要的复制与 mutation。** 保留为了训练一致性而做的 BF16 信息截断，但避免
   修改原始 `env_obs`，并检查 `.cpu()` 是否造成额外同步或复制。
5. **pinned memory + async H2D。** CPU 输出使用 pinned buffer，`.to(device,
   non_blocking=True)`，并在独立 CUDA copy stream 上预取。

每项优化都应先确认 action、previous logprob、value 和保存到 replay buffer 的
`forward_inputs` 与基线一致。

### 5.3 P2：两个 batch 的 CPU / VLM / action-head 三级流水线

`pipeline_stage_num=2` 时，一个 rollout worker 对应两个独立 env stage，因此可能同时持有
batch A 和 batch B 的 observation。当前实现仍要求 A 的完整 generation 返回后才开始 B
的 preprocessing。建议把一次 generation 拆成三个主要 stage：

```text
P：CPU preprocess + H2D
V：VLM backbone
A：action head + D2H/postprocess
```

目标时间线为：

```text
时间 ---------------------------------------------------------------------->

batch A: preprocess A ---- VLM A ---------------- action head A ---- post A
batch B:                  preprocess B ---------- VLM B ---------------- action head B
                                                ^
                                  VLM A 完成后即可开始 VLM B
```

具体依赖关系为：

- preprocess B 只依赖 batch B observation，可以和 VLM A、action head A 重叠；
- action head A 必须等待 VLM A 的 backbone output；
- VLM B 在语义上不依赖 VLM A，但 VLM 自身 SM Active 已接近 90%，因此先放在同一个
  VLM stream 中串行，避免两个高利用率 VLM 相互竞争；
- VLM B 排在 VLM A 后面，一旦 VLM A 完成即可执行，并尝试与 action head A 重叠；
- action head B 等待 VLM B，同时在 action-head stream 中排在 action head A 后面。

#### 5.3.1 使用 CUDA Event 表达依赖

不应由 CPU 轮询“前一 batch 的 VLM 是否完成”，也不应调用全局
`torch.cuda.synchronize()`。建议使用三个 CUDA stream 和 event：

```text
copy stream:
    H2D A -> h2d_done_A
    H2D B -> h2d_done_B

VLM stream:
    wait h2d_done_A -> VLM A -> vlm_done_A
    wait h2d_done_B -> VLM B -> vlm_done_B

high-priority action stream:
    wait vlm_done_A -> action head A -> action_done_A
    wait vlm_done_B -> action head B -> action_done_B
```

同一个 VLM stream 会自动保证 `VLM A -> VLM B` 的顺序；action stream 通过 event
只等待对应的 backbone output。`action_done` event 再控制 D2H、CPU postprocess 和向
对应 env stage 返回 action。

模型 API 需要拆成边界清晰的方法，例如：

```python
prepared = model.preprocess_action_batch(env_obs)        # CPU only
gpu_inputs = model.copy_preprocessed_to_gpu(prepared)    # copy stream
features = model.forward_backbone(gpu_inputs)            # VLM stream
outputs = model.forward_action_head(features, gpu_inputs)  # action stream
actions, result = model.postprocess_action(outputs)      # event + D2H/CPU
```

#### 5.3.2 仅创建 CUDA stream 还不够

当前 action head 每个 batch 发射约 1636 个 kernel，并在 Python 中推进四步 denoise。如果
仍由一个 Python 线程顺序执行：

```text
VLM A -> 提交完整 action head A -> VLM B
```

那么等 action-head Python 调用返回后再提交 VLM B，可能已经错过与 action head A
重叠的窗口。因此 GPU submission 还需要采用以下方案之一：

1. **两个固定 submission thread。** VLM thread 只驱动 backbone + VLM stream，action
   thread 只驱动 action head + action stream。两个子模块参数只读，但仍需验证 PyTorch
   module、autocast、allocator 和 GR00T transform 的线程安全。
2. **CUDA Graph。** 将固定 shape 的 action head 捕获成 graph，使四步 denoise 能以一次
   快速 replay 提交；单个 scheduler 随后可以及时把 VLM B 提交到另一个 stream。

建议先实现两个 submission thread 验证调度收益，再用 CUDA Graph 降低 action-head
launch overhead。Action stream 应使用较高 priority，优先保证 batch A 尽快返回 action；
VLM B 作为较低优先级工作填充 action-head 间隙。CUDA stream priority 不会抢占已经运行
的长 kernel，因此仍需实测尾延迟。

#### 5.3.3 Buffer、随机数与版本语义

每个 rollout worker 使用深度为 2 的有界队列和两套可复用 buffer：

- CPU/pinned observation buffer；
- GPU model input；
- backbone output；
- noise、chain、logprob、value 和 action output；
- `(chunk_step, stage_id, policy_version)` 元数据。

跨 stream 使用的 tensor 要保持引用并正确调用 `record_stream()`，防止 caching allocator
提前复用内存。Action head 使用随机初始 action 和每步 noise；并发后必须为每个 logical
batch 管理独立 generator 或预生成 noise，确保随机数序列可复现。

还需要处理：

- `env_all_done` 的 dummy result 复用；
- `final_obs` 和 bootstrap value generation；
- DAgger expert 分支；
- actor weight sync 不能发生在同一 logical batch 的 preprocess、VLM 和 action head 之间；
- async worker cancel、异常传播、队列 drain 和 stream/event 生命周期；
- 当前代码会修改 `env_obs["states"]`，pipeline 前应改成局部变量，避免 buffer 间数据竞争；
- 所有 in-flight batch 必须绑定同一个 policy version，并在切换权重前 drain 或建立版本屏障。

#### 5.3.4 性能上限与预期

可以分两步实现和评估。

第一步只重叠 CPU preprocess 与完整 GPU inference。按独立实验数据，CPU preprocess 约
70 ms，H2D + VLM + action head 约 219 ms，preprocess 理论上可被完整隐藏：

```text
290.37 ms / 219.34 ms = 1.32x
```

第二步再让 `VLM B || action head A` 重叠。若三个 stage 使用独立资源且完全无竞争，流水线
稳态周期的理论下限为最长 stage：

```text
max(preprocess=69.97 ms, VLM=128.54 ms, action_head=87.88 ms)
    = 128.54 ms

290.37 ms / 128.54 ms = 2.26x  # 仅为不可达的理想上限
```

VLM 与 action head 实际共享同一张 GPU，其 SM Active 需求相加超过 100%，必然发生资源
竞争；因此实际周期会明显高于 128.54 ms。Overlap 可能填充 action-head kernel gap、提高
总吞吐，也可能拉长当前 batch 的 action latency。必须同时比较 rollout steps/s、action
head p50/p95、env 等待 action 时间和 aggregate SM Active，不能只看 GPU 利用率。

此外，实际收益仍受 env output 到达时间、Channel 通信、bootstrap generation、CPU core
竞争和 pipeline 启动/排空 bubble 影响。上述结果不能直接外推到原始 80% MPS 生产运行。

### 5.4 P3：减少 action-head kernel gap

三级 pipeline 初步完成后，下一主要瓶颈仍是 action head。建议按以下顺序实验：

1. **固定 shape 的 CUDA Graph。** batch、action horizon 和四步 denoise 均较稳定，适合
   capture/replay；需要预分配输入、noise、timestep、chain 和输出 buffer。
2. **`torch.compile` action-head denoise body。** 先单独编译每个 denoise step，检查 graph
   break，再评估 elementwise fusion 和 launch 数下降。
3. **复用临时 tensor。** 避免四步循环内重复 allocation、随机 buffer 构造和 shape/mask
   tensor 创建。
4. **算子融合。** 优先处理短 elementwise、normalization、index/scatter 和 logprob 路径，
   而不是已经较高效的大 GEMM。
5. **增大有效 action-head batch。** 当两个 stage observation 同时可用时，可以评估合并
   batch 后一次执行 action head；需要权衡等待另一个 stage 带来的响应延迟。

减少 denoise step 数会直接改变 policy distribution、logprob 路径和效果，不应作为纯系统
优化；若尝试，必须作为算法消融重新评估 success rate 和训练稳定性。

### 5.5 P4：不要把提高 MPS 上限当成修复

原始 trace 的最大值达到 80%，说明 80% MPS cap 生效；但 cap 只是上限。CPU preprocessing
期间没有 kernel，action head 的小 kernel 和 launch gap 也无法靠提高上限自动填满。

MPS 可以让同一 GPU 上多个独立 CUDA process 的工作互补，但它不能消除单个 process
内部的串行 preprocessing，也不能保证两个 workload 的峰值恰好错开。应先解决 P1–P3，
再根据同卡 actor/rollout 的真实并发情况调 MPS 配额。

## 6. 实施与验证顺序

建议按以下阶段推进，每阶段保留可独立回退的配置开关：

1. **Baseline：** 在真实 RLinf generation 中加入 NVTX 和分段 latency；复现当前吞吐、
   success rate、SM Active 和 action 一致性。
2. **CPU-only 优化：** language cache、batch image transform、减少格式转换；目标是降低
   preprocess p50/p95，不改变 GPU 模型路径。
3. **CPU 双缓冲：** 先只 overlap CPU preprocess 和当前 batch GPU inference；比较
   pipeline stage 1/2、不同 queue depth，以及 CPU affinity 设置。
4. **异步 H2D：** pinned memory + copy stream，确认 timeline 中 copy 与 compute 重叠。
5. **VLM/action-head GPU pipeline：** 增加 VLM/action submission thread、独立 stream 和
   event；比较串行基线与 `VLM B || action head A` 的吞吐和 action 尾延迟。
6. **Action-head graph/compile：** 分别测试 CUDA Graph 与 `torch.compile`，用 Nsight 比较
   kernel 数、kernel busy、SM Active 和 action-head latency。
7. **进一步 batching：** 只有在前述优化稳定后，再评估合并 action-head batch。

验收指标至少包括：

| 类别 | 指标 |
| --- | --- |
| 性能 | rollout steps/s、episodes/s、generation p50/p95、端到端 epoch time |
| GPU | generation/VLM/action-head SM Active、kernel busy、kernel 数、H2D overlap |
| CPU | preprocess p50/p95、worker CPU 利用率、queue wait、线程/核竞争 |
| 内存 | GPU peak、pinned memory、buffer 数量、replay/forward-input 生命周期 |
| 正确性 | action、prev_logprobs、prev_values、forward_inputs、episode success rate |
| 稳定性 | all-done、bootstrap、weight sync、cancel/restart、多 epoch 长跑 |

建议先以数值容差比较相同 observation 和随机种子下 baseline 与 pipeline 的全部输出；确认
结果一致后，再进行至少一个完整训练 epoch 的吞吐和稳定性测试。

## 7. 最终结论

本次数据支持以下判断：

1. rollout-only 区间平均约 38% 并非 MPS 只分配了约 40%，而是接近空闲与接近 80%
   上限的阶段交替造成；
2. generation 内 VLM backbone 利用率较高，主要模型内部低利用率来自 action head 的四步
   denoise；
3. generation 每批还包含约 70 ms 的 CPU preprocessing，并且 RLinf 当前同一 rollout
   worker 必须等前一批完整 generation 返回后，才开始下一批 preprocessing；
4. 第一阶段应先通过深度 2 的 buffer 将 batch B CPU preprocessing 与 batch A GPU
   inference 重叠，理想吞吐上限约 1.32x；
5. 第二阶段应把 VLM 和 action head 拆成 event 驱动的独立 CUDA stage，使 batch B 的
   VLM 尝试与 batch A 的 action head 重叠；无资源竞争时 2.26x 只是不可达理论上限，实际
   收益必须同时考虑 action 尾延迟；
6. 随后通过 CUDA Graph、`torch.compile`、buffer 复用和算子融合进一步减少 action-head
   kernel gap；
7. 所有收益必须在生产 RLinf pipeline 中重新验证，尤其关注 stage 顺序、bootstrap、权重
   同步和数值一致性。
