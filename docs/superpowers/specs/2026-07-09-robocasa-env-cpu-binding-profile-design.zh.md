# RoboCasa Env CPU Binding Profile 设计

## 背景

端到端 RoboCasa 训练中，`cluster.resource_pool` 对 Env 进程和
RoboCasa 子环境做 CPU binding 后，整体收益不明显。但单独测试
RoboCasa/robosuite 环境时，显式绑核相对 OS 默认调度能提升吞吐。

当前代码里已经有几类可复用信号：

- `EnvWorker` 通过 `Worker.timer` 记录 `run_interact_once`、
  `env_interact_step` 等函数级耗时。
- `RolloutWorker` 记录 `generate_one_epoch`、`predict` 等函数级耗时。
- RoboCasa 子进程会记录每个真实 `robosuite.env.step` 的
  `wall_start_ns`、`wall_end_ns`、`duration_s`。
- `BaseVectorEnv` 会记录 chunk 调度 profile，包括 dispatch、wait、recv、
  stack、affinity、core donation 等耗时。
- resource pool 已能把 `RLINF_ENV_CPU_CORE_GROUPS` 传到 RoboCasa
  子进程，并在 EnvWorker 初始化阶段校验子进程 affinity。

本设计的目标不是先假设 CPU binding 无效，而是把收益链条拆开：

1. 确认单环境绑核收益是否在 RLinf/RoboCasa 子进程路径中仍然存在。
2. 确认多环境 rollout 后收益在哪个边界被抵消。
3. 回到真实端到端配置，从 rollout 内部时序解释 resource pool 收益较少的
   主要原因。
4. 产出可复现 profile 方法和 bottleneck 判定规则。

## 范围

本设计聚焦 RoboCasa + OpenPI embodied rollout 的 profile 和分析，不改变
训练算法、不重构 resource pool 调度器，也不把 profile 结果自动转成新资源
分配策略。若 profile 发现实现 bug 或明显低成本优化点，后续实现计划可以把
修复或优化作为独立任务处理。

使用环境为当前仓库路径下虚拟环境：

```bash
source /data1/miliang/RLinf/robocasa_openpi/bin/activate
```

## 总体策略

采用两阶段 profile：

- 阶段 A：严格 apples-to-apples 归因。固定 env 数、chunk mode、GPU/MPS、
  actor/rollout 配置，只切换 CPU binding 和调度策略。
- 阶段 B：回到当前真实端到端配置。复现收益较少的现象，并用阶段 A 的结果
  解释端到端瓶颈转移或收益抵消。

阶段 A 回答“CPU binding 本身有没有收益，以及收益在哪一层消失”。
阶段 B 回答“当前 resource pool 端到端为什么收益少”。

## 实验矩阵

### 阶段 A：严格归因

所有 case 尽量使用相同的：

- RoboCasa task，例如 `robocasa_closedrawer`
- `total_num_envs`
- `max_steps_per_rollout_epoch`
- `action_repeat_per_chunk_step`
- `actor.model.num_action_chunks`
- rollout/actor placement
- model checkpoint
- GPU 可见设备和 MPS 设置
- Ray 启动方式

建议从小规模开始，例如 8 或 16 env，先保证 profile 数据易读，再扩到 64 env。

| Case | CPU/调度设置 | 目的 |
|---|---|---|
| A0 | OS 默认调度，resource pool 关闭 | 无绑核 baseline |
| A1 | EnvWorker process 绑核，子环境不分组 | 验证 Ray worker 主进程绑核是否影响 |
| A2 | per-env 子进程静态绑核，`sync_time_major` | 对齐单环境 benchmark 的收益来源 |
| A3 | per-env 静态绑核，`latency_balanced_pair`，关闭 donation | 分离 pairing 调度本身收益和开销 |
| A4 | per-env 静态绑核，`latency_balanced_pair`，开启 donation | 判断 core donation 是否帮助或抵消收益 |

关键控制点：

- A0 和 A2 必须保持相同 env 数和 chunk mode，避免把“调度模式差异”误判为
  “绑核收益”。
- A3 和 A4 只差 core donation，避免把 dynamic pairing 和 donation 混在一起。
- 每个 case 至少做 warmup 后测量 3 次，记录均值、p50、p90、p95。

### 阶段 B：真实端到端解释

以当前 RoboCasa resource pool 配置为中心，逐项减少混杂变量：

| Case | 设置 | 目的 |
|---|---|---|
| B0 | 当前规模的无 resource pool 等价配置 | 真实端到端 baseline |
| B1 | 当前 resource pool 配置 | 复现“收益较少” |
| B2 | 只保留 CPU binding，关闭 dynamic donation | 看换核/捐核开销是否抵消收益 |
| B3 | CPU binding 固定，actor/rollout 资源固定 | 判断 bottleneck 是否转移到 rollout/actor/GPU |
| B4 | env 数 sweep，例如 8/16/32/64 | 判断是否存在 env 过饱和、尾延迟或 GPU 渲染瓶颈 |

阶段 B 不要求每个 case 都完整训练。`runner.max_steps=1`、`update_epoch=0`
足够用于 rollout 主路径分析；如需判断 actor training 覆盖效应，再打开最小
actor update。

## Rollout 耗时分解

一个训练 step 的 profile 需要拆成以下层级。

### Runner 层

已有 `EmbodiedRunner` 记录：

- `time/sync_weights`
- `time/generate_rollouts`
- `time/cal_adv_and_returns`
- `time/actor_training`
- `time/eval`
- `time/step`

需要用它判断端到端瓶颈是 rollout、actor、权重同步还是其他阶段。

### EnvWorker 层

需要拆分：

- `bootstrap_step`
- `_send_train_bootstrap`
- `recv_rollout_results`
- `compute_bootstrap_rewards`
- `env_interact_step`
- `send_env_batch`
- `send_rollout_trajectories`
- `store_last_obs_and_intervened_info`
- `finish_rollout`

现有 timer 已覆盖 `run_interact_once` 和 `env_interact_step`，但不足以区分
EnvWorker 在等待 rollout 还是执行 RoboCasa。因此需要补充轻量计时：

- env 等待 rollout result 的 wall time
- env 发送 obs 到 rollout 的 wall time
- env 发送 trajectory 到 actor 的 wall time
- reward model 关闭时确认 reward 路径为 0 或未参与

### RolloutWorker 层

需要拆分：

- `recv_env_output`
- obs merge
- `predict`
- `get_bootstrap_values`
- rollout result split
- `send_rollout_result`
- `generate_one_epoch`

现有 timer 覆盖 `predict` 和 `generate_one_epoch`，但需要补充 channel wait 和
batch merge/split 耗时。否则无法区分“模型推理慢”和“rollout 在等 env”。

### RoboCasa Env 主进程层

`RobocasaEnv.chunk_step` 之后需要拆分：

- `prepare_actions`
- `chunk_step` 总耗时
- `_wrap_obs`
- image flip / `torch.stack`
- state tensor conversion
- done/reward/metric 构造
- auto reset

其中 `_wrap_obs` 和图像 tensor 转换很可能吞掉子进程 step 的收益，必须单独计时。

### SubprocVectorEnv 层

已有 `_last_chunk_profile` 应进入 env metrics。重点字段：

- `operation`
- `env_count`
- `dispatch_s`
- `wait_recv_s`
- `wait_call_s`
- `recv_call_s`
- `time_to_first_ready_s`
- `first_ready_to_last_recv_s`
- `stack_s`
- `total_s`
- `affinity_s`
- `core_donation_count`
- `core_donation_s`
- `core_donation_restore_s`

这些字段用于判断：

- 子进程是否真的更快。
- 父进程是否被尾部 env 拖慢。
- donation 是否频繁改 affinity。
- stack/recv 是否成为新瓶颈。

### RoboCasa 子进程层

已有 `robocasa_env_step` JSONL 事件应保留，按以下维度聚合：

- rank
- stage
- chunk_step
- local_env
- global_env
- vector_step
- chunk_action_index
- repeat_index
- child_pid
- cpu_affinity
- duration_s
- wall_start_ns
- wall_end_ns

需要计算：

- 每个 env 的 mean/p50/p90/p95 step latency
- 每个 chunk 的最早开始、最晚结束、critical path
- 子进程实际并发度
- 每个 CPU core/group 上的 env 分布
- 每个 chunk 的尾延迟占比

## 输出数据

每次运行输出到独立目录，例如：

```text
logs/robocasa_cpu_binding_profile/<case>/<timestamp>/
```

建议包含：

- `resolved_config.yaml`
- `resource_pool_plan.json`
- `runner_metrics.jsonl`
- `env_rank_<rank>.jsonl`
- `rollout_rank_<rank>.jsonl`
- `worker_timer_metrics.jsonl`
- `summary.json`
- `timeline.csv`
- `bottleneck_report.md`

`summary.json` 至少包含：

- end-to-end step time
- rollout wall time
- actor training wall time
- env_interact_step total and per-rank max
- rollout predict total and per-rank max
- RoboCasa child step total, mean, p95
- SubprocVectorEnv wait/recv/stack/affinity/donation breakdown
- Env/Rollout waiting ratio
- estimated bottleneck stage

## 时间线对齐

所有 timeline 事件必须能通过 wall clock 对齐。

已有事件使用 `wall_ns` 或 `wall_start_ns/wall_end_ns`，rollout generation 事件也有
`wall_ns`。对齐时以 run start 的最小 wall timestamp 作为 `t0`，生成统一
`relative_start_s` 和 `relative_end_s`。

同一条 timeline 至少展示：

- runner 阶段
- EnvWorker chunk
- RoboCasa 子进程 step
- Rollout predict

关键视图：

- 每个 chunk 的 critical path 甘特图
- Env child step latency CDF
- SubprocVectorEnv breakdown stacked bar
- Rollout vs Env wait ratio

## Bottleneck 判定规则

按以下规则从上到下判断。

| 观察结果 | 结论 |
|---|---|
| 子进程 `robocasa_env_step` 变快，但 `env_interact_step` 不变 | 主进程调度、stack、obs wrap、tensor conversion 或 auto reset 吃掉收益 |
| `env_interact_step` 变快，但 `generate_rollouts` 不变 | rollout/model 或 channel 等待是瓶颈 |
| Env 和 rollout 都变快，但 step time 不变 | actor training、sync weights 或 runner 其他阶段覆盖收益 |
| `time_to_first_ready_s` 很低，`first_ready_to_last_recv_s` 很高 | 长尾 env 拖慢 chunk critical path |
| `core_donation_count` 高，`core_donation_s` 或 restore 高 | 动态换核开销可能抵消收益 |
| donation 时间低但收益低 | 可能是 cache/NUMA/GPU 渲染或模型推理瓶颈，而不是换核调用本身 |
| Env 子进程快，timeline 中存在大段空白 | Ray/channel 等待、父进程阻塞或 GPU 队列等待 |
| A 阶段收益明显，B 阶段收益小 | 端到端瓶颈已转移或真实配置有额外混杂变量 |

## 主要假设与待验证点

1. 单环境 benchmark 的收益主要来自稳定 CPU affinity 降低迁移和调度抖动。
2. 多环境端到端中，收益可能被以下因素抵消：
   - action_repeat 导致 GPU offscreen rendering 或 MuJoCo 内部路径成为瓶颈。
   - 子进程 step 变快后，EnvWorker 主进程 `_wrap_obs`、stack 和 tensor conversion
     成为瓶颈。
   - Rollout OpenPI 推理是更长的 critical path。
   - `latency_balanced_pair` 的动态 affinity 和 core donation 带来额外开销或
     cache 迁移。
   - actor training 或 weight sync 覆盖 rollout 提升。
   - 当前 baseline 与 resource pool case 在 env 数、chunk mode 或 placement 上
     不完全等价。
3. 如果 A2 相比 A0 没有收益，则说明单环境 benchmark 结果没有延续到 RLinf
   RoboCasa 子进程路径，需要优先检查 benchmark 与训练配置差异，例如相机数量、
   action repeat、reset、task、seed、render GPU 和 observation wrapping。

## 验证流程

1. 先运行最小 smoke case，确认 RoboCasa + OpenPI 路径能在
   `robocasa_openpi` 环境中启动。
2. 跑 A0 和 A2，小规模 8 env，确认 per-env CPU binding 是否降低
   `robocasa_env_step` p50/p95。
3. 若 A2 子进程变快，检查 `env_interact_step` 是否同步变快。
4. 若 Env 变快，检查 `generate_rollouts` 和 `predict` 是否成为 bottleneck。
5. 扩展到 A3/A4，量化 `latency_balanced_pair` 和 donation 的开销。
6. 跑 B0/B1 复现真实端到端收益较少。
7. 用 B2/B3/B4 验证收益较少是 donation、瓶颈转移、env 过饱和还是资源竞争。
8. 生成 `bottleneck_report.md`，列出证据链和主因排序。

## 预期交付物

- 一组 profile 配置或运行脚本，覆盖 A/B 实验矩阵。
- 轻量 profile 埋点，优先复用现有 `Worker.timer` 和 JSONL timestamp。
- 后处理脚本，将 Env/Rollout/Actor 内部事件对齐成 summary 和 timeline。
- 图表：
  - end-to-end phase breakdown
  - Env child step latency CDF
  - chunk critical path timeline
  - SubprocVectorEnv breakdown
- `bottleneck_report.md`，明确回答：
  - CPU binding 在哪一层有效。
  - 端到端收益少的主要原因。
  - 下一步优化优先级。

## 非目标

- 不在本轮自动选择最优 resource pool plan。
- 不重写 `latency_balanced_pair` 算法。
- 不引入高开销常驻 profiler 作为默认训练行为。
- 不把 Nsight Systems 或 Nsight Compute 作为必需项；它们只作为深挖 GPU 瓶颈时的
  可选工具。

## 风险与缓解

| 风险 | 缓解 |
|---|---|
| profile 埋点改变时序 | 默认使用轻量 `perf_counter` 和 JSONL，torch profiler 只在短窗口启用 |
| 日志过大 | 子进程 step 事件按 profile 开关启用，控制子进程事件采样频率 |
| Ray worker 日志分散 | 输出统一 run dir，并保存 resource pool plan 和 resolved config |
| apples-to-apples 不严格 | 每个 case 保存 resolved config，后处理检查关键字段一致 |
| GPU/MPS 配置不一致影响对比 | 每次运行记录 `CUDA_VISIBLE_DEVICES`、MPS 百分比和 resolved config |
| RoboCasa reset 或任务随机性带来波动 | 固定 seed，至少 3 次重复，报告 p50/p95 和方差 |

## 接受标准

完成后应能用数据回答以下问题：

1. 在 RLinf RoboCasa 子进程里，per-env CPU binding 相比 OS 默认调度是否降低
   `robocasa_env_step` latency。
2. 如果降低，降低幅度是否传递到 `env_interact_step`。
3. 如果没有传递，具体被哪一段主进程或调度开销抵消。
4. 如果传递到 EnvWorker，端到端收益是否被 rollout/model、actor training 或
   weight sync 覆盖。
5. 当前 resource pool 配置中，收益较少的主因排序是什么。
6. 后续最值得优化的 1 到 3 个点是什么，以及每个点的理论收益上限。
