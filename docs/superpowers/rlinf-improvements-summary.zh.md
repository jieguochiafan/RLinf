# 相对原版 RLinf 的系统改进总结

## 总体结论

本仓库的改进主要集中在 embodied RL 的性能和工程能力，而不是增加新的模型。相对原版 RLinf，当前版本更容易评测、定位瓶颈和优化环境—推理—训练流水线。

主要改进包括：

1. 无 Ray 的轻量 rollout 评测与性能测试。
2. 面向环境长尾的并行和延迟感知调度。
3. 基于 MPS/MIG 的 GPU 细粒度分配与自动配比。
4. Async GIPO 微粒度异步训练。
5. LIBERO、RoboCasa 的 reset 优化。
6. 更完整的 profiling 和诊断工具。

## 核心改进

### 1. 轻量评测与诊断

原版 embodied eval 主要依赖 Ray runner/worker，启动和调试成本较高。当前版本新增 `toolkits/rollout_eval/`，可以直接执行：

```text
env -> observation -> model -> action -> env.step
```

工具支持接口检查、固定 seed 稳定性检查、吞吐和延迟统计，并可分别测量 env-only、model-only 和并发 pipeline。

此外还增加了：

- LIBERO 单步延迟 profiler 和调度 benchmark。
- RoboCasa + OpenPI 的 MuJoCo diagnostics。
- actor training profile 和结构化 JSON/Markdown 报告。

### 2. 环境长尾调度

多环境 rollout 容易被最慢的 simulator 拖住。当前版本增加了 parallel shard、latency bin-packing 和 latency-balanced pair 等执行方式，根据历史 step latency 改善环境分组和执行顺序，减少 chunk 中的等待和 pipeline bubble。

该方向的主要价值来自延迟感知调度，而不是 CPU 绑核。

### 3. GPU 资源分配与自动编排

原版 `component_placement` 主要决定 worker 位于哪个 node/GPU。当前 resource pool 进一步支持：

- 使用 MPS 为 actor、rollout 等 worker 设置 GPU 配额。
- 绑定预先创建的 MIG UUID。
- 保存 worker-level `resource_pool_plan.json`。
- profile 环境、模型和 actor training，自动搜索 actor/rollout MPS 配比。

自动编排以缩短流水线较慢的一侧为目标：

```text
epoch_time = max(rollout_time, training_time)
```

当前版本主要面向 embodied FSDP 和 MPS，不负责启动 MPS daemon 或创建 MIG 实例。

> **注意：** 使用 MPS 分配 SM 配额前，必须提前在宿主机启动 NVIDIA MPS 服务。RLinf 只为 worker 设置 `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` 等环境变量，不会自动启动或检查 MPS daemon；如果 MPS 服务未启动，配置的 SM 配额不会按预期生效。

### 4. Async GIPO

原版 async PPO 仍存在较粗的 rollout batch 边界。Async GIPO 将训练拆成三个长驻服务：

```text
Env Worker -> observation request -> Rollout dynamic batcher
Env Worker -> trajectory segment  -> Actor replay buffer
Actor      -> weight sync         -> Rollout worker
```

Rollout worker 根据目标 batch size 或最大等待时间动态合批；actor 持续从 replay buffer 训练。GIPO 使用 Gaussian trust weight 处理异步训练中的 policy lag，使旧策略数据仍能提供平滑梯度。

核心 runner、worker、replay 和 loss 已实现，但 eval、完整指标和容错仍属于第一版。

### 5. Reset 优化

LIBERO 和 RoboCasa 的 full reset 会重复重建仿真、渲染和任务资产。当前版本新增：

```yaml
reset_mode: full  # full | state | task_aware
reset_full_on_state_mismatch: true
```

- LIBERO 在 task/BDDL 不变时跳过环境重建。
- RoboCasa 的 state/task-aware 模式使用 `hard_reset=False`。
- 记录 full/state/fallback 次数和各 reset 阶段耗时。

这是收益比较明确的优化：设计中的实测显示 LIBERO 可从秒级 full reset 降到更轻量的 state reset；RoboCasa hard reset 约为 5–7 秒，soft reset 约为 0.24 秒。

### 6. Profiling

当前版本增加了跨 worker timeline，可以观察：

- env 等待 action 和 rollout 等待 observation。
- 模型 predict、actor training 和权重同步。
- 子环境 step、通信、stack 和 reset 耗时。
- CPU per-core、CUDA kernel busy 和估算 occupancy。

这些工具的作用是把端到端耗时拆成环境、推理、训练、通信和 reset 等具体瓶颈。部分一键采集和后处理链路仍需完善。

## CPU 绑核的实际结论

CPU resource pool 已经支持 EnvWorker 和 SubprocVectorEnv 子进程绑核，但实验发现其常规训练收益较小。

此前观察到的主要收益实际上来自限制 OpenMP 线程数，避免多环境进程产生线程 oversubscription。设置：

```bash
export OMP_NUM_THREADS=1
```

可以达到与绑核近似的效果，而当前 Ray worker 在用户未覆盖时默认也会将该变量设为 `1`。因此：

- CPU affinity 不应作为本仓库的核心性能改进。
- 常规训练默认不需要开启 CPU binding。
- CPU affinity 主要保留给 NUMA、作业隔离、严格复现实验或明确存在迁核抖动的场景。
- CPU affinity 对比实验必须保持相同的 `OMP_NUM_THREADS`，否则无法区分线程限制与绑核收益。

resource pool 的主要实际价值应归于 GPU MPS/MIG 配额和可复现的 worker-level plan。

## 未保留的方案

VLM feature cache 曾尝试缓存 OpenPI、GR00T 和 OpenVLA-OFT 的 backbone feature，但最终因一致性、权重失效、集成复杂度和内存占用等问题被移除。当前 runtime 使用 direct-compute baseline，因此 feature cache 不属于最终改进。

## 总结

这批改造最重要的价值可以归纳为：

1. 更容易测量和解释 embodied RL 的性能瓶颈。
2. 通过环境长尾调度、reset 优化和 Async GIPO 提高流水化程度。
3. 通过 MPS/MIG 和自动编排改善 actor/rollout 的 GPU 配比。

CPU 绑核是一次有价值的实验探索，但不是主要收益来源。后续优化重点应放在环境调度、reset、GPU 配比和真正的异步训练流水线上。
