# Reset Mode Optimization Design

日期：2026-07-14

## 背景

当前 embodied rollout 中，`EnvWorker.bootstrap_step()` 在 `auto_reset=False`
或 v17 continuous collection 路径下会在每个 rollout epoch 开头调用
`env.reset()`。在 LIBERO 和 RoboCasa 这类 MuJoCo/robosuite 环境中，这个
reset 可能包含两种不同成本的动作：

- full reset：重建或重新加载任务资产、模型、仿真和渲染对象。
- state reset：保留已加载资产，只恢复 episode 初始状态和环境内部状态。

实测显示这两类 reset 成本差异很大：

- LIBERO `reset()` 约 1.10-1.31s，`set_init_state()` 约 0.0017s，close/recreate
  额外约 1s。
- RoboCasa `hard_reset=True` reset 约 5.3-6.8s，`hard_reset=False` reset 约
  0.24s。

真实 RL 训练中，很多 profiling、async 和单任务配置已经用 `task_ids_filter:
[0]`、`specific_reset_id: 0` 或单个 `task_names` 固定任务。此时任务资产不应
频繁变化，重复 full reset 会把不必要的资产重建成本放进 rollout bootstrap。

## 目标

- 显式区分 full reset 和 state reset。
- 为 embodied env 增加统一 reset 粒度配置。
- 默认保持现有行为，避免静默改变实验结果。
- 在 LIBERO 中，当任务或 BDDL 未变化时跳过资产重建。
- 在 RoboCasa 中，将统一 reset 策略映射到底层 `hard_reset`。
- 增加 reset 粒度和耗时指标，便于确认优化是否生效。
- 添加聚焦单元测试，并保留本地真实环境 smoke 验证路径。

## 非目标

- 不改变 rollout epoch、auto-reset、reward、advantage 或 actor training 语义。
- 不为所有环境一次性实现 state reset。
- 不默认启用轻量 reset。
- 不引入多任务资产池或环境缓存。
- 不把真实 LIBERO/RoboCasa smoke 测试纳入默认 CI。

## 配置与 API

在 `env.train` 和 `env.eval` 下新增统一配置：

```yaml
reset_mode: full
reset_full_on_state_mismatch: true
```

`reset_mode` 支持：

- `full`：保持现有行为，允许重建资产、模型、仿真和渲染对象。
- `state`：要求只恢复 state。若任务或资产需要变化，按
  `reset_full_on_state_mismatch` 处理。
- `task_aware`：任务或资产不变时走 state reset；任务或资产变化时走 full
  reset。

默认值：

- `reset_mode: full`
- `reset_full_on_state_mismatch: true`

环境对象内部支持 `reset(..., reset_mode=None)` 或从自身 config 读取
`reset_mode`。`EnvWorker` 不直接操作底层 simulator；它继续调用环境的
`reset()`，并通过环境返回的信息或环境内部 metrics 记录实际 reset mode。

## LIBERO 设计

LIBERO 当前 `_reconfigure()` 同时承担两件事：

1. 根据 reset state ids 计算目标 task、trial 和 BDDL。
2. 根据目标任务重建子进程环境，并调用 `reset()`、`set_init_state()` 和 settling
   steps。

新设计把这两部分拆开。每次 reset 先计算目标 task/trial/BDDL，再决定是否需要
重建资产：

- `full`：保持当前路径，调用 `reconfigure_env_fns()`。
- `state`：如果目标 task/BDDL 与当前一致，跳过 `reconfigure_env_fns()`，调用
  `set_init_state()` 恢复 trial state；如果目标 task/BDDL 变化，则根据
  `reset_full_on_state_mismatch` fallback 或 fail-fast。
- `task_aware`：目标 task/BDDL 不变时走 state reset；变化时走 full reset。

第一版采用保守 state reset：

- 跳过 `reconfigure_env_fns()`，但仍保留现有 `env.reset(id=env_idx)`、
  `set_init_state()` 和 15 次 zero-action settling。
- 增加实验开关 `libero_state_reset_skip_env_reset: false`。当该开关为 true 且
  同 BDDL 时，可跳过 `env.reset()`，直接执行 `set_init_state()` 相关路径。

保守路径先去掉 close/recreate 成本，风险较低。毫秒级路径作为显式实验开关，
需要额外验证 robosuite 内部计数器、observables、renderer 和 success checker
状态。

## RoboCasa 设计

RoboCasa 底层 `Kitchen` 已有 `hard_reset` 参数：

- `hard_reset=True`：reset 时 reload model、sim 和 render object。
- `hard_reset=False`：reset 时只执行 `sim.reset` 和 robosuite 内部变量重置。

RLinf 当前 `RobocasaEnv.get_env_fns()` 调用 `robosuite.make()` 时没有传
`hard_reset`，因此使用底层默认 `True`。

新设计将统一 reset 策略映射为：

- `reset_mode: full` -> `hard_reset=True`
- `reset_mode: state` -> `hard_reset=False`
- `reset_mode: task_aware` -> `hard_reset=False`

RoboCasa 当前每个子环境初始化后 task 固定，`update_reset_state_ids()` 是 no-op，
因此 `task_aware` 等价于 state reset。若未来支持动态 task 切换，再扩展为任务
变化时重建环境。

配置优先级：

1. 如果显式配置 `reset_mode`，由 `reset_mode` 决定 `hard_reset`。
2. 如果未配置 `reset_mode` 但配置了 `hard_reset`，尊重 `hard_reset`。
3. 两者都未配置时保持现状，使用 `hard_reset=True`。

## 数据流

训练主路径保持现有控制流：

1. `EnvWorker.bootstrap_step()` 调用 `env.reset()`。
2. 环境根据自身 config 和 reset state ids 决定实际 reset mode。
3. 环境返回普通 `(obs, infos)`，不改变 rollout payload。
4. `EnvWorker` 继续发送 bootstrap observation 到 rollout worker。
5. `finish_rollout()` 仍调用 `update_reset_state_ids()`，保持 task/trial 采样频率
   不变。

这意味着本设计只改变 reset 的实现粒度，不改变 RL 数据格式、episode 边界、
reward、bootstrap value 或 advantage 计算。

## 错误处理

默认行为为兼容优先：

- `reset_full_on_state_mismatch: true` 时，`state` 或 `task_aware` 遇到不支持
  state reset、任务变化或资产变化时 fallback 到 full reset，并记录 warning 和
  metric。
- `reset_full_on_state_mismatch: false` 时 fail-fast。错误信息包含 env type、
  env ids、requested reset mode、当前 task/BDDL 和目标 task/BDDL。

`state` 模式下 task/BDDL 变化属于配置不一致；`task_aware` 模式才允许自动 full
fallback。

## 指标

新增或汇总以下指标：

- `env/reset/full_count`
- `env/reset/state_count`
- `env/reset/fallback_count`
- `env/reset/reconfigure_time`
- `env/reset/base_reset_time`
- `env/reset/set_init_state_time`
- `env/reset/settle_time`
- `env/reset/total_time`
- `env/reset/hard_reset_enabled`，用于 RoboCasa。

这些指标应按 train/eval 和 stage 聚合，至少能在日志中解释一次 rollout 的
bootstrap reset 耗时由哪些部分构成。

## 测试计划

单元测试：

- LIBERO fake vector env：
  - `reset_mode=full` 时调用 `reconfigure_env_fns()`。
  - `reset_mode=task_aware` 且 task/BDDL 不变时不调用 `reconfigure_env_fns()`。
  - `reset_mode=task_aware` 且 task/BDDL 变化时调用 `reconfigure_env_fns()`。
  - `reset_mode=state` 且 task/BDDL 变化时按 fallback 或 fail-fast 配置处理。
- RoboCasa env factory：
  - `reset_mode=full` 传入 `hard_reset=True`。
  - `reset_mode=state` 和 `task_aware` 传入 `hard_reset=False`。
  - 未配置 `reset_mode` 时保持 `hard_reset=True`。
  - 未配置 `reset_mode` 但配置 `hard_reset` 时尊重显式 `hard_reset`。
- 配置验证：
  - 非法 `reset_mode` 报错。
  - 默认值为 `full`。

可选真实环境 smoke：

- 使用本地 `libero_openpi`，构造 1 个 LIBERO env，验证同 BDDL state reset 不触发
  reconfigure，并记录 reset 段耗时。
- 使用本地 `robocasa_openpi`，构造 1 个 RoboCasa env，验证
  `reset_mode=state/task_aware` 时底层 `hard_reset=False` 生效。

## 推进顺序

1. 添加配置默认值和校验。
2. 实现 RoboCasa `reset_mode` 到 `hard_reset` 的映射，并补单元测试。
3. 重构 LIBERO reset 决策，先实现保守 state reset，并补 fake env 单元测试。
4. 增加 reset metrics。
5. 用本地 `libero_openpi` 和 `robocasa_openpi` 做手动 smoke 验证。
6. 在 profiling 配置中显式开启 `reset_mode: task_aware`，收集 rollout timing。

## 验收标准

- 默认配置下行为与当前实现一致。
- RoboCasa `reset_mode=state/task_aware` 能使底层 `hard_reset=False`。
- LIBERO 固定任务或同 BDDL reset 时不再调用 `reconfigure_env_fns()`。
- task/BDDL 变化时 `task_aware` 仍会 full reset。
- 单元测试覆盖 reset mode 映射、fallback 和 fail-fast 行为。
- 本地 smoke 结果能显示 state reset 比 full reset 明显更快。
