# Async GIPO 微观全异步设计

日期：2026-07-05

## 背景

RLinf 已经具备宏观层面的异步 embodied 训练能力：环境 worker 和 rollout
worker 可以作为长驻服务运行，部分 actor 路径也可以通过 replay buffer
摄入轨迹数据。当前 async PPO 路径仍然保留较粗粒度的 rollout batch 边界：
rollout 生成会在每个训练 step 重新启动，env/rollout 之间的交互仍然是
epoch 形态，actor 也要等完整 rollout batch 到达后才开始训练。

本设计新增一条基于 GIPO（Gaussian Importance Sampling Policy
Optimization）的 embodied 微观全异步路径。GIPO 属于 PPO 系 actor-critic
目标，面向大量 replay 和 policy lag 场景：它用基于 log-ratio 的 Gaussian
trust weight 替代 PPO 的 hard clipping，使 stale replay 仍然能贡献平滑且
非零的 policy gradient。

参考：https://arxiv.org/abs/2603.03955

## 目标

- 解耦 simulator stepping、policy generation 和 trainer updates。
- 保持 simulator workers 不执行任何神经网络 forward。
- 将生成集中到 rollout workers，并通过 GPU-backed dynamic batching 提升吞吐。
- 使用 GIPO 从 replayed trajectory segments 训练，以处理 policy lag。
- 保留 RLinf 现有环境语义，尤其是 `auto_reset=False`。
- 为控制流和数据格式契约添加聚焦的单元测试。

## 非目标

- 不替换现有 async PPO、SAC 或 DAgger 路径。
- 不实现新的环境 API。
- 第一版 GIPO 微观异步骨架不支持 eval mode。
- 不加入复杂 retry/recovery 语义，只提供明确的 fail-fast 错误。
- 第一版不引入 ragged trajectory tensors。

## 高层架构

新增一条 `async_gipo` embodied training 路径。Runner 启动三个长驻服务：

```text
Env Workers  --obs request-->  Rollout Worker dynamic batcher  --action response--> Env Workers
Env Workers  --trajectory-->   Trainer ingest / TrajectoryReplayBuffer
Trainer      --weight sync-->  Rollout Worker
```

Env Worker 持有 simulator 状态、reset/step 和 trajectory assembly 逻辑，不运行
模型推理。Rollout Worker 持有 policy model 和 GPU。它接收 observation
requests，动态组成 batch，预测 actions，并返回 action 以及 behavior-policy
metadata。Trainer 从 replay 中采样 trajectory segments，重算当前策略的
logprobs 和 values，并应用 GIPO actor-critic objective。

新增 `AsyncGIPOEmbodiedRunner`，而不是修改 `AsyncPPOEmbodiedRunner`。新
runner 创建 request、response、trajectory 和 metric channels，并负责 worker
生命周期、metric draining、checkpointing 和 rollout weight sync。

## 请求与响应协议

新增轻量数据结构，例如放在 `rlinf/data/embodied_async.py`：

```text
InferenceRequest:
  request_id
  env_rank
  stage_id
  env_ids
  obs
  created_at
  policy_version_hint

InferenceResponse:
  request_id
  rollout_rank
  actions
  rollout_result
  policy_version
  timing
  error

AsyncTrajectoryEnvelope:
  env_rank
  stage_id
  segment_type: "fixed_horizon" | "episode"
  auto_reset
  trajectory
  completed_at
  last_policy_version
```

`request_id` 必须在 env ranks 和 stages 之间全局唯一，例如
`env_rank:stage_id:local_step:uuid`。Response 通过稳定的 channel key
定向路由，例如 `action:{env_rank}:{stage_id}:{request_id}`。

Env Worker 发送 `InferenceRequest` 后，await 对应 response。这个等待会让出
Ray async actor 的执行权，但物理环境在拿到 action 之前不会执行下一步。

## 动态批处理

Rollout Worker 维护一个本地 pending deque。serve loop 先阻塞等待第一个
request，记录 `first_request_time`，然后 drain 当前已经到达的 requests，直到
满足任一条件：

```text
len(pending) >= target_batch_size
or now - first_request_time >= max_wait_time
```

Flush 时，worker 沿 batch 维合并 request observations，调用现有 HuggingFace
rollout `predict()` 路径，再按照 request sizes 将生成的 `RolloutResult`
拆回去。Response 包含：

- actions
- behavior `prev_logprobs`
- 可用时的 behavior `prev_values`
- 训练用 `forward_inputs`
- rollout policy `versions`

第一版只处理 train mode，不混合不同 model instances 或 modes 的 requests。
Channel 的 `maxsize` 配置提供 backpressure，避免 requests 无界增长。

## 轨迹边界语义

本设计保留当前 RLinf 的 `auto_reset` 行为。

对于 `auto_reset=False`，done 信号不会立即 flush trajectory。Worker 会继续
执行，直到达到 `max_steps_per_rollout_epoch` 或 `max_episode_steps`，然后发出
一个 fixed-horizon segment。该 segment 保留 `dones`、`terminations` 和
`truncations`；trainer 或 actor-side processing 按照现有 PPO/GRPO 路径的方式
计算 `loss_mask`。

对于 `auto_reset=True`，done 信号可以结束单个 episode。Env Worker reset 该
env，并可以流式发送 completed episodes。为了避免 replay 写入过碎，第一版可以
按 stage 聚合 completed episodes，或者按配置的 minimum send batch flush。短
episode 仍然通过现有 `Trajectory` 字段表达，不引入新的 ragged format。

因此 replay payload 是 trajectory segment，不一定总是完整 episode：

```text
AsyncTrajectoryEnvelope.payload = fixed_horizon_segment | completed_episode
```

## GIPO 训练语义

Trainer 基于 replay buffer，但仍然是 PPO 系 actor-critic，而不是 SAC。Replay
entries 必须保留：

- `forward_inputs.action`
- behavior `prev_logprobs`
- behavior/proximal `prev_values`
- behavior `versions`
- `rewards`、`dones`、`terminations`、`truncations`
- 已知时的 `loss_mask` 和 `loss_mask_sum`
- `collect_transitions=true` 时的 `curr_obs` 和 `next_obs`

训练时从 replay 中采样 segments，将其处理为现有 `[T, B, ...]` rollout batch
格式，重算当前 policy logprobs，并构造：

```text
log_r = current_logprobs - prev_logprobs
rho = exp(log_r)
rho_bar = stop_gradient(clamp(rho, rho_min, rho_max))
omega = exp(-0.5 * (log(rho_bar) / sigma)^2)
policy_loss = -mean_masked(omega * rho * advantages)
```

Gaussian trust weight 在反向传播中作为常量系数处理。这对应 GIPO 论文中的
log-ratio trust-weighted surrogate。

注册新的 policy loss，例如 `loss_type: gipo_actor_critic`，并尽可能复用当前
actor-critic 的 value loss、entropy bonus、loss masks 和 metric utilities。

Advantage 处理方式可配置：

```yaml
algorithm:
  loss_type: gipo_actor_critic
  adv_type: gae
  gipo:
    advantage_source: recompute
    target_batch_segments: 8
    max_policy_lag: null
    gaussian_sigma: 1.0
    rho_min: 0.0067
    rho_max: 148.0
    normalize_advantages: true
```

默认 `advantage_source` 为 `recompute`：trainer 在 sampled replay 上重算 value
和 GAE，以降低 stale value bias。当某个模型路径无法可靠重算 values 时，可以
保留显式的 `stored` fallback mode。

默认情况下，staleness 只记录、不阻塞。Trainer 记录 `current_version -
versions`，只有配置了 `max_policy_lag` 时才丢弃超过阈值的 samples。

## Runner 生命周期

`AsyncGIPOEmbodiedRunner` 执行流程如下：

```text
init workers
sync initial rollout weights
start rollout.serve_inference(...)
start env.interact_gipo_async(...)
start actor.recv_trajectories_async(...) or internal ingest loop
loop:
  actor.run_training()
  update global_step
  request rollout weight sync at configured interval
  drain metrics
  checkpoint when due
stop all services
wait handles
```

Worker stop 语义遵循现有 async workers：每个 service 持有一个 task，`stop()`
取消 task 或设置 stop flag，runner 等待对应 handles。Rollout service 在退出前
尽可能 flush pending requests。Env workers 必须能在等待 action 时退出。
Trainer 的 replay-buffer wait loops 必须定期检查 stop state。

## 错误处理

第一版采用 fail-fast：

- 如果在 `inference_timeout_s` 前没有收到 inference response，Env Worker 抛出
  包含 `request_id`、`env_rank` 和 `stage_id` 的异常。
- 如果 rollout prediction 失败，rollout service 要么为所有 pending requests
  写入 error responses，要么让 worker task 带 pending request ids 失败。
- Queue backpressure 由 channel `maxsize` 处理；不允许无界 request 或
  trajectory accumulation。
- Weight sync 错误直接导致训练失败，而不是静默服务混合模型状态。

## 指标

至少记录：

- `rollout/request_qsize`
- `rollout/batch_size_mean`、`rollout/batch_size_p50`、`rollout/batch_size_p95`
- `rollout/max_wait_hit_rate`
- `rollout/target_batch_hit_rate`
- `rollout/inference_latency`
- `env/action_wait_time`
- `env/step_time`
- `train/replay_buffer_size`
- `train/policy_lag_mean`、`train/policy_lag_p95`、`train/policy_lag_max`
- `train/log_ratio_mean`、`train/log_ratio_std`、`train/log_ratio_p95`
- `train/gipo_weight_mean`、`train/gipo_weight_min`

## 测试

添加聚焦的单元测试。

1. Dynamic batching：
   - target batch size 触发 flush
   - max wait time 触发 flush
   - stop 能确定性地 flush 或 cancel pending work

2. Request/response routing：
   - 多个 env ranks、stages 和 request ids 不会串 response
   - response keys 能路由回正确的 Env Worker waiter

3. Data-flow format：
   - `InferenceRequest.obs` merge 后保持正确 batch 维度
   - `RolloutResult` split 后保留 actions、logprobs、values、versions 和
     nested `forward_inputs`
   - `AsyncTrajectoryEnvelope -> TrajectoryReplayBuffer -> sample` 保留必需字段
   - nested `forward_inputs`、`curr_obs` 和 `next_obs` 保持在 CPU，且符合预期的
     `[T, B, ...]` 或 `[B, ...]` shape 约定
   - `versions` 与 `prev_logprobs` 对齐，可用于 policy lag 和 log-ratio metrics

4. `auto_reset=False`：
   - done 不会立即 flush
   - fixed-horizon segment 只在达到配置 horizon 后 flush
   - 保留 `dones`、`terminations` 和 `truncations`，用于 loss mask 计算

5. GIPO loss：
   - 极端 log-ratios 产生有限 weights
   - loss mask 会排除无效 samples
   - 当 Gaussian weight 非零时，落在 PPO clip 区间外的 stale samples 仍然有非零梯度

6. Runner smoke：
   - fake env、fake rollout 和 fake actor services 可以启动
   - 一个 training step 可以完成
   - 所有 services 都能 stop，并且 handles 被等待

## 实现约束

将设计转化为实现计划时遵循以下约束：

- 将 async request 和 envelope dataclasses 放在 `rlinf/data/embodied_async.py`。
- 新增 runner，命名为 `AsyncGIPOEmbodiedRunner`。
- 新增 GIPO service methods 或 subclasses，不要原地改变现有 async PPO runner 行为。
- 默认 `algorithm.gipo.advantage_source` 为 `recompute`。当模型无法重算 values 时，
  保留 `stored` 作为显式 fallback mode。
- 对于 `auto_reset=True`，先使用 stage-level completed-episode aggregation。
  等第一批 data-flow tests 通过后，再添加更细粒度的 packing。

