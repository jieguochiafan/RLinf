# Async GIPO Micro-Asynchrony Design

Date: 2026-07-05

## Context

RLinf already has macro-level asynchronous embodied training: environment and
rollout workers can run as long-lived services, and some actor paths ingest
trajectories through a replay buffer. The current async PPO path still keeps a
coarse rollout batch boundary: rollout generation is restarted per training
step, the env/rollout exchange is epoch-shaped, and the actor trains after a
complete rollout batch arrives.

This design adds a new embodied path for micro-level full asynchrony based on
GIPO, Gaussian Importance Sampling Policy Optimization. GIPO is a PPO-family
replay-heavy objective designed for policy lag: it replaces PPO hard clipping
with a log-ratio Gaussian trust weight so stale replay can still contribute a
smooth, non-zero policy gradient.

Reference: https://arxiv.org/abs/2603.03955

## Goals

- Decouple simulator stepping, policy generation, and trainer updates.
- Keep simulator workers free of neural network forward passes.
- Centralize generation in rollout workers with GPU-backed dynamic batching.
- Train from replayed trajectory segments using GIPO to handle policy lag.
- Preserve existing RLinf environment semantics, especially `auto_reset=False`.
- Add focused unit tests for control flow and data-format contracts.

## Non-Goals

- Do not replace the existing async PPO, SAC, or DAgger paths.
- Do not implement a new environment API.
- Do not support eval mode in the first GIPO micro-async skeleton.
- Do not add retry/recovery semantics beyond clear fail-fast errors.
- Do not introduce ragged trajectory tensors in the first version.

## High-Level Architecture

Add a new `async_gipo` embodied training path. The runner starts three
long-lived services:

```text
Env Workers  --obs request-->  Rollout Worker dynamic batcher  --action response--> Env Workers
Env Workers  --trajectory-->   Trainer ingest / TrajectoryReplayBuffer
Trainer      --weight sync-->  Rollout Worker
```

The Env Worker owns simulator state, reset/step, and trajectory assembly. It
does not run model inference. The Rollout Worker owns the policy model and GPU.
It receives observation requests, batches them dynamically, predicts actions,
and returns action plus behavior-policy metadata. The Trainer samples trajectory
segments from replay, recomputes current logprobs and values, and applies the
GIPO actor-critic objective.

Create a new `AsyncGIPOEmbodiedRunner` rather than changing
`AsyncPPOEmbodiedRunner`. The new runner creates request, response, trajectory,
and metric channels, then manages worker lifecycle, metric draining,
checkpointing, and rollout weight sync.

## Request and Response Protocol

Introduce lightweight data structures, for example under
`rlinf/data/embodied_async.py`:

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

`request_id` must be unique across env ranks and stages, for example
`env_rank:stage_id:local_step:uuid`. Responses are routed back through a stable
channel key such as `action:{env_rank}:{stage_id}:{request_id}`.

The Env Worker sends an `InferenceRequest`, then awaits its matching response.
This wait yields the Ray async actor, but the physical environment does not step
again until its action is available.

## Dynamic Batching

The Rollout Worker maintains a local pending deque. Its serve loop blocks for
the first request, records `first_request_time`, then drains immediately
available requests until either condition is true:

```text
len(pending) >= target_batch_size
or now - first_request_time >= max_wait_time
```

On flush, the worker merges request observations along batch dimension, calls
the existing HuggingFace rollout `predict()` path, and splits the resulting
`RolloutResult` back by request sizes. The response includes:

- actions
- behavior `prev_logprobs`
- behavior `prev_values` when available
- training `forward_inputs`
- rollout policy `versions`

The first version handles train mode only and does not mix requests from
different model instances or modes. Channel `maxsize` config provides backpressure
so requests cannot grow without bound.

## Trajectory Boundary Semantics

The design preserves current RLinf `auto_reset` behavior.

For `auto_reset=False`, a done signal does not flush a trajectory immediately.
The worker continues until `max_steps_per_rollout_epoch` or `max_episode_steps`
is reached, then emits one fixed-horizon segment. The segment preserves
`dones`, `terminations`, and `truncations`; the trainer or actor-side processing
computes `loss_mask` the same way the existing PPO/GRPO path does.

For `auto_reset=True`, a done signal may complete an individual episode. The
Env Worker resets that env and can stream completed episodes. To avoid overly
small replay writes, the first version may aggregate completed episodes by
stage or flush by a configured minimum send batch. Short episodes are still
represented through existing `Trajectory` fields, not a new ragged format.

The replay payload is therefore a trajectory segment, not always a complete
episode:

```text
AsyncTrajectoryEnvelope.payload = fixed_horizon_segment | completed_episode
```

## GIPO Training Semantics

The Trainer is replay-buffer based but remains PPO-family actor-critic, not
SAC. Replay entries must preserve:

- `forward_inputs.action`
- behavior `prev_logprobs`
- behavior/proximal `prev_values`
- behavior `versions`
- `rewards`, `dones`, `terminations`, `truncations`
- `loss_mask` and `loss_mask_sum` when already known
- `curr_obs` and `next_obs` when `collect_transitions=true`

Training samples replayed segments, processes them into the existing
`[T, B, ...]` rollout batch format, recomputes current policy logprobs, and
builds:

```text
log_r = current_logprobs - prev_logprobs
rho = exp(log_r)
rho_bar = stop_gradient(clamp(rho, rho_min, rho_max))
omega = exp(-0.5 * (log(rho_bar) / sigma)^2)
policy_loss = -mean_masked(omega * rho * advantages)
```

The Gaussian trust weight is treated as a constant coefficient during
backpropagation. This follows the GIPO paper's log-ratio trust-weighted
surrogate.

Register a new policy loss, for example `loss_type: gipo_actor_critic`, and
reuse the current actor-critic value loss, entropy bonus, loss masks, and
metric utilities where possible.

Advantage handling is configurable:

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

Default `advantage_source` is `recompute`: the trainer recomputes value and GAE
on sampled replay to reduce stale-value bias. A `stored` mode can be kept as a
fallback when the model path cannot recompute values reliably.

Staleness should be measured, not blocked by default. The trainer records
`current_version - versions` and may drop samples only when `max_policy_lag` is
configured.

## Runner Lifecycle

`AsyncGIPOEmbodiedRunner` performs:

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

Worker stop semantics follow current async workers: each service owns a task,
`stop()` cancels the task or sets a stop flag, and the runner waits on handles.
The rollout service flushes pending requests before exit when possible. Env
workers must be able to exit while waiting for an action. Trainer replay-buffer
wait loops must periodically check stop state.

## Error Handling

The first version is fail-fast:

- If an inference response does not arrive before `inference_timeout_s`, the Env
  Worker raises an exception that includes `request_id`, `env_rank`, and
  `stage_id`.
- If rollout prediction fails, the rollout service either writes error
  responses for all pending requests or lets the worker task fail with the
  pending request ids in the message.
- Queue backpressure is handled by channel `maxsize`; no unbounded request or
  trajectory accumulation is allowed.
- Weight sync errors fail the training run rather than silently serving mixed
  model state.

## Metrics

Record at least:

- `rollout/request_qsize`
- `rollout/batch_size_mean`, `rollout/batch_size_p50`, `rollout/batch_size_p95`
- `rollout/max_wait_hit_rate`
- `rollout/target_batch_hit_rate`
- `rollout/inference_latency`
- `env/action_wait_time`
- `env/step_time`
- `train/replay_buffer_size`
- `train/policy_lag_mean`, `train/policy_lag_p95`, `train/policy_lag_max`
- `train/log_ratio_mean`, `train/log_ratio_std`, `train/log_ratio_p95`
- `train/gipo_weight_mean`, `train/gipo_weight_min`

## Tests

Add focused unit tests.

1. Dynamic batching:
   - target batch size triggers flush
   - max wait time triggers flush
   - stop flushes or cancels pending work deterministically

2. Request/response routing:
   - multiple env ranks, stages, and request ids do not cross responses
   - response keys route back to the correct Env Worker waiter

3. Data-flow format:
   - `InferenceRequest.obs` merge preserves batch dimensions
   - `RolloutResult` split preserves actions, logprobs, values, versions, and
     nested `forward_inputs`
   - `AsyncTrajectoryEnvelope -> TrajectoryReplayBuffer -> sample` keeps required
     fields
   - nested `forward_inputs`, `curr_obs`, and `next_obs` stay on CPU with the
     expected `[T, B, ...]` or `[B, ...]` shape convention
   - `versions` align with `prev_logprobs` for policy lag and log-ratio metrics

4. `auto_reset=False`:
   - done does not flush immediately
   - fixed-horizon segment flushes only after configured horizon
   - `dones`, `terminations`, and `truncations` are retained for loss mask
     computation

5. GIPO loss:
   - extreme log-ratios produce finite weights
   - loss mask excludes invalid samples
   - stale samples outside the PPO clip region still have non-zero gradient when
     their Gaussian weight is non-zero

6. Runner smoke:
   - fake env, fake rollout, and fake actor services start
   - one training step completes
   - all services stop and handles are waited

## Implementation Constraints

Use these constraints when turning the design into an implementation plan:

- Place async request and envelope dataclasses in `rlinf/data/embodied_async.py`.
- Add a new runner named `AsyncGIPOEmbodiedRunner`.
- Implement new GIPO service methods or subclasses instead of changing the
  existing async PPO runner behavior in place.
- Default `algorithm.gipo.advantage_source` to `recompute`. Keep `stored` as an
  explicit fallback mode when a model cannot recompute values.
- For `auto_reset=True`, start with stage-level completed-episode aggregation.
  Finer-grained packing can be added after the first data-flow tests pass.
