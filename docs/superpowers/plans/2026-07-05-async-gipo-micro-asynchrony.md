# Async GIPO Micro-Asynchrony 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 `superpowers-zh:subagent-driven-development`（推荐）或 `superpowers-zh:executing-plans` 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在 RLinf embodied 训练中新增一条基于 GIPO 的微观全异步路径，将 simulator、generation、training 解耦，并用 GIPO 处理 replay 中的 stale policy 数据。

**架构：** 新增 `async_gipo` 路径，不改写现有 async PPO/SAC 行为。Env Worker 发送 observation request，Rollout Worker 维护动态 batching 推理服务，Actor/Trainer 从 trajectory replay buffer 采样 `[T, B, ...]` 片段并用 GIPO actor-critic loss 训练。Runner 只负责编排长驻服务、权重同步、metrics 和 checkpoint。

**技术栈：** Python dataclasses、PyTorch、Ray async actors、RLinf Channel、Hydra/OmegaConf、TrajectoryReplayBuffer、FSDP actor workers、pytest。

---

## 文件结构

- 创建：`rlinf/data/embodied_async.py`
  - 定义 `InferenceRequest`、`InferenceResponse`、`AsyncTrajectoryEnvelope` 和 response key/request id helper。
- 修改：`rlinf/data/replay_buffer.py`
  - 增加 whole-trajectory / segment sampling API，返回 PPO/GIPO 需要的 `[T, B, ...]` batch。
- 创建：`rlinf/workers/rollout/hf/async_batching.py`
  - 放置可单测的动态 batching 控制逻辑，不依赖 Ray。
- 修改：`rlinf/workers/rollout/hf/async_huggingface_worker.py`
  - 新增 `serve_inference_gipo()` 长驻服务和 request flush 方法。
- 修改：`rlinf/workers/env/async_env_worker.py`
  - 新增 `interact_gipo_async()` 长驻交互服务，Env 只 step simulator，不 forward 模型。
- 修改：`rlinf/algorithms/losses.py`
  - 注册 `gipo_actor_critic` policy loss，复用 value loss。
- 创建：`rlinf/workers/actor/async_gipo_fsdp_worker.py`
  - 继承 PPO actor 的模型训练能力，加入 trajectory ingest、replay buffer、GIPO replay training。
- 创建：`rlinf/runners/async_gipo_embodied_runner.py`
  - 编排长驻 Env/Rollout/Actor 服务。
- 修改：`examples/embodiment/train_async.py`
  - 当 `algorithm.loss_type == "gipo_actor_critic"` 时选择 GIPO runner/actor。
- 创建：`examples/embodiment/config/libero_spatial_async_gipo_openpi_pi05.yaml`
  - 提供最小可运行配置。
- 创建/修改测试：
  - `tests/unit_tests/test_async_gipo_data_contracts.py`
  - `tests/unit_tests/test_async_gipo_dynamic_batching.py`
  - `tests/unit_tests/test_replay_buffer_trajectory_sampling.py`
  - `tests/unit_tests/test_gipo_loss.py`
  - `tests/unit_tests/test_async_gipo_env_flow.py`
  - `tests/unit_tests/test_async_gipo_runner.py`
  - `tests/unit_tests/test_async_embodied_launch.py`

## 任务 1：数据协议 dataclasses

**文件：**
- 创建：`rlinf/data/embodied_async.py`
- 测试：`tests/unit_tests/test_async_gipo_data_contracts.py`

- [ ] **步骤 1：编写失败的 dataclass 和路由测试**

```python
from rlinf.data.embodied_async import (
    AsyncTrajectoryEnvelope,
    InferenceRequest,
    InferenceResponse,
    build_inference_request_id,
    build_inference_response_key,
)


def test_inference_request_id_and_response_key_are_stable():
    request_id = build_inference_request_id(env_rank=2, stage_id=1, local_step=7)

    assert request_id.startswith("2:1:7:")
    assert build_inference_response_key(2, 1, request_id) == f"action:2:1:{request_id}"


def test_inference_response_carries_rollout_result_and_error_exclusively():
    response = InferenceResponse(
        request_id="2:1:7:test",
        rollout_rank=0,
        actions=None,
        rollout_result=None,
        policy_version=3,
        timing={"queue_wait_s": 0.01},
        error="boom",
    )

    assert response.has_error
    assert response.error == "boom"


def test_trajectory_envelope_segment_type_validation():
    envelope = AsyncTrajectoryEnvelope(
        env_rank=0,
        stage_id=0,
        segment_type="fixed_horizon",
        auto_reset=False,
        trajectory=object(),
        completed_at=1.0,
        last_policy_version=4,
    )

    assert envelope.segment_type == "fixed_horizon"
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/unit_tests/test_async_gipo_data_contracts.py -q`

预期：FAIL，报错包含 `ModuleNotFoundError: No module named 'rlinf.data.embodied_async'`。

- [ ] **步骤 3：实现最小数据协议**

在 `rlinf/data/embodied_async.py` 中加入：

```python
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal


SegmentType = Literal["fixed_horizon", "episode"]


def build_inference_request_id(env_rank: int, stage_id: int, local_step: int) -> str:
    return f"{env_rank}:{stage_id}:{local_step}:{uuid.uuid4().hex}"


def build_inference_response_key(env_rank: int, stage_id: int, request_id: str) -> str:
    return f"action:{env_rank}:{stage_id}:{request_id}"


@dataclass(kw_only=True)
class InferenceRequest:
    request_id: str
    env_rank: int
    stage_id: int
    env_ids: list[int]
    obs: dict[str, Any]
    created_at: float = field(default_factory=time.perf_counter)
    policy_version_hint: int | None = None

    @property
    def response_key(self) -> str:
        return build_inference_response_key(self.env_rank, self.stage_id, self.request_id)


@dataclass(kw_only=True)
class InferenceResponse:
    request_id: str
    rollout_rank: int
    actions: Any
    rollout_result: Any
    policy_version: int
    timing: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    @property
    def has_error(self) -> bool:
        return self.error is not None


@dataclass(kw_only=True)
class AsyncTrajectoryEnvelope:
    env_rank: int
    stage_id: int
    segment_type: SegmentType
    auto_reset: bool
    trajectory: Any
    completed_at: float
    last_policy_version: int | None = None

    def __post_init__(self) -> None:
        if self.segment_type not in ("fixed_horizon", "episode"):
            raise ValueError(f"Unsupported segment_type: {self.segment_type}")
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/unit_tests/test_async_gipo_data_contracts.py -q`

预期：PASS，3 个测试通过。

- [ ] **步骤 5：Commit**

```bash
git add rlinf/data/embodied_async.py tests/unit_tests/test_async_gipo_data_contracts.py
git commit -s -m "feat: add async GIPO data contracts"
```

## 任务 2：动态 batching 纯逻辑

**文件：**
- 创建：`rlinf/workers/rollout/hf/async_batching.py`
- 测试：`tests/unit_tests/test_async_gipo_dynamic_batching.py`

- [ ] **步骤 1：编写失败的 batching 触发测试**

```python
from rlinf.workers.rollout.hf.async_batching import DynamicBatchState


def test_target_batch_size_triggers_flush():
    state = DynamicBatchState(target_batch_size=4, max_wait_time_s=1.0)
    state.mark_first_request(now=10.0)

    assert state.should_flush(queue_size=4, now=10.1)


def test_max_wait_time_triggers_flush_before_target_size():
    state = DynamicBatchState(target_batch_size=8, max_wait_time_s=0.05)
    state.mark_first_request(now=10.0)

    assert state.should_flush(queue_size=2, now=10.06)


def test_empty_queue_never_flushes():
    state = DynamicBatchState(target_batch_size=1, max_wait_time_s=0.0)
    state.mark_first_request(now=10.0)

    assert not state.should_flush(queue_size=0, now=11.0)
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/unit_tests/test_async_gipo_dynamic_batching.py -q`

预期：FAIL，报错包含 `ModuleNotFoundError`。

- [ ] **步骤 3：实现最小触发器**

在 `rlinf/workers/rollout/hf/async_batching.py` 中加入：

```python
from dataclasses import dataclass


@dataclass
class DynamicBatchState:
    target_batch_size: int
    max_wait_time_s: float
    first_request_time: float | None = None

    def mark_first_request(self, now: float) -> None:
        if self.first_request_time is None:
            self.first_request_time = now

    def reset(self) -> None:
        self.first_request_time = None

    def should_flush(self, queue_size: int, now: float) -> bool:
        if queue_size <= 0:
            return False
        if queue_size >= self.target_batch_size:
            return True
        if self.first_request_time is None:
            return False
        return now - self.first_request_time >= self.max_wait_time_s
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/unit_tests/test_async_gipo_dynamic_batching.py -q`

预期：PASS，3 个测试通过。

- [ ] **步骤 5：Commit**

```bash
git add rlinf/workers/rollout/hf/async_batching.py tests/unit_tests/test_async_gipo_dynamic_batching.py
git commit -s -m "feat: add async GIPO dynamic batching policy"
```

## 任务 3：ReplayBuffer whole-trajectory sampling

**文件：**
- 修改：`rlinf/data/replay_buffer.py`
- 测试：`tests/unit_tests/test_replay_buffer_trajectory_sampling.py`

- [ ] **步骤 1：编写失败的 `[T, B, ...]` 采样测试**

```python
import torch

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer


def _make_traj(offset: int) -> Trajectory:
    t, b, c = 3, 2, 1
    rewards = torch.arange(offset, offset + t * b * c, dtype=torch.float32).reshape(t, b, c)
    dones = torch.zeros(t + 1, b, c, dtype=torch.bool)
    dones[-1] = True
    return Trajectory(
        max_episode_length=12,
        model_weights_id=f"w{offset}",
        rewards=rewards,
        dones=dones,
        terminations=dones.clone(),
        truncations=torch.zeros_like(dones),
        prev_logprobs=torch.zeros(t, b, c),
        prev_values=torch.zeros(t + 1, b, c),
        versions=torch.full((t, b, c), float(offset)),
        forward_inputs={"action": torch.zeros(t, b, c)},
    )


def test_sample_trajectory_batch_preserves_time_and_batch_dims():
    buffer = TrajectoryReplayBuffer(seed=0, enable_cache=True, sample_window_size=4)
    buffer.add_trajectories([_make_traj(0), _make_traj(10)])

    batch = buffer.sample_trajectory_batch(num_trajectories=2)

    assert batch["rewards"].shape == (3, 4, 1)
    assert batch["dones"].shape == (4, 4, 1)
    assert batch["forward_inputs"]["action"].shape == (3, 4, 1)
    assert batch["versions"].shape == (3, 4, 1)
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/unit_tests/test_replay_buffer_trajectory_sampling.py -q`

预期：FAIL，报错包含 `AttributeError: 'TrajectoryReplayBuffer' object has no attribute 'sample_trajectory_batch'`。

- [ ] **步骤 3：添加 full trajectory cache 和采样 API**

在 `TrajectoryReplayBuffer.__init__` 中初始化：

```python
self._trajectory_object_cache: dict[int, Trajectory] = {}
```

在 `add_trajectories()` 成功分配 `trajectory_id` 后保存窗口缓存：

```python
self._trajectory_object_cache[trajectory_id] = trajectory
while len(self._trajectory_object_cache) > max(1, int(self.sample_window_size)):
    oldest_id = next(iter(self._trajectory_object_cache))
    self._trajectory_object_cache.pop(oldest_id, None)
```

在类中新增：

```python
def _get_trajectory_object(self, trajectory_id: int) -> Trajectory:
    cached = self._trajectory_object_cache.get(trajectory_id)
    if cached is not None:
        return cached
    info = self._trajectory_index[trajectory_id]
    return self._load_trajectory(trajectory_id, info["model_weights_id"])

def sample_trajectories(self, num_trajectories: int) -> list[Trajectory]:
    if self.size == 0:
        return []
    window_size = max(0, int(self.sample_window_size))
    with self._index_lock:
        candidate_ids = (
            list(self._trajectory_id_list[-window_size:])
            if window_size > 0
            else list(self._trajectory_id_list)
        )
    if not candidate_ids:
        return []
    sample_count = min(int(num_trajectories), len(candidate_ids))
    indices = torch.randint(
        low=0,
        high=len(candidate_ids),
        size=(sample_count,),
        generator=self.random_generator,
    )
    return [self._get_trajectory_object(candidate_ids[int(i)]) for i in indices]

def sample_trajectory_batch(self, num_trajectories: int) -> dict[str, torch.Tensor]:
    from rlinf.data.embodied_io_struct import convert_trajectories_to_batch

    return convert_trajectories_to_batch(self.sample_trajectories(num_trajectories))
```

- [ ] **步骤 4：运行测试验证通过，并回归 chunk sampling**

运行：

```bash
pytest tests/unit_tests/test_replay_buffer_trajectory_sampling.py -q
python -m pytest tests/unit_tests -q -k "replay_buffer or embodied_io_struct"
```

预期：第一个命令 PASS；第二个命令没有和 replay buffer 相关的失败。

- [ ] **步骤 5：Commit**

```bash
git add rlinf/data/replay_buffer.py tests/unit_tests/test_replay_buffer_trajectory_sampling.py
git commit -s -m "feat: sample replay trajectories for GIPO"
```

## 任务 4：GIPO actor-critic loss

**文件：**
- 修改：`rlinf/algorithms/losses.py`
- 测试：`tests/unit_tests/test_gipo_loss.py`

- [ ] **步骤 1：编写失败的 GIPO loss 测试**

```python
import torch

from rlinf.algorithms.losses import compute_gipo_actor_loss
from rlinf.algorithms.registry import policy_loss


def test_gipo_loss_is_finite_for_extreme_log_ratios():
    logprobs = torch.tensor([10.0, -10.0, 0.1], requires_grad=True)
    old_logprobs = torch.zeros(3)
    advantages = torch.ones(3)
    mask = torch.tensor([True, True, True])

    loss, metrics = compute_gipo_actor_loss(
        logprobs=logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages,
        loss_mask=mask,
        gipo_sigma=1.0,
        gipo_rho_min=0.0067,
        gipo_rho_max=148.0,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(logprobs.grad).all()
    assert metrics["actor/gipo_weight_min"] >= 0


def test_policy_loss_registry_has_gipo_actor_critic():
    loss, metrics = policy_loss(
        task_type="embodied",
        loss_type="gipo_actor_critic",
        logprob_type="chunk_level",
        reward_type="chunk_level",
        single_action_dim=1,
        logprobs=torch.zeros(2, 1, requires_grad=True),
        old_logprobs=torch.zeros(2, 1),
        advantages=torch.ones(2, 1),
        returns=torch.ones(2, 1),
        values=torch.zeros(2, 1),
        prev_values=torch.zeros(2, 1),
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        value_clip=0.2,
        huber_delta=10.0,
        gipo_sigma=1.0,
        gipo_rho_min=0.0067,
        gipo_rho_max=148.0,
        max_episode_steps=None,
        loss_mask=None,
        loss_mask_sum=None,
    )

    assert torch.isfinite(loss)
    assert "actor/gipo_weight_mean" in metrics
    assert "critic/value_loss" in metrics
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/unit_tests/test_gipo_loss.py -q`

预期：FAIL，报错包含 `cannot import name 'compute_gipo_actor_loss'`。

- [ ] **步骤 3：实现 GIPO actor loss 和注册项**

在 `rlinf/algorithms/losses.py` 中新增：

```python
def compute_gipo_actor_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
    loss_agg_func: Optional[Callable[..., torch.Tensor]] = masked_mean,
    max_episode_steps: Optional[int] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
    critic_warmup: Optional[bool] = False,
    gipo_sigma: float = 1.0,
    gipo_rho_min: float = 0.0067,
    gipo_rho_max: float = 148.0,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    assert gipo_sigma > 0, "gipo_sigma must be positive"
    if loss_mask is None:
        loss_mask = torch.ones_like(logprobs, dtype=torch.bool)

    loss_mask_ratio = None
    if max_episode_steps is not None and loss_mask_sum is not None:
        loss_mask_ratio = (loss_mask_sum * 1.0) / max_episode_steps
        loss_agg_func = masked_mean_ratio

    log_ratio = logprobs.float() - old_logprobs.float()
    ratio = torch.exp(log_ratio)
    rho_bar = torch.clamp(ratio.detach(), min=gipo_rho_min, max=gipo_rho_max)
    trust_weight = torch.exp(-0.5 * (torch.log(rho_bar) / gipo_sigma).square())
    pg_loss = -trust_weight * ratio * advantages.float()
    policy_loss_abs = loss_agg_func(pg_loss.abs(), loss_mask, loss_mask_ratio)
    policy_loss = loss_agg_func(pg_loss, loss_mask, loss_mask_ratio)
    if critic_warmup:
        policy_loss = torch.tensor(0.0, device=logprobs.device)

    valid = loss_mask
    valid_count = valid.count_nonzero() or 1
    approx_kl = -torch.where(valid, log_ratio.detach(), 0.0).sum() / valid_count
    metrics_data = {
        "actor/policy_loss": policy_loss.detach(),
        "actor/policy_loss_abs": policy_loss_abs.detach(),
        "actor/gipo_ratio": masked_mean(ratio.detach(), valid),
        "actor/gipo_log_ratio_mean": masked_mean(log_ratio.detach(), valid),
        "actor/gipo_weight_mean": masked_mean(trust_weight.detach(), valid),
        "actor/gipo_weight_min": trust_weight.detach()[valid].min() if valid.any() else torch.tensor(0.0, device=logprobs.device),
        "actor/approx_kl": approx_kl.detach(),
    }
    return policy_loss, metrics_data


@register_policy_loss("gipo_actor_critic")
def compute_gipo_actor_critic_loss(**kwargs) -> tuple[torch.Tensor, dict]:
    metrics_data = {}
    actor_loss, actor_metrics_data = compute_gipo_actor_loss(**kwargs)
    critic_loss, critic_metrics_data = compute_ppo_critic_loss(**kwargs)
    loss = actor_loss + critic_loss
    metrics_data.update(actor_metrics_data)
    metrics_data.update(critic_metrics_data)
    return loss, metrics_data
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
pytest tests/unit_tests/test_gipo_loss.py -q
ruff check rlinf/algorithms/losses.py tests/unit_tests/test_gipo_loss.py
```

预期：PASS；Ruff 无错误。

- [ ] **步骤 5：Commit**

```bash
git add rlinf/algorithms/losses.py tests/unit_tests/test_gipo_loss.py
git commit -s -m "feat: add GIPO actor critic loss"
```

## 任务 5：Rollout Worker inference service skeleton

**文件：**
- 修改：`rlinf/workers/rollout/hf/async_huggingface_worker.py`
- 测试：`tests/unit_tests/test_async_gipo_dynamic_batching.py`

- [ ] **步骤 1：编写失败的 flush 数据格式测试**

在 `tests/unit_tests/test_async_gipo_dynamic_batching.py` 追加：

```python
import asyncio
import torch

from rlinf.data.embodied_async import InferenceRequest
from rlinf.data.embodied_io_struct import RolloutResult
from rlinf.workers.rollout.hf.async_huggingface_worker import AsyncMultiStepRolloutWorker


def test_gipo_split_rollout_result_preserves_request_sizes():
    worker = object.__new__(AsyncMultiStepRolloutWorker)
    rollout_result = RolloutResult(
        actions=torch.arange(6, dtype=torch.float32).reshape(3, 2),
        prev_logprobs=torch.zeros(3, 2),
        prev_values=torch.zeros(3, 1),
        forward_inputs={"action": torch.zeros(3, 2)},
        versions=torch.ones(3, 2),
    )

    pieces = worker._split_gipo_rollout_result_by_sizes(rollout_result, [1, 2])

    assert pieces[0].actions.shape == (1, 2)
    assert pieces[1].actions.shape == (2, 2)
    assert pieces[1].forward_inputs["action"].shape == (2, 2)
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/unit_tests/test_async_gipo_dynamic_batching.py::test_gipo_split_rollout_result_preserves_request_sizes -q`

预期：FAIL，报错包含 `AttributeError: 'AsyncMultiStepRolloutWorker' object has no attribute '_split_gipo_rollout_result_by_sizes'`。

- [ ] **步骤 3：实现 split helper 和服务方法签名**

在 `AsyncMultiStepRolloutWorker` 中加入：

```python
def _split_gipo_rollout_result_by_sizes(
    self, rollout_result: RolloutResult, sizes: list[int]
) -> list[RolloutResult]:
    return self._split_rollout_result(rollout_result, sizes)
```

再加入服务入口骨架：

```python
async def serve_inference_gipo(
    self,
    request_channel: Channel,
    response_channel: Channel,
    metric_channel: Channel,
):
    assert self._generate_task is None, "GIPO inference service is already running."
    self._generate_task = asyncio.create_task(
        self._serve_inference_gipo(request_channel, response_channel, metric_channel)
    )
    try:
        await self._generate_task
    except asyncio.CancelledError:
        pass
    finally:
        self._generate_task = None
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/unit_tests/test_async_gipo_dynamic_batching.py::test_gipo_split_rollout_result_preserves_request_sizes -q`

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add rlinf/workers/rollout/hf/async_huggingface_worker.py tests/unit_tests/test_async_gipo_dynamic_batching.py
git commit -s -m "feat: add GIPO rollout response splitting"
```

## 任务 6：Rollout Worker request flush loop

**文件：**
- 修改：`rlinf/workers/rollout/hf/async_huggingface_worker.py`
- 测试：`tests/unit_tests/test_async_gipo_dynamic_batching.py`

- [ ] **步骤 1：编写失败的 fake flush 测试**

追加一个不启动 Ray 的 fake channel 测试：

```python
class _AsyncWork:
    def __init__(self, value):
        self.value = value

    async def async_wait(self):
        return self.value


class _QueueChannel:
    def __init__(self, values=None):
        self.values = list(values or [])
        self.puts = []

    def get(self, async_op=False, key=None):
        assert async_op
        return _AsyncWork(self.values.pop(0))

    def get_nowait(self):
        if not self.values:
            raise asyncio.QueueEmpty
        return self.values.pop(0)

    def put(self, item, key=None, async_op=False):
        self.puts.append((key, item, async_op))


async def _flush_once(worker, requests):
    request_channel = _QueueChannel(requests)
    response_channel = _QueueChannel()
    await worker._flush_gipo_inference_requests(requests, response_channel)
    return response_channel.puts


def test_gipo_flush_sends_one_response_per_request():
    worker = object.__new__(AsyncMultiStepRolloutWorker)
    worker._rank = 0
    worker.version = 5
    worker.predict = lambda obs, profile_context=None: (
        torch.zeros(len(obs["states"]), 2),
        {
            "prev_logprobs": torch.zeros(len(obs["states"]), 2),
            "prev_values": torch.zeros(len(obs["states"]), 1),
            "forward_inputs": {"action": torch.zeros(len(obs["states"]), 2)},
        },
    )

    reqs = [
        InferenceRequest(request_id="r0", env_rank=0, stage_id=0, env_ids=[0], obs={"states": torch.zeros(1, 3)}),
        InferenceRequest(request_id="r1", env_rank=0, stage_id=0, env_ids=[1, 2], obs={"states": torch.zeros(2, 3)}),
    ]
    puts = asyncio.run(_flush_once(worker, reqs))

    assert len(puts) == 2
    assert puts[0][0] == "action:0:0:r0"
    assert puts[1][1].rollout_result.actions.shape == (2, 2)
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/unit_tests/test_async_gipo_dynamic_batching.py::test_gipo_flush_sends_one_response_per_request -q`

预期：FAIL，报错包含 `_flush_gipo_inference_requests` 缺失。

- [ ] **步骤 3：实现 request merge、predict、response put**

在 `AsyncMultiStepRolloutWorker` 中加入：

```python
def _merge_gipo_request_obs(self, requests: list[InferenceRequest]) -> dict[str, Any]:
    obs_batches = [{"obs": req.obs, "final_obs": None} for req in requests]
    return self._merge_obs_batches(obs_batches)["obs"]

async def _flush_gipo_inference_requests(
    self,
    requests: list[InferenceRequest],
    response_channel: Channel,
) -> None:
    if not requests:
        return
    merged_obs = self._merge_gipo_request_obs(requests)
    actions, result = self.predict(
        merged_obs,
        profile_context={"phase": "gipo_action_generation"},
    )
    rollout_result = RolloutResult(
        actions=actions,
        prev_logprobs=result.get("prev_logprobs"),
        prev_values=result.get("prev_values"),
        forward_inputs=result.get("forward_inputs", {}),
        versions=torch.full_like(
            result["prev_logprobs"],
            float(self.version),
            dtype=torch.float32,
        ),
    )
    sizes = [len(req.env_ids) for req in requests]
    pieces = self._split_gipo_rollout_result_by_sizes(rollout_result, sizes)
    for req, piece in zip(requests, pieces, strict=True):
        response_channel.put(
            InferenceResponse(
                request_id=req.request_id,
                rollout_rank=self._rank,
                actions=piece.actions,
                rollout_result=piece,
                policy_version=int(self.version),
            ),
            key=req.response_key,
            async_op=True,
        )
```

Import `InferenceRequest`, `InferenceResponse`, and `Any`.

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
pytest tests/unit_tests/test_async_gipo_dynamic_batching.py -q
ruff check rlinf/workers/rollout/hf/async_huggingface_worker.py tests/unit_tests/test_async_gipo_dynamic_batching.py
```

预期：PASS；Ruff 无错误。

- [ ] **步骤 5：Commit**

```bash
git add rlinf/workers/rollout/hf/async_huggingface_worker.py tests/unit_tests/test_async_gipo_dynamic_batching.py
git commit -s -m "feat: flush async GIPO inference requests"
```

## 任务 7：Env Worker GIPO fixed-horizon flow

**文件：**
- 修改：`rlinf/workers/env/async_env_worker.py`
- 测试：`tests/unit_tests/test_async_gipo_env_flow.py`

- [ ] **步骤 1：编写失败的 `auto_reset=False` 边界测试**

```python
import torch

from rlinf.data.embodied_io_struct import ChunkStepResult, EmbodiedRolloutResult, EnvOutput, RolloutResult
from rlinf.workers.env.async_env_worker import AsyncEnvWorker


def test_gipo_fixed_horizon_does_not_flush_on_done_before_horizon():
    worker = object.__new__(AsyncEnvWorker)
    worker.cfg = type("Cfg", (), {})()
    worker.cfg.env = type("EnvCfg", (), {})()
    worker.cfg.env.train = type("TrainCfg", (), {"auto_reset": False, "max_episode_steps": 10})()
    worker.stage_num = 1
    worker.train_num_envs_per_stage = 2
    worker.n_train_chunk_steps = 3
    worker.rollout_results = [EmbodiedRolloutResult(max_episode_length=10)]
    worker.compute_bootstrap_rewards = (
        lambda env_output, bootstrap_values, reward_model_output: env_output.rewards
    )

    dones = torch.zeros(2, 1, dtype=torch.bool)
    dones[0, 0] = True
    env_output = EnvOutput(
        obs={"states": torch.zeros(2, 3)},
        rewards=torch.ones(2, 1),
        dones=dones,
        terminations=dones.clone(),
        truncations=torch.zeros_like(dones),
    )
    rollout_result = RolloutResult(
        actions=torch.zeros(2, 2),
        prev_logprobs=torch.zeros(2, 2),
        prev_values=torch.zeros(2, 1),
        versions=torch.zeros(2, 2),
        forward_inputs={"action": torch.zeros(2, 2)},
    )

    should_flush = worker._append_gipo_step_and_check_flush(
        stage_id=0,
        env_output=env_output,
        rollout_result=rollout_result,
        chunk_step_idx=0,
    )

    assert not should_flush
    assert len(worker.rollout_results[0].dones) == 1
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/unit_tests/test_async_gipo_env_flow.py -q`

预期：FAIL，报错包含 `_append_gipo_step_and_check_flush` 缺失。

- [ ] **步骤 3：实现 step append helper**

在 `AsyncEnvWorker` 中加入：

```python
def _append_gipo_step_and_check_flush(
    self,
    stage_id: int,
    env_output: EnvOutput,
    rollout_result: RolloutResult,
    chunk_step_idx: int,
) -> bool:
    rewards = self.compute_bootstrap_rewards(
        env_output,
        rollout_result.bootstrap_values,
        reward_model_output=None,
    )
    step_result = ChunkStepResult(
        actions=rollout_result.forward_inputs.get("action", rollout_result.actions),
        prev_logprobs=rollout_result.prev_logprobs,
        prev_values=rollout_result.prev_values,
        forward_inputs=rollout_result.forward_inputs,
        versions=rollout_result.versions,
        dones=env_output.dones,
        truncations=env_output.truncations,
        terminations=env_output.terminations,
        rewards=rewards,
    )
    self.rollout_results[stage_id].append_step_result(step_result)
    if not self.cfg.env.train.auto_reset:
        return chunk_step_idx + 1 >= self.n_train_chunk_steps
    return bool(env_output.dones is not None and env_output.dones[:, -1].any().item())
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/unit_tests/test_async_gipo_env_flow.py -q`

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add rlinf/workers/env/async_env_worker.py tests/unit_tests/test_async_gipo_env_flow.py
git commit -s -m "feat: preserve fixed horizon flow for async GIPO envs"
```

## 任务 8：Env Worker request/response method skeleton

**文件：**
- 修改：`rlinf/workers/env/async_env_worker.py`
- 测试：`tests/unit_tests/test_async_gipo_env_flow.py`

- [ ] **步骤 1：编写失败的 request response routing 测试**

```python
from rlinf.data.embodied_async import build_inference_response_key


class _ResponseChannel:
    def __init__(self, response):
        self.response = response
        self.keys = []

    def get(self, key=None, async_op=False):
        self.keys.append((key, async_op))
        return _AsyncWork(self.response)


class _AsyncWork:
    def __init__(self, value):
        self.value = value

    async def async_wait(self):
        return self.value


def test_env_waits_on_request_specific_response_key():
    worker = object.__new__(AsyncEnvWorker)
    response = object()
    channel = _ResponseChannel(response)

    got = asyncio.run(worker._recv_gipo_inference_response(channel, 3, 1, "req"))

    assert got is response
    assert channel.keys == [(build_inference_response_key(3, 1, "req"), True)]
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/unit_tests/test_async_gipo_env_flow.py::test_env_waits_on_request_specific_response_key -q`

预期：FAIL，报错包含 `_recv_gipo_inference_response` 缺失。

- [ ] **步骤 3：实现 response wait helper 和 service entry**

在 `AsyncEnvWorker` 中加入：

```python
async def _recv_gipo_inference_response(
    self,
    response_channel: Channel,
    env_rank: int,
    stage_id: int,
    request_id: str,
):
    key = build_inference_response_key(env_rank, stage_id, request_id)
    response = await response_channel.get(key=key, async_op=True).async_wait()
    if getattr(response, "has_error", False):
        raise RuntimeError(f"GIPO inference failed for {request_id}: {response.error}")
    return response

async def interact_gipo_async(
    self,
    request_channel: Channel,
    response_channel: Channel,
    trajectory_channel: Channel,
    metric_channel: Channel,
):
    assert self._interact_task is None or self._interact_task.done(), (
        "Previous GIPO interact task is still running."
    )
    self._interact_task = asyncio.create_task(
        self._interact_gipo_async(
            request_channel,
            response_channel,
            trajectory_channel,
            metric_channel,
        )
    )
    try:
        await self._interact_task
    except asyncio.CancelledError:
        pass
```

Import `build_inference_response_key`.

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/unit_tests/test_async_gipo_env_flow.py -q`

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add rlinf/workers/env/async_env_worker.py tests/unit_tests/test_async_gipo_env_flow.py
git commit -s -m "feat: add async GIPO env response routing"
```

## 任务 9：GIPO actor worker replay ingest

**文件：**
- 创建：`rlinf/workers/actor/async_gipo_fsdp_worker.py`
- 测试：`tests/unit_tests/test_async_gipo_runner.py`

- [ ] **步骤 1：编写失败的 trajectory envelope drain 测试**

```python
from queue import Queue

from rlinf.data.embodied_async import AsyncTrajectoryEnvelope
from rlinf.workers.actor.async_gipo_fsdp_worker import AsyncGIPOEmbodiedFSDPActor


class _Buffer:
    def __init__(self):
        self.items = []

    def add_trajectories(self, items):
        self.items.extend(items)


def test_gipo_actor_drains_envelopes_into_replay_buffer():
    actor = object.__new__(AsyncGIPOEmbodiedFSDPActor)
    actor._recv_queue = Queue()
    actor.replay_buffer = _Buffer()
    actor._recv_queue.put(
        AsyncTrajectoryEnvelope(
            env_rank=0,
            stage_id=0,
            segment_type="fixed_horizon",
            auto_reset=False,
            trajectory="traj",
            completed_at=1.0,
            last_policy_version=2,
        )
    )

    actor._drain_received_trajectories()

    assert actor.replay_buffer.items == ["traj"]
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/unit_tests/test_async_gipo_runner.py::test_gipo_actor_drains_envelopes_into_replay_buffer -q`

预期：FAIL，报错包含 `ModuleNotFoundError: No module named 'rlinf.workers.actor.async_gipo_fsdp_worker'`。

- [ ] **步骤 3：实现 GIPO actor skeleton 和 drain**

创建 `rlinf/workers/actor/async_gipo_fsdp_worker.py`：

```python
import asyncio
import os
import queue
import threading

from rlinf.data.embodied_async import AsyncTrajectoryEnvelope
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.workers.actor.async_ppo_fsdp_worker import AsyncPPOEmbodiedFSDPActor


class AsyncGIPOEmbodiedFSDPActor(AsyncPPOEmbodiedFSDPActor):
    should_stop = False

    def setup_gipo_replay_buffer(self) -> None:
        seed = self.cfg.actor.get("seed", 1234)
        auto_save_path = self.cfg.algorithm.replay_buffer.get("auto_save_path", None)
        if auto_save_path is None:
            auto_save_path = os.path.join(
                self.cfg.runner.logger.log_path,
                f"gipo_replay_buffer/rank_{self._rank}",
            )
        else:
            auto_save_path = os.path.join(auto_save_path, f"rank_{self._rank}")
        self.replay_buffer = TrajectoryReplayBuffer(
            seed=seed,
            enable_cache=self.cfg.algorithm.replay_buffer.enable_cache,
            cache_size=self.cfg.algorithm.replay_buffer.cache_size,
            sample_window_size=self.cfg.algorithm.replay_buffer.sample_window_size,
            auto_save=self.cfg.algorithm.replay_buffer.get("auto_save", False),
            auto_save_path=auto_save_path,
            trajectory_format=self.cfg.algorithm.replay_buffer.get("trajectory_format", "pt"),
        )

    def init_worker(self) -> None:
        super().init_worker()
        self.setup_gipo_replay_buffer()
        self._recv_queue = queue.Queue()

    def _drain_received_trajectories(self, max_trajectories: int | None = None) -> int:
        recv_list = []
        while True:
            if max_trajectories is not None and len(recv_list) >= max_trajectories:
                break
            try:
                envelope = self._recv_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(envelope, AsyncTrajectoryEnvelope):
                recv_list.append(envelope.trajectory)
            else:
                recv_list.append(envelope)
        if recv_list:
            self.replay_buffer.add_trajectories(recv_list)
        return len(recv_list)
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/unit_tests/test_async_gipo_runner.py::test_gipo_actor_drains_envelopes_into_replay_buffer -q`

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add rlinf/workers/actor/async_gipo_fsdp_worker.py tests/unit_tests/test_async_gipo_runner.py
git commit -s -m "feat: add async GIPO actor replay ingest"
```

## 任务 10：GIPO actor replay training loop

**文件：**
- 修改：`rlinf/workers/actor/async_gipo_fsdp_worker.py`
- 测试：`tests/unit_tests/test_async_gipo_runner.py`

- [ ] **步骤 1：编写失败的 replay batch loading 测试**

```python
import torch
from omegaconf import OmegaConf


class _ReadyBuffer:
    def __init__(self):
        self.size = 1

    async def is_ready_async(self, min_size):
        return True

    def sample_trajectory_batch(self, num_trajectories):
        return {
            "prev_logprobs": torch.zeros(2, 1, 1),
            "prev_values": torch.zeros(3, 1, 1),
            "rewards": torch.ones(2, 1, 1),
            "dones": torch.zeros(3, 1, 1, dtype=torch.bool),
            "terminations": torch.zeros(3, 1, 1, dtype=torch.bool),
            "truncations": torch.zeros(3, 1, 1, dtype=torch.bool),
            "versions": torch.zeros(2, 1, 1),
            "forward_inputs": {"action": torch.zeros(2, 1, 1)},
        }


def test_gipo_actor_loads_replay_rollout_batch_before_training():
    actor = object.__new__(AsyncGIPOEmbodiedFSDPActor)
    actor.replay_buffer = _ReadyBuffer()
    actor._drain_received_trajectories = lambda max_trajectories=None: 0
    actor.cfg = OmegaConf.create(
        {
            "actor": {"recv_drain_max_trajectories": 256},
            "algorithm": {
                "replay_buffer": {"min_buffer_size": 1},
                "gipo": {"target_batch_segments": 1},
            },
        }
    )
    actor.load_batch = lambda batch: {"reward": 1.0}

    metrics = asyncio.run(actor._prepare_gipo_replay_batch())

    assert metrics == {"reward": 1.0}
    assert actor.rollout_batch["prev_logprobs"].shape == (2, 1, 1)
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/unit_tests/test_async_gipo_runner.py::test_gipo_actor_loads_replay_rollout_batch_before_training -q`

预期：FAIL，报错包含 `_prepare_gipo_replay_batch` 缺失。

- [ ] **步骤 3：实现 replay batch prepare 和 async training wrapper**

在 `AsyncGIPOEmbodiedFSDPActor` 中加入：

```python
async def _wait_for_replay_buffer_ready(self, min_buffer_size: int) -> None:
    while not self.should_stop:
        self._drain_received_trajectories(
            max_trajectories=self.cfg.actor.get("recv_drain_max_trajectories", 256)
            if hasattr(self.cfg, "actor")
            else None
        )
        if await self.replay_buffer.is_ready_async(min_buffer_size):
            return
        await asyncio.sleep(1.0)

async def _prepare_gipo_replay_batch(self) -> dict:
    min_buffer_size = self.cfg.algorithm.replay_buffer.get("min_buffer_size", 1)
    await self._wait_for_replay_buffer_ready(min_buffer_size)
    target_segments = self.cfg.algorithm.get("gipo", {}).get("target_batch_segments", 1)
    batch = self.replay_buffer.sample_trajectory_batch(target_segments)
    self.rollout_batch = batch
    return self.load_batch(batch)

async def run_training(self):
    await self._prepare_gipo_replay_batch()
    return await asyncio.to_thread(AsyncPPOEmbodiedFSDPActor.run_training, self)
```

Ensure `AsyncPPOEmbodiedFSDPActor` is imported in this file.

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
pytest tests/unit_tests/test_async_gipo_runner.py -q
ruff check rlinf/workers/actor/async_gipo_fsdp_worker.py tests/unit_tests/test_async_gipo_runner.py
```

预期：PASS；Ruff 无错误。

- [ ] **步骤 5：Commit**

```bash
git add rlinf/workers/actor/async_gipo_fsdp_worker.py tests/unit_tests/test_async_gipo_runner.py
git commit -s -m "feat: train GIPO actor from replay segments"
```

## 任务 11：GIPO runner lifecycle

**文件：**
- 创建：`rlinf/runners/async_gipo_embodied_runner.py`
- 测试：`tests/unit_tests/test_async_gipo_runner.py`

- [ ] **步骤 1：编写失败的 runner lifecycle smoke test**

```python
from types import SimpleNamespace

from rlinf.runners.async_gipo_embodied_runner import AsyncGIPOEmbodiedRunner


class _DoneHandle:
    def __init__(self, value=None):
        self.value = value if value is not None else [None]

    def wait(self):
        return self.value

    def consume_durations(self, return_per_rank=False):
        return ({}, [{}]) if return_per_rank else {}


class _Service:
    worker_group_name = "Group"

    def __init__(self):
        self.started = 0
        self.stopped = 0
        self.trained = 0

    def set_global_step(self, step):
        return _DoneHandle()

    def sync_model_from_actor(self):
        return _DoneHandle()

    def sync_model_to_rollout(self):
        return _DoneHandle()

    def serve_inference_gipo(self, **kwargs):
        self.started += 1
        return _DoneHandle()

    def interact_gipo_async(self, **kwargs):
        self.started += 1
        return _DoneHandle()

    def recv_trajectories_async(self, **kwargs):
        self.started += 1
        return _DoneHandle()

    def run_training(self):
        self.trained += 1
        return _DoneHandle([{"loss": 0.1}])

    def stop(self):
        self.stopped += 1
        return _DoneHandle()


def test_async_gipo_runner_starts_and_stops_services():
    runner = object.__new__(AsyncGIPOEmbodiedRunner)
    runner.global_step = 0
    runner.max_steps = 1
    runner.cfg = SimpleNamespace(runner=SimpleNamespace(save_interval=-1, val_check_interval=-1))
    runner.actor = _Service()
    runner.rollout = _Service()
    runner.env = _Service()
    runner.metric_logger = SimpleNamespace(log=lambda *a, **k: None, finish=lambda: None)
    runner.timer = SimpleNamespace(consume_durations=lambda: {}, __call__=lambda self, name: self)
    runner.update_rollout_weights = lambda: None
    runner._aggregate_numeric_metrics = lambda metrics: metrics[0] if metrics else {}
    runner.print_metrics_table_async = lambda *a, **k: None
    runner._save_checkpoint = lambda: None
    runner._drain_gipo_metrics = lambda: {}
    runner._create_gipo_channels_for_test()

    runner.run()

    assert runner.actor.trained == 1
    assert runner.env.stopped == 1
    assert runner.rollout.stopped == 1
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/unit_tests/test_async_gipo_runner.py::test_async_gipo_runner_starts_and_stops_services -q`

预期：FAIL，报错包含 `ModuleNotFoundError: No module named 'rlinf.runners.async_gipo_embodied_runner'`。

- [ ] **步骤 3：实现 runner skeleton**

创建 `rlinf/runners/async_gipo_embodied_runner.py`，继承 `EmbodiedRunner`，创建 channels：

```python
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Channel


class AsyncGIPOEmbodiedRunner(EmbodiedRunner):
    def __init__(self, cfg, actor, rollout, env, reward=None, critic=None):
        super().__init__(cfg, actor, rollout, env, critic, reward)
        self._create_gipo_channels()

    def _create_gipo_channels(self) -> None:
        self.gipo_request_channel = Channel.create("GIPOInferenceRequest")
        self.gipo_response_channel = Channel.create("GIPOInferenceResponse")
        self.gipo_trajectory_channel = Channel.create("GIPOTrajectory")
        self.gipo_metric_channel = Channel.create("GIPOMetric")

    def _create_gipo_channels_for_test(self) -> None:
        self.gipo_request_channel = object()
        self.gipo_response_channel = object()
        self.gipo_trajectory_channel = object()
        self.gipo_metric_channel = object()

    def _drain_gipo_metrics(self) -> dict:
        return {}
```

Implement `run()` with service start, one training loop, stop, and handle waits.

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/unit_tests/test_async_gipo_runner.py::test_async_gipo_runner_starts_and_stops_services -q`

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add rlinf/runners/async_gipo_embodied_runner.py tests/unit_tests/test_async_gipo_runner.py
git commit -s -m "feat: add async GIPO embodied runner"
```

## 任务 12：入口脚本和配置选择

**文件：**
- 修改：`examples/embodiment/train_async.py`
- 修改：`tests/unit_tests/test_async_embodied_launch.py`

- [ ] **步骤 1：编写失败的 launch selection 测试**

在 `tests/unit_tests/test_async_embodied_launch.py` 追加：

```python
def test_train_async_selects_gipo_runner_for_gipo_loss():
    source = Path("examples/embodiment/train_async.py").read_text()

    assert "gipo_actor_critic" in source
    assert "AsyncGIPOEmbodiedRunner" in source
    assert "AsyncGIPOEmbodiedFSDPActor" in source
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/unit_tests/test_async_embodied_launch.py::test_train_async_selects_gipo_runner_for_gipo_loss -q`

预期：FAIL，断言 `gipo_actor_critic` 不存在。

- [ ] **步骤 3：接入 train_async 选择分支**

在 `examples/embodiment/train_async.py` 的 loss_type 分支中加入：

```python
elif cfg.algorithm.loss_type == "gipo_actor_critic":
    from rlinf.runners.async_gipo_embodied_runner import AsyncGIPOEmbodiedRunner
    from rlinf.workers.actor.async_gipo_fsdp_worker import AsyncGIPOEmbodiedFSDPActor

    runner_cls = AsyncGIPOEmbodiedRunner
    actor_worker_cls = AsyncGIPOEmbodiedFSDPActor
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
pytest tests/unit_tests/test_async_embodied_launch.py::test_train_async_selects_gipo_runner_for_gipo_loss -q
ruff check examples/embodiment/train_async.py tests/unit_tests/test_async_embodied_launch.py
```

预期：PASS；Ruff 无错误。

- [ ] **步骤 5：Commit**

```bash
git add examples/embodiment/train_async.py tests/unit_tests/test_async_embodied_launch.py
git commit -s -m "feat: route async embodied GIPO training"
```

## 任务 13：配置校验和示例 YAML

**文件：**
- 修改：`rlinf/config.py`
- 创建：`examples/embodiment/config/libero_spatial_async_gipo_openpi_pi05.yaml`
- 测试：`tests/unit_tests/test_async_embodied_launch.py`

- [ ] **步骤 1：编写失败的配置文本测试**

```python
def test_async_gipo_example_config_contains_required_sections():
    path = Path("examples/embodiment/config/libero_spatial_async_gipo_openpi_pi05.yaml")
    text = path.read_text()

    assert "loss_type: gipo_actor_critic" in text
    assert "replay_buffer:" in text
    assert "gipo:" in text
    assert "target_batch_size:" in text
    assert "max_wait_time_s:" in text
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/unit_tests/test_async_embodied_launch.py::test_async_gipo_example_config_contains_required_sections -q`

预期：FAIL，报错包含 file not found。

- [ ] **步骤 3：创建示例配置并放宽 value head 校验**

复制 `examples/embodiment/config/libero_spatial_async_ppo_openpi_pi05.yaml` 为新文件，最小修改：

```yaml
runner:
  logger:
    experiment_name: "libero_spatial_async_gipo_openpi_pi05"

algorithm:
  loss_type: gipo_actor_critic
  adv_type: gae
  replay_buffer:
    min_buffer_size: 1
    enable_cache: true
    cache_size: 8
    sample_window_size: 64
    auto_save: false
    trajectory_format: pt
  gipo:
    advantage_source: recompute
    target_batch_segments: 8
    max_policy_lag: null
    gaussian_sigma: 1.0
    rho_min: 0.0067
    rho_max: 148.0
  async_inference:
    target_batch_size: 32
    max_wait_time_s: 0.01
    request_queue_maxsize: 1024
    inference_timeout_s: 60.0
```

在 `validate_embodied_cfg()` 中让 value head 校验包含 GIPO：

```python
cfg.algorithm.loss_type in (
    "actor_critic",
    "decoupled_actor_critic",
    "gipo_actor_critic",
)
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
pytest tests/unit_tests/test_async_embodied_launch.py::test_async_gipo_example_config_contains_required_sections -q
ruff check rlinf/config.py
```

预期：PASS；Ruff 无错误。

- [ ] **步骤 5：Commit**

```bash
git add rlinf/config.py examples/embodiment/config/libero_spatial_async_gipo_openpi_pi05.yaml tests/unit_tests/test_async_embodied_launch.py
git commit -s -m "feat: add async GIPO embodied config"
```

## 任务 14：最小 GIPO service loop 集成

**文件：**
- 修改：`rlinf/workers/rollout/hf/async_huggingface_worker.py`
- 修改：`rlinf/workers/env/async_env_worker.py`
- 修改：`rlinf/workers/actor/async_gipo_fsdp_worker.py`
- 测试：`tests/unit_tests/test_async_gipo_runner.py`

- [ ] **步骤 1：编写失败的 stop method contract 测试**

```python
def test_gipo_services_expose_stop_safe_methods():
    from rlinf.workers.env.async_env_worker import AsyncEnvWorker
    from rlinf.workers.rollout.hf.async_huggingface_worker import AsyncMultiStepRolloutWorker

    rollout = object.__new__(AsyncMultiStepRolloutWorker)
    rollout._generate_task = None
    assert rollout.stop() is None

    env = object.__new__(AsyncEnvWorker)
    env._interact_task = None
    assert asyncio.run(env.stop()) is None

    actor = object.__new__(AsyncGIPOEmbodiedFSDPActor)
    actor.should_stop = False
    assert asyncio.run(actor.stop()) is None
    assert actor.should_stop
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/unit_tests/test_async_gipo_runner.py::test_gipo_services_expose_stop_safe_methods -q`

预期：FAIL，缺少 actor `stop()` 或 stop flag 行为。

- [ ] **步骤 3：补齐 service loop stop 合约**

在 `AsyncGIPOEmbodiedFSDPActor` 中加入：

```python
async def recv_trajectories_async(self, input_channel):
    if getattr(self, "_recv_queue", None) is None:
        self._recv_queue = queue.Queue()
    if getattr(self, "_recv_thread", None) is None or not self._recv_thread.is_alive():
        self._recv_thread = threading.Thread(
            target=self._recv_thread_main,
            args=(input_channel,),
            daemon=True,
        )
        self._recv_thread.start()

def _recv_thread_main(self, input_channel):
    while not self.should_stop:
        envelope = input_channel.get()
        self._recv_queue.put(envelope)

async def stop(self):
    self.should_stop = True
    recv_thread = getattr(self, "_recv_thread", None)
    if recv_thread is not None and recv_thread.is_alive():
        await asyncio.to_thread(recv_thread.join, 5)
```

In rollout `_serve_inference_gipo()`, loop until cancelled, use `DynamicBatchState`,
and call `_flush_gipo_inference_requests()` when trigger fires.

In env `_interact_gipo_async()`, implement the first version as fixed-horizon
request -> response -> `env_interact_step()` loop for train mode.

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
pytest tests/unit_tests/test_async_gipo_runner.py tests/unit_tests/test_async_gipo_env_flow.py tests/unit_tests/test_async_gipo_dynamic_batching.py -q
ruff check rlinf/workers/rollout/hf/async_huggingface_worker.py rlinf/workers/env/async_env_worker.py rlinf/workers/actor/async_gipo_fsdp_worker.py
```

预期：PASS；Ruff 无错误。

- [ ] **步骤 5：Commit**

```bash
git add rlinf/workers/rollout/hf/async_huggingface_worker.py rlinf/workers/env/async_env_worker.py rlinf/workers/actor/async_gipo_fsdp_worker.py tests/unit_tests/test_async_gipo_runner.py
git commit -s -m "feat: wire async GIPO service loops"
```

## 任务 15：最终验证

**文件：**
- 修改：按验证失败的实际文件最小修复

- [ ] **步骤 1：运行 focused unit tests**

运行：

```bash
pytest \
  tests/unit_tests/test_async_gipo_data_contracts.py \
  tests/unit_tests/test_async_gipo_dynamic_batching.py \
  tests/unit_tests/test_replay_buffer_trajectory_sampling.py \
  tests/unit_tests/test_gipo_loss.py \
  tests/unit_tests/test_async_gipo_env_flow.py \
  tests/unit_tests/test_async_gipo_runner.py \
  tests/unit_tests/test_async_embodied_launch.py \
  -q
```

预期：PASS。

- [ ] **步骤 2：运行 lint**

运行：

```bash
ruff check \
  rlinf/data/embodied_async.py \
  rlinf/data/replay_buffer.py \
  rlinf/algorithms/losses.py \
  rlinf/workers/rollout/hf/async_batching.py \
  rlinf/workers/rollout/hf/async_huggingface_worker.py \
  rlinf/workers/env/async_env_worker.py \
  rlinf/workers/actor/async_gipo_fsdp_worker.py \
  rlinf/runners/async_gipo_embodied_runner.py \
  examples/embodiment/train_async.py \
  tests/unit_tests/test_async_gipo_data_contracts.py \
  tests/unit_tests/test_async_gipo_dynamic_batching.py \
  tests/unit_tests/test_replay_buffer_trajectory_sampling.py \
  tests/unit_tests/test_gipo_loss.py \
  tests/unit_tests/test_async_gipo_env_flow.py \
  tests/unit_tests/test_async_gipo_runner.py
```

预期：PASS。

- [ ] **步骤 3：运行配置导入 smoke**

运行：

```bash
python -m py_compile \
  rlinf/data/embodied_async.py \
  rlinf/workers/rollout/hf/async_batching.py \
  rlinf/workers/actor/async_gipo_fsdp_worker.py \
  rlinf/runners/async_gipo_embodied_runner.py
```

预期：命令退出码 0。

- [ ] **步骤 4：检查未提交改动**

运行：`git status --short`

预期：只显示本任务有意修改的文件；没有临时文件、缓存文件或无关文件被加入。

- [ ] **步骤 5：Commit 验证修复**

如果步骤 1-3 需要修复代码：

```bash
git add <fixed-files>
git commit -s -m "test: verify async GIPO micro-asynchrony"
```

如果没有修复，不创建空 commit。
