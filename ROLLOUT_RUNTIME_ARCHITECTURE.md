# Rollout Runtime 架构大纲

> 状态：讨论稿  
> 目标：在现有 `rlinf_rollout` 基础上设计通用 Rollout Runtime，作为 Agent、RL
> 训练、批量评测及其他具身应用的统一后端。本文固定总体边界和模块关系，具体模块改动由后续
> 设计分别展开。

## 1. 目标与原则

Rollout Runtime 负责高效执行虚拟环境和 VLA 等模型推理，对上层应用隐藏 Ray、worker
拓扑、Channel、batch 和模型部署细节。

设计原则：

1. 上层应用继续使用自身原生接口，通过 Adapter 与 Runtime 交互。
2. Runtime 数据面只包含两类计算 worker：`EnvWorker` 和 `RolloutWorker`。
3. Runtime 只理解 session、observation、action、request 和生命周期，不理解具体 Agent
   tool 或 RL 算法。
4. 工具语义、轨迹组装和训练语义位于 Adapter 或可选插件中。
5. 优先复用现有 `rlinf_rollout` 的 scheduler、环境、模型、路由和 decoupled batching
   能力，不从零重写。
6. 外部协议和内部传输解耦；上层不直接连接 Ray Channel。

## 2. 总体架构

```text
Applications
├── RPent / other Agents
├── RL collectors and evaluators
├── Gym / VectorEnv applications
└── Batch evaluation systems
          │ native application interfaces
          ▼
Application Adapter Layer
├── Agent Toolkit Adapter       execute_tool(...)
├── RL Rollout Adapter          collect(...)
├── Gym Adapter                 reset() / step()
└── Evaluation Adapter          run_episodes(...)
          │ unified Runtime API
          ▼
Rollout Runtime Gateway
├── Session Manager
├── Request Router
├── Admission / Backpressure
├── Adapter Plugin Executor
└── Status / Metrics
          │ internal typed requests over Ray Channel
          ▼
┌────────────────────────┐       ┌──────────────────────────┐
│ EnvWorkerGroup         │       │ RolloutWorkerGroup       │
│                        │ obs   │                          │
│ Environment creation   ├──────►│ Inference scheduler      │
│ Session / env slots    │       │ Dynamic micro-batching   │
│ reset / step / chunk   │◄──────┤ VLA / policy inference   │
│ Episode state          │ action│ Replicas / TP / PP       │
└────────────────────────┘       └──────────────────────────┘
```

Gateway 和 Adapter 是轻量 CPU 控制面，不执行仿真或模型计算。Runtime 内部可以继续使用
Ray，但该实现细节不进入上层 SDK。

## 3. 分层职责

### 3.1 Application Adapter Layer

Adapter 保持上层应用接口不变，并将其编译为统一 Runtime 操作。

| Adapter | 上层接口 | Runtime 操作 |
|---|---|---|
| Agent Toolkit | `execute_tool(name, args)` | `observe`、`policy_step`、`action_step` 的组合 |
| Gym | `reset()` / `step(action)` | `create_session`、`reset`、`action_step` |
| RL rollout | `collect(horizon)` | 批量 session 和 `policy_step`，组装应用所需轨迹 |
| Evaluation | `run_episodes(tasks)` | 批量创建 session，运行至终止 |

工具的循环和停止条件不进入 Runtime Core。例如 `pi0_pick` Adapter 可以多次调用
`policy_step`，根据 observation 判断是否结束，最终只向 Agent 返回一次 tool result。

Adapter 可有两种部署形式：

- 客户端 Adapter：实现简单，适合低频、单步操作。
- Gateway Plugin：减少高层 primitive 的跨进程往返，适合 Agent tool 和闭环技能。

### 3.2 Rollout Runtime Gateway

Gateway 是面向 session 的协议无关控制入口。它决定哪个应用可以在什么时候操作哪个
session，并把命令可靠地送到持有该 session 的 EnvWorker。Gateway 不进入 EnvWorker 与
RolloutWorker 之间的推理数据路径：`policy_step` 对 Gateway 是一条原子环境命令，只有
EnvWorker 会将它展开为 observation、模型推理和 `chunk_step`。

Gateway 的职责包括：

- session 创建、查找、租约和回收；
- 外部请求到内部 worker request 的转换；
- session 到 EnvWorker binding 的粘性路由；
- 请求鉴权、配额、优先级、deadline 和背压；
- request 状态查询、取消和错误归一化；
- Adapter Plugin 执行；
- health、status、metrics 和 tracing。

Gateway 不执行环境 step 或模型 forward，不保存大规模 trajectory，也不把 observation 和
episode 状态作为自身的权威状态。Gateway 可以缓存最近一次 worker 返回的状态摘要或
payload reference，但 EnvWorker 始终是环境执行状态的唯一事实源。

```text
Application / Adapter
        │ Runtime API
        ▼
Runtime Gateway
  session / admission / operation / routing
        │ EnvCommand
        ▼
EnvWorker ───────── InferenceRequest ────────► RolloutWorker
          ◄──────── ActionResponse ──────────
```

#### 3.2.1 上层应用与 Session 创建

Session 由上层应用或 Adapter 按需创建，是上层业务会话对 Runtime 环境资源的一次有状态
占用。Gateway 只理解稳定的关联字段，不理解 Agent tool、评测任务或 RL collector 的具体
业务语义。建议的创建请求为：

```text
CreateSessionRequest
  application_id          # 鉴权、配额、指标和审计的归属
  client_session_key      # 上层应用提供的幂等键
  env_spec                # 环境类型、配置和资源需求
  default_policy_id       # 可选；policy_step 未指定 policy 时使用
  lease_seconds
  metadata                # tool_call_id、trace_id 等透传字段
```

Session 创建由 Gateway 发起和协调，物理资源分配由 EnvWorker 完成：

```text
Application / Adapter
  │ create_session(request)
  ▼
Gateway
  │ 1. 鉴权、配额和幂等检查
  │ 2. 生成 session_id，记录 CREATING 状态
  │ 3. 按 EnvSpec、capacity 和 locality 选择 EnvWorker
  ▼
EnvWorker
  │ 4. 选择或创建 env pool，分配 env slot
  │ 5. 建立 session/slot 映射，返回 opaque binding_token
  ▼
Gateway
  │ 6. 提交 session binding，将状态更新为 READY
  ▼
Application receives SessionHandle
```

只有 EnvWorker 成功返回 binding 后，Gateway 才能发布可用的 `SessionHandle`。如果分配
失败，Gateway 将 session 标记为 `FAILED`，短期保留错误供调用方查询，然后清理记录。创建
成功后，上层 Adapter 持有并复用 handle，直到主动关闭或 lease 过期。

```text
SessionHandle
  session_id
  application_id
  env_spec_digest
  default_policy_id
  lease_expiration
  gateway_epoch
```

`SessionHandle` 不暴露 EnvWorker rank、env slot 或 `binding_token`。这些字段属于 Runtime
内部路由，避免 worker 重启、placement 调整或 transport 变化破坏上层协议。

`create_session` 与 `reset` 应保持分离：前者分配并绑定环境资源，后者初始化具体 episode
的 task、seed、instruction 和初始状态。因此同一 session 可以通过多次 reset 连续执行多个
episode。Agent 和 Gym Adapter 通常显式持有长期 session；Evaluation 和 RL Adapter 可以批量
创建临时 session，并在 episode 结束后自动关闭。

#### 3.2.2 状态所有权与内部组件

Gateway 和 EnvWorker 不能共同修改同一份 session 状态。各模块的权威状态划分如下：

| 状态 | 唯一事实源 |
|---|---|
| application、tenant、lease、逻辑生命周期、EnvWorker 路由 | Gateway |
| env pool、env slot、observation、reward 和 episode 状态 | EnvWorker |
| 外部 operation 的幂等、查询、取消和短期结果 | Gateway |
| operation 是否已开始 inference 或 env step | EnvWorker，通过状态事件通知 Gateway |
| inference queue、batch、模型版本 | RolloutWorker |
| trajectory 和应用专属结果 | Adapter 或 Sink |

Gateway 维护的逻辑生命周期建议采用以下状态机：

```text
CREATING ── allocation committed ──► READY ── close / lease expired ──► CLOSING
    │                                  │                                  │
    └── allocation failed ──► FAILED   └── worker binding lost ──► LOST   │
                                                                         ▼
                                                                       CLOSED
```

`READY` 表示 session 已绑定环境资源，可以接收操作；它不表示 episode 已 reset。`FAILED` 和
`LOST` 短期保留用于错误查询，但不能继续 step。第一版不透明恢复 `LOST` session，因为新建
环境无法保证延续原物理状态。

建议将 Gateway 拆成以下逻辑组件。第一版可以把这些组件部署在一个单 rank CPU worker 中，
以单 writer 简化 session 和 operation 的一致性：

```text
RuntimeGateway
├── GatewayCore             # 协议无关 Runtime API
├── SessionManager          # session binding、生命周期和 lease
├── EnvWorkerRegistry       # worker capability、capacity 和 heartbeat
├── AdmissionController     # 鉴权、配额、deadline 和背压
├── OperationRegistry       # 幂等、状态、取消和短期结果
├── CommandDispatcher       # 粘性路由、批量 scatter/gather
└── PluginExecutor          # 可选的高层 Adapter Plugin
```

`PluginExecutor` 只能调用与普通 Adapter 相同的 Runtime API，不能绕过 SessionManager 直接
操作 EnvWorker。生产环境中应把插件放入独立进程或受限执行池，避免插件阻塞 Gateway 的请求
循环；第一版可以只定义接口而不开放第三方插件加载。

#### 3.2.3 请求、顺序与取消语义

Gateway 的最小调度单位是一条只操作一个 session 的 `EnvCommand`。批量 API 在 Gateway
内部拆成多条命令并发 scatter，再按输入顺序 gather。不同 session 可以部分成功，不提供跨
session 事务。

同一 session 的状态变更命令必须串行。每个 binding 使用以下字段阻止迟到请求污染新状态：

- `binding_token`：EnvWorker 生成的不透明绑定标识；env slot 被释放并复用后，旧 token
  立即失效。
- `episode_id`：每次成功 reset 后递增，拒绝前一 episode 产生的迟到响应。
- `operation_seq`：每次状态变更操作递增，确定同一 session 的执行顺序。

`request_id` 用于 operation 幂等。同一 ID 和相同请求返回已有状态或结果；同一 ID 对应不同
请求时拒绝执行。Gateway 只能保证进程生命周期内的 effectively-once，不能承诺故障场景下的
exactly-once。

RPC 等待超时不等于取消环境操作。取消按执行阶段处理：

- 尚未派发：Gateway 可以直接取消。
- 已派发但仍在等待 inference：EnvWorker 可以停止等待，并丢弃迟到 action。
- env step 已开始：动作不能回滚，必须等待完成，并在结果中返回
  `side_effect_applied=true`。
- worker 失联且无法判断 step 是否发生：返回 `OUTCOME_UNKNOWN`，不得自动重放。

#### 3.2.4 Gateway 伪代码

```python
class RuntimeGateway:

    async def start(self):
        # 启动 worker heartbeat、命令结果接收、lease 回收和状态发布循环。
        pass

    async def create_sessions(self, requests):
        # 对每个请求执行鉴权、配额和 client_session_key 幂等检查，选择 EnvWorker，
        # 分配 env binding；每个 session 完成全部创建步骤后才发布其对应 handle。
        pass

    async def allocate_session(self, request):
        # 创建 CREATING 记录，向匹配 EnvSpec 的 EnvWorker 请求 env slot，并原子提交
        # worker route 和 binding_token；失败时释放可能已分配的资源。
        pass

    async def get_session(self, session_id):
        # 返回逻辑生命周期、lease 和 worker 摘要，不把缓存摘要声明为环境权威状态。
        pass

    async def renew_sessions(self, session_ids, lease_seconds):
        # 校验调用方所有权，延长 Gateway lease，并批量通知对应 EnvWorker watchdog。
        pass

    async def reset(self, session_ids, reset_spec):
        # 为每个 session 生成下一 operation_seq，并派发 RESET；成功响应携带新的
        # episode_id 和初始 observation。
        pass

    async def observe(self, session_ids):
        # 在线性一致模式下排在此前已接受的变更命令之后，再从 EnvWorker 读取状态。
        pass

    async def action_step(self, session_ids, actions):
        # 将每个 action 封装为单 session EnvCommand，并发派发后按输入顺序聚合结果。
        pass

    async def policy_step(self, session_ids, policy_request):
        # 向 EnvWorker 派发原子 POLICY_STEP；Gateway 不直接调用 RolloutWorker。
        pass

    async def dispatch_command(self, session_id, operation, payload, context):
        # 校验 binding、lifecycle、deadline 和配额，通过 request_id 去重，串行化同一
        # session 的变更操作，记录 operation 后发送到粘性绑定的 EnvWorker。
        pass

    async def handle_worker_event(self, event):
        # 更新 operation 状态和非权威 session 摘要，完成等待中的请求并记录指标。
        pass

    async def get_request_status(self, request_id):
        # 返回 ACCEPTED/QUEUED/RUNNING/SUCCEEDED/FAILED/CANCELLED/OUTCOME_UNKNOWN。
        pass

    async def cancel_request(self, request_id):
        # 未派发时直接取消；已派发时转发 best-effort cancel，并报告是否可能已产生副作用。
        pass

    async def close_sessions(self, session_ids):
        # 将 session 标记为 CLOSING，停止接收新操作，通知 EnvWorker 安全释放 env slot，
        # 收到确认后标记 CLOSED。
        pass

    async def recover_expired_sessions(self):
        # 回收 lease 已过期的 session；有执行中操作时先标记 reap_pending，待操作结束后关闭。
        pass

    async def register_env_worker(self, worker_info):
        # 更新 EnvWorker 的 capability、capacity、heartbeat 和可服务 EnvSpec。
        pass
```

Gateway 到 EnvWorker 的 command/result channel 必须有界，以便从数据面向入口逐级传播背压。
cancel、heartbeat 和 shutdown 等控制消息建议使用独立的高优先级控制通道，避免被普通环境命令
阻塞。

### 3.3 EnvWorker

EnvWorker 负责：

- 按 `EnvSpec` 创建不同类型的虚拟环境；
- 管理 vector env、session/env slot 映射和 episode 状态；
- 执行 `reset`、`observe`、`step`、`chunk_step`；
- 将 observation 转换为统一内部 schema；
- 向 RolloutWorker 发出推理请求并接收 action；
- 将 action 应用于正确的 env slot；
- 维护终止、截断、reward、info 和必要的 observation 缓存。

同一 session 的变更操作必须串行，不同 session 可以并发。

#### 3.3.1 调用模式与 Session

Runtime 对外支持两类环境服务，但两者应共用同一套 EnvWorker 和环境执行原语：

| 调用模式 | 典型应用 | EnvWorker 操作 | Session 使用方式 |
|---|---|---|---|
| 单步交互 | Agent tool、人工控制 | `policy_step`：执行一次 VLA 推理和 `chunk_step` | 调用方显式持有 session，多次请求持续操作同一环境 |
| 完整 episode | Evaluation、RL rollout | `run_episode`：循环执行 `policy_step` 直至结束 | 可由 Runtime 内部创建临时 session，并在 episode 结束后关闭 |

Session 是一段有状态环境交互的逻辑句柄，不是环境实例本身。它将多次请求粘性绑定到同一个
EnvWorker 和 env slot。一个逻辑 session 的状态分布在 Gateway 与 EnvWorker 中，但字段所有权
互不重叠：

```text
session_id
  -> Gateway: application / lifecycle / lease / active operation / worker route
  -> EnvWorker: env pool / env slot / current observation / episode state
```

Session 的主要职责包括：

- 保证后续请求继续操作同一个 env slot；
- 由 EnvWorker 缓存当前 observation 和 episode 状态；
- 串行化同一 session 的 `reset`、`action_step` 和 `policy_step`；
- 由 Gateway 协调关闭和租约，由 EnvWorker 管理 reset 与终止状态；
- 通过 `binding_token`、`episode_id` 和 `operation_seq` 拒绝迟到命令或响应。

需要区分三个概念：env slot 是实际运行环境的物理资源；session 是应用对 env slot 的逻辑占用；
episode 是一次从 reset 到 terminated/truncated 的执行过程。一个 session 可以通过多次 reset
执行多个 episode。

#### 3.3.2 EnvWorker 伪代码

```python
class RuntimeEnvWorker:

    async def run(self):
        # 启动请求接收、任务调度和资源回收循环。
        pass

    async def create_session(self, session_id, env_spec, lease_expiration):
        # 为 Gateway 已生成的 session_id 分配 env slot，创建 binding_token，并记录
        # EnvWorker 本地的 session/slot 映射；不创建外部 SessionHandle。
        pass

    async def reset(self, session_id, reset_spec):
        # 重置指定 session，并缓存初始 observation 和 episode 状态。
        pass

    async def observe(self, session_id):
        # 返回当前缓存的 observation，不改变环境状态。
        pass

    async def action_step(self, session_id, actions):
        # 将外部 action 转换为环境 action，并执行一次 chunk_step。
        pass

    async def policy_step(self, session_id, policy_request):
        # 原子执行 current observation -> VLA inference -> chunk_step。
        pass

    async def run_episode(self, session_id, episode_request):
        # 循环执行 policy_step，直至终止、截断、取消、超时或达到步数上限。
        pass

    async def request_inference(self, session_id, observation, policy_request):
        # 向 RolloutWorker 提交推理请求，并等待与 request/session 对应的 action。
        pass

    async def step_env(self, session_id, raw_actions):
        # 执行 action 转换和 chunk_step，并更新 session 的 observation 与终止状态。
        pass

    async def publish_transition(self, episode_request, step_result):
        # 将 episode transition 写入 evaluation 或 RL Adapter/sink。
        pass

    def should_finish_episode(self, session_id, episode_request):
        # 检查 terminated、truncated、cancel、deadline 和最大步数条件。
        pass

    async def cancel_request(self, request_id):
        # 取消尚未开始的操作；已经开始的 env step 只能等待其完成。
        pass

    async def close_session(self, session_id):
        # 释放 env slot，并删除对应的 session 状态。
        pass

    async def recover_expired_sessions(self):
        # 回收 lease 已过期且没有执行中操作的 session。
        pass
```

### 3.4 RolloutWorker

RolloutWorker 以 serving system 的形式运行：

- 常驻加载一个或多个 policy/VLA；
- 消费来自多个 EnvWorker 的 `InferenceRequest`；
- 按兼容性键分桶并动态组 batch；
- 执行预处理、`predict_action_batch` 和后处理；
- 按 request 路由信息拆分结果并返回原 EnvWorker；
- 支持模型 replica、tensor/pipeline parallel 和模型版本；
- 暴露队列长度、batch 利用率、延迟和错误指标。

建议的 batch 兼容性键：

```text
policy_id
+ model_version
+ observation_schema
+ inference_parameters
+ device/dtype constraints
```

触发推理的条件为 `max_batch_size` 或 `max_wait_ms` 任一达到。调度还需考虑 deadline、
priority、应用配额和跨 session 公平性。

#### 3.4.1 Placement 与多卡实例

EnvWorker 和 RolloutWorker 都复用 RLinf 的 `HybridComponentPlacement` 和 `WorkerGroup`。
Placement 分别生成 `env`、`rollout` 两套放置策略，WorkerGroup 根据策略启动 Ray worker，
并设置 `RANK`、`WORLD_SIZE`、`LOCAL_RANK` 和可见加速器等运行环境。

```python
def launch_runtime_workers(cfg, cluster):
    # 解析 component_placement，分别计算 env 和 rollout 的节点及硬件分配。
    placement = HybridComponentPlacement(cfg, cluster)

    # 按 env placement 启动 EnvWorkerGroup；每个 rank 管理自己的 env slot pool。
    env_group = RuntimeEnvWorker.create_group(cfg).launch(
        cluster,
        name=cfg.env.group_name,
        placement_strategy=placement.get_strategy("env"),
    )

    # 按 rollout placement 启动 RolloutWorkerGroup；每个 rank 是一个推理实例。
    rollout_group = RuntimeRolloutWorker.create_group(cfg).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=placement.get_strategy("rollout"),
    )

    return env_group, rollout_group
```

第一版 VLA Runtime 采用现有 HF embodied rollout 的数据并行方式：

- `rollout.tensor_parallel_size = 1`，每个 RolloutWorker rank 占用一张 GPU；
- 每个 rank 加载一份完整 VLA，并拥有独立请求队列和本地 Scheduler；
- EnvWorker 与 RolloutWorker 的数量相互独立，由现有 decoupled routing 将请求分散到各
  rollout rank，并按记录的 route 返回结果；
- Env placement 只决定 EnvWorker 的进程和硬件位置；session 创建后再粘性绑定到具体
  `env rank/env slot`。

如果以后单个 VLA 需要跨卡，可仿照 RLinf 的 SGLang/vLLM placement：

```python
def build_model_parallel_rollout_placement(cfg, rollout_gpus):
    # 一个逻辑推理实例占用 TP * PP 张 GPU；剩余维度为数据并行实例数。
    gpus_per_instance = cfg.rollout.tensor_parallel_size * cfg.rollout.pipeline_parallel_size
    rollout_dp_size = len(rollout_gpus) // gpus_per_instance

    # Placement 只保证每个 worker 看见对应 GPU，不负责切分模型。
    strategy = PackedPlacementStrategy(
        rollout_gpus.first,
        rollout_gpus.last,
        num_hardware_per_process=gpus_per_instance,
    )
    return strategy, rollout_dp_size
```

此时 `PolicyInferenceCore` 或其底层 serving backend 必须实现 TP/PP。Placement 本身不能把
普通单卡 VLA 自动变成模型并行实现。

#### 3.4.2 RolloutWorker 伪代码

```python
class RuntimeRolloutWorker(Worker):
    def initialize(self):
        # 在当前 rank 分配到的 GPU 上加载 VLA，并创建本地 InferenceBatchScheduler。
        pass

    async def serve(self, input_channel, output_channel):
        # 常驻运行请求接收循环和 batch 执行循环，直到收到停止信号。
        pass

    async def receive_requests(self, input_channel):
        # 接收多个 EnvWorker 的请求，记录回包 route，并提交到本地 Scheduler。
        pass

    async def execute_batches(self, output_channel):
        # 按 max_batch_size、max_wait_ms、兼容性和优先级持续取得可执行 batch。
        pass

    def infer_batch(self, requests):
        # 预处理 observation，执行 predict_action_batch，并生成逐请求 ActionResponse。
        pass

    async def send_responses(self, responses, output_channel):
        # 按 request_id、session_id 和已记录 route 将结果返回原 EnvWorker。
        pass

    async def update_weights(self, model_version):
        # 在版本边界同步或切换模型权重，不让同一 batch 混用不同版本。
        pass

    async def stop(self):
        # 停止接收新请求，排空或拒绝队列，并释放模型与通信资源。
        pass
```

Scheduler 是每个 RolloutWorker rank 内部的 serving 调度器，只负责排队、兼容性分桶、
micro-batching、公平性和背压；Ray Channel 负责跨 worker 传输，两者职责分离。完整 episode
不会进入 RolloutWorker，它只会看到 EnvWorker 连续发来的普通 `InferenceRequest`。

## 4. Runtime API 草案

建议新增 `rlinf_rollout/runtime_api/v1/`，不要修改已经冻结的 `api/v1`。

基础接口：

```text
create_sessions(create_requests) -> SessionHandle[]
get_session(session_id) -> SessionStatus
renew_sessions(session_ids, lease_seconds)
reset(session_ids, reset_spec) -> ObservationBatch
observe(session_ids) -> ObservationBatch
action_step(session_ids, actions) -> StepBatch
policy_step(session_ids, policy_request) -> StepBatch
close_sessions(session_ids)
get_request_status(request_id)
cancel_request(request_id)
```

基础消息：

```text
EnvCommand
  request_id
  session_id
  binding_token
  episode_id
  operation_seq
  operation
  deadline
  priority
  payload
  trace_context

InferenceRequest
  request_id
  session_id
  binding_token
  episode_id
  operation_seq
  policy_id
  observation
  instruction_override
  inference_parameters
  routing_token

ActionResponse
  request_id
  session_id
  binding_token
  episode_id
  operation_seq
  actions
  model_version
  auxiliary_outputs
  error

StepResult
  request_id
  session_id
  binding_token
  episode_id
  operation_seq
  observation
  reward
  terminated
  truncated
  info
  side_effect_applied
```

外部 transport 可以是 HTTP、gRPC 或本地 Python SDK；内部仍可使用 Ray Channel 和优化后的
tensor communication。业务 ID 与内部 Channel key/worker rank 必须分离。

## 5. Session 与请求语义

Gateway 的 SessionManager 至少维护：

```text
session_id
  -> application_id / client_session_key
  -> EnvWorker rank / opaque binding_token
  -> logical lifecycle state
  -> env spec / default policy
  -> next operation sequence / active operation
  -> lease expiration
  -> optional last worker summary reference
```

EnvWorker 维护 `binding_token -> env pool/env slot`、当前 observation、episode、reward 和
terminated/truncated 状态。Gateway 中的 worker summary 仅用于查询和观测，不能作为执行
下一条环境命令的依据。

约束：

- 同一 session 同时最多有一个状态变更请求。
- session 必须通过 binding 粘性路由到同一个 EnvWorker/env slot。
- `terminated`/`truncated` 后拒绝 step，除非显式 reset。
- lease 超时后回收环境资源。
- `request_id` 用于去重和结果查询。
- `binding_token`、`episode_id` 和 `operation_seq` 用于拒绝迟到命令和响应。
- 已开始的物理动作或仿真 step 不能因 RPC 超时自动重放。
- cancel 是 best effort，响应必须说明尚未开始、执行中或已经完成。

## 6. 调度与背压

### 6.1 Env 调度

- EnvWorker 维护固定或可扩缩的 env slot pool。
- 创建 session 时按 env type、配置兼容性和负载选择 pool。
- 可批量 reset/step 的 session 尽量落到兼容 vector env。
- 不兼容配置使用独立 pool，避免在每次 step 动态重建环境。

### 6.2 推理调度

- 从多个 EnvWorker 收集请求，短时间窗口内组成 micro-batch。
- 根据兼容性键分桶，不允许不兼容 observation/model 混批。
- 交互 Agent 请求可使用较短 deadline；离线 RL/eval 请求可偏向吞吐。
- 每个 application/session 设置并发上限，避免单一消费者占满队列。
- 队列满时显式拒绝或延迟，不允许无界堆积。
- 响应按 `request_id/session_id/routing_token` 精确返回。

当前训练期 dynamic scheduler 主要负责资源伸缩，不应直接充当 Runtime micro-batcher；可复用
其监控和实例管理思想，但需要新的在线推理调度器。

## 7. RLinf Rollout 可复用模块

### 7.1 可直接复用

- `scheduler/cluster`：Ray 集群接入和节点管理。
- `scheduler/worker`：`Worker`、`WorkerGroup`、worker 生命周期。
- `scheduler/channel`：异步队列、命名连接和分布式 Channel。
- `scheduler/placement`、`utils/placement.py`：组件放置与异构节点。
- `scheduler/worker/routing.py`：batch split/merge、普通与 decoupled 路由。
- `envs/`：环境注册、vector env、wrapper 和 action utility。
- `models/`：模型注册、`BasePolicy`、`get_model` 和各 VLA 实现。
- HF rollout worker 中的模型加载、预处理、推理、CUDA graph、compile/offload。
- `data/embodied_io_struct.py` 中的 `EnvOutput` 和内部 action result 表达。
- `serve/` 中的服务生命周期、配置装配、status 和 control-plane 基础。

### 7.2 抽取后复用

- 从 `AsyncEnvWorker` 抽出 `EnvExecutionCore`：
  环境构建、reset、step、chunk step、observation 转换。
- 从 HF rollout worker 抽出 `PolicyInferenceCore`：
  模型加载、输入预处理、`predict_action_batch`、输出构建。
- 将现有 `decoupled_generate` 的收集、合批和返回路由抽成
  `InferenceBatchScheduler`。
- 将 `RolloutService` 扩展为 `runtime` 服务模式，装配 Runtime Gateway 和两类 worker。

旧的批量 collect/eval worker 应继续调用上述 core，避免产生两套环境和模型实现。

### 7.3 不进入 Runtime Core

- advantage、return 和 RL loss；
- actor/critic world size；
- trainer trajectory partition；
- 训练 epoch/minibatch 生命周期；
- RPent 或其他 Agent 的具体 tool schema；
- 具体 primitive 的循环和成功判定；
- 应用专属 trajectory 格式。

这些能力由 Adapter、sink 或可选插件提供。

## 8. 建议目录结构

```text
rlinf_rollout/
├── runtime_api/v1/           # session/request/response 稳定协议
├── runtime/
│   ├── gateway.py            # 协议无关 Runtime Gateway
│   ├── session_manager.py    # session 状态机与 lease
│   ├── request_router.py     # admission、路由、状态与取消
│   ├── plugin.py             # Adapter Plugin 接口
│   └── service.py            # Runtime 装配和生命周期
├── workers/runtime/
│   ├── env_worker.py         # request-driven EnvWorker
│   └── rollout_worker.py     # serving-style RolloutWorker
├── execution/
│   ├── env_core.py           # 从现有 EnvWorker 抽取
│   └── policy_core.py        # 从现有 HF worker 抽取
├── scheduling/
│   └── inference_batcher.py  # 在线 micro-batching
├── transports/
│   ├── http.py
│   └── grpc.py               # 可后置
└── adapters/
    ├── gym.py
    ├── evaluation.py
    └── rl.py                 # 只负责应用协议，不包含算法

RPent repository
└── robots/.../rollout_adapter.py
    # Toolkit handler / primitive -> Runtime API
```

目录名称是讨论建议，不作为最终实现约束。

## 9. 分阶段实施建议

### Phase A：协议与 core 抽取

1. 定义 `runtime_api/v1` 和 session 状态机。
2. 抽取 `EnvExecutionCore` 和 `PolicyInferenceCore`。
3. 保持现有 collect/eval 测试与行为不变。

### Phase B：Runtime 数据面

1. 实现 request-driven `RuntimeEnvWorker`。
2. 基于 `decoupled_generate` 实现 `RuntimeRolloutWorker`。
3. 增加 micro-batching、correlation routing、deadline 和 backpressure。
4. 用 fake env/model 验证多 session 乱序响应和错误隔离。

### Phase C：Gateway 与通用 Adapter

1. 实现 Runtime Gateway、Session Manager 和 Python SDK。
2. 实现 Gym Adapter，验证 Runtime 不依赖 Agent 语义。
3. 实现批量 evaluation Adapter。

### Phase D：RPent 接入

1. 实现 RPent Toolkit Adapter。
2. 先覆盖 `observe`、`action_step`、`policy_step`。
3. 再迁移 `pi0_pick` 等高层 primitive 为 Gateway Plugin。
4. 对比迁移前后的 RPC 次数、图像传输量、延迟和吞吐。

### Phase E：性能与生产能力

1. 根据 profiling 决定是否引入 gRPC、共享内存或对象存储。
2. 增加 replica 调度、自动扩缩和故障恢复。
3. GPU/模拟器实机验证 continuous batching 和多租户隔离。

## 10. 模块设计讨论清单

后续与其他 agent 讨论时，需要分别确定：

1. Runtime API 字段、兼容性和序列化格式。
2. session 状态机、lease、reset 和故障恢复语义。
3. env slot pool 是否允许动态创建，以及 vector env 的兼容分组规则。
4. micro-batcher 的队列结构、兼容性键、公平性与 deadline 策略。
5. request/result 在 Ray Channel 上的路由和 tensor 传输方式。
6. `EnvExecutionCore`、`PolicyInferenceCore` 的最小公共接口。
7. Adapter Plugin 的进程位置、权限、超时与资源限制。
8. HTTP/gRPC/Python SDK 的第一版选择。
9. metrics、tracing、审计和容量指标。
10. 从现有 collect/eval 模式迁移时的兼容测试和实机验收标准。

## 11. 当前结论

现有 `rlinf_rollout` 已经具备 Runtime 所需的大部分底层能力，尤其是环境/模型实现、Ray
worker、placement、Channel、batch 拆并和 decoupled request routing。主要新增工作应集中在：

- session/request 协议；
- request-driven EnvWorker；
- serving-style RolloutWorker 和通用 micro-batcher；
- Runtime Gateway；
- 面向不同应用的透明 Adapter。

不建议重写 scheduler、环境或模型体系，也不建议让上层应用直接依赖现有 `RolloutClient` 的
Ray 控制面与训练任务语义。
