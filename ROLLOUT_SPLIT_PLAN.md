# Rollout 系统拆分方案（feat/rollout）

> 目标：将 RLinf 的 rollout 系统（具身 HF rollout + env、LLM SGLang/vLLM rollout）拆分为**独立系统**，
> 用于 rollout 效率优化，不含训练系统，并能方便地接入其他训练框架。

## 1. 背景与决策

| 决策项 | 结论 |
|---|---|
| 仓库形态 | 独立仓库。先在本仓 `feat/rollout` 分支下以顶层新包 `rlinf_rollout/` 开发，完成后用 `git subtree split` 导出 |
| 内部调度 | **保留 Ray**（vendor 整个 `rlinf/scheduler/`），仅对外接口去 Ray 化 |
| 覆盖范围 | 具身链路（EnvWorker + HF rollout）**和** LLM 链路（SGLang/vLLM/server） |
| 运行形态 | **仅异步/服务式**（常驻协程 + 守护进程）；同步训练由客户端 SDK 以「采集 N 步 → 取轨迹」模拟 |
| 代码共享 | 模型等代码**先复制**，后续再考虑收敛为共享包 |
| 接口协议 | **v1 冻结 schema**，dataclass + `SCHEMA_VERSION = "v1"` 字段，版本化演进 |
| Phase 5（回接主仓验证） | **暂不做** |

## 2. 现状分析摘要

### 2.1 数据流（具身链路）

```
Actor(FSDP) ──WeightSyncer(NCCL broadcast, 带版本号)──▶ MultiStepRolloutWorker(predict_action_batch)
                                                          ▲ obs │ actions(RolloutResult)
                                                          │     ▼
Reward ◀──obs── EnvWorker(stage_num × VecEnv.chunk_step, prepare_actions, 轨迹缓冲)
   └──reward──▶            │
                           └── Trajectory ──▶ actor_channel ──▶ Actor 训练
```

### 2.2 主要耦合点（需解除）

| 耦合点 | 位置 | 解法 |
|---|---|---|
| 权重同步握手硬编码 actor 组名/rank 拓扑 | `rlinf/workers/rollout/hf/huggingface_worker.py` 的 `setup_weight_sync` | 收敛为 `WeightReceiver` 接口，src 拓扑由握手协议描述 |
| 配置纠缠（读 `cfg.actor.*`、`cfg.algorithm.*`） | env_worker / huggingface_worker | 自包含 `RolloutConfig` |
| 优势/价值逻辑渗入采集（`calculate_adv_and_returns`、bootstrap value） | env_worker | 下沉为可选后处理插件 |
| 轨迹按 actor world_size 切分、`forward_inputs` 隐式约定 | env_worker → actor_channel | `TrajectorySink` 接口，切分粒度由消费方声明，schema 自描述 |
| DAgger expert / RLT 逻辑 | huggingface_worker import `rlinf.algorithms.expert/rlt` | 移入 `plugins/`（可选） |
| Megatron resharding（LLM 权重发送侧） | `rlinf/utils/resharding/` | 作为发送端 SDK 的可选适配器，非 rollout 核心 |

### 2.3 rollout/env worker 的依赖清单（import 分析结果）

- **scheduler**（全部保留、整体 vendor）：`Worker` / `WorkerGroup` / `Channel` / `Cluster` / `CommMapper` / placement / dynamic_scheduler
- **utils 最小集**：`placement`、`data_iter_utils`、`distributed`、`metric_utils`、`nested_dict_process`、`data_process`、`http_client`、`utils`
- **data**：`io_struct.py`、`embodied_io_struct.py`（升级为对外公开 v1 schema）
- **weight_syncer**：`rlinf/hybrid_engines/weight_syncer/`（bucket/patch/compressor，已与拓扑解耦，只依赖注入的 send/recv 协程）
- **envs / models**：`rlinf/envs/` 全部、`rlinf/models/embodiment/`（含 `BasePolicy`、`get_model`）
- **hybrid_engines**：sglang / vllm 引擎封装（LLM 链路）

## 3. 目标架构

```
rlinf_rollout/
├── pyproject.toml            # extras: [embodied] [sglang] [vllm]
├── api/
│   └── v1/                   # 冻结协议（dataclass，SCHEMA_VERSION="v1"）
│       ├── weight.py         # WeightUpdateRequest/Ack, WeightReceiver ABC
│       ├── trajectory.py     # Trajectory / RolloutResult, TrajectorySink ABC
│       └── task.py           # RolloutTask, TaskSource ABC
├── scheduler/                # vendor 自 rlinf/scheduler/（Ray 基建）
├── utils/                    # utils 最小集
├── data/                     # io_struct（内部表示，对外经 api/v1 转换）
├── weight_sync/              # bucket/patch syncer + compressor
├── envs/                     # 复制自 rlinf/envs/
├── models/                   # 复制自 rlinf/models/embodiment/
├── workers/
│   ├── env/                  # 仅异步路径
│   └── rollout/
│       ├── hf/               # 具身 HF rollout（仅异步）
│       ├── sglang/  vllm/    # LLM 引擎 worker
│       └── server/           # OpenAI 兼容 HTTP server / router
├── postprocess/              # 可选：bootstrap value / adv 预计算插件
├── plugins/                  # 可选：DAgger expert / RLT
├── serve/                    # rollout-serve 守护入口
└── client/                   # 训练侧接入 SDK：push_weights / get_trajectories / submit_tasks
```

对外三接口（训练框架无关）：

1. **权重更新**：`WeightReceiver.recv(request) -> Ack`；后端插件化——NCCL broadcast（现有 WeightSyncer）、checkpoint 路径轮询、HTTP push。保留版本号语义（版本随 `RolloutResult.versions` 传出，供消费方做 staleness 处理）。
2. **轨迹/生成输出**：具身 `TrajectorySink.put(Trajectory)`；LLM 输出 `RolloutResult`，另保留 OpenAI 兼容 HTTP 形态。
3. **任务输入**：`TaskSource`——LLM prompt 批次；具身 episode/评测规格。

## 4. 实施计划

### Phase 0 — 骨架与协议先行 ✅ 已完成

1. 新建顶层 `rlinf_rollout/`，独立 `pyproject.toml`（extras：`embodied`/`sglang`/`vllm`），禁止 import 主仓 `rlinf.*`。
2. 编写 `api/v1/` 冻结协议：
   - `WeightUpdateRequest/Ack`（version、bucket/patch 模式、张量清单、src 拓扑描述）
   - `Trajectory` / `RolloutResult`（从 `rlinf/data/embodied_io_struct.py`、`io_struct.py` 提炼；去掉按 actor world_size 切分；`forward_inputs` 改为策略自描述字段）
   - `RolloutTask` + 三个 ABC：`WeightReceiver`、`TrajectorySink`、`TaskSource`

落地情况：

- `rlinf_rollout/{__init__.py,pyproject.toml,README.md,LICENSE}`：`package-dir = {"rlinf_rollout" = "."}`，
  在主仓可直接 `import rlinf_rollout`，subtree split 后同一份配置仍可用。
- `rlinf_rollout/api/v1/common.py`：`SCHEMA_VERSION = "v1"`、`SchemaBase`（`schema_version` 字段 +
  `field_names()` / `to_dict()` / 严格 `from_dict()`）、`SchemaError`、`Metadata`。
- `weight.py`：`WeightSyncMode`/`WeightTransport`/`WeightUpdateStatus`、`TensorSpec`、
  `SourceTopology`（group_name / src_ranks / parallel_sizes / rank_map，取代硬编码 actor 拓扑）、
  `WeightUpdateRequest`、`WeightUpdateAck`、`WeightReceiver`。
- `trajectory.py`：`TensorLayout`/`PartitionAxis`/`FinishReason`、`PolicyInputs`（替代 `forward_inputs`）、
  `ActionChunkResult`、`Trajectory`、`RolloutResult`（LLM）、`ConsumerSpec`（消费方声明切分粒度）、`TrajectorySink`；
  不含 advantages / returns / ref_logprobs / bootstrap_values。
- `task.py`：`TaskKind`/`RolloutMode`、`SamplingParams`、`PromptSpec`、`EpisodeSpec`、`RolloutTask`、`TaskSource`。
- `rlinf_rollout/tests/test_api_v1_schema.py`：锁定全部字段集合与枚举值、校验行为、ABC 可实现性，
  并断言 `rlinf_rollout/` 内不出现 `rlinf.*` import；主仓 CI 经
  `tests/unit_tests/test_rlinf_rollout.py` 代跑该套件。

### Phase 1 — 基建复制（机械性，最小改动）

3. 复制 `rlinf/scheduler/` → `rlinf_rollout/scheduler/`（全目录）。
4. 复制 utils 最小集 + `data/io_struct*` + `weight_syncer/`；批量替换 import 前缀 `rlinf.` → `rlinf_rollout.`；确保 `python -c "import rlinf_rollout.scheduler"` 通过。

### Phase 2 — 具身链路（仅异步形态）

5. 复制 `rlinf/envs/`、`rlinf/models/embodiment/`（含 `get_model` 相关 wiring）。
6. 复制 env worker（`env_worker.py` + `async_env_worker.py`）与 HF rollout worker（`huggingface_worker.py` + `async_huggingface_worker.py`），只保留异步路径，并解耦：
   - 删 `calculate_adv_and_returns` / bootstrap value → `postprocess/` 可选插件
   - `cfg.actor.*` / `cfg.algorithm.*` → 自包含 `RolloutConfig`（新 schema + 校验）
   - `algorithms.expert` / `rlt` → `plugins/`
   - 轨迹出口 `actor_channel.put` → `TrajectorySink`
   - `setup_weight_sync` → `WeightReceiver`（NCCL 后端实现）

### Phase 3 — LLM 链路

7. 复制 `rlinf/workers/rollout/` 的 sglang / vllm / sglang_server / server + `rlinf/hybrid_engines/` 的 sglang / vllm 引擎封装：
   - `ModelParallelComponentPlacement` 的 actor 侧字段改由 `WeightUpdateRequest` 内 src 拓扑描述提供
   - 保留 offload / sleep-wake、dynamic_scheduler 扩缩容
   - 保留 OpenAI 兼容 HTTP server 形态

### Phase 4 — 服务化入口 + 客户端 SDK

8. `rollout-serve --config x.yaml` 守护入口：attach/启动 Ray → 按 placement 拉起 env+rollout（或 LLM engine）worker → 常驻协程 + `RolloutController`。
9. `client/` SDK：`push_weights()`（NCCL 发送协程或 ckpt 路径）、`get_trajectories()`、`submit_tasks()`。
10. 冒烟验证：
    - 具身：eval-only 配置（无 actor，参考 `evaluations/eval_embodied_agent.py`）跑通采集
    - LLM：固定权重 + prompt 文件跑通生成

### Phase 5 —（暂不做）回接主仓 e2e 验证与效率优化

## 5. 验收标准

- [ ] `pip install -e rlinf_rollout[embodied]` / `[sglang]` 可独立安装，不依赖主仓 `rlinf` 包
      （Phase 0 已验证 core：wheel 构建 + `pip install -e --no-deps` 可 import；extras 依赖待 Phase 2/3 实机验证）
- [x] `api/v1` 全部类型带 `SCHEMA_VERSION`，有单元测试锁定字段集合（防止意外破坏兼容）
- [ ] 具身链路 eval-only 冒烟通过；LLM 链路固定权重生成冒烟通过
- [ ] rollout/env worker 代码中不再出现 `cfg.actor.` / `cfg.algorithm.` / actor 组名硬编码
- [ ] `rollout-serve` 可独立启动并常驻，客户端 SDK 可 push 权重、拉取轨迹

## 6. 风险与注意事项

- 复制后两仓代码漂移：v1 协议冻结 + 单测锁 schema 缓解；模型代码后续再收敛
- `hybrid_engines` 对 SGLang/vLLM 版本敏感（按版本分派）：extras 里 pin 版本区间
- 具身 env 重依赖（ManiSkill/LIBERO 等）：保持惰性 import（`get_env_cls` 现状即如此）
- Ray 版本需与 vendored scheduler 匹配；对外接口不暴露任何 Ray 对象
