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
├── config/                   # SupportedModel + 自包含 RolloutConfig（schema + 校验）
├── data/                     # io_struct（内部表示）+ convert.py（→ api/v1 边界转换）
├── engines/                  # sglang / vllm 引擎封装（vendor 自 hybrid_engines/）
├── weight_sync/              # bucket/patch syncer + compressor + receiver.py（NCCL 后端）
                              # + llm.py（SourceTopology → 引擎 rank 配对）
├── sinks/                    # TrajectorySink 后端（channel / null）
├── envs/                     # 复制自 rlinf/envs/
├── models/                   # 复制自 rlinf/models/（get_model + embodiment/）
├── workers/
│   ├── env/                  # 仅异步路径（AsyncEnvWorker）
│   └── rollout/
│       ├── hf/               # 具身 HF rollout（仅异步）
│       ├── sglang/  vllm/    # LLM 引擎 worker
│       └── server/           # OpenAI 兼容 HTTP server / router
├── postprocess/              # 可选：bootstrap 塑形 + TrajectoryPostprocessor hook
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

### Phase 1 — 基建复制（机械性，最小改动）✅ 已完成

3. 复制 `rlinf/scheduler/` → `rlinf_rollout/scheduler/`（全目录）。
4. 复制 utils 最小集 + `data/io_struct*` + `weight_syncer/`；批量替换 import 前缀 `rlinf.` → `rlinf_rollout.`；确保 `python -c "import rlinf_rollout.scheduler"` 通过。

落地情况：

- `rlinf_rollout/scheduler/`：整体 vendor 自 `rlinf/scheduler/`（channel / cluster / collective /
  dynamic_scheduler / hardware{accelerators,robots} / manager / placement / worker，共 11 个子包）。
- `rlinf_rollout/utils/`：`placement`、`data_iter_utils`、`distributed`、`metric_utils`、
  `nested_dict_process`、`data_process`、`http_client`、`utils`，另按依赖闭包补 `logging`（`get_logger`）
  与 `timers`（`distributed.ScopedTimer` 依赖 `NamedTimer`）。
- `rlinf_rollout/data/`：`io_struct.py`、`embodied_io_struct.py`，另补闭包所需 `utils.py`
  （`batch_pad_to_fixed_len`）。这些是**内部表示**，对外仍经 `api/v1` 转换。
- `rlinf_rollout/weight_sync/`：vendor 自 `rlinf/hybrid_engines/weight_syncer/`
  （`base` / `bucket_syncer` / `patch_syncer` / `compressor`），import 路径同时改名
  `rlinf.hybrid_engines.weight_syncer` → `rlinf_rollout.weight_sync`。
- import 重写：`rlinf.` → `rlinf_rollout.`（79 个文件，25 个发生改写），随后 `ruff check --preview` +
  `ruff format` 全绿；主仓 `pyproject.toml` 的 docstring 规则 per-file-ignores 扩展为
  `!{rlinf,rlinf_rollout}/scheduler/**.py`，让 vendored scheduler 与原版同规则受检。
- 唯一非机械改动：`Cluster` 的 Ray code-sync（`RLINF_CODE_WORKING_DIR`）原本硬编码 `rlinf` 包名与
  “repo root 含 pyproject.toml + rlinf/” 布局。改为 `Cluster.PACKAGE_NAME = "rlinf_rollout"`，
  并接受三种写法：`auto`（由已导入包定位）、包目录绝对路径、含 `pyproject.toml` 的 checkout 根；
  这样在主仓内和 subtree split 后（`pyproject.toml` 位于包目录内）都成立。
  `Cluster.SYS_NAME` 保持 `"RLinf"`，以沿用 `RLINF_NODE_RANK` 等既有环境变量名。
- 前向引用（保持惰性/type-only，带 `TODO(agent)` 标注，由测试收口成白名单）：
  `scheduler/hardware/robots/franka.py` 的 Lumos 相机（Phase 2 的 `envs`）、
  `scheduler/dynamic_scheduler/manager.py` 的 `SGLangWorker`（Phase 3 的 `workers`）。
- `pyproject.toml`：`packages` 列出全部 17 个子包（由测试与目录树比对保持同步）；核心依赖按
  import 闭包补齐 `packaging` / `pyyaml` / `typing-extensions` / `aiohttp` / `requests` / `pillow`，
  移除未被使用的 `sortedcontainers`。
- `rlinf_rollout/tests/test_phase1_vendoring.py`：断言 vendored 模块齐备、全部内部 import 可解析
  （仅允许白名单前向引用）、无残留 `rlinf.` 路径、`pyproject.toml` 的 `packages` 与目录树一致，
  并逐个 import 全部 vendored 模块 + 覆盖 code-sync 三种写法与两类报错。
  验证：`ray 2.56.1 / torch 2.7.0` 环境下 `pytest rlinf_rollout/tests` = **142 passed**；
  `python -c "import rlinf_rollout.scheduler"` 通过；wheel 构建含 79 个 py 文件、不含 tests，
  `pip install --no-deps` 后可在任意目录 import。


### Phase 2 — 具身链路（仅异步形态）✅ 已完成

5. 复制 `rlinf/envs/`、`rlinf/models/embodiment/`（含 `get_model` 相关 wiring）。
6. 复制 env worker（`env_worker.py` + `async_env_worker.py`）与 HF rollout worker（`huggingface_worker.py` + `async_huggingface_worker.py`），只保留异步路径，并解耦：
   - 删 `calculate_adv_and_returns` / bootstrap value → `postprocess/` 可选插件
   - `cfg.actor.*` / `cfg.algorithm.*` → 自包含 `RolloutConfig`（新 schema + 校验）
   - `algorithms.expert` / `rlt` → `plugins/`
   - 轨迹出口 `actor_channel.put` → `TrajectorySink`
   - `setup_weight_sync` → `WeightReceiver`（NCCL 后端实现）

落地情况：

- **vendor（机械复制 + import 前缀重写，358 个文件扫描 / 192 改写）**
  - `rlinf_rollout/envs/`：`rlinf/envs/` 全量（20 个子包 + `action_utils` / `utils` / `venv` / `wrappers`，
    含 behavior / metaworld / robotwin / habitat / calvin 的非 py 资产）。
  - `rlinf_rollout/models/`：`rlinf/models/`（`get_model` 注册表 + `embodiment/` 19 个子包）。
    删掉 `lingbotvla/sft_builder.py`（SFT-only，且引用主仓也不存在的 `lingbotvla.data.*`）。
  - `rlinf_rollout/utils/` 追加依赖闭包：`rot6d`、`cuda_graph`、`patcher`、`pytree`、
    `torch_functionals`、`omega_resolver`。
  - `rlinf_rollout/data/` 追加：`lerobot_writer`、`lerobot_paths`、
    `datasets/{item,vlm,world_model}.py`、`datasets/dreamzero/**`；`datasets/__init__.py` 用最小
    stub 取代主仓的 `create_rl_dataset`（后者属训练侧）。
  - 唯一新增前向引用：`envs/realworld/franka/franka_env.py` 里 standalone realworld reward worker
    （`rlinf_rollout.workers.reward.reward_worker`，惰性 import + `TODO(agent)`，进白名单）。

- **配置解耦** `rlinf_rollout/config/`
  - `models.py`：`SupportedModel`（开放注册表）/ `EMBODIED_MODEL` / `torch_dtype_from_precision`，
    只取主仓 `rlinf/config.py` 的这三块，不带任何训练侧 builder。
  - `rollout.py`：`DEFAULT_ROLLOUT_CONFIG` + `build_rollout_config()` + `validate_rollout_config()` +
    `RolloutConfig`（派生 batch/chunk-step，两个 worker 共用同一份推导）。key 映射表：

    | 训练仓 | rollout 系统 |
    |---|---|
    | `actor.model` | `policy.model` |
    | `actor.group_name` | `rollout.weight_sync.source.group_name` |
    | `actor.sync_weight_no_wait` | `rollout.weight_sync.no_wait` |
    | `weight_syncer` | `rollout.weight_sync` |
    | `runner.only_eval` | `rollout.mode: eval` |
    | `runner.enable_decoupled_mode` | `rollout.decoupled` |
    | `runner.ckpt_path` / `expert_ckpt_path` | `policy.ckpt_path` / `rollout.plugins.dagger.expert_ckpt_path` |
    | `algorithm.loss_type=rlt_ac` / `embodied_dagger`、`adv_type=opd` | `rollout.plugins.{rlt,dagger,opd}.enabled` |
    | `algorithm.dagger.*` / `rlt_schedule` | `rollout.plugins.dagger.*` / `rollout.plugins.rlt.schedule` |
    | `algorithm.staleness_threshold` | `rollout.staleness_threshold` |
    | `algorithm.{gamma,bootstrap_type}` | `rollout.postprocess.bootstrap.{gamma,type}` |
    | actor world_size（轨迹切分） | `sink.num_shards` |

    `validate_rollout_config` 直接**拒绝** `actor` / `algorithm` / `critic` / `runner` 段并在报错里给出映射提示，
    使解耦成为硬约束而非约定。

- **postprocess / plugins**
  - `postprocess/bootstrap.py`：`BootstrapRewardShaper`（reward-model 混合 + 截断步 `gamma*V(s_final)`，
    `standard` / `done` 两种 bootstrap）与 `estimate_bootstrap_values()`（原 `get_bootstrap_values`）。
  - `postprocess/base.py`：`TrajectoryPostprocessor` ABC + `load_trajectory_postprocessor("module:attr")`。
    优势/回报计算**不再内置**，只能经该 hook 由训练侧插件提供；`use_training_pipeline` 的 actor
    micro-batch 打包路径整体删除。
  - `plugins/expert.py`（expert 配置构建）、`plugins/rlt/`（route / rollout / transition / expert）；
    `build_rlt_route` 改读 `rollout.plugins.rlt.schedule`。二者均惰性 import。

- **两个对外接口的实现**
  - `weight_sync/receiver.py`：`CollectiveWeightReceiver` 实现 `api/v1.WeightReceiver`，
    src 组名 / root rank / world_size 全部来自 `WeightUpdateRequest.source`（`SourceTopology`），
    失败经 `WeightUpdateAck(status=FAILED)` 返回而非抛异常。collective 是**接收侧驱动**（pull）：
    `request.version` 只是「最小期望版本」，`ack.served_version` 才是权威值。
    `weight_sync/__init__.py` 改为 PEP 562 懒加载，使 receiver 在缺少 `torch.distributed.tensor.DTensor`
    的 torch 上仍可 import。
  - `sinks/channel.py`：`ChannelTrajectorySink`（按 `ConsumerSpec` 分片 + 可选 per-partition channel key，
    round-robin，`flush()` 等待所有 async put）与 `NullTrajectorySink`（eval / 测试）。
  - `data/convert.py`：内部 `embodied_io_struct` → `api/v1` 的边界转换
    （`trajectory_to_api` / `chunk_result_to_api`，`forward_inputs` → 自描述 `PolicyInputs`）。
    默认 `validate=False`：epoch 末尾观测只贡献 reward/done 不贡献 action，各字段步数天然差 1。

- **workers（仅异步）**
  - `workers/rollout/hf/huggingface_worker.py`：`AsyncMultiStepRolloutWorker`（原 base + async 两个类合一）。
    常驻 `generate()`；`sync_weights()` / `build_weight_update_request()` 取代
    `setup_weight_sync` + `sync_model_from_actor`；后台权重同步、staleness 节流、decoupled 路由、
    DAgger/RLT/OPD 分支保留；epoch 末尾那一次 serve 保持原来的「无 logprobs/versions」语义。
  - `workers/env/env_worker.py`：`AsyncEnvWorker`。删除 `_init_pipeline_params` /
    `compute_advantages_and_returns` / `prepare_pipeline_batch` / `pack_pipeline_micro_batches` /
    `send_rollout_trajectories_pipeline` / `get_actor_split_num`；轨迹经
    `publish_trajectories()` → `TrajectorySink`（`set_trajectory_sink()` 注入，或由
    `sink.num_shards` 自动构建 channel sink）；reward worker 交互仍走 channel（组名来自 `reward.group_name`）。

- **测试**（`rlinf_rollout/tests/`，均不需要 Ray / GPU / 模拟器 SDK）
  - `test_phase2_embodied.py`：envs/models 子包齐备；`get_env_cls` 覆盖 `SupportedEnvType` 全部成员；
    模型注册表包含全部具身策略；**AST 扫描**证明两个 worker 的代码（跳过 docstring）不含
    `cfg.actor.` / `cfg.algorithm.` / `cfg.runner.`、不含 `"actor"` 组名与 `actor_channel`、
    不含 `OmegaConf.select(cfg, "actor.…")`；已删方法确实消失；入口函数只有 async 形态；
    `RolloutConfig` 默认值 / 派生量 / 7 类非法配置 / 4 个训练段拒绝。
  - `test_phase2_seams.py`：`BootstrapRewardShaper`（standard/done/disabled/no-auto-reset/reward 混合）、
    `estimate_bootstrap_values`、postprocess hook 加载与报错、api/v1 转换、两个 sink 的分片与 flush、
    `CollectiveWeightReceiver`（apply / 握手只跑一次 / FAILED ack / 非 collective 传输拒绝 / describe）。
  - `test_phase1_vendoring.py` 同步更新：forward-reference 白名单加 reward worker；
    stale-path 检查改为正则 `(?<![\w.])rlinf\.[A-Za-z_]`，从而不再误报散文里的 “rlinf.”
    和 vendored `models/embodiment/openvla_oft/rlinf` 子包（其名字来自 `implement_version`）。

  验证（本机 torch 2.4 / ray 2.39）：`pytest rlinf_rollout/tests` = **217 passed, 45 skipped**；
  `ruff check --preview rlinf_rollout` + `ruff format --check` 全绿；
  worker 全链路 import 在 shim 掉本机 `ray<2.47` / `DTensor` 位置差异后逐个通过；
  wheel 构建含 461 个 py + 6 个资产文件、不含 tests。

- `pyproject.toml`：`packages` 扩到 113 项（由测试与目录树比对保持同步）；`package-data` 增加
  `**/*.json` / `**/*.jsonl`，并显式补 `envs/calvin/calvin_cfg/.hydra/*.yaml`
  （setuptools 的递归 glob 会跳过点目录，而 `CalvinEnv` 运行时要读它）。

### Phase 3 — LLM 链路 ✅ 已完成

7. 复制 `rlinf/workers/rollout/` 的 sglang / vllm / sglang_server / server + `rlinf/hybrid_engines/` 的 sglang / vllm 引擎封装：
   - `ModelParallelComponentPlacement` 的 actor 侧字段改由 `WeightUpdateRequest` 内 src 拓扑描述提供
   - 保留 offload / sleep-wake、dynamic_scheduler 扩缩容
   - 保留 OpenAI 兼容 HTTP server 形态

落地情况：

- **引擎封装** `rlinf_rollout/engines/`（vendor 自 `rlinf/hybrid_engines/{sglang,vllm}`）
  - `engines/sglang/common/`：`io_struct` / `sgl_engine` / `sgl_scheduler` /
    `tokenizer_manager` / `detokenizer_manager`；`engines/vllm/vllm_0_8_5/`：
    `executor` / `worker` / `weight_loader`。
  - 版本分派从 worker 层下移到引擎层：`engines/sglang/__init__.py`（`sglang` 0.4.4–0.5.4 →
    `Engine` / `io_struct`）、`engines/vllm/__init__.py`（`vllm` ≥0.8.5,<0.9 → `VLLMExecutor`），
    于 import 时即报版本错，而非等到起引擎才失败。
  - `Patcher.add_patch` 的替换目标是**字符串**，一并改写为
    `rlinf_rollout.engines.sglang.common.*`（测试用 AST 取出全部字符串常量断言前缀）。

- **权重同步去 actor 化** `rlinf_rollout/weight_sync/llm.py`
  - `SourceLayout.from_topology(SourceTopology)`：`parallel_sizes["tp"|"pp"]` / `world_size` /
    `group_name` 取代 `placement.actor_{tp,pp,world}_size` 与 `cfg.actor.group_name`。
  - `RankMapper` / `CollocateRankMapper` / `DisaggRankMapper`：算法逐行照搬，仅把入参从
    placement 换成 `SourceLayout`，并把标识符 `actor_*` 全量改名 `source_*`（测试断言
    `RankMapper` 的方法签名里不再出现 `placement`，且 LLM 链路代码中不出现 `actor_tp_size` 等）。
    输出与原实现逐项一致（tp=4→rollout tp=2, world=8 的映射经交叉核对）。
  - `SourceTopology.rank_map`（键 `"<dp_rank>,<tp_rank>"`，`format_rollout_rank` /
    `parse_rank_map` 负责编解码）可**完全绕过**推导，供做特殊 resharding 的训练侧直接指定配对。
  - `EngineWeightSyncSetup`：跨进程（`spawn`）传给引擎的**纯数据**载荷（frozen dataclass，
    可 pickle，不含 Ray / OmegaConf 对象），字段为 `source` / `rollout_{tp,world}_size` /
    `sync_mode` / `presharded_weights`（原 `cfg.actor.training_backend != "fsdp"`）/
    `validate_first_sync`。取代原来直接把 `(placement, 整份训练 cfg)` 塞进引擎进程的做法。
  - `sgl_scheduler.Scheduler.init_rlinf_worker(parent_address, weight_reload,
    weight_sync_setup, enforce_eager)` 与 `vllm ... worker.VLLMWorker(weight_sync_setup,
    parent_address, enforce_eager, ...)`：`self.cfg` / `_actor_group_name` /
    `actor_weight_rank` / `rollout_sync_mode` 全部消失，改为 `_weight_setup` +
    `_weight_source_{group_name,rank}`。原 `cfg.runner.resume_dir is not None` 静默关闭首次
    校验的逻辑**不再内置**：改由调用方显式设 `rollout.validate_weight_first_sync: false`
    （config 注释与 dataclass docstring 都点明了这条约束）。

- **rollout-only placement** `rlinf_rollout/utils/rollout_placement.py`
  - `RolloutComponentPlacement`：只要求 `rollout` 组件（`reward` 可选），不再断言 actor GPU 存在；
    模式由 `rollout.placement_mode`（collocated / disaggregated / auto）**声明**而非由 actor/rollout
    GPU 是否重叠推断；collocated 的 stride 由
    `rollout.weight_sync.source.parallel_sizes.tp // rollout.tensor_parallel_size` 得到——
    这正是「actor 侧字段改由 src 拓扑提供」的落点。对外保留
    `is_{collocated,disaggregated,auto,pipeline}` / `rollout_{dp,tp,world}_size` /
    `rollout_sync_mode` / `num_gpus_per_engine`。
  - 唯一改动 vendored 文件的地方：`PlacementMode` / `RolloutSyncMode` /
    `placement_mode_to_rollout_sync_mode` 抽到无 Ray 依赖的 `utils/placement_modes.py`，
    `utils/placement.py` 原地 re-export（既有 `from rlinf_rollout.utils.placement import
    PlacementMode` 写法不变）。这样 `weight_sync/llm.py` 的纯算术逻辑可以在没有 Ray 的环境里
    直接单测（测试断言 `placement_modes.py` 只 import `enum`）。

- **配置** `rlinf_rollout/config/rollout.py`
  - 新增 `rollout.kind`（`embodied` 默认 / `llm`）作为链路开关；`build_rollout_config` /
    `validate_rollout_config` 据此分派，两条链路共用「拒绝训练段」与 weight-sync 校验。
    `DEFAULT_LLM_ROLLOUT_CONFIG` 独立于具身默认值，因此 LLM 配置里不会冒出 `policy` /
    `env` / `plugins` / `postprocess`。key 映射表：

    | 训练仓 | rollout 系统 |
    |---|---|
    | `algorithm.sampling_params` | `rollout.sampling_params` |
    | `algorithm.group_size` | `rollout.group_size` |
    | `data.rollout_batch_size` | `rollout.batch_size` |
    | `runner.seq_length` | `rollout.max_model_len` |
    | `runner.resume_dir is not None` | `rollout.validate_weight_first_sync: false` |
    | `actor.tokenizer.trust_remote_code` | `rollout.model.trust_remote_code` |
    | `actor.group_name` | `rollout.weight_sync.source.group_name` |
    | `actor.training_backend != "fsdp"` | `rollout.weight_sync.source.presharded` |
    | `placement.actor_{tp,pp}_size` | `rollout.weight_sync.source.parallel_sizes` |
    | `placement.actor_world_size` | `rollout.weight_sync.source.world_size` |
    | 由 actor/rollout GPU 重叠推断的模式 | `rollout.placement_mode` |
    | `rollout_server.online_router` | `rollout.online_router` |
    | `rollout_server.tracking_rollout` | `rollout.tracking_server` |

  - `LLMRolloutConfig`：派生视图（`total_tasks = batch_size * group_size`、
    `num_gpus_per_engine`、`trust_remote_code`、`source_presharded`、`only_eval`），
    并提供 `weight_source_topology()` 直接产出 api/v1 `SourceTopology`；eval 模式调用它会报错
    （无权重发送方）。校验覆盖 backend 白名单、并行度/批量正整数、`max_new_tokens`、
    `placement_mode`、`serving_mode`、以及 `source.world_size % source.parallel_sizes.tp == 0`。

- **workers** `rlinf_rollout/workers/rollout/`
  - `utils.py`：`RunningStatusManager` / `RolloutEngineStats` / `MetaInfoStatsCollector` /
    打印助手 + `get_rollout_backend_worker`（读 `rollout.rollout_backend` 与
    `rollout.sglang.serving_mode`）。`RankMapper` 已迁往 `weight_sync/llm.py`。
  - `sglang/sglang_worker.py`（`SGLangWorker`）：`sync_model_from_actor` → `sync_weights`，
    新增 `build_weight_sync_setup()`；保留 `Release/ResumeMemoryOccupation` 的 offload、
    `RolloutScalingScheduler` 扩缩容、`rollout_serverless` 常驻路径。`config_rollout` 覆盖时，
    派生视图按 `OmegaConf.merge(cfg, {"rollout": config_rollout})` 生成，多引擎组各用自己的
    weight source 与批量。
  - `sglang/sglang_worker_server.py`（`SGLangWorkerWithHTTPServer`）：逐字保留
    `/v1/chat/completions`（含 0.4.x/0.5.x 两套 OpenAI adapter 与 tool-call 400 降级），
    只把 host/port 改读 `rollout.sglang.server.*`。
  - `vllm/vllm_worker.py`（`VLLMWorker`）：`sync_model_from_actor` → `sync_weights`；
    `enable_sleep_mode` + `collective_rpc("offload_model_weights")` 原样保留；
    worker class 常量指向 `rlinf_rollout.engines.vllm.vllm_0_8_5.worker.VLLMWorker`。
    顺带修掉两个上游隐患：`_validate_weight_at_first` 里 `print_vllm_outputs(request_output,
    tokenizer)` 的实参数量不符，以及 detokenize 分支下 `check_input_ids()` 会对 `None` 断言。
  - `sglang_server/`（`launcher` / `server_worker` / `router_worker`）：纯机械 vendor，
    本就只消费传入的 `router_server_args`（通常是 `cfg.rollout`），无训练段耦合。
  - `server/`：`ServerRolloutWorker` 的 `runner.seq_length - max_new_tokens` 改为
    `rollout.max_model_len - rollout.sampling_params.max_new_tokens`（并显式断言两者关系），
    storage 配置由 `rollout.tracking_server.storage` 提供（原来硬编码 `None`）；
    `OnlineRouterWorker` 的 placement 换成 `RolloutComponentPlacement`（只用 `rollout_dp_size`），
    并修掉 `request.stop is None` 时 `sampling_params` 未定义的分支。

- **仍留在 vendored scheduler 里的训练侧编排**：`dynamic_scheduler` 的
  `RolloutManager` / `ActorManager` / `SchedulerWorker` 读 `cfg.algorithm.*` / `cfg.actor.*`，
  它们是 driver 侧的自动扩缩容编排，不在 rollout worker 进程内运行；rollout 侧只用
  `RolloutScalingScheduler`（仅依赖 worker + channel）。Phase 4 的 `serve/` 入口决定是否
  把 driver 侧编排一并自包含。同理 `data/io_struct.py` 的 `BatchResizingIterator`
  （actor 训练用迭代器）仍带 `cfg.actor.seed`，rollout 链路不引用它。

- **测试** `rlinf_rollout/tests/test_phase3_llm.py`（140 项，不需要 Ray / GPU / sglang / vllm）
  - vendor 完整性：25 个引擎/worker 模块存在、`hybrid_engines/` 已消失、两个引擎的版本闸门、
    `Patcher` 字符串目标全部指向 vendored 路径、vLLM worker class 常量对应真实文件。
  - 解耦（对 14 个源文件逐个 AST 扫描，跳过 docstring）：不含 `cfg.{actor,algorithm,runner,data}.`
    与 `rollout_server.`；不含以 `actor.` / `algorithm.` / `runner.` / `rollout_server.` 开头的
    字符串选择器；不含 `_actor_group_name` / `actor_weight_rank` / `"actor"`；
    不含 `ModelParallelComponentPlacement` / `actor_tp_size` / `actor_world_size`。
  - 跨进程契约（这类漂移只会在起引擎时才暴露，故静态锁定）：`init_rlinf_worker` 的位置实参个数
    落在 `[必填, 全部]` 区间内；vLLM executor 交给 worker 的 kwargs 字典是
    `VLLMWorker.__init__` 形参的子集且含 `weight_sync_setup` / `enforce_eager` / `parent_address`。
  - 行为：LLM 配置默认值/派生量/`SourceTopology` 生成/eval 无 source/4 个训练段拒绝/12 类非法配置；
    `SourceLayout` 投影与欠定拓扑报错；rank_map 键往返与两种 sync mode 下的满射配对；
    tp=1 时的 row-major 映射；显式 rank_map 短路；`EngineWeightSyncSetup` 的 describe/报错/可 pickle；
    `weight_sync` 包导出；`placement_modes` 只 import `enum`；`RolloutComponentPlacement`
    只读 `rollout`/`reward` 两个组件且 stride 源自 src 拓扑。
  - 保留项：offload/sleep-wake 与 `RolloutScalingScheduler`、`/v1/chat/completions`、
    `/v1/completions`、`launch_sglang_router_and_server` 均有断言。
  - `test_phase1_vendoring.py` 同步：`workers.rollout.sglang.sglang_worker` 从
    forward-reference 白名单移除（已真实存在），`dynamic_scheduler/manager.py` 的
    `TODO(agent)` 改为「type-only，避免让 scheduler 的使用者都装上 SGLang」。

  验证（本机 torch 2.4 / ray 2.39 / 无可用 sglang / vllm 0.6.x）：
  `pytest rlinf_rollout/tests` = **349 passed, 53 skipped**；
  主仓 `pytest tests/unit_tests/test_rlinf_rollout.py` = 1 passed；
  `ruff check --preview rlinf_rollout` + `ruff format --check` 全绿（F821 等未定义名检查覆盖全树）；
  引擎/worker 的新增 `self._weight_*` / `self._enforce_eager` 属性经 AST 核对「有写有读、无孤儿」；
  wheel 构建含 490 个 py 文件（Phase 2 为 461）、不含 tests，13 个 `engines/` 与 16 个
  `workers/rollout/` 模块齐备。引擎级真机冒烟（起 SGLang/vLLM、跑权重同步）需 GPU，留待 Phase 4。

- `pyproject.toml`：`packages` 扩到 122 项（新增 5 个 `engines.*` 与 4 个
  `workers.rollout.*`，由测试与目录树比对保持同步）；`[sglang]` / `[vllm]` extras 补
  `fastapi` / `uvicorn`（HTTP server 与 router worker 需要）。

### Phase 4 — 服务化入口 + 客户端 SDK

8. `rollout-serve --config x.yaml` 守护入口：attach/启动 Ray → 按 placement 拉起 env+rollout（或 LLM engine）worker → 常驻协程 + `RolloutController`。
9. `client/` SDK：`push_weights()`（NCCL 发送协程或 ckpt 路径）、`get_trajectories()`、`submit_tasks()`。
10. 冒烟验证：
    - 具身：eval-only 配置（无 actor，参考 `evaluations/eval_embodied_agent.py`）跑通采集
    - LLM：固定权重 + prompt 文件跑通生成

### Phase 5 —（暂不做）回接主仓 e2e 验证与效率优化

## 5. 验收标准

- [ ] `pip install -e rlinf_rollout[embodied]` / `[sglang]` 可独立安装，不依赖主仓 `rlinf` 包
      （Phase 1 已验证 core：wheel 构建 + `pip install --no-deps` 后可在任意目录 import，
      且 `rlinf_rollout` 内不出现 `rlinf.*` import；Phase 2 wheel 含 461 py + 6 资产、不含 tests；
      Phase 3 wheel 含 490 py、13 个 `engines/` 与 16 个 `workers/rollout/` 模块；
      `[embodied]` / `[sglang]` / `[vllm]` extras 的引擎与模拟器依赖待实机验证）
- [x] `api/v1` 全部类型带 `SCHEMA_VERSION`，有单元测试锁定字段集合（防止意外破坏兼容）
- [ ] 具身链路 eval-only 冒烟通过；LLM 链路固定权重生成冒烟通过
- [x] rollout/env worker 代码中不再出现 `cfg.actor.` / `cfg.algorithm.` / actor 组名硬编码
      （`test_phase2_embodied.py` 与 `test_phase3_llm.py` 以 AST 扫描锁定具身与 LLM 两条链路，
      含 `OmegaConf.select` 里的字符串选择器、`ModelParallelComponentPlacement` 与
      `actor_{tp,world}_size`；`validate_rollout_config` 另在运行期拒绝
      `actor` / `algorithm` / `critic` / `runner` 配置段）
- [ ] `rollout-serve` 可独立启动并常驻，客户端 SDK 可 push 权重、拉取轨迹

## 6. 风险与注意事项

- 复制后两仓代码漂移：v1 协议冻结 + 单测锁 schema 缓解；模型代码后续再收敛
- `hybrid_engines` 对 SGLang/vLLM 版本敏感（按版本分派）：extras 里 pin 版本区间
- 具身 env 重依赖（ManiSkill/LIBERO 等）：保持惰性 import（`get_env_cls` 现状即如此）
- Ray 版本需与 vendored scheduler 匹配；对外接口不暴露任何 Ray 对象
