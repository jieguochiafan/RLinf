# RoboCasa Bin Packing Affinity 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 增加一个独立的 `latency_bin_packing` chunk-step 实验模式，在每个 EnvGroup 内用 K=4 个 CPU bin/core groups 服务 8 个 RoboCasa env，并与“不绑核 + 关闭 batch 优化”的 baseline 对比。

**架构：** 新模式不替换 `latency_balanced_pair`/Core Donation V2。每个 chunk 开始时，用 EMA 预测 latency 通过 LPT/bin packing 将 env 分到 K 个 bin；bin 间并发执行，bin 内同一时刻只执行一个 env 的完整 action chunk，完成后串行调度同 bin 下一个 env；实际耗时回写 EMA。

**技术栈：** Python、NumPy、PyTorch、Hydra/OmegaConf、RLinf `SubprocVectorEnv`、pytest、Ruff、RoboCasa。

---

## 文件结构

- 修改 `rlinf/envs/chunk_runner.py`：注册新 chunk mode 名称。
- 修改 `rlinf/envs/venv/venv.py`：新增 `latency_bin_packing_chunk_step()`、EMA 状态和 LPT 分配 helper。
- 修改 `rlinf/envs/robocasa/robocasa_env.py`：在 RoboCasa `chunk_step()` 中接入新模式，并把返回值转换为现有 rollout 需要的 tensor shape。
- 修改 `rlinf/config.py`：校验 `env.train.latency_bin_packing.bin_count=4` 等参数，并写回规范化 config。
- 修改 `tests/unit_tests/test_chunk_step_parallel.py`：新增 K=4 bin packing 行为单测。

## 任务 1：新增失败测试

**文件：**
- 修改：`tests/unit_tests/test_chunk_step_parallel.py`

- [ ] **步骤 1：编写失败的测试**

在 `ScriptedWorker` 测试区域添加一个 worker，记录发送顺序并由测试手动释放 ready queue。测试 8 个 env、4 个 bins，预测 latency `[8, 7, 6, 5, 4, 3, 2, 1]` 时 LPT 分配应为 `[[0, 7], [1, 6], [2, 5], [3, 4]]`；初始只 dispatch 4 个 env，释放某个 bin 后才 dispatch 同 bin 的第二个 env。

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
source /data1/miliang/RLinf/robocasa_openpi/bin/activate
pytest tests/unit_tests/test_chunk_step_parallel.py::test_latency_bin_packing_dispatches_k_bins_with_serial_envs -v
```

预期：FAIL，报错包含 `latency_bin_packing_chunk_step` 不存在或 mode 未注册。

## 任务 2：实现 VectorEnv bin packing

**文件：**
- 修改：`rlinf/envs/chunk_runner.py`
- 修改：`rlinf/envs/venv/venv.py`
- 测试：`tests/unit_tests/test_chunk_step_parallel.py`

- [ ] **步骤 1：注册模式**

把 `latency_bin_packing` 加入 `CHUNK_STEP_MODES`。

- [ ] **步骤 2：新增状态**

在 `BaseVectorEnv.__init__()` 中增加：

```python
self._bin_packing_predicted_latency_s: list[float] | None = None
self._bin_packing_logged = False
```

- [ ] **步骤 3：新增 LPT helper**

增加 `_build_latency_bin_packing_groups(env_ids, bin_count)`：按预测 latency 降序排序 env local positions，放到当前预测 load 最小的 bin，并返回 `(groups, predicted_loads)`。

- [ ] **步骤 4：新增 chunk step**

实现 `latency_bin_packing_chunk_step(chunk_action, *, bin_count=4, ema_alpha=0.3, initial_latency_ms=None)`：
- 要求 `bin_count >= 1` 且 `bin_count <= len(id)`。
- 要求存在 CPU core groups，用 `_get_slot_cpu_core_groups(bin_count)` 取前 K 个独立 groups。
- 每个 bin 首个 env 初始 dispatch；收到结果后更新该 env EMA，并 dispatch 同 bin 下一个 env。
- dispatch 前把目标 env worker 绑定到该 bin core group。
- 记录 timestamp extra：`bin_index`、`bin_offset`、`predicted_latency_s`、`predicted_bin_load_s`。
- 写 `_last_chunk_profile`：`operation`、`env_count`、`bin_count`、`bin_loads_predicted_s`、`bin_loads_actual_s`、`wait_recv_s`、`dispatch_call_s`、`stack_s`、`total_s`。

- [ ] **步骤 5：运行单测验证通过**

运行：

```bash
pytest tests/unit_tests/test_chunk_step_parallel.py::test_latency_bin_packing_dispatches_k_bins_with_serial_envs -v
```

预期：PASS。

## 任务 3：接入 RoboCasa 和 config

**文件：**
- 修改：`rlinf/envs/robocasa/robocasa_env.py`
- 修改：`rlinf/config.py`
- 测试：`tests/unit_tests/test_chunk_step_parallel.py`

- [ ] **步骤 1：RoboCasa 分支**

在 `RobocasaEnv.chunk_step()` 中新增 `latency_bin_packing` 分支；新增 `_chunk_step_latency_bin_packing()`，调用 `self.env.latency_bin_packing_chunk_step()` 并复用 V2 的奖励、done collapse、auto reset、timing event 处理方式。

- [ ] **步骤 2：Config 校验**

在 `validate_chunk_step_cfg()` 中，当 mode 为 `latency_bin_packing` 时校验：
- `bin_count >= 1`
- `bin_count <= local_num_envs`
- `ema_alpha in (0, 1]`
- `initial_latency_ms` 为空或正数

写回：

```python
env_cfg.latency_bin_packing = OmegaConf.create({
    "bin_count": bin_count,
    "ema_alpha": ema_alpha,
    "initial_latency_ms": initial_latency_ms,
})
```

- [ ] **步骤 3：运行相关单测和 Ruff**

运行：

```bash
ruff check rlinf/envs/chunk_runner.py rlinf/envs/venv/venv.py rlinf/envs/robocasa/robocasa_env.py rlinf/config.py tests/unit_tests/test_chunk_step_parallel.py
pytest tests/unit_tests/test_chunk_step_parallel.py -k 'latency_bin_packing or latency_balanced_pair' -v
```

预期：Ruff PASS；相关 pytest PASS。

## 任务 4：RoboCasa K=4 性能实验

**文件：**
- 生成日志：`logs/latency_balance_eval/20260610_robocasa_pnpcountertocab_latency_bin_packing_k4`

- [ ] **步骤 1：执行实验**

使用 RoboCasa venv 和同一任务/模型/环境数，运行 1 epoch。关键 override：

```bash
env.train.chunk_step_mode=latency_bin_packing
env.train.latency_bin_packing.bin_count=4
cluster.resource_pool.cpu.enabled=true
```

- [ ] **步骤 2：解析结果**

抽取 rollout tqdm wall time、env total mean/max、chunk_loop mean/max、step mean/p95、bin load profile，并与已有 no-binding baseline：

```text
logs/latency_balance_eval/20260610_robocasa_pnpcountertocab_no_cpu_binding_sync_time_major_rerun
```

对比。
