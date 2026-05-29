# Profile-Based Resource Orchestration 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 构建 `toolkits.resource_orchestration`，对 embodiment YAML 的 actor/rollout MPS 配额做 profile、估算和选择，并输出兼容 resource pool 的 plan JSON。

**架构：** 新增独立 orchestration toolkit，负责配置解析、候选 SM pair 管理、profile 结果归一化、时间估算、候选选择、报告和 plan JSON 写入。env/model profile 复用 `toolkits.rollout_eval.benchmark`，actor profile 通过可替换 adapter 接入 `toolkits.training_eval` 或测试 stub，scheduler 运行时不变。

**技术栈：** Python dataclasses、Hydra/OmegaConf、现有 `rlinf.scheduler.resource_pool` binding 类型、pytest、JSON/Markdown 报告。

---

## 文件结构

- 创建：`toolkits/resource_orchestration/__init__.py`
  - 包入口，导出公共 dataclass 和主函数。
- 创建：`toolkits/resource_orchestration/types.py`
  - 定义 `CandidatePair`、`StageThroughput`、`CandidateProfile`、`CandidateEstimate`、`SelectionResult`、`OrchestrationRequest`、`ConfigSummary`。
- 创建：`toolkits/resource_orchestration/candidates.py`
  - 解析 `actor_sm:rollout_sm` 字符串，默认候选生成，SM 合法性验证。
- 创建：`toolkits/resource_orchestration/config_loader.py`
  - 读取 Hydra config，提取 chunk/rollout/resource_pool/placement 摘要。
- 创建：`toolkits/resource_orchestration/estimator.py`
  - 吞吐归一化与 rollout/training/epoch 时间估算。
- 创建：`toolkits/resource_orchestration/selector.py`
  - 根据 `max(rollout_time, training_time)` 和 tie-breaker 选择候选。
- 创建：`toolkits/resource_orchestration/plan_writer.py`
  - 使用现有 placement/resource_pool solver 生成或继承 bindings，更新 actor/rollout GPU SM 并写 plan JSON。
- 创建：`toolkits/resource_orchestration/reporting.py`
  - 写 `profiles/*.json`、`summary.json`、`summary.md`。
- 创建：`toolkits/resource_orchestration/profilers.py`
  - 定义 profile adapter 协议，提供 rollout_eval/training_eval 接入点和可测试 orchestrator 调用接口。
- 创建：`toolkits/resource_orchestration/run.py`
  - CLI 入口，串起 config、profile、estimate、select、report、plan。
- 创建：`tests/unit_tests/test_resource_orchestration_candidates.py`
- 创建：`tests/unit_tests/test_resource_orchestration_estimator.py`
- 创建：`tests/unit_tests/test_resource_orchestration_selector.py`
- 创建：`tests/unit_tests/test_resource_orchestration_plan_writer.py`
- 创建：`tests/unit_tests/test_resource_orchestration_reporting.py`
- 创建：`tests/unit_tests/test_resource_orchestration_run.py`

## 任务 1：候选 MPS Pair 解析

**文件：**
- 创建：`toolkits/resource_orchestration/__init__.py`
- 创建：`toolkits/resource_orchestration/types.py`
- 创建：`toolkits/resource_orchestration/candidates.py`
- 测试：`tests/unit_tests/test_resource_orchestration_candidates.py`

- [ ] **步骤 1：编写失败的候选解析测试**

在 `tests/unit_tests/test_resource_orchestration_candidates.py` 写入：

```python
import pytest

from toolkits.resource_orchestration.candidates import (
    default_candidate_pairs,
    parse_candidate_pairs,
)
from toolkits.resource_orchestration.types import CandidatePair


def test_parse_candidate_pairs_accepts_actor_rollout_pairs() -> None:
    assert parse_candidate_pairs("30:70,40:60") == (
        CandidatePair(actor_sm=30, rollout_sm=70),
        CandidatePair(actor_sm=40, rollout_sm=60),
    )


def test_default_candidate_pairs_are_complementary() -> None:
    assert default_candidate_pairs() == (
        CandidatePair(actor_sm=20, rollout_sm=80),
        CandidatePair(actor_sm=30, rollout_sm=70),
        CandidatePair(actor_sm=40, rollout_sm=60),
        CandidatePair(actor_sm=50, rollout_sm=50),
        CandidatePair(actor_sm=60, rollout_sm=40),
        CandidatePair(actor_sm=70, rollout_sm=30),
        CandidatePair(actor_sm=80, rollout_sm=20),
    )


@pytest.mark.parametrize("raw", ["30", "30-70", "30:70:0", "15:85", "60:50"])
def test_parse_candidate_pairs_rejects_invalid_values(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_candidate_pairs(raw)
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
pytest tests/unit_tests/test_resource_orchestration_candidates.py -v
```

预期：FAIL，报错包含 `ModuleNotFoundError: No module named 'toolkits.resource_orchestration'`。

- [ ] **步骤 3：实现类型与解析函数**

创建 `toolkits/resource_orchestration/__init__.py`：

```python
"""Profile-based resource orchestration toolkit."""
```

创建 `toolkits/resource_orchestration/types.py`：

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidatePair:
    """Actor/rollout MPS SM allocation candidate."""

    actor_sm: int
    rollout_sm: int

    @property
    def candidate_id(self) -> str:
        """Stable id used for reports and profile files."""
        return f"actor{self.actor_sm}_rollout{self.rollout_sm}"
```

创建 `toolkits/resource_orchestration/candidates.py`：

```python
from __future__ import annotations

from rlinf.scheduler.resource_pool.gpu_binding import validate_sm_percent

from toolkits.resource_orchestration.types import CandidatePair


def _validate_pair(pair: CandidatePair) -> CandidatePair:
    validate_sm_percent(pair.actor_sm)
    validate_sm_percent(pair.rollout_sm)
    if pair.actor_sm + pair.rollout_sm > 100:
        raise ValueError(
            "candidate pair exceeds one GPU SM budget: "
            f"actor_sm={pair.actor_sm}, rollout_sm={pair.rollout_sm}"
        )
    return pair


def default_candidate_pairs() -> tuple[CandidatePair, ...]:
    """Return default complementary actor/rollout MPS pairs."""
    return tuple(
        CandidatePair(actor_sm=actor_sm, rollout_sm=100 - actor_sm)
        for actor_sm in range(20, 90, 10)
    )


def parse_candidate_pairs(raw: str | None) -> tuple[CandidatePair, ...]:
    """Parse comma-separated actor:rollout SM candidate pairs."""
    if raw is None or not raw.strip():
        return default_candidate_pairs()

    pairs: list[CandidatePair] = []
    for item in raw.split(","):
        token = item.strip()
        parts = token.split(":")
        if len(parts) != 2:
            raise ValueError(f"candidate pair must use actor:rollout syntax, got {token!r}")
        try:
            actor_sm = int(parts[0])
            rollout_sm = int(parts[1])
        except ValueError as exc:
            raise ValueError(f"candidate pair values must be integers, got {token!r}") from exc
        pairs.append(_validate_pair(CandidatePair(actor_sm=actor_sm, rollout_sm=rollout_sm)))

    if not pairs:
        raise ValueError("at least one candidate pair is required")
    return tuple(pairs)
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
pytest tests/unit_tests/test_resource_orchestration_candidates.py -v
```

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add toolkits/resource_orchestration tests/unit_tests/test_resource_orchestration_candidates.py
git commit -s -m "feat: add resource orchestration candidates"
```

## 任务 2：配置摘要加载与 Chunk 数计算

**文件：**
- 修改：`toolkits/resource_orchestration/types.py`
- 创建：`toolkits/resource_orchestration/config_loader.py`
- 测试：`tests/unit_tests/test_resource_orchestration_config_loader.py`

- [ ] **步骤 1：编写失败的配置摘要测试**

创建 `tests/unit_tests/test_resource_orchestration_config_loader.py`：

```python
from omegaconf import OmegaConf

from toolkits.resource_orchestration.config_loader import build_config_summary


def test_build_config_summary_extracts_rollout_chunk_count() -> None:
    cfg = OmegaConf.create(
        {
            "cluster": {
                "component_placement": {"actor": "0-1", "rollout": "0-1", "env": "0-1"},
                "resource_pool": {"enabled": True, "gpu": {"enabled": True, "mode": "mps"}},
            },
            "env": {
                "train": {
                    "total_num_envs": 8,
                    "max_steps_per_rollout_epoch": 40,
                    "max_episode_steps": 40,
                }
            },
            "actor": {
                "global_batch_size": 16,
                "micro_batch_size": 4,
                "model": {"num_action_chunks": 5},
            },
            "algorithm": {"rollout_epoch": 2, "update_epoch": 4},
            "rollout": {"pipeline_stage_num": 2},
        }
    )

    summary = build_config_summary(cfg)

    assert summary.total_num_envs == 8
    assert summary.chunk_size == 5
    assert summary.chunk_steps_per_env == 8
    assert summary.rollout_chunk_count == 128
    assert summary.actor_global_batch_size == 16
    assert summary.update_epoch == 4
    assert summary.resource_pool_mode == "mps"


def test_build_config_summary_rejects_non_mps_resource_pool() -> None:
    cfg = OmegaConf.create(
        {
            "cluster": {"resource_pool": {"enabled": True, "gpu": {"enabled": True, "mode": "mig"}}},
            "env": {"train": {"total_num_envs": 1, "max_steps_per_rollout_epoch": 5}},
            "actor": {"global_batch_size": 1, "micro_batch_size": 1, "model": {"num_action_chunks": 5}},
            "algorithm": {"rollout_epoch": 1, "update_epoch": 1},
            "rollout": {"pipeline_stage_num": 1},
        }
    )

    try:
        build_config_summary(cfg)
    except ValueError as exc:
        assert "MPS" in str(exc)
    else:
        raise AssertionError("expected non-MPS config to be rejected")
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
pytest tests/unit_tests/test_resource_orchestration_config_loader.py -v
```

预期：FAIL，报错包含 `ModuleNotFoundError` 或 `cannot import name 'build_config_summary'`。

- [ ] **步骤 3：扩展类型并实现配置摘要**

在 `toolkits/resource_orchestration/types.py` 追加：

```python
@dataclass(frozen=True)
class ConfigSummary:
    """Config fields required by resource orchestration."""

    total_num_envs: int
    episode_env_steps: int
    chunk_size: int
    chunk_steps_per_env: int
    rollout_epoch: int
    rollout_chunk_count: int
    update_epoch: int
    actor_global_batch_size: int
    actor_micro_batch_size: int
    pipeline_stage_num: int
    resource_pool_mode: str
```

创建 `toolkits/resource_orchestration/config_loader.py`：

```python
from __future__ import annotations

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from toolkits.resource_orchestration.types import ConfigSummary


def _select_int(cfg: DictConfig, path: str, default: int | None = None) -> int:
    value = OmegaConf.select(cfg, path, default=default)
    if value is None:
        raise ValueError(f"missing required config value: {path}")
    return int(value)


def load_hydra_config(
    *,
    config_path: str,
    config_name: str,
    overrides: tuple[str, ...] = (),
) -> DictConfig:
    """Load a Hydra config from an absolute or relative config directory."""
    from pathlib import Path

    abs_config_path = str(Path(config_path).resolve())
    with initialize_config_dir(version_base="1.1", config_dir=abs_config_path):
        return compose(config_name=config_name, overrides=list(overrides))


def build_config_summary(cfg: DictConfig) -> ConfigSummary:
    """Extract orchestration fields from a loaded RLinf config."""
    mode = str(OmegaConf.select(cfg, "cluster.resource_pool.gpu.mode", default=""))
    if mode != "mps":
        raise ValueError(f"profile-based orchestration v1 requires MPS resource_pool, got {mode!r}")

    total_num_envs = _select_int(cfg, "env.train.total_num_envs")
    episode_env_steps = _select_int(
        cfg,
        "env.train.max_steps_per_rollout_epoch",
        default=OmegaConf.select(cfg, "env.train.max_episode_steps"),
    )
    chunk_size = _select_int(cfg, "actor.model.num_action_chunks")
    if chunk_size <= 0:
        raise ValueError(f"actor.model.num_action_chunks must be > 0, got {chunk_size}")
    if episode_env_steps % chunk_size != 0:
        raise ValueError(
            "env.train.max_steps_per_rollout_epoch must be divisible by "
            "actor.model.num_action_chunks"
        )
    chunk_steps_per_env = episode_env_steps // chunk_size
    rollout_epoch = _select_int(cfg, "algorithm.rollout_epoch", default=1)
    update_epoch = _select_int(cfg, "algorithm.update_epoch", default=1)
    rollout_chunk_count = total_num_envs * rollout_epoch * chunk_steps_per_env

    return ConfigSummary(
        total_num_envs=total_num_envs,
        episode_env_steps=episode_env_steps,
        chunk_size=chunk_size,
        chunk_steps_per_env=chunk_steps_per_env,
        rollout_epoch=rollout_epoch,
        rollout_chunk_count=rollout_chunk_count,
        update_epoch=update_epoch,
        actor_global_batch_size=_select_int(cfg, "actor.global_batch_size"),
        actor_micro_batch_size=_select_int(cfg, "actor.micro_batch_size"),
        pipeline_stage_num=_select_int(cfg, "rollout.pipeline_stage_num", default=1),
        resource_pool_mode=mode,
    )
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
pytest tests/unit_tests/test_resource_orchestration_config_loader.py -v
```

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add toolkits/resource_orchestration/types.py toolkits/resource_orchestration/config_loader.py tests/unit_tests/test_resource_orchestration_config_loader.py
git commit -s -m "feat: summarize orchestration config"
```

## 任务 3：吞吐归一化与时间估算

**文件：**
- 修改：`toolkits/resource_orchestration/types.py`
- 创建：`toolkits/resource_orchestration/estimator.py`
- 测试：`tests/unit_tests/test_resource_orchestration_estimator.py`

- [ ] **步骤 1：编写失败的估算测试**

创建 `tests/unit_tests/test_resource_orchestration_estimator.py`：

```python
from toolkits.resource_orchestration.estimator import estimate_candidate
from toolkits.resource_orchestration.types import (
    CandidatePair,
    ConfigSummary,
    StageThroughput,
)


def _summary() -> ConfigSummary:
    return ConfigSummary(
        total_num_envs=8,
        episode_env_steps=40,
        chunk_size=5,
        chunk_steps_per_env=8,
        rollout_epoch=2,
        rollout_chunk_count=128,
        update_epoch=4,
        actor_global_batch_size=16,
        actor_micro_batch_size=4,
        pipeline_stage_num=2,
        resource_pool_mode="mps",
    )


def test_estimate_candidate_uses_pipeline_bottleneck() -> None:
    estimate = estimate_candidate(
        candidate=CandidatePair(actor_sm=40, rollout_sm=60),
        summary=_summary(),
        throughput=StageThroughput(
            env_chunk_steps_per_sec=32.0,
            model_chunk_steps_per_sec=16.0,
            actor_chunk_steps_per_sec=64.0,
        ),
    )

    assert estimate.rollout_time_s == 8.0
    assert estimate.training_time_s == 2.0
    assert estimate.epoch_time_s == 8.0
    assert estimate.bottleneck_stage == "model"


def test_estimate_candidate_rejects_zero_throughput() -> None:
    try:
        estimate_candidate(
            candidate=CandidatePair(actor_sm=40, rollout_sm=60),
            summary=_summary(),
            throughput=StageThroughput(
                env_chunk_steps_per_sec=0.0,
                model_chunk_steps_per_sec=16.0,
                actor_chunk_steps_per_sec=64.0,
            ),
        )
    except ValueError as exc:
        assert "throughput" in str(exc)
    else:
        raise AssertionError("expected zero throughput to fail")
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
pytest tests/unit_tests/test_resource_orchestration_estimator.py -v
```

预期：FAIL，报错包含 `cannot import name 'estimate_candidate'`。

- [ ] **步骤 3：实现估算类型与函数**

在 `toolkits/resource_orchestration/types.py` 追加：

```python
@dataclass(frozen=True)
class StageThroughput:
    """Per-stage throughput normalized to chunk steps per second."""

    env_chunk_steps_per_sec: float
    model_chunk_steps_per_sec: float
    actor_chunk_steps_per_sec: float
    pipeline_samples_per_sec: float | None = None


@dataclass(frozen=True)
class CandidateEstimate:
    """Estimated rollout/training time for one candidate."""

    candidate: CandidatePair
    throughput: StageThroughput
    rollout_chunk_count: int
    rollout_time_s: float
    training_time_s: float
    epoch_time_s: float
    balance_gap_s: float
    bottleneck_stage: str
```

创建 `toolkits/resource_orchestration/estimator.py`：

```python
from __future__ import annotations

from toolkits.resource_orchestration.types import (
    CandidateEstimate,
    CandidatePair,
    ConfigSummary,
    StageThroughput,
)


def _require_positive(value: float, name: str) -> float:
    if value <= 0:
        raise ValueError(f"{name} throughput must be positive, got {value}")
    return float(value)


def estimate_candidate(
    *,
    candidate: CandidatePair,
    summary: ConfigSummary,
    throughput: StageThroughput,
) -> CandidateEstimate:
    """Estimate rollout and training time for one profile candidate."""
    env_tput = _require_positive(
        throughput.env_chunk_steps_per_sec, "environment chunk-step"
    )
    model_tput = _require_positive(
        throughput.model_chunk_steps_per_sec, "model chunk-step"
    )
    actor_tput = _require_positive(
        throughput.actor_chunk_steps_per_sec, "actor chunk-step"
    )

    rollout_bottleneck_tput = min(env_tput, model_tput)
    bottleneck_stage = "env" if env_tput <= model_tput else "model"
    rollout_time_s = summary.rollout_chunk_count / rollout_bottleneck_tput
    training_time_s = summary.rollout_chunk_count / actor_tput
    epoch_time_s = max(rollout_time_s, training_time_s)

    return CandidateEstimate(
        candidate=candidate,
        throughput=throughput,
        rollout_chunk_count=summary.rollout_chunk_count,
        rollout_time_s=rollout_time_s,
        training_time_s=training_time_s,
        epoch_time_s=epoch_time_s,
        balance_gap_s=abs(rollout_time_s - training_time_s),
        bottleneck_stage=bottleneck_stage,
    )
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
pytest tests/unit_tests/test_resource_orchestration_estimator.py -v
```

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add toolkits/resource_orchestration/types.py toolkits/resource_orchestration/estimator.py tests/unit_tests/test_resource_orchestration_estimator.py
git commit -s -m "feat: estimate resource orchestration timing"
```

## 任务 4：候选选择器

**文件：**
- 修改：`toolkits/resource_orchestration/types.py`
- 创建：`toolkits/resource_orchestration/selector.py`
- 测试：`tests/unit_tests/test_resource_orchestration_selector.py`

- [ ] **步骤 1：编写失败的选择器测试**

创建 `tests/unit_tests/test_resource_orchestration_selector.py`：

```python
from toolkits.resource_orchestration.selector import select_best_candidate
from toolkits.resource_orchestration.types import (
    CandidateEstimate,
    CandidatePair,
    StageThroughput,
)


def _estimate(actor_sm: int, rollout_sm: int, rollout: float, training: float) -> CandidateEstimate:
    return CandidateEstimate(
        candidate=CandidatePair(actor_sm=actor_sm, rollout_sm=rollout_sm),
        throughput=StageThroughput(1.0, 1.0, 1.0),
        rollout_chunk_count=100,
        rollout_time_s=rollout,
        training_time_s=training,
        epoch_time_s=max(rollout, training),
        balance_gap_s=abs(rollout - training),
        bottleneck_stage="env",
    )


def test_select_best_candidate_minimizes_epoch_time() -> None:
    result = select_best_candidate(
        [
            _estimate(30, 70, rollout=10.0, training=4.0),
            _estimate(40, 60, rollout=8.0, training=7.0),
        ]
    )

    assert result.selected.candidate == CandidatePair(actor_sm=40, rollout_sm=60)


def test_select_best_candidate_uses_balance_within_tolerance() -> None:
    result = select_best_candidate(
        [
            _estimate(30, 70, rollout=10.0, training=1.0),
            _estimate(40, 60, rollout=9.9, training=9.0),
        ],
        tolerance=0.03,
    )

    assert result.selected.candidate == CandidatePair(actor_sm=40, rollout_sm=60)


def test_select_best_candidate_prefers_higher_actor_sm_after_ties() -> None:
    result = select_best_candidate(
        [
            _estimate(40, 60, rollout=10.0, training=8.0),
            _estimate(50, 50, rollout=10.0, training=8.0),
        ]
    )

    assert result.selected.candidate == CandidatePair(actor_sm=50, rollout_sm=50)
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
pytest tests/unit_tests/test_resource_orchestration_selector.py -v
```

预期：FAIL，报错包含 `cannot import name 'select_best_candidate'`。

- [ ] **步骤 3：实现选择器**

在 `toolkits/resource_orchestration/types.py` 追加：

```python
@dataclass(frozen=True)
class SelectionResult:
    """Best candidate and ranked valid estimates."""

    selected: CandidateEstimate
    ranked: tuple[CandidateEstimate, ...]
```

创建 `toolkits/resource_orchestration/selector.py`：

```python
from __future__ import annotations

from toolkits.resource_orchestration.types import CandidateEstimate, SelectionResult


def select_best_candidate(
    estimates: list[CandidateEstimate] | tuple[CandidateEstimate, ...],
    *,
    tolerance: float = 0.03,
) -> SelectionResult:
    """Select the best candidate by epoch time, balance, then actor SM."""
    if not estimates:
        raise ValueError("at least one valid candidate estimate is required")
    if tolerance < 0:
        raise ValueError("selection tolerance must be non-negative")

    best_epoch = min(item.epoch_time_s for item in estimates)
    threshold = best_epoch * (1.0 + tolerance)
    near_best = [item for item in estimates if item.epoch_time_s <= threshold]
    ranked = tuple(
        sorted(
            near_best,
            key=lambda item: (
                item.balance_gap_s,
                item.epoch_time_s,
                -item.candidate.actor_sm,
            ),
        )
    )
    selected = ranked[0]

    full_ranked = tuple(
        sorted(
            estimates,
            key=lambda item: (
                0 if item is selected else 1,
                item.epoch_time_s,
                item.balance_gap_s,
                -item.candidate.actor_sm,
            ),
        )
    )
    return SelectionResult(selected=selected, ranked=full_ranked)
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
pytest tests/unit_tests/test_resource_orchestration_selector.py -v
```

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add toolkits/resource_orchestration/types.py toolkits/resource_orchestration/selector.py tests/unit_tests/test_resource_orchestration_selector.py
git commit -s -m "feat: select resource orchestration candidate"
```

## 任务 5：Plan JSON 写入

**文件：**
- 创建：`toolkits/resource_orchestration/plan_writer.py`
- 测试：`tests/unit_tests/test_resource_orchestration_plan_writer.py`

- [ ] **步骤 1：编写失败的 plan writer 测试**

创建 `tests/unit_tests/test_resource_orchestration_plan_writer.py`：

```python
import json

from rlinf.scheduler.resource_pool.bindings import (
    CpuBinding,
    GpuBinding,
    WorkerResourceBinding,
)
from toolkits.resource_orchestration.plan_writer import write_mps_plan
from toolkits.resource_orchestration.types import CandidatePair


def test_write_mps_plan_updates_actor_and_rollout_gpu_sm(tmp_path) -> None:
    bindings = {
        "actor": [
            WorkerResourceBinding(
                component="actor",
                rank=0,
                cluster_node_rank=0,
                node_group_label="cluster",
                cpu=CpuBinding(process_cpu_cores=(0, 1)),
                gpu=GpuBinding(mode="mps", sm_percent=50, visible_devices=("0",), parent_gpu=0),
            )
        ],
        "rollout": [
            WorkerResourceBinding(
                component="rollout",
                rank=0,
                cluster_node_rank=0,
                node_group_label="cluster",
                cpu=CpuBinding(process_cpu_cores=(2, 3)),
                gpu=GpuBinding(mode="mps", sm_percent=50, visible_devices=("0",), parent_gpu=0),
            )
        ],
        "env": [
            WorkerResourceBinding(
                component="env",
                rank=0,
                cluster_node_rank=0,
                node_group_label="node",
                cpu=CpuBinding(process_cpu_cores=(4, 5)),
                gpu=None,
            )
        ],
    }

    output = tmp_path / "plan.json"
    write_mps_plan(
        output_path=output,
        base_bindings=bindings,
        candidate=CandidatePair(actor_sm=30, rollout_sm=70),
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    parsed = [WorkerResourceBinding.from_json(json.dumps(item)) for item in payload["bindings"]]

    actor = next(item for item in parsed if item.component == "actor")
    rollout = next(item for item in parsed if item.component == "rollout")
    env = next(item for item in parsed if item.component == "env")
    assert actor.gpu.sm_percent == 30
    assert rollout.gpu.sm_percent == 70
    assert env.gpu is None
    assert actor.cpu.process_cpu_cores == (0, 1)
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
pytest tests/unit_tests/test_resource_orchestration_plan_writer.py -v
```

预期：FAIL，报错包含 `cannot import name 'write_mps_plan'`。

- [ ] **步骤 3：实现 plan writer**

创建 `toolkits/resource_orchestration/plan_writer.py`：

```python
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from rlinf.scheduler.resource_pool.bindings import (
    GpuBinding,
    WorkerResourceBinding,
)

from toolkits.resource_orchestration.types import CandidatePair


def _with_sm(binding: WorkerResourceBinding, sm_percent: int) -> WorkerResourceBinding:
    gpu = binding.gpu
    if gpu is None:
        return binding
    if gpu.mode != "mps":
        raise ValueError(f"expected MPS GPU binding for {binding.component}:{binding.rank}")
    return WorkerResourceBinding(
        component=binding.component,
        rank=binding.rank,
        cluster_node_rank=binding.cluster_node_rank,
        node_group_label=binding.node_group_label,
        cpu=binding.cpu,
        gpu=GpuBinding(
            mode="mps",
            sm_percent=sm_percent,
            visible_devices=gpu.visible_devices,
            mig_device_uuid=None,
            parent_gpu=gpu.parent_gpu,
        ),
    )


def build_mps_plan_payload(
    *,
    base_bindings: dict[str, list[WorkerResourceBinding]],
    candidate: CandidatePair,
) -> dict[str, list[dict]]:
    """Build a JSON-compatible plan payload for a selected candidate."""
    output_bindings: list[WorkerResourceBinding] = []
    for component in sorted(base_bindings):
        for binding in sorted(base_bindings[component], key=lambda item: item.rank):
            if component == "actor":
                output_bindings.append(_with_sm(binding, candidate.actor_sm))
            elif component == "rollout":
                output_bindings.append(_with_sm(binding, candidate.rollout_sm))
            else:
                output_bindings.append(binding)

    payload = {"bindings": [asdict(binding) for binding in output_bindings]}
    for item in payload["bindings"]:
        WorkerResourceBinding.from_json(json.dumps(item))
    return payload


def write_mps_plan(
    *,
    output_path: str | Path,
    base_bindings: dict[str, list[WorkerResourceBinding]],
    candidate: CandidatePair,
) -> None:
    """Write a ResourcePoolSolver-compatible MPS plan JSON."""
    payload = build_mps_plan_payload(base_bindings=base_bindings, candidate=candidate)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
pytest tests/unit_tests/test_resource_orchestration_plan_writer.py -v
```

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add toolkits/resource_orchestration/plan_writer.py tests/unit_tests/test_resource_orchestration_plan_writer.py
git commit -s -m "feat: write resource orchestration plans"
```

## 任务 6：报告输出

**文件：**
- 创建：`toolkits/resource_orchestration/reporting.py`
- 测试：`tests/unit_tests/test_resource_orchestration_reporting.py`

- [ ] **步骤 1：编写失败的 reporting 测试**

创建 `tests/unit_tests/test_resource_orchestration_reporting.py`：

```python
import json

from toolkits.resource_orchestration.reporting import write_reports
from toolkits.resource_orchestration.types import (
    CandidateEstimate,
    CandidatePair,
    SelectionResult,
    StageThroughput,
)


def _estimate() -> CandidateEstimate:
    return CandidateEstimate(
        candidate=CandidatePair(actor_sm=40, rollout_sm=60),
        throughput=StageThroughput(
            env_chunk_steps_per_sec=20.0,
            model_chunk_steps_per_sec=10.0,
            actor_chunk_steps_per_sec=30.0,
            pipeline_samples_per_sec=9.0,
        ),
        rollout_chunk_count=100,
        rollout_time_s=10.0,
        training_time_s=3.3333333333,
        epoch_time_s=10.0,
        balance_gap_s=6.6666666667,
        bottleneck_stage="model",
    )


def test_write_reports_outputs_summary_and_profile_json(tmp_path) -> None:
    estimate = _estimate()
    write_reports(
        output_dir=tmp_path,
        estimates=[estimate],
        selection=SelectionResult(selected=estimate, ranked=(estimate,)),
        plan_output="plan.json",
        failed_profiles=[],
    )

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    profile = json.loads(
        (tmp_path / "profiles" / "actor40_rollout60.json").read_text(encoding="utf-8")
    )

    assert summary["selected"]["candidate_id"] == "actor40_rollout60"
    assert summary["plan_output"] == "plan.json"
    assert profile["throughput"]["model_chunk_steps_per_sec"] == 10.0
    assert "actor40_rollout60" in (tmp_path / "summary.md").read_text(encoding="utf-8")
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
pytest tests/unit_tests/test_resource_orchestration_reporting.py -v
```

预期：FAIL，报错包含 `cannot import name 'write_reports'`。

- [ ] **步骤 3：实现 reporting**

创建 `toolkits/resource_orchestration/reporting.py`：

```python
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from toolkits.resource_orchestration.types import CandidateEstimate, SelectionResult


def _estimate_record(estimate: CandidateEstimate) -> dict[str, Any]:
    payload = asdict(estimate)
    payload["candidate_id"] = estimate.candidate.candidate_id
    return payload


def write_reports(
    *,
    output_dir: str | Path,
    estimates: list[CandidateEstimate],
    selection: SelectionResult,
    plan_output: str,
    failed_profiles: list[dict[str, Any]],
) -> dict[str, Any]:
    """Write profile and summary reports."""
    root = Path(output_dir)
    profiles_dir = root / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    for estimate in estimates:
        record = _estimate_record(estimate)
        (profiles_dir / f"{estimate.candidate.candidate_id}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    summary = {
        "selected": _estimate_record(selection.selected),
        "ranked_candidate_ids": [
            estimate.candidate.candidate_id for estimate in selection.ranked
        ],
        "plan_output": plan_output,
        "failed_profiles": failed_profiles,
        "candidates": [_estimate_record(estimate) for estimate in estimates],
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    lines = [
        "# Profile-Based Resource Orchestration Summary",
        "",
        f"- Selected: {selection.selected.candidate.candidate_id}",
        f"- Plan output: {plan_output}",
        "",
        "| candidate | epoch_s | rollout_s | training_s | bottleneck | env_chunk/s | model_chunk/s | actor_chunk/s |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for estimate in selection.ranked:
        lines.append(
            "| {candidate} | {epoch:.6f} | {rollout:.6f} | {training:.6f} | {bottleneck} | {env:.6f} | {model:.6f} | {actor:.6f} |".format(
                candidate=estimate.candidate.candidate_id,
                epoch=estimate.epoch_time_s,
                rollout=estimate.rollout_time_s,
                training=estimate.training_time_s,
                bottleneck=estimate.bottleneck_stage,
                env=estimate.throughput.env_chunk_steps_per_sec,
                model=estimate.throughput.model_chunk_steps_per_sec,
                actor=estimate.throughput.actor_chunk_steps_per_sec,
            )
        )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
pytest tests/unit_tests/test_resource_orchestration_reporting.py -v
```

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add toolkits/resource_orchestration/reporting.py tests/unit_tests/test_resource_orchestration_reporting.py
git commit -s -m "feat: report resource orchestration results"
```

## 任务 7：Profile Adapter 与 Stub-Orchestrator

**文件：**
- 修改：`toolkits/resource_orchestration/types.py`
- 创建：`toolkits/resource_orchestration/profilers.py`
- 创建：`toolkits/resource_orchestration/orchestrator.py`
- 测试：`tests/unit_tests/test_resource_orchestration_run.py`

- [ ] **步骤 1：编写失败的 orchestrator stub 测试**

创建 `tests/unit_tests/test_resource_orchestration_run.py`：

```python
from rlinf.scheduler.resource_pool.bindings import GpuBinding, WorkerResourceBinding

from toolkits.resource_orchestration.orchestrator import run_orchestration
from toolkits.resource_orchestration.types import (
    CandidatePair,
    ConfigSummary,
    StageThroughput,
)


class StubProfiler:
    def profile(self, candidate: CandidatePair) -> StageThroughput:
        if candidate.actor_sm == 30:
            return StageThroughput(10.0, 10.0, 20.0)
        return StageThroughput(10.0, 10.0, 5.0)


def test_run_orchestration_with_stub_profiler_writes_plan_and_summary(tmp_path) -> None:
    bindings = {
        "actor": [
            WorkerResourceBinding(
                component="actor",
                rank=0,
                cluster_node_rank=0,
                node_group_label="cluster",
                gpu=GpuBinding(mode="mps", sm_percent=50, visible_devices=("0",), parent_gpu=0),
            )
        ],
        "rollout": [
            WorkerResourceBinding(
                component="rollout",
                rank=0,
                cluster_node_rank=0,
                node_group_label="cluster",
                gpu=GpuBinding(mode="mps", sm_percent=50, visible_devices=("0",), parent_gpu=0),
            )
        ],
    }
    summary = ConfigSummary(
        total_num_envs=10,
        episode_env_steps=10,
        chunk_size=5,
        chunk_steps_per_env=2,
        rollout_epoch=1,
        rollout_chunk_count=20,
        update_epoch=1,
        actor_global_batch_size=10,
        actor_micro_batch_size=5,
        pipeline_stage_num=1,
        resource_pool_mode="mps",
    )

    result = run_orchestration(
        config_summary=summary,
        base_bindings=bindings,
        candidates=(
            CandidatePair(actor_sm=30, rollout_sm=70),
            CandidatePair(actor_sm=70, rollout_sm=30),
        ),
        profiler=StubProfiler(),
        output_dir=tmp_path,
        plan_output=tmp_path / "plan.json",
    )

    assert result.selected.candidate == CandidatePair(actor_sm=30, rollout_sm=70)
    assert (tmp_path / "plan.json").exists()
    assert (tmp_path / "summary.json").exists()
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
pytest tests/unit_tests/test_resource_orchestration_run.py -v
```

预期：FAIL，报错包含 `cannot import name 'run_orchestration'`。

- [ ] **步骤 3：实现 Profile 协议和 orchestrator**

创建 `toolkits/resource_orchestration/profilers.py`：

```python
from __future__ import annotations

from typing import Protocol

from toolkits.resource_orchestration.types import CandidatePair, StageThroughput


class ThroughputProfiler(Protocol):
    """Profiles one candidate and returns normalized stage throughput."""

    def profile(self, candidate: CandidatePair) -> StageThroughput:
        """Profile a candidate MPS allocation."""
```

创建 `toolkits/resource_orchestration/orchestrator.py`：

```python
from __future__ import annotations

from pathlib import Path

from rlinf.scheduler.resource_pool.bindings import WorkerResourceBinding

from toolkits.resource_orchestration.estimator import estimate_candidate
from toolkits.resource_orchestration.plan_writer import write_mps_plan
from toolkits.resource_orchestration.profilers import ThroughputProfiler
from toolkits.resource_orchestration.reporting import write_reports
from toolkits.resource_orchestration.selector import select_best_candidate
from toolkits.resource_orchestration.types import (
    CandidateEstimate,
    CandidatePair,
    ConfigSummary,
    SelectionResult,
)


def run_orchestration(
    *,
    config_summary: ConfigSummary,
    base_bindings: dict[str, list[WorkerResourceBinding]],
    candidates: tuple[CandidatePair, ...],
    profiler: ThroughputProfiler,
    output_dir: str | Path,
    plan_output: str | Path,
    selection_tolerance: float = 0.03,
) -> SelectionResult:
    """Profile candidates, select the best one, write reports and plan."""
    estimates: list[CandidateEstimate] = []
    failed_profiles: list[dict[str, str]] = []
    for candidate in candidates:
        try:
            throughput = profiler.profile(candidate)
            estimates.append(
                estimate_candidate(
                    candidate=candidate,
                    summary=config_summary,
                    throughput=throughput,
                )
            )
        except Exception as exc:  # noqa: BLE001
            failed_profiles.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    if not estimates:
        write_reports(
            output_dir=output_dir,
            estimates=[],
            selection=None,  # type: ignore[arg-type]
            plan_output=str(plan_output),
            failed_profiles=failed_profiles,
        )
        raise RuntimeError("all resource orchestration candidates failed")

    selection = select_best_candidate(estimates, tolerance=selection_tolerance)
    write_mps_plan(
        output_path=plan_output,
        base_bindings=base_bindings,
        candidate=selection.selected.candidate,
    )
    write_reports(
        output_dir=output_dir,
        estimates=estimates,
        selection=selection,
        plan_output=str(plan_output),
        failed_profiles=failed_profiles,
    )
    return selection
```

同时修改 `toolkits/resource_orchestration/reporting.py`，让失败全空场景可写 summary。将签名改为：

```python
def write_reports(
    *,
    output_dir: str | Path,
    estimates: list[CandidateEstimate],
    selection: SelectionResult | None,
    plan_output: str,
    failed_profiles: list[dict[str, Any]],
) -> dict[str, Any]:
```

并在函数开头生成：

```python
selected_record = _estimate_record(selection.selected) if selection is not None else None
ranked = selection.ranked if selection is not None else tuple(estimates)
```

将所有 `selection.selected` 使用改成 `selected_record` 或 guarded 文本；Markdown selected 行在 `selection is None` 时写 `- Selected: none`。

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
pytest tests/unit_tests/test_resource_orchestration_run.py tests/unit_tests/test_resource_orchestration_reporting.py -v
```

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add toolkits/resource_orchestration/profilers.py toolkits/resource_orchestration/orchestrator.py toolkits/resource_orchestration/reporting.py tests/unit_tests/test_resource_orchestration_run.py
git commit -s -m "feat: orchestrate resource profile estimates"
```

## 任务 8：CLI 与 Base Bindings 解析

**文件：**
- 修改：`toolkits/resource_orchestration/config_loader.py`
- 创建：`toolkits/resource_orchestration/run.py`
- 测试：`tests/unit_tests/test_resource_orchestration_cli.py`

- [ ] **步骤 1：编写失败的 CLI 参数测试**

创建 `tests/unit_tests/test_resource_orchestration_cli.py`：

```python
from toolkits.resource_orchestration.run import parse_args


def test_parse_args_accepts_required_cli_options() -> None:
    args = parse_args(
        [
            "--config-path",
            "examples/embodiment/config",
            "--config-name",
            "libero",
            "--candidate-pairs",
            "30:70",
            "--output-dir",
            "out",
            "--plan-output",
            "plan.json",
            "--override",
            "runner.max_epochs=1",
        ]
    )

    assert args.config_path == "examples/embodiment/config"
    assert args.config_name == "libero"
    assert args.candidate_pairs == "30:70"
    assert args.override == ["runner.max_epochs=1"]
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
pytest tests/unit_tests/test_resource_orchestration_cli.py -v
```

预期：FAIL，报错包含 `No module named 'toolkits.resource_orchestration.run'`。

- [ ] **步骤 3：实现 CLI parse 和 plan file binding 读取**

在 `toolkits/resource_orchestration/config_loader.py` 追加：

```python
import json
from pathlib import Path

from rlinf.scheduler.resource_pool.bindings import WorkerResourceBinding


def load_plan_bindings(plan_path: str | Path) -> dict[str, list[WorkerResourceBinding]]:
    """Load resource bindings from an existing plan JSON."""
    payload = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    bindings: dict[str, list[WorkerResourceBinding]] = {}
    for item in payload.get("bindings", []):
        binding = WorkerResourceBinding.from_json(json.dumps(item))
        bindings.setdefault(binding.component, []).append(binding)
    return {
        component: sorted(items, key=lambda binding: binding.rank)
        for component, items in bindings.items()
    }


def load_base_bindings(
    *,
    cfg: DictConfig,
    base_plan: str | Path | None,
) -> dict[str, list[WorkerResourceBinding]]:
    """Load base bindings from --base-plan or cfg allocation_plan_path."""
    plan_path = base_plan or OmegaConf.select(
        cfg,
        "cluster.resource_pool.allocation_plan_path",
    )
    if not plan_path:
        raise ValueError(
            "resource orchestration v1 requires --base-plan or "
            "cluster.resource_pool.allocation_plan_path"
        )
    return load_plan_bindings(plan_path)
```

创建 `toolkits/resource_orchestration/run.py`：

```python
from __future__ import annotations

import argparse

from toolkits.resource_orchestration.candidates import parse_candidate_pairs
from toolkits.resource_orchestration.config_loader import (
    build_config_summary,
    load_base_bindings,
    load_hydra_config,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse resource orchestration CLI arguments."""
    parser = argparse.ArgumentParser(description="Profile-based resource orchestration")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--candidate-pairs", default=None)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=20)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--plan-output", required=True)
    parser.add_argument("--base-plan", default=None)
    parser.add_argument("--selection-tolerance", type=float, default=0.03)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for argument validation."""
    args = parse_args(argv)
    cfg = load_hydra_config(
        config_path=args.config_path,
        config_name=args.config_name,
        overrides=tuple(args.override),
    )
    build_config_summary(cfg)
    parse_candidate_pairs(args.candidate_pairs)
    load_base_bindings(cfg=cfg, base_plan=args.base_plan)


if __name__ == "__main__":
    main()
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
pytest tests/unit_tests/test_resource_orchestration_cli.py -v
```

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add toolkits/resource_orchestration/config_loader.py toolkits/resource_orchestration/run.py tests/unit_tests/test_resource_orchestration_cli.py
git commit -s -m "feat: add resource orchestration cli"
```

## 任务 9：真实 Profiler Adapter 接入

**文件：**
- 修改：`toolkits/resource_orchestration/profilers.py`
- 修改：`toolkits/resource_orchestration/run.py`
- 测试：`tests/unit_tests/test_resource_orchestration_profilers.py`

- [ ] **步骤 1：编写失败的 profiler metric 转换和调用测试**

创建 `tests/unit_tests/test_resource_orchestration_profilers.py`：

```python
from unittest.mock import Mock

from omegaconf import OmegaConf

from toolkits.resource_orchestration.profilers import (
    ToolkitProfileFunctions,
    ToolkitThroughputProfiler,
    combine_profile_metrics,
)
from toolkits.resource_orchestration.types import CandidatePair, ConfigSummary


def test_combine_profile_metrics_converts_env_steps_to_chunk_steps() -> None:
    got = combine_profile_metrics(
        env_steps_per_sec=50.0,
        model_infers_per_sec=11.0,
        actor_chunk_steps_per_sec=7.0,
        chunk_size=5,
        pipeline_samples_per_sec=9.0,
    )

    assert got.env_chunk_steps_per_sec == 10.0
    assert got.model_chunk_steps_per_sec == 11.0
    assert got.actor_chunk_steps_per_sec == 7.0
    assert got.pipeline_samples_per_sec == 9.0


def test_toolkit_profiler_calls_rollout_and_training_functions() -> None:
    cfg = OmegaConf.create({})
    summary = ConfigSummary(
        total_num_envs=8,
        episode_env_steps=40,
        chunk_size=5,
        chunk_steps_per_env=8,
        rollout_epoch=1,
        rollout_chunk_count=64,
        update_epoch=4,
        actor_global_batch_size=16,
        actor_micro_batch_size=4,
        pipeline_stage_num=1,
        resource_pool_mode="mps",
    )
    rollout_profile = Mock(
        return_value={
            "env_steps_per_sec": 50.0,
            "model_infers_per_sec": 11.0,
            "pipeline_samples_per_sec": 9.0,
        }
    )
    training_profile = Mock(return_value={"actor_chunk_steps_per_sec": 7.0})

    profiler = ToolkitThroughputProfiler(
        cfg=cfg,
        summary=summary,
        warmup_steps=2,
        measure_steps=3,
        functions=ToolkitProfileFunctions(
            rollout_profile=rollout_profile,
            training_profile=training_profile,
        ),
    )

    got = profiler.profile(CandidatePair(actor_sm=30, rollout_sm=70))

    assert got.env_chunk_steps_per_sec == 10.0
    assert got.model_chunk_steps_per_sec == 11.0
    assert got.actor_chunk_steps_per_sec == 7.0
    rollout_profile.assert_called_once()
    training_profile.assert_called_once()
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
pytest tests/unit_tests/test_resource_orchestration_profilers.py -v
```

预期：FAIL，报错包含 `cannot import name 'combine_profile_metrics'` 或 `cannot import name 'ToolkitThroughputProfiler'`。

- [ ] **步骤 3：实现 metric 转换和真实 profiler adapter**

修改 `toolkits/resource_orchestration/profilers.py`：

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from omegaconf import DictConfig

from toolkits.resource_orchestration.types import (
    CandidatePair,
    ConfigSummary,
    StageThroughput,
)


class ThroughputProfiler(Protocol):
    """Profiles one candidate and returns normalized stage throughput."""

    def profile(self, candidate: CandidatePair) -> StageThroughput:
        """Profile a candidate MPS allocation."""


RolloutProfileFn = Callable[
    [DictConfig, CandidatePair, int, int],
    dict[str, float],
]
TrainingProfileFn = Callable[
    [DictConfig, CandidatePair, ConfigSummary, int, int],
    dict[str, float],
]


@dataclass(frozen=True)
class ToolkitProfileFunctions:
    """Callable hooks used by ToolkitThroughputProfiler."""

    rollout_profile: RolloutProfileFn
    training_profile: TrainingProfileFn


def combine_profile_metrics(
    *,
    env_steps_per_sec: float,
    model_infers_per_sec: float,
    actor_chunk_steps_per_sec: float,
    chunk_size: int,
    pipeline_samples_per_sec: float | None = None,
) -> StageThroughput:
    """Normalize raw toolkit metrics to chunk-step throughput."""
    return StageThroughput(
        env_chunk_steps_per_sec=float(env_steps_per_sec) / float(chunk_size),
        model_chunk_steps_per_sec=float(model_infers_per_sec),
        actor_chunk_steps_per_sec=float(actor_chunk_steps_per_sec),
        pipeline_samples_per_sec=(
            None if pipeline_samples_per_sec is None else float(pipeline_samples_per_sec)
        ),
    )


@dataclass
class ToolkitThroughputProfiler:
    """Adapter for rollout_eval and training_eval profiling."""

    cfg: DictConfig
    summary: ConfigSummary
    warmup_steps: int
    measure_steps: int
    functions: ToolkitProfileFunctions

    def profile(self, candidate: CandidatePair) -> StageThroughput:
        """Run real toolkit profilers for one candidate."""
        rollout_metrics = self.functions.rollout_profile(
            self.cfg,
            candidate,
            self.warmup_steps,
            self.measure_steps,
        )
        training_metrics = self.functions.training_profile(
            self.cfg,
            candidate,
            self.summary,
            self.warmup_steps,
            self.measure_steps,
        )
        return combine_profile_metrics(
            env_steps_per_sec=float(rollout_metrics["env_steps_per_sec"]),
            model_infers_per_sec=float(rollout_metrics["model_infers_per_sec"]),
            actor_chunk_steps_per_sec=float(
                training_metrics["actor_chunk_steps_per_sec"]
            ),
            chunk_size=self.summary.chunk_size,
            pipeline_samples_per_sec=rollout_metrics.get("pipeline_samples_per_sec"),
        )
```

- [ ] **步骤 4：实现默认 profile 函数加载**

在 `toolkits/resource_orchestration/profilers.py` 追加：

```python
def default_rollout_profile(
    cfg: DictConfig,
    candidate: CandidatePair,
    warmup_steps: int,
    measure_steps: int,
) -> dict[str, float]:
    """Run rollout_eval benchmark for env/model throughput."""
    import os
    from contextlib import contextmanager

    from toolkits.rollout_eval.adapters import build_env_adapter, build_model_adapter
    from toolkits.rollout_eval.benchmark.resource_binding import build_process_env
    from toolkits.rollout_eval.benchmark.single_runner import (
        run_env_only_case,
        run_model_only_case,
    )

    @contextmanager
    def temporary_environ(updates: dict[str, str]):
        previous = {key: os.environ.get(key) for key in updates}
        try:
            for key, value in updates.items():
                os.environ[key] = value
            yield
        finally:
            for key, old_value in previous.items():
                if old_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_value

    merged_env = build_process_env(
        base_env=os.environ,
        mps_active_thread_percentage=candidate.rollout_sm,
    )
    updates = {
        key: value
        for key, value in merged_env.items()
        if os.environ.get(key) != value
    }

    with temporary_environ(updates):
        env_adapter = build_env_adapter(cfg, split="eval", profile_output_dir=None)
        action_dim_override = int(cfg.actor.model.get("action_dim", 0) or 0)
        env_result = run_env_only_case(
            env_adapter=env_adapter,
            warmup_steps=warmup_steps,
            measure_steps=measure_steps,
            action_dim_override=action_dim_override,
        )

        model_env_adapter = build_env_adapter(
            cfg,
            split="eval",
            profile_output_dir=None,
        )
        reset_obs, _ = model_env_adapter.reset()
        model_adapter = build_model_adapter(cfg, split_model_stages=False)
        model_result = run_model_only_case(
            env_adapter=model_env_adapter,
            model_adapter=model_adapter,
            warmup_steps=warmup_steps,
            measure_steps=measure_steps,
            obs_batch=reset_obs,
        )

    return {
        "env_steps_per_sec": env_result.metrics.env_steps_per_sec,
        "model_infers_per_sec": model_result.metrics.model_infers_per_sec,
        "pipeline_samples_per_sec": model_result.metrics.pipeline_samples_per_sec,
    }


def default_training_profile(
    cfg: DictConfig,
    candidate: CandidatePair,
    summary: ConfigSummary,
    warmup_steps: int,
    measure_steps: int,
) -> dict[str, float]:
    """Run training_eval benchmark for actor throughput."""
    try:
        from toolkits.training_eval.run import run_training_profile
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "toolkits.training_eval is required for actor training profiling"
        ) from exc

    return run_training_profile(
        cfg=cfg,
        actor_sm=candidate.actor_sm,
        warmup_steps=warmup_steps,
        measure_steps=measure_steps,
        rollout_chunk_count=summary.rollout_chunk_count,
    )


def default_profile_functions() -> ToolkitProfileFunctions:
    """Return default profile hooks used by the CLI."""
    return ToolkitProfileFunctions(
        rollout_profile=default_rollout_profile,
        training_profile=default_training_profile,
    )
```

Add a unit test that patches `build_env_adapter`, `build_model_adapter`,
`run_env_only_case`, and `run_model_only_case` in
`toolkits.resource_orchestration.profilers`. The test should assert that
`default_rollout_profile()` returns the three numeric keys and passes
`candidate.rollout_sm` into `build_process_env`.

- [ ] **步骤 5：运行测试验证通过**

运行：

```bash
pytest tests/unit_tests/test_resource_orchestration_profilers.py -v
```

预期：PASS。

- [ ] **步骤 6：Commit**

```bash
git add toolkits/resource_orchestration/profilers.py tests/unit_tests/test_resource_orchestration_profilers.py
git commit -s -m "feat: adapt resource orchestration profilers"
```

## 任务 10：CLI 串起真实 Orchestrator

**文件：**
- 修改：`toolkits/resource_orchestration/run.py`
- 测试：`tests/unit_tests/test_resource_orchestration_cli.py`

- [ ] **步骤 1：编写失败的 CLI main 调用测试**

在 `tests/unit_tests/test_resource_orchestration_cli.py` 追加：

```python
from unittest.mock import Mock, patch

from omegaconf import OmegaConf


def test_main_wires_orchestrator_with_default_profiler(tmp_path) -> None:
    from toolkits.resource_orchestration import run

    cfg = OmegaConf.create(
        {
            "cluster": {
                "resource_pool": {
                    "allocation_plan_path": str(tmp_path / "base.json"),
                    "gpu": {"mode": "mps"},
                }
            },
            "env": {"train": {"total_num_envs": 1, "max_steps_per_rollout_epoch": 5}},
            "actor": {
                "global_batch_size": 1,
                "micro_batch_size": 1,
                "model": {"num_action_chunks": 5},
            },
            "algorithm": {"rollout_epoch": 1, "update_epoch": 1},
            "rollout": {"pipeline_stage_num": 1},
        }
    )

    with (
        patch.object(run, "load_hydra_config", return_value=cfg),
        patch.object(run, "load_base_bindings", return_value={"actor": [], "rollout": []}),
        patch.object(run, "ToolkitThroughputProfiler") as profiler_cls,
        patch.object(run, "default_profile_functions", return_value=Mock()),
        patch.object(run, "run_orchestration") as run_orchestration,
    ):
        run.main(
            [
                "--config-path",
                "examples/embodiment/config",
                "--config-name",
                "libero",
                "--candidate-pairs",
                "30:70",
                "--output-dir",
                str(tmp_path),
                "--plan-output",
                str(tmp_path / "plan.json"),
            ]
        )

    profiler_cls.assert_called_once()
    run_orchestration.assert_called_once()
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
pytest tests/unit_tests/test_resource_orchestration_cli.py::test_main_wires_orchestrator_with_default_profiler -v
```

预期：FAIL，`run_orchestration` 未被调用。

- [ ] **步骤 3：实现 CLI wiring**

修改 `toolkits/resource_orchestration/run.py` imports：

```python
from toolkits.resource_orchestration.orchestrator import run_orchestration
from toolkits.resource_orchestration.profilers import (
    ToolkitThroughputProfiler,
    default_profile_functions,
)
```

修改 `main()`：

```python
def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for profile-based resource orchestration."""
    args = parse_args(argv)
    cfg = load_hydra_config(
        config_path=args.config_path,
        config_name=args.config_name,
        overrides=tuple(args.override),
    )
    summary = build_config_summary(cfg)
    candidates = parse_candidate_pairs(args.candidate_pairs)
    base_bindings = load_base_bindings(cfg=cfg, base_plan=args.base_plan)
    profiler = ToolkitThroughputProfiler(
        cfg=cfg,
        summary=summary,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        functions=default_profile_functions(),
    )
    run_orchestration(
        config_summary=summary,
        base_bindings=base_bindings,
        candidates=candidates,
        profiler=profiler,
        output_dir=args.output_dir,
        plan_output=args.plan_output,
        selection_tolerance=args.selection_tolerance,
    )
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
pytest tests/unit_tests/test_resource_orchestration_cli.py -v
```

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add toolkits/resource_orchestration/run.py tests/unit_tests/test_resource_orchestration_cli.py
git commit -s -m "feat: wire resource orchestration cli"
```

## 任务 11：完整测试与文档验证

**文件：**
- 修改：按前面任务产生的实际文件
- 测试：所有 resource orchestration unit tests

- [ ] **步骤 1：运行 resource orchestration 单测**

运行：

```bash
pytest \
  tests/unit_tests/test_resource_orchestration_candidates.py \
  tests/unit_tests/test_resource_orchestration_config_loader.py \
  tests/unit_tests/test_resource_orchestration_estimator.py \
  tests/unit_tests/test_resource_orchestration_selector.py \
  tests/unit_tests/test_resource_orchestration_plan_writer.py \
  tests/unit_tests/test_resource_orchestration_reporting.py \
  tests/unit_tests/test_resource_orchestration_run.py \
  tests/unit_tests/test_resource_orchestration_cli.py \
  tests/unit_tests/test_resource_orchestration_profilers.py \
  -v
```

预期：全部 PASS。

- [ ] **步骤 2：运行相关既有 resource_pool 测试**

运行：

```bash
pytest \
  tests/unit_tests/test_resource_pool_gpu_binding.py \
  tests/unit_tests/test_resource_pool_bindings.py \
  tests/unit_tests/test_resource_pool_solver.py \
  -v
```

预期：全部 PASS。

- [ ] **步骤 3：运行 Ruff 格式检查**

运行：

```bash
ruff check toolkits/resource_orchestration tests/unit_tests/test_resource_orchestration_*.py
```

预期：PASS。

- [ ] **步骤 4：修复验证中发现的问题**

如果步骤 1-3 任一失败，按报错修改对应文件，然后重新运行失败命令。常见修复：

```python
# 未使用 import：删除 import
# 行过长：拆成多行
# 类型名不一致：以 toolkits/resource_orchestration/types.py 为准
```

- [ ] **步骤 5：最终 Commit**

如果步骤 4 有修改：

```bash
git add toolkits/resource_orchestration tests/unit_tests/test_resource_orchestration_*.py
git commit -s -m "test: verify resource orchestration toolkit"
```

如果步骤 4 没有修改，不创建空提交。

## 实施注意事项

- 不要修改用户已有的未提交工作，尤其是当前 worktree 里的 embodiment config、worker、runner 和 resource_pool 改动。
- 手动编辑文件使用 `apply_patch`。
- 每个任务完成后单独 commit。
- `ToolkitThroughputProfiler.profile()` 不能保留未实现路径；CLI 默认路径必须能调用真实 profile hooks，或者在缺少 `toolkits.training_eval` 时明确失败并不写 plan。
- 真实 GPU-heavy 逻辑只能放在 `profilers.py` 或被它调用的小 helper 中，不要塞进 estimator/selector/reporting。

## 自检清单

- 规格中的架构、数据流、公式、选择规则、plan 输出、CLI、报告、错误处理、测试均有对应任务。
- 计划没有要求自动修改 YAML。
- 计划没有把 scheduler runtime 变成 profile 流程。
- 计划优先用 stub profiler 保障 orchestrator 可测试。
- Plan writer 使用现有 `WorkerResourceBinding` schema 验证输出。
