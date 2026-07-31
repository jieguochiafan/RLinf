# RLinf Rollout

Standalone rollout system for embodied and LLM policies, extracted from
[RLinf](https://github.com/RLinf/RLinf). It contains rollout and environment
execution only — no training loop — so it can be driven by any training
framework.

Status: **Phase 1** — frozen `v1` protocol plus the vendored infrastructure
(Ray scheduler, utils, io structs, weight sync). Envs, models and workers arrive
in Phase 2–3 (see `../ROLLOUT_SPLIT_PLAN.md`).

## Install

```bash
pip install -e rlinf_rollout            # core (Ray + torch)
pip install -e "rlinf_rollout[embodied]" # embodied policies and envs
pip install -e "rlinf_rollout[sglang]"   # LLM rollout on SGLang
pip install -e "rlinf_rollout[vllm]"     # LLM rollout on vLLM
```

The package never imports the training-side `rlinf` package; everything it
needs is vendored under `rlinf_rollout/`.

## Layout

| Directory | Contents | Source |
|---|---|---|
| `api/v1/` | frozen protocol (weights in, data out, work in) | new |
| `scheduler/` | Ray basics: `Cluster`, `Worker`, `WorkerGroup`, `Channel`, collectives, placement, dynamic scheduler, hardware probes | vendored from `rlinf/scheduler/` |
| `utils/` | placement, data iteration, distributed helpers, metrics, nested-dict ops, HTTP client, logging, timers | vendored from `rlinf/utils/` |
| `data/` | `io_struct` (LLM) and `embodied_io_struct` (trajectories) — internal representation, converted to `api/v1` at the boundary | vendored from `rlinf/data/` |
| `weight_sync/` | bucket / patch syncers and compressors | vendored from `rlinf/hybrid_engines/weight_syncer/` |

Two deliberate differences from the vendored originals:

- `Cluster.PACKAGE_NAME` is `rlinf_rollout`, so Ray code sync
  (`RLINF_CODE_WORKING_DIR`) ships this package. It accepts `auto`, the package
  directory, or a checkout root — the last two keep working after
  `git subtree split`, where `pyproject.toml` lives inside the package directory.
- `Cluster.SYS_NAME` stays `RLinf`, so the scheduler env vars keep their
  `RLINF_*` names (`RLINF_NODE_RANK`, `RLINF_COMM_NET_DEVICES`, ...).

Two imports point at modules that later phases vendor
(`rlinf_rollout.envs...lumos_camera`, `rlinf_rollout.workers.rollout.sglang`).
Both are lazy or type-only and carry a `TODO(agent)` note; the test suite keeps
that list closed.

## The `v1` protocol

`rlinf_rollout.api.v1` is the only stable surface. Every message carries
`schema_version == "v1"`, and the field set of each type is locked by
`tests/test_api_v1_schema.py`.

| Direction | Interface | Messages |
|---|---|---|
| weights in | `WeightReceiver` | `WeightUpdateRequest` → `WeightUpdateAck` |
| data out | `TrajectorySink` | `Trajectory` (embodied), `RolloutResult` (LLM), shaped by `ConsumerSpec` |
| work in | `TaskSource` | `RolloutTask` (`PromptSpec` \| `EpisodeSpec`) |

Three deliberate departures from the in-tree RLinf structs:

- Outputs are not split by trainer world size; the consumer declares its
  partitioning through `ConsumerSpec`.
- `forward_inputs` becomes `PolicyInputs`, which names its producer, layout and
  required keys.
- Trainer-only quantities (advantages, returns, reference logprobs, bootstrap
  values) are not part of the protocol.

```python
from rlinf_rollout.api.v1 import EpisodeSpec, RolloutTask, TaskKind

task = RolloutTask(
    kind=TaskKind.EMBODIED_EPISODE,
    episode=EpisodeSpec(env_type="maniskill", num_envs=8, group_size=4),
)
```

## Tests

```bash
pytest rlinf_rollout/tests
```

`test_api_v1_schema.py` locks the protocol; `test_phase1_vendoring.py` checks the
vendored tree (every internal import resolves, no `rlinf.` paths survive,
`pyproject.toml` packages match the directory tree) and imports every vendored
module. The import checks skip when `ray>=2.47` / `torch>=2.5` are unavailable.
The main repo runs this suite through `tests/unit_tests/test_rlinf_rollout.py`.

