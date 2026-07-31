# RLinf Rollout

Standalone rollout system for embodied and LLM policies, extracted from
[RLinf](https://github.com/RLinf/RLinf). It contains rollout and environment
execution only — no training loop — so it can be driven by any training
framework.

Status: **Phase 0** — skeleton plus the frozen `v1` protocol. Scheduler, envs,
models and workers arrive in Phase 1–3 (see `../ROLLOUT_SPLIT_PLAN.md`).

## Install

```bash
pip install -e rlinf_rollout            # core (Ray + torch)
pip install -e "rlinf_rollout[embodied]" # embodied policies and envs
pip install -e "rlinf_rollout[sglang]"   # LLM rollout on SGLang
pip install -e "rlinf_rollout[vllm]"     # LLM rollout on vLLM
```

The package never imports the training-side `rlinf` package; everything it
needs is vendored under `rlinf_rollout/`.

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
