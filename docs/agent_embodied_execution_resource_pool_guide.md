# Agent Embodied Execution and Resource Pool Guide

This note is for agents that need to reproduce the current embodied training
run, switch to another simulator/model pair, or enable CPU resource-pool and
core-donation scheduling.

## Current RoboCasa + OpenPI Run

Use this command to run RoboCasa + OpenPI for one global step with four rollout
epochs. It writes logs to a timestamped `profile_data` directory.

```bash
source /data1/miliang/RLinf/robocasa_openpi/bin/activate

OUT=/data1/miliang/RLinf/profile_data/robocasa_step_timing_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"

export EMBODIED_PATH=/data1/miliang/RLinf/examples/embodiment
export PYTHONPATH=/data1/miliang/RLinf:${PYTHONPATH:-}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export ROBOT_PLATFORM=LIBERO
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python examples/embodiment/train_embodied_agent.py \
  --config-path /data1/miliang/RLinf/examples/embodiment/config \
  --config-name robocasa_realA_profile_openpi \
  runner.logger.log_path="$OUT/logs" \
  runner.logger.logger_backends=[] \
  runner.max_epochs=1 \
  runner.max_steps=1 \
  algorithm.rollout_epoch=4 \
  runner.val_check_interval=-1 \
  runner.save_interval=-1 \
  2>&1 | tee "$OUT/train.log"
```

The base config is:

```text
examples/embodiment/config/robocasa_realA_profile_openpi.yaml
```

Important overrides:

- `runner.max_steps=1`: one global training step.
- `algorithm.rollout_epoch=4`: four rollout epochs inside that step.
- `runner.val_check_interval=-1` and `runner.save_interval=-1`: disable eval and
  checkpoint save for profiling/smoke runs.
- `runner.logger.logger_backends=[]`: avoid wandb/tensorboard dependencies.

## RoboCasa Step Timing Logs

When `env.train.log_sim_timestamps: true`, env timing logs are written to:

```text
<runner.logger.log_path>/env_sim_timestamps/env_rank_<rank>.jsonl
```

RoboCasa real simulator step records use:

```json
{"event": "robocasa_env_step"}
```

Useful fields:

- `rank`: EnvWorker rank.
- `global_env`: global env id across ranks.
- `local_env`: env id local to one EnvWorker.
- `duration_s`: elapsed time for one real RoboCasa `env.step()`.
- `wall_start_ns` and `wall_end_ns`: wall-clock timestamps.
- `chunk_step`: EnvWorker chunk-step index.
- `vector_step`: vector-env call index.
- `chunk_action_index` and `repeat_index`: present for subprocess chunk paths.

Quick summary command:

```bash
python - <<'PY'
import json
import os
from pathlib import Path
from statistics import mean

root = Path(os.environ["OUT"]) / "logs" / "env_sim_timestamps"
vals = []
by_rank = {}
by_global_env = {}

for path in sorted(root.glob("env_rank_*.jsonl")):
    rank = int(path.stem.split("_")[-1])
    count = 0
    for line in path.open():
        rec = json.loads(line)
        if rec.get("event") != "robocasa_env_step":
            continue
        vals.append(float(rec["duration_s"]))
        count += 1
        global_env = rec.get("global_env")
        by_global_env[global_env] = by_global_env.get(global_env, 0) + 1
    by_rank[rank] = count

vals = sorted(vals)

def pct(p):
    return vals[round((len(vals) - 1) * p / 100)]

print("files:", len(by_rank))
print("events:", len(vals))
print("by_rank:", by_rank)
print(
    "global_envs:",
    len(by_global_env),
    "min_events_per_env:",
    min(by_global_env.values()),
    "max_events_per_env:",
    max(by_global_env.values()),
)
print(
    "duration_s:",
    "mean", mean(vals),
    "min", vals[0],
    "p50", pct(50),
    "p90", pct(90),
    "p95", pct(95),
    "p99", pct(99),
    "max", vals[-1],
)
PY
```

Export `OUT` before running the summary command:

```bash
export OUT=/data1/miliang/RLinf/profile_data/<run_dir>
```

## Switch Simulator and Model

First look for an existing full config:

```bash
ls examples/embodiment/config/*.yaml
```

Common embodied configs:

```text
libero_10_ppo_openpi_pi05
libero_10_grpo_openvlaoft
maniskill_ppo_openpi_pi05
maniskill_ppo_openvla_quickstart
maniskill_ppo_openvlaoft
metaworld_45_ppo_openpi
robocasa_realA_profile_openpi
robocasa_profile_pairing_dynamic_core_donation
```

Run another existing config with the same entrypoint:

```bash
source /data1/miliang/RLinf/libero_openpi/bin/activate

export EMBODIED_PATH=/data1/miliang/RLinf/examples/embodiment
export PYTHONPATH=/data1/miliang/RLinf:${PYTHONPATH:-}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export ROBOT_PLATFORM=LIBERO
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python examples/embodiment/train_embodied_agent.py \
  --config-path examples/embodiment/config \
  --config-name libero_10_ppo_openpi_pi05 \
  runner.logger.logger_backends=[] \
  runner.max_steps=1 \
  runner.val_check_interval=-1 \
  runner.save_interval=-1
```

If no full config exists, create a new YAML by composing env and model defaults:

```yaml
defaults:
  - env/<env_name>@env.train
  - env/<env_name>@env.eval
  - model/<model_name>@actor.model
  - training_backend/fsdp@actor.fsdp_config
  - weight_syncer/patch_syncer@weight_syncer
  - override hydra/job_logging: stdout
```

Available env defaults:

```bash
ls examples/embodiment/config/env
```

Available model defaults:

```bash
ls examples/embodiment/config/model
```

Supported embodied model types include:

```text
openpi
openvla
openvla_oft
gr00t
starvla
dexbotic_pi
dexbotic_dm0
dreamzero
mlp_policy
cnn_policy
flow_policy
lingbotvla
```

Do not assume every env/model pair is compatible. Prefer starting from the
nearest existing config and changing only model path, env default, and small
batch/env counts.

Common overrides when changing a pair:

```bash
actor.model.model_path=/path/to/checkpoint \
rollout.model.model_path=/path/to/checkpoint \
rollout.unnorm_key=<norm_key> \
env.train.total_num_envs=32 \
env.train.max_steps_per_rollout_epoch=160 \
actor.micro_batch_size=32 \
actor.global_batch_size=512
```

## Resource Pool CPU Binding

Resource-pool config lives under `cluster.resource_pool`. Static per-env CPU
binding for env subprocesses looks like this:

```yaml
cluster:
  resource_pool:
    enabled: true
    allocation_mode: default
    cpu:
      enabled: true
      pools:
        env_cpu:
          node_group: cluster
          cores: ${oc.env:RLINF_ENV_CPU_CORES,0-111}
      components:
        env:
          pool: env_cpu
          granularity: per_env
```

Set the CPU pool before running:

```bash
export RLINF_ENV_CPU_CORES=0-111
```

`granularity: per_env` splits each EnvWorker CPU allocation into one core group
per local env subprocess. Supported per-env backends include:

```text
calvin
habitat
libero
metaworld
robocasa
d4rl with use_subproc_vector_env
```

For unsupported env backends, use `granularity: process` or disable the resource
pool.

## GPU Resource Pool

For env rendering GPU visibility, the current configs use MPS mode with
`sm_percent: 0`, meaning the resource pool controls visible devices but does not
reserve a positive SM quota.

```yaml
cluster:
  resource_pool:
    gpu:
      enabled: true
      mode: mps
      pools:
        env_render_gpu:
          node_group: cluster
          devices: ${oc.env:RLINF_ENV_RENDER_GPUS,0-7}
      components:
        env:
          pool: env_render_gpu
          sm_percent: 0
```

Set:

```bash
export RLINF_ENV_RENDER_GPUS=0-7
```

## Core Donation V2

Core donation is the retained dynamic CPU scheduling mode. It requires per-env
CPU core groups from the resource pool.

```yaml
env:
  train:
    chunk_step_mode: latency_balanced_pair
    latency_balanced_pair:
      envs_per_core: 1
      ema_alpha: 0.3
      initial_latency_ms: null
      dynamic_affinity: true
      core_donation_enabled: true
      core_donation_max_extra_groups: 1
```

Current validation requires:

- `envs_per_core: 1`
- `dynamic_affinity: true`
- `core_donation_enabled: true`
- `ema_alpha` in `(0, 1]`
- `initial_latency_ms` unset or positive
- `core_donation_max_extra_groups >= 0`

Use `sync_time_major` for baselines without dynamic scheduling:

```yaml
env:
  train:
    chunk_step_mode: sync_time_major
```

Recommended comparison set:

1. No CPU binding, no scheduling: `sync_time_major` and no env CPU resource pool.
2. Static CPU binding: `sync_time_major` plus env `per_env` CPU resource pool.
3. Core donation v2: `latency_balanced_pair` plus env `per_env` CPU resource pool.

## RoboCasa Core-Donation Run

Use the existing config:

```text
examples/embodiment/config/robocasa_profile_pairing_dynamic_core_donation.yaml
```

Command:

```bash
source /data1/miliang/RLinf/robocasa_openpi/bin/activate

OUT=/data1/miliang/RLinf/profile_data/robocasa_core_donation_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"

export EMBODIED_PATH=/data1/miliang/RLinf/examples/embodiment
export PYTHONPATH=/data1/miliang/RLinf:${PYTHONPATH:-}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export ROBOT_PLATFORM=LIBERO
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RLINF_ENV_CPU_CORES=0-111
export RLINF_ENV_RENDER_GPUS=0-7

python examples/embodiment/train_embodied_agent.py \
  --config-path examples/embodiment/config \
  --config-name robocasa_profile_pairing_dynamic_core_donation \
  runner.logger.log_path="$OUT/logs" \
  runner.logger.logger_backends=[] \
  runner.max_epochs=1 \
  runner.max_steps=1 \
  algorithm.rollout_epoch=4 \
  runner.val_check_interval=-1 \
  runner.save_interval=-1 \
  2>&1 | tee "$OUT/train.log"
```

## LIBERO Core-Donation Run

LIBERO configs use the same `latency_balanced_pair` mechanism. Example:

```bash
source /data1/miliang/RLinf/libero_openpi/bin/activate

export EMBODIED_PATH=/data1/miliang/RLinf/examples/embodiment
export PYTHONPATH=/data1/miliang/RLinf:${PYTHONPATH:-}
export LIBERO_TYPE=standard
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export ROBOT_PLATFORM=LIBERO
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RLINF_ENV_CPU_CORES=0-111
export RLINF_ENV_RENDER_GPUS=0-7

python examples/embodiment/train_embodied_agent.py \
  --config-path examples/embodiment/config \
  --config-name 0libero90_envs_per_core1_profile_test \
  runner.logger.logger_backends=[] \
  runner.max_epochs=1 \
  runner.max_steps=1 \
  runner.val_check_interval=-1 \
  runner.save_interval=-1
```

## Troubleshooting

If Ray fails to auto-start with a GCS port-file timeout inside a sandbox, rerun
outside the sandbox or start Ray manually before launching:

```bash
ray start --head
```

Then run the training command on the head node.

If RoboCasa cannot render, check:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

If CPU binding fails, check that the configured cores are visible:

```bash
taskset -pc $$
nproc
```

If `latency_balanced_pair` errors, verify that env CPU resource pool is enabled
with `granularity: per_env`. Core donation has no core groups to donate without
per-env CPU binding.

## Verification Commands

After changing RoboCasa timing or resource-pool code, run:

```bash
source /data1/miliang/RLinf/libero_openpi/bin/activate

ruff check \
  rlinf/envs/robocasa/venv.py \
  rlinf/envs/robocasa/robocasa_env.py \
  tests/unit_tests/test_robocasa_env.py

pytest -q tests/unit_tests/test_robocasa_env.py -q
```

For resource-pool-only changes, add:

```bash
pytest -q tests/unit_tests/test_resource_pool_config.py
pytest -q tests/unit_tests/test_resource_pool_solver.py
pytest -q tests/unit_tests/test_resource_pool_env_binding.py
```
