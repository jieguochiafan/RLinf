# Async GIPO 128-Trajectory Execution Guide

This note summarizes the async GIPO LIBERO/OpenPI run used to validate
128 training trajectory segments. It is intended as a handoff document for
agents that need to reproduce, inspect, or extend the test.

## Goal

Run async GIPO training outside the sandbox with a fixed 128-trajectory
training-consumption target, then report:

- end-to-end training wall time
- training throughput
- actor-training vs non-actor-training time
- rollout trajectory duration distribution
- known gaps for simulator/generation bubble analysis

## Repository And Environment

- Repository: `/data1/miliang/RLinf`
- Python environment: `/data1/miliang/RLinf/libero_openpi`
- Entry point: `examples/embodiment/train_async.py`
- Config: `examples/embodiment/config/libero_spatial_async_gipo_openpi_pi05_verify.yaml`
- Model path used by the config:
  `/data1/gaobowen/model/RLinf-Pi05-LIBERO-SFT`

Use the virtualenv explicitly before launching:

```bash
source /data1/miliang/RLinf/libero_openpi/bin/activate
```

The run should be executed outside the Codex sandbox because it needs GPUs,
Ray workers, CUDA/EGL, and long-running processes.

## Key Config Values

The verification YAML uses a single node and this placement:

```yaml
cluster:
  num_nodes: 1
  component_placement:
    actor: 6-7
    rollout: 4-5
    env: 4-7
```

Training environment parallelism:

```yaml
env:
  train:
    total_num_envs: 128
rollout:
  pipeline_stage_num: 2
algorithm:
  gipo:
    target_batch_segments: 8
```

Parallelism breakdown:

- Env world size: 4 ranks, from placement `env: 4-7`
- Pipeline stages: 2
- Envs per env rank per stage: `128 / 4 / 2 = 16`
- Envs per env rank total: `16 * 2 = 32`
- Total concurrent train env instances: 128

Actor/trajectory training-consumption target:

- Actor ranks: 2, from placement `actor: 6-7`
- Target sampled segments per actor per step: 8
- Run steps: 8
- Training-consumed trajectory segments: `8 steps * 2 actors * 8 segments = 128`

## Launch Command

The run that completed used this command shape:

```bash
source /data1/miliang/RLinf/libero_openpi/bin/activate
export EMBODIED_PATH=/data1/miliang/RLinf/examples/embodiment
export PYTHONPATH=/data1/miliang/RLinf:${PYTHONPATH}
export LIBERO_TYPE=standard
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export ROBOT_PLATFORM=LIBERO
export RLINF_TRAINING_EVAL_LOCAL_RAY=1

CUDA_VISIBLE_DEVICES=4,5,6,7 \
python examples/embodiment/train_async.py \
  --config-path /data1/miliang/RLinf/examples/embodiment/config \
  --config-name libero_spatial_async_gipo_openpi_pi05_verify \
  runner.max_epochs=8 \
  runner.max_steps=8 \
  runner.val_check_interval=-1 \
  runner.save_interval=-1 \
  runner.logger.logger_backends=[] \
  runner.logger.log_path=/tmp/rlinf_async_gipo_128traj_8step_YYYYMMDD_HHMMSS \
  +actor.recv_drain_max_trajectories=8
```

Notes:

- Do not set an external timeout if the goal is to finish the full 8-step run.
- `runner.max_epochs=8 runner.max_steps=8` is intentional. In this async runner,
  `max_steps` is the direct stop condition.
- `+actor.recv_drain_max_trajectories=8` limits each actor rank to draining
  up to 8 newly received trajectories per training step after warmup/backlog.
- The log path should include a timestamp. The completed reference run used:
  `/tmp/rlinf_async_gipo_128traj_8step_20260709_124959`.

## Resource Utilization Profiling

Use the wrapper script to collect RLinf-only CPU samples, chunked
`torch.profiler` traces, derived CSV summaries, and the final three-panel
resource utilization figure:

```bash
RUN_DIR=/tmp/rlinf_async_gipo_resource_$(date +%Y%m%d_%H%M%S) \
bash tools/run_async_gipo_resource_profile.sh
```

The script runs the same 8-step GIPO configuration as the launch command
above, with resource profiling enabled. It also accepts extra Hydra overrides,
for example:

```bash
PROFILE_ACTIVE_STEPS=5 \
bash tools/run_async_gipo_resource_profile.sh \
  runner.max_epochs=1 \
  runner.max_steps=1
```

Metric definitions and limitations:

- CPU utilization is RLinf-only. The sampler reads Ray/RLinf process threads
  every 100 ms by default and attributes each thread delta to the logical CPU
  observed in the later sample. Migration counts are therefore approximate.
- GPU `kernel_busy_pct` is the union of CUDA kernel time intervals per physical
  GPU. Overlapping kernels are de-duplicated before computing busy percentage.
- Worker `est_sm_occupancy_pct` comes from `torch.profiler` kernel launch
  metadata and is weighted by kernel overlap with each time bucket.
- Neither `kernel_busy_pct` nor `est_sm_occupancy_pct` is hardware
  `SM_ACTIVE`. Treat them as profiler-derived approximations, not direct GPU
  performance-counter measurements.
- Full profiling increases runtime and disk usage because each worker exports
  repeated trace chunks.
- Before using the plot for analysis, inspect
  `<RUN_DIR>/resource_profile/derived/coverage.json` for missing traces,
  unmapped devices, missing occupancy fields, CPU over-100% buckets, and
  unmatched phase windows.

Primary outputs:

```text
<RUN_DIR>/resource_profile/cpu/thread_core_samples.csv
<RUN_DIR>/resource_profile/torch/*/trace_manifest.jsonl
<RUN_DIR>/resource_profile/derived/cpu_core_1s.csv
<RUN_DIR>/resource_profile/derived/gpu_device_1s.csv
<RUN_DIR>/resource_profile/derived/gpu_worker_1s.csv
<RUN_DIR>/resource_profile/derived/phase_windows.csv
<RUN_DIR>/resource_profile/derived/coverage.json
<RUN_DIR>/resource_profile/resource_utilization.png
<RUN_DIR>/resource_profile/resource_utilization.pdf
```

## Output Artifacts

The trajectory timestamp instrumentation writes JSONL files under:

```text
<runner.logger.log_path>/rollout_trajectory_timestamps/
```

Expected files:

```text
actor_rank_0.jsonl
actor_rank_1.jsonl
```

Each line is one drained rollout trajectory envelope:

```json
{
  "auto_reset": true,
  "completed_at": 2052720.957057892,
  "duration_s": 198.99873820901848,
  "env_rank": 0,
  "last_policy_version": 0,
  "segment_type": "episode",
  "stage_id": 0,
  "started_at": 2052521.958319683
}
```

The reference run also generated a derived CSV report:

```text
<runner.logger.log_path>/rollout_trajectory_duration_report.csv
```

That CSV was generated from JSONL after the run. It included:

- inferred training step
- actor rank
- JSONL line number
- env rank
- stage id
- segment type
- policy version
- start timestamp
- completion timestamp
- duration in seconds

## Timestamp Instrumentation Locations

The current implementation records rollout trajectory timing in these places:

- `rlinf/data/embodied_async.py`
  - `AsyncTrajectoryEnvelope.started_at`
  - `AsyncTrajectoryEnvelope.completed_at`
  - `AsyncTrajectoryEnvelope.duration_s`
  - `AsyncTrajectoryEnvelope.timing_payload()`
- `rlinf/workers/env/async_env_worker.py`
  - starts the GIPO trajectory timer when a stage begins a new segment
  - sets `completed_at` when the stage flushes an episode/fixed-horizon segment
- `rlinf/workers/actor/async_gipo_fsdp_worker.py`
  - records timing payloads when actor drains trajectory envelopes
  - writes JSONL files under `rollout_trajectory_timestamps`
  - adds aggregated training metrics:
    - `rollout/trajectory_count`
    - `rollout/trajectory_duration_mean`
    - `rollout/trajectory_duration_min`
    - `rollout/trajectory_duration_max`
    - `rollout/trajectory_collect_window`

Related regression tests:

```bash
pytest -q tests/unit_tests/test_async_gipo_runner.py::test_gipo_actor_records_rollout_trajectory_timestamps
pytest -q tests/unit_tests/test_async_gipo_runner.py::test_gipo_actor_pads_replay_batch_to_common_time_dim
```

## Reference Run Result

The completed reference run reached:

```text
Global Step: 8/8
exit code: 0
```

Ray logs were checked for critical failures and did not match:

```text
Traceback|RuntimeError|IndexError|Exception occurred|NCCL error|CUDA out of memory
```

Training-consumption count:

- Target/consumed training segments: 128
- Actual drained JSONL rollout trajectories: 139

The 11 extra JSONL rows came from first-step warmup/backlog drain. Use 128 as
the main training-consumption denominator, and 139 only when analyzing all
drained rollout envelopes.

## Reference Timing Summary

End-to-end wall time:

- Final displayed elapsed time: `18:15`
- Converted wall time: `1095s`

Actor training:

- Sum of actor-training times: `939.724s`

Non-actor-training time:

- `1095s - 939.724s = 155.276s`
- Full-run non-actor-training bubble ratio:
  `155.276 / 1095 = 14.18%`

Warmup-excluded estimate, steps 2-8:

- Step 2-8 wall time: about `610.8s`
- Step 2-8 actor training: `565.424s`
- Step 2-8 non-actor-training time: about `45.4s`
- Step 2-8 non-actor-training bubble ratio: about `7.4%`

Throughput:

- Training-consumption throughput:
  `128 trajectories / 1095s = 0.1169 traj/s`
- Equivalent:
  - `7.01 traj/min`
  - `420.8 traj/hour`

If using all drained JSONL rows:

- `139 trajectories / 1095s = 0.1270 traj/s`
- Equivalent:
  - `7.62 traj/min`
  - `457.0 traj/hour`

Use the 128-trajectory throughput as the main training throughput.

## Per-Step Timing Reconstruction

The Rich metric table prints `Step Time` as elapsed average per completed step,
not as a direct per-step delta. To reconstruct incremental wall time:

```text
incremental_step_time[N] =
  displayed_average_step_time[N] * N
  - displayed_average_step_time[N - 1] * (N - 1)
```

Reference reconstruction:

| Step | Wall-clock delta | Actor training | Other / wait approx |
| ---: | ---------------: | -------------: | ------------------: |
| 1 | 485.000s | 374.300s | 110.700s |
| 2 | 87.058s | 81.811s | 5.247s |
| 3 | 88.272s | 82.890s | 5.382s |
| 4 | 77.762s | 72.644s | 5.118s |
| 5 | 102.363s | 96.359s | 6.004s |
| 6 | 53.809s | 48.234s | 5.575s |
| 7 | 97.888s | 86.778s | 11.110s |
| 8 | 103.656s | 96.708s | 6.948s |

The `Other / wait approx` column is:

```text
incremental wall-clock delta - actor_training
```

It includes scheduling, data readiness, synchronization, and async pipeline wait.
It is not the same as rollout-internal simulator/generation bubble.

## Rollout Trajectory Duration Summary

Reference full JSONL distribution, 139 drained trajectories:

| Metric | Duration |
| -----: | -------: |
| mean | 24.532s |
| min | 3.551s |
| p50 | 8.280s |
| p90 | 48.358s |
| p95 | 118.885s |
| p99 | 225.767s |
| max | 230.595s |

Global collection window:

```text
607.086s
```

By actor rank:

| Actor rank | Count | Mean | p50 | Max |
| ---------: | ----: | ---: | --: | --: |
| 0 | 71 | 20.897s | 8.511s | 198.999s |
| 1 | 68 | 28.328s | 7.960s | 230.595s |

Longest observed trajectories:

| Duration | Actor rank | Env rank | Stage |
| -------: | ---------: | -------: | ----: |
| 230.595s | 1 | 1 | 0 |
| 227.156s | 1 | 3 | 0 |
| 223.501s | 1 | 2 | 0 |
| 198.999s | 0 | 0 | 0 |

## Post-Run JSONL Summary Script

Use this script after a new run to summarize JSONL files:

```bash
python - <<'PY'
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

run_dir = Path("/tmp/rlinf_async_gipo_128traj_8step_YYYYMMDD_HHMMSS")
ts_dir = run_dir / "rollout_trajectory_timestamps"
records = []

for path in sorted(ts_dir.glob("actor_rank_*.jsonl")):
    rank = int(path.stem.split("_")[-1])
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            row = json.loads(line)
            row["actor_rank"] = rank
            row["line_no"] = line_no
            records.append(row)

def pct(vals, q):
    vals = sorted(vals)
    pos = (len(vals) - 1) * q / 100
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)

def summary(rows):
    vals = [float(r["duration_s"]) for r in rows]
    return {
        "count": len(vals),
        "mean": statistics.fmean(vals),
        "min": min(vals),
        "p50": pct(vals, 50),
        "p90": pct(vals, 90),
        "p95": pct(vals, 95),
        "p99": pct(vals, 99),
        "max": max(vals),
    }

print("global", summary(records))
print(
    "global_collect_window",
    max(r["completed_at"] for r in records) - min(r["started_at"] for r in records),
)

by_rank = defaultdict(list)
for row in records:
    by_rank[row["actor_rank"]].append(row)

for rank, rows in sorted(by_rank.items()):
    print(f"actor_rank_{rank}", summary(rows))
PY
```

## Bubble Definitions

Use two separate bubble definitions:

1. Rollout-internal simulator/generation bubble
   - Simulator waiting for generation:
     env has sent an inference request and waits for an action response.
   - Generation waiting for simulator:
     rollout worker/GPU has no pending request and waits for env requests.
   - The reference run cannot compute this precisely because the recorded JSONL
     only has trajectory start/end timestamps.

2. Non-actor-training bubble
   - Wall-clock time not spent inside actor training.
   - This can be approximated from the metric table:
     `wall-clock - actor_training`.
   - Reference value:
     - full run: about `14.2%`
     - excluding first-step warmup/backlog: about `7.4%`

## Current Trace Limitations

The current trajectory timestamp trace is enough for:

- per-trajectory rollout duration
- rollout duration distribution by actor rank, env rank, and stage id
- training-consumption throughput
- non-actor-training bubble approximation

It is not enough for precise rollout-internal bubble analysis. To compute
simulator/generation waiting separately, add per-action or per-batch events:

- Env side:
  - `sim_step_start`
  - `request_sent_at`
  - `response_received_at`
  - `sim_step_end`
- Rollout side:
  - `request_received_at`
  - `generation_start`
  - `generation_end`
  - `response_sent_at`

The codebase already has related optional timestamp mechanisms:

- env sim timestamps under `env_sim_timestamps`
- rollout generation timestamps under `rollout_generation_timestamps`

Future analysis should either enable those fields if they cover GIPO action
inference, or add explicit GIPO request/response timestamps with a shared
request id.

## Verification Commands

Lightweight code/test checks used during this work:

```bash
source /data1/miliang/RLinf/libero_openpi/bin/activate

ruff check \
  rlinf/workers/actor/async_gipo_fsdp_worker.py \
  tests/unit_tests/test_async_gipo_runner.py

pytest -q \
  tests/unit_tests/test_async_gipo_data_contracts.py \
  tests/unit_tests/test_async_gipo_runner.py \
  tests/unit_tests/test_async_gipo_env_flow.py \
  tests/unit_tests/test_replay_buffer_trajectory_sampling.py
```

The reference unit-test result was:

```text
20 passed
```

## Common Pitfalls

- Do not use `runner.max_epochs=1 runner.max_steps=8` for this config if the
  runner derives `max_steps` through a `min(...)` path; earlier testing showed
  this can stop after one step in some runner paths. For the async GIPO command
  above, set both to 8.
- Do not interpret JSONL row count as training-consumed count. JSONL records
  drained trajectories; replay sampling consumes `target_batch_segments` per
  actor rank per training step.
- The first step includes warmup/backlog and is not representative of steady
  throughput or bubble.
- `/tmp` run outputs are ephemeral. Copy or regenerate summaries before relying
  on old paths.
