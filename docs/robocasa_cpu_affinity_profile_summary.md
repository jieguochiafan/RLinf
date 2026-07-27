# RoboCasa CPU Affinity Profile Summary

Date: 2026-07-10

This note summarizes the RoboCasa/OpenPI CPU affinity profiling state so another agent can continue analysis without re-reading the full conversation. The original runs below used 16 train environments, `env.train.max_steps_per_rollout_epoch=300`, GPUs 4-7, rollout profiling enabled, and the venv `/data1/miliang/RLinf/robocasa_openpi`. A 32-environment follow-up is documented at the end.

## Important Baseline Definition

The fair no-binding baseline for the main question is **OS default scheduling**:

- No CPU resource pool binding.
- No `taskset`.
- No process or child affinity restriction.
- Env and subprocesses are scheduled by the OS over the machine's available cores.

The later "16-core no-per-env" run is only an ablation. It restricts Env workers to 16 CPU cores but does not bind one env to one core. Do not treat it as the OS default baseline.

## Main Runs

| Label | Run directory | CPU policy | Status |
|---|---|---|---|
| OS default no-binding | `profile_data/rollout_profile_phys4_7_16env_300step_nobind_sync_20260710_164430` | `cluster.resource_pool.cpu.enabled=false`; no affinity restriction | 0 |
| Permanent per-env binding | `profile_data/rollout_profile_phys4_7_16env_300step_bind1core_sync_20260710_165731` | `cores=0-15`, `granularity=per_env`, `cpu_affinity_scope=process`; each env child permanently bound to one core | 0 |
| Step-only per-env binding | `profile_data/rollout_profile_phys4_7_16env_300step_steponly_sync_20260710_173200` | per-env core groups exist; child binds to its env core only during `step` / `chunk_step`, then restores | 0 |
| 16-core no-per-env ablation | `profile_data/rollout_profile_phys4_7_16env_300step_16core_sync_noperenv_20260710_201655` | `cores=0-15`, `granularity=process`; each EnvGroup gets 4 cores, its 4 children share those cores | 0 |
| 16-core per-env retest | `profile_data/rollout_profile_phys4_7_16env_300step_16core_sync_perenv_process_20260710_202151` | `cores=0-15`, `granularity=per_env`, permanent one env per core | 0 |

## End-to-End Metrics

| Run | `step` | `generate_rollouts` | `env/interact` | `env/env_interact_step` | `env/recv_rollout_results` | `rollout/predict` |
|---|---:|---:|---:|---:|---:|---:|
| OS default no-binding | 51.274s | 47.678s | 47.438s | 20.593s | 16.117s | 14.563s |
| Permanent per-env binding | 160.800s | 137.700s | 137.500s | 19.876s | 74.660s | 16.220s |
| Step-only per-env binding | 48.788s | 45.654s | 45.396s | 19.679s | 15.296s | 14.054s |
| 16-core no-per-env ablation | 48.574s | 45.767s | 45.549s | 18.488s | 15.018s | 14.083s |
| 16-core per-env retest | 49.531s | 46.480s | 46.256s | 18.400s | 14.958s | 13.700s |

## Low-Level Event Metrics

Mean durations from `rollout_profile/*.jsonl`:

| Run | `robocasa.child_step` | `env.chunk_step` | `rollout.predict` | `env.recv_rollout_results` | `rollout.recv_env_output` |
|---|---:|---:|---:|---:|---:|
| OS default no-binding | 47.515ms | 318.063ms | 233.828ms | 252.753ms | 508.102ms |
| Permanent per-env binding | 46.689ms | 311.786ms | 244.758ms | 661.307ms | 1966.076ms |
| Step-only per-env binding | 45.903ms | 305.993ms | 226.539ms | 242.151ms | 490.285ms |
| 16-core no-per-env ablation | 43.993ms | 290.032ms | 227.359ms | 243.186ms | 494.237ms |
| 16-core per-env retest | 43.679ms | 288.282ms | 223.742ms | 239.748ms | 497.504ms |

## Interpretation So Far

Standalone simulator tests reportedly show about 30% step throughput improvement from CPU binding. The end-to-end runs do not show the same scale because the bound work is only part of the full rollout critical path.

For OS default versus step-only:

- `robocasa.child_step` improves from 47.515ms to 45.903ms, about 3.4%.
- `env/env_interact_step` improves from 20.593s to 19.679s, about 4.4%.
- Total `step` improves from 51.274s to 48.788s, about 4.9%.

Permanent per-env binding is not viable in the current end-to-end path:

- The raw child step is slightly faster than OS default.
- But `rollout.recv_env_output` has extreme long-tail waiting: mean 1966ms and max about 99.8s.
- `env.recv_rollout_results` also has long-tail waiting: mean 661ms and max about 60.3s.
- Total `step` regresses to 160.8s.

The 16-core ablation answers a different question: if both policies are restricted to 16 Env CPU cores, does one-env-one-core help versus four envs sharing four cores per EnvGroup?

- No-per-env ablation: `step=48.574s`.
- Per-env retest: `step=49.531s`.
- The raw child step is slightly faster with per-env binding, but end-to-end is about 2% slower.
- This supports the hypothesis that per-env binding helps local step execution but does not necessarily improve the rollout critical path.

## Affinity Evidence

OS default no-binding:

- No Env CPU binding log lines.
- `cluster.resource_pool.cpu.enabled=false`.

Permanent per-env binding:

- EnvGroup 0 uses `(0, 1, 2, 3)` and children are `(0,)`, `(1,)`, `(2,)`, `(3,)`.
- EnvGroup 1 uses `(4, 5, 6, 7)` and children are `(4,)`, `(5,)`, `(6,)`, `(7,)`.
- EnvGroup 2 uses `(8, 9, 10, 11)` and children are `(8,)`, `(9,)`, `(10,)`, `(11,)`.
- EnvGroup 3 uses `(12, 13, 14, 15)` and children are `(12,)`, `(13,)`, `(14,)`, `(15,)`.

Step-only run:

- Config has `cpu_affinity_scope=step_only`.
- EnvGroup parent and child startup affinity remained broad, e.g. `(0..111)`.
- `env_sim_timestamps` showed `subenv_start` affinity as single-core groups `(0,)` through `(15,)` during step.
- All 240 chunk start records showed children back at full affinity before entering the next step section.

16-core no-per-env ablation:

- EnvGroup process affinity is limited to four cores per rank.
- Child affinities match the EnvGroup four-core pool, not single-core binding.
- Example: EnvGroup 0 children all have `(0, 1, 2, 3)`.

## Failed/Discarded Runs

These should not be used as final profile comparisons:

- `profile_data/rollout_profile_phys4_7_16env_300step_16core_process_noperenv_20260710_200813`
  - Failed because `latency_balanced_pair` requires per-env CPU core groups.
  - Log says to use `sync_time_major` for the no-CPU-binding baseline.
- `profile_data/rollout_profile_phys4_7_16env_300step_16core_sync_noperenv_20260710_201224`
  - Rollout completed, but actor training failed because `240 is not divisible by 512`.
  - The later successful run fixed this by matching the old profile overrides: `actor.micro_batch_size=16`, `actor.global_batch_size=64`, `algorithm.update_epoch=0`.

## Current Working Conclusion

Use these three as the main comparison for the original question:

1. OS default no-binding: `51.274s`.
2. Permanent per-env binding: `160.800s`, rejected due to communication/wait long tails.
3. Step-only per-env binding: `48.788s`, about 4.9% faster than OS default but far below the standalone simulator's reported 30% local step improvement.

The most likely explanation is that standalone testing measures a CPU-bound tight loop around simulator step, while end-to-end rollout includes policy prediction, env/rollout send-recv, rank synchronization, bootstrap/reset work, and long-tail waits. CPU binding can improve local child step latency, but only improvements on the rollout critical path produce end-to-end speedup.

## Suggested Next Analysis

- Compare standalone simulator benchmark code against end-to-end `robocasa.child_step` instrumentation to ensure they measure the same operation.
- Inspect per-rank timelines for the permanent binding run to identify why `rollout.recv_env_output` and `env.recv_rollout_results` developed extreme long tails.
- For step-only, check whether limiting restoration to an Env CPU pool instead of full-machine affinity changes the result, but only if code changes are allowed.
- Consider running 3 repeats for OS default and step-only because the observed 4.9% gain is small enough that noise and reset/task randomness may matter.

## 32-Environment Follow-up

Date: 2026-07-10 to 2026-07-11

The 32-environment follow-up kept the end-to-end workload fixed:

- Task: RoboCasa `CloseDrawer`
- Train environments: 32
- Env/Rollout ranks: 4, with 8 environments per rank
- Physical rollout horizon: 300 steps, or 60 action chunks
- Actor/Rollout/Env placement: GPUs 4-7
- Scheduling mode: `sync_time_major`
- Actor update disabled with `algorithm.update_epoch=0`
- Rollout profiling and RoboCasa child-step tracing enabled

### End-to-End Runs

| Label | Run directory | CPU policy | Status |
|---|---|---|---|
| OS default | `profile_data/rollout_profile_phys4_7_32env_300step_nobind_sync_20260710_230645` | CPU resource pool disabled; affinity remains `0-111` | 0 |
| 32-core process pool | `profile_data/rollout_profile_phys4_7_32env_300step_32core_sync_noperenv_20260710_231403` | `cores=0-31`, `granularity=process`; each rank and its 8 children share 8 cores | 0 |
| GPU-local NUMA pool | `profile_data/rollout_profile_phys4_7_32env_300step_28core_numa1_sync_noperenv_20260710_233133` | `cores=28-55`, `granularity=process`; each rank and its 8 children share 7 cores | 0 |

The `0-31` run is the direct 32-env extension of the earlier process-pool
ablation. The `28-55` run was added because GPUs 4-7 are local to NUMA node 1,
whose physical CPU cores are 28-55. It is a topology check, not an equal-core
comparison, because that socket has only 28 physical cores.

### End-to-End Metrics

| Run | `step` | vs OS default | `generate_rollouts` | `env/interact` | `env/env_interact_step` | `rollout/predict` |
|---|---:|---:|---:|---:|---:|---:|
| OS default | 60.452s | baseline | 57.328s | 57.060s | 24.532s | 19.798s |
| 32-core process pool | 62.298s | 3.05% slower | 59.341s | 59.054s | 24.529s | 19.621s |
| GPU-local 28-core pool | 65.986s | 9.15% slower | 62.891s | 62.617s | 26.050s | 20.445s |

The direct 32-core comparison does not show an affinity speedup. The metric
that most directly covers simulation execution, `env/env_interact_step`, is
effectively unchanged: 24.532s to 24.529s. Total step time regresses because
the rollout critical path and receive waits get longer.

### Profile Event Metrics

| Run | `robocasa.child_step` mean | `env.chunk_step` mean | `rollout.recv_env_output` mean | `rollout.predict` mean |
|---|---:|---:|---:|---:|
| OS default | 48.634ms | 361.269ms | 569.937ms | 321.157ms |
| 32-core process pool | 48.809ms | 366.948ms | 606.476ms | 318.935ms |
| GPU-local 28-core pool | 53.259ms | 407.053ms | 671.894ms | 324.180ms |

Relative to OS default, the 32-core pool changes local child-step latency by
only +0.36%, while `rollout.recv_env_output` increases by 6.41%. The GPU-local
28-core pool is under-provisioned for this process-level policy: each rank has
8 child environments plus the parent and helper threads but only 7 cores. Its
child-step mean increases by 9.51% and its receive wait increases by 17.89%.

### Rank Critical Path

For the direct OS-default versus `0-31` comparison:

| Rank | OS-default child-step mean | `0-31` child-step mean | OS-default chunk loop | `0-31` chunk loop |
|---|---:|---:|---:|---:|
| 0 | 45.433ms | 43.674ms | 41.8s | 40.4s |
| 1 | 49.013ms | 47.800ms | 42.3s | 41.0s |
| 2 | 49.150ms | 47.734ms | 45.7s | 45.7s |
| 3 | 50.941ms | 56.028ms | 41.5s | 44.6s |

Ranks 0-2 get small or no local improvements, but rank 3 regresses enough to
erase them. Its pool, `24-31`, crosses the NUMA boundary: cores 24-27 are on
NUMA 0 and cores 28-31 are on NUMA 1. Since synchronous rollout completion is
limited by the slowest rank, average local improvements on non-critical ranks
do not produce an end-to-end gain.

The topology-only run confirms that crossing NUMA is not the sole cause. Using
only GPU-local cores 28-55 removes the cross-socket rank, but 8 environments
sharing 7 cores increases every rank's simulation time. The slowest rank total
increases from 56.4s for OS default to 61.9s.

### Toolkits Validation

Two implementation details are important before using
`toolkits/rollout_eval` as proof that CPU binding is effective:

1. In `toolkits/rollout_eval/benchmark/run.py`, CLI strategy `default` is
   converted to `none`. The active example in
   `toolkits/rollout_eval/benchmark_run.sh` therefore runs OS-default scheduling,
   despite using the `env_only_cpu_core` scenario name.
2. In `toolkits/rollout_eval/benchmark/orchestrator.py`, the current benchmark
   applies the union of all logical environment core groups to the environment
   process. It does not permanently pin each child environment to its reported
   single-core group.

A single 32-env toolkits process could not be used because all 32 RoboCasa EGL
contexts were created on one GPU and failed with `mujoco.FatalError: Offscreen
framebuffer is not complete`. The test was therefore reshaped to match the
training layout: four toolkits processes, each with 8 environments and one of
GPUs 4-7. The bound processes used core pools `0-7`, `8-15`, `16-23`, and
`24-31`. Each case used 3 warmup steps and 20 measured steps.

| Execution order | Policy | Per-rank vector-step mean | Aggregate logical env steps/s |
|---|---|---|---:|
| OS default first | OS default | 281-405ms | 91.97 |
| OS default first | 8-core process union | 1919-2030ms | 16.24 |
| Binding first | OS default | 68-193ms | 248.54 |
| Binding first | 8-core process union | 1928-2021ms | 16.37 |

Artifacts:

- `profile_data/toolkits_robocasa_4x8env_affinity_retry2_20260711`
- `profile_data/toolkits_robocasa_4x8env_affinity_reverse_20260711`

Reversing the execution order reproduces the binding regression, so it is not
caused by running the bound case after the default case. OS-default results are
noisy, but the bound result is consistently about 2 seconds per vector step.
Under the current union-affinity implementation, the fixed 8-core pool contains
the parent process, 8 child simulators, and their EGL/library helper threads.
The data is consistent with severe CPU oversubscription or contention inside
that restricted pool.

#### Direct 32-Environment, 32-Core Step Test

A final longer run directly compared OS-default scheduling against a total of
32 disjoint CPU cores. The test used four concurrent toolkit processes so the
32 RoboCasa environments could be distributed across GPUs 4-7 without the
single-GPU EGL framebuffer failure:

- 4 toolkit processes, 8 environments per process
- Bound pools: `0-7`, `8-15`, `16-23`, and `24-31`
- 10 warmup steps and 100 measured steps
- `max_episode_steps=1000` and `auto_reset=false`, so reset latency is excluded
- Artifact: `profile_data/toolkits_robocasa_32env_32core_direct_20260711_225657`

| Policy | Per-rank step mean | Aggregate logical env steps/s | Relative throughput |
|---|---:|---:|---:|
| OS default | 534.659-553.974ms | 58.785 | 3.567x faster |
| 32-core process-union affinity | 1925.425-1967.406ms | 16.482 | 71.96% lower |

All eight toolkit cases passed, and the bound summaries report the expected
effective affinities for all four disjoint pools. Under the current toolkit
process-union semantics, OS-default scheduling is decisively faster for 32
concurrent RoboCasa environments on this machine.

The existing single-environment affinity artifact also does not establish a
large binding win. In
`logs/latency_balance_eval/robocasa_single_env_affinity_pnpcountertocab_1_2_4_8.json`,
all cases are affinity-restricted; there is no OS-default case. Its reported
step means are 44.49ms for one core and 43.91ms for eight cores, a 1.3%
difference with run-to-run standard deviations around 8ms.

### 32-Environment Conclusion

The 32-env data strengthens the original explanation:

1. Process-level CPU affinity is active and verifiable from parent and child
   affinity logs, but it does not reduce the measured RoboCasa child-step
   latency in this workload.
2. End-to-end time is controlled by the slowest rank and channel waits, not the
   sum or average of all child-step durations.
3. Fixed core pools can introduce rank imbalance, NUMA splits, helper-thread
   contention, or outright CPU oversubscription. These costs can erase a local
   simulator benefit before it reaches the rollout critical path.
4. The current toolkits process-union benchmark does not provide evidence of a
   large binding benefit for 32 envs. In the matching 4x8 layout it shows a
   large regression, while the single-env core-count sweep shows only a small,
   noisy difference and lacks an OS-default baseline.

The defensible result is therefore narrower than "binding is effective": CPU
binding can be effective for specific high-concurrency scheduling schemes, as
seen in the historical 64-env `PnPCounterToCab` chunk-loop experiments, but the
benefit is not reproduced by the current 32-env `CloseDrawer` process-pool or
toolkits union-affinity tests. Any future claim should compare identical tasks,
environment counts, GPU layouts, reset behavior, and affinity semantics, with
at least three paired repeats.
