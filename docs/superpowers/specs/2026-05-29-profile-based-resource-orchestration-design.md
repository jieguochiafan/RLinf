# Profile-Based Resource Orchestration Design

## 1. Context and Goal

RLinf now has fine-grained resource pool support and rollout profiling tools, but
there is no end-to-end tool that profiles rollout and training stages, compares
MPS allocations, and emits a resource pool plan for a target embodiment config.

This design adds a profile-based resource orchestration toolkit that:

- profiles environment chunk stepping, model generation, and actor training
  throughput for a target embodiment YAML;
- evaluates multiple actor/rollout MPS SM allocation pairs;
- estimates rollout and training time using a shared chunk-step unit;
- selects the allocation that minimizes the estimated pipelined epoch time; and
- writes a `ResourcePoolSolver`-compatible plan JSON for
  `cluster.resource_pool.allocation_mode: plan_file`.

The first version targets MPS resource pool plans for embodied FSDP training.

## 2. Confirmed Decisions

- Add a new orchestration layer under `toolkits/resource_orchestration/`.
- Keep rollout profiling, actor training profiling, and production scheduling
  separated.
- Reuse `toolkits.rollout_eval.benchmark` for environment and model generation
  profiling.
- Reuse or implement `toolkits.training_eval` for actor training profiling on the
  real embodied FSDP update path.
- Output a report plus a new resource pool plan JSON.
- Do not automatically edit the source YAML.
- Accept arbitrary CLI candidate pairs such as `30:70,40:60,50:50`.
- Use complementary MPS pairs from `20:80` through `80:20` as the default sweep.
- Select the candidate that minimizes `max(rollout_time, training_time)`.
- Use rollout/training balance as a tie-breaker.

## 3. Scope

### In Scope

- Hydra config loading for existing embodiment YAMLs.
- MPS candidate pair expansion and validation.
- Per-candidate profiling for env, model generation, and actor training.
- Chunk-step throughput normalization.
- Pipelined rollout and actor training time estimation.
- Candidate selection and report generation.
- Plan JSON generation compatible with existing resource pool bindings.
- Unit tests for parsing, estimation, selection, plan writing, and reporting.

### Non-Goals

- No MIG allocation recommendation in v1.
- No automatic MPS daemon management.
- No automatic YAML rewriting.
- No integration into training startup.
- No change to scheduler runtime semantics.
- No Megatron or tensor-parallel training profile support in v1.

## 4. Architecture

Add `toolkits/resource_orchestration/` with these modules:

- `run.py`
  - CLI entrypoint.
  - Parses Hydra config args, MPS candidate pairs, profile controls, output paths,
    and overrides.
- `config_loader.py`
  - Loads and validates the target config.
  - Extracts placement, GPU count, actor and rollout ranks, resource pool mode,
    environment counts, rollout horizon, action chunk size, and training batch
    fields.
- `profilers.py`
  - Coordinates per-candidate profiling.
  - Calls rollout benchmark functionality for env/model measurements.
  - Calls training evaluation functionality for actor measurements.
- `estimator.py`
  - Converts raw stage metrics to chunk-step throughput.
  - Computes rollout and training time estimates.
- `selector.py`
  - Filters invalid candidates.
  - Applies the objective and tie-break rules.
- `plan_writer.py`
  - Emits a resource pool plan JSON matching existing
    `WorkerResourceBinding` schema.
- `reporting.py`
  - Writes per-candidate profile details, `summary.json`, and `summary.md`.
- `types.py`
  - Defines dataclasses for candidate pairs, stage metrics, estimates,
    selection results, and output records.

The orchestration layer owns only coordination and recommendation. It should not
duplicate environment adapters, model adapters, or actor update logic.

## 5. Data Flow

1. Load the target Hydra YAML and apply any CLI overrides.
2. Validate that `cluster.resource_pool.gpu.mode` is `mps`.
3. Determine physical GPU count and existing actor/rollout/env rank placement.
4. Expand candidate MPS pairs.
5. For each candidate:
   - apply actor and rollout SM percentages to the profile environment;
   - profile environment chunk stepping;
   - profile model generation;
   - profile actor training;
   - normalize metrics to chunk steps per second; and
   - estimate rollout time, training time, and epoch time.
6. Select the best valid candidate.
7. Write reports and the selected resource pool plan JSON.

## 6. Chunk-Step Semantics

The shared unit is `chunk_step`: one policy generation that produces one action
chunk for one environment stream.

Config-derived values:

- `chunk_size = actor.model.num_action_chunks`
- `episode_env_steps = env.train.max_steps_per_rollout_epoch`
- if `max_steps_per_rollout_epoch` is unavailable, fall back to
  `env.train.max_episode_steps`
- `chunk_steps_per_env = episode_env_steps / chunk_size`
- `rollout_chunk_count =
  env.train.total_num_envs * algorithm.rollout_epoch * chunk_steps_per_env`

The config validator already requires rollout steps to be divisible by
`actor.model.num_action_chunks`; the orchestration tool should surface a clear
error if this assumption is violated.

## 7. Profiling Metrics

### Environment

Preferred metric:

- `env_chunk_steps_per_sec`

If the reused profiler reports environment step throughput only, convert with:

- `env_chunk_steps_per_sec = env_steps_per_sec / chunk_size`

### Model Generation

Preferred metric:

- `model_chunk_steps_per_sec`

One model inference is treated as one generated action chunk.

### Actor Training

Preferred metric:

- `actor_chunk_steps_per_sec = measured_training_chunks / measured_seconds`

This throughput must include the cost of `algorithm.update_epoch`. The measured
chunk count is therefore the amount of rollout chunk work consumed by the full
training update path, including repeated PPO epochs.

Actor profiling should use the real embodied FSDP update path with synthetic
trajectory data shaped like real rollout batches. A simplified loss-only
microbenchmark is not acceptable for the primary metric.

## 8. Estimation Algorithm

Rollout is modeled as an env/model pipeline. In steady state, the slower stage is
the bottleneck:

```text
rollout_time =
    rollout_chunk_count / min(env_chunk_steps_per_sec, model_chunk_steps_per_sec)
```

Actor training time is:

```text
training_time =
    rollout_chunk_count / actor_chunk_steps_per_sec
```

The orchestration report may also record warmup and drain corrections, but v1
selection uses the steady-state formulas above.

If the existing concurrent rollout pipeline profiler reports
`pipeline_samples_per_sec`, include it in the report as a sanity-check metric.
The selected candidate should still be explained from the env/model stage
throughputs.

## 9. Selection Algorithm

For each candidate:

```text
epoch_time = max(rollout_time, training_time)
balance_gap = abs(rollout_time - training_time)
```

Selection rules:

1. Drop candidates with failed profiles, zero throughput, invalid SM values, or
   `actor_sm + rollout_sm > 100`.
2. Choose the smallest `epoch_time`.
3. If candidates are within a configurable tolerance, default `3%`, choose the
   smaller `balance_gap`.
4. If still tied, choose the higher actor SM percentage.

If no candidate is valid, the command exits non-zero and does not write a plan.

## 10. Plan JSON Output

The generated plan must be compatible with
`cluster.resource_pool.allocation_mode: plan_file`.

Plan generation rules:

- Preserve placement: do not change rank count, parent GPU, or visible devices.
- Preserve CPU binding where possible.
- Actor ranks receive `gpu.sm_percent = selected.actor_sm`.
- Rollout ranks receive `gpu.sm_percent = selected.rollout_sm`.
- Env ranks receive GPU bindings only if the target YAML already configures env
  as a GPU-bound resource pool component.
- `gpu.mode` is `mps`.
- `visible_devices` and `parent_gpu` come from the existing placement solution.
- If the input YAML already uses a plan file, `--base-plan` can be used to inherit
  CPU bindings and non-target component bindings.

The plan writer should validate its output by parsing each binding through
`WorkerResourceBinding.from_json()`.

## 11. CLI Contract

Example:

```bash
python -m toolkits.resource_orchestration.run \
  --config-path examples/embodiment/config \
  --config-name libero_10_async_ppo_openpi_pi05_resource_pool_112env_stage2 \
  --candidate-pairs 20:80,30:70,40:60,50:50,60:40,70:30,80:20 \
  --warmup-steps 5 \
  --measure-steps 20 \
  --output-dir ./resource_orchestration_output/libero_10 \
  --plan-output examples/embodiment/config/resource_pool/libero_10_auto_mps_plan.json
```

CLI options:

- `--config-path`: Hydra config directory.
- `--config-name`: target Hydra config name.
- `--override`: repeatable Hydra override.
- `--candidate-pairs`: comma-separated `actor_sm:rollout_sm` pairs.
- `--warmup-steps`: profile warmup steps.
- `--measure-steps`: measured profile steps.
- `--output-dir`: report and intermediate profile directory.
- `--plan-output`: output path for the selected resource pool plan JSON.
- `--base-plan`: optional existing plan to inherit CPU bindings from.
- `--selection-tolerance`: relative tolerance for tie-breaking, default `0.03`.

Default `--candidate-pairs`:

```text
20:80,30:70,40:60,50:50,60:40,70:30,80:20
```

## 12. Reports

Artifacts:

- `summary.json`
- `summary.md`
- `profiles/<candidate_id>.json`
- selected plan JSON at `--plan-output`

Each candidate record includes:

- candidate id
- actor SM percentage
- rollout SM percentage
- status and failure reason if applicable
- env chunk steps/s
- model chunk steps/s
- actor chunk steps/s
- optional rollout pipeline samples/s
- rollout chunk count
- estimated rollout time
- estimated training time
- estimated epoch time
- bottleneck stage

The global summary includes the selected candidate and the final plan path.

## 13. Error Handling

- A failed candidate profile is recorded and the sweep continues.
- Invalid candidate syntax or SM percentages fail fast.
- Candidate pairs with `actor_sm + rollout_sm > 100` fail validation.
- Non-MPS resource pool configs are rejected in v1.
- Missing actor training profile support is a command failure because the final
  recommendation would be incomplete.
- If all candidates fail, no plan is written.
- If plan validation fails, reports are still written, but plan output is not
  claimed as usable.

## 14. Testing Strategy

Unit tests:

- candidate pair parsing and default expansion;
- config extraction for chunk size, rollout chunk count, rank counts, and MPS
  mode;
- estimator formulas and throughput conversion;
- selector filtering, objective ordering, tolerance handling, and tie-breaks;
- plan writer output parseable by `WorkerResourceBinding.from_json()`;
- report schema contains required fields.

Lightweight integration tests:

- run the full orchestrator with stub profilers and a small Hydra config;
- assert a selected candidate is reported;
- assert a plan JSON is written and parseable;
- assert no source YAML is modified.

Manual validation:

- run on an MPS-enabled GPU host with a LIBERO/OpenPI resource pool config;
- verify all candidates produce per-stage throughput;
- verify the selected plan can be used as `allocation_plan_path` in a training
  launch.

## 15. Done Criteria

- `python -m toolkits.resource_orchestration.run` can evaluate candidate MPS
  pairs for an embodiment config.
- Reports include per-stage throughput and estimated rollout/training times.
- The selected candidate follows the objective:
  minimize `max(rollout_time, training_time)`.
- The generated plan JSON is compatible with existing resource pool plan loading.
- Unit tests cover parsing, estimation, selection, reporting, and plan writing.
- Existing rollout benchmark and scheduler behavior remain unchanged.
