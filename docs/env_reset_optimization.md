# Environment Reset Optimization

This note summarizes the LIBERO and RoboCasa reset latency optimizations used for embodied RL rollout.

## Goals

- Avoid rebuilding simulator assets when the task has not changed.
- Keep full reset available as the correctness fallback.
- Make the optimized path explicit through config, so default behavior remains conservative.

## Config Switches

### Global switch

```yaml
reset_optimization_enabled: true
```

Default: `false`.

When disabled, RLinf forces full reset behavior even if `reset_mode` is set. Enable this only for envs where the optimized reset semantics have been validated.

### Reset mode

```yaml
reset_mode: task_aware
```

Supported values:

- `full`: always use full reset semantics.
- `state`: require state reset; can fall back to full on mismatch depending on `reset_full_on_state_mismatch`.
- `task_aware`: use full reset when task/assets change, otherwise use optimized state reset.

Recommended training value: `task_aware`.

### LIBERO task-affine sampling

```yaml
reset_sampling_strategy: task_affine_fixed
```

Default: `random`.

With `task_affine_fixed`, each env/group is assigned a fixed task and reset samples only trials/init states from that task. This avoids frequent task switches from global random reset-state sampling.

## LIBERO Behavior

When optimization is enabled and `reset_mode: task_aware`:

- If task/BDDL changes, LIBERO performs full reset.
- If task/BDDL is unchanged, LIBERO uses soft reset plus `set_init_state`, avoiding MuJoCo sim/render-context reconstruction.

For multi-task LIBERO training, `task_affine_fixed` is important. Without it, global random reset-state sampling can switch tasks on most resets, forcing full reset and reducing the benefit of `task_aware`.

Example:

```yaml
env:
  train:
    reset_optimization_enabled: true
    reset_mode: task_aware
    reset_sampling_strategy: task_affine_fixed
```

Current `examples/embodiment/config/env/libero_spatial.yaml` includes:

```yaml
reset_optimization_enabled: true
reset_mode: task_aware
reset_sampling_strategy: task_affine_fixed
```

## RoboCasa Behavior

Current RLinf RoboCasa envs do not change task on reset. The task is assigned when the subprocess env is created.

When optimization is enabled:

```yaml
reset_optimization_enabled: true
reset_mode: task_aware
```

RoboCasa maps this to `hard_reset=False`, reusing simulator assets across resets.

Current `examples/embodiment/config/env/robocasa_closedrawer.yaml` includes this optimized mode.

### Low-cost reset randomization

RoboCasa's optimized reset reuses the compiled MuJoCo model, but it can still
vary physical initial state on every reset:

```yaml
reset_randomization:
  enabled: true
  drawer_open_range: [0.65, 1.0]
  resample_object_placements: true
  placement_sampling_attempts: 3
```

`drawer_open_range` overrides the task's drawer state during its normal reset.
For tasks with movable objects, `resample_object_placements` samples new poses
from the existing placement initializer. Neither operation rebuilds the MuJoCo
model or render context. Layout, style, fixture selection, and object identity
remain fixed until a full reset.

## Measured Results

### LIBERO

Direct benchmark, `libero_spatial`, 8 envs, 3 epochs, `settle_steps=1`:

| Mode | Mean Reset Time |
| --- | ---: |
| full | 7.07 s |
| task-aware optimized | 0.86 s |

Estimated saving: about 87.8%.

With default `settle_steps=15`, the optimized reset still helps, but settle time dominates more of the remaining latency.

### RoboCasa

Direct benchmark, `CloseDrawer`, 8 env seeds, 3 epochs:

| Mode | Mean Reset Time | p95 |
| --- | ---: | ---: |
| full (`hard_reset=True`) | 7.84 s | 10.53 s |
| task-aware optimized (`hard_reset=False`) | 0.60 s | 0.83 s |

Estimated saving: about 92.4%.

## Rollout-Level Impact

During training, rollout rank 0 shows the reset time received with each epoch's
bootstrap batch in the rollout progress bar:

```text
Generating Rollout Epochs: 12%|...| 1/8 [01:34<..., 94.38s/it, reset=47.08s]
```

The postfix is reported whether reset optimization is enabled or disabled. When
multiple environment workers feed rollout rank 0, it reports the maximum reset
time among those mapped workers.

LIBERO spatial simulation, 8 envs, 64 rollout epochs:

| Sampling Strategy | Full Reset Ratio | State Reset Ratio |
| --- | ---: | ---: |
| global random reset-state sampling | 89.1% | 10.9% |
| `task_affine_fixed` | 0% | 100% |

This is why `task_affine_fixed` is recommended for multi-task LIBERO training when reset latency matters.

## Caveats

- `task_affine_fixed` fixes each env/group to a task. If the number of env groups is smaller than the number of tasks, not all tasks are covered at once.
- For full task coverage with fewer envs, add a future rotation strategy, for example rotating fixed task assignments every N rollout epochs.
- Evaluation configs should enable these options only when fixed-task sampling is intended.
- Cross-task state reset is not considered safe for LIBERO; task/BDDL mismatch must still use full reset.

## Quick Enablement

LIBERO training:

```yaml
env:
  train:
    reset_optimization_enabled: true
    reset_mode: task_aware
    reset_sampling_strategy: task_affine_fixed
```

RoboCasa training:

```yaml
env:
  train:
    reset_optimization_enabled: true
    reset_mode: task_aware
```
