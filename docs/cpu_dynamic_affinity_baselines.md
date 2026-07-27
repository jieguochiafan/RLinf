# CPU Dynamic Affinity Baselines

This note summarizes the CPU affinity baselines currently kept for env chunk
execution experiments. The retained set is intentionally small: two baselines
and one dynamic scheduling method.

## Retained Configurations

| Method | CPU binding | Dynamic scheduling | Chunk mode | Purpose |
| --- | --- | --- | --- | --- |
| No CPU binding, no scheduling | Disabled | Disabled | `sync_time_major` | Default baseline without env subprocess affinity constraints. |
| Static CPU binding, no scheduling | Enabled | Disabled | `sync_time_major` | Measures the effect of fixed env-to-core binding alone. |
| Static CPU binding + core donation v2 | Enabled | Enabled | `latency_balanced_pair` | Reassigns idle env core groups to unfinished envs during a chunk. |

## 1. No CPU Binding, No Scheduling

This is the default baseline. Env subprocesses are not pinned to dedicated CPU
core groups, and chunk execution follows the normal time-major path.

Expected configuration shape:

```yaml
env:
  train:
    chunk_step_mode: sync_time_major
```

Do not enable the env CPU resource pool for this baseline.

## 2. Static CPU Binding, No Scheduling

This baseline enables the env CPU resource pool so each env subprocess gets a
fixed CPU core group. There is no dynamic rebinding or core donation during
chunk execution.

Expected configuration shape:

```yaml
env:
  train:
    chunk_step_mode: sync_time_major
```

Enable env CPU resource allocation through the cluster/resource-pool config.
This isolates the effect of static CPU affinity from the scheduling algorithm.

## 3. Static CPU Binding + Core Donation V2

This is the retained dynamic scheduling method. It uses latency-balanced pair
chunk execution with `core_donation_v2`: when some envs finish their chunk work
earlier, their CPU core groups can be temporarily donated to still-running envs.
The affinities are restored after the chunk finishes.

Expected configuration shape:

```yaml
env:
  train:
    chunk_step_mode: latency_balanced_pair
    latency_balanced_pair:
      dynamic_affinity: true
      rebalance_mode: core_donation_v2
```

Enable env CPU resource allocation for this method. Without per-env CPU core
groups, core donation has no resource groups to donate.

## Removed Experimental Variants

The following variants were explored but are no longer part of the kept
comparison set:

- `envs_per_core=2`
- `latency_balanced_pair` with `dynamic_affinity: false`
- dynamic pairing variants that only rebind cores without core donation
- unbound-CPU dynamic scheduling variants

## Recommended Comparison

Use the three retained methods above as the main comparison:

1. No CPU binding, no scheduling.
2. Static CPU binding, no scheduling.
3. Static CPU binding with `core_donation_v2`.

This separates the benefit of CPU pinning itself from the additional benefit of
dynamic core donation.
