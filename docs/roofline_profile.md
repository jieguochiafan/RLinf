# VLA Roofline Profiling — Spec for Implementation

## Goal

Produce a **roofline scatter plot** that visualizes the low hardware utilization of VLA (Vision-Language-Action) models during embodied RL rollout and training. The figure will replace/augment `figs/sm_occupancy_cdf.pdf` in `weijun/Background.tex` §Limitation 1.

The story the figure must tell:
1. Most VLA kernels sit **far below both the compute roof and the memory roof**.
2. **Generation** (especially action-head denoising) clusters in the **low-AI, memory-bound** region.
3. **Training** kernels reach higher AI but still fall short of the compute roof.
4. Compared to an optimized LLM decode reference, VLA points are visibly lower — supporting the claim in §L159–161 that VLA lacks kernel-level optimization.

---

## 1. Figure Spec

| Item | Value |
|---|---|
| X-axis | Arithmetic Intensity = FLOPs / DRAM Bytes  (log scale, FLOP/Byte) |
| Y-axis | Achieved Performance = FLOPs / kernel_time  (log scale, TFLOP/s) |
| Compute roof | Horizontal line at hardware peak (e.g. A100 FP16 TensorCore = 312 TFLOP/s; H100 FP16 TC = 989 TFLOP/s) |
| Memory roof | Sloped line `y = peak_BW * x` (e.g. A100 HBM = 1555 GB/s; H100 HBM3 = 3350 GB/s) |
| Ridge point | Intersection of the two roofs; right = compute-bound, left = memory-bound |
| Point groups | (a) VLM prefill GEMM/attn, (b) VLM decode, (c) Action-head denoising, (d) Training fwd, (e) Training bwd, (f) [optional] LLM decode reference |
| Point size | Proportional to `sum(kernel_time)` over the run (so time-dominant kernels are visually prominent) |
| Point color | By group |
| Annotations | Ridge point marker; label the top-3 time-dominant kernels in each group |

Output: `figs/vla_roofline.pdf` (and `.svg` for editing).

---

## 2. Data to Collect via `torch.profiler`

### 2.1 Profiler setup

```python
import torch
from torch.profiler import profile, ProfilerActivity, record_function, schedule

prof_schedule = schedule(wait=1, warmup=3, active=10, repeat=1)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=prof_schedule,
    record_shapes=True,      # required for shape/bytes estimation
    with_flops=True,         # required for FLOPs (mm/bmm/conv/attn only)
    profile_memory=True,
    with_stack=True,
    on_trace_ready=lambda p: p.export_chrome_trace(f"traces/trace_{p.step_num}.json"),
) as prof:
    for step in range(20):
        with record_function("vlm_prefill"):
            ...  # VLM forward on image+lang tokens
        with record_function("vlm_decode"):
            ...  # if applicable
        with record_function("action_head_denoise"):
            for t in range(num_denoise_steps):
                ...  # each denoise step
        # For training pass:
        with record_function("train_fwd"):
            ...
        with record_function("train_bwd"):
            loss.backward()
        prof.step()
```

**Important**: wrap each logical stage with `record_function(...)` so every child CUDA kernel inherits a stage tag — this is what enables grouping in the roofline.

### 2.2 Fields to extract per CUDA kernel event

From the Kineto trace JSON (chrome trace format), for every `cat == "kernel"` event:

| Field | Source | Purpose |
|---|---|---|
| `name` | event.name | kernel identifier (e.g. `ampere_fp16_s16816gemm_...`) |
| `dur` (μs) | event.dur | denominator for FLOP/s and bytes/s |
| `args.stream` | event.args.stream | to distinguish concurrent streams |
| parent op name | walk up via `External id` / `Ev Idx` | e.g. `aten::mm`, `aten::_scaled_dot_product_flash_attention` |
| `input_shapes`, `input_dims`, `input_types` | parent op args (needs `record_shapes=True`) | for byte estimation |
| `flops` | parent op args (needs `with_flops=True`) | FLOP count; **only populated for mm/bmm/conv/attn** |
| `stage_tag` | nearest ancestor `record_function` label | grouping (prefill / decode / denoise / train_fwd / train_bwd) |

**Recommended path**: use `prof.key_averages(group_by_input_shape=True, group_by_stack_n=0)` for aggregated stats, and parse the chrome trace for per-kernel timing + stage tag.

### 2.3 Runs / sweep

Collect the same measurement under multiple configurations so the scatter tells a story:

| Sweep | Values | Rationale |
|---|---|---|
| Batch size | 1, 8, 32, 128 | Show AI shifts right but never reaches ridge (supports §L156 batch curve) |
| Denoising steps | typical value (e.g. 10) | Action-head kernels dominate step count |
| Model | OpenPI 3B (primary); optionally π₀-lite | Show size effect |
| Precision | model-native (e.g. bf16) | Match production; note peak accordingly |
| Reference | Llama-7B decode (batch=1, batch=32) | Anchor for "well-optimized" points |

Save each run's aggregated CSV separately, then combine at plot time.

---

## 3. Computing FLOPs and Bytes

### 3.1 FLOPs

- Prefer profiler-provided `flops` field (populated for `aten::mm`, `aten::bmm`, `aten::addmm`, `aten::_scaled_dot_product_flash_attention`, `aten::conv2d`).
- For ops without profiler FLOPs (LayerNorm, SiLU, elementwise), fall back to shape-based formulas:
  - `LayerNorm`: `~8 * numel(input)` FLOPs
  - `SiLU / GELU`: `~4 * numel(input)` FLOPs
  - Elementwise unary: `~1 * numel(input)`
  - Elementwise binary: `~1 * numel(output)`
- If a kernel cannot be attributed to a known op, mark it `flops=NaN` and exclude from the plot (but include its time in the group-total for sanity).

### 3.2 DRAM Bytes — two-tier approach

**Tier A (default, cheap): shape-based estimate**
```
bytes_estimate = sum(numel(t) * dtype_size(t) for t in inputs)
               + sum(numel(t) * dtype_size(t) for t in outputs)
```
- Exact for elementwise / norm / softmax.
- Lower bound for GEMM (assumes no re-load — real bytes ≥ estimate).
- Enough to show relative positioning; document the assumption in the caption.

**Tier B (calibration, ~10–20 kernels): Nsight Compute**
```
ncu --set full \
    --section SpeedOfLight_RooflineChart \
    --section MemoryWorkloadAnalysis \
    --replay-mode application \
    --launch-skip 100 --launch-count 20 \
    --csv --log-file ncu.csv \
    python run_vla_single_step.py
```
Extract:
- `dram__bytes.sum` → real DRAM bytes
- `sm__cycles_active.avg.pct_of_peak_sustained_elapsed` → real compute utilization
- `achieved_occupancy`

Use ncu numbers to compute a per-kernel-class correction factor and apply it to the Tier-A estimate for the full run. Report both in the paper (e.g. "shape-based AI, calibrated via ncu on N representative kernels").

### 3.3 Achieved TFLOP/s
```
achieved_flops_per_s = flops / (cuda_time_us * 1e-6) / 1e12   # TFLOP/s
```

### 3.4 Arithmetic Intensity
```
AI = flops / bytes    # FLOP / Byte
```

---

## 4. Data Processing Pipeline

Suggested file layout:

```
profile/
├── run.py                 # one entry point per (stage, batch, model) config
├── configs/               # yaml files: {stage, batch, model, num_denoise, ...}
├── traces/                # raw chrome_trace_*.json (one per config)
├── ncu/                   # raw ncu csvs
├── parse_trace.py         # trace.json -> per-kernel csv
├── flops_bytes.py         # add flops/bytes columns (Tier A) + ncu calibration (Tier B)
├── aggregate.py           # per-kernel rows -> per-(kernel,stage) aggregates
├── plot_roofline.py       # scatter plot
└── out/
    ├── per_kernel_<config>.csv
    ├── aggregated.csv
    └── vla_roofline.pdf
```

### 4.1 `parse_trace.py`
- Load chrome trace JSON.
- Build a stack of `record_function` events (they are `cat=="user_annotation"` or `cat=="cpu_op"`).
- For each `cat=="kernel"` event, find the enclosing `record_function` on the CPU side (by external_id link, not just by time overlap) to assign `stage_tag`.
- For each kernel, find its parent aten op (usually same external_id) → pull `input_shapes`, `input_types`, `flops`.
- Emit rows: `(config, stage_tag, op_name, kernel_name, input_shapes, dtype, flops, dur_us)`.

### 4.2 `flops_bytes.py`
- Fill missing `flops` via shape-based formulas (see §3.1).
- Compute `bytes_est` via shape-based formula (§3.2).
- Left-join `ncu.csv` on kernel_name to get `bytes_real` where available; compute per-op-class correction ratio; apply to remaining rows.
- Emit `bytes_final`, `AI = flops / bytes_final`, `achieved_tflops = flops / dur / 1e12`.

### 4.3 `aggregate.py`
- Group by `(stage_tag, kernel_name, input_shape_bucket)` where `input_shape_bucket` collapses variable batch-dim to a canonical bucket. This prevents thousands of near-duplicate points.
- For each group: mean AI, mean achieved_tflops, sum(dur_us) as `total_time`.
- Also keep the top-K time-dominant groups per stage for annotation.

### 4.4 `plot_roofline.py`
- Read `aggregated.csv`.
- Compute roofs from `--peak-tflops` and `--peak-bw-gbs` CLI args (make hardware explicit).
- Log-log axes.
- Scatter: color by `stage_tag`, size by `total_time` (e.g. `s = sqrt(total_time_us) * k`).
- Draw roofs; mark ridge point.
- Annotate top-3 kernels per stage.
- Save PDF + SVG.

---

## 5. Sanity Checks (do before drawing)

1. **Time coverage**: `sum(kernel dur) / wall_time_of_stage` should be > 0.7 for GPU-bound stages. If not, big idle gaps → either your instrumentation missed kernels or the stage is CPU-blocked (which is itself a finding, note it).
2. **FLOPs coverage**: fraction of kernels with non-null FLOPs. Aim > 80% of total time. If low, extend shape-based fallback list.
3. **No point above roof**: any point above the compute or memory roof indicates a bug (usually double-counted FLOPs or wrong dtype in peak). Fix before publishing.
4. **Ridge point plausibility**: for A100 FP16 TC, ridge point ≈ 312 / 1.555 ≈ 200 FLOP/Byte. Sanity-check against this.
5. **Determinism**: profiler adds overhead; a warmup phase (first N steps discarded via `schedule(wait,warmup,active)`) is mandatory or GPU frequency ramp / cudnn autotune will pollute the first few kernels.

---

## 6. Deliverables

The implementing agent should produce:

1. `profile/run.py` + configs — a reproducible script that, given a config, runs the target VLA on synthetic or replayed inputs with the profiler enabled.
2. Raw traces under `profile/traces/`.
3. `profile/out/aggregated.csv` with columns:
   `config, stage, kernel_name, op_name, shape_bucket, flops, bytes, AI, achieved_tflops, total_time_us, dtype`.
4. `figs/vla_roofline.pdf` and `figs/vla_roofline.svg`.
5. A short `profile/README.md` documenting hardware used (GPU, driver, CUDA, torch versions), peak numbers assumed, and calibration factors.

---

## 7. Caption Draft (for the paper)

> **Figure X. Roofline analysis of VLA execution on <GPU>.** Each point is one CUDA kernel class, positioned by its arithmetic intensity (x) and achieved throughput (y); point size encodes cumulative time. Generation kernels — especially action-head denoising steps — cluster in the low-intensity, memory-bound region and stay one to two orders of magnitude below the memory roof. Training kernels reach higher intensity but still fall well short of the compute roof. For reference, an optimized Llama-7B decode reaches close to the memory roof. This gap quantifies the kernel-level inefficiency discussed in §Limitation 1.
