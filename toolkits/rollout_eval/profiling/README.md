# GR00T generation stage profiling

`gr00t_generation_profile.py` isolates one GR00T rollout inference call and
profiles the same phase boundaries used by the production worker:

- CPU observation preprocessing;
- VLM backbone;
- action head, including every denoising step and the value head;
- CPU action postprocessing.

The default batch size is 12, matching a 192-environment, 8-rollout-worker,
2-pipeline-stage run. The default MPS active thread percentage is 80. Random
LIBERO-shaped observations make the run deterministic and remove simulator
noise while preserving model tensor shapes.

Run the lightweight CUDA-event measurement with the GR00T environment:

```bash
/data1/gaobowen/RLinf/.venv-libero-gr00t/bin/python -m \
  toolkits.rollout_eval.profiling.gr00t_generation_profile \
  --gpu 0 \
  --mps-sm 80 \
  --batch-size 12 \
  --warmup-steps 5 \
  --measure-steps 10
```

The command starts an isolated MPS daemon by default. Pass `--no-manage-mps`
to connect to an existing daemon.

For a kernel/operator breakdown, add `--torch-profiler`. This produces
`torch_trace.json` and `torch_operators.txt` in the output directory.

For Nsight Systems GPU metrics aligned with the named NVTX phases, add
`--nsys` instead:

```bash
/data1/gaobowen/RLinf/.venv-libero-gr00t/bin/python -m \
  toolkits.rollout_eval.profiling.gr00t_generation_profile \
  --gpu 0 \
  --mps-sm 80 \
  --batch-size 12 \
  --warmup-steps 5 \
  --measure-steps 10 \
  --nsys
```

Do not combine `--torch-profiler` and `--nsys`; both use CUPTI. The JSON and
Markdown summaries report wall latency and CUDA-event stream span. CUDA-event
span includes gaps between kernel launches. Use the PyTorch trace or the
Nsight report to measure kernel busy time and utilization within each phase.
