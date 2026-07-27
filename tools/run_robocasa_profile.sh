#!/usr/bin/env bash
# One-shot profiling run for robocasa+openpi (RLinf, collocated baseline).
# Place this next to cpu_probe.sh and gpu_sm.sh, then:
#   sudo sysctl kernel.perf_event_paranoid=0      # in another shell, once, for IPC
#   bash run_robocasa_profile.sh [OUTPUT_DIR]
#   sudo sysctl kernel.perf_event_paranoid=4      # restore afterwards
set -uo pipefail

# ---- config (edit if paths differ) ----
RLINF=/data1/miliang/RLinf
VENV=$RLINF/robocasa_openpi/bin/activate
GPUS_PHYS="0,1,2,3,4,5,6,7" # physical GPU ids for nvidia-smi query
SAMPLE_INT=0.1             # sampler interval (s)
CONFIG_NAME=${ROBOCASA_CONFIG_NAME:-robocasa_realA_profile_openpi}
# ----------------------------------------

PROBE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=${1:-$RLINF/profile_data/run_$(date +%Y%m%d_%H%M%S)}
if [[ $# -gt 0 ]]; then
  shift
fi
mkdir -p "$OUT"/{cpu,gpu,torch}
echo "Output base: $OUT"

# perf permission check (this script cannot self-sudo)
PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo 4)
if [[ "$PARANOID" -gt 0 ]]; then
  echo "WARNING: perf_event_paranoid=$PARANOID -> IPC will be empty."
  echo "  Run in another shell:  sudo sysctl kernel.perf_event_paranoid=0"
  echo "  (frequency + GPU still collected). Continuing in 5s..."
  sleep 5
fi

# shellcheck disable=SC1090
source "$VENV"
cd "$RLINF"
export EMBODIED_PATH=$RLINF/examples/embodiment
export PYTHONPATH=$RLINF:${PYTHONPATH:-}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export ROBOT_PLATFORM=LIBERO
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RLINF_TORCH_PROFILE=${RLINF_TORCH_PROFILE:-1}
export RLINF_TORCH_PROFILE_DIR=$OUT/torch

# start full-epoch samplers (background)
bash "$PROBE_DIR/gpu_sm.sh"   -o "$OUT/gpu" -i "$GPUS_PHYS" -t "$SAMPLE_INT" & GPU_PID=$!
bash "$PROBE_DIR/cpu_probe.sh" -o "$OUT/cpu" -i "$SAMPLE_INT"               & CPU_PID=$!
python "$PROBE_DIR/worker_cpu_core_util.py" -o "$OUT/cpu" -i "$SAMPLE_INT" & WORKER_CPU_PID=$!
echo "samplers started (gpu=$GPU_PID cpu=$CPU_PID worker_cpu=$WORKER_CPU_PID); warming up 2s..."
sleep 2

# run training in foreground (torch.profiler runs inside actor worker)
TRAIN_STATUS=0
python examples/embodiment/train_embodied_agent.py \
  --config-path "$RLINF/examples/embodiment/config" \
  --config-name "$CONFIG_NAME" \
  runner.logger.log_path="$OUT/logs" \
  runner.logger.logger_backends=[] \
  runner.max_epochs=1 runner.max_steps=1 \
  runner.val_check_interval=-1 runner.save_interval=-1 \
  "$@" \
  2>&1 | tee "$OUT/train.log" || TRAIN_STATUS=${PIPESTATUS[0]}

# stop samplers (SIGTERM triggers their cleanup/flush)
kill "$GPU_PID" "$CPU_PID" "$WORKER_CPU_PID" 2>/dev/null || true
sleep 1
if [[ "${RLINF_TORCH_PROFILE:-0}" == "1" ]]; then
  python "$PROBE_DIR/summarize_torch_phase_sm.py" "$OUT/torch" \
    -o "$OUT/torch/env_generation_sm_summary.csv" || true
fi
echo ""
echo "DONE. Artifacts in $OUT"
echo "  cpu/cpu_freq.csv   cpu/cpu_ipc.csv"
echo "  cpu/worker_cpu_core_util.csv   cpu/worker_cpu_cores.csv"
echo "  gpu/gpu_util_10hz.csv"
echo "  torch/rank*/ torch/rollout_rank*/ torch/env_rank*/ (*.pt.trace.json + op_summary.txt + time_anchor.json, only when profiler is enabled)"
echo "  torch/env_generation_sm_summary.csv"
echo "  train.log"
exit "$TRAIN_STATUS"
