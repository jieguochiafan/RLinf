#!/usr/bin/env bash
set -uo pipefail

RLINF=${RLINF:-/data1/miliang/RLinf}
VENV=${VENV:-$RLINF/robocasa_openpi/bin/activate}
CONFIG_NAME=${ROBOCASA_CONFIG_NAME:-robocasa_async_ppo_openpi_pipeline2_profile}
OUT=${RLINF_PROFILE_OUT:-$RLINF/profile_data/robocasa_async_pipeline2_profile_$(date +%Y%m%d_%H%M%S)}
if [[ $# -gt 0 && "$1" != *=* && "$1" != +* && "$1" != ~* ]]; then
  OUT=$1
  shift
fi
CPU_INTERVAL=${CPU_INTERVAL:-0.1}
CPU_REFRESH_INTERVAL=${CPU_REFRESH_INTERVAL:-0.5}
GPU_INTERVAL_MS=${GPU_INTERVAL_MS:-100}
MEM_INTERVAL=${MEM_INTERVAL:-0.2}
NUM_CPUS=${NUM_CPUS:-112}

mkdir -p "$OUT"/{cpu,gpu,torch}
ln -sfn "$OUT" "$RLINF/profile_data/robocasa_async_pipeline2_profile_latest"

# shellcheck disable=SC1090
source "$VENV"
cd "$RLINF" || exit 1

export EMBODIED_PATH=$RLINF/examples/embodiment
export PYTHONPATH=$RLINF:${PYTHONPATH:-}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export ROBOT_PLATFORM=${ROBOT_PLATFORM:-ROBOCASA}
export MPLCONFIGDIR="$OUT/matplotlib"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/rlinf_mps_miliang_20260609}
export CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-/tmp/rlinf_mps_logs_miliang_20260609}
export RLINF_ENV_CPU_CORES=${RLINF_ENV_CPU_CORES:-0-111}
export RLINF_ENV_RENDER_GPUS=${RLINF_ENV_RENDER_GPUS:-0-7}
export RLINF_TORCH_PROFILE=${RLINF_TORCH_PROFILE:-1}
export RLINF_TORCH_PROFILE_DIR=$OUT/torch
export RLINF_TRAINING_EVAL_LOCAL_RAY=1
unset RAY_ADDRESS
mkdir -p "$MPLCONFIGDIR"

STOP_FILE="$OUT/monitor.stop"
rm -f "$STOP_FILE"

python tools/highres_cpu_monitor.py \
  --output "$OUT/cpu_highres.csv" \
  --stop-file "$STOP_FILE" \
  --interval "$CPU_INTERVAL" \
  --refresh-interval "$CPU_REFRESH_INTERVAL" \
  --role-log "$OUT/train.log" \
  --num-cpus "$NUM_CPUS" \
  > "$OUT/cpu_monitor.log" 2>&1 &
CPU_MON_PID=$!

python tools/worker_memory_monitor.py \
  --output "$OUT/memory_workers.csv" \
  --stop-file "$STOP_FILE" \
  --interval "$MEM_INTERVAL" \
  --role-log "$OUT/train.log" \
  > "$OUT/memory_monitor.log" 2>&1 &
MEM_MON_PID=$!

python tools/worker_cpu_core_util.py \
  -o "$OUT/cpu" \
  -i "$CPU_INTERVAL" \
  > "$OUT/worker_cpu_core_monitor.log" 2>&1 &
WORKER_CPU_PID=$!

nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw \
  --format=csv,nounits \
  -lms "$GPU_INTERVAL_MS" \
  > "$OUT/gpu_highres.csv" 2>&1 &
GPU_MON_PID=$!

TRAIN_STATUS=0
python examples/embodiment/train_async.py \
  --config-path "$RLINF/examples/embodiment/config" \
  --config-name "$CONFIG_NAME" \
  runner.logger.log_path="$OUT/logs" \
  runner.logger.logger_backends=[] \
  runner.max_epochs=1 \
  runner.max_steps=1 \
  runner.val_check_interval=-1 \
  runner.save_interval=-1 \
  "$@" \
  2>&1 | python tools/timestamp_stream.py | tee "$OUT/train.log" || TRAIN_STATUS=${PIPESTATUS[0]}

touch "$STOP_FILE"
kill "$GPU_MON_PID" "$WORKER_CPU_PID" "$CPU_MON_PID" "$MEM_MON_PID" 2>/dev/null || true
wait "$CPU_MON_PID" 2>/dev/null || true
wait "$MEM_MON_PID" 2>/dev/null || true
wait "$WORKER_CPU_PID" 2>/dev/null || true
wait "$GPU_MON_PID" 2>/dev/null || true

if [[ "${RLINF_TORCH_PROFILE:-0}" == "1" ]]; then
  python tools/summarize_torch_phase_sm.py "$OUT/torch" \
    -o "$OUT/torch/env_generation_sm_summary.csv" \
    > "$OUT/torch/summarize_torch_phase_sm.log" 2>&1 || true
  python tools/plot_worker_sm_timeline.py "$OUT/torch" \
    --output-prefix "$OUT/torch/worker_sm_timeline" \
    > "$OUT/torch/plot_worker_sm_timeline.log" 2>&1 || true
  python tools/plot_worker_sm_cdf.py "$OUT/torch" \
    --output-dir "$OUT/torch" \
    > "$OUT/torch/plot_worker_sm_cdf.log" 2>&1 || true
fi

python tools/plot_highres_robocasa_resource.py \
  --profile-dir "$OUT" \
  --num-cpus "$NUM_CPUS" \
  --smooth-window 1 \
  --output-stem resource_utilization_highres_roles \
  > "$OUT/plot_highres_resource.log" 2>&1 || true

python tools/plot_cpu_core_util_timeline.py "$OUT" \
  --smooth-window-s 5 \
  --x-pad-s 10 \
  --bin-s 0.5 \
  > "$OUT/plot_cpu_core_util_timeline.log" 2>&1 || true

python tools/plot_cpu_gpu_util_timeline.py "$OUT" \
  --cpu-smooth-window-s 5 \
  --gpu-smooth-window-s 5 \
  --x-pad-s 10 \
  --bin-s 0.5 \
  > "$OUT/plot_cpu_gpu_util_timeline.log" 2>&1 || true

python tools/plot_worker_memory_timeline.py "$OUT" \
  > "$OUT/plot_worker_memory_timeline.log" 2>&1 || true

{
  printf "TRAIN_STATUS=%s\n" "$TRAIN_STATUS"
  printf "PROFILE_DIR=%s\n" "$OUT"
  printf "CONFIG=%s\n" "$CONFIG_NAME"
  printf "CPU_INTERVAL=%ss\n" "$CPU_INTERVAL"
  printf "GPU_INTERVAL=%sms\n" "$GPU_INTERVAL_MS"
  printf "MEM_INTERVAL=%ss\n" "$MEM_INTERVAL"
  printf "CUDA_VISIBLE_DEVICES=%s\n" "$CUDA_VISIBLE_DEVICES"
  printf "CUDA_MPS_PIPE_DIRECTORY=%s\n" "$CUDA_MPS_PIPE_DIRECTORY"
  printf "CUDA_MPS_LOG_DIRECTORY=%s\n" "$CUDA_MPS_LOG_DIRECTORY"
  printf "RLINF_ENV_CPU_CORES=%s\n" "$RLINF_ENV_CPU_CORES"
  printf "RLINF_ENV_RENDER_GPUS=%s\n" "$RLINF_ENV_RENDER_GPUS"
} | tee "$OUT/status.txt"

exit "$TRAIN_STATUS"
