#!/usr/bin/env bash
set -euo pipefail

PROFILE_DIR="${PROFILE_DIR:-/tmp/robocasa_default56_highres_50ms_$(date +%Y%m%d_%H%M%S)}"
CPU_INTERVAL="${CPU_INTERVAL:-0.05}"
CPU_REFRESH_INTERVAL="${CPU_REFRESH_INTERVAL:-0.5}"
GPU_INTERVAL_MS="${GPU_INTERVAL_MS:-50}"
NUM_CPUS="${NUM_CPUS:-112}"

mkdir -p "$PROFILE_DIR"
ln -sfn "$PROFILE_DIR" /tmp/robocasa_default56_highres_50ms_latest

source /data1/miliang/RLinf/robocasa_openpi/bin/activate
cd /data1/miliang/RLinf

export EMBODIED_PATH=/data1/miliang/RLinf/examples/embodiment
export PYTHONPATH=/data1/miliang/RLinf:${PYTHONPATH:-}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export ROBOT_PLATFORM=LIBERO
export MPLCONFIGDIR="$PROFILE_DIR/matplotlib"
mkdir -p "$MPLCONFIGDIR"

STOP_FILE="$PROFILE_DIR/monitor.stop"
rm -f "$STOP_FILE"

python tools/highres_cpu_monitor.py \
  --output "$PROFILE_DIR/cpu_highres.csv" \
  --stop-file "$STOP_FILE" \
  --interval "$CPU_INTERVAL" \
  --refresh-interval "$CPU_REFRESH_INTERVAL" \
  --role-log "$PROFILE_DIR/train.log" \
  --num-cpus "$NUM_CPUS" \
  > "$PROFILE_DIR/cpu_monitor.log" 2>&1 &
CPU_MON_PID=$!

nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw \
  --format=csv,nounits \
  -lms "$GPU_INTERVAL_MS" \
  > "$PROFILE_DIR/gpu_highres.csv" 2>&1 &
GPU_MON_PID=$!

TRAIN_STATUS=0
python examples/embodiment/train_embodied_agent.py \
  --config-path /data1/miliang/RLinf/examples/embodiment/config \
  --config-name robocasa_baseline_profile_openpi \
  runner.logger.log_path="$PROFILE_DIR/logs" \
  runner.logger.logger_backends=[] \
  runner.max_epochs=1 \
  runner.max_steps=1 \
  runner.val_check_interval=-1 \
  runner.save_interval=-1 \
  env.train.total_num_envs=56 \
  env.train.action_repeat_per_chunk_step=1 \
  algorithm.update_epoch=1 \
  actor.global_batch_size=56 \
  actor.micro_batch_size=1 \
  2>&1 | python tools/timestamp_stream.py | tee "$PROFILE_DIR/train.log" || TRAIN_STATUS=${PIPESTATUS[0]}

touch "$STOP_FILE"
kill "$GPU_MON_PID" 2>/dev/null || true
wait "$CPU_MON_PID" 2>/dev/null || true
wait "$GPU_MON_PID" 2>/dev/null || true

python tools/plot_highres_robocasa_resource.py \
  --profile-dir "$PROFILE_DIR" \
  --num-cpus "$NUM_CPUS" \
  --smooth-window 1 \
  --output-stem resource_utilization_highres_roles \
  > "$PROFILE_DIR/plot.log" 2>&1 || true

{
  printf "TRAIN_STATUS=%s\n" "$TRAIN_STATUS"
  printf "PROFILE_DIR=%s\n" "$PROFILE_DIR"
  printf "CONFIG=robocasa_baseline_profile_openpi default scheduling, 56 envs, no CPU core limit, normal rollout/training, CPU_INTERVAL=%ss, GPU_INTERVAL=%sms\n" "$CPU_INTERVAL" "$GPU_INTERVAL_MS"
} | tee "$PROFILE_DIR/status.txt"

exit "$TRAIN_STATUS"
