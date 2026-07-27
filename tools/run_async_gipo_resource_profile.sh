#!/usr/bin/env bash
set -uo pipefail

RLINF=/data1/miliang/RLinf
RUN_DIR=${RUN_DIR:-/tmp/rlinf_async_gipo_resource_$(date +%Y%m%d_%H%M%S)}
CPU_INTERVAL=${CPU_INTERVAL:-0.1}
PROFILE_ACTIVE_STEPS=${PROFILE_ACTIVE_STEPS:-25}
TRAJECTORIES_PER_TRAIN=${TRAJECTORIES_PER_TRAIN:-16}

mkdir -p "$RUN_DIR/resource_profile/cpu" "$RUN_DIR/resource_profile/torch"

set +u
source "$RLINF/libero_openpi/bin/activate"
set -u
cd "$RLINF"

export EMBODIED_PATH=$RLINF/examples/embodiment
export PYTHONPATH=$RLINF:${PYTHONPATH:-}
export LIBERO_TYPE=standard
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export ROBOT_PLATFORM=LIBERO
export RLINF_TRAINING_EVAL_LOCAL_RAY=1
export RLINF_RESOURCE_PROFILE=1
export RLINF_TORCH_PROFILE=1
export RLINF_TORCH_PROFILE_DIR=$RUN_DIR/resource_profile/torch
export RLINF_TORCH_PROFILE_WAIT=0
export RLINF_TORCH_PROFILE_WARMUP=0
export RLINF_TORCH_PROFILE_ACTIVE=$PROFILE_ACTIVE_STEPS
export RLINF_TORCH_PROFILE_REPEAT=0
export MPLCONFIGDIR=$RUN_DIR/matplotlib

STOP_FILE=$RUN_DIR/resource_profile/cpu/monitor.stop
rm -f "$STOP_FILE"
python tools/rlinf_cpu_core_monitor.py \
  --output "$RUN_DIR/resource_profile/cpu/thread_core_samples.csv" \
  --stop-file "$STOP_FILE" \
  --interval "$CPU_INTERVAL" \
  > "$RUN_DIR/resource_profile/cpu/monitor.log" 2>&1 &
CPU_MONITOR_PID=$!

cleanup() {
  touch "$STOP_FILE"
  kill "$CPU_MONITOR_PID" 2>/dev/null || true
  wait "$CPU_MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

TRAIN_STATUS=0
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python examples/embodiment/train_async.py \
  --config-path "$RLINF/examples/embodiment/config" \
  --config-name libero_spatial_async_gipo_openpi_pi05_verify \
  runner.max_epochs=8 \
  runner.max_steps=8 \
  runner.val_check_interval=-1 \
  runner.save_interval=-1 \
  runner.logger.logger_backends=[] \
  runner.logger.log_path="$RUN_DIR" \
  env.train.v17_continuous_collect.target_trajectories="$TRAJECTORIES_PER_TRAIN" \
  algorithm.replay_buffer.min_buffer_size="$TRAJECTORIES_PER_TRAIN" \
  algorithm.replay_buffer.cache_size="$TRAJECTORIES_PER_TRAIN" \
  algorithm.replay_buffer.sample_window_size="$TRAJECTORIES_PER_TRAIN" \
  +actor.recv_drain_max_trajectories="$TRAJECTORIES_PER_TRAIN" \
  "$@" \
  2>&1 | python tools/timestamp_stream.py | tee "$RUN_DIR/train.log"
PIPE_STATUSES=("${PIPESTATUS[@]}")
TRAIN_STATUS=${PIPE_STATUSES[0]}

cleanup
trap - EXIT INT TERM

PROFILE_STATUS=0
python tools/async_gipo_resource_profile.py "$RUN_DIR" --bin-s 1.0 --num-cpus 112
STATUS=$?
if [[ "$STATUS" -ne 0 ]]; then
  PROFILE_STATUS=$STATUS
fi

python tools/plot_async_gipo_resource_utilization.py \
  "$RUN_DIR/resource_profile/derived" \
  --output-prefix "$RUN_DIR/resource_profile/resource_utilization" \
  --num-cpus 112
STATUS=$?
if [[ "$PROFILE_STATUS" -eq 0 && "$STATUS" -ne 0 ]]; then
  PROFILE_STATUS=$STATUS
fi

if [[ "$TRAIN_STATUS" -ne 0 ]]; then
  exit "$TRAIN_STATUS"
fi
exit "$PROFILE_STATUS"
