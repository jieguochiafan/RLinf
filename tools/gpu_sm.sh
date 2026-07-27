#!/usr/bin/env bash
# GPU utilization sampler.
#   nvidia-smi query -> safe to run CONCURRENTLY with torch.profiler (no PerfWorks).
#   dcgmi dmon       -> finer SM_ACTIVE/SM_OCCUPANCY, but uses PerfWorks:
#                       DO NOT run on the same GPU as torch.profiler at the same time.
#
# Usage:
#   ./gpu_sm.sh -o OUTDIR [-i "5,6,7"] [-t INTERVAL_SEC] [-d DURATION_SEC] [--dcgm]
set -uo pipefail

INTERVAL=0.1
DURATION=0
GPUS=""
OUTDIR="gpu_sm_$(date +%Y%m%d_%H%M%S)"
USE_DCGM=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o) OUTDIR="$2"; shift 2;;
    -i) GPUS="$2"; shift 2;;
    -t) INTERVAL="$2"; shift 2;;
    -d) DURATION="$2"; shift 2;;
    --dcgm) USE_DCGM=1; shift;;
    *) shift;;
  esac
done
mkdir -p "$OUTDIR"
echo "Output dir : $OUTDIR  (GPUs=${GPUS:-all}, interval=${INTERVAL}s, dcgm=$USE_DCGM)"

ISEL=""; [[ -n "$GPUS" ]] && ISEL="-i $GPUS"

# nvidia-smi query mode supports millisecond sampling; utilization.gpu is GPU busy %, not SM occupancy.
MS=$(awk "BEGIN{print int($INTERVAL*1000)}")
if [[ "$MS" -le 0 ]]; then
  MS=100
fi
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory \
  --format=csv,noheader,nounits -lms "$MS" $ISEL \
  > "$OUTDIR/gpu_util_10hz.csv" 2>&1 &
SMI_PID=$!
echo "nvidia-smi query (${MS}ms) -> $OUTDIR/gpu_util_10hz.csv"

DCGM_PID=""
if [[ "$USE_DCGM" -eq 1 ]]; then
  if command -v dcgmi >/dev/null 2>&1; then
    MS=$(awk "BEGIN{print int($INTERVAL*1000)}")
    GSEL=""; [[ -n "$GPUS" ]] && GSEL="-i $GPUS"
    # 1002 SM_ACTIVE, 1003 SM_OCCUPANCY, 1004 TENSOR_ACTIVE
    dcgmi dmon $GSEL -e 1002,1003,1004 -d "$MS" > "$OUTDIR/gpu_dcgm.txt" 2>&1 &
    DCGM_PID=$!
    echo "dcgmi dmon (PerfWorks) -> $OUTDIR/gpu_dcgm.txt"
    echo "  -> reminder: do NOT run torch.profiler on these GPUs during this run."
  else
    echo "WARN: --dcgm set but dcgmi not found; nvidia-smi only."
  fi
fi

cleanup() {
  kill "$SMI_PID" 2>/dev/null || true
  [[ -n "$DCGM_PID" ]] && kill "$DCGM_PID" 2>/dev/null || true
  echo "Saved to $OUTDIR"
}
trap cleanup EXIT INT TERM

if [[ "$DURATION" -gt 0 ]]; then
  sleep "$DURATION"
else
  echo "Sampling... press Ctrl-C when the epoch finishes."
  wait "$SMI_PID"
fi
