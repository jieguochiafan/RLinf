#!/usr/bin/env bash
# CPU fine-grained probe: per-core frequency (sysfs, no root) + IPC (perf).
# Start this RIGHT AFTER the RLinf training process is launched.
#
# Usage:
#   ./cpu_probe.sh -o OUTDIR [-i INTERVAL_SEC] [-d DURATION_SEC] \
#                  [-p "PID1,PID2"] [-n PROC_PATTERN]
#   -p  explicit comma-separated PIDs to attach perf to
#   -n  pattern to resolve PIDs via pgrep -f (e.g. "rlinf" or "train")
#   if neither given, perf runs system-wide (-a) and captures all tenants.
set -uo pipefail

INTERVAL=1
DURATION=0          # 0 = run until Ctrl-C
OUTDIR="cpu_probe_$(date +%Y%m%d_%H%M%S)"
PIDS=""
PATTERN=""

while getopts "o:i:d:p:n:" opt; do
  case $opt in
    o) OUTDIR="$OPTARG" ;;
    i) INTERVAL="$OPTARG" ;;
    d) DURATION="$OPTARG" ;;
    p) PIDS="$OPTARG" ;;
    n) PATTERN="$OPTARG" ;;
  esac
done
mkdir -p "$OUTDIR"

if [[ -z "$PIDS" && -n "$PATTERN" ]]; then
  PIDS=$(pgrep -d, -f "$PATTERN" || true)
fi
echo "Target PIDs: ${PIDS:-<system-wide -a>}"
echo "Output dir : $OUTDIR  (interval=${INTERVAL}s)"

# --- per-core frequency sampler (no root needed) ---
freq_loop() {
  local csv="$OUTDIR/cpu_freq.csv"
  echo "timestamp,cpu,khz" > "$csv"
  while true; do
    ts=$(date +%s.%N)
    for f in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq; do
      [[ -r "$f" ]] || continue
      cpu=$(echo "$f" | grep -o 'cpu[0-9]*' | head -1)
      echo "$ts,$cpu,$(cat "$f" 2>/dev/null || echo NaN)" >> "$csv"
    done
    sleep "$INTERVAL"
  done
}
freq_loop & FREQ_PID=$!

# --- IPC via perf (instructions / cycles) ---
PERF_PID=""
if command -v perf >/dev/null 2>&1; then
  PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo 4)
  INTERVAL_MS=$(awk "BEGIN{print int($INTERVAL*1000)}")
  if [[ -n "$PIDS" ]]; then
    perf stat -e cycles,instructions -p "$PIDS" -I "$INTERVAL_MS" -x, \
      -o "$OUTDIR/cpu_ipc.csv" 2>/dev/null & PERF_PID=$!
  else
    perf stat -e cycles,instructions -a -I "$INTERVAL_MS" -x, \
      -o "$OUTDIR/cpu_ipc.csv" 2>/dev/null & PERF_PID=$!
  fi
  echo "perf running (perf_event_paranoid=$PARANOID)."
  echo "  -> if cpu_ipc.csv stays empty, perf lacks permission (need <=2, or sudo / CAP_PERFMON)."
else
  echo "WARN: perf not found; IPC NOT collected. Install linux-tools-\$(uname -r)."
fi

cleanup() {
  kill "$FREQ_PID" 2>/dev/null || true
  [[ -n "$PERF_PID" ]] && kill -INT "$PERF_PID" 2>/dev/null || true
  echo "Saved: $OUTDIR/cpu_freq.csv  $OUTDIR/cpu_ipc.csv"
}
trap cleanup EXIT INT TERM

if [[ "$DURATION" -gt 0 ]]; then
  sleep "$DURATION"
else
  echo "Sampling... press Ctrl-C when the epoch finishes."
  wait "$FREQ_PID"
fi
