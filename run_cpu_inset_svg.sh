#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

set +u
source /data1/miliang/RLinf/libero_openpi/bin/activate
set -u

RUN_DIR="${RUN_DIR:-profile_data/runA_mb32_envgen_worker_cpu_profile_20260605_130306}"
OUTPUT_SVG="${OUTPUT_SVG:-cpu_gpu_util_timeline_inset_bigger_text_cpu_inset.svg}"
BIN_S="${BIN_S:-1.0}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"
export RUN_DIR OUTPUT_SVG BIN_S MPLCONFIGDIR
mkdir -p "${MPLCONFIGDIR}"

python - <<'PY'
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path("tools").resolve()))
import plot_cpu_gpu_util_timeline as timeline  # noqa: E402

run_dir = Path(os.environ["RUN_DIR"])
output_svg = Path(os.environ["OUTPUT_SVG"])
bin_s = float(os.environ["BIN_S"])

process_cpu_series = timeline.read_process_phase_cpu_series(
    run_dir,
    bin_s=bin_s,
    rebuild_cache=False,
)
start_s = min(series[0][0] for series in process_cpu_series.values() if len(series[0]))
end_s = max(series[0][-1] for series in process_cpu_series.values() if len(series[0]))
times = np.arange(start_s, end_s + 1, dtype=float)
mean_util = np.zeros_like(times)

window = timeline.select_cpu_inset_window(
    times,
    mean_util,
    phases=[],
    detail_windows=[],
    process_cpu_series=process_cpu_series,
)
if window is None:
    raise SystemExit("failed to select CPU inset window")

written_path = timeline.save_cpu_inset_svg(
    output_svg,
    window,
    process_cpu_series,
)
if written_path is None:
    raise SystemExit("failed to save CPU inset SVG")

print(written_path)
PY
