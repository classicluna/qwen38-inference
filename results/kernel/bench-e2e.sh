#!/usr/bin/env bash
# End-to-end llama-bench, strictly serial on the single GPU (.omp/rules/gpu-experiments-strictly-serial).
# Usage: bench-e2e.sh <label> <bin-dir> <model> [depths=0,32768] [extra llama-bench args...]
set -uo pipefail
cd /home/evank/dev/inference
L=$1 BIN=$2 M=$3 D=${4:-0,32768}; shift 4 2>/dev/null || shift $#
if pgrep -af 'llama-server|llama-bench|llama-cli|test-backend-ops'; then echo "GPU busy — abort"; exit 1; fi
for i in $(seq 1 60); do [ "$(cat /sys/class/drm/card1/device/mem_info_vram_used)" -lt 2000000000 ] && break; sleep 2; done
export LD_LIBRARY_PATH=/home/evank/rocm-runtime/opt/rocm/lib:/opt/rocm/lib
"$BIN/llama-bench" -m "$M" -ngl 99 -fa 1 -ctk f16 -ctv f16 -ub 256 -p 512 -n 128 -d "$D" -r 2 -o json "$@" \
  > "results/kernel/$L.json" 2> "results/kernel/$L.log"
jq -r --arg l "$L" '.[] | "\($l) d=\(.n_depth) \(if .n_gen>0 then "tg" else "pp" end) \(.avg_ts|.*10|round/10) t/s ±\(.stddev_ts|.*10|round/10)"' "results/kernel/$L.json"
