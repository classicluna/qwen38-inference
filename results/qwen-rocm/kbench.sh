#!/usr/bin/env bash
# Kernel-level Qwen3.8-27B IQ4_XS sweep: pp512 / tg128 at several depths, per backend and KV type.
# Usage: kbench.sh <label> <bin-dir> <kvtype> [depths] [extra llama-bench args]. Strictly serial GPU.
set -uo pipefail
cd /home/evank/dev/inference
L=$1 BIN=$2 KV=$3 D=${4:-0,8192,32768}; shift 4 2>/dev/null || shift $#
pgrep -a -r D,R,S,T 'llama-server|llama-bench|llama-perplexity' && { echo "GPU busy — abort"; exit 1; }
for i in $(seq 60); do [ "$(cat /sys/class/drm/card1/device/mem_info_vram_used)" -lt 2000000000 ] && break; sleep 2; done
export LD_LIBRARY_PATH=/home/evank/rocm-runtime/opt/rocm/lib
"$BIN/llama-bench" -m models/Qwen3.8-27B-UD-IQ4_XS.gguf -ngl 99 -fa 1 -ctk "$KV" -ctv "$KV" -ub 256 \
  -p 512 -n 128 -d "$D" -r 2 -o json "$@" > "results/qwen-rocm/k-$L.json" 2> "results/qwen-rocm/k-$L.log"
jq -r --arg l "$L" '.[] | "\($l) d=\(.n_depth) \(if .n_gen>0 then "tg128" else "pp512" end) \(.avg_ts|.*10|round/10) ±\(.stddev_ts|.*10|round/10)"' "results/qwen-rocm/k-$L.json"
