#!/usr/bin/env bash
# HIP runtime env A/B on decode only (tg128 @ depth 0), PTQ1_0 on the kernel build. Strictly serial.
set -uo pipefail
cd /home/evank/dev/inference
BIN=${BIN:-/home/evank/llama-prism-kern/build-kern/bin}
M=${M:-models/Ternary-Bonsai-2-27B-PTQ1_0.gguf}
export LD_LIBRARY_PATH=/home/evank/rocm-runtime/opt/rocm/lib:/opt/rocm/lib
run() { # label env...
  local l=$1; shift
  pgrep -af 'llama-server|llama-bench|llama-cli|test-backend-ops' && { echo "GPU busy — abort"; exit 1; }
  env "$@" "$BIN/llama-bench" -m "$M" -ngl 99 -fa 1 -ctk f16 -ctv f16 -ub 256 -p 0 -n 128 -r 3 -o json 2>/dev/null \
    | jq -r --arg l "$l" '.[] | "\($l) tg128 \(.avg_ts|.*100|round/100) ±\(.stddev_ts|.*100|round/100)"'
}
for spec in "$@"; do
  l=${spec%%:*}; e=${spec#*:}; [ "$e" = "$spec" ] && e=""
  # shellcheck disable=SC2086
  run "$l" $e
done
