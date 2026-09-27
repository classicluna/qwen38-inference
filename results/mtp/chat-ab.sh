#!/usr/bin/env bash
# Realistic chat A/B for MTP: run-server.sh defaults (thinking on, server-default sampling), 4 prompts x 2.
set -uo pipefail
cd /home/evank/dev/inference
PROMPTS=(
  "Write a short paragraph about the history of Waterloo, Ontario."
  "Write a Python function that parses an ISO-8601 duration like P1DT2H30M into seconds, with tests."
  "Explain why a hash map lookup is O(1) on average but O(n) worst case."
  "Refactor this into idiomatic Python: for i in range(len(xs)): if xs[i] % 2 == 0: out.append(xs[i]*2)"
)
for spec in "$@"; do  # label:MTP_N (0 = off)
  l=${spec%%:*}; n=${spec#*:}
  pkill -f run-server.sh; pkill -f tracker.py; pkill -f llama-server; sleep 3
  for i in $(seq 60); do [ $(cat /sys/class/drm/card1/device/mem_info_vram_used) -lt 2000000000 ] && break; sleep 1; done
  if [ "$n" = 0 ]; then MTP= ./run-server.sh >/dev/null 2>&1 & else MTP_N=$n ./run-server.sh >/dev/null 2>&1 & fi
  for i in $(seq 150); do curl -sf localhost:8080/health >/dev/null && break; sleep 2; done
  for rep in 1 2; do for p in "${PROMPTS[@]}"; do
    curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
      -d "$(jq -n --arg p "$p" '{messages:[{role:"user",content:$p}],max_tokens:1500}')" \
      | jq -r --arg l "$l" '"\($l) \(.timings.predicted_per_second) \(.timings.draft_n_accepted // 0) \(.timings.draft_n // 0)"'
  done; done
done | awk '{t[$1]+=$2; c[$1]++; a[$1]+=$3; d[$1]+=$4} END{for(k in t) printf "%-6s %.1f t/s  acc %s\n", k, t[k]/c[k], d[k]?sprintf("%.0f%%",100*a[k]/d[k]):"-"}'
