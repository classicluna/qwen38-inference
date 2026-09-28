#!/usr/bin/env bash
# Realistic 0k-context decode A/B through run-server.sh (PROFILE=qwen): chat API, thinking on, server-default
# sampling, fresh short prompts. Each spec is  label|ENV=val ENV=val ...  (env passed to run-server.sh).
# Prints mean output t/s, MTP acceptance and peak VRAM per label. Restores the systemd service at the end.
set -uo pipefail
cd /home/evank/dev/inference
PROMPTS=(
  "Write a short paragraph about the history of Waterloo, Ontario."
  "Write a Python function that parses an ISO-8601 duration like P1DT2H30M into seconds, with tests."
  "Explain why a hash map lookup is O(1) on average but O(n) worst case."
  "Refactor this into idiomatic Python: for i in range(len(xs)): if xs[i] % 2 == 0: out.append(xs[i]*2)"
)
V=/sys/class/drm/card1/device/mem_info_vram_used
stop_all() {
  systemctl --user stop omp-llama; pkill -f run-server.sh; pkill -f tracker.py; pkill -x llama-server
  for i in $(seq 90); do pgrep -x -r D,R,S,T llama-server >/dev/null || [ "$(cat $V)" -gt 2000000000 ] || break; sleep 1; done
  for i in $(seq 60); do [ "$(cat $V)" -lt 2000000000 ] && break; sleep 1; done
}
for spec in "$@"; do
  label=${spec%%|*}; envs=${spec#*|}
  stop_all
  # shellcheck disable=SC2086
  env $envs ./run-server.sh > "results/qwen-50/server-$label.log" 2>&1 &
  ok=0; for i in $(seq 300); do curl -sf localhost:8080/health >/dev/null && { ok=1; break; }; sleep 1; done
  [ $ok = 1 ] || { echo "$label: server did not come up"; continue; }
  peak=0
  ( while :; do cat $V; sleep 0.2; done ) > "/tmp/vram-$label" & SMP=$!
  for rep in 1 2; do for p in "${PROMPTS[@]}"; do
    curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
      -d "$(jq -n --arg p "$p" '{messages:[{role:"user",content:$p}],max_tokens:1500}')" \
      | jq -r '"\(.timings.predicted_per_second) \(.timings.draft_n_accepted // 0) \(.timings.draft_n // 0) \(.timings.predicted_n)"'
  done; done | awk -v l="$label" '{t+=$1*$4; n+=$4; a+=$2; d+=$3} END{printf "%-18s %.1f t/s (token-weighted)  acc %s  ", l, t/n, d?sprintf("%.0f%%",100*a/d):"-"}'
  kill $SMP; peak=$(sort -n "/tmp/vram-$label" | tail -1); rm -f "/tmp/vram-$label"
  awk -v p="$peak" 'BEGIN{printf "peak VRAM %.2f GiB (free %.2f)\n", p/2^30, 15.98-p/2^30}'
done
stop_all
systemctl --user start omp-llama
