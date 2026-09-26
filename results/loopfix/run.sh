#!/usr/bin/env bash
# Loop-control A/B on Bonsai PQ2_0: same server policy as results/head2head (effort=low, budget 1024,
# greedy, 48k ctx, f16 KV), plus one loop-control knob per config. Items: the 8 items that hit the
# 4096-token limit in the head-to-head, plus a regression sample of previously-passing items.
set -uo pipefail
cd /home/evank/dev/inference
PY=.venv-eval/bin/python
H=results/head2head/bonsai-pq2
ids() { jq -r "$1" "$H/$2.jsonl" | paste -sd, -; }
LOOP_HP=$(ids 'select(.finish=="length").id' humanevalplus)
LOOP_FX=$(ids 'select(.finish=="length").id' humanevalfix)
LOOP_MM=$(ids 'select(.finish=="length").id' mmlupro)
# regression sample: every 8th previously-passing item per suite
REG() { jq -r 'select(.pass and .finish!="length").id' "$H/$1.jsonl" | awk 'NR%8==1' | paste -sd, -; }
HP="$LOOP_HP,$(REG humanevalplus)"; FX="$LOOP_FX,$(REG humanevalfix)"; MM="$LOOP_MM,$(REG mmlupro)"
MSG=$'\n\nTime is up. I must stop deliberating and give the final answer now, directly and concisely.\n'
declare -A CFG=(
  [base]=""
  [msg]="MSG"
  [dry]="--dry-multiplier 0.8 --dry-allowed-length 4 --dry-penalty-last-n 1024"
  [msgdry]="MSG --dry-multiplier 0.8 --dry-allowed-length 4 --dry-penalty-last-n 1024"
)
wait_free() { for i in $(seq 1 90); do [ "$(cat /sys/class/drm/card1/device/mem_info_vram_used)" -lt 2000000000 ] && return 0; sleep 2; done; exit 1; }
wait_up()   { for i in $(seq 1 150); do curl -sf http://127.0.0.1:8080/health >/dev/null && return 0; sleep 2; done; exit 1; }
for name in msg dry msgdry; do
  pgrep -af 'llama-server|llama-bench|llama-cli' && { echo "GPU busy — abort"; exit 1; }
  wait_free
  extra=()
  for w in ${CFG[$name]}; do [ "$w" = MSG ] && extra+=(--reasoning-budget-message "$MSG") || extra+=("$w"); done
  LD_LIBRARY_PATH=/home/evank/rocm-runtime/opt/rocm/lib:/opt/rocm/lib /home/evank/rocm-bin/llama-server \
    --model models/Ternary-Bonsai-2-27B-PQ2_0.gguf --alias bonsai-2-27b --n-gpu-layers 99 --ctx-size 49152 \
    --flash-attn on --cache-type-k f16 --cache-type-v f16 --ubatch-size 256 --batch-size 2048 --parallel 1 \
    --jinja --chat-template-kwargs '{"reasoning_effort":"low"}' --reasoning-budget 1024 \
    --host 127.0.0.1 --port 8080 "${extra[@]}" > "results/loopfix/server-$name.log" 2>&1 &
  S=$!; wait_up; echo "== $name UP $(date -u +%T)"
  $PY head2head-eval.py --tag "../loopfix/$name" --suites humanevalplus --ids "$HP" 2>&1 | grep '^=='
  $PY head2head-eval.py --tag "../loopfix/$name" --suites humanevalfix --ids "$FX" 2>&1 | grep '^=='
  $PY head2head-eval.py --tag "../loopfix/$name" --suites mmlupro --ids "$MM" 2>&1 | grep '^=='
  kill $S; wait $S 2>/dev/null
done
wait_free; echo "ALL DONE $(date -u +%T)"
