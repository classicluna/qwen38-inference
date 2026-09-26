#!/usr/bin/env bash
# Serial head-to-head: Qwen3.8-27B UD-IQ4_XS (Vulkan) then Bonsai 2 PQ2_0 (ROCm), identical
# reasoning policy (effort=low, --reasoning-budget 1024), greedy, 1 slot, text only.
set -uo pipefail
cd /home/evank/dev/inference
PY=.venv-eval/bin/python
COMMON=(--n-gpu-layers 99 --ctx-size 49152 --flash-attn on --ubatch-size 256 --batch-size 2048
        --parallel 1 --jinja --chat-template-kwargs '{"reasoning_effort":"low"}' --reasoning-budget 1024
        --host 127.0.0.1 --port 8080 --metrics)

wait_free() { for i in $(seq 1 90); do u=$(cat /sys/class/drm/card1/device/mem_info_vram_used); [ "$u" -lt 2000000000 ] && return 0; sleep 2; done; echo "VRAM never freed"; exit 1; }
wait_up()   { for i in $(seq 1 150); do curl -sf http://127.0.0.1:8080/health >/dev/null && return 0; sleep 2; done; echo "server never healthy"; exit 1; }
guard()     { if pgrep -af 'llama-server|llama-bench|llama-cli'; then echo "GPU busy — abort"; exit 1; fi; }

run_model() { # tag lm_eval_too
  local tag=$1
  $PY head2head-eval.py --tag "$tag" 2>&1 | grep -v Warning
  if [ "$2" = yes ]; then
    results/20260917T003336Z/run-lm-eval.sh "$tag-100" --limit 100
    mkdir -p "results/head2head/$tag" && mv results/20260917T003336Z/quality/lm-eval-"$tag"-100* "results/head2head/$tag/"
  fi
}

guard; wait_free
llama.cpp/build/bin/llama-server --model models/Qwen3.8-27B-UD-IQ4_XS.gguf --alias qwen3.8-27b \
  --cache-type-k q8_0 --cache-type-v q8_0 "${COMMON[@]}" > results/head2head/server-qwen.log 2>&1 &
S=$!; wait_up; echo "QWEN UP $(date -u +%FT%TZ)"
run_model qwen-iq4 no
kill $S; wait $S 2>/dev/null; wait_free

guard
LD_LIBRARY_PATH=/home/evank/rocm-runtime/opt/rocm/lib:/opt/rocm/lib /home/evank/rocm-bin/llama-server \
  --model models/Ternary-Bonsai-2-27B-PQ2_0.gguf --alias bonsai-2-27b \
  --cache-type-k f16 --cache-type-v f16 "${COMMON[@]}" > results/head2head/server-bonsai.log 2>&1 &
S=$!; wait_up; echo "BONSAI UP $(date -u +%FT%TZ)"
run_model bonsai-pq2 yes
kill $S; wait $S 2>/dev/null; wait_free
echo "ALL DONE $(date -u +%FT%TZ)"
