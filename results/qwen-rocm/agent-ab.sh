#!/usr/bin/env bash
# Agent-bench A/B for the Qwen backends, each started through run-server.sh (PROFILE=qwen) like OMP uses it.
set -uo pipefail
cd /home/evank/dev/inference
for spec in "$@"; do  # tag:bindir:kvtype:ctx
  IFS=: read -r tag bin kv ctx <<<"$spec"
  pkill -f run-server.sh; pkill -f tracker.py; pkill -f llama-server; sleep 3
  for i in $(seq 60); do [ "$(cat /sys/class/drm/card1/device/mem_info_vram_used)" -lt 2000000000 ] && break; sleep 1; done
  PROFILE=qwen LLAMA_BIN_DIR=$bin KVTYPE=$kv CTX=$ctx ./run-server.sh > "results/qwen-rocm/agent-server-$tag.log" 2>&1 &
  for i in $(seq 900); do curl -sf localhost:8080/health >/dev/null && break; sleep 2; done
  python3 agent-bench/run.py --tag "qwen-$tag" --model llama.cpp/qwen3.8-27b --max-time 900 | tail -1
done
pkill -f run-server.sh; pkill -f tracker.py; pkill -f llama-server; true
