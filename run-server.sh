#!/usr/bin/env bash
# Bonsai 2 27B (PrismML) on RX 7800 XT (16 GB) — ROCm PTQ1_0 profile on the patched fork
# (branch bonsai-kern in /home/evank/llama-prism-kern, built by `COMPILER=amdclang toolchain/build.sh build-amd`; see
# results/kernel/NOTES.md). vs official b10685 + PQ2_0: decode 46.9 -> 51.8 t/s @d0,
# 39.8 -> 43.1 @32k; prefill 283 -> 432 t/s; weights 1.26 GB smaller; identical perplexity.
# LLAMA_BIN_DIR=/home/evank/rocm-bin MODEL=models/Ternary-Bonsai-2-27B-PQ2_0.gguf reverts.
#
# 2026-09-18 depth-campaign findings (results/20260918-fast/):
#   * KV quantization COSTS throughput on this build: at 32k depth, decode is 39.4 t/s with f16 KV
#     vs 19.5 t/s with q4_0 and 14.2 t/s with q8_0; prefill 207.5 / 169.7 / 53.7 t/s respectively.
#     f16 KV is ~66 KiB/token, so the context that fits in 16 GiB is much smaller than with q4_0.
#   * Measured footprints: f16 KV @65k ctx + weights ≈ 13.1 GiB (llama-bench); leaving room for the
#     vision projector and compute buffers. 98k+ needs MMPROJ unset.
#   * Defaults below therefore use f16 KV at 65536 ctx. To get 262k back, run with
#     `CTX=262144 KVTYPE=q4_0` and accept ~2x slower decode at depth.
#   * DFlash2 draft heads (ProCreations/z-lab) do NOT load in this fork build (needs DFlash2 runtime).
set -euo pipefail
cd "$(dirname "$0")"
# Binary directory: override with LLAMA_BIN_DIR (e.g. /home/evank/llama-prism-src/build-noall/bin
# for a from-source build; those binaries carry their own RUNPATH, so only the dir changes).
ROCM_DIR="${LLAMA_BIN_DIR:-/home/evank/llama-prism-kern/build-amd/bin}"
if [ -x "$ROCM_DIR/llama-server" ] && [ -d "/home/evank/rocm-runtime/opt/rocm/lib" ]; then
  BIN="$ROCM_DIR/llama-server"
  export LD_LIBRARY_PATH="/home/evank/rocm-runtime/opt/rocm/lib:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
  MODEL_DEFAULT="models/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
else
  BIN="llama.cpp/build/bin/llama-server"
  MODEL_DEFAULT="models/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
fi
MODEL="${MODEL:-$MODEL_DEFAULT}"
# Context default: with f16 KV + vision projector this leaves ~1.5 GiB of VRAM headroom on 16 GiB.
# Omitting --ctx-size makes the server auto-fit (measured 114432, only ~0.4 GiB headroom).
CTX="${CTX:-81920}"
UB="${UB:-256}"
PUBLIC_PORT="${PORT:-8080}"
BACKEND_PORT="${BACKEND_PORT:-8085}"
# f16 KV is the fast path on ROCm (see header); q4_0 only for 262k contexts.
KVTYPE="${KVTYPE:-f16}"
export KVTYPE
# Optional: vision encoder (0.60 GiB Q8_0) — set MMPROJ to enable.
MMPROJ="${MMPROJ:-models/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf}"

args=(
  --model "$MODEL"
  --n-gpu-layers 99
  --ctx-size "$CTX"
  --alias "${ALIAS:-bonsai-27b,bonsai-2-27b,ternary-bonsai-27b}"
  --flash-attn on
  --cache-type-k "$KVTYPE"
  --cache-type-v "$KVTYPE"
  --ubatch-size "$UB"
  --batch-size 2048
  --parallel "${SLOTS:-1}"
  --jinja
  --reasoning-format deepseek
  --reasoning on
  --reasoning-preserve
  --chat-template-kwargs "{\"reasoning_effort\":\"${REASONING_EFFORT:-medium}\"}"
  --host 127.0.0.1
  --port "$BACKEND_PORT"
  --metrics
)
# Optional: hard cap on thinking tokens (0 = no thinking at all). Guards against a client
# requesting xhigh reasoning and burning minutes per turn on a 27B local model.
REASONING_BUDGET="${REASONING_BUDGET-1024}"
# When the budget runs out, this message is forced in before </think>: measured 0/62 length cut-offs
# vs 8/468 without it, and better pass rates than DRY (results/loopfix/). BUDGET_MESSAGE= disables.
BUDGET_MESSAGE="${BUDGET_MESSAGE-$'\n\nTime is up. I must stop deliberating and give the final answer now, directly and concisely.\n'}"
[ -n "$REASONING_BUDGET" ] && args+=( --reasoning-budget "$REASONING_BUDGET" )
[ -n "$REASONING_BUDGET" ] && [ -n "$BUDGET_MESSAGE" ] && args+=( --reasoning-budget-message "$BUDGET_MESSAGE" )
[ -n "$MMPROJ" ] && args+=( --mmproj "$MMPROJ" )
# Optional: encode images on CPU instead of GPU (frees ~0.9 GiB VRAM, slower vision prefill).
[ -n "${NO_MMPROJ_OFFLOAD:-}" ] && args+=( --no-mmproj-offload )
# MTP speculation, n-max 1: real chat (thinking, default sampling) 51.0 -> 53.6 t/s; n2 50.2, n3 44.3
# (results/mtp/chat-ab.txt). Greedy-only benches favoured n3 (+32 % prose) but acceptance halves when sampling.
# With vision: 80k ctx peaks 14.61 GiB (98k peaks 15.70 — too tight). MTP= disables.
MTP="${MTP-models/MTP/mtp-Qwen3.8-27B-Q4_0.gguf}"
[ -n "$MTP" ] && args+=( --spec-draft-model "$MTP" --spec-type draft-mtp --spec-draft-n-max "${MTP_N:-1}" )

"$BIN" "${args[@]}" &
LLAMA_PID=$!

cleanup() {
  kill -TERM "$LLAMA_PID" 2>/dev/null || true
  wait "$LLAMA_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

exec .venv/bin/python telemetry/tracker.py --port "$PUBLIC_PORT" --backend-port "$BACKEND_PORT"
