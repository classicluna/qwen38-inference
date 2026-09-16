#!/usr/bin/env bash
# Qwen3.8-27B on RX 7800 XT (16 GB) — full-VRAM profile.
# Measured peaks of 15.98 GiB total: 14.37 GiB text-only @32k (1.6 GiB free);
#   14.07 GiB with vision on CPU (MMPROJ + NO_MMPROJ_OFFLOAD=1, 1.9 GiB free);
#   14.64 GiB @CTX=16384 with vision on GPU.
#   GPU vision at 32k ctx peaks at 15.84 GiB — do not use (0.15 GiB free).
set -euo pipefail
cd "$(dirname "$0")"

BIN="llama.cpp/build/bin/llama-server"
MODEL="${MODEL:-models/Qwen3.8-27B-UD-Q3_K_XL.gguf}"
CTX="${CTX:-32768}"
UB="${UB:-256}"
PORT="${PORT:-8080}"
# Optional: vision encoder costs ~0.86 GiB VRAM (f16) — set MMPROJ to enable.
MMPROJ="${MMPROJ:-}"

args=(
  --model "$MODEL"
  --alias "${ALIAS:-qwen3.8-27b}"
  --n-gpu-layers 99
  --ctx-size "$CTX"
  --flash-attn on
  --cache-type-k "${KVTYPE:-q8_0}"
  --cache-type-v "${KVTYPE:-q8_0}"
  --ubatch-size "$UB"
  --batch-size 2048
  --parallel "${SLOTS:-1}"
  --jinja
  --chat-template-kwargs "{\"reasoning_effort\":\"${REASONING_EFFORT:-medium}\"}"
  --host 127.0.0.1
  --port "$PORT"
  --metrics
)
[ -n "$MMPROJ" ] && args+=( --mmproj "$MMPROJ" )
# Optional: encode images on CPU instead of GPU (frees ~0.9 GiB VRAM, slower vision prefill).
[ -n "${NO_MMPROJ_OFFLOAD:-}" ] && args+=( --no-mmproj-offload )
# Optional: speculative decoding with the model's own MTP head — ~+45% decode, costs ~1.5 GiB VRAM.
# Set MTP=models/MTP/mtp-Qwen3.8-27B-Q4_0.gguf (needs a shorter ctx to stay inside 16 GB).
[ -n "${MTP:-}" ] && args+=( --spec-draft-model "$MTP" --spec-type draft-mtp )

exec "$BIN" "${args[@]}"
