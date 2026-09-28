#!/usr/bin/env bash
# Local model server for OMP on RX 7800 XT (16 GB): llama-server behind telemetry/tracker.py on :8080.
#
# PROFILE=qwen (default) — Qwen3.8-27B UD-IQ4_XS, the agent daily driver (8/8 vs Bonsai 7/8 on
#   agent-bench, half the wall time; results/agent-bench/). Settings: results/qwen-rocm/NOTES.md.
# PROFILE=bonsai — Bonsai 2 27B PTQ1_0 on the patched ROCm fork (branch bonsai-kern, built by
#   `COMPILER=amdclang toolchain/build.sh build-amd`): 51.8 t/s decode, 511 t/s prefill, 80k ctx + vision
#   + MTP (results/kernel/NOTES.md, results/mtp/). Faster chat/long-context reader, weaker agent.
#
# Every setting below is an env override (MODEL, CTX, KVTYPE, MMPROJ, MTP, LLAMA_BIN_DIR, ...).
set -euo pipefail
cd "$(dirname "$0")"
PROFILE="${PROFILE:-qwen}"
ROCM_BIN=/home/evank/llama-prism-kern/build-amd/bin
case "$PROFILE" in
  qwen)
    # Vulkan beats the patched ROCm build on the agent workload (~90 % decode time): decode 30.0 vs 26.9 t/s
    # short, 26.3 vs 22.7 @40k; ROCm only wins prefill (+10-15 %). q4_0 KV decodes as fast as q8_0 on Vulkan
    # and buys 64k ctx (peak 14.45 GiB). LLAMA_BIN_DIR=$ROCM_BIN switches to ROCm (results/qwen-rocm/NOTES.md).
    # Upstream 4da6337 (2026-09-27, worktree llama.cpp-new, build dir build-up): +6 % decode, +7-13 % prefill vs
    # the fb27a52 build, same perplexity, agent-bench 8/8 at 29.2 vs 28.6 eff t/s. llama.cpp/build/bin reverts.
    : "${LLAMA_BIN_DIR:=llama.cpp-new/build-up/bin}"
    : "${MODEL:=models/Qwen3.8-27B-UD-IQ4_XS.gguf}"
    : "${ALIAS:=qwen3.8-27b}"
    # Built-in MTP (the GGUF's own nextn layer), 2 draft tokens, graphics queue: 0k-context chat 33.2 -> 59.4 t/s
    # (72 % acceptance with default sampling). MTP's rollback states + draft KV cost ~0.8 GiB, so ctx is 48k
    # (peak 15.2 GiB on a plain desktop). MTP= CTX=65536 restores the 64k no-MTP profile (results/qwen-50/).
    : "${CTX:=49152}"
    : "${KVTYPE:=q4_0}"
    : "${MMPROJ:=}"
    : "${MTP:=builtin}"
    : "${MTP_N:=2}"
    : "${GFXQ:=1}"
    : "${REASONING_EFFORT:=low}"
    ;;
  bonsai)
    : "${LLAMA_BIN_DIR:=$ROCM_BIN}"
    : "${MODEL:=models/Ternary-Bonsai-2-27B-PTQ1_0.gguf}"
    : "${ALIAS:=bonsai-27b,bonsai-2-27b,ternary-bonsai-27b}"
    # f16 KV is the fast path for Bonsai on ROCm; 80k ctx + vision + MTP peaks 14.61 GiB (98k: 15.70, too tight).
    : "${CTX:=81920}"
    : "${KVTYPE:=f16}"
    : "${MMPROJ=models/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf}"
    # MTP n-max 1: real chat 51.0 -> 53.6 t/s; n2 50.2, n3 44.3 (results/mtp/chat-ab.txt).
    : "${MTP=models/MTP/mtp-Qwen3.8-27B-Q4_0.gguf}"
    : "${REASONING_EFFORT:=medium}"
    ;;
  *) echo "unknown PROFILE=$PROFILE (qwen|bonsai)" >&2; exit 1 ;;
esac
BIN="$LLAMA_BIN_DIR/llama-server"
export LD_LIBRARY_PATH="/home/evank/rocm-runtime/opt/rocm/lib:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
UB="${UB:-256}"
PUBLIC_PORT="${PORT:-8080}"
BACKEND_PORT="${BACKEND_PORT:-8085}"
export KVTYPE

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
# Speculative decoding with the Qwen3.8 MTP head (also valid for Bonsai, a Qwen3.8 derivative).
# MTP=builtin uses the target GGUF's own nextn layer (no sidecar; shares token_embd/output, ~0.75 GiB less VRAM).
if [ "$MTP" = builtin ]; then
  args+=( --spec-type draft-mtp --spec-draft-n-max "${MTP_N:-1}" )
elif [ -n "$MTP" ]; then
  args+=( --spec-draft-model "$MTP" --spec-type draft-mtp --spec-draft-n-max "${MTP_N:-1}" )
fi
# The MTP draft context keeps its own (1-layer, full-context) KV cache, f16 by default: match the target's type.
[ -n "$MTP" ] && args+=( --spec-draft-type-k "$KVTYPE" --spec-draft-type-v "$KVTYPE" )
# RADV: allow the graphics queue for compute (+5 % decode on Vulkan, results/qwen-50/). GFXQ=0 disables.
[ "${GFXQ:-0}" = 1 ] && export GGML_VK_ALLOW_GRAPHICS_QUEUE=1
# Extra speculation flags, word-split (e.g. SPEC_ARGS="--spec-type ngram-map-k").
[ -n "${SPEC_ARGS:-}" ] && read -r -a spec_extra <<<"$SPEC_ARGS" && args+=( "${spec_extra[@]}" )

# Never load a second model: two 27B servers exhaust the 15 GiB of RAM (+swap) and the OOM killer takes the
# desktop session with them (2026-09-27 18:35). Refuse to start while any llama-server or the ports are busy.
if pgrep -x -r D,R,S,T llama-server >/dev/null || ss -ltn "( sport = :$BACKEND_PORT or sport = :$PUBLIC_PORT )" | grep -q LISTEN; then
  echo "run-server.sh: a llama-server is already running or :$PUBLIC_PORT/:$BACKEND_PORT is taken — refusing to start" >&2
  pgrep -ax llama-server | cut -c1-120 >&2 || true
  exit 1
fi

# The shell execs into the tracker below, so a bash trap can never clean up. Instead the kernel sends
# llama-server SIGTERM when its parent (this process, later the tracker) dies, however it dies.
setpriv --pdeathsig TERM -- "$BIN" "${args[@]}" &

exec .venv/bin/python telemetry/tracker.py --port "$PUBLIC_PORT" --backend-port "$BACKEND_PORT"
