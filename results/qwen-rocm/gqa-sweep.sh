#!/usr/bin/env bash
# Sweep GQA vec-attention params: rebuild build-amd per (NTKQ, MAXC2), tg64 @ 32k depth, q8_0 and q4_0 KV.
set -uo pipefail
cd /home/evank/dev/inference
F=/home/evank/llama-prism-kern/ggml/src/ggml-cuda/fattn-vec.cuh
T=toolchain
for spec in "$@"; do  # NTKQ:MAXC2
  a=${spec%%:*}; c=${spec#*:}
  sed -i -E "s/^#define GGML_FATTN_VEC_GQA_NTKQ [0-9]+/#define GGML_FATTN_VEC_GQA_NTKQ $a/; s/^#define GGML_FATTN_VEC_GQA_MAXC2 [0-9]+/#define GGML_FATTN_VEC_GQA_MAXC2 $c/" "$F"
  PATH=$T/lld/usr/bin:$PATH LD_LIBRARY_PATH=$T/lld/usr/lib:$T/rocm-llvm/opt/rocm/lib/llvm/lib \
    .venv/bin/ninja -C /home/evank/llama-prism-kern/build-amd -j8 llama-bench >/dev/null || { echo "build failed $spec"; exit 1; }
  for kv in q8_0 q4_0; do
    pgrep -af 'llama-server|llama-bench' && { echo "GPU busy"; exit 1; }
    LD_LIBRARY_PATH=/home/evank/rocm-runtime/opt/rocm/lib /home/evank/llama-prism-kern/build-amd/bin/llama-bench \
      -m models/Qwen3.8-27B-UD-IQ4_XS.gguf -ngl 99 -fa 1 -ctk $kv -ctv $kv -ub 256 -p 0 -n 64 -d 32768 -r 1 -o json 2>/dev/null \
      | jq -r --arg s "$spec" --arg kv $kv '.[] | "ntkq:maxc2=\($s) \($kv) tg64@32k \(.avg_ts|.*10|round/10)"'
  done
done
