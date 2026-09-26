#!/usr/bin/env bash
# Rebuild the kernel worktree with GGML_MMVQ_RDNA3_PTQ1_0_NWARPS=N and measure tg128 @ d0 (serial).
set -uo pipefail
cd /home/evank/dev/inference
W=/home/evank/llama-prism-kern
F=$W/ggml/src/ggml-cuda/mmvq.cu
T=/home/evank/dev/inference/toolchain
MAC=${MAC:-GGML_MMVQ_RDNA3_PTQ1_0_NWARPS}
for n in "$@"; do
  sed -i -E "s/^#define $MAC [0-9]+/#define $MAC $n/" "$F"
  PATH=$T/lld/usr/bin:$PATH LD_LIBRARY_PATH=$T/lld/usr/lib .venv/bin/ninja -C $W/build-kern -j8 llama-bench test-backend-ops >/dev/null || { echo "build failed n=$n"; exit 1; }
  results/kernel/env-sweep.sh "$MAC=$n:X=1"
done
