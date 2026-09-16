#!/usr/bin/env bash
# Wave 2: tune around the MTP winner (IQ4_XS @8k, q4_0 KV + MTP draft).
cd "$(dirname "$0")"
P=.venv/bin/python
IQ4=models/Qwen3.8-27B-UD-IQ4_XS.gguf
MTP=models/MTP/mtp-Qwen3.8-27B-Q4_0.gguf
BASE=(--model $IQ4 --ctx 8192 --kt q4_0 --vt q4_0 --draft $MTP --spec-type draft-mtp)

run() { $P bench-server.py "$@" || echo "RUN FAILED: $*"; }

# how many tokens the MTP head may draft (default 3)
run --label w2-nmax1 "${BASE[@]}" --spec-nmax 1
run --label w2-nmax2 "${BASE[@]}" --spec-nmax 2
run --label w2-nmax4 "${BASE[@]}" --spec-nmax 4
# vulkan submission/queue knobs
run --label w2-graphicsq "${BASE[@]}" --env GGML_VK_ALLOW_GRAPHICS_QUEUE=1
run --label w2-xferq "${BASE[@]}" --env GGML_VK_ASYNC_USE_TRANSFER_QUEUE=1
run --label w2-nodes1 "${BASE[@]}" --env GGML_VK_MAX_NODES_PER_SUBMIT=1
# host-side knobs
run --label w2-t4 "${BASE[@]}" --extra "-t 4"
run --label w2-nommap "${BASE[@]}" --extra "--no-mmap"
# KV quality upgrade at the same context (does it still fit with MTP?)
run --label w2-q8kv "${BASE[@]}" --kt q8_0 --vt q8_0
echo "WAVE2 DONE"
