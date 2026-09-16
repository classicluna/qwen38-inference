#!/usr/bin/env bash
# Optimization matrix. Same quant as the target run (IQ4_XS) plus our default (Q3_K_XL).
# Each config is measured with real 512-in/512-out requests (see bench-server.py).
cd "$(dirname "$0")"
P=.venv/bin/python
IQ4=models/Qwen3.8-27B-UD-IQ4_XS.gguf
Q3K=models/Qwen3.8-27B-UD-Q3_K_XL.gguf
MTP=models/MTP/mtp-Qwen3.8-27B-Q4_0.gguf

run() { $P bench-server.py "$@" || echo "RUN FAILED: $*"; }

# reference: current shipping config
run --label iq4-32k-q8 --model $IQ4 --ctx 32768 --kt q8_0 --vt q8_0
# target-run shape (2048 ctx, q4_0 KV) without and with MTP
run --label iq4-2k-q4 --model $IQ4 --ctx 2048 --kt q4_0 --vt q4_0
run --label iq4-2k-q4-mtp --model $IQ4 --ctx 2048 --kt q4_0 --vt q4_0 \
    --draft $MTP --spec-type draft-mtp
# matched 8k pair to isolate the MTP effect
run --label iq4-8k-q4 --model $IQ4 --ctx 8192 --kt q4_0 --vt q4_0
run --label iq4-8k-q4-mtp --model $IQ4 --ctx 8192 --kt q4_0 --vt q4_0 \
    --draft $MTP --spec-type draft-mtp
# same as above + forced MMVQ decode kernel
run --label iq4-8k-q4-mtp-mmvq --model $IQ4 --ctx 8192 --kt q4_0 --vt q4_0 \
    --draft $MTP --spec-type draft-mtp --env GGML_VK_FORCE_MMVQ=1
# our default quant, matched pair
run --label q3k-16k-q4 --model $Q3K --ctx 16384 --kt q4_0 --vt q4_0
run --label q3k-16k-q4-mtp --model $Q3K --ctx 16384 --kt q4_0 --vt q4_0 \
    --draft $MTP --spec-type draft-mtp
echo "OPT MATRIX DONE"
