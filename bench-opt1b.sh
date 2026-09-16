#!/usr/bin/env bash
# Wave 1b: re-measure the decisive configs with the corrected harness
# (512-token prompt, continuous VRAM/GTT sampling).
cd "$(dirname "$0")"
P=.venv/bin/python
IQ4=models/Qwen3.8-27B-UD-IQ4_XS.gguf
Q3K=models/Qwen3.8-27B-UD-Q3_K_XL.gguf
MTP=models/MTP/mtp-Qwen3.8-27B-Q4_0.gguf

run() { $P bench-server.py "$@" || echo "RUN FAILED: $*"; }

# head-to-head with the target run's shape (2048 ctx, q4_0 KV), with and without MTP
run --label r2-iq4-2k-q4      --model $IQ4 --ctx 2048  --kt q4_0 --vt q4_0
run --label r2-iq4-2k-q4-mtp  --model $IQ4 --ctx 2048  --kt q4_0 --vt q4_0 --draft $MTP --spec-type draft-mtp
# matched 8k pair
run --label r2-iq4-8k-q4      --model $IQ4 --ctx 8192  --kt q4_0 --vt q4_0
run --label r2-iq4-8k-q4-mtp  --model $IQ4 --ctx 8192  --kt q4_0 --vt q4_0 --draft $MTP --spec-type draft-mtp
# current shipping config, for continuity
run --label r2-iq4-32k-q8     --model $IQ4 --ctx 32768 --kt q8_0 --vt q8_0
# our default quant: MTP at a usable context
run --label r2-q3k-16k-q4     --model $Q3K --ctx 16384 --kt q4_0 --vt q4_0
run --label r2-q3k-16k-q4-mtp --model $Q3K --ctx 16384 --kt q4_0 --vt q4_0 --draft $MTP --spec-type draft-mtp
echo "WAVE1B DONE"
