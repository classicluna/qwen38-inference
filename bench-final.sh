#!/usr/bin/env bash
# Final wave: head-to-head vs the target run + knob tuning, all with the 512-token prompt.
cd "$(dirname "$0")"
P=.venv/bin/python
IQ4=models/Qwen3.8-27B-UD-IQ4_XS.gguf
Q3K=models/Qwen3.8-27B-UD-Q3_K_XL.gguf
MTP=models/MTP/mtp-Qwen3.8-27B-Q4_0.gguf
BASE=(--model $IQ4 --ctx 8192 --kt q4_0 --vt q4_0 --draft $MTP --spec-type draft-mtp)

run() { $P bench-server.py "$@" || echo "RUN FAILED: $*"; }

# head-to-head against the submitted run's shape
run --label r3-iq4-2k-q4      --model $IQ4 --ctx 2048 --kt q4_0 --vt q4_0
run --label r3-iq4-2k-q4-mtp  --model $IQ4 --ctx 2048 --kt q4_0 --vt q4_0 --draft $MTP --spec-type draft-mtp
# shipping config + practical MTP config
run --label r3-iq4-32k-q8     --model $IQ4 --ctx 32768 --kt q8_0 --vt q8_0
run --label r3-q3k-16k-q4-mtp --model $Q3K --ctx 16384 --kt q4_0 --vt q4_0 --draft $MTP --spec-type draft-mtp
# knob tuning on the MTP base
run --label r3-mtp-8k "${BASE[@]}"
run --label r3-nmax1 "${BASE[@]}" --spec-nmax 1
run --label r3-nmax2 "${BASE[@]}" --spec-nmax 2
run --label r3-graphicsq "${BASE[@]}" --env GGML_VK_ALLOW_GRAPHICS_QUEUE=1
run --label r3-xferq "${BASE[@]}" --env GGML_VK_ASYNC_USE_TRANSFER_QUEUE=1
run --label r3-q8kv "${BASE[@]}" --kt q8_0 --vt q8_0
run --label r3-t4 "${BASE[@]}" --extra "-t 4"
echo "FINAL WAVE DONE"
