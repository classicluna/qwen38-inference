#!/usr/bin/env bash
# Strict throughput A/B: both quants in one invocation, repeated with reversed order to
# cancel thermal/ordering bias. Shader caches are already warm from bench-both.sh.
cd "$(dirname "$0")"
BIN=llama.cpp/build/bin/llama-bench
COMMON=(-ngl 99 -fa on -ctk q8_0 -ctv q8_0 -ub 256 -b 2048 -p 512 -n 128 -d 0,8192
        -r 5 --delay 2 -t 8 --poll 0 -o jsonl)
IQ4=models/Qwen3.8-27B-UD-IQ4_XS.gguf
Q3K=models/Qwen3.8-27B-UD-Q3_K_XL.gguf

echo "=== pass 1: IQ4_XS then Q3_K_XL ==="
"$BIN" -m "$IQ4,$Q3K" "${COMMON[@]}" > strict-pass1.jsonl || echo "PASS1 FAILED"
echo "=== pass 2: Q3_K_XL then IQ4_XS ==="
"$BIN" -m "$Q3K,$IQ4" "${COMMON[@]}" > strict-pass2.jsonl || echo "PASS2 FAILED"
echo "STRICT RUNS DONE"
