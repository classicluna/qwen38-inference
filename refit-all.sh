#!/usr/bin/env bash
# Re-measure all VRAM-fit profiles with the ctx-scaled smoke-test haystack, so every
# profile in the report is produced by an identical harness.
cd "$(dirname "$0")"
.venv/bin/python bench-quant.py --model models/Qwen3.8-27B-UD-Q3_K_XL.gguf --tag q3kx-refit   --ctx 32768 --fit-only || echo "Q3K 32k FAILED"
.venv/bin/python bench-quant.py --model models/Qwen3.8-27B-UD-IQ4_XS.gguf --tag iq4xs-refit  --ctx 32768 --fit-only || echo "IQ4 32k FAILED"
.venv/bin/python bench-quant.py --model models/Qwen3.8-27B-UD-IQ4_XS.gguf --tag iq4xs-16k    --ctx 16384 --fit-only || echo "IQ4 16k FAILED"
echo "REFIT DONE"
