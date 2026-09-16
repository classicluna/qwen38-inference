#!/usr/bin/env bash
# Run the identical protocol for both quants, sequentially (VRAM is exclusive).
cd "$(dirname "$0")"
.venv/bin/python bench-quant.py --model models/Qwen3.8-27B-UD-IQ4_XS.gguf --tag iq4xs || echo "IQ4_XS RUN FAILED"
.venv/bin/python bench-quant.py --model models/Qwen3.8-27B-UD-Q3_K_XL.gguf --tag q3kx || echo "Q3_K_XL RUN FAILED"
echo "BOTH RUNS DONE"
