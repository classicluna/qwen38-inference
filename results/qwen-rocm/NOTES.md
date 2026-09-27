# Qwen3.8-27B UD-IQ4_XS: Vulkan vs patched ROCm — 2026-09-27

Question: does the ROCm/amdclang build + kernel work that helped Bonsai also speed up Qwen?
Answer: prefill yes (+10-15 %), decode no (−10 to −13 %). Agent sessions are ~90 % decode, so the
shipped default is **Vulkan, q4_0 KV, 64k ctx** — same speed as the old Vulkan q8_0/48k config
with 16k more context and 1.5 GiB free VRAM.

## End-to-end server benchmark (serve-bench.py; 256 output tokens, temp 0.6, fresh cache, median of 2)

| config | prompt | TTFT | latency | prefill t/s | output t/s |
|---|---:|---:|---:|---:|---:|
| **before**: Vulkan q8_0 48k | 508 | 1.46 s | 10.00 s | 361 | 29.9 |
| | 4 161 | 8.95 s | 17.61 s | 467 | 29.4 |
| | 16 340 | 37.93 s | 46.96 s | 431 | 28.3 |
| | 40 083 | 110.58 s | 120.30 s | 363 | 26.2 |
| **shipped**: Vulkan q4_0 64k | 508 | 1.56 s | 10.06 s | 344 | 30.0 |
| | 4 161 | 8.93 s | 17.57 s | 468 | 29.5 |
| | 16 340 | 38.07 s | 47.07 s | 430 | 28.3 |
| | 40 083 | 111.94 s | 121.65 s | 358 | 26.3 |
| ROCm (patched) q4_0 64k | 508 | 1.26 s | 10.73 s | 417 | 26.9 |
| | 4 161 | 7.75 s | 17.31 s | 539 | 26.7 |
| | 16 340 | 33.63 s | 43.74 s | 487 | 25.2 |
| | 40 083 | 100.19 s | 111.41 s | 400 | 22.7 |

Peak VRAM: Vulkan q8 48k 14.91 GiB · Vulkan q4 64k 14.45 · ROCm q8 48k 15.53 · ROCm q4 48k 14.83 · ROCm q4 64k 15.18.

## Real agent workload (agent-bench through run-server.sh, telemetry/usage.db)

| backend | pass | output tok | prefill time | decode time | effective out t/s |
|---|---|---:|---:|---:|---:|
| Vulkan q4_0 64k | 7/8 | 18 527 | 1.2 min | 10.8 min | 28.6 |
| ROCm q4_0 64k | 7/8 | 19 236 | 1.0 min | 12.4 min | 25.8 |

Pass/fail and per-task times vary more between runs than between backends (different tasks failed);
the token-weighted rate is the comparable number: Vulkan ~7 % faster per output token.

## Kernel work on the ROCm fork (commit a9395acda, branch bonsai-kern)

Profile (rocprofv3, d0): MMVQ weight reads already ~84-88 % of 624 GB/s; the ROCm decode gap is 1 775
kernels/token of launch gaps + small ops. At 32k the q8_0 `flash_attn_ext_vec` took 720 µs/layer: every Q
head re-read its K/V head (GQA 6:1), ~99 GB/s of useful bandwidth.

1. **GQA-batched vec flash attention** for quantized KV decode (one block serves ncols2 Q heads of a K/V
   head). Sweep (tg64 @32k): ncols2=6 spills (256 VGPR + scratch) → 19.3-22.2; **ncols2=3, 8 lanes/K row
   → 23.0 (q8_0)**, q4_0 17.8 → 21.8. Attention 720 → ~461 µs/layer. Tests: new GQA-3/6 single-token
   FLASH_ATTN_EXT cases for q8_0/q4_0 pass vs CPU; greedy decode token-identical on/off.
   `GGML_CUDA_FATTN_VEC_GQA=0` disables.
2. **Shared Q8_1 activations** across MMVQ consumers of one tensor (461 quantize launches/token → fewer):
   tg128 27.14 → 27.47 (+1.2 %), token-identical. `GGML_CUDA_NO_MMVQ_Q8_SHARE=1` disables.

Kernel-level (llama-bench pp512 / tg128), before the fixes:

| | d0 | @8k | @32k |
|---|---|---|---|
| Vulkan q8_0 | 492 / 30.7 | 430 / 29.6 | 317 / 27.2 |
| ROCm q8_0 | 568 / 27.3 | 466 / 25.5 | 334 / 21.0 |
| ROCm q4_0 | 570 / 27.1 | 485 / 23.7 | 334 / 17.8 |

## Rejected

- MTP for Qwen IQ4_XS: does not fit at 48k on either backend (server hangs in memory fitting); it only fits
  at ≤8-16k, too small for OMP sessions.
- ROCm f16 KV: fastest ROCm decode at depth (tile kernel near peak) but 48k f16 KV does not fit with 13.3 GiB
  of weights.
