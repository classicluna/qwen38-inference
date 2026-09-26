# Bonsai 2 kernel campaign — 2026-09-25/26

Fork branch `bonsai-kern` at `/home/evank/llama-prism-kern` (base 7dffb158d = b10685), built with
`SRC=/home/evank/llama-prism-kern toolchain/build.sh build-kern` (rootless: lld, hipblas-common,
rocm-llvm, rocprofiler extracted under `toolchain/`). All GPU runs strictly serial.

## Result (llama-bench, f16 KV, fa on, ub 256, r=2)

| build / model | pp512 d0 | tg128 d0 | pp512 @32k | tg128 @32k |
|---|---:|---:|---:|---:|
| official b10685, PQ2_0 (old default) | 282.8 | 46.9 | 209.9 | 39.8 |
| bonsai-kern, PQ2_0 | 195.2 | 47.0 | 152.8 | 39.8 |
| bonsai-kern, PTQ1_0, no fusions | 437.9 | 49.9 | 278.2 | 41.8 |
| **bonsai-kern, PTQ1_0, all changes** | **431.5** | **51.8** | **279.0** | **43.1** |

Net vs the old default: **+10 % decode @d0, +8 % @32k, +53 % prefill @d0, +33 % @32k**, and
1.26 GB less VRAM for weights (5.95 vs 7.21 GB) → more room for context.

## Changes (commits 22656db45, c5d0d0480)

1. **PTQ1_0 on HIP**: the HIP vec-dot was a scalar 128-multiply loop (why PTQ1_0 ">13 min" at
   depth). Ported the dp4a trit decoder to HIP with `v_perm_b32`-based widen + digit→{-1,0,1} LUT
   (2 perms, no subtract), and enabled PTQ1_0 MMQ on RDNA3 (WMMA tile loader, RDNA3 configs
   mirrored from PQ2_0). PTQ1_0 prefill beats PQ2_0 by 55 % because this build's PQ2_0 MMQ is slow
   (official binary's amdclang build is fast for PQ2_0; PTQ1_0 sidesteps it).
2. **Fused FWHT + Q8_1**: the Hadamard block kernel now also writes the rotated activation as Q8_1;
   the MMVQ consumers reuse it and skip `quantize_q8_1` (−~180 launches/token). +3.4 % decode.
   `GGML_CUDA_NO_FWHT_Q8=1` disables (A/B knob).
3. **GDN rows mode on CUDA/ROCm**: the recurrent op reads each sequence's state row in place
   instead of a gather copy (was Metal-only). Active only when MTP/speculation sets n_rs_seq > 0.
   `GGML_GDN_STATE_GATHER=1` restores the gather.

## Correctness

- `test-backend-ops` MUL_MAT for ptq1_0/pq2_0: 90/90 OK vs CPU; GATED_DELTA_NET: all OK.
- Greedy decode, 5 prompts × 256 tokens: official-PQ2_0 = kern-PQ2_0 (fused and unfused) token-identical;
  rows = gather token-identical. PTQ1_0 vs PQ2_0 diverges late on 2/5 prompts (34, 210) — different
  packing, not a bug:
- Perplexity (corpus.txt, 12×4096): official PQ2_0 16.7365, kern PQ2_0 16.7336, kern PTQ1_0 16.7336.

## Measured and rejected

- MMVQ nwarps for PTQ1_0: 1 → 50.1, 2 → 50.2, 4 → 36.7, 8 → 33.0 t/s. Kept 1.
- MMVQ rows/block: 1 → 52.0, 2 → 49.9, 4 → 45.3. Kept 1.
- HIP runtime env (dev kernargs, graph packet capture, HDP WA, AMD_OPT_FLUSH): all within noise or
  worse; `ROC_SYSTEM_SCOPE_SIGNAL=0` hung the bench — do not use. HIP graphs off: −6 %.

## Where decode time goes now (rocprofv3, PTQ1_0, d0)

21.9 ms/token span, 16.0 ms GPU busy (27 % idle = launch gaps, ~1565 kernels/token);
MMVQ 12.1 ms = 76 % of busy at ~490 GB/s effective on 5.95 GB. Remaining ideas, not done:
fuse rms_norm+FWHT, remove per-layer cpy/concat in the DeltaNet path, GQA-6 batched attention
for depth. Upper bound from launch gaps alone ≈ +25 %.

## Tried and reverted (2026-09-26)

- rms_norm + norm-weight + sign mul + FWHT + Q8_1 in one kernel: fired (80/token, 1565 → 1484 kernels)
  but 51.9 vs 51.8 t/s (noise), span 21.89 → 21.68 ms. Launch gaps cost ~2.6 µs each, and the fused
  kernel recomputes the row RMS per block. Greedy output drifted by float reduction order. Removed.
  Consequence: kernel-count reduction is worth ~1 %/80 launches here, so the "halve launches" idea is
  worth ~+7 %, not the +15-20 % first estimated.

## Loop control + quality (results/loopfix/, results/head2head/bonsai-ptq1-kern-msg/)

`--reasoning-budget 1024` + budget message ("Time is up… give the final answer now") on the 8 looping
items + regression sample: msg 21/24, 21/22, 15/16 with 0 cut-offs; DRY 18/24, 14/22, 14/16 (3 cut-offs);
both 20/24, 17/22, 14/16. Full HumanEvalFix on the final build + PTQ1_0 + msg: **160/164, 0 cut-offs,
18.5 s/item** (old Bonsai 157/164 with 2 cut-offs at 21.7 s; Qwen IQ4_XS 159/164). Now the run-server.sh default.
