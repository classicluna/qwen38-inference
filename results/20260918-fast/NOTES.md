# Bonsai 2 speed campaign — 2026-09-18

Goal: attribute and fix the `43 t/s @ depth 0 → 10.4 t/s @ 114k` decode collapse, then
layer speculative decoding on top of whatever wins.

## Test harness

- `llama-bench` from `/home/evank/rocm-bin` (build 10685, commit 7dffb158d), `-ngl 99 -fa on -ub 256`,
  `-p 512 -n 128`, `-o json`, output tee'd into this directory.
- Newer build staged at `/home/evank/rocm-bin-b10709` (build 10709, commit 9a9394a89) for A/B.
- Live server (stopped for the campaign, restore after): `/home/evank/rocm-bin/llama-server`
  `--model models/Ternary-Bonsai-2-27B-PQ2_0.gguf --n-gpu-layers 99`
  `--alias bonsai-27b,bonsai-2-27b,ternary-bonsai-27b --flash-attn on`
  `--cache-type-k q4_0 --cache-type-v q4_0 --ubatch-size 256 --batch-size 2048 --parallel 1 --jinja`
  `--reasoning-format deepseek --reasoning on --reasoning-preserve`
  `--chat-template-kwargs {"reasoning_effort":"medium"} --host 127.0.0.1 --port 8085 --metrics`
  `--mmproj models/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` (ctx 262144, started by `run-server.sh`).

## Known-good facts before testing

- GPU is not throttled during decode: sclk 2555 MHz (max), mclk 1218 MHz (max), 199 W, 76 C junction.
- Live VRAM at 262k ctx + q4_0 KV + mmproj: 14.6 / 16.0 GiB used; GTT ~0.05 GiB (no spill).
- Live depth curve from `telemetry/usage.db` (engine `timings.predicted_per_second`):
  43.1 @ 52 tok · 27.5 @ 23k · 22.6 @ 37k · 19.1 @ 44k · 16.5 @ 60k · 15.1 @ 70k ·
  12.1 @ 82k · 11.3 @ 98k · 10.9 @ 108k · 10.4 @ 114k.
- The prior matrix (`results/optimization-matrix.json`) never measured a non-zero depth: `llama-bench`
  was invoked without `-d`, so "KV quant is within noise" was concluded with an empty KV cache.

## Results

### Depth 0 (build 10685, PQ2_0, q4_0 KV, ub 256, r=2)

- pp512 = 277.31 t/s (σ 0.08), tg128 = 44.21 t/s (σ 0.27) — `bench-d0-q4.json`.
  Reproduces the matrix row (42.54) within sampling tolerance.

### Depth 32768 KV sweep (build 10685, PQ2_0, ub 256, r=1)

| KV (k/v) | pp512 @32k | tg128 @32k | vs depth 0 (pp 277.3 / tg 44.2) |
|---|---|---|---|
| q4_0 | 169.67 t/s | 19.52 t/s | −39 % / −56 % |
| q8_0 | 53.73 t/s | 14.25 t/s | −81 % / −68 % |
| **f16** | **207.52 t/s** | **39.37 t/s** | **−25 % / −11 %** |

**Headline: f16 KV is 2.02x faster than q4_0 for decode at depth** (39.4 vs 19.5 t/s) and 1.22x
faster for prefill. The quantized-KV attention path on this build is the depth bottleneck — the
opposite of the naive "smaller KV = faster" assumption the old matrix encoded.

Consequences: the KV-quantization premise has to be inverted for this backend/build — quantization
saves VRAM but costs ~half the decode throughput at depth. The config space is now "how much context
can we afford at f16 KV" (f16 KV ≈ 66 KiB/token vs q4_0 ≈ 16.5 KiB/token, so 262k f16 KV ≈ 16 GiB is
impossible; the practical ceiling is being measured).

### Secondary knobs at depth 32768 (build 10685, PQ2_0, q4_0 KV)

| knob | pp512 | tg128 |
|---|---|---|
| `-ub 256` | 169.67 t/s | 19.52 t/s |
| `-ub 512` | 205.21 t/s | 23.78 t/s |

`-ub 512` buys +21 % prefill and (unexpectedly) +22 % decode at the same KV type — the ubatch size
changes the compute-buffer layout, not just prefill batching. Worth combining with f16 KV.

Aborted/invalid: `-fa off` (>9 min, no result), PTQ1_0 at depth (>13 min, no result), all b10709 runs
(see INVALIDATED above).

### f16 KV deeper (build 10685, PQ2_0, ub 256, r=1)

| depth | pp512 | tg128 |
|---|---|---|
| 0 | 277.3 | 44.2 |
| 32768 | 207.5 | 39.4 |
| 65536 | 165.0 | **34.0** |
| 98304 | 134.6 | **30.1** |

f16 curve is flat-ish: −32 % decode from 0 to the full production context (98304). For comparison the
live pane measured 11.3 t/s at 98k depth on q4_0 — the new default is **~2.7x that at the same depth**.

### Why f16 KV is faster than q4_0 — mechanism

The measured ordering falsifies the naive explanation (bytes moved). At identical depth and identical
weights:

| KV | bytes/token | decode @32k | prefill @32k |
|---|---:|---:|---:|
| q4_0 | ~16.5 KiB | 19.5 | 169.7 |
| q8_0 | ~33 KiB | 14.2 | 53.7 |
| f16 | ~66 KiB | **39.4** | **207.5** |

Bandwidth would rank them exactly opposite (q4_0 fastest, f16 slowest by 4x). Instead the *lowest-byte*
types are the slowest, so the cost is in the **quantized-KV attention code path**, not in memory traffic:
every attention step must dequantize the K (and V) slice before the f16 dot/accumulate, paying that cost
once per token over the *entire* context, layered over all 16 full-attention layers — which is precisely
the depth-proportional term that produced 43 → 27.5 → 15 → 10.4 t/s on q4_0.

Code-level context (`ggml/src/ggml-cuda/fattn.cu`, upstream lineage): the tensor-core flash-attention path
is excluded on HIP builds (`#if !defined(GGML_USE_HIP)` guards the MMA include), leaving the vector kernel
(decode-sized batches) and the tile kernel (prefill). With quantized K/V the vector kernel is selected only
for `Q->ne[1] <= 2`; prefill falls to the tile kernel, which for quantized K/V must dequantize per tile —
matching the 54-170 t/s prefill spread against 207 for f16. Whether the q4_0-q4_0 vector instance is
compiled at all is a build-time choice (`GGML_CUDA_FA_QUANTS`, default `q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16`);
a missing instance prints `no FlashAttention vector kernel compiled ... converting K and V to f16 instead
(slow)` once at startup. [INFERENCE on the exact per-kernel accounting; the measurements above are facts.]

Consequence: quantized KV is never a "free" memory saving on this backend — it trades ~2x decode
throughput for 4x context. A rebuild with `-DGGML_CUDA_FA_ALL_QUANTS=ON` (or explicit
`GGML_CUDA_FA_QUANTS`) is the experiment that would test whether more compiled K/V combinations buy
q4_0 back its speed, which would restore 262k context at f16-like rates.

## FA-quants rebuild experiment (2026-09-19)

### Results from the rebuilt binary (build-faquants, ALL_QUANTS on)

| config | old build (b10685) | new build (ALL_QUANTS) |
|---|---|---|
| f16/f16 @32k | 207.5 / 39.4 | **146.1 / 39.2** |
| q4_0/q4_0 @32k | 169.7 / **19.5** | **144.3 / 32.5** |
| f16-K / q4_0-V @32k | not compilable | 138.2 / 32.9 |
| q8_0-K / q4_0-V @32k | not compilable | 144.7 / 28.3 |

### Depth 98304 comparison (the production depth)

| build | KV | prefill | decode |
|---|---|---:|---:|
| official b10685 | q4_0 | — | 11.3 (live) |
| official b10685 | f16 | 134.6 | **30.1** |
| rebuilt, ALL_QUANTS | q4_0 | 106.3 | **20.5** |

So the rebuilt binary nearly doubles q4_0 at depth (11.3 → 20.5 t/s) but does not reach the official
build's f16 rate — and it pays ~20 % worse prefill. Net: the shipped default (official + f16 + 98k ctx)
stays the fastest configuration; the rebuilt binary's value is **full 262k context with q4_0** at
~17-20 t/s (vs ~9-10 t/s in the old pane experience). Adoption is one env var
(`LLAMA_BIN_DIR=/home/evank/llama-prism-src/build-faquants/bin CTX=262144 KVTYPE=q4_0`).

### Control build — ALL_QUANTS=OFF, same toolchain

| build (all @32k depth) | f16 prefill / decode | q4_0 prefill / decode |
|---|---|---|
| official b10685 (release) | 207.5 / 39.4 | 169.7 / **19.5** |
| from source, ALL_QUANTS **off** | 154.4 / 39.4 | 139.1 / **32.6** |
| from source, ALL_QUANTS **on** | 146.1 / 39.2 | 144.3 / 32.5 |

**Verdict: the quantized-KV speedup comes from the toolchain, not the flag.** A from-source build with
ALL_QUANTS off matches the ALL_QUANTS build on q4_0 decode (32.6 vs 32.5 t/s) while keeping ~5 % better
prefill (154.4 vs 146.1). The official release binary was simply compiled with a compiler/config that
produces a slow quantized-KV attention kernel for this GPU (the same source commit gives 19.5 t/s there
vs 32.6 here). Mixed K/V combos — the only thing the flag actually adds — are dominated by q4_0/q4_0 at
2.5x its memory, so the ALL_QUANTS build has no use here.

Cost of the from-source builds: prefill is ~25 % lower than the official binary on the f16 path
(154 vs 207), i.e. slightly worse time-to-first-token.

### Recommended configurations (final)

| use | command | @98k decode |
|---|---|---|
| **default (shipped)** | official binary, f16 KV, `ctx 98304` | **30.1 t/s** |
| full-context option | `LLAMA_BIN_DIR=/home/evank/llama-prism-src/build-noall/bin CTX=262144 KVTYPE=q4_0 ./run-server.sh` | ~20 t/s (measured 20.5 on the ALL_QUANTS build, which is identical on this path at 32k) |

The from-source binaries carry their own RUNPATH, so only `LLAMA_BIN_DIR` changes (wired into
`run-server.sh`). Trade for the long-context option: ~25 % slower prefill, but 262k context at ~2x the
decode rate the pane used to get with q4_0.

Conclusions so far:

1. **The official binary's quantized-KV path was ~1.7x slower than a from-source build** (q4_0 decode
   19.5 → 32.5 t/s at the same commit). That gap was toolchain/build codegen, not the kernel design.
2. **Mixed KV is dominated**: f16-K/q4_0-V performs like q4_0 (32.9 vs 32.5 t/s) while needing 2.5x the
   KV memory of q4_0 — so the residual penalty lives on the **V dequantization** side (K in f16 buys
   nothing). q4_0/q4_0 is the efficient point.
3. On the rebuilt binary, q4_0 costs only ~17 % decode vs f16 (32.5 vs 39.2) at 4x less KV memory →
   **q4_0 + large context becomes the best total configuration**, pending the depth measurement.
4. Prefill is ~30 % lower on the rebuilt binary across all configs (146 vs 207 for f16) — a real TTFT
   regression, unexplained (same source, same flags except ALL_QUANTS, single gfx1101 target).

Source findings in the fork at our build commit `7dffb158d`:

- Default (`GGML_CUDA_FA_ALL_QUANTS=OFF`) vec-kernel instances are only the **matched pairs**
  `f16-f16, q4_0-q4_0, q8_0-q8_0, bf16-bf16` — so the q4_0/q8_0 slowness is *not* a missing instance.
- With ALL_QUANTS **off**, the dispatcher rejects mixed K/V outright
  (`if (K->type != V->type) return BEST_FATTN_KERNEL_NONE;` in `ggml/src/ggml-cuda/fattn.cu`, fork
  revision 7dffb158d), and `ggml_cuda_fattn_kv_type_supported()` also rejects q4_1/q5_0/q5_1.
- Feature marker: the build advertises `FA_ALL_QUANTS` (`ggml-cuda.cu`); neither `/home/evank/rocm-bin`
  nor the official b10709 release lib contains it → both shipped with the flag OFF.

Implication: ALL_QUANTS unlocks *mixed* KV (e.g. K=f16 + V=q4_0 ≈ 2.5 B/elem vs f16's 4 B → ~62 % KV
memory) which may buy back context without giving up the fast f16 dot path.

Build recipe used (source at `/home/evank/llama-prism-src`, checkout `7dffb158d`):

```bash
# toolchain shims (no root): cmake+ninja via repo venv; lld and hipblas-common extracted locally
.venv/bin/pip install cmake ninja
#   lld 22.1.8  -> /tmp/lld-local   (extracted from Arch 'extra' package)
#   hipblas-common 7.2.4 -> /tmp/hbc-local  (runtime's hipblas.h includes <hipblas-common/hipblas-common.h>)
export PATH=/tmp/lld-local/usr/bin:$PWD/.venv/bin:$PATH
export LD_LIBRARY_PATH=/tmp/lld-local/usr/lib:$LD_LIBRARY_PATH
cmake -B build-faquants -G Ninja -DCMAKE_MAKE_PROGRAM=$PWD/.venv/bin/ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1101 -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF \
  -DCMAKE_HIP_COMPILER=/usr/bin/clang++ \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT=/home/evank/rocm-runtime/opt/rocm \
  -DCMAKE_HIP_ARCHITECTURES=gfx1101 \
  -DCMAKE_HIP_FLAGS='--rocm-device-lib-path=/opt/rocm/amdgcn/bitcode -I/tmp/hbc-local/opt/rocm/include' \
  -DCMAKE_PREFIX_PATH='/tmp/hbc-local/opt/rocm;/home/evank/rocm-runtime/opt/rocm;/opt/rocm'
ninja -C build-faquants -j8 llama-server llama-bench
```

## Production config restored + verified (2026-09-19 01:07Z)

`run-server.sh` changes: `KVTYPE` default f16, explicit `--ctx-size "$CTX"` (the variable existed but was
never passed — the server had been auto-fitting itself to 114432 with only ~0.4 GiB headroom), `CTX`
default 98304, comments updated to the measured numbers, `export KVTYPE` so the tracker records the
real KV type.

Verified: `/health` ok, `/slots` n_ctx = 98304, VRAM 14.8 GiB used / GTT 0.1 GiB (no spill), live request
through the full path (8080 → tracker → backend) decodes at **43.61 t/s** and is logged with
`kv_quant=f16, n_ctx=98304`. The same path earlier today: **8.98 t/s** at 140k depth on q4_0 / 262144.

Revert path: `CTX=262144 KVTYPE=q4_0 ./run-server.sh` (or `hub restart omp-llama` with those env vars).

## Follow-ups (2026-09-19)

- **f16 + `-ub 512` @32k**: 211.3 / 39.3 t/s vs 207.5 / 39.4 at ub 256 → the ub512 gain does **not**
  reproduce on the f16 path (it was q4_0-specific or noise). Production stays at `UB=256`.
  (`bench-d32768-f16-ub512.json`)
- **b10709 + DFlash2 head**: same failure as b10685 — `wrong number of tensors; expected 81, got 58`.
  DFlash2 needs the newer draft runtime (upstream PR #27342), not a newer fork build.
- **b10709 vs b10685, f16 KV @32k depth**: 175.7 / **29.4** t/s vs 207.5 / **39.4** t/s — the newer
  fork build is 15 % slower on prefill and 25 % slower on decode. No reason to upgrade; b10685 stays
  (`bench-d32768-f16-b10709.json`).
- DFlash2 head anatomy (from its GGUF): arch `dflash`, 5 blocks, block_size 8, two-tap convs
  (kernel 2, groups 16), selector rank 256 / top-k 16, 81 tensors vs the 58 the fork's loader builds.

## Summary — what to run (2026-09-18)

| Question | Answer |
|---|---|
| Is 10.4 t/s at 114k expected? | Yes *for q4_0 KV* — it is a config artifact, not a hardware limit |
| Root cause | the quantized-KV attention path collapses with depth: f16 KV is **2.02x** faster than q4_0 for decode at 32k depth (39.4 vs 19.5 t/s), 1.22x for prefill |
| Fastest decode measured | f16 KV, 39.4 t/s @32k depth, 44.2 t/s @0 (vs 19.5 / 14.2 for q4_0 / q8_0 @32k) |
| Prefill knob | `-ub 512` gives +21 % prefill, +22 % decode at 32k on q4_0, but **nothing on f16** (211.3 / 39.3 vs 207.5 / 39.4) — keep `UB=256` |
| Speculation | MTP head works: **+12-16 %** at 24.5k depth, 51 % acceptance, mean accepted len 2.54, +1.3 GiB VRAM |
| Speculation not available | DFlash2 heads do not load in this fork build or in b10709 (81 vs 58 tensor mismatch; needs upstream PR #27342 runtime) |
| Do not use | q8_0 KV (worst on both axes), PTQ1_0 at depth (>13 min/config), `-fa off` at depth (no result in 9 min) |
| Build b10709 | **tested and rejected**: 25 % slower decode, 15 % slower prefill than b10685 on the f16 path |
| Production default now | `run-server.sh`: **f16 KV, ctx 98304** (14.8 GiB used) |

Tradeoff to be explicit about: f16 KV costs ~66 KiB/token, so 262k context is impossible with it.
`CTX=262144 KVTYPE=q4_0` restores full context at ~2x slower decode; `MMPROJ=` (unset) frees ~0.6 GiB
and allows ~98k ctx with f16.

## Queued GPU waves (serialized on the single GPU)

- **Wave 2** (`bg_5`): `-fa off` @32k with f16 KV; PTQ1_0 model @32k with q4_0 KV; `-ub 512` @32k with
  q4_0 KV (prefill knob).
- **Wave 3** (`bg_2`): build b10709 at depth 0 (r=2) and depth 32768 with q4_0 KV — checks whether the
  3-day-newer fork release fixes the quantized-KV attention path.
- **Wave 4** (`bg_9`): b10709 at depth 32768 with f16 KV; build 10685 with f16 KV at depth 65536 and
  131072 — establishes the f16 speed curve and finds the VRAM ceiling for f16 KV.
- **Wave 5** (spec decode, server-based via `bench-server.py --attach`): baseline vs
  `--spec-type draft-mtp` (local Qwen3.8 MTP head) vs `--spec-type draft-dflash` (Bonsai-2 DFlash2 head,
  n-max 3/5) vs `--spec-type ngram-simple`, all at 32k ctx on the winning KV config. Acceptance is read
  from `timings.draft_n` / `timings.draft_n_accepted` (strings confirmed in `libllama-server-impl.so`).

## Spec decode wave (server-based, f16 KV, ctx 32768, ub 256, build 10685)

Attach harness: `bench-server.py --attach`, prose workload, prompt 24576 tokens, greedy, n_predict 512.

| config | prefill | decode | acceptance |
|---|---|---|---|
| baseline (no spec) | 245.76 t/s | 36.79 t/s (warmup) / 28.46 t/s (scored) | — |
| + `draft-mtp` (Qwen3.8 MTP head, n-max 3) | 228.54 t/s | **42.72 t/s (warmup)** | **51.2 %** (309/603), mean len 2.54 |
| + `draft-dflash` (Bonsai-2 DFlash2 head) | n/a | n/a | **runtime refuses to load it** |

DFlash2 verdict (2026-09-18): the PrismML fork build b10685 cannot load a DFlash2 draft head —
`llama_model_load: error loading model: done_getting_tensors: wrong number of tensors; expected 81,
got 58` for `models/dflash2/Bonsai-2-27B-DFlash2-Q8_0.gguf` (verified intact: 2,056,415,104 B,
sha256 matches the repo's SHA256SUMS). The fork's `draft-dflash` runtime implements the original
DFlash/DFly tensor set; DFlash2 heads (selector + two-tap convolutions, 81 tensors) need the newer
runtime from upstream PR #27342. Both downloaded heads (`ProCreations/Ternary-Bonsai-2-27B-DFlash2`,
`z-lab/Qwen3.8-27B-DFlash2-GGUF`) are therefore unusable on this build — keep them staged for when the
fork picks up DFlash2.

MTP gives only +16 % at 24.5k depth (vs +30 % measured at depth 0 in the earlier campaign) because
acceptance falls to 51 % with depth. Prefill drops ~7 % with speculation enabled.

## INVALIDATED runs — 2026-09-18 23:44Z

Two benchmark chains were launched concurrently (wave 3 and wave 4 both started at ~23:44:20Z; the
wave-3 wait-guard used a path-specific pgrep pattern that missed the b10709 binary). The GPU ran out of
memory: kernel log shows `amdgpu 0000:28:00.0: [drm] *ERROR* Not enough memory for command submission!`
at 23:44:29Z, the b10709 `llama-bench` (pid 197675) got SIGSEGV at 23:44:32Z, and Hyprland/`quickshell`
were killed at 23:44:35-37Z. **All b10709 measurements are void** (both output files are empty):
`bench-d0-q4-b10709.json`, `bench-d32768-f16-b10709.json`, plus `bench-d32768-q4-b10709.json`.
Also void: `bench-d32768-f16-faoff.json` (aborted, >9 min without finishing) and
`bench-d32768-ptq1_0-q4.json` (aborted, >13 min).

Rule going forward (`.omp/rules/gpu-experiments-strictly-serial`): one chain at a time, version-agnostic
wait-guards (`pgrep -f llama-bench`), immediate pre-launch re-check.

Valid measurements (single chain, no overlap): d0 q4_0, d32k q4_0/q8_0/f16, d32k q4_0 ub512.

## Drafters staged

- `models/dflash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf` — 1,143,006,816 B (size verified).
- `models/dflash2/Bonsai-2-27B-DFlash2-Q8_0.gguf` — downloading, expected 2,056,415,104 B.
- `models/MTP/mtp-Qwen3.8-27B-Q4_0.gguf` — pre-existing Qwen3.8 MTP head (measured 55.3 t/s @ depth 0,
  68.8 % acceptance in the prior campaign).

## Notes

- Subagent workers (`task`, `scout`) produced no turns in this environment — all prep work was done
  inline instead. Two scout items failed with empty artifacts.
