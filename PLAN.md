# Qwen3.8-27B on RX 7800 XT (16 GB) — plan

Goal: run **Qwen3.8-27B** fully in VRAM (no CPU/disk offload, no GTT spill) on this box, for
interactive/agentic use, with vision as a stretch goal.

## Measured facts (2026-09-16)

| Item | Value | Source |
|---|---|---|
| GPU | RX 7800 XT, Navi 32 `gfx1101`, 1× `renderD128` (`/dev/dri/card1`) | `lspci -d ::0300`, `/dev/dri` |
| VRAM total | 17,163,091,968 B = **15.98 GiB** | `/sys/class/drm/card1/device/mem_info_vram_total` |
| VRAM used (idle desktop) | 879,091,712 B = **0.82 GiB** | `mem_info_vram_used` |
| VRAM usable | ~**15.1 GiB** | total − desktop |
| System RAM | 15 GiB total, 12 GiB available | `free -h` — **no CPU-offload safety net** |
| Disk | 901 GB free on `/home` | `df -h` |
| Vulkan | RADV present: `vulkan-radeon 26.2.2`, `libvulkan_radeon.so`, ICD `radeon_icd.json` | pacman, glob |
| ROCm | not installed (optional later; `gfx1101` support is fiddly) | pacman |
| Toolchain | gcc 16, git, make, glslc (shaderc), spirv-tools, python 3.14 (+venv) ✅; **cmake, ninja, vulkan-headers, spirv-headers missing** | `command -v`, pacman |
| llama.cpp upstream | `LLM_ARCH_QWEN35` = `qwen35` registered in `src/llama-arch.cpp` | upstream master |
| Vulkan kernels | `GGML_OP_GATED_DELTA_NET` implemented, `S_V ∈ {16,32,64,128}` → model's 128 ✓; SSM conv/scan fusions present; FA requires `head_dim % 8 == 0` (model: 256 ✓) and accepts `q8_0` KV | `ggml/src/ggml-vulkan/ggml-vulkan.cpp` |

Model (Qwen/Qwen3.8-27B, Apache-2.0, released 2026-08-14):
64 layers = 16 × (3 × Gated DeltaNet + 1 × Gated Attention), hidden 5120, FFN 17408,
24 Q / 4 KV heads @ head_dim 256 (partial RoPE 0.25), 262 144 native ctx, MTP head, VL.

## VRAM budget math

KV cache is only for the **16 full-attention layers**: 16 × 4 KV heads × 256 dim × 2 (K+V)
= 32 768 values/token.

| | 16k ctx | 32k ctx | 64k ctx | 128k ctx |
|---|---|---|---|---|
| KV f16 (64 KiB/tok) | 1.00 GiB | 2.00 GiB | 4.00 GiB | 8.00 GiB |
| KV q8_0 (34 KiB/tok) | 0.53 GiB | **1.06 GiB** | 2.12 GiB | 4.25 GiB |

Fixed extras: DeltaNet recurrent state ≈ 0.15 GiB (48 layers × ~3.3 MB) + compute/graph
buffers ≈ 0.3–1.0 GiB (scales with `--ubatch-size`) + mmproj vision 0.86 GiB (only if loaded).

### Quant ladder (unsloth `Qwen3.8-27B-GGUF`, exact file sizes)

| Quant | Size | bpw | +1.06 KV +0.15 state +0.6 compute | Verdict |
|---|---|---|---|---|
| UD-IQ3_XXS | 10.18 GiB | 3.20 | 12.0 GiB | too lossy for the win |
| UD-IQ3_S | 11.21 GiB | 3.53 | 13.0 GiB | long-context pick (64k fits: 14.1) |
| **UD-Q3_K_XL** | **12.24 GiB** | **3.85** | **14.05 GiB** | ✅ **recommended** (1.1 GiB headroom @32k) |
| UD-IQ4_XS | 13.27 GiB | 4.17 | 15.08 GiB | ⚠️ only at 16k + lean desktop |
| UD-Q4_K_S | 14.30 GiB | 4.50 | 16.1 GiB | ❌ spills to GTT/RAM |
| Q4_0 / UD-Q4_K_M | 14.95 / 15.33 GiB | 4.7 / 4.8 | ❌ | ❌ |
| + MTP (Q4_0) | +1.28 GiB | | | stretch goal, 16k ctx only |
| mmproj-F16 | +0.86 GiB | | | vision only; quantize to Q8_0 if budget bites |

**Decision: `Qwen3.8-27B-UD-Q3_K_XL.gguf` (12.24 GiB), 32k ctx, q8_0 KV, `-ngl 99`.**
Unsloth Dynamic V3.0 3-bit ≈ older Q4 quality at ~3.85 bpw; it is the largest quant that
leaves real headroom for a browser/compositor without risking a spill.
Fallback ladder if measurements say otherwise: IQ3_S (long ctx) → IQ4_XS (quality, lean desktop).

## Measured results (implemented 2026-09-16)

Built llama.cpp `fb27a52` against RADV via a rootless local SDK prefix (`sdk/` + `third_party/`),
no distro packages needed. Model files verified against the HF LFS digests
(`8c2a45ff…` for UD-Q3_K_XL, `cbb841a9…` for mmproj-F16).

| Profile | Peak VRAM | Free of 15.98 | Verdict |
|---|---|---|---|
| 32k ctx, q8_0 KV, ub 256, text only | 14.37 GiB | 1.60 GiB | ✅ default |
| 32k ctx + vision encoder on CPU | 14.07 GiB | 1.91 GiB | ✅ best headroom, vision included |
| 16k ctx + vision encoder on GPU | 14.64 GiB | 1.34 GiB | ✅ fast vision, shorter ctx |
| 32k ctx + vision encoder on GPU | 15.84 GiB | 0.15 GiB | ❌ rejected — will OOM the desktop |

Throughput (`llama-bench -ngl 99 -fa 1 -ctk/-ctv q8_0 -ub 256`):
pp512 500.8–506.1 t/s, pp512@d8192 436.8 t/s, tg128 33.3 t/s, tg128@d8192 31.8 t/s.
ubatch sweep: 256 best for prefill (506 vs 467 @512, 514 @128); decode flat ~33 t/s everywhere.

Functional checks — all pass, via a live `llama-server` on :8080:
thinking emits `reasoning_content` and still answers correctly; native tool call returns
`get_weather{"city":"Reykjavik"}`; 18 742-token needle recall; vision OCR reads `4782` and names `Red`.

### Quant A/B — UD-Q3_K_XL vs UD-IQ4_XS (measured 2026-09-16)

Protocol: both quants interleaved in one `llama-bench` invocation (`-m A,B`), 5 reps, 2 s delay,
`t=8 poll=0`, a second pass with reversed model order (decode reproduced to ±0.1% between passes —
ordering bias negligible); VRAM fit measured at 32k ctx against the same 25.2k-token haystack;
quality via `llama-perplexity` on an identical 12×4096-token corpus slice. GTT sampled throughout
(equal at 0.188 GiB for both — neither quant spills to system memory).

| Metric | UD-Q3_K_XL (3.85 bpw, 12.24 GiB) | UD-IQ4_XS (4.17 bpw, 13.27 GiB) | Δ |
|---|---|---|---|
| pp512 | 506.5 t/s | 503.6 t/s | −0.6% (within pass noise) |
| pp512 @ 8k depth | 442.0 t/s | 442.3 t/s | +0.1% |
| tg128 | 33.81 t/s | 32.36 t/s | **−4.3%** |
| tg128 @ 8k depth | 32.56 t/s | 31.20 t/s | **−4.2%** |
| Peak VRAM @ 32k ctx | 14.49 GiB (1.50 free) | 15.45 GiB (**0.53 free**) | **+0.96 GiB** |
| Perplexity (12×4096) | 7.549 ± 0.117 | 7.035 ± 0.108 | **−6.8%, lower is better** |
| 25.2k-token prefill | 59.1 s | 59.2 s | equal |
| Thinking / tool call / needle recall | all pass | all pass | |

**Decision rule:** Q3_K_XL remains the default — 1.5 GiB of desktop headroom and ~4% faster decode.
Take IQ4_XS when quality matters more than 4% decode and the desktop is quiet, or run it at 16k ctx
(peak 14.86 GiB, 1.12 GiB free) where both are comfortable. Third-party numbers on analogue models
agree on direction (IQ4_XS: lower KLD, +1.2% top-1) — see `bench-data.json` → `reference[]`.

Artifacts: `run-server.sh` (env knobs `CTX`, `UB`, `MMPROJ`, `NO_MMPROJ_OFFLOAD`, `PORT`),
`smoke-test.py` (ctx-aware), `vision-test.py` + `make-image.py`, `bench-quant.py`, `bench-both.sh`,
`bench-strict.sh`, `refit-all.sh`, `make-report-data.py`, `embed-data.py`, `report.html`.

## Beating the localmaxxing RX 7800 XT run (2026-09-16)

Target: `localmaxxing.com` run `cmt0nfmo00f1jms01zkdjkvc7` — same GPU, llama.cpp,
**Unsloth-Dynamic-IQ4_XS**, 512-in/512-out, submitted **28.0 t/s decode / 500 t/s prefill**
(`-c 154000 -b 2048 -ub 256 -fa on -ck q4_0 -cv q4_0`, mmproj on CPU, Ryzen 7 7700 / 32 GB).

Measured here, kernel level (`llama-bench`, IQ4_XS, `-ngl 99 -fa on`):

| | this box | target | delta |
|---|---|---|---|
| pp512 | **499.7 t/s** (q4_0 KV) / 503.6 (q8_0 KV) | 500 | parity |
| tg128 | **32.00 t/s** (q4_0 KV) / 32.36 (q8_0 KV) | 28.0 | **+14 to +16%** |

Server protocol (real 481-in/512-out through `llama-server`, continuous 50 ms VRAM/GTT sampling):

| config | decode | prefill | peak VRAM | free | GTT |
|---|---|---|---|---|---|
| IQ4_XS 2k ctx q4_0 (their shape) | 31.48 | 393.5 | 13.92 | 2.07 | 0.167 |
| IQ4_XS 2k ctx q4_0 **+ MTP** | **50.11** | 356.2 | 15.53 | 0.46 | 0.223 |
| IQ4_XS 8k ctx q4_0 + MTP | 50.29 | 351.1 | 15.66 | 0.33 | 0.229 |
| IQ4_XS 8k ctx q4_0 + MTP + graphics queue | **52.66** | 358.1 | 15.64 | 0.34 | 0.229 |
| **Q3_K_XL 16k ctx q4_0 + MTP** | **49.79** | 367.5 | 14.79 | **1.19** | 0.237 |
| IQ4_XS 32k ctx q8_0 (old shipping config) | 31.57 | 395.0 | 14.96 | 1.02 | 0.182 |

**The win is the MTP head**: `--spec-draft-model models/MTP/mtp-Qwen3.8-27B-Q4_0.gguf --spec-type draft-mtp`
(the model's own multi-token-prediction layer, shipped as a sidecar) lifts decode ~+50-88% over the
submitted 28 t/s. It costs ~1.5 GiB VRAM and ~8% of the reported prefill number, so it is only viable
at reduced context on a 16 GB card; no config spilled (GTT ≤ 0.24 GiB everywhere).

Recommended fast profile: **Q3_K_XL + MTP @16k ctx = 49.8 t/s decode with 1.19 GiB still free**.
Raw peak: IQ4_XS + MTP @8k = 52.7 t/s but only 0.33 GiB free.

Server-reported prefill is diluted by ~0.19 s of fixed per-request work (481-token prompt ⇒ 1.22 s
vs 1.99 s for 840 tokens at the same config): the net prefill rate is ~465-500 t/s, matching the
kernel-level number. Compare prefill like-for-like via `llama-bench`, not the server's counter.

Reproduce: `.venv/bin/python bench-server.py --label demo --model models/Qwen3.8-27B-UD-Q3_K_XL.gguf
--ctx 16384 --kt q4_0 --vt q4_0 --draft models/MTP/mtp-Qwen3.8-27B-Q4_0.gguf --spec-type draft-mtp`
or `MTP=models/MTP/mtp-Qwen3.8-27B-Q4_0.gguf CTX=16384 ./run-server.sh`.

## Wiring into OMP (2026-09-16)

OMP ships a **built-in keyless `llama.cpp` provider**, auto-discovered at `LLAMA_CPP_BASE_URL` or
`http://127.0.0.1:8080`. No `models.yml` entry and no `config.yml` change are needed — the server
just has to be up when OMP starts.

Agent profile (chosen for headroom at usable context):

| config | peak VRAM | free | prefill | decode |
|---|---|---|---|---|
| **Q3_K_XL @24k ctx + MTP + q8_0 KV** | 15.18 GiB | **0.80** | 372.8 t/s | **47.3 t/s** |
| IQ4_XS @16k ctx + MTP + q8_0 KV | 15.89 GiB | 0.10 ❌ | 366.6 t/s | 47.8 t/s |

Q3_K_XL is the better agent quant despite lower standalone quality: it is 1 GiB smaller, so it buys
8k more context *and* 0.8 GiB more headroom than IQ4_XS at the same speed.

```bash
CTX=24576 MODEL=models/Qwen3.8-27B-UD-Q3_K_XL.gguf \
MTP=models/MTP/mtp-Qwen3.8-27B-Q4_0.gguf REASONING_EFFORT=low ./run-server.sh
```

- OMP lists it as `llama.cpp/qwen3.8-27b` with the **real** context (25K) and thinking levels
  (`low, medium, xhigh`) auto-detected; switch in-session with `/model llama.cpp/qwen3.8-27b`.
- End-to-end verified: `omp -p "Reply with exactly: LOCAL OK" --model llama.cpp/qwen3.8-27b --no-session`
  → `LOCAL OK` in 22.7 s.
- Reliability: 9/9 chat requests healthy (6 instruct @ temp 0.7 thinking-off, 3 thinking @ temp 1.0).
  The stray 1-token completions in benchmarking come from raw `/completion` with synthetic filler
  prompts (the model EOSes on them), not from the chat endpoint.
- Not reboot-persistent; a systemd user unit is the follow-up if this becomes the daily driver.

## Phases

### 1. Build — done, rootless (no sudo needed)
`cmake`/`ninja` come from pip wheels in `.venv`; Vulkan + SPIRV headers are vendored in
`third_party/` at the loader's exact version (`vulkan-sdk-1.4.357.0`) and installed into the
local prefix `sdk/`. Reproduce with:
```bash
python3 -m venv .venv && .venv/bin/pip install -U cmake ninja huggingface_hub pillow
git clone -b vulkan-sdk-1.4.357.0 --depth 1 https://github.com/KhronosGroup/Vulkan-Headers third_party/Vulkan-Headers
git clone -b vulkan-sdk-1.4.357.0 --depth 1 https://github.com/KhronosGroup/SPIRV-Headers third_party/SPIRV-Headers
.venv/bin/cmake -S third_party/SPIRV-Headers -B third_party/SPIRV-Headers/build -DCMAKE_INSTALL_PREFIX=$PWD/sdk
.venv/bin/cmake --build third_party/SPIRV-Headers/build --target install
ln -sfn $PWD/third_party/Vulkan-Headers/include/vulkan   sdk/include/vulkan
ln -sfn $PWD/third_party/Vulkan-Headers/include/vk_video sdk/include/vk_video
ln -sf /usr/bin/glslc sdk/bin/glslc
PATH=$PWD/.venv/bin:$PATH VULKAN_SDK=$PWD/sdk CPATH=$PWD/sdk/include \
  cmake -S llama.cpp -B llama.cpp/build -G Ninja -DGGML_VULKAN=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF
PATH=$PWD/.venv/bin:$PATH VULKAN_SDK=$PWD/sdk CPATH=$PWD/sdk/include cmake --build llama.cpp/build -j8
```
`vk_video/` was the trap — symlinking only `vulkan/` fails on `vulkan_core.h`. Optional cleanup:
`sudo pacman -S --needed cmake ninja vulkan-headers spirv-headers vulkan-tools` for distro packages.

### 2. Download
```bash
python3 -m venv .venv && .venv/bin/pip install -U "huggingface_hub[cli,hf_transfer]"
export HF_HUB_ENABLE_HF_TRANSFER=1
.venv/bin/hf download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-UD-Q3_K_XL.gguf mmproj-F16.gguf --local-dir models
```
~13 GB. (mmproj optional at this stage.)

### 3. Fit verification — the actual acceptance test
```bash
llama-server -m models/Qwen3.8-27B-UD-Q3_K_XL.gguf \
  -ngl 99 -c 32768 -fa on -ctk q8_0 -ctv q8_0 -ub 256 \
  --jinja --host 127.0.0.1 --port 8080
```
Pass criteria:
- log says all layers offloaded, no "offloaded to system memory" / GTT fallback;
- `Vulkan0 model buffer` + `KV buffer` + `compute buffer` ≈ weights + 1.06 + ~0.6 GiB;
- `cat /sys/class/drm/card1/device/mem_info_vram_used` stays < **15.1 GiB** under load;
- `llama-bench -m ... -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 512 -n 128 -d 0,8192` recorded
  (decode is bandwidth-bound: 624 GB/s ÷ ~12.8 GB ≈ ~45 tok/s ceiling, expect less).

### 4. Functional smoke tests
- thinking toggle: `--chat-template-kwargs '{"reasoning_effort":"medium"}'` vs default `xhigh`;
- tool call round-trip (the embedded jinja template supports it) — needed for agent harnesses;
- RoPE/long-context: 16k-token prompt stays coherent, no repetition collapse;
- vision (if mmproj kept): image → text via `llama-mtmd-cli`, and quantize mmproj to Q8_0 if VRAM is tight;
- persist: `run-server.sh` in `~/dev/inference` with final flags (OpenAI-compatible at :8080).

### 5. Tuning / optional
- `-ub` sweep (128/256/512) for prefill vs VRAM; `-fa on` vs off;
- try UD-IQ4_XS at 16k on a lean desktop for the quality comparison;
- MTP speculative decoding (`MTP/mtp-Qwen3.8-27B-Q4_0.gguf`) at 16k ctx — 1.28 GiB extra;
- ROCm/HIP build for gfx1101 comparison (big install, `HSA_OVERRIDE_GFX_VERSION=11.0.0` possible);
- optionally expose :8080 to the OMP/opencode config as an OpenAI-compatible provider.

## Risks
1. **No RAM safety net** (15 GiB): any VRAM spill lands in GTT → thrash, not just slowdown.
   Mitigation: stay on the recommended quant, keep `-c` at 32k, watch the log + sysfs.
2. Vulkan OOM at first attempt from a big `-ub`/other apps holding VRAM → drop `-ub` to 128, then ctx.
3. Early Gated-DeltaNet Vulkan bugs → pin a build that benchmarks sanely; compare against
   `-ngl 0` CPU output for a known-good sample if output looks broken.
4. `mmproj` VRAM cost (0.86 GiB) makes vision + 32k + Q3_K_XL borderline → quantize mmproj or
   run vision at 16k.

## Non-goals
bf16/FP8 (28–56 GB), vLLM/SGLang (no fit for a dense 27B on 16 GB; ROCm on gfx1101 is extra pain),
full 262k ctx (would need 8 GiB KV f16 / 4.25 GiB q8 — out of budget).
