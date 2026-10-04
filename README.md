# 27B LLM inference on a 16 GB RX 7800 XT

Running Qwen3.8-27B (and the ternary Bonsai 2 27B derivative) entirely in VRAM on a
consumer AMD card with llama.cpp, and making it fast enough to drive a coding agent.
Every change in this repo was accepted or rejected on measurements, which are checked
in next to the scripts that produced them.

Hardware: RX 7800 XT (Navi 32, gfx1101, 16 GiB VRAM, 624 GB/s), Ryzen 7 3700X, 15 GiB RAM.
No system-RAM safety net: any VRAM spill lands in GTT and thrashes, so fit was the first
acceptance test for every configuration.

## Headline results

| What | Before | After | Where |
|---|---:|---:|---|
| Qwen3.8-27B decode, 0k context chat | 33.2 t/s | **59.4 t/s** | [`results/qwen-50/NOTES.md`](results/qwen-50/NOTES.md) |
| Agent benchmark (8 tasks), wall time | 12.7 min | **8.3 min**, 8/8 pass | [`results/qwen-50/NOTES.md`](results/qwen-50/NOTES.md) |
| Bonsai 2 prefill (HIP kernel work) | 282.8 t/s | **431.5 t/s** (+53%) | [`results/kernel/NOTES.md`](results/kernel/NOTES.md) |
| Bonsai 2 decode (HIP kernel work) | 46.9 t/s | **51.8 t/s** (+10%) | [`results/kernel/NOTES.md`](results/kernel/NOTES.md) |
| Reasoning-loop cut-offs | 8/468 | **0/62** | [`results/loopfix/`](results/loopfix/) |
| HumanEvalFix (Bonsai, final build) | 157/164 | **160/164** | [`results/kernel/NOTES.md`](results/kernel/NOTES.md) |

## How the decode number was reached

Plain decode is bandwidth-bound: 14.25 GB of IQ4_XS weights per token at 624 GB/s is a
43.8 t/s ceiling, and upstream Vulkan was already at 71% of it. Kernel tuning alone could
not reach 50 t/s, so the gain had to come from speculation.

The shipped configuration (`run-server.sh`, `PROFILE=qwen`) uses the model's own
multi-token-prediction layer as the draft (`--spec-type draft-mtp`, 2 draft tokens,
~72% acceptance), q4_0 KV cache, 48k context and RADV's graphics queue for compute.
N-gram speculation was tried first and rejected: 3x faster on file rewrites in chat,
but 2-5% acceptance on real agent traffic made it a net loss
([`results/ngram/NOTES.md`](results/ngram/NOTES.md)).

## HIP kernel work

A fork of the PrismML llama.cpp branch, built rootless for gfx1101 with
[`toolchain/build.sh`](toolchain/build.sh):

- Ported the 1.75 bpw ternary (`PTQ1_0`) dequant to HIP with a `v_perm_b32` based
  widen and digit-to-{-1,0,1} LUT, and enabled its MMQ path on RDNA3.
- Fused the Hadamard block rotation with Q8_1 activation quantization so MMVQ consumers
  skip a separate quantize pass (about 180 fewer launches per token).
- GQA-batched vector flash attention for quantized-KV decode: one block serves several
  Q heads of a K/V head, attention 720 to ~461 us per layer at 32k context.

Correctness: `test-backend-ops` 90/90 against CPU, greedy decode token-identical with
each change on and off, perplexity unchanged. Profiles were taken with rocprofv3; the
notes record what was measured and rejected as well as what shipped.

## Repository layout

| Path | Contents |
|---|---|
| [`PLAN.md`](PLAN.md) | VRAM budget model, quant ladder, the Q3_K_XL vs IQ4_XS A/B, build steps, agent profile |
| [`run-server.sh`](run-server.sh) | The production server launcher; every setting is an env override |
| `results/*/NOTES.md` | One write-up per campaign: `qwen-50`, `qwen-rocm`, `kernel`, `ngram`, `20260918-fast` |
| `results/agent-bench/`, `agent-bench/` | 8-task coding-agent benchmark and per-run transcripts |
| `results/head2head/` | Qwen vs Bonsai on lm-eval and HumanEvalFix |
| `results/loopfix/` | Reasoning-budget stop message vs DRY sampling |
| `results/mtp/` | MTP draft-length and context-fit sweeps |
| `telemetry/` | Transparent proxy in front of llama-server that logs per-request tokens, timings and VRAM to SQLite |
| `bench-*.py`, `opt-*.json` | Server-level benchmark harness and the records behind the numbers in `PLAN.md` |
| `toolchain/build.sh` | Rootless HIP build for gfx1101 |

Model weights, build trees and logs are not tracked; `PLAN.md` records the exact model
digests and the llama.cpp commits used.

## Running it

```bash
./run-server.sh                      # Qwen3.8-27B IQ4_XS, MTP, 48k ctx, on :8080
PROFILE=bonsai ./run-server.sh       # Bonsai 2 27B PTQ1_0 on the patched ROCm build
MTP= CTX=65536 ./run-server.sh       # plain decode, 64k context
```

The server is OpenAI-compatible and sits behind `telemetry/tracker.py`, which proxies
`:8080` to the backend and records every request. `run-server.sh` refuses to start a
second model (two 27B loads exhaust RAM and the OOM killer takes the desktop with them)
and ties the server's lifetime to its parent with `pdeathsig`.
