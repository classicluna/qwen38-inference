# Qwen3.8-27B IQ4_XS: 50+ t/s at 0k context — 2026-09-28

Plain decode ceiling: 14.25 GB of weights per token at 624 GB/s = 43.8 t/s (100 % bandwidth); upstream
Vulkan already reaches 71 % (30.9 t/s). 50 t/s therefore needs speculation; kernel tuning alone cannot.

Winning config (run-server.sh qwen default): upstream Vulkan 4da6337, q4_0 KV, 48k ctx,
GGML_VK_ALLOW_GRAPHICS_QUEUE=1, built-in MTP (the GGUF's own nextn layer, `--spec-type draft-mtp` with no
draft model), n-max 2, draft KV q4_0.

## Realistic chat, 0k ctx (chat-ab.sh: thinking on, server-default sampling, token-weighted over 8 prompts)

| config | out t/s | MTP acceptance | peak VRAM |
|---|---:|---:|---:|
| before: no MTP, 64k | 33.2 | — | 15.32 |
| built-in MTP n1, 32k | 51.7 | 84 % | 15.40 |
| **built-in MTP n2, 48k, draft KV q4_0 (shipped)** | **59.4** | 72 % | 15.77 (Steam open) |
| built-in MTP n3, 32k | 62.6 | 67 % | 15.69 |
| built-in MTP n3, 48k | 61.3 | 65 % | 15.86 — too tight |

Greedy /completion bench (mtp-bench.py, 8k ctx): off 32.7 · sidecar MTP n2 59.9 · built-in n3 66.4 on code.
Graphics queue: +5 % on its own (31.0 -> 32.7). The built-in nextn layer beats the 1.27 GiB sidecar GGUF: same
acceptance, 0.75 GiB less VRAM (shares token_embd/output with the target).

## Agent workload (agent-bench through run-server.sh, telemetry)

| config | pass | wall | effective out t/s |
|---|---|---:|---:|
| upstream, no MTP, 64k | 8/8 | 12.7 min | 29.2 |
| **upstream + MTP n2, 48k** | **8/8** | **8.3 min** | **55.6** |

Unlike n-gram speculation (2-47 % acceptance, net loss), MTP drafts are accepted 70-80 % of the time, so the
recurrent-state rollback cost is amortized.

## Caveats

- Greedy MTP output diverges from plain greedy after ~31 tokens (float rounding of the batched verify pass;
  the text stays coherent and on-topic). Quality check: agent-bench 8/8 either way.
- MTP costs ~0.8 GiB (rollback states + draft KV), so the context drops 64k -> 48k. `MTP= CTX=65536` restores it.
