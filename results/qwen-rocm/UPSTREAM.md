# Vulkan upstream bump — 2026-09-28

Worktree `llama.cpp-new` at upstream 4da6337 (from fb27a52), built into `llama.cpp-new/build-up`
(same flags as PLAN.md: GGML_VULKAN, static, Release, native). The worktree's pre-existing `build/`
(dated Sep 16) is unrelated and untouched.

llama-bench, IQ4_XS, q4_0 KV, fa on, ub 256 (second run, new build first — rules out order bias):

| | old fb27a52 | new 4da6337 |
|---|---:|---:|
| tg128 d0 | 29.1 | 30.9 (+6 %) |
| tg128 @32k | 25.8 | 27.4 (+6 %) |
| pp512 d0 | 473 | 533 (+13 %) |
| pp512 @32k | 289 | 308 (+7 %) |

Perplexity (corpus 12×4096, q4_0 KV): old 7.0676 ± 0.109, new 7.0308 ± 0.108.
agent-bench through run-server.sh: 8/8, 12.7 min, effective 29.2 t/s (old 28.6). Now the qwen default.
