# N-gram speculative decoding for Qwen3.8-27B (Vulkan) — 2026-09-27, paused

Chat-style A/B (ngram-ab.py, thinking on): file rewrites 29.9 -> 90.5 t/s (simple,map-k4v), short rename
29.9 -> 40.3, prose/explain unchanged. The first A/B run was void: an orphaned server held :8085 and the
tracker proxied every config to it (draft_n = 0 everywhere); ngram-ab.py now asserts /slots speculative.

Real agent traffic (agent-bench, telemetry): `--spec-type ngram-simple,ngram-map-k4v` is a net LOSS —
8/8 passed but 16.3 min vs 12.2, effective output 17.3 vs 28.6 t/s. Acceptance in agent sessions is 2-5 %:
repeated tool output/code makes n-grams match, the model rarely continues them, and each rejected 48-token
draft costs a verify batch plus a recurrent-state checkpoint restore.

Next (not run): shorter drafts + min-hits 2 (`--spec-type ngram-simple --spec-ngram-simple-size-m 12
--spec-ngram-simple-min-hits 2` via ~/.config/omp-llama.env SPEC_ARGS), agent-bench + telemetry rate.
Until that beats 28.6 t/s, speculation stays off.

## Resumed 2026-09-28 — tuned run, verdict

`--spec-type ngram-simple --spec-ngram-simple-size-m 12 --spec-ngram-simple-min-hits 2`: acceptance rose to
47 % (1 689 / 3 576) but agent-bench 6/8, 17.5 min, effective 19.1 t/s vs 28.6 without speculation. Every
rejected draft restores the Gated-DeltaNet recurrent state from a checkpoint ("speculative decoding will use
checkpoints"), which costs more than accepted tokens save. N-gram speculation stays OFF for Qwen3.8 agents.
