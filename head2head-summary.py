#!/usr/bin/env python3
"""Summarize results/head2head/<A> vs <B>: pass rate, paired wins/losses, exact sign-test p, failure modes."""
import json, math, sys
from pathlib import Path

R = Path(__file__).resolve().parent / "results" / "head2head"
A, B = (sys.argv[1:3] if len(sys.argv) > 2 else ("qwen-iq4", "bonsai-pq2"))


def load(tag, suite):
    p = R / tag / f"{suite}.jsonl"
    return {r["id"]: r for r in map(json.loads, p.open())} if p.exists() else {}


def sign_p(b, c):  # two-sided exact binomial on discordant pairs
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


print(f"| suite | n | {A} | {B} | only {A} | only {B} | sign-test p | {B} length-cut / no-answer |")
print("|---|---|---|---|---|---|---|---|")
for suite in ("humanevalplus", "humanevalfix", "mmlupro"):
    a, b = load(A, suite), load(B, suite)
    ids = sorted(a.keys() & b.keys())
    if not ids:
        continue
    pa = sum(a[i]["pass"] for i in ids); pb = sum(b[i]["pass"] for i in ids)
    oa = sum(a[i]["pass"] and not b[i]["pass"] for i in ids)
    ob = sum(b[i]["pass"] and not a[i]["pass"] for i in ids)
    cut = lambda d: sum(d[i].get("finish") == "length" for i in ids)
    noans = lambda d: sum(d[i]["detail"] in ("no code block", "request error") or "got=None" in d[i]["detail"] for i in ids)
    n = len(ids)
    print(f"| {suite} | {n} | {pa/n:.1%} | {pb/n:.1%} | {oa} | {ob} | {sign_p(oa, ob):.3g} | "
          f"{cut(b)} / {noans(b)} (vs {cut(a)} / {noans(a)}) |")
    tok = lambda d: sum(d[i].get("completion_tokens") or 0 for i in ids) / n
    wall = lambda d: sum(d[i].get("wall_s") or 0 for i in ids) / n
    print(f"|  ↳ mean tokens / wall s | | {tok(a):.0f} / {wall(a):.1f} | {tok(b):.0f} / {wall(b):.1f} | | | | |")
