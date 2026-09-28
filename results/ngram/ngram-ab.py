#!/usr/bin/env python3
"""N-gram speculative decoding A/B on the live Qwen profile (run-server.sh, chat API, thinking on).

Workloads: copy-heavy code edits (what agents emit) plus low-overlap explanation/prose (regression check).
Per config the systemd unit is stopped and run-server.sh is started with SPEC_ARGS; the unit is restarted at
the end. Reports output t/s, draft acceptance and wall time per workload (mean of REPS).
Usage: ngram-ab.py label='SPEC ARGS' ...   (label=  with empty args = baseline)
"""
import json, os, statistics, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path("/home/evank/dev/inference")
OUT = ROOT / "results/ngram/ngram-ab.json"
REPS = int(os.environ.get("REPS", 2))
src = (ROOT / "agent-bench/run.py").read_text()
cfg = (ROOT / "agent-bench/tasks/07-config-parser/repo/cfg/parser.py").read_text()
WORK = {
    "edit-docstrings": f"Add a concise docstring to every function in this file and return the complete updated file "
                       f"in one ```python block, changing nothing else.\n\n```python\n{src}```",
    "edit-rename": f"In this file rename the local variable `out` to `result` everywhere and return the complete "
                   f"updated file in one ```python block.\n\n```python\n{cfg}```",
    "explain": f"Explain in a few paragraphs what this script does and how grading works.\n\n```python\n{src}```",
    "prose": "Write a short paragraph about the history of Waterloo, Ontario.",
}


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def stop_all():
    sh("systemctl --user stop omp-llama; pkill -f run-server.sh; pkill -f tracker.py; pkill -f llama-server")
    for _ in range(90):
        if int(Path("/sys/class/drm/card1/device/mem_info_vram_used").read_text()) < 2_000_000_000:
            return
        time.sleep(1)


def chat(prompt):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": 3000}).encode()
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        j = json.load(r)
    t = j["timings"]
    return {"wall": time.time() - t0, "tps": t["predicted_per_second"], "n": t["predicted_n"],
            "draft": t.get("draft_n") or 0, "acc": t.get("draft_n_accepted") or 0}


res = json.loads(OUT.read_text()) if OUT.exists() else {}
try:
    for spec in sys.argv[1:]:
        label, _, args = spec.partition("=")
        if label in res:
            continue
        stop_all()
        p = subprocess.Popen(["./run-server.sh"], cwd=ROOT, env={**os.environ, "SPEC_ARGS": args},
                             stdout=open(ROOT / f"results/ngram/server-{label}.log", "w"), stderr=subprocess.STDOUT)
        for _ in range(450):
            if p.poll() is not None:
                sys.exit(f"{label}: run-server.sh exited (see results/ngram/server-{label}.log)")
            try:
                urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2); break
            except Exception:
                time.sleep(2)
        # the backend must be the server we just started, with speculation exactly as requested
        with urllib.request.urlopen("http://127.0.0.1:8085/slots", timeout=10) as r:
            spec_on = json.load(r)[0]["speculative"]
        if spec_on != bool(args.strip()):
            sys.exit(f"{label}: backend speculative={spec_on}, expected {bool(args.strip())} — stale server?")
        row = {}
        for w, prompt in WORK.items():
            rs = [chat(prompt) for _ in range(REPS)]
            row[w] = {"tps": statistics.mean(r["tps"] for r in rs), "wall": statistics.mean(r["wall"] for r in rs),
                      "n": statistics.mean(r["n"] for r in rs),
                      "acc": sum(r["acc"] for r in rs) / max(1, sum(r["draft"] for r in rs)),
                      "drafted": sum(r["draft"] for r in rs)}
            print(f"{label:12s} {w:16s} {row[w]['tps']:5.1f} t/s  {row[w]['n']:6.0f} tok  wall {row[w]['wall']:6.1f}s"
                  f"  acc {row[w]['acc']:.0%} of {row[w]['drafted']}", flush=True)
        res[label] = row
        OUT.write_text(json.dumps(res, indent=1))
        p.terminate(); p.wait()
finally:
    stop_all()
    sh("systemctl --user start omp-llama")
