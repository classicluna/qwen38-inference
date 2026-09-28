#!/usr/bin/env python3
"""MTP speculative decoding A/B on the patched fork + PTQ1_0 (strictly serial GPU use).

Per config: one llama-server, greedy /completion on short code, short prose and a ~24k-token prose
prompt; records decode t/s, draft acceptance, peak VRAM, and token equality vs the no-MTP baseline.
"""
import json, os, subprocess, sys, threading, time, urllib.request
from pathlib import Path

ROOT = Path("/home/evank/dev/inference")
OUT = ROOT / os.environ.get("OUT", "results/mtp/mtp-bench.json")
BIN = os.environ.get("BIN", "/home/evank/llama-prism-kern/build-kern/bin")
MODEL = os.environ.get("MODEL", "models/Ternary-Bonsai-2-27B-PTQ1_0.gguf")
MTP = "models/MTP/mtp-Qwen3.8-27B-Q4_0.gguf"
VRAM = Path("/sys/class/drm/card1/device/mem_info_vram_used")
CTX = int(os.environ.get("CTX", 32768))
BASE = ["-m", MODEL, "-ngl", "99", "-c", str(CTX), "-fa", "on", "-ctk", os.environ.get("KV", "f16"), "-ctv", os.environ.get("KV", "f16"), "-ub", "256",
        "--parallel", "1", "--host", "127.0.0.1", "--port", "8090"]
CONFIGS_ALL = {
    "off": [],
    "mtp-n1": ["--spec-draft-model", MTP, "--spec-type", "draft-mtp", "--spec-draft-n-max", "1"],
    "mtp-n2": ["--spec-draft-model", MTP, "--spec-type", "draft-mtp", "--spec-draft-n-max", "2"],
    "mtp-n3": ["--spec-draft-model", MTP, "--spec-type", "draft-mtp", "--spec-draft-n-max", "3"],
    # built-in nextn layer of the target GGUF (no sidecar: shares token_embd/output with the target)
    "mtpi-n1": ["--spec-type", "draft-mtp", "--spec-draft-n-max", "1"],
    "mtpi-n2": ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"],
    "mtpi-n3": ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"],
}
CONFIGS = {k: v for k, v in CONFIGS_ALL.items() if k in os.environ.get("ONLY", ",".join(CONFIGS_ALL)).split(",")}
corpus = (ROOT / "corpus.txt").read_text()
code = (ROOT / "head2head-eval.py").read_text()
PROMPTS = {
    "code-short": "Here is a Python program:\n\n" + code[:3000] + "\n\nRewrite the function `main` with clearer names:\n\n```python\n",
    "prose-short": corpus[5000:8000],
    "prose-24k": corpus[100000:100000 + 95000],
}
if os.environ.get("PROMPTS"):
    PROMPTS = {k: v for k, v in PROMPTS.items() if k in os.environ["PROMPTS"].split(",")}


def post(body):
    req = urllib.request.Request("http://127.0.0.1:8090/completion", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        return json.load(r)


res = json.loads(OUT.read_text()) if OUT.exists() else {}
for name, extra in CONFIGS.items():
    if name in res:
        continue
    if subprocess.run(["pgrep", "-r", "D,R,S,T", "-f", "llama-server|llama-bench|llama-perplexity"], capture_output=True).returncode == 0:
        sys.exit("GPU busy — abort")
    peak = [0]
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0], int(VRAM.read_text()))
            time.sleep(0.05)
    threading.Thread(target=sample, daemon=True).start()
    p = subprocess.Popen([f"{BIN}/llama-server", *BASE, *extra],
                         env={**os.environ, "LD_LIBRARY_PATH": "/home/evank/rocm-runtime/opt/rocm/lib"},
                         stdout=subprocess.DEVNULL, stderr=open(ROOT / f"results/mtp/server-{name}.log", "w"))
    try:
        for _ in range(300):
            try:
                urllib.request.urlopen("http://127.0.0.1:8090/health", timeout=2); break
            except Exception:
                time.sleep(1)
        row = {}
        for pn, pr in PROMPTS.items():
            r = post({"prompt": pr, "n_predict": 256, "temperature": 0, "top_k": 1, "cache_prompt": False,
                      "return_tokens": True, "seed": 1})
            t = r["timings"]
            row[pn] = {"tps": t["predicted_per_second"], "n_prompt": t["prompt_n"], "tokens": r["tokens"],
                       "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted")}
            print(f"{name:7s} {pn:11s} ctx={t['prompt_n']:6d} {t['predicted_per_second']:6.1f} t/s "
                  f"draft {t.get('draft_n_accepted')}/{t.get('draft_n')}", flush=True)
        row["peak_vram_gib"] = peak[0] / 2**30
        res[name] = row
        OUT.write_text(json.dumps(res))
    finally:
        stop.set(); p.terminate(); p.wait()
        for _ in range(60):
            if int(VRAM.read_text()) < 2_000_000_000: break
            time.sleep(1)

base = res["off"]
print("\n| config | " + " | ".join(PROMPTS) + " | peak VRAM | identical to off |")
for name, row in res.items():
    cells = []
    for pn in PROMPTS:
        sp = row[pn]["tps"] / base[pn]["tps"] - 1
        acc = f", acc {row[pn]['draft_acc']/row[pn]['draft_n']:.0%}" if row[pn].get("draft_n") else ""
        cells.append(f"{row[pn]['tps']:.1f} ({sp:+.0%}{acc})")
    same = all(row[pn]["tokens"] == base[pn]["tokens"] for pn in PROMPTS)
    print(f"| {name} | " + " | ".join(cells) + f" | {row['peak_vram_gib']:.2f} | {same} |")
