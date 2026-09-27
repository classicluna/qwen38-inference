#!/usr/bin/env python3
"""Greedy decode equivalence across builds/models/env knobs (strictly serial GPU use).

Starts one llama-server per config, generates N_PREDICT tokens greedily for fixed prompts via
/completion (raw prompts, no chat template, no thinking), records token ids, and reports the
first divergence position of every config against the first (reference) config.
"""
import json, os, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path("/home/evank/dev/inference")
OUT = ROOT / os.environ.get("EQUIV_OUT", "results/kernel/decode-equiv.json")
N_PREDICT = int(os.environ.get("N_PREDICT", 256))
LIB = "/home/evank/rocm-runtime/opt/rocm/lib:/opt/rocm/lib"
KERN = "/home/evank/llama-prism-kern/build-kern/bin"
PQ2, PTQ1 = "models/Ternary-Bonsai-2-27B-PQ2_0.gguf", "models/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
CONFIGS = [  # label, bin dir, model, extra env
    ("official-pq2", "/home/evank/rocm-bin", PQ2, {}),
    ("kern-pq2-nofuse", KERN, PQ2, {"GGML_CUDA_NO_FWHT_Q8": "1"}),
    ("kern-pq2-fuse", KERN, PQ2, {}),
    ("kern-ptq1-fuse", KERN, PTQ1, {}),
]
if os.environ.get("EQUIV_CONFIGS"):
    CONFIGS = [tuple(c) for c in json.loads(os.environ["EQUIV_CONFIGS"])]
corpus = (ROOT / "corpus.txt").read_text()
PROMPTS = [corpus[i * 20000:i * 20000 + 1500] for i in range(3)] + [
    "def quicksort(arr):\n    \"\"\"Sort a list of integers.\"\"\"\n",
    "The three most important causes of the French Revolution were",
]


def gpu_idle():
    r = subprocess.run(["pgrep", "-af", "llama-server|llama-bench|llama-cli|test-backend-ops"], capture_output=True)
    return r.returncode != 0


def post(path, body):
    req = urllib.request.Request("http://127.0.0.1:8090" + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


results = json.loads(OUT.read_text()) if OUT.exists() else {}
for label, bindir, model, env in CONFIGS:
    if label in results:
        continue
    if not gpu_idle():
        sys.exit("GPU busy — abort")
    p = subprocess.Popen([f"{bindir}/llama-server", "-m", model, "-ngl", "99", "-c", "8192", "-fa", "on",
                          "-ctk", "f16", "-ctv", "f16", "-ub", "256", "--parallel", "1", "--port", "8090",
                          "--host", "127.0.0.1"], env={**os.environ, "LD_LIBRARY_PATH": LIB, **env},
                         stdout=subprocess.DEVNULL, stderr=open(ROOT / f"results/kernel/equiv-{label}.log", "w"))
    try:
        for _ in range(900):
            try:
                urllib.request.urlopen("http://127.0.0.1:8090/health", timeout=2)
                break
            except Exception:
                time.sleep(1)
        toks = []
        for pr in PROMPTS:
            r = post("/completion", {"prompt": pr, "n_predict": N_PREDICT, "temperature": 0, "top_k": 1,
                                     "cache_prompt": False, "return_tokens": True, "seed": 1})
            toks.append(r["tokens"])
        results[label] = toks
        OUT.write_text(json.dumps(results))
        print(label, "done", [len(t) for t in toks], flush=True)
    finally:
        p.terminate(); p.wait()
        for _ in range(60):
            if int(Path("/sys/class/drm/card1/device/mem_info_vram_used").read_text()) < 2_000_000_000:
                break
            time.sleep(1)

ref_label = CONFIGS[0][0]
for label, *_ in CONFIGS:
    div = []
    for a, b in zip(results[ref_label], results[label]):
        n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        div.append(n)
    print(f"{label:18s} first divergence vs {ref_label}: {div}")
def div(x, y):
    return [next((i for i, (p, q) in enumerate(zip(a, b)) if p != q), len(a)) for a, b in zip(results[x], results[y])]


if "kern-pq2-nofuse" in results and "kern-pq2-fuse" in results:
    print("fuse vs nofuse (same build):", div("kern-pq2-nofuse", "kern-pq2-fuse"))
