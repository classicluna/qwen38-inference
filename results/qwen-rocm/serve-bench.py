#!/usr/bin/env python3
"""End-to-end llama-server benchmark for Qwen3.8-27B configs (strictly serial GPU use).

Per config: start llama-server, then for each prompt size send a streaming /completion (fresh cache,
256 output tokens, temperature 0.6 / top-p 0.95 / seed 1 so MTP acceptance is realistic) and record:
  ttft_s      client wall time to the first streamed token
  latency_s   client wall time to the last token
  prefill_tps server prompt_per_second
  decode_tps  server predicted_per_second (output tok/s)
  e2e_tps     output tokens / latency (what a user perceives)
  draft acc   accepted / drafted (MTP)
plus peak VRAM/GTT over the whole config. Each size runs REPS times; the median is reported.

Usage: serve-bench.py <config> [...]   (names from CONFIGS; OUT env picks the json file)
"""
import json, os, statistics, subprocess, sys, threading, time, urllib.request
from pathlib import Path

ROOT = Path("/home/evank/dev/inference")
OUT = ROOT / os.environ.get("OUT", "results/qwen-rocm/serve-bench.json")
VK = str(ROOT / "llama.cpp/build/bin")
RC = "/home/evank/llama-prism-kern/build-amd/bin"
MODEL = "models/Qwen3.8-27B-UD-IQ4_XS.gguf"
MTP = "models/MTP/mtp-Qwen3.8-27B-Q4_0.gguf"
VRAM = Path("/sys/class/drm/card1/device/mem_info_vram_used")
GTT = Path("/sys/class/drm/card1/device/mem_info_gtt_used")
REPS = int(os.environ.get("REPS", 2))
SIZES = [int(x) for x in os.environ.get("SIZES", "512,4096,16384,40000").split(",")]


def srv(ctx, kv, extra=()):
    return ["-m", MODEL, "-ngl", "99", "-c", str(ctx), "-fa", "on", "-ctk", kv, "-ctv", kv, "-ub", "256",
            "-b", "2048", "--parallel", "1", "--host", "127.0.0.1", "--port", "8090", *extra]


MTP_ARGS = ["--spec-draft-model", MTP, "--spec-type", "draft-mtp", "--spec-draft-n-max", "1"]
CONFIGS = {
    # label: (bin dir, server args, env)
    "vk-q8-48k":       (VK, srv(49152, "q8_0"), {}),                       # shipped baseline
    "rocm-q8-48k":     (RC, srv(49152, "q8_0"), {}),
    "rocm-q4-48k":     (RC, srv(49152, "q4_0"), {}),
    "rocm-q4-64k":     (RC, srv(65536, "q4_0"), {}),
    "rocm-q4-48k-mtp": (RC, srv(49152, "q4_0", MTP_ARGS), {}),
    "vk-q8-16k-mtp":   (VK, srv(16384, "q8_0", MTP_ARGS), {}),
}
for k, v in json.loads(os.environ.get("EXTRA_CONFIGS", "{}")).items():
    CONFIGS[k] = (v[0], srv(*v[1], v[2] if len(v) > 2 else ()), {})

corpus = (ROOT / "corpus.txt").read_text()


def prompt(ntok):
    return corpus[50000:50000 + int(ntok * 3.9)] + "\n\nSummarize the passage above in detail:\n"


def run(label):
    bindir, args, env = CONFIGS[label]
    if subprocess.run(["pgrep", "-f", "llama-server|llama-bench|llama-perplexity"], capture_output=True).returncode == 0:
        sys.exit("GPU busy — abort")
    peak = {"vram": 0, "gtt": 0}
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            peak["vram"] = max(peak["vram"], int(VRAM.read_text()))
            peak["gtt"] = max(peak["gtt"], int(GTT.read_text()))
            time.sleep(0.05)
    threading.Thread(target=sample, daemon=True).start()
    log = open(ROOT / f"results/qwen-rocm/server-{label}.log", "w")
    p = subprocess.Popen([f"{bindir}/llama-server", *args], stdout=subprocess.DEVNULL, stderr=log,
                         env={**os.environ, "LD_LIBRARY_PATH": "/home/evank/rocm-runtime/opt/rocm/lib", **env})
    rows = {}
    try:
        for _ in range(900):
            if p.poll() is not None:
                return {"error": "server exited (see log)"}
            try:
                urllib.request.urlopen("http://127.0.0.1:8090/health", timeout=2); break
            except Exception:
                time.sleep(1)
        ctx = int(args[args.index("-c") + 1])
        for size in SIZES:
            if size + 300 > ctx:
                continue
            reps = []
            for _ in range(REPS):
                body = {"prompt": prompt(size), "n_predict": 256, "temperature": 0.6, "top_p": 0.95, "seed": 1,
                        "cache_prompt": False, "stream": True, "ignore_eos": True}
                req = urllib.request.Request("http://127.0.0.1:8090/completion", json.dumps(body).encode(),
                                             {"Content-Type": "application/json"})
                t0 = time.time(); ttft = None; timings = None
                with urllib.request.urlopen(req, timeout=1800) as r:
                    for line in r:
                        if not line.startswith(b"data: "):
                            continue
                        ev = json.loads(line[6:])
                        if ttft is None and ev.get("content"):
                            ttft = time.time() - t0
                        if ev.get("stop"):
                            timings = ev["timings"]
                lat = time.time() - t0
                reps.append({"ttft_s": ttft, "latency_s": lat, "prompt_n": timings["prompt_n"],
                             "prefill_tps": timings["prompt_per_second"], "decode_tps": timings["predicted_per_second"],
                             "e2e_tps": timings["predicted_n"] / lat,
                             "draft_n": timings.get("draft_n"), "draft_acc": timings.get("draft_n_accepted")})
            med = {k: statistics.median(r[k] for r in reps) for k in ("ttft_s", "latency_s", "prefill_tps", "decode_tps", "e2e_tps")}
            med["prompt_n"] = reps[0]["prompt_n"]
            if reps[0]["draft_n"]:
                med["accept"] = sum(r["draft_acc"] for r in reps) / sum(r["draft_n"] for r in reps)
            rows[str(size)] = med
            print(f"{label:16s} {med['prompt_n']:6d} tok  ttft {med['ttft_s']:6.2f}s  lat {med['latency_s']:6.2f}s  "
                  f"prefill {med['prefill_tps']:6.1f}  out {med['decode_tps']:5.1f} t/s  e2e {med['e2e_tps']:5.1f}"
                  + (f"  acc {med['accept']:.0%}" if "accept" in med else ""), flush=True)
    finally:
        stop.set(); p.terminate(); p.wait()
        for _ in range(60):
            if int(VRAM.read_text()) < 2_000_000_000: break
            time.sleep(1)
    rows["peak_vram_gib"] = peak["vram"] / 2**30
    rows["peak_gtt_gib"] = peak["gtt"] / 2**30
    print(f"{label:16s} peak VRAM {rows['peak_vram_gib']:.2f} GiB (free {15.98 - rows['peak_vram_gib']:.2f}), "
          f"GTT {rows['peak_gtt_gib']:.2f} GiB", flush=True)
    return rows


res = json.loads(OUT.read_text()) if OUT.exists() else {}
for label in sys.argv[1:]:
    res[label] = run(label)
    OUT.write_text(json.dumps(res, indent=1))
