#!/usr/bin/env python3
"""Measure a llama-server configuration the way localmaxxing does: real 512-in/512-out requests.

- Prompt is padded to ~512 tokens so prefill tok/s is meaningful (not request-overhead noise).
- VRAM and GTT are sampled continuously (50 ms) by a background thread, so transient peaks and
  any spill into system memory are actually observed rather than aliased away.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BIN = ROOT / "llama.cpp/build/bin/llama-server"
VRAM = Path("/sys/class/drm/card1/device/mem_info_vram_used")
GTT = Path("/sys/class/drm/card1/device/mem_info_gtt_used")
TOTAL_VRAM = int(Path("/sys/class/drm/card1/device/mem_info_vram_total").read_text())
GiB = 2**30

TASK = ("Write a detailed technical explanation of how flash attention reduces memory traffic "
        "during transformer inference, covering the tiling strategy and the online softmax "
        "recurrence, then explain how grouped-query attention interacts with it.")
FILLER = [f"Background note {i}: routine system telemetry, sensor group {i % 11}, "
          f"sample value {(i * 7919) % 9973}." for i in range(19)]
PROMPT = "\n".join(FILLER) + "\n\n" + TASK


class Sampler(threading.Thread):
    def __init__(self, interval=0.05):
        super().__init__(daemon=True)
        self.interval, self.stop_at = interval, False
        self.vram = self.gtt = 0
        self.vram_max = self.gtt_max = 0
        self.n = 0

    def run(self):
        while not self.stop_at:
            try:
                self.vram = int(VRAM.read_text())
                self.gtt = int(GTT.read_text())
            except OSError:
                continue
            self.vram_max = max(self.vram_max, self.vram)
            self.gtt_max = max(self.gtt_max, self.gtt)
            self.n += 1
            time.sleep(self.interval)


def get(path, port, timeout=5):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as r:
        return json.loads(r.read())


def wait_ready(port, proc, timeout=420):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return None
        try:
            if get("/health", port, 2).get("status") == "ok":
                return time.time() - t0
        except Exception:
            pass
        time.sleep(0.5)
    return None


def completion(port, n_predict, timeout=1200):
    payload = {"prompt": PROMPT, "n_predict": n_predict, "temperature": 1.0,
               "top_p": 0.95, "top_k": 20, "min_p": 0.0, "cache_prompt": False}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--kt", default="q8_0")
    ap.add_argument("--vt", default="q8_0")
    ap.add_argument("--ub", type=int, default=256)
    ap.add_argument("--draft", default="")
    ap.add_argument("--spec-type", default="")
    ap.add_argument("--spec-nmax", type=int, default=0)
    ap.add_argument("--extra", default="")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n-predict", type=int, default=512)
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    cmd = [str(BIN), "--model", args.model, "--n-gpu-layers", "99",
           "--ctx-size", str(args.ctx), "--flash-attn", "on",
           "--cache-type-k", args.kt, "--cache-type-v", args.vt,
           "--ubatch-size", str(args.ub), "--batch-size", "2048",
           "--parallel", "1", "--host", "127.0.0.1", "--port", str(args.port)]
    if args.draft:
        cmd += ["--spec-draft-model", args.draft]
    if args.spec_type:
        cmd += ["--spec-type", args.spec_type]
    if args.spec_nmax:
        cmd += ["--spec-draft-n-max", str(args.spec_nmax)]
    if args.extra:
        cmd += args.extra.split()

    env = os.environ.copy()
    for e in args.env:
        k, _, v = e.partition("=")
        env[k] = v

    log_path = ROOT / f"opt-{args.label}.log"
    log = open(log_path, "w")
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
    ready = wait_ready(args.port, proc)
    if ready is None:
        proc.kill()
        log.close()
        print(json.dumps({"label": args.label, "error": "server failed to start",
                          "log": log_path.name}), flush=True)
        return 1

    sampler = Sampler()
    sampler.start()
    results = []
    try:
        completion(args.port, 32)          # warm-up, discarded
        for _ in range(args.reps):
            t = completion(args.port, args.n_predict)["timings"]
            results.append({"prefill": t["prompt_per_second"], "decode": t["predicted_per_second"],
                            "prompt_tokens": t["prompt_n"], "predicted_tokens": t["predicted_n"]})
    finally:
        sampler.stop_at = True
        sampler.join(timeout=5)
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()

    out = {
        "label": args.label,
        "ready_s": round(ready, 1),
        "prefill_t_s": round(statistics.median(r["prefill"] for r in results), 2),
        "decode_t_s": round(statistics.median(r["decode"] for r in results), 2),
        "prefill_all": [round(r["prefill"], 1) for r in results],
        "decode_all": [round(r["decode"], 1) for r in results],
        "prompt_tokens": results[0]["prompt_tokens"],
        "predicted_tokens": results[0]["predicted_tokens"],
        "peak_vram_GiB": round(sampler.vram_max / GiB, 2),
        "free_at_peak_GiB": round((TOTAL_VRAM - sampler.vram_max) / GiB, 2),
        "peak_gtt_GiB": round(sampler.gtt_max / GiB, 3),
        "gtt_final_GiB": round(sampler.gtt / GiB, 3),
        "samples": sampler.n,
        "elapsed_s": round(time.time() - t0, 1),
        "cmd": " ".join(cmd[1:]),
        "env": args.env,
    }
    print(json.dumps(out), flush=True)
    (ROOT / f"opt-{args.label}.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
