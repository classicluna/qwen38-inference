#!/usr/bin/env python3
"""Identical measurement protocol for one GGUF quant.

  throughput (llama-bench) + load/ready time + 32k-ctx VRAM fit under a real long-context
  workload (smoke-test.py) + perplexity on a fixed corpus -> bench-<tag>.json
"""
import argparse
import json
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BENCH = ROOT / "llama.cpp/build/bin/llama-bench"
PPL_BIN = ROOT / "llama.cpp/build/bin/llama-perplexity"
VRAM = Path("/sys/class/drm/card1/device/mem_info_vram_used")
GTT = Path("/sys/class/drm/card1/device/mem_info_gtt_used")
TOTAL = int(Path("/sys/class/drm/card1/device/mem_info_vram_total").read_text())
GiB = 2**30


def sample_until_exit(proc, interval=0.25, want_gtt=False):
    peak_v = peak_g = 0
    while proc.poll() is None:
        peak_v = max(peak_v, int(VRAM.read_text()))
        if want_gtt:
            peak_g = max(peak_g, int(GTT.read_text()))
        time.sleep(interval)
    return peak_v, peak_g


def throughput(model):
    cmd = [str(BENCH), "-m", str(model), "-ngl", "99", "-fa", "on",
           "-ctk", "q8_0", "-ctv", "q8_0", "-ub", "256",
           "-p", "512", "-n", "128", "-r", "3", "-d", "0,8192", "-o", "json"]
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    peak, _ = sample_until_exit(proc)
    out = proc.stdout.read()
    rows = {}
    try:
        for entry in json.loads(out):
            test = f"{entry['test']}@{entry['n_depth']}"
            rows[test] = {"t_s": entry["avg_ts"], "stdev": entry["stddev_ts"]}
    except Exception:
        rows = {"parse_error": out[-400:]}
    return rows, peak


def fit_test(model, tag, ctx=32768):
    baseline = int(VRAM.read_text())
    env = os.environ | {"MODEL": str(model), "CTX": str(ctx), "PORT": "8080"}
    log = open(ROOT / f"server-{tag}.log", "w")
    t0 = time.time()
    srv = subprocess.Popen(["./run-server.sh"], cwd=ROOT, stdout=log,
                           stderr=subprocess.STDOUT, env=env)
    ready = None
    while time.time() - t0 < 300 and srv.poll() is None:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2) as r:
                if json.loads(r.read()).get("status") == "ok":
                    ready = time.time() - t0
                    break
        except Exception:
            pass
        time.sleep(0.5)
    if ready is None:
        srv.kill()
        log.close()
        return {"error": "server never became ready", "baseline_vram": baseline}

    smoke = subprocess.Popen([".venv/bin/python", "smoke-test.py"], cwd=ROOT,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    peak_v, peak_g = sample_until_exit(smoke, want_gtt=True)
    smoke_out = smoke.stdout.read()

    srv.terminate()
    try:
        srv.wait(timeout=60)
    except subprocess.TimeoutExpired:
        srv.kill()
    log.close()

    needle = re.search(r"needle recall — (\d+) prompt tokens, ([\d.]+)s", smoke_out)
    return {
        "baseline_vram": baseline,
        "ready_s": round(ready, 1),
        "peak_vram": peak_v,
        "peak_gtt": peak_g,
        "peak_vram_GiB": round(peak_v / GiB, 2),
        "free_at_peak_GiB": round((TOTAL - peak_v) / GiB, 2),
        "over_baseline_GiB": round((peak_v - baseline) / GiB, 2),
        "smoke_pass": "ALL PASS" in smoke_out,
        "needle_prompt_tokens": int(needle.group(1)) if needle else None,
        "needle_seconds": float(needle.group(2)) if needle else None,
        "smoke_output": smoke_out.strip().splitlines(),
    }


def perplexity(model):
    cmd = [str(PPL_BIN), "-m", str(model), "-f", str(ROOT / "corpus.txt"),
           "-c", "4096", "--chunks", "12", "-ngl", "99", "-fa", "on",
           "-ctk", "q8_0", "-ctv", "q8_0", "-ub", "256"]
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    peak, _ = sample_until_exit(proc)
    out = proc.stdout.read()
    m = re.search(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)", out)
    return {"ppl": float(m.group(1)) if m else None,
            "ppl_err": float(m.group(2)) if m else None,
            "peak_vram_GiB": round(peak / GiB, 2),
            "tail": out.strip().splitlines()[-3:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--fit-only", action="store_true",
                    help="skip throughput/perplexity; measure the 32k-or-other-ctx VRAM fit only")
    args = ap.parse_args()
    model = Path(args.model)

    result = {
        "tag": args.tag,
        "model": model.name,
        "size_bytes": model.stat().st_size,
        "size_GiB": round(model.stat().st_size / GiB, 2),
        "vram_total_GiB": round(TOTAL / GiB, 2),
        "baseline_vram_GiB": round(int(VRAM.read_text()) / GiB, 2),
        "ctx": args.ctx,
    }

    if args.fit_only:
        print(f"=== {args.tag}: {args.ctx} ctx fit ===", flush=True)
        result["fit"] = fit_test(model, args.tag, args.ctx)
        print(json.dumps({k: v for k, v in result["fit"].items() if k != "smoke_output"}, indent=2), flush=True)
        (ROOT / f"bench-{args.tag}.json").write_text(json.dumps(result, indent=2))
        print(f"wrote bench-{args.tag}.json", flush=True)
        return

    print(f"=== {args.tag}: throughput ===", flush=True)
    rows, peak = throughput(model)
    result["throughput"] = rows
    result["bench_peak_vram_GiB"] = round(peak / GiB, 2)
    print(json.dumps(rows, indent=2), flush=True)

    print(f"=== {args.tag}: 32k fit + smoke ===", flush=True)
    result["fit"] = fit_test(model, args.tag, args.ctx)
    print(json.dumps({k: v for k, v in result["fit"].items() if k != "smoke_output"}, indent=2), flush=True)

    print(f"=== {args.tag}: perplexity ===", flush=True)
    result["perplexity"] = perplexity(model)
    print(json.dumps(result["perplexity"], indent=2), flush=True)

    (ROOT / f"bench-{args.tag}.json").write_text(json.dumps(result, indent=2))
    print(f"wrote bench-{args.tag}.json", flush=True)


if __name__ == "__main__":
    main()
