#!/usr/bin/env python3
"""Merge bench-quant.py + strict A/B jsonl runs into bench-data.json for the HTML report."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TAGS = {"models/Qwen3.8-27B-UD-IQ4_XS.gguf": "iq4xs", "models/Qwen3.8-27B-UD-Q3_K_XL.gguf": "q3kx"}
NAMES = {"iq4xs": "UD-IQ4_XS", "q3kx": "UD-Q3_K_XL"}


def test_key(r):
    """jsonl output omits the computed 'test' field — rebuild it from n_prompt/n_gen/n_depth."""
    depth = r.get("n_depth", 0)
    if r.get("n_prompt", 0) > 0 and r.get("n_gen", 0) == 0:
        base = f"pp{r['n_prompt']}"
    elif r.get("n_gen", 0) > 0 and r.get("n_prompt", 0) == 0:
        base = f"tg{r['n_gen']}"
    else:
        base = f"mix{r.get('n_prompt', 0)}x{r.get('n_gen', 0)}"
    return f"{base}_d{depth}"


def strict_runs():
    """{tag: {test_key: [t/s per pass]}} from the two interleaved passes."""
    out = {}
    for name in ("strict-pass1.jsonl", "strict-pass2.jsonl"):
        path = ROOT / name
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue  # llama-bench interleaves non-JSON progress lines
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            tag = TAGS.get(r.get("model_filename"))
            if tag is None or "avg_ts" not in r:
                continue
            out.setdefault(tag, {}).setdefault(test_key(r), []).append(round(r["avg_ts"], 2))
    return out


def main():
    strict = strict_runs()
    quants = []
    for tag in ("q3kx", "iq4xs"):
        f = ROOT / f"bench-{tag}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        fit_src = ROOT / f"bench-{tag}-refit.json"
        fit = json.loads((fit_src if fit_src.exists() else f).read_text())["fit"]
        bench = {f"{k}_d0" if "@0" in k else f"{k.replace('@', '_d')}": v
                 for k, v in d["throughput"].items() if isinstance(v, dict)}
        runs = strict.get(tag, {})
        # strict A/B wins where available; keep the r=3 run as fallback/context
        tp = {}
        for key in ("pp512_d0", "tg128_d0", "pp512_d8192", "tg128_d8192"):
            vals = runs.get(key)
            entry = {"single_pass_t_s": bench.get(key, {}).get("t_s"),
                     "single_pass_stdev": bench.get(key, {}).get("stdev")}
            if vals:
                entry.update({"pass_values": vals,
                              "mean_t_s": round(sum(vals) / len(vals), 2),
                              "half_range": round((max(vals) - min(vals)) / 2, 2)})
            tp[key] = entry
        quants.append({
            "tag": tag, "name": NAMES[tag],
            "size_GiB": d["size_GiB"], "size_bytes": d["size_bytes"],
            "bpw": {"q3kx": 3.85, "iq4xs": 4.17}[tag],
            "throughput": tp,
            "fit": {k: v for k, v in fit.items() if k != "smoke_output"},
            "smoke_output": fit.get("smoke_output", []),
            "perplexity": {k: v for k, v in d["perplexity"].items() if k != "tail"},
        })

    extra = []
    alt = ROOT / "bench-iq4xs-16k.json"
    if alt.exists():
        d = json.loads(alt.read_text())
        f = d["fit"]
        extra.append({
            "tag": d["tag"], "model": d["model"], "ctx": d.get("ctx", 16384),
            "size_GiB": d.get("size_GiB"),
            "peak_vram_GiB": f.get("peak_vram_GiB"), "free_at_peak_GiB": f.get("free_at_peak_GiB"),
            "baseline_vram_GiB": d.get("baseline_vram_GiB"),
            "net_GiB": round(f.get("peak_vram_GiB", 0) - d.get("baseline_vram_GiB", 0), 2),
            "note": "same IQ4_XS weights, 16k context — the safe-headroom profile",
        })

    # server-protocol configs (512-in / 512-out), including MTP speculative decoding
    optimization = []
    for p in sorted(ROOT.glob("opt-r3-*.json")):
        d = json.loads(p.read_text())
        if "error" in d or "decode_t_s" not in d:
            continue
        label = d["label"].removeprefix("r3-")
        optimization.append({
            "label": label,
            "decode_t_s": d["decode_t_s"], "prefill_t_s": d["prefill_t_s"],
            "peak_vram_GiB": d["peak_vram_GiB"], "free_at_peak_GiB": d["free_at_peak_GiB"],
            "peak_gtt_GiB": d["peak_gtt_GiB"],
            "prompt_tokens": d["prompt_tokens"], "predicted_tokens": d["predicted_tokens"],
            "mtp": "--spec-draft-model" in d["cmd"],
            "ctx": int(next(a for i, a in enumerate(d["cmd"].split()) if d["cmd"].split()[i - 1] == "--ctx-size")),
            "quants": "IQ4_XS" if "IQ4_XS" in d["cmd"] else "Q3_K_XL",
            "env": d.get("env", []),
        })

    target_run = {
        "source": ("https://www.localmaxxing.com/en/hardware/DISCRETE_GPU%3Arx%207800%20xt"
                   "?name=RX+7800+XT&run=cmt0nfmo00f1jms01zkdjkvc7"),
        "decode_t_s": 28.0, "prefill_t_s": 500.0,
        "quantization": "Unsloth-Dynamic-IQ4_XS", "engine": "llama.cpp",
        "context": 2048, "prompt_tokens": 512, "output_tokens": 512,
        "gpu": "RX 7800 XT", "cpu": "Ryzen 7 7700", "ram_Gb": 32, "os": "CachyOS",
        "flags": "-c 154000 -b 2048 -ub 256 -fa on -ctk q4_0 -ctv q4_0 --parallel 1",
    }

    data = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": {
            "gpu": "AMD Radeon RX 7800 XT (RADV NAVI32, gfx1101)",
            "vram_GiB": round(17163091968 / 2**30, 2),
            "desktop_baseline_GiB": 0.82,
            "ram_GiB": 15,
            "cpu": "AMD Ryzen 7 3700X (8C/16T)",
            "backend": "llama.cpp Vulkan (RADV / Mesa 26.2.2)",
            "llama_cpp_commit": "fb27a52",
        },
        "protocol": {
            "throughput": ("llama-bench, both models interleaved in one invocation, 5 reps, 2 s delay, "
                           "t=8 poll=0; two passes with reversed model order; mean of passes reported"),
            "fit": ("llama-server --ctx-size 32768, q8_0 KV, ub 256, fa on, --n-gpu-layers 99, "
                    "peak VRAM sampled at 0.25 s during an 18.7k-token recall request"),
            "quality": "llama-perplexity on the same ~12x4096-token corpus slice, identical flags",
        },
        "quants": quants,
        "extra_profiles": extra,
        "optimization": optimization,
        "target_run": target_run,
        "reference": [
            {"source": "https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks.md",
             "model": "Qwen3.5 (analogue)", "q3_k_xl": {"ppl": 6.7245, "kld_mean": 0.0308},
             "iq4_xs": {"ppl": 6.6447, "kld_mean": 0.0235},
             "note": "different model size — direction only, not this 27B on this box"},
            {"source": "https://unsloth.ai/docs/models/qwen3.8-next.md",
             "model": "Qwen3.8-Flash-Next 125B MoE (analogue)",
             "q3_k_xl": {"top1_acc_pct": 88.315, "kld_mean": 0.106504},
             "iq4_xs": {"top1_acc_pct": 89.554, "kld_mean": 0.08363},
             "note": "analogue model; shows the same direction (IQ4_XS more accurate)"},
        ],
    }
    out = ROOT / "bench-data.json"
    out.write_text(json.dumps(data, indent=2))
    print(f"wrote {out}")
    for q in quants:
        tp = q["throughput"]
        print(f"{q['name']}: size={q['size_GiB']} GiB  "
              f"pp512={tp['pp512_d0'].get('mean_t_s')} tg128={tp['tg128_d0'].get('mean_t_s')} "
              f"pp512@8k={tp['pp512_d8192'].get('mean_t_s')} tg128@8k={tp['tg128_d8192'].get('mean_t_s')} "
              f"peakVRAM={q['fit'].get('peak_vram_GiB')} ppl={q['perplexity'].get('ppl')}")


if __name__ == "__main__":
    main()
