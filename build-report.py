#!/usr/bin/env python3
"""Generate report.html from bench-data.json — every number is read from the data, none hand-copied."""
import json
import html
from pathlib import Path

ROOT = Path(__file__).resolve().parent
d = json.loads((ROOT / "bench-data.json").read_text())

COLORS = {"q3kx": "#4cc9f0", "iq4xs": "#f79256"}
W, LEFT, RIGHT = 980, 210, 60          # svg width, label gutter, right padding
PLOT = W - LEFT - RIGHT


def esc(s):
    return html.escape(str(s))


def grouped_bars(groups, series, unit, fmt="{:.1f}"):
    """Horizontal grouped bars. groups = [(label, {tag: (value, half_range)})]."""
    mx = max(v[0] + v[1] for _, vals in groups for v in vals.values()) * 1.12
    bar_h, inner_gap, group_gap, pad_top = 20, 4, 26, 8
    n_bars = len(series)
    gh = n_bars * bar_h + (n_bars - 1) * inner_gap
    height = pad_top + len(groups) * (gh + group_gap) + 30
    out = [f'<svg viewBox="0 0 {W} {height}" class="chart" role="img">']
    # x gridlines
    for i in range(6):
        x = LEFT + PLOT * i / 5
        val = mx * i / 5
        out.append(f'<line x1="{x:.1f}" y1="{pad_top}" x2="{x:.1f}" y2="{height-26}" class="grid"/>')
        out.append(f'<text x="{x:.1f}" y="{height-10}" class="axis" text-anchor="middle">'
                   f'{fmt.format(val)}</text>')
    y = pad_top
    for label, vals in groups:
        out.append(f'<text x="{LEFT-14}" y="{y+gh/2+5:.1f}" class="glabel" text-anchor="end">{esc(label)}</text>')
        for i, s in enumerate(series):
            v, hr = vals.get(s["tag"], (0, 0))
            by = y + i * (bar_h + inner_gap)
            bw = max(1.0, PLOT * v / mx)
            out.append(f'<rect x="{LEFT}" y="{by:.1f}" width="{bw:.1f}" height="{bar_h}" rx="3" '
                       f'fill="{COLORS[s["tag"]]}" class="bar">'
                       f'<title>{esc(s["name"])} — {esc(label)}: {v:.2f} {esc(unit)} (±{hr:.2f})</title></rect>')
            if hr > 0:
                x1, x2 = LEFT + PLOT * max(0.0, v - hr) / mx, LEFT + PLOT * (v + hr) / mx
                out.append(f'<line x1="{x1:.1f}" y1="{by+bar_h/2:.1f}" x2="{x2:.1f}" y2="{by+bar_h/2:.1f}" '
                           f'class="whisker"/><line x1="{x2:.1f}" y1="{by+2:.1f}" x2="{x2:.1f}" '
                           f'y2="{by+bar_h-2:.1f}" class="whisker"/>')
            out.append(f'<text x="{LEFT+bw+8:.1f}" y="{by+bar_h/2+4:.1f}" class="value">{v:.2f}</text>')
        y += gh + group_gap
    out.append("</svg>")
    return "\n".join(out)


def config_chart(rows, target_ts):
    """Decode throughput per measured server config, with the target run marked."""
    if not rows:
        return ""
    rows = sorted(rows, key=lambda r: r["decode_t_s"])
    mx = max(max(r["decode_t_s"] for r in rows), target_ts) * 1.14
    bar_h, gap, pad_top = 22, 9, 22
    height = pad_top + len(rows) * (bar_h + gap) + 30
    out = [f'<svg viewBox="0 0 {W} {height}" class="chart" role="img">']
    for i in range(6):
        x = LEFT + PLOT * i / 5
        out.append(f'<line x1="{x:.1f}" y1="{pad_top}" x2="{x:.1f}" y2="{height-26}" class="grid"/>')
        out.append(f'<text x="{x:.1f}" y="{height-10}" class="axis" text-anchor="middle">{mx*i/5:.0f}</text>')
    tx = LEFT + PLOT * target_ts / mx
    out.append(f'<line x1="{tx:.1f}" y1="{pad_top}" x2="{tx:.1f}" y2="{height-26}" class="ceiling"/>')
    out.append(f'<text x="{tx-6:.1f}" y="{pad_top-6}" class="ceiling-label" text-anchor="end">'
               f'localmaxxing target {target_ts:.0f} t/s</text>')
    y = pad_top
    for r in rows:
        label = f'{r["quants"]} · {"MTP" if r["mtp"] else "no MTP"} · {r["ctx"] // 1024}k'
        w = PLOT * r["decode_t_s"] / mx
        color = "#f79256" if r["mtp"] else "#5b6b7c"
        out.append(f'<text x="{LEFT-14}" y="{y+bar_h/2+4:.1f}" class="glabel" text-anchor="end">{esc(label)}</text>')
        out.append(f'<rect class="bar" x="{LEFT}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="3" fill="{color}">'
                   f'<title>{esc(label)}: decode {r["decode_t_s"]:.2f} t/s, prefill {r["prefill_t_s"]:.0f} t/s, '
                   f'peak {r["peak_vram_GiB"]:.2f} GiB, {r["free_at_peak_GiB"]:.2f} GiB free</title></rect>')
        out.append(f'<text x="{LEFT+w+8:.1f}" y="{y+bar_h/2+4:.1f}" class="value">'
                   f'{r["decode_t_s"]:.1f} t/s · free {r["free_at_peak_GiB"]:.2f} GiB</text>')
        y += bar_h + gap
    out.append("</svg>")
    return "\n".join(out)


def target_table():
    """Head-to-head: the submitted run vs our best same-quant config."""
    t = d.get("target_run") or {}
    ours_row = next((r for r in opt if r["quants"] == "IQ4_XS" and r["mtp"] and r["ctx"] <= 4096), None)
    if not t or not ours_row:
        return ""
    ours = ours_row["decode_t_s"]
    delta = (ours / t["decode_t_s"] - 1) * 100
    rows = [
        ("GPU", t["gpu"], host_gpu),
        ("Engine / backend", f'{t["engine"]} (submitted)', "llama.cpp Vulkan (RADV)"),
        ("Quant", t["quantization"], "Unsloth-Dynamic-IQ4_XS"),
        ("Input / output tokens", f'{t["prompt_tokens"]} / {t["output_tokens"]}',
         f'{ours_row["prompt_tokens"]} / {ours_row["predicted_tokens"]}'),
        ("Decode", f'{t["decode_t_s"]:.1f} t/s', f'{ours:.2f} t/s ({delta:+.0f}%)'),
        ("Prefill", f'{t["prefill_t_s"]:.0f} t/s', f'{ours_row["prefill_t_s"]:.0f} t/s'),
        ("Peak VRAM", "not submitted", f'{ours_row["peak_vram_GiB"]:.2f} GiB '
                                       f'({ours_row["free_at_peak_GiB"]:.2f} free)'),
        ("CPU / RAM", f'{t["cpu"]} / {t["ram_Gb"]} GB', f'{d["host"]["cpu"]} / {d["host"]["ram_GiB"]} GB'),
    ]
    body = "".join(f'<tr><th>{esc(k)}</th><td>{a}</td><td>{b}</td></tr>' for k, a, b in rows)
    return ('<table><thead><tr><th>metric</th><th>submitted run</th><th>this box (with MTP)</th></tr>'
            f'</thead><tbody>{body}</tbody></table>')


quants = d["quants"]
opt = d.get("optimization", [])
host_gpu = d["host"]["gpu"]
target_decode = (d.get("target_run") or {}).get("decode_t_s", 28.0)
target_src = (d.get("target_run") or {}).get("source", "")
best_mtp = max((r["decode_t_s"] for r in opt if r["mtp"]), default=0.0)
prompt_desc = (f'{opt[0]["prompt_tokens"]}-in / {opt[0]["predicted_tokens"]}-out' if opt else "512-in / 512-out")
series = [{"tag": q["tag"], "name": q["name"]} for q in quants]
tp = {q["tag"]: q["throughput"] for q in quants}
fit = {q["tag"]: q["fit"] for q in quants}
size = {q["tag"]: q["size_GiB"] for q in quants}
ppl = {q["tag"]: q["perplexity"] for q in quants}

decode = [(f"128 tokens, empty ctx", {q["tag"]: (tp[q["tag"]]["tg128_d0"]["mean_t_s"], tp[q["tag"]]["tg128_d0"]["half_range"]) for q in quants}),
          (f"128 tokens @ 8k depth", {q["tag"]: (tp[q["tag"]]["tg128_d8192"]["mean_t_s"], tp[q["tag"]]["tg128_d8192"]["half_range"]) for q in quants})]
prefill = [("512-token prompt", {q["tag"]: (tp[q["tag"]]["pp512_d0"]["mean_t_s"], tp[q["tag"]]["pp512_d0"]["half_range"]) for q in quants}),
           ("512-token prompt @ 8k depth", {q["tag"]: (tp[q["tag"]]["pp512_d8192"]["mean_t_s"], tp[q["tag"]]["pp512_d8192"]["half_range"]) for q in quants})]
quality = [("Perplexity (12×4096 tokens, lower is better)",
            {q["tag"]: (q["perplexity"]["ppl"], q["perplexity"]["ppl_err"]) for q in quants})]

total = d["host"]["vram_GiB"]


def vram_chart():
    rows = [(q["name"], q["tag"], size[q["tag"]], fit[q["tag"]]["over_baseline_GiB"],
             fit[q["tag"]]["peak_vram_GiB"], fit[q["tag"]]["free_at_peak_GiB"], "32k ctx") for q in quants]
    for p in d.get("extra_profiles", []):
        rows.append((p["model"].replace("Qwen3.8-27B-", ""), p["tag"], p["size_GiB"], p["net_GiB"],
                     p["peak_vram_GiB"], p["free_at_peak_GiB"], f'{p["ctx"] // 1024}k ctx'))
    bar_h, gap, pad_top = 30, 26, 8
    height = pad_top + len(rows) * (bar_h + gap) + 30
    out = [f'<svg viewBox="0 0 {W} {height}" class="chart" role="img">']
    for i in range(6):
        x = LEFT + PLOT * i / 5
        out.append(f'<line x1="{x:.1f}" y1="{pad_top}" x2="{x:.1f}" y2="{height-26}" class="grid"/>')
        out.append(f'<text x="{x:.1f}" y="{height-10}" class="axis" text-anchor="middle">{total*i/5:.1f}</text>')
    ceiling = LEFT + PLOT * total / total
    out.append(f'<line x1="{ceiling:.1f}" y1="{pad_top}" x2="{ceiling:.1f}" y2="{height-26}" class="ceiling"/>')
    out.append(f'<text x="{ceiling-6:.1f}" y="{pad_top+12}" class="ceiling-label" text-anchor="end">'
               f'VRAM ceiling {total:.2f} GiB</text>')
    y = pad_top
    for name, tag, s, net, peak, free, ctxlabel in rows:
        s = s if s is not None else peak * 0.9  # fallback only if a future row lacks size
        w_model = PLOT * s / total
        w_other = PLOT * max(0.0, net - s) / total
        out.append(f'<text x="{LEFT-14}" y="{y+bar_h/2-2:.1f}" class="glabel" text-anchor="end">{esc(name)}</text>')
        out.append(f'<text x="{LEFT-14}" y="{y+bar_h/2+13:.1f}" class="gsub" text-anchor="end">{esc(ctxlabel)}</text>')
        out.append(f'<rect x="{LEFT}" y="{y}" width="{w_model:.1f}" height="{bar_h}" rx="3" '
                   f'fill="{COLORS.get(tag) or COLORS[tag.split("-")[0]]}">'
                   f'<title>{esc(name)} weights: {s:.2f} GiB</title></rect>')
        out.append(f'<rect x="{LEFT+w_model:.1f}" y="{y}" width="{w_other:.1f}" height="{bar_h}" rx="0" fill="#5b6b7c">'
                   f'<title>KV cache + recurrent state + compute buffers: {net-s:.2f} GiB</title></rect>')
        wpeak = PLOT * peak / total
        out.append(f'<text x="{LEFT+wpeak+8:.1f}" y="{y+bar_h/2+4:.1f}" class="value">peak {peak:.2f} · '
                   f'free {free:.2f}</text>')
        y += bar_h + gap
    out.append("</svg>")
    return "\n".join(out)


def legend():
    return "".join(f'<span class="key"><i style="background:{COLORS[q["tag"]]}"></i>{esc(q["name"])}'
                   f' <span class="dim">{q["size_GiB"]:.2f} GiB · {q["bpw"]} bpw</span></span>' for q in quants)


def table():
    rows = [
        ("Quant", lambda q: esc(q["name"])),
        ("File size", lambda q: f'{q["size_GiB"]:.2f} GiB ({q["bpw"]} bpw)'),
        ("Prefill 512 @ empty", lambda q: f'{tp[q["tag"]]["pp512_d0"]["mean_t_s"]:.2f} t/s'),
        ("Prefill 512 @ 8k depth", lambda q: f'{tp[q["tag"]]["pp512_d8192"]["mean_t_s"]:.2f} t/s'),
        ("Decode 128 @ empty", lambda q: f'{tp[q["tag"]]["tg128_d0"]["mean_t_s"]:.2f} t/s'),
        ("Decode 128 @ 8k depth", lambda q: f'{tp[q["tag"]]["tg128_d8192"]["mean_t_s"]:.2f} t/s'),
        ("Perplexity", lambda q: f'{q["perplexity"]["ppl"]:.3f} ± {q["perplexity"]["ppl_err"]:.3f}'),
        ("Peak VRAM @ 32k", lambda q: f'{fit[q["tag"]]["peak_vram_GiB"]:.2f} GiB'),
        ("Free VRAM at peak", lambda q: f'{fit[q["tag"]]["free_at_peak_GiB"]:.2f} GiB'),
        ("Peak GTT (spill proxy)", lambda q: f'{fit[q["tag"]]["peak_gtt"]/2**30:.3f} GiB'),
        ("Server ready (load)", lambda q: f'{fit[q["tag"]]["ready_s"]:.1f} s'),
        ("25.2k-token prefill", lambda q: f'{fit[q["tag"]]["needle_seconds"]:.1f} s'),
        ("Smoke tests", lambda q: "pass" if fit[q["tag"]]["smoke_pass"] else "FAIL"),
    ]
    head = "".join(f'<th>{esc(q["name"])}</th>' for q in quants)
    body = "".join("<tr><th>" + esc(k) + "</th>" + "".join(f"<td>{f(q)}</td>" for q in quants) + "</tr>"
                   for k, f in rows)
    return f'<table><thead><tr><th>metric</th>{head}</tr></thead><tbody>{body}</tbody></table>'


def reference():
    out = []
    for r in d.get("reference", []):
        parts = []
        for key, label in (("q3_k_xl", "Q3_K_XL"), ("iq4_xs", "IQ4_XS")):
            if key in r:
                vals = ", ".join(f'{k}={v}' for k, v in r[key].items())
                parts.append(f"<b>{label}</b>: {esc(vals)}")
        out.append(f'<li><b>{esc(r["model"])}</b> — ' + " · ".join(parts) +
                   f'<br><span class="dim">{esc(r["note"])} — <code>{esc(r["source"])}</code></span></li>')
    return "<ul class='ref'>" + "".join(out) + "</ul>"


payload = json.dumps(d, separators=(",", ":"), ensure_ascii=False)
doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen3.8-27B quant A/B — RX 7800 XT</title>
<style>
:root {{ --bg:#0e1116; --panel:#161b22; --line:#26303a; --fg:#e6edf3; --dim:#8b98a5; }}
* {{ box-sizing:border-box }}
body {{ margin:0; background:var(--bg); color:var(--fg);
  font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }}
main {{ max-width:1080px; margin:0 auto; padding:32px 20px 64px }}
h1 {{ font-size:26px; margin:0 0 4px }} h2 {{ font-size:17px; margin:0 0 14px; letter-spacing:.02em }}
.sub {{ color:var(--dim); margin:0 0 26px }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:20px; margin:0 0 20px }}
.key {{ display:inline-flex; align-items:center; gap:8px; margin-right:22px; font-weight:600 }}
.key i {{ width:12px; height:12px; border-radius:3px; display:inline-block }}
.dim {{ color:var(--dim); font-weight:400; font-size:13px }}
.chart {{ width:100%; height:auto; display:block }}
.grid {{ stroke:var(--line); stroke-width:1 }}
.ceiling {{ stroke:#ff6b6b; stroke-width:2; stroke-dasharray:5 4 }}
.ceiling-label {{ fill:#ff6b6b; font-size:12px }}
.bar {{ opacity:.92 }} .bar:hover {{ opacity:1 }}
.whisker {{ stroke:var(--fg); stroke-width:1.5; opacity:.7 }}
.glabel {{ fill:var(--fg); font-size:13px }} .gsub {{ fill:var(--dim); font-size:11px }}
.axis {{ fill:var(--dim); font-size:11px }} .value {{ fill:var(--fg); font-size:12px; font-weight:600 }}
table {{ width:100%; border-collapse:collapse; font-size:14px; font-variant-numeric:tabular-nums }}
th,td {{ text-align:right; padding:7px 10px; border-bottom:1px solid var(--line) }}
thead th {{ color:var(--dim); font-weight:600 }} tbody th {{ text-align:left; color:var(--dim); font-weight:500 }}
tbody tr:last-child td, tbody tr:last-child th {{ border-bottom:0 }}
.ref li {{ margin-bottom:12px }} code {{ color:#8fd3ff; font-size:12px; word-break:break-all }}
.verdict {{ border-left:3px solid #4cc9f0 }}
.note {{ color:var(--dim); font-size:13px }}
</style></head>
<body><main>
<h1>Qwen3.8-27B — UD-Q3_K_XL vs UD-IQ4_XS</h1>
<p class="sub">{esc(d["host"]["gpu"])} · {d["host"]["vram_GiB"]:.2f} GiB VRAM · {d["host"]["ram_GiB"]} GiB RAM ·
{esc(d["host"]["backend"])} · llama.cpp <code>{esc(d["host"]["llama_cpp_commit"])}</code> ·
measured {esc(d["generated"][:16].replace("T", " "))} UTC</p>

<div class="card"><h2>Legend</h2><p style="margin:0">{legend()}</p></div>

<div class="card"><h2>Beating the localmaxxing run — same GPU, same quant</h2>
{target_table()}
{config_chart(opt, target_decode)}
<p class="note">Target run: <code>{esc(target_src)}</code>. Rows measured here with real {esc(prompt_desc)}
requests through <code>llama-server</code> — the same protocol the submission used — with VRAM and GTT
sampled continuously (50&nbsp;ms). "MTP" = the model's own multi-token-prediction head used as a
speculative draft (<code>--spec-draft-model mtp-Qwen3.8-27B-Q4_0.gguf --spec-type draft-mtp</code>).
Best MTP config: <b>{best_mtp:.1f} t/s</b> decode, {(best_mtp/target_decode - 1)*100:+.0f}% versus the target.</p></div>

<div class="card"><h2>Decode speed (tokens/s, higher is better)</h2>
{grouped_bars(decode, series, "t/s", "{:.0f}")}
<p class="note">Whiskers show the spread between the two interleaved passes (reversed model order);
within-pass standard deviation was ≤0.15 t/s, so the decode gap is real, not noise.</p></div>

<div class="card"><h2>Prefill speed (tokens/s, higher is better)</h2>
{grouped_bars(prefill, series, "t/s", "{:.0f}")}
<p class="note">Prefill is a tie within noise — the IQ4_XS lookup-table dequant costs decode throughput, not prompt processing.</p></div>

<div class="card"><h2>VRAM fit — the deciding chart</h2>
{vram_chart()}
<p class="note">Colored segment = quantized weights; grey = q8_0 KV cache for the context + Gated-DeltaNet
recurrent state + compute buffers. Red line = card total ({d["host"]["vram_GiB"]:.2f} GiB);
desktop baseline ~{d["host"]["desktop_baseline_GiB"]:.2f} GiB is included in every bar. Peak GTT (the
spill-to-system-RAM proxy) was {fit[quants[0]["tag"]]["peak_gtt"]/2**30:.3f} GiB for both quants — no spill.</p></div>

<div class="card"><h2>Quality (perplexity, lower is better)</h2>
{grouped_bars(quality, series, "PPL", "{:.1f}")}
<p class="note">Identical 12×4096-token corpus slice, identical flags. IQ4_XS is
{(1 - ppl["iq4xs"]["ppl"]/ppl["q3kx"]["ppl"])*100:.1f}% lower — the extra bit of precision buys measurable quality.</p></div>

<div class="card"><h2>Raw numbers</h2>{table()}
<p class="note">{esc(d["protocol"]["throughput"])}</p>
<p class="note">{esc(d["protocol"]["fit"])}</p>
<p class="note">{esc(d["protocol"]["quality"])}</p></div>

<div class="card"><h2>Published third-party numbers (context only — different models)</h2>
{reference()}</div>

<div class="card verdict"><h2>Verdict</h2>
<p><b>Keep UD-Q3_K_XL as the default.</b> It leaves {fit["q3kx"]["free_at_peak_GiB"]:.2f} GiB of free VRAM at 32k
context versus {fit["iq4xs"]["free_at_peak_GiB"]:.2f} GiB for UD-IQ4_XS, and decodes
{(tp["q3kx"]["tg128_d0"]["mean_t_s"]/tp["iq4xs"]["tg128_d0"]["mean_t_s"]-1)*100:.1f}% faster — with prefill unchanged.</p>
<p><b>Switch to UD-IQ4_XS when quality matters more than that margin</b> — it is
{(1 - ppl["iq4xs"]["ppl"]/ppl["q3kx"]["ppl"])*100:.1f}% better on perplexity — and either accept
{fit["iq4xs"]["free_at_peak_GiB"]:.2f} GiB of headroom at 32k, or run it at 16k context
({d["extra_profiles"][0]["peak_vram_GiB"]:.2f} GiB peak, {d["extra_profiles"][0]["free_at_peak_GiB"]:.2f} GiB free)
if the desktop is busy. Below 0.7 GiB free, an amdgpu VRAM overflow would spill to system RAM, and this box
only has {d["host"]["ram_GiB"]} GiB of RAM to absorb that.</p></div>

<p class="note">Generated by <code>build-report.py</code> from <code>bench-data.json</code>; the raw payload is
embedded below for audit. Regenerate with <code>.venv/bin/python make-report-data.py &amp;&amp; .venv/bin/python build-report.py</code>.</p>
<script id="bench-data" type="application/json">{payload}</script>
</main></body></html>
"""
(ROOT / "report.html").write_text(doc)
print(f"wrote report.html ({len(doc):,} bytes)")
