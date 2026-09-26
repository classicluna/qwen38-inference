#!/usr/bin/env python3
"""Render results/head2head/<A> vs <B> into a self-contained Bauhaus-style report.html."""
import html, json, math, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
R = Path(os.environ.get("H2H_DIR", ROOT / "results" / "head2head"))
A, B = "qwen-iq4", "bonsai-pq2"
NAMES = {A: "Qwen3.8-27B · UD-IQ4_XS", B: "Bonsai 2 27B · PQ2_0"}
RED, BLUE, YEL, INK, PAPER = "#d62828", "#1d3fbb", "#f4c20d", "#111111", "#f2ede3"
SUITES = [("humanevalplus", "HumanEval+", "write working code"),
          ("humanevalfix", "HumanEvalFix", "find & fix a hidden bug"),
          ("mmlupro", "MMLU-Pro", "knowledge & reasoning")]


def load(tag, suite):
    p = R / tag / f"{suite}.jsonl"
    return {r["id"]: r for r in map(json.loads, p.open())} if p.exists() else {}


def sign_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def lm_eval(tag_dir):
    files = sorted(tag_dir.rglob("results*.json")) if tag_dir.exists() else []
    if not files:
        return None
    res = json.loads(files[-1].read_text())["results"]
    return {"GSM8K": res["gsm8k"]["exact_match,strict-match"],
            "IFEval": res["ifeval"]["prompt_level_strict_acc,none"]}


rows = []
for key, title, sub in SUITES:
    a, b = load(A, key), load(B, key)
    ids = sorted(a.keys() & b.keys())
    if not ids:
        continue
    n = len(ids)
    pa = sum(a[i]["pass"] for i in ids) / n
    pb = sum(b[i]["pass"] for i in ids) / n
    oa = sum(a[i]["pass"] and not b[i]["pass"] for i in ids)
    ob = sum(b[i]["pass"] and not a[i]["pass"] for i in ids)
    mean = lambda d, f: sum(d[i].get(f) or 0 for i in ids) / n
    cut = lambda d: sum(d[i].get("finish") == "length" or d[i]["detail"] in ("no code block", "request error")
                        or "got=None" in d[i]["detail"] for i in ids)
    rows.append(dict(key=key, title=title, sub=sub, n=n, pa=pa, pb=pb, oa=oa, ob=ob, p=sign_p(oa, ob),
                     ta=mean(a, "completion_tokens"), tb=mean(b, "completion_tokens"),
                     wa=mean(a, "wall_s"), wb=mean(b, "wall_s"), ca=cut(a), cb=cut(b)))

lm = {A: lm_eval(ROOT / "results/20260917T003336Z/quality/lm-eval-baseline-100"),
      B: lm_eval(R / B / "lm-eval-bonsai-pq2-100")}

# ---- verdict ------------------------------------------------------------------------------------
sig = [r for r in rows if r["p"] < 0.05]
gap = sum(r["pa"] - r["pb"] for r in rows) / len(rows) if rows else 0
if not rows:
    verdict = "NO DATA YET"
elif all(r["pa"] >= r["pb"] for r in rows) and sig:
    verdict = f"QWEN WINS · {gap*100:+.1f} PTS AVG"
elif all(r["pb"] >= r["pa"] for r in rows) and sig:
    verdict = f"BONSAI WINS · {-gap*100:+.1f} PTS AVG"
else:
    verdict = f"TOO CLOSE TO CALL · {gap*100:+.1f} PTS"
complete = all(r["n"] == (140 if r["key"] == "mmlupro" else 164) for r in rows) and len(rows) == 3


def bar(v, color, label):
    w = v * 100
    return (f'<div class="barrow"><div class="bar"><div class="fill" style="width:{w:.1f}%;background:{color}"></div>'
            f'<span class="who">{label}</span></div><span class="val">{v*100:.1f}%</span></div>')


cards = []
for i, r in enumerate(rows):
    shape = ["circle", "square", "triangle"][i % 3]
    color = [RED, BLUE, YEL][i % 3]
    sigtxt = (f"p = {r['p']:.3g} — significant" if r["p"] < 0.05 else f"p = {r['p']:.2g} — not significant")
    cards.append(f"""
<section class="card">
  <div class="shape {shape}" style="--c:{color}"></div>
  <h2>{html.escape(r['title'])}</h2><p class="sub">{html.escape(r['sub'])} · n = {r['n']}</p>
  {bar(r['pa'], BLUE, 'QWEN')}
  {bar(r['pb'], RED, 'BONSAI')}
  <div class="duel">
    <div><b style="color:{BLUE}">{r['oa']}</b><small>only Qwen solved</small></div>
    <div><b style="color:{RED}">{r['ob']}</b><small>only Bonsai solved</small></div>
    <div><b>{r['cb']}<i>/</i>{r['ca']}</b><small>cut off / no answer<br>Bonsai / Qwen</small></div>
  </div>
  <p class="sig">{sigtxt}</p>
  <p class="cost">{r['ta']:.0f} vs {r['tb']:.0f} tokens · {r['wa']:.1f}s vs {r['wb']:.1f}s per item (Qwen vs Bonsai)</p>
</section>""")

lm_html = ""
if lm[A] and lm[B]:
    items = "".join(f"<div class='lmrow'><span>{k}</span>{bar(lm[A][k], BLUE, 'QWEN')}{bar(lm[B][k], RED, 'BONSAI')}</div>"
                    for k in lm[A])
    lm_html = f"<section class='wide'><h2>SATURATED CONTROLS</h2><p class='sub'>lm-eval, first 100 items each — both near ceiling, shown for completeness</p>{items}</section>"

page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen vs Bonsai — head to head</title>
<style>
:root{{--red:{RED};--blue:{BLUE};--yel:{YEL};--ink:{INK};--paper:{PAPER}}}
*{{box-sizing:border-box;margin:0}}
body{{background:var(--paper);color:var(--ink);font-family:"Futura","Century Gothic","Avenir Next","Jost",
 "Helvetica Neue",Arial,sans-serif;padding:0 0 80px}}
header{{display:grid;grid-template-columns:1.2fr 1fr;border-bottom:12px solid var(--ink);min-height:360px}}
.hl{{padding:48px;display:flex;flex-direction:column;justify-content:space-between}}
h1{{font-size:clamp(48px,8vw,112px);line-height:.85;font-weight:900;letter-spacing:-.04em;text-transform:uppercase}}
h1 span{{display:block}} h1 .r{{color:var(--red)}} h1 .b{{color:var(--blue)}}
.kicker{{font-weight:700;letter-spacing:.3em;text-transform:uppercase;font-size:13px}}
.hr{{background:var(--ink);position:relative;overflow:hidden}}
.hr .c{{position:absolute;width:62%;aspect-ratio:1;border-radius:50%;background:var(--red);left:-10%;top:12%}}
.hr .s{{position:absolute;width:40%;aspect-ratio:1;background:var(--yel);right:8%;top:8%}}
.hr .t{{position:absolute;right:-4%;bottom:-2%;width:0;height:0;border-left:140px solid transparent;
 border-right:140px solid transparent;border-bottom:240px solid var(--blue)}}
.verdict{{background:var(--yel);border-bottom:12px solid var(--ink);padding:28px 48px;font-size:clamp(28px,4.5vw,56px);
 font-weight:900;letter-spacing:-.02em}}
.verdict small{{display:block;font-size:14px;letter-spacing:.25em;font-weight:700;margin-bottom:6px}}
main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:0;border-bottom:12px solid var(--ink)}}
.card,.wide{{padding:40px 36px;border-right:6px solid var(--ink);position:relative;background:var(--paper)}}
.card:last-child{{border-right:none}}
.wide{{border-right:none;border-bottom:12px solid var(--ink)}}
h2{{font-size:34px;font-weight:900;text-transform:uppercase;letter-spacing:-.02em}}
.sub{{font-size:13px;letter-spacing:.15em;text-transform:uppercase;margin:6px 0 26px}}
.shape{{position:absolute;top:36px;right:36px;width:44px;height:44px;background:var(--c)}}
.circle{{border-radius:50%}} .triangle{{background:none!important;width:0;height:0;border-left:24px solid transparent;
 border-right:24px solid transparent;border-bottom:44px solid var(--c)}}
.barrow{{display:grid;grid-template-columns:1fr 92px;align-items:center;gap:12px;margin:10px 0}}
.bar{{position:relative;height:46px;border:4px solid var(--ink);background:#fff}}
.fill{{height:100%}}
.val{{font-weight:900;font-size:24px;text-align:right}}
.who{{position:absolute;left:10px;top:50%;transform:translateY(-50%);font-weight:900;font-size:13px;letter-spacing:.2em;
 color:#fff}}
.duel{{display:grid;grid-template-columns:repeat(3,1fr);border:4px solid var(--ink);margin-top:22px}}
.duel div{{padding:12px;border-right:4px solid var(--ink)}} .duel div:last-child{{border-right:none}}
.duel b{{font-size:34px;font-weight:900;display:block}} .duel i{{font-style:normal;opacity:.4}}
.duel small{{font-size:11px;letter-spacing:.08em;text-transform:uppercase}}
.sig{{margin-top:18px;font-weight:700}} .cost{{font-size:13px;margin-top:6px;opacity:.75}}
.lmrow{{display:grid;grid-template-columns:120px 1fr 1fr;gap:16px;align-items:center}}
.lmrow span{{font-weight:900;font-size:20px}}
footer{{padding:36px 48px;font-size:13px;line-height:1.7;max-width:1100px}}
footer b{{letter-spacing:.2em}}
.pending{{background:var(--red);color:#fff;padding:10px 48px;font-weight:700;letter-spacing:.2em}}
@media(max-width:800px){{header{{grid-template-columns:1fr}}.hr{{min-height:220px}}.card{{border-right:none;border-bottom:6px solid var(--ink)}}}}
</style></head><body>
<header><div class="hl"><p class="kicker">Local inference · RX 7800 XT · 16 GB</p>
<h1><span class="b">Qwen</span><span>versus</span><span class="r">Bonsai</span></h1>
<p class="kicker">{html.escape(NAMES[A])} &nbsp;■&nbsp; {html.escape(NAMES[B])}</p></div>
<div class="hr"><div class="c"></div><div class="s"></div><div class="t"></div></div></header>
{'' if complete else '<div class="pending">PARTIAL RESULTS — CAMPAIGN STILL RUNNING</div>'}
<div class="verdict"><small>VERDICT</small>{html.escape(verdict)}</div>
<main>{''.join(cards)}</main>
{lm_html}
<footer><b>FAILURE MODE</b><br>
Every Bonsai cut-off (8 of 468 items; Qwen: 0) has the same shape: the 1024-token thinking cap ends, then Bonsai keeps
deliberating inside its visible answer until the 4096-token output limit — e.g. HumanEval/132 repeats
"String '[][[[]' no." until truncated. This is the overthinking loop community reports describe, not wrong knowledge.
</footer>
<footer><b>METHOD</b><br>
Identical server policy for both models: llama-server, greedy (temperature 0), thinking effort <i>low</i> with a hard
1024-token thinking budget, max 4096 output tokens, 1 slot, 48k context, text only. Qwen on Vulkan (q8_0 KV);
Bonsai on PrismML's ROCm fork (f16 KV). Code is executed locally against the benchmark's tests; HumanEvalFix does
not show the tests to the model. MMLU-Pro = first 10 questions of each of 14 categories. "Only X solved" counts
paired items; p is a two-sided exact sign test on those discordant pairs. HumanEval/32's reference solution fails
the Plus tests, so that item is unsolvable for both. Harness: <code>head2head-eval.py</code>.</footer>
</body></html>"""

(R / "report.html").write_text(page)
print(R / "report.html", "complete" if complete else "partial", verdict)
