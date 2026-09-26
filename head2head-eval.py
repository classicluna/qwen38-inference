#!/usr/bin/env python3
"""Head-to-head quality eval against whatever OpenAI-compatible llama-server is on --base-url.

Suites (all executed/graded locally, deterministic item order, greedy decoding):
  humanevalplus  — evalplus/humanevalplus (164): write the function, graded by the Plus test suite.
  humanevalfix   — bigcode/humanevalpack python (164): buggy function + docstring, "fix the bug",
                   graded by the original HumanEval tests (tests are NOT shown to the model).
  mmlupro        — TIGER-Lab/MMLU-Pro test, first N per category (default 10 x 14 = 140), 10-way MC.

Results append to <out>/<suite>.jsonl, one row per item; reruns skip finished items (resumable).
Usage: head2head-eval.py --tag qwen-iq4 --suites humanevalplus,humanevalfix,mmlupro
"""
import argparse, json, os, re, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path

import datasets

ROOT = Path(__file__).resolve().parent


def chat(base_url, prompt, max_tokens, timeout):
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": max_tokens, "seed": 42,
    }).encode()
    req = urllib.request.Request(base_url + "/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.load(r)
    msg = resp["choices"][0]["message"]
    return {
        "content": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content") or "",
        "finish": resp["choices"][0].get("finish_reason"),
        "completion_tokens": resp.get("usage", {}).get("completion_tokens"),
        "decode_tps": resp.get("timings", {}).get("predicted_per_second"),
        "wall_s": round(time.time() - t0, 2),
    }


def last_code_block(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return blocks[-1] if blocks else None


def run_python(program, timeout=30):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.py"
        p.write_text(program)
        try:
            r = subprocess.run([sys.executable, str(p)], cwd=d, capture_output=True,
                               text=True, timeout=timeout)
            return r.returncode == 0, r.stderr[-400:]
        except subprocess.TimeoutExpired:
            return False, "timeout"


def ensure_callable(code, prefix, entry_point):
    # Model may return only a body or omit imports: fall back to prompt prefix + its code.
    if code is None:
        return None
    if re.search(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", code, re.M):
        return prefix.split("def ")[0] + code  # keep original imports (typing etc.)
    return prefix + code


CODE_SUFFIX = ("\n\nReturn the complete corrected/implemented function, including its signature and "
               "any imports it needs, in a single ```python code block.")


def items_humanevalplus():
    ds = datasets.load_dataset("evalplus/humanevalplus", split="test")
    for r in ds:
        prompt = ("Implement the following Python function.\n\n```python\n" + r["prompt"] + "```"
                  + CODE_SUFFIX)

        def grade(content, r=r):
            code = ensure_callable(last_code_block(content), r["prompt"], r["entry_point"])
            if code is None:
                return False, "no code block"
            return run_python(code + "\n\n" + r["test"] + f"\n\ncheck({r['entry_point']})\n")
        yield r["task_id"], prompt, grade


def items_humanevalfix():
    ds = datasets.load_dataset("bigcode/humanevalpack", "python", split="test")
    for r in ds:
        buggy = r["prompt"] + r["buggy_solution"]
        prompt = (f"The function `{r['entry_point']}` below has a bug: its behaviour does not match its "
                  "docstring. Find and fix the bug.\n\n```python\n" + buggy + "```" + CODE_SUFFIX)

        def grade(content, r=r):
            code = ensure_callable(last_code_block(content), r["prompt"], r["entry_point"])
            if code is None:
                return False, "no code block"
            return run_python(code + "\n\n" + r["test"] + "\n")
        yield r["task_id"], prompt, grade


def items_mmlupro(per_cat):
    ds = datasets.load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    seen = {}
    for r in ds:
        c = r["category"]
        if seen.get(c, 0) >= per_cat:
            continue
        seen[c] = seen.get(c, 0) + 1
        letters = "ABCDEFGHIJ"
        opts = "\n".join(f"({letters[i]}) {o}" for i, o in enumerate(r["options"]))
        prompt = (f"The following is a multiple-choice question about {c}.\n\n{r['question']}\n\n{opts}\n\n"
                  'Finish your reply with exactly "The answer is (X)" where X is the letter of the correct option.')

        def grade(content, r=r):
            m = re.findall(r"answer is \(?([A-J])\)?", content)
            got = m[-1] if m else None
            return got == r["answer"], f"got={got} want={r['answer']}"
        yield f"{c}/{r['question_id']}", prompt, grade


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--suites", default="humanevalplus,humanevalfix,mmlupro")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--ids", default="", help="comma list of item ids to run (default: all)")
    ap.add_argument("--mmlu-per-cat", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=1200)
    a = ap.parse_args()
    out = ROOT / "results" / "head2head" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    gens = {"humanevalplus": items_humanevalplus, "humanevalfix": items_humanevalfix,
            "mmlupro": lambda: items_mmlupro(a.mmlu_per_cat)}
    for suite in a.suites.split(","):
        path = out / f"{suite}.jsonl"
        done = {json.loads(l)["id"] for l in path.open()} if path.exists() else set()
        n = 0
        for iid, prompt, grade in gens[suite]():
            if a.ids and iid not in a.ids.split(","):
                continue
            if a.limit and n >= a.limit:
                break
            n += 1
            if iid in done:
                continue
            try:
                res = chat(a.base_url, prompt, a.max_tokens, a.timeout)
                ok, detail = grade(res["content"])
            except Exception as e:  # recorded as a failure, not skipped: errors count against the model
                res, ok, detail = {"error": repr(e)}, False, "request error"
            row = {"id": iid, "pass": ok, "detail": detail, **res}
            with path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[{a.tag}] {suite} {iid} pass={ok} {res.get('wall_s')}s", flush=True)
        rows = [json.loads(l) for l in path.open()]
        print(f"== {a.tag} {suite}: {sum(r['pass'] for r in rows)}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
