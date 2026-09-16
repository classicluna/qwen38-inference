#!/usr/bin/env python3
"""Smoke tests for the local Qwen3.8-27B llama-server: thinking, tools, long-context recall.

Run against a live server:  ./run-server.sh &  then  .venv/bin/python smoke-test.py
"""
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8080"


def post(path, payload, timeout=900):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def get(path, timeout=60):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read().decode())


fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


# 1. server up
h = get("/health")
check("health endpoint ok", h.get("status") == "ok", str(h)[:120])

# 2. thinking mode emits a reasoning trace, and the answer is still correct
t0 = time.time()
r = post("/v1/chat/completions", {
    "messages": [{"role": "user", "content": (
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
        "How much does the ball cost? Answer with just the number.")}],
    "temperature": 0.0,
    "max_tokens": 2000,
    "chat_template_kwargs": {"reasoning_effort": "low"},
})
msg = r["choices"][0]["message"]
reasoning = msg.get("reasoning_content") or ""
content = msg.get("content") or ""
check("thinking emits reasoning_content", len(reasoning) > 20,
      f"{len(reasoning)} chars, {time.time() - t0:.1f}s")
check("answer correct (0.05)", "0.05" in content, repr(content)[:100])

# 3. native tool calling through the embedded jinja template
tools = [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Get the current weather for a city",
    "parameters": {"type": "object",
                   "properties": {"city": {"type": "string", "description": "City name"}},
                   "required": ["city"]}}}]
r = post("/v1/chat/completions", {
    "messages": [{"role": "user", "content": "What is the weather in Reykjavik right now? Use the get_weather tool."}],
    "tools": tools,
    "temperature": 0.0,
    "max_tokens": 2000,
    "chat_template_kwargs": {"reasoning_effort": "low"},
})
msg = r["choices"][0]["message"]
tc = msg.get("tool_calls") or []
ok = bool(tc) and tc[0]["function"]["name"] == "get_weather"
check("tool_call emitted", ok, json.dumps(msg.get("tool_calls"))[:160])
if ok:
    check("tool args carry the city",
          "Reykjavik" in str(tc[0]["function"]["arguments"]),
          str(tc[0]["function"]["arguments"])[:120])

# 4. long-context recall across the hybrid (DeltaNet + full attention) stack.
# Scale the haystack to the server's actual context window so this works at 16k and 32k.
NEEDLE = "ORBITAL-MANATEE-7731"
try:
    props = get("/props")
    n_ctx = (props.get("default_generation_settings") or {}).get("n_ctx") or props.get("n_ctx") or 32768
except Exception as e:  # older builds may not expose /props
    print(f"note: /props unavailable ({e}); assuming 32768 ctx")
    n_ctx = 32768
n_lines = max(64, int(n_ctx * 0.55 / 15))  # ~15 tokens per filler line
lines = [f"line {i}: routine telemetry record, channel {i % 7}, value {(i * 37) % 1000}."
         for i in range(n_lines)]
lines.insert(len(lines) // 2, f"IMPORTANT NOTE: the magic passphrase is {NEEDLE}. Remember it.")
print(f"note: ctx={n_ctx}, haystack={n_lines} lines")
prompt = ("\n".join(lines)
          + "\n\nQuestion: what is the magic passphrase? Answer with just the passphrase.")
t0 = time.time()
r = post("/v1/chat/completions", {
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0.0,
    "max_tokens": 200,
    "chat_template_kwargs": {"enable_thinking": False},
})
msg = r["choices"][0]["message"]
check("long-context needle recall", NEEDLE in json.dumps(msg),
      f"{r['usage']['prompt_tokens']} prompt tokens, {time.time() - t0:.1f}s")

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
