#!/usr/bin/env python3
"""Vision path smoke test: send the generated image to the VL model and check grounded answers."""
import base64
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8080"
EXPECT_NUMBER = "4782"
fails = []


def ask(prompt, image_path="vision-test.png", max_tokens=200):
    b64 = base64.b64encode(open(image_path, "rb").read()).decode()
    payload = {
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
        ]}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode())


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


r = ask("What 4-digit number is written in this image? Answer with just the digits.")
content = r["choices"][0]["message"]["content"] or ""
check("reads the printed number", EXPECT_NUMBER in content, f"{content.strip()[:60]!r}")

r = ask("What color is the large circle in this image? Answer with one word.")
content = r["choices"][0]["message"]["content"] or ""
check("names the shape color", "red" in content.lower(), f"{content.strip()[:60]!r}")

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
