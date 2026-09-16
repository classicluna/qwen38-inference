#!/usr/bin/env python3
"""Re-embed bench-data.json into report.html's data block without touching layout code."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
html = (ROOT / "report.html").read_text()
data = json.dumps(json.loads((ROOT / "bench-data.json").read_text()),
                  separators=(",", ":"), ensure_ascii=False)
if "</script" in data.lower():
    sys.exit("refusing: payload would break out of the script tag")

pat = re.compile(r'(<script id="bench-data" type="application/json">)(.*?)(</script>)', re.S)
new, n = pat.subn(lambda m: m.group(1) + data + m.group(3), html)
if n != 1:
    sys.exit(f"expected exactly 1 data block in report.html, found {n}")
(ROOT / "report.html").write_text(new)
print(f"embedded {len(data)} bytes of JSON into report.html")
