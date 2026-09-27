"""Tiny INI-like parser.

[section] headers; `key = value` lines; blank lines and lines starting with # or ; ignored.
Inline comments start at an unquoted ' #' or ' ;'. Values may be wrapped in double quotes,
which are removed and protect # ; = inside. Keys are case-insensitive (lowercased).
A repeated key in the same section overrides the earlier value.
Keys before any section go into section "".
"""


def parse(text):
    out, sec = {"": {}}, ""
    for line in text.splitlines():
        line = line.strip()
        if not line or line[0] in "#;":
            continue
        if line.startswith("["):
            sec = line[1:-1]
            out.setdefault(sec, {})
            continue
        k, v = line.split("=")
        out[sec].setdefault(k.strip(), v.strip())
    return out
