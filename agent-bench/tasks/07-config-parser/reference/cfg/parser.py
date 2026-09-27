"""Tiny INI-like parser.

[section] headers; `key = value` lines; blank lines and lines starting with # or ; ignored.
Inline comments start at an unquoted ' #' or ' ;'. Values may be wrapped in double quotes,
which are removed and protect # ; = inside. Keys are case-insensitive (lowercased).
A repeated key in the same section overrides the earlier value.
Keys before any section go into section "".
"""


def _value(raw):
    raw = raw.strip()
    if raw.startswith('"'):
        end = raw.index('"', 1)
        return raw[1:end]
    for i in range(1, len(raw)):
        if raw[i] in "#;" and raw[i - 1].isspace():
            return raw[:i].strip()
    return raw


def parse(text):
    out, sec = {"": {}}, ""
    for line in text.splitlines():
        line = line.strip()
        if not line or line[0] in "#;":
            continue
        if line.startswith("["):
            sec = line[1:line.index("]")]
            out.setdefault(sec, {})
            continue
        k, v = line.split("=", 1)
        out[sec][k.strip().lower()] = _value(v)
    return out
