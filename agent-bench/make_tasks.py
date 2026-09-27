#!/usr/bin/env python3
"""Generate agent-bench/tasks/<NN-name>/{repo,hidden_tests,reference,task.md}.

reference/ holds whole replacement files (paths relative to repo/) that constitute a correct fix.
Re-running regenerates everything from the definitions below.
"""
import shutil, textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
T = HERE / "tasks"


def d(s):
    return textwrap.dedent(s).lstrip("\n")


TASKS = {}

# 1 ── UTC vs local time cutoff ------------------------------------------------------------------
TASKS["01-utc-window"] = dict(
    task="The dashboard's \"events in the last 24 hours\" count is too low on servers that aren't in UTC: "
         "recent events are missing. Events are stored with UTC timestamps. Find and fix the bug. "
         "Run the tests with `python -m pytest -q`.",
    repo={
        "wildlife/__init__.py": "",
        "wildlife/store.py": d('''
            from dataclasses import dataclass
            from datetime import datetime


            @dataclass
            class Event:
                species: str
                ts: datetime  # naive datetime, always UTC


            class Store:
                def __init__(self):
                    self.events = []

                def add(self, species, ts):
                    self.events.append(Event(species, ts))
            '''),
        "wildlife/stats.py": d('''
            from datetime import datetime, timedelta


            def _cutoff(hours):
                return datetime.now() - timedelta(hours=hours)


            def recent_count(store, hours=24, now=None):
                """Number of events whose (UTC) timestamp is within the last `hours` hours."""
                cutoff = (now - timedelta(hours=hours)) if now is not None else _cutoff(hours)
                return sum(1 for e in store.events if e.ts >= cutoff)


            def by_species(store, hours=24, now=None):
                cutoff = (now - timedelta(hours=hours)) if now is not None else _cutoff(hours)
                out = {}
                for e in store.events:
                    if e.ts >= cutoff:
                        out[e.species] = out.get(e.species, 0) + 1
                return out
            '''),
        "tests/test_stats.py": d('''
            from datetime import datetime, timedelta
            from wildlife.store import Store
            from wildlife.stats import recent_count


            def test_explicit_now():
                s = Store()
                now = datetime(2026, 1, 1, 12)
                s.add("zebra", now - timedelta(hours=1))
                s.add("zebra", now - timedelta(hours=30))
                assert recent_count(s, now=now) == 1
            '''),
    },
    hidden={"test_hidden.py": d('''
        import os, time
        from datetime import datetime, timedelta, timezone
        from wildlife.store import Store
        from wildlife.stats import recent_count, by_species


        def _utcnow():
            return datetime.now(timezone.utc).replace(tzinfo=None)


        def test_default_now_is_utc_west(monkeypatch):
            monkeypatch.setenv("TZ", "America/Los_Angeles"); time.tzset()
            try:
                s = Store()
                s.add("lion", _utcnow() - timedelta(hours=2))
                s.add("lion", _utcnow() - timedelta(hours=25))
                assert recent_count(s) == 1
                assert by_species(s) == {"lion": 1}
            finally:
                monkeypatch.delenv("TZ"); time.tzset()


        def test_default_now_is_utc_east(monkeypatch):
            monkeypatch.setenv("TZ", "Africa/Nairobi"); time.tzset()
            try:
                s = Store()
                s.add("elephant", _utcnow() - timedelta(minutes=10))
                assert recent_count(s) == 1
                assert by_species(s) == {"elephant": 1}
            finally:
                monkeypatch.delenv("TZ"); time.tzset()
        ''')},
    ref={"wildlife/stats.py": d('''
        from datetime import datetime, timedelta, timezone


        def _cutoff(hours):
            return datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hours)


        def recent_count(store, hours=24, now=None):
            """Number of events whose (UTC) timestamp is within the last `hours` hours."""
            cutoff = (now - timedelta(hours=hours)) if now is not None else _cutoff(hours)
            return sum(1 for e in store.events if e.ts >= cutoff)


        def by_species(store, hours=24, now=None):
            cutoff = (now - timedelta(hours=hours)) if now is not None else _cutoff(hours)
            out = {}
            for e in store.events:
                if e.ts >= cutoff:
                    out[e.species] = out.get(e.species, 0) + 1
            return out
        ''')},
)

# 2 ── pagination off-by-one across layers --------------------------------------------------------
TASKS["02-pagination"] = dict(
    task="Users report that paging through /items skips one item between pages and the last page is "
         "sometimes missing. Pages are 1-based. Fix it so every item appears exactly once across pages, "
         "and `total_pages` is right. Run `python -m pytest -q`.",
    repo={
        "shop/__init__.py": "",
        "shop/service.py": d('''
            ITEMS = [f"item-{i}" for i in range(23)]


            def fetch(offset, limit):
                return ITEMS[offset:offset + limit]


            def count():
                return len(ITEMS)
            '''),
        "shop/api.py": d('''
            from . import service


            def list_items(page=1, per_page=10):
                """Return one 1-based page of items plus paging metadata."""
                offset = page * per_page - per_page + (page - 1)
                items = service.fetch(offset, per_page)
                total_pages = service.count() // per_page
                return {"page": page, "items": items, "total_pages": total_pages}
            '''),
        "tests/test_api.py": d('''
            from shop.api import list_items


            def test_first_page():
                r = list_items(1, 10)
                assert r["items"][0] == "item-0"
                assert len(r["items"]) == 10
            '''),
    },
    hidden={"test_hidden.py": d('''
        from shop.api import list_items
        from shop import service


        def test_all_items_exactly_once():
            for per in (1, 3, 5, 10, 23, 30):
                first = list_items(1, per)
                seen = []
                for p in range(1, first["total_pages"] + 1):
                    seen += list_items(p, per)["items"]
                assert seen == service.ITEMS, per


        def test_total_pages():
            assert list_items(1, 10)["total_pages"] == 3
            assert list_items(1, 23)["total_pages"] == 1
            assert list_items(1, 5)["total_pages"] == 5
        ''')},
    ref={"shop/api.py": d('''
        from . import service


        def list_items(page=1, per_page=10):
            """Return one 1-based page of items plus paging metadata."""
            offset = (page - 1) * per_page
            items = service.fetch(offset, per_page)
            total_pages = -(-service.count() // per_page)
            return {"page": page, "items": items, "total_pages": total_pages}
        ''')},
)

# 3 ── implement a missing function used by two callers -------------------------------------------
TASKS["03-implement-slugify"] = dict(
    task="`blog.text.slugify` is still a stub, so post URLs and tag pages are broken. Implement it "
         "according to its docstring; both `blog/posts.py` and `blog/tags.py` depend on it. "
         "Run `python -m pytest -q`.",
    repo={
        "blog/__init__.py": "",
        "blog/text.py": d('''
            def slugify(title, max_len=50):
                """Turn a title into a URL slug.

                - lowercase ASCII; accented Latin letters are reduced to their base letter (é -> e)
                - every run of characters that are not a-z or 0-9 becomes a single "-"
                - no leading or trailing "-"
                - truncated to at most max_len characters, never ending in "-"
                - an empty result becomes "untitled"
                """
                raise NotImplementedError
            '''),
        "blog/posts.py": d('''
            from .text import slugify


            def post_url(post_id, title):
                return f"/posts/{post_id}/{slugify(title)}"
            '''),
        "blog/tags.py": d('''
            from .text import slugify


            def tag_url(tag):
                return f"/tags/{slugify(tag, max_len=20)}"
            '''),
        "tests/test_urls.py": d('''
            from blog.posts import post_url


            def test_simple():
                assert post_url(7, "Hello World") == "/posts/7/hello-world"
            '''),
    },
    hidden={"test_hidden.py": d('''
        from blog.text import slugify
        from blog.tags import tag_url


        def test_rules():
            assert slugify("  Crème Brûlée: 10 Tips!! ") == "creme-brulee-10-tips"
            assert slugify("a---b___c") == "a-b-c"
            assert slugify("!!!") == "untitled"
            assert slugify("") == "untitled"
            s = slugify("word " * 40)
            assert len(s) <= 50 and not s.endswith("-")


        def test_tag():
            assert tag_url("Machine Learning & AI Systems Design") == "/tags/machine-learning-ai"
        ''')},
    ref={"blog/text.py": d('''
        import re
        import unicodedata


        def slugify(title, max_len=50):
            """Turn a title into a URL slug.

            - lowercase ASCII; accented Latin letters are reduced to their base letter (é -> e)
            - every run of characters that are not a-z or 0-9 becomes a single "-"
            - no leading or trailing "-"
            - truncated to at most max_len characters, never ending in "-"
            - an empty result becomes "untitled"
            """
            s = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode().lower()
            s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
            s = s[:max_len].rstrip("-")
            return s or "untitled"
        ''')},
)

# 4 ── rename across the package -------------------------------------------------------------------
TASKS["04-rename-api"] = dict(
    task="Rename the public function `calc_total` in `billing/core.py` to `compute_invoice_total` "
         "everywhere in this package (definitions, call sites, re-exports, tests). Do not leave an alias "
         "with the old name. Run `python -m pytest -q` when done.",
    repo={
        "billing/__init__.py": "from .core import calc_total\n\n__all__ = [\"calc_total\"]\n",
        "billing/core.py": d('''
            def calc_total(lines, tax_rate=0.0):
                subtotal = sum(q * p for q, p in lines)
                return round(subtotal * (1 + tax_rate), 2)
            '''),
        "billing/invoice.py": d('''
            from .core import calc_total


            def render(lines, tax_rate=0.0):
                return f"TOTAL {calc_total(lines, tax_rate):.2f}"
            '''),
        "billing/report.py": d('''
            from . import core


            def monthly(invoices):
                return sum(core.calc_total(lines) for lines in invoices)
            '''),
        "tests/test_billing.py": d('''
            from billing import calc_total
            from billing.invoice import render


            def test_total():
                assert calc_total([(2, 1.5)], 0.1) == 3.3


            def test_render():
                assert render([(1, 2.0)]) == "TOTAL 2.00"
            '''),
    },
    hidden={"test_hidden.py": d('''
        import pathlib, billing
        from billing import compute_invoice_total
        from billing.report import monthly


        def test_new_name():
            assert compute_invoice_total([(2, 1.5)], 0.1) == 3.3
            assert monthly([[(1, 1.0)], [(2, 2.0)]]) == 5.0


        def test_old_name_gone():
            assert not hasattr(billing, "calc_total")
            root = pathlib.Path(billing.__file__).parent.parent
            for p in list(root.glob("billing/*.py")) + list(root.glob("tests/*.py")):
                assert "calc_total" not in p.read_text(), p
        ''')},
    ref={
        "billing/__init__.py": "from .core import compute_invoice_total\n\n__all__ = [\"compute_invoice_total\"]\n",
        "billing/core.py": d('''
            def compute_invoice_total(lines, tax_rate=0.0):
                subtotal = sum(q * p for q, p in lines)
                return round(subtotal * (1 + tax_rate), 2)
            '''),
        "billing/invoice.py": d('''
            from .core import compute_invoice_total


            def render(lines, tax_rate=0.0):
                return f"TOTAL {compute_invoice_total(lines, tax_rate):.2f}"
            '''),
        "billing/report.py": d('''
            from . import core


            def monthly(invoices):
                return sum(core.compute_invoice_total(lines) for lines in invoices)
            '''),
        "tests/test_billing.py": d('''
            from billing import compute_invoice_total
            from billing.invoice import render


            def test_total():
                assert compute_invoice_total([(2, 1.5)], 0.1) == 3.3


            def test_render():
                assert render([(1, 2.0)]) == "TOTAL 2.00"
            '''),
    },
)

# 5 ── add a CLI flag ------------------------------------------------------------------------------
TASKS["05-cli-flag"] = dict(
    task="Add a `--min-size BYTES` option to the `dusum` CLI (`dusum/cli.py`). When given, files smaller "
         "than BYTES are ignored in both the per-extension totals and the grand total. Default: no "
         "filtering. Add a test for it. Run `python -m pytest -q`.",
    repo={
        "dusum/__init__.py": "",
        "dusum/scan.py": d('''
            import os


            def scan(root):
                """Yield (path, size) for every regular file under root."""
                for dirpath, _, files in os.walk(root):
                    for f in files:
                        p = os.path.join(dirpath, f)
                        yield p, os.path.getsize(p)
            '''),
        "dusum/cli.py": d('''
            import argparse, os
            from .scan import scan


            def summarize(root):
                by_ext, total = {}, 0
                for path, size in scan(root):
                    ext = os.path.splitext(path)[1] or "<none>"
                    by_ext[ext] = by_ext.get(ext, 0) + size
                    total += size
                return by_ext, total


            def main(argv=None):
                ap = argparse.ArgumentParser(prog="dusum")
                ap.add_argument("root")
                a = ap.parse_args(argv)
                by_ext, total = summarize(a.root)
                for ext in sorted(by_ext):
                    print(f"{ext}\\t{by_ext[ext]}")
                print(f"TOTAL\\t{total}")
                return 0
            '''),
        "tests/test_cli.py": d('''
            from dusum.cli import main


            def test_basic(tmp_path, capsys):
                (tmp_path / "a.txt").write_bytes(b"x" * 10)
                main([str(tmp_path)])
                assert "TOTAL\\t10" in capsys.readouterr().out
            '''),
    },
    hidden={"test_hidden.py": d('''
        from dusum.cli import main


        def _tree(p):
            (p / "a.txt").write_bytes(b"x" * 10)
            (p / "b.txt").write_bytes(b"x" * 100)
            (p / "c.py").write_bytes(b"x" * 5)


        def test_min_size(tmp_path, capsys):
            _tree(tmp_path)
            main([str(tmp_path), "--min-size", "10"])
            out = capsys.readouterr().out
            assert "TOTAL\\t110" in out and ".py" not in out and ".txt\\t110" in out


        def test_default(tmp_path, capsys):
            _tree(tmp_path)
            main([str(tmp_path)])
            assert "TOTAL\\t115" in capsys.readouterr().out
        ''')},
    ref={"dusum/cli.py": d('''
        import argparse, os
        from .scan import scan


        def summarize(root, min_size=0):
            by_ext, total = {}, 0
            for path, size in scan(root):
                if size < min_size:
                    continue
                ext = os.path.splitext(path)[1] or "<none>"
                by_ext[ext] = by_ext.get(ext, 0) + size
                total += size
            return by_ext, total


        def main(argv=None):
            ap = argparse.ArgumentParser(prog="dusum")
            ap.add_argument("root")
            ap.add_argument("--min-size", type=int, default=0, metavar="BYTES")
            a = ap.parse_args(argv)
            by_ext, total = summarize(a.root, a.min_size)
            for ext in sorted(by_ext):
                print(f"{ext}\\t{by_ext[ext]}")
            print(f"TOTAL\\t{total}")
            return 0
        ''')},
)

# 6 ── mutable default argument ------------------------------------------------------------------
TASKS["06-mutable-default"] = dict(
    task="Our request handlers leak state between requests: headers added for one request show up on "
         "later, unrelated requests. Track down why and fix it. Run `python -m pytest -q`.",
    repo={
        "web/__init__.py": "",
        "web/request.py": d('''
            class Request:
                def __init__(self, path, headers={}):
                    self.path = path
                    self.headers = headers

                def with_header(self, k, v):
                    self.headers[k] = v
                    return self
            '''),
        "web/handlers.py": d('''
            from .request import Request


            def build(path, auth_token=None):
                r = Request(path)
                if auth_token:
                    r.with_header("Authorization", f"Bearer {auth_token}")
                return r
            '''),
        "tests/test_handlers.py": d('''
            from web.handlers import build


            def test_auth():
                assert build("/a", "t").headers == {"Authorization": "Bearer t"}
            '''),
    },
    hidden={"test_hidden.py": d('''
        from web.handlers import build
        from web.request import Request


        def test_no_leak():
            build("/secret", "tok")
            assert build("/public").headers == {}


        def test_explicit_dict_still_used():
            h = {"X": "1"}
            assert Request("/", h).headers is h
        ''')},
    ref={"web/request.py": d('''
        class Request:
            def __init__(self, path, headers=None):
                self.path = path
                self.headers = {} if headers is None else headers

            def with_header(self, k, v):
                self.headers[k] = v
                return self
        ''')},
)

# 7 ── config parser edge cases ------------------------------------------------------------------
TASKS["07-config-parser"] = dict(
    task="`cfg.parser.parse` mishandles several inputs from our real config files: inline comments, "
         "quoted values containing '#' or '=', and repeated keys (the last one should win). Make the "
         "parser handle these per the module docstring. Run `python -m pytest -q`.",
    repo={
        "cfg/__init__.py": "",
        "cfg/parser.py": d('''
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
            '''),
        "tests/test_parser.py": d('''
            from cfg.parser import parse


            def test_basic():
                assert parse("[a]\\nx = 1\\n") == {"": {}, "a": {"x": "1"}}
            '''),
    },
    hidden={"test_hidden.py": d('''
        from cfg.parser import parse

        SRC = """
        top = 1
        ; comment
        [Server]
        Host = example.com   # the host
        path = "/a#b=c"  ; trailing
        port = 80
        PORT = 8080
        eq = a=b
        """


        def test_all():
            assert parse(SRC) == {
                "": {"top": "1"},
                "Server": {"host": "example.com", "path": "/a#b=c", "port": "8080", "eq": "a=b"},
            }
        ''')},
    ref={"cfg/parser.py": d('''
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
        ''')},
)

# 8 ── bug in module A surfaced by module B's test ------------------------------------------------
TASKS["08-cross-module"] = dict(
    task="`test_report.py` fails: the weekly report shows the wrong average latency. The report code "
         "looks right to me — find the real cause and fix it there. Run `python -m pytest -q`.",
    repo={
        "metrics/__init__.py": "",
        "metrics/window.py": d('''
            class Window:
                """Keeps the last `size` samples."""

                def __init__(self, size):
                    self.size = size
                    self.samples = []

                def push(self, x):
                    self.samples.append(x)
                    if len(self.samples) > self.size:
                        self.samples.pop()

                def mean(self):
                    return sum(self.samples) / len(self.samples) if self.samples else 0.0
            '''),
        "metrics/report.py": d('''
            from .window import Window


            def weekly_latency(samples, window=7):
                w = Window(window)
                for s in samples:
                    w.push(s)
                return round(w.mean(), 2)
            '''),
        "tests/test_report.py": d('''
            from metrics.report import weekly_latency


            def test_weekly():
                assert weekly_latency([100, 1, 2, 3, 4, 5, 6, 7]) == 4.0
            '''),
    },
    hidden={"test_hidden.py": d('''
        from metrics.window import Window
        from metrics.report import weekly_latency


        def test_window_keeps_latest():
            w = Window(3)
            for x in range(10):
                w.push(x)
            assert w.samples == [7, 8, 9]


        def test_report():
            assert weekly_latency([100, 1, 2, 3, 4, 5, 6, 7]) == 4.0
            assert weekly_latency([5]) == 5.0
        ''')},
    ref={"metrics/window.py": d('''
        class Window:
            """Keeps the last `size` samples."""

            def __init__(self, size):
                self.size = size
                self.samples = []

            def push(self, x):
                self.samples.append(x)
                if len(self.samples) > self.size:
                    self.samples.pop(0)

            def mean(self):
                return sum(self.samples) / len(self.samples) if self.samples else 0.0
        ''')},
)


def write(base, files):
    for rel, content in files.items():
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


if T.exists():
    shutil.rmtree(T)
for name, t in TASKS.items():
    b = T / name
    write(b / "repo", t["repo"])
    write(b / "hidden_tests", t["hidden"])
    write(b / "reference", t["ref"])
    (b / "task.md").write_text(t["task"] + "\n")
print(f"wrote {len(TASKS)} tasks to {T}")
