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
        print(f"{ext}\t{by_ext[ext]}")
    print(f"TOTAL\t{total}")
    return 0
