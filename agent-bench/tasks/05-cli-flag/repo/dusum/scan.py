import os


def scan(root):
    """Yield (path, size) for every regular file under root."""
    for dirpath, _, files in os.walk(root):
        for f in files:
            p = os.path.join(dirpath, f)
            yield p, os.path.getsize(p)
