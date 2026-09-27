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
