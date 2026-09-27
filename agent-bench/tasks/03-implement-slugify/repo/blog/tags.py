from .text import slugify


def tag_url(tag):
    return f"/tags/{slugify(tag, max_len=20)}"
