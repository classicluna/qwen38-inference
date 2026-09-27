from .text import slugify


def post_url(post_id, title):
    return f"/posts/{post_id}/{slugify(title)}"
