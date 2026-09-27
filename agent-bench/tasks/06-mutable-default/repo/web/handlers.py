from .request import Request


def build(path, auth_token=None):
    r = Request(path)
    if auth_token:
        r.with_header("Authorization", f"Bearer {auth_token}")
    return r
