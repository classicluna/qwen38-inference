from web.handlers import build
from web.request import Request


def test_no_leak():
    build("/secret", "tok")
    assert build("/public").headers == {}


def test_explicit_dict_still_used():
    h = {"X": "1"}
    assert Request("/", h).headers is h
