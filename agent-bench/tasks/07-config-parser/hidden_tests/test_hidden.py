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
