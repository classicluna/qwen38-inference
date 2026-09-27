from cfg.parser import parse


def test_basic():
    assert parse("[a]\nx = 1\n") == {"": {}, "a": {"x": "1"}}
