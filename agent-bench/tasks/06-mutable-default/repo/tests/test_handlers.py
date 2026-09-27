from web.handlers import build


def test_auth():
    assert build("/a", "t").headers == {"Authorization": "Bearer t"}
