from shop.api import list_items


def test_first_page():
    r = list_items(1, 10)
    assert r["items"][0] == "item-0"
    assert len(r["items"]) == 10
