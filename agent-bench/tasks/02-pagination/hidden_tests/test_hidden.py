from shop.api import list_items
from shop import service


def test_all_items_exactly_once():
    for per in (1, 3, 5, 10, 23, 30):
        first = list_items(1, per)
        seen = []
        for p in range(1, first["total_pages"] + 1):
            seen += list_items(p, per)["items"]
        assert seen == service.ITEMS, per


def test_total_pages():
    assert list_items(1, 10)["total_pages"] == 3
    assert list_items(1, 23)["total_pages"] == 1
    assert list_items(1, 5)["total_pages"] == 5
