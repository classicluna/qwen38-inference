ITEMS = [f"item-{i}" for i in range(23)]


def fetch(offset, limit):
    return ITEMS[offset:offset + limit]


def count():
    return len(ITEMS)
