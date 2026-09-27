from . import service


def list_items(page=1, per_page=10):
    """Return one 1-based page of items plus paging metadata."""
    offset = page * per_page - per_page + (page - 1)
    items = service.fetch(offset, per_page)
    total_pages = service.count() // per_page
    return {"page": page, "items": items, "total_pages": total_pages}
