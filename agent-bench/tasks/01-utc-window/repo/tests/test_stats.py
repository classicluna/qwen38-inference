from datetime import datetime, timedelta
from wildlife.store import Store
from wildlife.stats import recent_count


def test_explicit_now():
    s = Store()
    now = datetime(2026, 1, 1, 12)
    s.add("zebra", now - timedelta(hours=1))
    s.add("zebra", now - timedelta(hours=30))
    assert recent_count(s, now=now) == 1
