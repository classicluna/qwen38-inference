import os, time
from datetime import datetime, timedelta, timezone
from wildlife.store import Store
from wildlife.stats import recent_count, by_species


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def test_default_now_is_utc_west(monkeypatch):
    monkeypatch.setenv("TZ", "America/Los_Angeles"); time.tzset()
    try:
        s = Store()
        s.add("lion", _utcnow() - timedelta(hours=2))
        s.add("lion", _utcnow() - timedelta(hours=25))
        assert recent_count(s) == 1
        assert by_species(s) == {"lion": 1}
    finally:
        monkeypatch.delenv("TZ"); time.tzset()


def test_default_now_is_utc_east(monkeypatch):
    monkeypatch.setenv("TZ", "Africa/Nairobi"); time.tzset()
    try:
        s = Store()
        s.add("elephant", _utcnow() - timedelta(minutes=10))
        assert recent_count(s) == 1
        assert by_species(s) == {"elephant": 1}
    finally:
        monkeypatch.delenv("TZ"); time.tzset()
