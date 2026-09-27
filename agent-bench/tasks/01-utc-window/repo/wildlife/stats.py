from datetime import datetime, timedelta


def _cutoff(hours):
    return datetime.now() - timedelta(hours=hours)


def recent_count(store, hours=24, now=None):
    """Number of events whose (UTC) timestamp is within the last `hours` hours."""
    cutoff = (now - timedelta(hours=hours)) if now is not None else _cutoff(hours)
    return sum(1 for e in store.events if e.ts >= cutoff)


def by_species(store, hours=24, now=None):
    cutoff = (now - timedelta(hours=hours)) if now is not None else _cutoff(hours)
    out = {}
    for e in store.events:
        if e.ts >= cutoff:
            out[e.species] = out.get(e.species, 0) + 1
    return out
