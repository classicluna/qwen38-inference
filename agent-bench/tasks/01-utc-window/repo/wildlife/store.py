from dataclasses import dataclass
from datetime import datetime


@dataclass
class Event:
    species: str
    ts: datetime  # naive datetime, always UTC


class Store:
    def __init__(self):
        self.events = []

    def add(self, species, ts):
        self.events.append(Event(species, ts))
