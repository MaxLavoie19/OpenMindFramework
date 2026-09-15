from dataclasses import dataclass

from openmind.rhetoric.model.position import Position


@dataclass(frozen=True, slots=True)
class Tell:
    """A message the planner decides on: telling an addressee positions, in a tone. Tell(joe, disgust, [do I want his
    fish: no, is his fish disgusting: yes])."""

    addressee: str
    tone: str
    positions: tuple[Position, ...]
