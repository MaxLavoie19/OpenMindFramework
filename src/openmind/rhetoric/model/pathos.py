from dataclasses import dataclass

from openmind.rhetoric.model.position import Position


@dataclass(frozen=True, slots=True)
class Pathos:
    """An audience member's positions on questions, what they believe and how much it matters to them: who they really
    are (effective) or what a speaker perceives them to be (projective)."""

    positions: tuple[Position, ...]
