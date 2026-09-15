from dataclasses import dataclass

from openmind.rhetoric.model.position import Position


@dataclass(frozen=True, slots=True)
class Ethos:
    """A speaker's positions on questions: who the speaker is (effective) or how the speaker shows themselves to one
    audience member (projective)."""

    positions: tuple[Position, ...]
