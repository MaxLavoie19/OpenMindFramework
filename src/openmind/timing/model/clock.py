from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Clock:
    """A player's time: the seconds left, the seconds each step adds, and whether it ran out.

    A clock reaching zero is flagged, as a chess flag falls at zero, and a flagged clock gains nothing, the game being
    lost on time. What is left may be below zero on a flagged clock: by how much the step overran."""

    remaining: float
    increment: float = 0.0
    flagged: bool = False

    def after(self, spent: float) -> "Clock":
        """The clock after a step that took `spent` seconds: flagged if that reached zero, the increment added
        otherwise. A negative time spent, or a step on a clock already flagged, raises ValueError."""
        if spent < 0:
            raise ValueError(f"A step can't take negative time, not {spent} seconds")
        if self.flagged:
            raise ValueError("A clock that ran out has no more steps")
        left = self.remaining - spent
        if left <= 0.0:
            return Clock(left, self.increment, True)
        return Clock(left + self.increment, self.increment)
