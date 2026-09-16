from dataclasses import dataclass

from openmind.timing.model.clock import Clock


@dataclass(frozen=True, slots=True)
class TimeControl:
    """What each player starts a game with and gains each step, in seconds: 3 minutes and 2 seconds a move is
    `TimeControl(180, 2)`. A base that isn't above zero, or a negative increment, raises ValueError."""

    base_seconds: float
    increment_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.base_seconds <= 0:
            raise ValueError(f"A time control's base must be above zero, not {self.base_seconds} seconds")
        if self.increment_seconds < 0:
            raise ValueError(f"A time control's increment can't be negative, not {self.increment_seconds} seconds")

    def clock(self) -> Clock:
        """The clock a player starts a game with."""
        return Clock(self.base_seconds, self.increment_seconds)
