from typing import Protocol

from openmind.timing.model.clock import Clock


class TimeBudgetEstimator(Protocol):
    """How long a player's next step may take: the seconds, from the player's clock and the steps it has played so far. A
    flagged clock or negative steps played raise ValueError."""

    def budget(self, clock: Clock, steps_played: int) -> float: ...
