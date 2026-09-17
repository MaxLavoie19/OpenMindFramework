import json
import logging

from openmind.timing.model.clock import Clock

logger = logging.getLogger(__name__)


class PlainTimeBudgetEstimator:
    """The plain rule: the time left above the reserve over the steps still expected, plus the increment, but never more
    than the time left above the reserve, since the increment only comes once the step is done; 0 once the clock is at
    or below the reserve. `expected_steps` is how many steps are still expected at any point of a game, not a game's
    length; below 1 raises ValueError, and so does a negative reserve. The steps played change nothing here."""

    def __init__(self, expected_steps: int, reserve_seconds: float = 0.0) -> None:
        if expected_steps < 1:
            raise ValueError(f"At least 1 step needs to be expected, not {expected_steps}")
        if reserve_seconds < 0.0:
            raise ValueError(f"A time reserve can't be negative, not {reserve_seconds}")
        self._expected_steps = expected_steps
        self._reserve = reserve_seconds

    def describe(self) -> str:
        """The rule and its steps expected, as JSON."""
        return json.dumps({"rule": "plain", "expected_steps": self._expected_steps, "reserve_seconds": self._reserve})

    def budget(self, clock: Clock, steps_played: int) -> float:
        if clock.flagged:
            raise ValueError("A clock that ran out has no budget left")
        if steps_played < 0:
            raise ValueError(f"Steps played can't be negative, not {steps_played}")
        usable = clock.remaining - self._reserve
        if usable <= 0.0:
            logger.info(
                "A step may take 0 seconds: %.1f seconds left, at or below the %g second reserve", clock.remaining, self._reserve
            )
            return 0.0
        planned = usable / self._expected_steps + clock.increment
        budget = min(planned, usable)
        logger.info(
            "A step may take %.3f seconds: %.1f seconds left above a %g second reserve over %d steps expected, plus a %g "
            "second increment%s",
            budget,
            usable,
            self._reserve,
            self._expected_steps,
            clock.increment,
            ", limited to the time left" if planned > usable else "",
        )
        return budget
