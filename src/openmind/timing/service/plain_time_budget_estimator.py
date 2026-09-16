import json
import logging

from openmind.timing.model.clock import Clock

logger = logging.getLogger(__name__)


class PlainTimeBudgetEstimator:
    """The plain rule: the time left over the steps still expected, plus the increment, but never more than the time left,
    since the increment only comes once the step is done. `expected_steps` is how many steps are still expected at any
    point of a game, not a game's length; below 1 raises ValueError. The steps played change nothing here."""

    def __init__(self, expected_steps: int) -> None:
        if expected_steps < 1:
            raise ValueError(f"At least 1 step needs to be expected, not {expected_steps}")
        self._expected_steps = expected_steps

    def describe(self) -> str:
        """The rule and its steps expected, as JSON."""
        return json.dumps({"rule": "plain", "expected_steps": self._expected_steps})

    def budget(self, clock: Clock, steps_played: int) -> float:
        if clock.flagged:
            raise ValueError("A clock that ran out has no budget left")
        if steps_played < 0:
            raise ValueError(f"Steps played can't be negative, not {steps_played}")
        planned = clock.remaining / self._expected_steps + clock.increment
        budget = min(planned, clock.remaining)
        logger.info(
            "A step may take %.3f seconds: %.1f seconds left over %d steps expected, plus a %g second increment%s",
            budget,
            clock.remaining,
            self._expected_steps,
            clock.increment,
            ", limited to the time left" if planned > clock.remaining else "",
        )
        return budget
