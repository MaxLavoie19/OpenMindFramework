from collections.abc import Callable

from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.timing.model.clock import Clock
from openmind.timing.model.time_control import TimeControl
from openmind.timing.model.time_source import TimeSource
from openmind.timing.service.wall_time_source import WallTimeSource


class Timekeeper:
    """What a referee needs to keep the players' clocks: a clock for each player, and a choice timed. What running out of
    time does is the game's own rule, over its own variables; OMF has none."""

    def __init__(self, time_source: TimeSource | None = None) -> None:
        self._time_source = WallTimeSource() if time_source is None else time_source

    def clocks(self, rbs: RuleBasedGame, control: TimeControl) -> tuple[Clock, ...]:
        """Each player's starting clock, in the order of the players' names."""
        return tuple(control.clock() for _ in rbs.players().names)

    def timed[T](self, choose: Callable[[], T]) -> tuple[T, float]:
        """What the choice gave, and the seconds it took."""
        started = self._time_source.now()
        chosen = choose()
        return chosen, self._time_source.now() - started
