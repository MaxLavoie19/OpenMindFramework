from collections.abc import Callable

from openmind.agent.constant.agent_constant import FLAGGED
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.rbs.service.rule_caller import RuleCaller
from openmind.timing.model.clock import Clock
from openmind.timing.model.time_control import TimeControl
from openmind.timing.model.time_source import TimeSource
from openmind.timing.service.wall_time_source import WallTimeSource
from openmind.world.model.state import State


class Timekeeper:
    """What a referee needs to keep the players' clocks: a clock for each player, a choice timed, and what the domain says
    running out of time does."""

    def __init__(self, rule_caller: RuleCaller, time_source: TimeSource | None = None) -> None:
        self._rule_caller = rule_caller
        self._time_source = WallTimeSource() if time_source is None else time_source

    def clocks(self, rbs: RuleBasedSystem, control: TimeControl) -> tuple[Clock, ...]:
        """Each player's starting clock, in the order of the players' names. A game that doesn't say what running out
        of time does raises ValueError."""
        if not rbs.timed():
            raise ValueError(f"{rbs.context} can't be played on a clock: it has no timeout rule")
        return tuple(control.clock() for _ in rbs.players().names)

    def timed[T](self, choose: Callable[[], T]) -> tuple[T, float]:
        """What the choice gave, and the seconds it took."""
        started = self._time_source.now()
        chosen = choose()
        return chosen, self._time_source.now() - started

    def flag(self, rbs: RuleBasedSystem, state: State, player: str) -> State:
        """The state after the player's clock ran out, by the game's timeout rule, given the player as `flagged`."""
        flagged = rbs.flagged(state, **{FLAGGED: player})
        if flagged is None:
            raise ValueError(f"{rbs.context} has no timeout rule")
        return flagged
