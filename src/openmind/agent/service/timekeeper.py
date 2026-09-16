from collections.abc import Callable

from openmind.agent.constant.agent_constant import FLAGGED
from openmind.agent.model.domain import Domain
from openmind.rule.service.rule_caller import RuleCaller
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

    def clocks(self, domain: Domain, control: TimeControl) -> tuple[Clock, ...]:
        """Each player's starting clock, in the order of the players' names. A domain that doesn't say what running out
        of time does raises ValueError."""
        if domain.timeout is None:
            raise ValueError(f"{domain.name} can't be played on a clock: it has no timeout rule")
        return tuple(control.clock() for _ in domain.players.names)

    def timed[T](self, choose: Callable[[], T]) -> tuple[T, float]:
        """What the choice gave, and the seconds it took."""
        started = self._time_source.now()
        chosen = choose()
        return chosen, self._time_source.now() - started

    def flag(self, domain: Domain, state: State, player: str) -> State:
        """The state after the player's clock ran out, by the domain's timeout rule, given the player as `flagged`."""
        if domain.timeout is None:
            raise ValueError(f"{domain.name} has no timeout rule")
        return self._rule_caller.apply(domain.timeout, state, {FLAGGED: player}, domain.transitions.definitions)
