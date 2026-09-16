from dataclasses import dataclass

from openmind.timing.model.clock import Clock
from openmind.world.model.action import Action


@dataclass(frozen=True, slots=True)
class MatchGame:
    """One game of a match: the final payoffs in the order of the players' names, the player whose time ran out if any,
    the plies, why it ended when the domain says or a player's time ran out, the actions played where players took
    turns, the seeds it was played from, and on a clock each choice's seconds in order and each player's clock at the
    end."""

    payoffs: tuple[float, ...]
    flagged: str | None
    plies: int
    ending: str | None
    actions: tuple[Action, ...]
    policy_seed: int
    outcome_seed: int
    seconds: tuple[float, ...] = ()
    clocks: tuple[Clock, ...] = ()
