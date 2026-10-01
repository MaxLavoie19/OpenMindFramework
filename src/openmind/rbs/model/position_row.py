from dataclasses import dataclass

from openmind.world.model.state import State


@dataclass(frozen=True, slots=True)
class PositionRow:
    """A position valued for a player, by name, and the payoff its value is fitted to."""

    state: State
    player: str
    target: float
    #: What somebody who knows makes of this position, or None where nobody was asked.
    #:
    #: **A second anchor beside the payoff, filled by whoever has one to offer.** The payoff is what the game
    #: came to, credited back from the end; a teller gives a number for this board on its own. Nothing here
    #: knows what a teller is — a caller that has one fills this, and a caller that has not leaves it alone.
    #:
    #: None rather than nought, because nought is a real evaluation. A column of them would read as a teller
    #: that called every position level, and anything correlating against it would come back with nothing and
    #: say nothing about why.
    told: float | None = None
