from dataclasses import dataclass

from openmind.world.model.action import Action
from openmind.world.model.state import State


@dataclass(frozen=True, slots=True)
class Disagreement:
    """Where what a set of constraints refuses and what the game refuses differ.

    This scores constraints against evidence. It is not how evidence arrives, and nothing is learned from it
    directly: a learner driven by its own mistakes only ever hears about the cases it got wrong, while the position
    it got them wrong in has thousands of others to say why.

    `allowed` are candidates no constraint refused and the game does not list: the constraints say too little, and
    something is missing. `forbade` are candidates some constraint refused and the game lists: a constraint says
    too much, and it is repaired where a reading tells the cases apart and dropped where none does.

    The two are not symmetrical in what they cost. Letting through what the game refuses makes a move OMF would
    play and the game would reject; turning away what the game allows makes a move OMF will never find, and
    nothing will ever tell it what it missed."""

    where: State
    allowed: tuple[Action, ...] = ()
    forbade: tuple[Action, ...] = ()
    #: For each candidate wrongly refused, the constraints that refused it.
    #:
    #: **A refusal nobody can attribute is a refusal nobody can argue with.** Saying that a move the game allows
    #: was turned away names the fault and not the rule, so every time one turned up the only way to the rule
    #: responsible was to write a script asking each constraint in turn. Chasing a single one cost a day.
    #:
    #: Only for the ones wrongly refused, because those are the few that matter — a position offers fourteen
    #: thousand candidates and nearly all of them are rightly refused by something.
    blamed: tuple[tuple[Action, tuple], ...] = ()

    @property
    def settled(self) -> bool:
        """Whether the constraints and the game agree here, case for case."""
        return not self.allowed and not self.forbade

    @property
    def readable(self) -> str:
        return f"{len(self.allowed)} allowed that are refused, {len(self.forbade)} refused that are allowed"

    @property
    def why(self) -> str:
        """The moves wrongly refused, each with the constraints that refused it, for somebody to read."""
        return "; ".join(
            f"{dict(action.parameters)} by {' and '.join(one.readable for one in clauses)}"
            for action, clauses in self.blamed
        )
