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

    @property
    def settled(self) -> bool:
        """Whether the constraints and the game agree here, case for case."""
        return not self.allowed and not self.forbade

    @property
    def readable(self) -> str:
        return f"{len(self.allowed)} allowed that are refused, {len(self.forbade)} refused that are allowed"
