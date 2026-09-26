from dataclasses import dataclass

from openmind.world.model.action import Action
from openmind.world.model.change import Change
from openmind.world.model.state import State


@dataclass(frozen=True, slots=True)
class Watched:
    """One action played and what was seen of it: the position before, the action, and the changes it made.

    **The game says what its action did; OMF does not work it out by comparing positions.** Two positions differ
    in some squares and never say which difference was the point — a move that takes a piece and a move onto an
    empty square differ in the same two squares, and a pawn taken in passing stands on neither square the move
    names. A learner given only the before and after is being asked to guess the thing whose conditions it is
    supposed to learn.

    `whole` says whether what was seen is all there was. Chess shows the whole outcome; liar's dice shows a
    player their own dice and nothing else. A prediction about what was not seen cannot be scored when it is
    made, and saying so here is what keeps a learner from reading silence as agreement."""

    where: State
    action: Action
    changes: tuple[Change, ...]
    whole: bool = True
