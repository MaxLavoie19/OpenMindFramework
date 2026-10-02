from dataclasses import dataclass

from openmind.world.model.action import Action
from openmind.world.model.state import State


@dataclass(frozen=True, slots=True)
class MoveRow:
    """One action in one position, with what it is worth to the player taking it: a row to fit move rules on.

    **The position value's counterpart, and a different question.** A position row asks what a board is worth;
    this asks what a *move* is worth there, which a move rule answers by reading the action as well as the
    board. One position gives as many rows as it had legal moves.

    **Its target is what a search settled on, which is the whole point.** A move heuristic is what makes the
    next search cheaper: rating moves directly costs one reading where valuing the position each move leads to
    costs one per move, about thirty-five of them in chess. So the thing to learn is which moves the search
    kept coming back to — and that is known, as the chance it settled on, for every position it searched.

    Nothing else states it. What a game paid says who won forty moves later and nothing about which move was
    better here; the move actually played is one bit where a distribution is a shape."""

    state: State
    action: Action
    player: str
    target: float
    #: How well grounded that target is, as a share of an ordinary row's say in the fit; see `PositionRow`.
    #:
    #: A search that settled after a thousand visits has shown more than one that settled after three, and the
    #: fit should hear them accordingly. One is an ordinary row and nought is a row it should not hear.
    certainty: float = 1.0
