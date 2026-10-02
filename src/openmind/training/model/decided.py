from dataclasses import dataclass

from openmind.heuristic.model.node import Node
from openmind.world.model.action import Action


@dataclass(frozen=True, slots=True)
class Decided:
    """One decision a finished game contains, with what the game went on to pay for it.

    **What a played game holds that a position does not.** A position says where the players were; a decision
    says what could be done there, what was done, by whom, and what it came to. All four are needed to ask a
    heuristic that never played whether it would have done the same: the actions on offer are what it rates,
    the action taken is what it is measured against, and the payoff is what agreeing with that player is
    worth.

    `paid` is what the game paid `player`, on the scale the accuracy of a played model already uses: one for a
    win, half for a draw, none for a loss. A drawn game is evidence and calling it a loss for both would be a
    claim neither game made; a game that never finished is not handed here at all, since it paid nobody and
    agreeing with either side would be evidence of nothing.
    """

    node: Node
    offered: tuple[Action, ...]
    taken: Action
    player: str
    paid: float
    #: What a search settled on here, as a chance per action — the expanded game, to be distilled into a
    #: heuristic that has not looked ahead. Empty where nothing searched this position.
    #:
    #: **This is what a heuristic should be selected for agreeing with.** What was played is one move and
    #: carries the whole weight of a result decided forty moves later; what a search concluded is a
    #: distribution over every move on offer, about *this* position, and it is the reading of a heuristic
    #: improved by looking ahead. A heuristic that reproduces it without the looking ahead is the thing the
    #: loop is for, and the next search starts from that better reading.
    #:
    #: **A distribution rather than its best move, because nearly right is worth saying.** Scored against one
    #: move, a heuristic that ranks the top three in the right order counts the same as one that guessed.
    searched: tuple[tuple[Action, float], ...] = ()
