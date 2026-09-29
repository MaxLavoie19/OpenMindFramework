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
