from dataclasses import dataclass

from openmind.world.model.action import Action
from openmind.world.model.state import State


@dataclass(frozen=True, slots=True)
class MoveRow:
    """An action valued for the player taking it, and the payoff its value is fitted to.

    `PositionRow` with the action added, and the difference is the whole of why the two tasks are two. A
    position is worth one thing to each player; an action is worth one thing to whoever takes it, and asking
    what my move is worth to my opponent is a question nothing asks.

    **What it is fitted to is a payoff and not a preference.** The rows say what the action was shown to pay,
    never that one action is better than another: a rank would be an opinion about a position, and a payoff is
    a fact about a finished game. Where nothing could show what an action pays, there is no row for it —
    knowing nothing about a move is not the same as the move being worth nothing, and a row saying zero is a
    claim the evidence never made."""

    state: State
    action: Action
    player: str
    target: float
