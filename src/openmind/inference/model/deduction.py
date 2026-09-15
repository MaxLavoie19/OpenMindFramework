from dataclasses import dataclass

from openmind.world.model.action import Action
from openmind.world.model.state import State


@dataclass(frozen=True, slots=True)
class Deduction:
    """What reasoning about a position proved for the player to act: the best action, each player's payoffs it leads to,
    in the order of the players' names, and the line showing it, each action with the state it led to (the likeliest
    outcome); and how many plies were searched in full. When nothing was proven, the action and payoffs are None and the
    line is empty."""

    state: State
    player: str
    action: Action | None
    payoffs: tuple[float, ...] | None
    line: tuple[tuple[Action, State], ...]
    plies: int

    @property
    def proven(self) -> bool:
        return self.action is not None
