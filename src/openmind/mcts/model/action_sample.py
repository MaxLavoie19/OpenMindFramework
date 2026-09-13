from dataclasses import dataclass

from openmind.world.model.action import Action
from openmind.world.model.state import State


@dataclass(frozen=True, slots=True)
class ActionSample:
    """An action expanded in a search tree: the state and player to act, its visits, and its mean payoff for that player."""

    state: State
    player: int
    action: Action
    visits: int
    mean_payoff: float
