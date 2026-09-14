from dataclasses import dataclass

from openmind.world.model.action import Action
from openmind.world.model.state import State


@dataclass(frozen=True, slots=True)
class ActionRow:
    """An action as the searches saw it in a state: its visits over every search that expanded it, its visit-weighted
    mean payoff for the player to act, and its advantage, that mean payoff minus the best among the actions of the same
    state."""

    state: State
    action: Action
    visits: int
    mean_payoff: float
    advantage: float
