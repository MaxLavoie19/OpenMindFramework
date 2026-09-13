from typing import Protocol

from openmind.world.model.action import Action
from openmind.world.model.state import State


class ActionRater(Protocol):
    """A model rating actions: each one's expected payoff for the player to act, or None where it knows nothing."""

    def rate(self, state: State, actions: tuple[Action, ...]) -> tuple[float | None, ...]: ...
