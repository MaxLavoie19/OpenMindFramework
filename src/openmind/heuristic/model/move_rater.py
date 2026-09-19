from typing import Protocol

from openmind.world.model.action import Action
from openmind.world.model.state import State


class MoveRater[Model](Protocol):
    """The move value task: what each action is worth to the player taking it, in the actions' order, None for an
    action the model knows nothing about. A stateless service fills it, given the model it runs."""

    def rate(
        self, model: Model, state: State, actions: tuple[Action, ...], player: str
    ) -> tuple[float | None, ...]: ...
