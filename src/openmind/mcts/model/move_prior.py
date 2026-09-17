from typing import Protocol

from openmind.world.model.action import Action
from openmind.world.model.state import State


class MovePrior(Protocol):
    """How likely each legal action is to be the one worth following, before searching: one prior per action, in the
    actions' order, summing to 1. `name` says which prior it is in the logs."""

    @property
    def name(self) -> str: ...

    def priors(self, state: State, actions: tuple[Action, ...]) -> tuple[float, ...]: ...
