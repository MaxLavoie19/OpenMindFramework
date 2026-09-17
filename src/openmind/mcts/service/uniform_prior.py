import json

from openmind.world.model.action import Action
from openmind.world.model.state import State


class UniformPrior:
    """Every legal action alike."""

    name = "uniform"

    def priors(self, state: State, actions: tuple[Action, ...]) -> tuple[float, ...]:
        return tuple(1.0 / len(actions) for _ in actions)

    def describe(self) -> str:
        return json.dumps({"prior": self.name})
