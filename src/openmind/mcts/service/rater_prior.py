import json

from openmind.mcts.model.action_rater import ActionRater
from openmind.mcts.service.softmax import softmax
from openmind.world.model.action import Action
from openmind.world.model.state import State


class RaterPrior:
    """A rater's ratings through a softmax at the temperature: a rating it doesn't give takes the mean of those it does,
    and with none every action is alike. A temperature of 0 or less raises ValueError."""

    name = "rater"

    def __init__(self, rater: ActionRater, temperature: float) -> None:
        if temperature <= 0.0:
            raise ValueError(f"A prior needs a temperature above 0, not {temperature}")
        self._rater = rater
        self._temperature = temperature

    def priors(self, state: State, actions: tuple[Action, ...]) -> tuple[float, ...]:
        return softmax(self._rater.rate(state, actions), self._temperature)

    def describe(self) -> str:
        describe = getattr(self._rater, "describe", None)
        rater = json.loads(describe()) if callable(describe) else type(self._rater).__qualname__
        return json.dumps({"prior": self.name, "temperature": self._temperature, "rater": rater})
