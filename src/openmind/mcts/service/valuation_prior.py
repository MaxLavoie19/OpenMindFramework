import json
import math

from openmind.mcts.model.position_valuer import PositionValuer
from openmind.mcts.service.softmax import softmax
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.world.model.action import Action
from openmind.world.model.players import Players
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader


class ValuationPrior:
    """Each action's value for the player to act, the valuer's value of its outcomes weighed by their probabilities,
    through a softmax at the temperature. A finished outcome counts at its payoffs; an action with an outcome the valuer
    can't value takes the mean of the actions it can, and with none every action is alike. A temperature of 0 or less
    raises ValueError."""

    name = "value"

    def __init__(
        self,
        valuer: PositionValuer,
        rbs: RuleBasedGame,
        players: Players,
        temperature: float,
        state_reader: StateReader | None = None,
    ) -> None:
        if temperature <= 0.0:
            raise ValueError(f"A prior needs a temperature above 0, not {temperature}")
        self._valuer = valuer
        self._rbs = rbs
        self._players = players
        self._temperature = temperature
        self._state_reader = StateReader() if state_reader is None else state_reader

    def priors(self, state: State, actions: tuple[Action, ...]) -> tuple[float, ...]:
        player = self._players.names.index(self._rbs.acting_player(state))
        return softmax([self._value(state, action, player) for action in actions], self._temperature)

    def describe(self) -> str:
        describe = getattr(self._valuer, "describe", None)
        valuer = json.loads(describe()) if callable(describe) else type(self._valuer).__qualname__
        return json.dumps({"prior": self.name, "temperature": self._temperature, "valuer": valuer})

    def _value(self, state: State, action: Action, player: int) -> float | None:
        parts: list[float] = []
        for outcome, probability in self._rbs.outcomes(state, action).outcomes:
            values = self._finished(outcome) or self._valuer.values(outcome)
            if values is None:
                return None
            parts.append(probability * values[player])
        return math.fsum(parts)

    def _finished(self, state: State) -> tuple[float, ...] | None:
        try:
            return self._state_reader.payoffs(state, self._players)
        except ValueError:
            return None
