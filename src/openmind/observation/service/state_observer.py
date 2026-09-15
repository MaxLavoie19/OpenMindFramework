import math

from openmind.observation.constant.observation_constant import HIDDEN, PLAYER
from openmind.observation.model.observation import Observation
from openmind.rule.service.rule_caller import RuleCaller
from openmind.world.model.state import State


class StateObserver:
    """What a player sees of a state, by an observation, and the states that could be true given what they see."""

    def __init__(self, rule_caller: RuleCaller) -> None:
        self._rule_caller = rule_caller

    def observe(self, observation: Observation, state: State, player: str) -> State:
        """The state with every variable hidden from the player showing HIDDEN; a hidden name the state doesn't have
        raises KeyError."""
        hidden = self._hidden(observation, state, player)
        return State(tuple((name, HIDDEN if name in hidden else value) for name, value in state.variables))

    def completions(self, observation: Observation, observed: State, player: str) -> tuple[tuple[State, float], ...]:
        """Each state that could be true given what the player sees, with its probability: what they see, with the values
        of one completion in its hidden variables. A completion that doesn't give exactly the hidden variables, or
        probabilities that don't sum to 1, raise ValueError."""
        hidden = self._hidden(observation, observed, player)
        given = self._rule_caller.value(observation.completions, observed, {PLAYER: player}, None, observation.definitions)
        completions: list[tuple[State, float]] = []
        for values, probability in given:  # type: ignore[attr-defined]
            if set(values) != hidden:
                raise ValueError(
                    f"A completion for {player} gives {sorted(values)}, not the hidden variables {sorted(hidden)}"
                )
            variables = tuple((name, values[name] if name in hidden else value) for name, value in observed.variables)
            completions.append((State(variables), float(probability)))
        total = math.fsum(probability for _, probability in completions)
        if not math.isclose(total, 1.0):
            raise ValueError(f"The completions for {player} have probabilities summing to {total}, not 1")
        return tuple(completions)

    def _hidden(self, observation: Observation, state: State, player: str) -> frozenset[str]:
        given = self._rule_caller.value(observation.hidden, state, {PLAYER: player}, None, observation.definitions)
        hidden = frozenset(given)  # type: ignore[arg-type]
        if unknown := sorted(hidden - {name for name, _ in state.variables}):
            raise KeyError(f"Unknown state variable: {unknown[0]!r}")
        return hidden
