import math
from collections.abc import Hashable

from openmind.agent.model.domain import Domain
from openmind.mcts.model.hypothesis import Hypothesis
from openmind.observation.constant.observation_constant import HIDDEN, PLAYER
from openmind.observation.service.state_observer import StateObserver
from openmind.rule.model.python_rule import PythonRule
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.world.model.action import Action
from openmind.world.model.state import State


class CompletionTheory:
    """A theory of mind that knows nothing about the other players: it believes what the domain's own completions say.
    The states that could be true are grouped into hypotheses by a label, each weighted by its states' summed
    probabilities. Without a label rule, the label is the values of every variable the player can't see; a label rule,
    a value rule reading a state that could be true, the parameter `player` and the observation's definitions, says
    what the hypotheses are about, such as the other player's hidden choice alone, leaving the rest hidden."""

    def __init__(
        self,
        state_observer: StateObserver,
        rule_compiler: RuleCompiler,
        rule_runner: RuleRunner,
        label: PythonRule | None = None,
    ) -> None:
        self._state_observer = state_observer
        self._rule_compiler = rule_compiler
        self._rule_runner = rule_runner
        self._label = label

    def hypotheses(self, domain: Domain, observed: State, player: str) -> tuple[tuple[Hypothesis, float], ...]:
        """A domain without an observation raises ValueError."""
        observation = domain.observation
        if observation is None:
            raise ValueError("A theory of mind needs a domain with an observation")
        hidden = [name for name, value in observed.variables if value == HIDDEN]
        compiled = (
            None
            if self._label is None
            else self._rule_compiler.compile_value(self._label, (PLAYER,), observation.definitions)
        )
        groups: dict[Hashable, list[tuple[State, float]]] = {}
        for state, probability in self._state_observer.completions(observation, observed, player):
            if compiled is None:
                values = dict(state.variables)
                label: Hashable = tuple((name, values[name]) for name in hidden)
            else:
                label = self._rule_runner.value(compiled, state, {PLAYER: player})  # type: ignore[assignment]
            groups.setdefault(label, []).append((state, probability))
        hypotheses: list[tuple[Hypothesis, float]] = []
        for label, members in groups.items():
            total = math.fsum(probability for _, probability in members)
            if total > 0.0:
                completions = tuple((state, probability / total) for state, probability in members)
                hypotheses.append((Hypothesis(label, completions), total))
        return tuple(hypotheses)

    def strategy(
        self, domain: Domain, state: State, player: str, other: str
    ) -> tuple[tuple[Action, float], ...] | None:
        """No prediction: knowing nothing about the other player, it lets the search find its strategy."""
        return None
