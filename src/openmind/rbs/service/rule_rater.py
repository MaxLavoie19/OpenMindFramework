from openmind.expression.model.expression import Expression
from openmind.expression.service.interpreter import Interpreter
from openmind.rbs.model.rule import Rule
from openmind.rbs.model.rule_base import RuleBase
from openmind.world.model.action import Action
from openmind.world.model.state import State


class RuleRater:
    """Rates actions with a rule base: each action gets the expected value of its most specific matching rule."""

    def __init__(self, rule_base: RuleBase, interpreter: Interpreter) -> None:
        self._interpreter = interpreter
        self._rules: dict[str, list[Rule]] = {}
        for rule in rule_base.rules:
            self._rules.setdefault(rule.action, []).append(rule)
        for rules in self._rules.values():
            rules.sort(key=lambda rule: (-len(rule.conditions), -rule.visits))

    def rate(self, state: State, actions: tuple[Action, ...]) -> tuple[float | None, ...]:
        return tuple(
            None if (rule := self.explain(state, action)) is None else rule.expected_value for action in actions
        )

    def explain(self, state: State, action: Action) -> Rule | None:
        """The rule behind an action's rating: the one with the most conditions, then the most visits, that holds."""
        for rule in self._rules.get(action.name, ()):
            if all(self._holds(condition, state, action) for condition in rule.conditions):
                return rule
        return None

    def _holds(self, condition: Expression, state: State, action: Action) -> bool:
        try:
            return self._interpreter.evaluate(condition, state, action) is True
        except KeyError:
            return False
