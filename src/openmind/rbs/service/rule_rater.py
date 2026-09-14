from openmind.agent.model.domain import Domain
from openmind.rbs.constant.consequence_constant import ACTION
from openmind.rbs.model.rule import Rule
from openmind.rbs.model.rule_base import RuleBase
from openmind.rbs.service.consequence_library import ConsequenceLibrary
from openmind.rule.model.python_rule import PythonRule
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.world.model.action import Action
from openmind.world.model.state import State


class RuleRater:
    """Rates a domain's actions with a rule base: each action gets the expected value of its first matching rule,
    priority rules first, then the rules with the most conditions, then the most visited. Conditions read the action's
    parameters, the action as `action`, and the consequence library's names."""

    def __init__(
        self,
        rule_base: RuleBase,
        domain: Domain,
        rule_compiler: RuleCompiler,
        rule_runner: RuleRunner,
        consequence_library: ConsequenceLibrary,
    ) -> None:
        self._domain = domain
        self._rule_compiler = rule_compiler
        self._rule_runner = rule_runner
        self._consequence_library = consequence_library
        self._rules: dict[str, list[Rule]] = {}
        for rule in rule_base.rules:
            self._rules.setdefault(rule.action, []).append(rule)
        for rules in self._rules.values():
            rules.sort(key=lambda rule: (not rule.priority, -len(rule.conditions), -rule.visits))

    def rate(self, state: State, actions: tuple[Action, ...]) -> tuple[float | None, ...]:
        names = self._consequence_library.names(self._domain, state)
        return tuple(
            None if (rule := self._explain(state, action, names)) is None else rule.expected_value
            for action in actions
        )

    def explain(self, state: State, action: Action) -> Rule | None:
        """The rule behind an action's rating."""
        return self._explain(state, action, self._consequence_library.names(self._domain, state))

    def _explain(self, state: State, action: Action, names: dict[str, object]) -> Rule | None:
        for rule in self._rules.get(action.name, ()):
            if all(self._holds(condition, state, action, names) for condition in rule.conditions):
                return rule
        return None

    def _holds(self, condition: PythonRule, state: State, action: Action, names: dict[str, object]) -> bool:
        """A condition reading something the state or the action doesn't have doesn't hold."""
        parameters: dict[str, object] = {**dict(action.parameters), ACTION: action}
        compiled = self._rule_compiler.compile_value(condition, parameters)
        try:
            return self._rule_runner.value(compiled, state, parameters, names) is True
        except (KeyError, NameError, TypeError):
            return False
