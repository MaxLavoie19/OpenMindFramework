from collections.abc import Mapping

from openmind.rule.model.compiled_rule import CompiledRule
from openmind.rule.service.rule_runner import RuleRunner
from openmind.world.model.state import State
from openmind.world.model.value import Value


class ConstraintChecker:
    """Checks compiled constraints for an action with some parameter values in a state."""

    def __init__(self, rule_runner: RuleRunner) -> None:
        self._rule_runner = rule_runner

    def holds(self, constraint: CompiledRule, state: State, action: str, values: Mapping[str, Value]) -> bool:
        """Whether the constraint holds; a constraint must give true or false."""
        result = self._rule_runner.value(constraint, state, values)
        if not isinstance(result, bool):
            raise TypeError(
                f"Constraint of {action!r} gave {result!r} instead of true or false: {constraint.rule.source}"
            )
        return result
