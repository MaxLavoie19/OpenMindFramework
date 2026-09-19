from collections.abc import Mapping

from openmind.rbs.model.called_rule import CalledRule
from openmind.rbs.service.rule_caller import RuleCaller
from openmind.world.model.state import State
from openmind.structure.model.value import Value


class ConstraintChecker:
    """Checks a prepared constraint for an action with some parameter values in a state."""

    def __init__(self, rule_caller: RuleCaller) -> None:
        self._rule_caller = rule_caller

    def holds(self, constraint: CalledRule, state: State, action: str, values: Mapping[str, Value]) -> bool:
        """Whether the constraint holds; a constraint must give true or false."""
        result = self._rule_caller.call(constraint, state, values)
        if not isinstance(result, bool):
            raise TypeError(f"Constraint of {action!r} gave {result!r} instead of true or false: {constraint.source}")
        return result
