from collections.abc import Mapping
from operator import itemgetter

from openmind.expression.model.expression import Expression
from openmind.expression.service.interpreter import Interpreter
from openmind.world.model.action import Action
from openmind.world.model.state import State
from openmind.world.model.value import Value


class ConstraintChecker:
    """Evaluates constraints and operands for an action with some parameter values in a state."""

    def __init__(self, interpreter: Interpreter) -> None:
        self._interpreter = interpreter

    def holds(self, constraint: Expression, state: State, action: str, values: Mapping[str, Value]) -> bool:
        """Whether the constraint holds; a constraint must give true or false."""
        result = self.value(constraint, state, action, values)
        if not isinstance(result, bool):
            raise TypeError(f"Constraint of {action!r} gave {result!r} instead of true or false: {constraint!r}")
        return result

    def value(self, expression: Expression, state: State, action: str, values: Mapping[str, Value]) -> Value:
        return self._interpreter.evaluate(
            expression, state, Action(action, tuple(sorted(values.items(), key=itemgetter(0))))
        )
