from openmind.expression.model.action_parameter import ActionParameter
from openmind.expression.model.all_different import AllDifferent
from openmind.expression.model.all_of import AllOf
from openmind.expression.model.any_of import AnyOf
from openmind.expression.model.constant import Constant
from openmind.expression.model.equals import Equals
from openmind.expression.model.expression import Expression
from openmind.expression.model.not_ import Not
from openmind.expression.model.state_variable import StateVariable
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.action import Action
from openmind.world.model.state import State
from openmind.world.model.value import Value


class Interpreter:
    """Evaluates expressions against a state and an action; it understands only the expression vocabulary."""

    def __init__(self, variable_name_mapper: VariableNameMapper) -> None:
        self._variable_name_mapper = variable_name_mapper

    def evaluate(self, expression: Expression, state: State, action: Action) -> Value:
        match expression:
            case Constant(value):
                return value
            case StateVariable(base, indices):
                index_values = tuple(self.evaluate(index, state, action) for index in indices)
                name = self._variable_name_mapper.to_name(base, index_values)
                return self._find(state.variables, name, "state variable")
            case ActionParameter(name):
                return self._find(action.parameters, name, f"parameter of {action.name!r}")
            case Equals(left, right):
                return self.evaluate(left, state, action) == self.evaluate(right, state, action)
            case Not(operand):
                return not self.evaluate(operand, state, action)
            case AllOf(operands):
                return all(self.evaluate(operand, state, action) for operand in operands)
            case AnyOf(operands):
                return any(self.evaluate(operand, state, action) for operand in operands)
            case AllDifferent(operands):
                values = [self.evaluate(operand, state, action) for operand in operands]
                return len(set(values)) == len(values)
            case _:
                raise TypeError(f"Not an expression: {expression!r}")

    def _find(self, pairs: tuple[tuple[str, Value], ...], name: str, kind: str) -> Value:
        for key, value in pairs:
            if key == name:
                return value
        raise KeyError(f"Unknown {kind}: {name!r}")
