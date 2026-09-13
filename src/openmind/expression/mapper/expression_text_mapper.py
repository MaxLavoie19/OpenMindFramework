from openmind.expression.model.action_parameter import ActionParameter
from openmind.expression.model.all_of import AllOf
from openmind.expression.model.any_of import AnyOf
from openmind.expression.model.constant import Constant
from openmind.expression.model.equals import Equals
from openmind.expression.model.expression import Expression
from openmind.expression.model.not_ import Not
from openmind.expression.model.state_variable import StateVariable
from openmind.world.mapper.variable_name_mapper import VariableNameMapper


class ExpressionTextMapper:
    """Maps an expression to readable text: all(payoff(X) == None, cell(row,col) == None)."""

    def __init__(self, variable_name_mapper: VariableNameMapper) -> None:
        self._variable_name_mapper = variable_name_mapper

    def to_text(self, expression: Expression) -> str:
        match expression:
            case Constant(value):
                return repr(value)
            case StateVariable(base, indices):
                return self._variable_name_mapper.to_name(
                    base, tuple(self._index_to_text(index) for index in indices)
                )
            case ActionParameter(name):
                return name
            case Equals(left, right):
                return f"{self.to_text(left)} == {self.to_text(right)}"
            case Not(operand):
                return f"not({self.to_text(operand)})"
            case AllOf(operands):
                return f"all({', '.join(self.to_text(operand) for operand in operands)})"
            case AnyOf(operands):
                return f"any({', '.join(self.to_text(operand) for operand in operands)})"
            case _:
                raise TypeError(f"Not an expression: {expression!r}")

    def _index_to_text(self, index: Expression) -> str:
        match index:
            case Constant(value):
                return str(value)
            case _:
                return self.to_text(index)
