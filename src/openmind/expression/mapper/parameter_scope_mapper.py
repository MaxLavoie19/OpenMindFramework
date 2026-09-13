from openmind.expression.model.action_parameter import ActionParameter
from openmind.expression.model.all_different import AllDifferent
from openmind.expression.model.all_of import AllOf
from openmind.expression.model.any_of import AnyOf
from openmind.expression.model.constant import Constant
from openmind.expression.model.equals import Equals
from openmind.expression.model.expression import Expression
from openmind.expression.model.not_ import Not
from openmind.expression.model.state_variable import StateVariable


class ParameterScopeMapper:
    """Maps an expression to its scope: the names of the action parameters it reads."""

    def to_scope(self, expression: Expression) -> frozenset[str]:
        match expression:
            case Constant():
                return frozenset()
            case ActionParameter(name):
                return frozenset((name,))
            case StateVariable(_, indices):
                return frozenset().union(*(self.to_scope(index) for index in indices))
            case Equals(left, right):
                return self.to_scope(left) | self.to_scope(right)
            case Not(operand):
                return self.to_scope(operand)
            case AllOf(operands) | AnyOf(operands) | AllDifferent(operands):
                return frozenset().union(*(self.to_scope(operand) for operand in operands))
            case _:
                raise TypeError(f"Not an expression: {expression!r}")
