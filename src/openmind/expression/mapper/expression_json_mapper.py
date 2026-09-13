from typing import Any

from openmind.expression.model.action_parameter import ActionParameter
from openmind.expression.model.all_different import AllDifferent
from openmind.expression.model.all_of import AllOf
from openmind.expression.model.any_of import AnyOf
from openmind.expression.model.constant import Constant
from openmind.expression.model.equals import Equals
from openmind.expression.model.expression import Expression
from openmind.expression.model.not_ import Not
from openmind.expression.model.state_variable import StateVariable


class ExpressionJsonMapper:
    """Maps an expression to JSON-compatible data and back: {"equals": [{"action_parameter": "row"}, {"constant": 2}]}."""

    def to_data(self, expression: Expression) -> dict[str, Any]:
        match expression:
            case Constant(value):
                return {"constant": value}
            case StateVariable(base, indices):
                return {"state_variable": base, "indices": [self.to_data(index) for index in indices]}
            case ActionParameter(name):
                return {"action_parameter": name}
            case Equals(left, right):
                return {"equals": [self.to_data(left), self.to_data(right)]}
            case Not(operand):
                return {"not": self.to_data(operand)}
            case AllOf(operands):
                return {"all_of": [self.to_data(operand) for operand in operands]}
            case AnyOf(operands):
                return {"any_of": [self.to_data(operand) for operand in operands]}
            case AllDifferent(operands):
                return {"all_different": [self.to_data(operand) for operand in operands]}
            case _:
                raise TypeError(f"Not an expression: {expression!r}")

    def from_data(self, data: dict[str, Any]) -> Expression:
        if "constant" in data:
            return Constant(data["constant"])
        if "state_variable" in data:
            return StateVariable(data["state_variable"], tuple(self.from_data(index) for index in data["indices"]))
        if "action_parameter" in data:
            return ActionParameter(data["action_parameter"])
        if "equals" in data:
            left, right = data["equals"]
            return Equals(self.from_data(left), self.from_data(right))
        if "not" in data:
            return Not(self.from_data(data["not"]))
        if "all_of" in data:
            return AllOf(tuple(self.from_data(operand) for operand in data["all_of"]))
        if "any_of" in data:
            return AnyOf(tuple(self.from_data(operand) for operand in data["any_of"]))
        if "all_different" in data:
            return AllDifferent(tuple(self.from_data(operand) for operand in data["all_different"]))
        raise ValueError(f"Not an expression: {data!r}")
