import pytest

from openmind.expression.model.action_parameter import ActionParameter
from openmind.expression.model.all_different import AllDifferent
from openmind.expression.model.all_of import AllOf
from openmind.expression.model.any_of import AnyOf
from openmind.expression.model.constant import Constant
from openmind.expression.model.equals import Equals
from openmind.expression.model.expression import Expression
from openmind.expression.model.not_ import Not
from openmind.expression.model.state_variable import StateVariable
from openmind.expression.service.interpreter import Interpreter
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.action import Action
from openmind.world.model.state import State
from openmind.world.model.value import Value


def evaluate(expression: Expression) -> Value:
    state = State((("cell(2,3)", "X"), ("turn", "O")))
    action = Action("place", (("col", 3), ("row", 2)))
    return Interpreter(VariableNameMapper()).evaluate(expression, state, action)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        (Constant(7), 7),
        (StateVariable("turn"), "O"),
        (StateVariable("cell", (ActionParameter("row"), ActionParameter("col"))), "X"),
        (ActionParameter("row"), 2),
        (Equals(StateVariable("turn"), Constant("O")), True),
        (Equals(StateVariable("turn"), Constant("X")), False),
        (Not(Constant(False)), True),
        (Not(Constant(True)), False),
        (AllOf((Constant(True), Constant(True))), True),
        (AllOf((Constant(True), Constant(False))), False),
        (AnyOf((Constant(False), Constant(True))), True),
        (AnyOf((Constant(False), Constant(False))), False),
        (AllDifferent((Constant(1), StateVariable("turn"), ActionParameter("row"))), True),
        (AllDifferent((ActionParameter("row"), Constant(2))), False),
    ],
)
def test_evaluate(expression: Expression, expected: Value) -> None:
    assert evaluate(expression) == expected


def test_unknown_state_variable_raises() -> None:
    with pytest.raises(KeyError, match=r"cell\(1,1\)"):
        evaluate(StateVariable("cell", (Constant(1), Constant(1))))


def test_unknown_action_parameter_raises() -> None:
    with pytest.raises(KeyError, match="player"):
        evaluate(ActionParameter("player"))


def test_non_expression_raises() -> None:
    with pytest.raises(TypeError):
        evaluate("turn")
