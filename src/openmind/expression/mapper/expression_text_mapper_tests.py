import pytest

from openmind.expression.mapper.expression_text_mapper import ExpressionTextMapper
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


@pytest.mark.parametrize(
    ("expression", "text"),
    [
        (Constant("X"), "'X'"),
        (Constant(None), "None"),
        (StateVariable("turn"), "turn"),
        (StateVariable("cell", (ActionParameter("row"), ActionParameter("col"))), "cell(row,col)"),
        (StateVariable("payoff", (Constant("X"),)), "payoff(X)"),
        (StateVariable("payoff", (StateVariable("turn"),)), "payoff(turn)"),
        (ActionParameter("row"), "row"),
        (Equals(StateVariable("turn"), Constant("X")), "turn == 'X'"),
        (Not(Constant(True)), "not(True)"),
        (AllOf((Constant(True), Constant(False))), "all(True, False)"),
        (AnyOf((Constant(True), Constant(False))), "any(True, False)"),
        (AllDifferent((ActionParameter("cell(1,3)"), StateVariable("cell(1,1)"))), "all_different(cell(1,3), cell(1,1))"),
    ],
)
def test_to_text(expression: Expression, text: str) -> None:
    assert ExpressionTextMapper(VariableNameMapper()).to_text(expression) == text


def test_non_expression_raises() -> None:
    with pytest.raises(TypeError):
        ExpressionTextMapper(VariableNameMapper()).to_text("turn")
