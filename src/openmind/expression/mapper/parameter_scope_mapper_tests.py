import pytest

from openmind.expression.mapper.parameter_scope_mapper import ParameterScopeMapper
from openmind.expression.model.action_parameter import ActionParameter
from openmind.expression.model.all_different import AllDifferent
from openmind.expression.model.all_of import AllOf
from openmind.expression.model.any_of import AnyOf
from openmind.expression.model.constant import Constant
from openmind.expression.model.equals import Equals
from openmind.expression.model.expression import Expression
from openmind.expression.model.not_ import Not
from openmind.expression.model.state_variable import StateVariable


@pytest.mark.parametrize(
    ("expression", "scope"),
    [
        (Constant(1), set()),
        (StateVariable("turn"), set()),
        (ActionParameter("row"), {"row"}),
        (StateVariable("cell", (ActionParameter("row"), ActionParameter("col"))), {"row", "col"}),
        (Equals(ActionParameter("a"), StateVariable("light")), {"a"}),
        (Not(Equals(ActionParameter("a"), ActionParameter("b"))), {"a", "b"}),
        (AllOf((ActionParameter("a"), Constant(True))), {"a"}),
        (AnyOf((ActionParameter("a"), ActionParameter("c"))), {"a", "c"}),
        (AllDifferent((ActionParameter("a"), ActionParameter("b"), StateVariable("cell(1,1)"))), {"a", "b"}),
    ],
)
def test_to_scope(expression: Expression, scope: set[str]) -> None:
    assert ParameterScopeMapper().to_scope(expression) == scope


def test_non_expression_raises() -> None:
    with pytest.raises(TypeError):
        ParameterScopeMapper().to_scope("row")
