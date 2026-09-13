import pytest

from openmind.csp.service.constraint_checker import ConstraintChecker
from openmind.expression.model.action_parameter import ActionParameter
from openmind.expression.model.constant import Constant
from openmind.expression.model.equals import Equals
from openmind.expression.model.state_variable import StateVariable
from openmind.expression.service.interpreter import Interpreter
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.state import State


def new_checker() -> ConstraintChecker:
    return ConstraintChecker(Interpreter(VariableNameMapper()))


def test_holds_evaluates_the_constraint_with_the_given_values() -> None:
    constraint = Equals(StateVariable("cell", (ActionParameter("row"), ActionParameter("col"))), Constant(None))
    state = State((("cell(1,1)", "X"), ("cell(1,2)", None)))

    assert new_checker().holds(constraint, state, "place", {"row": 1, "col": 2}) is True
    assert new_checker().holds(constraint, state, "place", {"row": 1, "col": 1}) is False


def test_holds_rejects_a_constraint_that_is_not_true_or_false() -> None:
    with pytest.raises(TypeError, match="place"):
        new_checker().holds(ActionParameter("row"), State(()), "place", {"row": 1})


def test_value_evaluates_any_expression() -> None:
    assert new_checker().value(StateVariable("turn"), State((("turn", "O"),)), "place", {}) == "O"
