import pytest

from openmind.csp.service.constraint_checker import ConstraintChecker
from openmind.rule.factory.rule_factory import create_rule_caller
from openmind.rule.model.called_rule import CalledRule
from openmind.rule.model.python_rule import PythonRule
from openmind.structure.model.grid import Grid
from openmind.world.model.state import State


def new_checker() -> ConstraintChecker:
    return ConstraintChecker(create_rule_caller())


def constraint(source: str, *parameters: str) -> CalledRule:
    return create_rule_caller().prepare(PythonRule(source), parameters)


def test_holds_checks_the_constraint_with_the_given_values() -> None:
    cell_is_empty = constraint("cell[row, col] is None", "row", "col")
    state = State.of(cell=Grid((1, 2), ("X", None)))

    assert new_checker().holds(cell_is_empty, state, "place", {"row": 1, "col": 2}) is True
    assert new_checker().holds(cell_is_empty, state, "place", {"row": 1, "col": 1}) is False


def test_holds_rejects_a_constraint_that_is_not_true_or_false() -> None:
    with pytest.raises(TypeError, match="place"):
        new_checker().holds(constraint("row", "row"), State(()), "place", {"row": 1})
