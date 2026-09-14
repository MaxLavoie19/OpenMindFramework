import pytest

from openmind.csp.service.constraint_checker import ConstraintChecker
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rule.model.compiled_rule import CompiledRule
from openmind.rule.model.python_rule import PythonRule
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.state import State


def new_checker() -> ConstraintChecker:
    return ConstraintChecker(RuleRunner(StateNamespaceMapper(VariableNameMapper())))


def constraint(source: str, *parameters: str) -> CompiledRule:
    return RuleCompiler().compile_value(PythonRule(source), parameters)


def test_holds_checks_the_constraint_with_the_given_values() -> None:
    cell_is_empty = constraint("cell[row, col] is None", "row", "col")
    state = State((("cell(1,1)", "X"), ("cell(1,2)", None)))

    assert new_checker().holds(cell_is_empty, state, "place", {"row": 1, "col": 2}) is True
    assert new_checker().holds(cell_is_empty, state, "place", {"row": 1, "col": 1}) is False


def test_holds_rejects_a_constraint_that_is_not_true_or_false() -> None:
    with pytest.raises(TypeError, match="place"):
        new_checker().holds(constraint("row", "row"), State(()), "place", {"row": 1})
