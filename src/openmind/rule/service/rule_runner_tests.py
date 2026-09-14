import pickle
import traceback

import pytest

from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rule.model.compiled_rule import CompiledRule
from openmind.rule.model.python_rule import PythonRule
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.state import State

STATE = State((("cell(1,1)", "X"), ("cell(1,2)", None), ("payoff(O)", None), ("payoff(X)", None), ("turn", "O")))


def new_runner() -> RuleRunner:
    return RuleRunner(StateNamespaceMapper(VariableNameMapper()))


def value_rule(source: str, parameters: tuple[str, ...] = (), definitions: str | None = None) -> CompiledRule:
    return RuleCompiler().compile_value(
        PythonRule(source), parameters, None if definitions is None else PythonRule(definitions)
    )


def effects_rule(source: str, definitions: str | None = None) -> CompiledRule:
    return RuleCompiler().compile_effects(PythonRule(source), None if definitions is None else PythonRule(definitions))


def test_a_value_rule_reads_state_variables_and_the_parameters_given() -> None:
    empty_and_o_to_play = value_rule("cell[row, col] is None and turn == 'O'", ("row", "col"))
    runner = new_runner()

    assert runner.value(empty_and_o_to_play, STATE, {"row": 1, "col": 2}) is True
    assert runner.value(empty_and_o_to_play, STATE, {"row": 1, "col": 1}) is False


def test_parameters_are_visible_inside_comprehensions() -> None:
    row_is_full = value_rule("all(cell[row, col] is not None for col in (1, 2))", ("row",))

    assert new_runner().value(row_is_full, STATE, {"row": 1}) is False


def test_a_parameter_the_rule_reads_but_is_not_given_raises() -> None:
    with pytest.raises(KeyError, match="col"):
        new_runner().value(value_rule("cell[1, col]", ("col",)), STATE, {})


def test_rules_see_their_definitions_imports_included_and_all_different() -> None:
    rule = value_rule("LIMIT == 3 and all_different(1, 2, LIMIT)", definitions="import math\nLIMIT = math.floor(3.5)")

    assert new_runner().value(rule, STATE) is True


def test_added_names_are_visible_to_a_value_rule() -> None:
    runner = new_runner()
    names = {"double": lambda value: 2 * value, "me": "O"}

    assert runner.value(value_rule("double(row) == 4 and turn == me", ("row",)), STATE, {"row": 2}, names) is True


def test_an_added_name_that_is_also_a_state_variable_raises() -> None:
    with pytest.raises(ValueError, match="turn"):
        new_runner().value(value_rule("turn"), STATE, None, {"turn": "X"})


def test_effects_assign_state_variables_to_give_the_next_state() -> None:
    place = effects_rule("row = 1\ncell[row, col] = turn\npayoff[turn] = WIN\nturn = 'X'", definitions="WIN = 1.0")

    assert new_runner().apply(place, STATE, {"col": 2}) == State(
        (("cell(1,1)", "X"), ("cell(1,2)", "O"), ("payoff(O)", 1.0), ("payoff(X)", None), ("turn", "X"))
    )


def test_effects_leave_the_given_state_and_its_names_unchanged() -> None:
    runner = new_runner()
    read_cell = value_rule("cell[1, 2]")
    runner.value(read_cell, STATE)

    runner.apply(effects_rule("cell[1, 2] = 'O'"), STATE)

    assert runner.value(read_cell, STATE) is None


def test_an_effect_on_a_variable_the_state_lacks_raises() -> None:
    with pytest.raises(KeyError, match=r"cell\(3,3\)"):
        new_runner().apply(effects_rule("cell[3, 3] = 'X'"), STATE)


def test_a_parameter_named_like_a_state_variable_raises() -> None:
    with pytest.raises(ValueError, match="turn"):
        new_runner().apply(effects_rule("pass"), STATE, {"turn": "X"})


def test_a_copy_sent_to_another_process_leaves_its_namespaces_behind() -> None:
    runner, rule = new_runner(), value_rule("zero + 1", ("zero",))
    runner.value(rule, STATE, {"zero": 1})

    copy = pickle.loads(pickle.dumps(runner))

    assert copy.value(rule, STATE, {"zero": 1}) == 2


def test_a_failing_rule_shows_its_own_source_in_the_traceback() -> None:
    with pytest.raises(ZeroDivisionError) as raised:
        new_runner().value(value_rule("1 / zero", ("zero",)), STATE, {"zero": 0})

    assert "1 / zero" in "".join(traceback.format_exception(raised.value))
