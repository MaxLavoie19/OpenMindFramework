import pytest

from openmind.rule.mapper.call_operand_mapper import CallOperandMapper
from openmind.rule.model.python_rule import PythonRule


def test_a_single_call_gives_its_arguments_as_rules() -> None:
    assert CallOperandMapper().to_operands(PythonRule("all_different(a, b, cell[1, 1])"), "all_different") == (
        PythonRule("a"),
        PythonRule("b"),
        PythonRule("cell[1, 1]"),
    )


@pytest.mark.parametrize(
    "source", ["all_different(a, b) and c", "any_of(a, b)", "all_different(*values)", "all_different(a, key=b)", "a =="]
)
def test_anything_but_one_plain_call_to_the_function_gives_none(source: str) -> None:
    assert CallOperandMapper().to_operands(PythonRule(source), "all_different") is None
