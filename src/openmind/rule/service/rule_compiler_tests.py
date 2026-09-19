import pickle

import pytest

from openmind.rule.constant.rule_constant import DEFINITIONS, EFFECTS, VALUE
from openmind.rule.model.python_rule import PythonRule
from openmind.rule.service.rule_compiler import RuleCompiler


def test_a_value_rule_takes_only_the_parameters_it_reads_in_the_given_order() -> None:
    compiled = RuleCompiler().compile_value(PythonRule("cell[row, col] is None"), ("col", "row", "player"))

    assert (compiled.kind, compiled.arguments, compiled.definitions) == (VALUE, ("col", "row"), None)


def test_a_script_returning_its_value_is_a_value_rule() -> None:
    compiled = RuleCompiler().compile_value(PythonRule("total = a + b\nreturn total > 2"), ("a", "b"))

    assert (compiled.kind, compiled.arguments) == (VALUE, ("a", "b"))


def test_the_same_rule_compiles_once() -> None:
    compiler = RuleCompiler()

    assert compiler.compile_value(PythonRule("a > 1"), ("a",)) is compiler.compile_value(PythonRule("a > 1"), ("a",))


def test_effects_and_definitions_compile_as_modules_and_effects_keep_their_definitions() -> None:
    compiler = RuleCompiler()
    definitions = PythonRule("WIN = 1.0")

    effects = compiler.compile_effects(PythonRule("payoff = WIN"), definitions)

    assert (effects.kind, effects.arguments) == (EFFECTS, ())
    assert effects.definitions == compiler.compile_definitions(definitions)
    assert effects.definitions is not None and effects.definitions.kind == DEFINITIONS


def test_an_indented_rule_compiles_as_if_it_started_at_the_margin() -> None:
    compiled = RuleCompiler().compile_effects(PythonRule("\n    score = 1\n    if score:\n        score = 2\n"))

    assert compiled.kind == EFFECTS


def test_a_copy_sent_to_another_process_leaves_compiled_rules_behind_and_compiles_again() -> None:
    compiler = RuleCompiler()
    compiler.compile_value(PythonRule("a + 1"), ("a",))

    copy = pickle.loads(pickle.dumps(compiler))

    assert copy.compile_value(PythonRule("a + 1"), ("a",)).arguments == ("a",)


def test_a_syntax_error_names_the_rule_and_the_line() -> None:
    with pytest.raises(SyntaxError, match=r"in rule 'cell\[row, col\] ==', line 1"):
        RuleCompiler().compile_value(PythonRule("cell[row, col] =="), ("row", "col"))
