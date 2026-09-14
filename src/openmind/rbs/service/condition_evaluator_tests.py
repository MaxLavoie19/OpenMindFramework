import pytest

from openmind.parallel.service.task_runner import TaskRunner
from openmind.rbs.builder.consequence_library_builder import ConsequenceLibraryBuilder
from openmind.rbs.model.action_row import ActionRow
from openmind.rbs.service.condition_evaluator import ConditionEvaluator
from openmind.rbs.service.consequence_library_tests import place, position, strip_domain
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rule.model.python_rule import PythonRule
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.action import Action

ROWS = (
    ActionRow(position({1: "X"}, "X"), place(2), 10, 1.0, 0.0),
    ActionRow(position({1: "X"}, "X"), place(4), 10, 0.2, -0.8),
)


def new_evaluator(workers: int = 1) -> ConditionEvaluator:
    return ConditionEvaluator(
        RuleCompiler(),
        RuleRunner(StateNamespaceMapper(VariableNameMapper())),
        ConsequenceLibraryBuilder().build(),
        TaskRunner(workers),
    )


def test_values_read_the_parameters_the_action_and_the_consequences() -> None:
    values = new_evaluator().values(strip_domain(), ROWS, PythonRule("(col, win_chance(action), near(action, 0, -1))"))

    assert values == [(2, 1.0, "X"), (4, 0.0, None)]


def test_the_mask_is_where_the_rule_gives_true() -> None:
    mask = new_evaluator().mask(strip_domain(), ROWS, PythonRule("win_chance(action) >= 1"))

    assert mask is not None and mask.tolist() == [True, False]


def test_a_rule_raising_on_a_row_gives_none() -> None:
    assert new_evaluator().mask(strip_domain(), ROWS, PythonRule("lamp == 1")) is None


def test_a_parameter_named_action_is_rejected() -> None:
    rows = (ActionRow(position({}, "X"), Action("place", (("action", 1),)), 10, 0.5, 0.0),)

    with pytest.raises(ValueError, match="action"):
        new_evaluator().values(strip_domain(), rows, PythonRule("True"))


@pytest.mark.log_level("INFO")
def test_several_rules_split_between_workers_give_what_each_gives_alone() -> None:
    rows = (*ROWS, ActionRow(position({4: "O"}, "X"), place(3), 10, 0.5, 0.0))
    rules = (PythonRule("win_chance(action) >= 1"), PythonRule("lamp == 1"), PythonRule("(col, near(action, 0, 1))"))

    split = new_evaluator(workers=2)

    assert split.all_values(strip_domain(), rows, rules) == [new_evaluator().values(strip_domain(), rows, rule) for rule in rules]
    assert [None if mask is None else mask.tolist() for mask in split.masks(strip_domain(), rows, rules)] == [
        [True, False, False],
        None,
        [False, False, False],
    ]
