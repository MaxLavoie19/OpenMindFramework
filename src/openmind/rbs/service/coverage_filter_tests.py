from dataclasses import replace

from openmind.parallel.service.task_runner import TaskRunner
from openmind.rbs.builder.consequence_library_builder import ConsequenceLibraryBuilder
from openmind.rbs.model.coverage import Coverage
from openmind.rbs.model.rule import Rule
from openmind.rbs.service.condition_evaluator import ConditionEvaluator
from openmind.rbs.service.consequence_library_tests import strip_domain
from openmind.rbs.service.coverage_filter import CoverageFilter
from openmind.rbs.service.hypothesis_discoverer_tests import WINS, rows
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rule.model.python_rule import PythonRule
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.world.mapper.variable_name_mapper import VariableNameMapper

#: In the rows, winning moves have an advantage of 0 and the others -0.6.
BASE = Rule("place", (), 0.5, 100)


def new_filter() -> CoverageFilter:
    evaluator = ConditionEvaluator(
        RuleCompiler(),
        RuleRunner(StateNamespaceMapper(VariableNameMapper())),
        ConsequenceLibraryBuilder().build(),
        TaskRunner(1),
    )
    return CoverageFilter(evaluator)


def test_a_rule_matching_moves_as_good_as_a_simpler_rule_s_is_covered_whatever_its_expected_value() -> None:
    x_to_play = Rule("place", (PythonRule("turn == 'X'"),), 0.9, 100)

    assert new_filter().keep(strip_domain(), rows(), (BASE, x_to_play), 0.05) == ((BASE,), (Coverage(x_to_play, BASE),))


def test_a_rule_matching_better_moves_is_kept_whatever_its_expected_value() -> None:
    wins = Rule("place", (WINS,), 0.5, 40)

    assert new_filter().keep(strip_domain(), rows(), (BASE, wins), 0.05) == ((BASE, wins), ())


def test_a_rule_matching_rows_a_simpler_rule_does_not_is_not_covered_by_it() -> None:
    left = Rule("place", (WINS, PythonRule("col <= 2")), 1.0, 20)
    right = Rule("place", (WINS, PythonRule("col >= 3")), 1.0, 20)

    assert new_filter().keep(strip_domain(), rows(), (BASE, left, right), 0.05) == ((BASE, left, right), ())


def test_a_priority_rule_is_covered_only_by_a_priority_rule() -> None:
    first_empty_wins = Rule("place", (PythonRule("cell[1, 1] == None"), WINS), 1.0, 20, priority=True)
    ordinary_wins = Rule("place", (WINS,), 1.0, 40)
    priority_wins = replace(ordinary_wins, priority=True)

    assert new_filter().keep(strip_domain(), rows(), (BASE, ordinary_wins, first_empty_wins), 0.05) == (
        (BASE, ordinary_wins, first_empty_wins),
        (),
    )
    assert new_filter().keep(strip_domain(), rows(), (BASE, priority_wins, first_empty_wins), 0.05) == (
        (BASE, priority_wins),
        (Coverage(first_empty_wins, priority_wins),),
    )


def test_a_rule_that_cannot_be_evaluated_on_every_row_is_kept() -> None:
    outside = Rule("place", (PythonRule("cell[9, 9] == None"),), 0.5, 10)

    assert new_filter().keep(strip_domain(), rows(), (BASE, outside), 0.05) == ((BASE, outside), ())
