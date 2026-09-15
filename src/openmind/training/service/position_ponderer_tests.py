import pytest

from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain
from openmind.csp.factory.csp_factory import create_solver
from openmind.inference.model.deduction_budget import DeductionBudget
from openmind.inference.service.deduction_inducer import DeductionInducer
from openmind.inference.service.expression_generator import ExpressionGenerator
from openmind.inference.service.position_deducer import PositionDeducer
from openmind.inference.service.position_deducer_tests import WIN_IN_3, position
from openmind.parallel.service.task_runner import TaskRunner
from openmind.predictor.factory.predictor_factory import create_predictor
from openmind.rbs.builder.consequence_library_builder import ConsequenceLibraryBuilder
from openmind.rbs.model.position_row import PositionRow
from openmind.rbs.model.value_base import ValueBase
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rule.model.python_rule import PythonRule
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.training.model.pondering_settings import PonderingSettings
from openmind.training.service.position_ponderer import PositionPonderer
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.service.state_reader import StateReader

pytestmark = pytest.mark.log_level("INFO")

BUDGET = DeductionBudget(3, 60.0)


def new_ponderer() -> PositionPonderer:
    solver, predictor, state_reader = create_solver(), create_predictor(), StateReader()
    generator = ExpressionGenerator(VariableNameMapper())
    return PositionPonderer(
        PositionDeducer(solver, predictor, state_reader, ActionTextMapper()),
        DeductionInducer(generator, VariableNameMapper()),
        generator,
        solver,
        RuleCompiler(),
        RuleRunner(StateNamespaceMapper(VariableNameMapper())),
        ConsequenceLibraryBuilder().build(),
        TaskRunner(1),
    )


def test_the_position_missed_most_is_pondered_and_once_proven_its_rows_take_the_proven_payoffs() -> None:
    domain = create_tictactoe_domain()
    win, start = position(WIN_IN_3), domain.initial_state
    rows = (PositionRow(win, "X", 0.0), PositionRow(start, "X", 0.5), PositionRow(start, "O", 0.5))

    pondering = new_ponderer().ponder(domain, rows, None, PonderingSettings(1, BUDGET))

    assert [deduction.state for deduction in pondering.deductions] == [win]
    assert [row.target for row in pondering.rows] == [1.0, 0.5, 0.5]
    assert pondering.sources[0] == PythonRule(
        "here.best(me, lambda v3: v3.worst(other, lambda v2: v2.best(me, lambda v1: v1.payoff[me] == 1.0)))"
    )
    assert len(pondering.seeds) == len(pondering.sources) == 2


def test_with_rules_a_miss_is_how_far_the_target_is_from_their_value() -> None:
    domain = create_tictactoe_domain()
    win, start = position(WIN_IN_3), domain.initial_state
    # Without rules the win misses the mean target most; rules valuing everything 0.5 miss the start most.
    rows = (PositionRow(win, "X", 0.5), PositionRow(start, "X", 0.0), PositionRow(start, "O", 0.0))
    halves = ValueBase("tictactoe", 0.0, 0.0, 1.0, ())

    without = new_ponderer().ponder(domain, rows, None, PonderingSettings(1, BUDGET))
    with_rules = new_ponderer().ponder(domain, rows, halves, PonderingSettings(1, BUDGET))

    assert [deduction.state for deduction in without.deductions] == [win]
    assert [deduction.state for deduction in with_rules.deductions] == [start]
    assert (with_rules.rows, with_rules.seeds) == (rows, ())


def test_pondering_no_position_leaves_the_rows_as_they_are() -> None:
    domain = create_tictactoe_domain()
    rows = (PositionRow(position(WIN_IN_3), "X", 0.0),)

    assert new_ponderer().ponder(domain, rows, None, PonderingSettings(0, BUDGET)).rows == rows
