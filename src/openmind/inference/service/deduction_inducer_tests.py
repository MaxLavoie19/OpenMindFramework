import pytest

from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain
from openmind.inference.model.deduction_budget import DeductionBudget
from openmind.inference.service.deduction_inducer import DeductionInducer
from openmind.inference.service.expression_generator import ExpressionGenerator
from openmind.inference.service.position_deducer_tests import LOST_IN_2, WIN_IN_3, new_deducer, position
from openmind.rbs.model.position_row import PositionRow
from openmind.rbs.service.term_evaluator_tests import new_evaluator
from openmind.world.mapper.variable_name_mapper import VariableNameMapper

pytestmark = pytest.mark.log_level("INFO")


def new_inducer() -> DeductionInducer:
    return DeductionInducer(ExpressionGenerator(VariableNameMapper()), VariableNameMapper())


def test_a_forced_win_induces_its_proof_as_a_look_ahead_and_what_its_first_move_changed_as_a_pattern() -> None:
    domain, generator, state = create_tictactoe_domain(), ExpressionGenerator(VariableNameMapper()), position(WIN_IN_3)
    deduction = new_deducer().deduce(domain, state, DeductionBudget(3, 60.0))

    proof, pattern = new_inducer().seeds(domain, deduction, generator.vocabulary(domain, [state]), 1.0)

    assert proof.template == (
        "{view}.best(me, lambda v3: v3.worst(other, lambda v2: v2.best(me, lambda v1: v1.payoff[me] == 1.0)))"
    )
    assert (proof.clauses, proof.plies) == (4, 3)
    assert pattern.template == "sum(1 for at in {view}.cell if {view}.cell[at] == None)"
    column = new_evaluator().column(domain, [PositionRow(state, "X", 0.0)], generator.source(proof))
    assert column is not None and list(column) == [1.0]


def test_a_forced_loss_reads_whatever_i_do_the_other_player_can_win_next() -> None:
    domain, generator, state = create_tictactoe_domain(), ExpressionGenerator(VariableNameMapper()), position(LOST_IN_2)
    deduction = new_deducer().deduce(domain, state, DeductionBudget(2, 60.0))

    proof, _ = new_inducer().seeds(domain, deduction, generator.vocabulary(domain, [state]), 1.0)

    assert proof.template == "{view}.worst(me, lambda v2: v2.best(other, lambda v1: v1.payoff[other] == 1.0))"
    column = new_evaluator().column(domain, [PositionRow(state, "O", 0.0)], generator.source(proof))
    assert column is not None and list(column) == [1.0]


def test_a_deduction_that_proved_nothing_induces_nothing() -> None:
    domain, generator, state = create_tictactoe_domain(), ExpressionGenerator(VariableNameMapper()), position(WIN_IN_3)
    deduction = new_deducer().deduce(domain, state, DeductionBudget(2, 60.0))

    assert new_inducer().seeds(domain, deduction, generator.vocabulary(domain, [state]), 1.0) == ()
