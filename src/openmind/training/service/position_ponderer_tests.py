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
from openmind.training.model.played_game import PlayedGame
from openmind.training.model.pondering_settings import PonderingSettings
from openmind.training.service.position_ponderer import PositionPonderer
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.action import Action
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


def played(*moves: tuple[int, int]) -> PlayedGame:
    """A tic-tac-toe game playing the moves, (row, col) each, from the start: every position searched from and the final
    payoffs."""
    domain, predictor, state_reader = create_tictactoe_domain(), create_predictor(), StateReader()
    state, states = domain.initial_state, []
    for row, col in moves:
        states.append(state)
        ((state, _),) = predictor.predict(domain.transitions, state, Action("place", (("col", col), ("row", row)))).outcomes
    return PlayedGame((), tuple(states), (0.5,) * len(states), state_reader.payoffs(state, domain.players))


#: X wins along the top row: X, O, X, O, X.
X_WINS = ((1, 1), (2, 1), (1, 2), (2, 2), (1, 3))


def test_a_decisive_game_is_walked_back_from_its_end_until_a_position_isn_t_proven(caplog: pytest.LogCaptureFixture) -> None:
    domain, game = create_tictactoe_domain(), played(*X_WINS)
    rows = tuple(PositionRow(state, "X", 0.5) for state in game.states)

    pondering = new_ponderer().ponder(domain, rows, None, PonderingSettings(0, BUDGET, 10), (game,))

    (walk,) = pondering.walks
    last, before = game.states[-1], game.states[-2]
    assert (walk.game, [deduction.state for deduction in walk.deductions], walk.proven) == (0, [last, before], 1)
    assert pondering.deductions == ()
    assert [row.target for row in pondering.rows] == [0.5, 0.5, 0.5, 0.5, 1.0]
    assert pondering.seeds
    assert any(message.startswith("Walked back 1 decisive games of 1, 10 positions each at most: 1 of 2") for message in caplog.messages)


def test_a_walk_stops_at_its_share_of_the_endings_and_a_drawn_game_isn_t_walked() -> None:
    domain = create_tictactoe_domain()
    decisive, drawn = played(*X_WINS), played((2, 2), (1, 1), (1, 2), (3, 2), (2, 1), (2, 3), (1, 3), (3, 1), (3, 3))
    rows = tuple(PositionRow(state, "X", 0.5) for game in (decisive, drawn) for state in game.states)

    pondering = new_ponderer().ponder(domain, rows, None, PonderingSettings(0, BUDGET, 1), (decisive, drawn))

    assert drawn.payoffs == (0.5, 0.5)
    assert [(walk.game, len(walk.deductions)) for walk in pondering.walks] == [(0, 1)]


def test_the_positions_missed_most_leave_out_those_the_walks_proved() -> None:
    domain, game = create_tictactoe_domain(), played(*X_WINS)
    rows = (PositionRow(game.states[-1], "X", 0.0), PositionRow(game.states[0], "X", 0.5), PositionRow(game.states[1], "X", 0.5))

    pondering = new_ponderer().ponder(domain, rows, None, PonderingSettings(1, BUDGET, 10), (game,))

    assert game.states[-1] not in [deduction.state for deduction in pondering.deductions]
    assert len(pondering.deductions) == 1 and pondering.rows[0].target == 1.0
