from collections.abc import Callable
import pytest

from openmind.inference.model.deduction_budget import DeductionBudget
from openmind.inference.service.position_deducer import PositionDeducer
from openmind.training.service.ending_walker import EndingWalker
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.model.action import Action
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader

pytestmark = pytest.mark.log_level("INFO")

type Game = Callable[[str], RuleBasedSystem]

BUDGET = DeductionBudget(3, 60.0)
#: X wins along the top row: X, O, X, O, X.
X_WINS = ((1, 1), (2, 1), (1, 2), (2, 2), (1, 3))


def new_walker() -> EndingWalker:
    return EndingWalker(PositionDeducer(StateReader(), ActionTextMapper()))


def positions(game: Game, *moves: tuple[int, int]) -> tuple[State, ...]:
    """Every tic-tac-toe position the moves, (row, col) each, are played from, starting at the start."""
    rbs = game("tictactoe")
    state, states = rbs.start(), []
    for row, col in moves:
        states.append(state)
        ((state, _),) = rbs.outcomes(state, Action("place", (("col", col), ("row", row)))).outcomes
    return tuple(states)


def test_a_game_is_walked_back_from_its_end_until_a_position_isn_t_proven(game: Game) -> None:
    rbs, states = game("tictactoe"), positions(game, *X_WINS)

    deductions = new_walker().walk_back(rbs, states, BUDGET, 10)

    assert [deduction.state for deduction in deductions] == [states[-1], states[-2]]
    assert [deduction.payoffs is not None for deduction in deductions] == [True, False]


def test_a_walk_stops_at_its_limit(game: Game) -> None:
    rbs, states = game("tictactoe"), positions(game, *X_WINS)

    assert [deduction.state for deduction in new_walker().walk_back(rbs, states, BUDGET, 1)] == [states[-1]]
