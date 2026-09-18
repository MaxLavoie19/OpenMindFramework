from collections.abc import Callable
import pytest

from openmind.evaluation.service.exact_search import ExactSearch
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.world.builder.state_builder import StateBuilder
from openmind.world.model.action import Action
from openmind.world.service.state_reader import StateReader

pytestmark = pytest.mark.log_level("INFO")

type Game = Callable[[str], RuleBasedSystem]


@pytest.fixture(scope="module")
def exact_search() -> ExactSearch:
    return ExactSearch(StateReader())


def test_tictactoe_has_4520_positions_with_a_legal_action(game: Game, exact_search: ExactSearch) -> None:
    assert len(exact_search.positions(game("tictactoe"))) == 4520


def test_every_first_move_is_optimal(game: Game, exact_search: ExactSearch) -> None:
    rbs = game("tictactoe")

    assert len(exact_search.optimal_actions(rbs, rbs.start())) == 9


def test_taking_an_immediate_win_is_the_optimal_move(game: Game, exact_search: ExactSearch) -> None:
    # X X . / O O . / . . .
    rbs = game("tictactoe")
    builder = StateBuilder()
    cells = {"cell(1,1)": "X", "cell(1,2)": "X", "cell(2,1)": "O", "cell(2,2)": "O"}
    for name, value in (dict(rbs.start().variables) | cells).items():
        builder.with_variable(name, value)

    assert exact_search.optimal_actions(rbs, builder.build()) == (Action("place", (("col", 3), ("row", 1))),)
