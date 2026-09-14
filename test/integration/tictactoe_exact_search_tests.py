import pytest

from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain
from openmind.csp.factory.csp_factory import create_solver
from openmind.evaluation.service.exact_search import ExactSearch
from openmind.predictor.factory.predictor_factory import create_predictor
from openmind.world.builder.state_builder import StateBuilder
from openmind.world.model.action import Action
from openmind.world.service.state_reader import StateReader

pytestmark = pytest.mark.log_level("INFO")


@pytest.fixture(scope="module")
def exact_search() -> ExactSearch:
    return ExactSearch(create_solver(), create_predictor(), StateReader())


def test_tictactoe_has_4520_positions_with_a_legal_action(exact_search: ExactSearch) -> None:
    assert len(exact_search.positions(create_tictactoe_domain())) == 4520


def test_every_first_move_is_optimal(exact_search: ExactSearch) -> None:
    domain = create_tictactoe_domain()

    assert len(exact_search.optimal_actions(domain, domain.initial_state)) == 9


def test_taking_an_immediate_win_is_the_optimal_move(exact_search: ExactSearch) -> None:
    # X X . / O O . / . . .
    domain = create_tictactoe_domain()
    builder = StateBuilder()
    cells = {"cell(1,1)": "X", "cell(1,2)": "X", "cell(2,1)": "O", "cell(2,2)": "O"}
    for name, value in (dict(domain.initial_state.variables) | cells).items():
        builder.with_variable(name, value)

    assert exact_search.optimal_actions(domain, builder.build()) == (Action("place", (("col", 3), ("row", 1))),)
