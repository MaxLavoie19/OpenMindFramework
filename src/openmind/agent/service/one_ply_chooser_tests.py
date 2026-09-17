import random

from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain
from openmind.agent.service.one_ply_chooser import OnePlyChooser
from openmind.csp.factory.csp_factory import create_solver
from openmind.mcts.constant.mcts_constant import ONE_PLY_OPTION, RANDOM_OPTION
from openmind.mcts.service.rater_prior_tests import CENTER
from openmind.mcts.service.tree_search_tests import Ticking
from openmind.mcts.service.valuation_prior_tests import CenterValued
from openmind.predictor.factory.predictor_factory import create_predictor
from openmind.timing.model.deadline import Deadline
from openmind.world.service.state_reader import StateReader


def chooser() -> OnePlyChooser:
    return OnePlyChooser(create_solver(), create_predictor(), StateReader())


def test_one_ply_plays_the_move_valued_highest_for_the_player_to_act() -> None:
    domain = create_tictactoe_domain()

    result = chooser().best(domain, domain.initial_state, CenterValued(), random.Random(1))

    assert result is not None and result.chosen == CENTER and result.option == ONE_PLY_OPTION
    assert len(result.statistics) == 9 and all(item.visits == 1 for item in result.statistics)


def test_one_ply_gives_up_once_its_deadline_passes() -> None:
    domain = create_tictactoe_domain()
    source = Ticking(1.0)

    assert chooser().best(domain, domain.initial_state, CenterValued(), random.Random(1), Deadline(source.now() + 3.0, source)) is None


def test_a_random_move_is_a_legal_move_at_the_value_given_for_what_isn_t_known() -> None:
    domain = create_tictactoe_domain()

    result = chooser().random(domain, domain.initial_state, random.Random(1), 0.5)

    assert result.chosen in create_solver().solve(domain.problem, domain.initial_state)
    assert result.option == RANDOM_OPTION and [item.mean_payoff for item in result.statistics] == [0.5]
