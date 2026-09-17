import random

from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain
from openmind.agent.service.one_ply_chooser import OnePlyChooser
from openmind.csp.factory.csp_factory import create_solver
from openmind.mcts.constant.mcts_constant import RANDOM_OPTION
from openmind.mcts.service.tree_search_tests import Ticking
from openmind.mcts.service.valuation_prior_tests import CenterValued
from openmind.predictor.factory.predictor_factory import create_predictor
from openmind.timing.model.deadline import Deadline
from openmind.world.service.state_reader import StateReader


def chooser() -> OnePlyChooser:
    return OnePlyChooser(create_solver(), create_predictor(), StateReader())


def test_each_legal_move_is_valued_for_the_player_to_act_until_the_deadline_passes() -> None:
    domain = create_tictactoe_domain()
    actions = chooser().legal(domain, domain.initial_state)
    source = Ticking(1.0)

    values = chooser().values(domain, domain.initial_state, actions, CenterValued())

    assert values is not None and max(values) == 0.9 and values.count(0.9) == 1
    assert chooser().values(domain, domain.initial_state, actions, CenterValued(), Deadline(source.now() + 3.0, source)) is None


def test_a_random_move_is_a_legal_move_at_the_value_given_for_what_isn_t_known() -> None:
    domain = create_tictactoe_domain()

    result = chooser().random(domain, domain.initial_state, random.Random(1), 0.5)

    assert result.chosen in create_solver().solve(domain.problem, domain.initial_state)
    assert result.option == RANDOM_OPTION and [item.mean_payoff for item in result.statistics] == [0.5]
