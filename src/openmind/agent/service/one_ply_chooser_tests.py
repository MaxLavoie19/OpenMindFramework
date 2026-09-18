from collections.abc import Callable
import math
import random

from openmind.agent.service.one_ply_chooser import OnePlyChooser
from openmind.mcts.constant.mcts_constant import RANDOM_OPTION
from openmind.mcts.service.tree_search_tests import Ticking
from openmind.mcts.service.valuation_prior_tests import CenterValued
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.timing.model.deadline import Deadline
from openmind.world.service.state_reader import StateReader


type Game = Callable[[str], RuleBasedSystem]


def chooser() -> OnePlyChooser:
    return OnePlyChooser(StateReader())


def test_each_legal_move_is_valued_for_the_player_to_act_until_the_deadline_passes(game: Game) -> None:
    rbs = game("tictactoe")
    actions = chooser().legal(rbs, rbs.start())
    source = Ticking(1.0)

    values = chooser().values(rbs, rbs.start(), actions, CenterValued())

    assert values is not None and max(values) == 0.9 and values.count(0.9) == 1
    assert chooser().values(rbs, rbs.start(), actions, CenterValued(), Deadline(source.now() + 3.0, source)) is None


def test_a_random_move_is_a_legal_move_without_a_value(game: Game) -> None:
    rbs = game("tictactoe")

    result = chooser().random(rbs, rbs.start(), random.Random(1))

    assert result.chosen in rbs.actions(rbs.start())
    assert result.option == RANDOM_OPTION and all(math.isnan(item.mean_payoff) for item in result.statistics)
