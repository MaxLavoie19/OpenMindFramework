from collections.abc import Callable
import math

from openmind.mcts.service.rater_prior_tests import CENTER
from openmind.mcts.service.uniform_prior_tests import opening
from openmind.mcts.service.valuation_prior import ValuationPrior
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.world.model.state import State


class CenterValued:
    """Values a position 0.9 for X when X holds the center, 0.1 otherwise, and 1 − that for O."""

    def values(self, state: State) -> tuple[float, ...] | None:
        x = 0.9 if state.model("cell")[2, 2] == "X" else 0.1
        return x, 1.0 - x


type Game = Callable[[str], RuleBasedSystem]


def test_the_value_prior_follows_each_action_s_outcome_for_the_player_to_act(game: Game) -> None:
    rbs = game("tictactoe")
    state, actions = opening(game)
    prior = ValuationPrior(CenterValued(), rbs, rbs.players(), 0.1)

    priors = dict(zip(actions, prior.priors(state, actions), strict=True))

    assert math.isclose(sum(priors.values()), 1.0)
    assert priors[CENTER] > 0.9
