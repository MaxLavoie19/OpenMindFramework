import math

from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain
from openmind.mcts.service.rater_prior_tests import CENTER
from openmind.mcts.service.uniform_prior_tests import opening
from openmind.mcts.service.valuation_prior import ValuationPrior
from openmind.predictor.factory.predictor_factory import create_predictor
from openmind.world.model.state import State


class CenterValued:
    """Values a position 0.9 for X when X holds the center, 0.1 otherwise, and 1 − that for O."""

    def value(self, state: State) -> tuple[float, ...] | None:
        x = 0.9 if dict(state.variables)["cell(2,2)"] == "X" else 0.1
        return x, 1.0 - x


def test_the_value_prior_follows_each_action_s_outcome_for_the_player_to_act() -> None:
    domain = create_tictactoe_domain()
    state, actions = opening()
    prior = ValuationPrior(CenterValued(), create_predictor(), domain.transitions, domain.players, 0.1)

    priors = dict(zip(actions, prior.priors(state, actions), strict=True))

    assert math.isclose(sum(priors.values()), 1.0)
    assert priors[CENTER] > 0.9
