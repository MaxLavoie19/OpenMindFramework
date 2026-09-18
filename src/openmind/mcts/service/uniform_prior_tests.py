from collections.abc import Callable

from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.mcts.service.uniform_prior import UniformPrior
from openmind.world.model.action import Action
from openmind.world.model.state import State


type Game = Callable[[str], RuleBasedSystem]


def opening(game: Game) -> tuple[State, tuple[Action, ...]]:
    rbs = game("tictactoe")
    return rbs.start(), rbs.actions(rbs.start())


def test_the_uniform_prior_gives_every_action_the_same_share(game: Game) -> None:
    state, actions = opening(game)

    assert UniformPrior().priors(state, actions) == tuple(1 / 9 for _ in range(9))
