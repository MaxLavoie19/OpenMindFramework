from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain
from openmind.csp.factory.csp_factory import create_solver
from openmind.mcts.service.uniform_prior import UniformPrior
from openmind.world.model.action import Action
from openmind.world.model.state import State


def opening() -> tuple[State, tuple[Action, ...]]:
    domain = create_tictactoe_domain()
    return domain.initial_state, create_solver().solve(domain.problem, domain.initial_state)


def test_the_uniform_prior_gives_every_action_the_same_share() -> None:
    state, actions = opening()

    assert UniformPrior().priors(state, actions) == tuple(1 / 9 for _ in range(9))
