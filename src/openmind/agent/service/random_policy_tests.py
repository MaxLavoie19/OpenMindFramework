import random

import pytest

from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain
from openmind.agent.service.random_policy import RandomPolicy
from openmind.csp.factory.csp_factory import create_solver
from openmind.world.model.state import State

pytestmark = pytest.mark.log_level("INFO")


def test_chooses_a_legal_action() -> None:
    domain = create_tictactoe_domain()

    action = RandomPolicy(create_solver(), random.Random(1)).choose(domain, domain.initial_state)

    assert action in create_solver().solve(domain.problem, domain.initial_state)


def test_the_same_seed_gives_the_same_choices() -> None:
    domain = create_tictactoe_domain()
    first, second = RandomPolicy(create_solver(), random.Random(7)), RandomPolicy(create_solver(), random.Random(7))

    assert [first.choose(domain, domain.initial_state) for _ in range(5)] == [
        second.choose(domain, domain.initial_state) for _ in range(5)
    ]


def test_no_legal_action_raises() -> None:
    domain = create_tictactoe_domain()
    finished = State(tuple((name, 1.0 if name.startswith("payoff") else value) for name, value in domain.initial_state.variables))

    with pytest.raises(ValueError, match="No legal action"):
        RandomPolicy(create_solver(), random.Random(1)).choose(domain, finished)
