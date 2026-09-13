import random

import pytest

from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain
from openmind.agent.service.random_policy import RandomPolicy
from openmind.csp.service.solver import Solver
from openmind.expression.mapper.expression_text_mapper import ExpressionTextMapper
from openmind.expression.service.interpreter import Interpreter
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.state import State

pytestmark = pytest.mark.log_level("INFO")


def new_solver() -> Solver:
    names = VariableNameMapper()
    return Solver(Interpreter(names), ExpressionTextMapper(names), ActionTextMapper())


def test_chooses_a_legal_action() -> None:
    domain = create_tictactoe_domain()

    action = RandomPolicy(new_solver(), random.Random(1)).choose(domain, domain.initial_state)

    assert action in new_solver().solve(domain.problem, domain.initial_state)


def test_the_same_seed_gives_the_same_choices() -> None:
    domain = create_tictactoe_domain()
    first, second = RandomPolicy(new_solver(), random.Random(7)), RandomPolicy(new_solver(), random.Random(7))

    assert [first.choose(domain, domain.initial_state) for _ in range(5)] == [
        second.choose(domain, domain.initial_state) for _ in range(5)
    ]


def test_no_legal_action_raises() -> None:
    domain = create_tictactoe_domain()
    finished = State(tuple((name, 1.0 if name.startswith("payoff") else value) for name, value in domain.initial_state.variables))

    with pytest.raises(ValueError, match="No legal action"):
        RandomPolicy(new_solver(), random.Random(1)).choose(domain, finished)
