import pytest

from openmind.agent.factory.agent_factory import create_agent
from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain
from openmind.world.builder.state_builder import StateBuilder
from openmind.world.model.action import Action
from openmind.world.model.state import State
from openmind.world.model.value import Value

pytestmark = pytest.mark.log_level("INFO")


def position(cells: dict[str, Value]) -> State:
    builder = StateBuilder()
    for name, value in (dict(create_tictactoe_domain().initial_state.variables) | cells).items():
        builder.with_variable(name, value)
    return builder.build()


def test_x_takes_an_immediate_win() -> None:
    # X X . / O O . / . . .
    state = position({"cell(1,1)": "X", "cell(1,2)": "X", "cell(2,1)": "O", "cell(2,2)": "O"})

    action = create_agent(iterations=500, seed=1).choose(create_tictactoe_domain(), state)

    assert action == Action("place", (("col", 3), ("row", 1)))


def test_x_blocks_an_immediate_win_of_o() -> None:
    # X . . / O O . / . X .
    state = position({"cell(1,1)": "X", "cell(3,2)": "X", "cell(2,1)": "O", "cell(2,2)": "O"})

    action = create_agent(iterations=500, seed=1).choose(create_tictactoe_domain(), state)

    assert action == Action("place", (("col", 3), ("row", 2)))
