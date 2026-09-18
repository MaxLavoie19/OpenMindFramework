from collections.abc import Callable

import pytest

from openmind.agent.factory.agent_factory import create_agent
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.world.builder.state_builder import StateBuilder
from openmind.world.model.action import Action
from openmind.world.model.state import State
from openmind.world.model.value import Value

pytestmark = pytest.mark.log_level("INFO")

type Game = Callable[[str], RuleBasedSystem]


def position(rbs: RuleBasedSystem, cells: dict[str, Value]) -> State:
    builder = StateBuilder()
    for name, value in (dict(rbs.start().variables) | cells).items():
        builder.with_variable(name, value)
    return builder.build()


def test_x_takes_an_immediate_win(game: Game) -> None:
    rbs = game("tictactoe")
    # X X . / O O . / . . .
    state = position(rbs, {"cell(1,1)": "X", "cell(1,2)": "X", "cell(2,1)": "O", "cell(2,2)": "O"})

    action = create_agent(iterations=500, seed=1).choose(rbs, state)

    assert action == Action("place", (("col", 3), ("row", 1)))


def test_x_blocks_an_immediate_win_of_o(game: Game) -> None:
    rbs = game("tictactoe")
    # X . . / O O . / . X .
    state = position(rbs, {"cell(1,1)": "X", "cell(3,2)": "X", "cell(2,1)": "O", "cell(2,2)": "O"})

    action = create_agent(iterations=500, seed=1).choose(rbs, state)

    assert action == Action("place", (("col", 3), ("row", 2)))
