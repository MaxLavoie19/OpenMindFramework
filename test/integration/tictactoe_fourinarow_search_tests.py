from collections.abc import Callable
import pytest

from openmind.agent.constant.tictactoe_constant import VARIANTS
from openmind.agent.factory.agent_factory import create_agent
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.world.builder.state_builder import StateBuilder
from openmind.world.model.action import Action
from openmind.world.model.state import State
from openmind.world.model.value import Value

pytestmark = pytest.mark.log_level("INFO")

type Game = Callable[[str], RuleBasedSystem]

FOURINAROW = "tictactoe/" + VARIANTS["fourinarow"].name


def position(rbs: RuleBasedSystem, cells: dict[str, Value]) -> State:
    builder = StateBuilder()
    for name, value in (dict(rbs.start().variables) | cells).items():
        builder.with_variable(name, value)
    return builder.build()


def test_x_takes_an_immediate_win(game: Game) -> None:
    rbs = game(FOURINAROW)
    # bottom row X X X . . . O, with O on (5,1) and (5,2)
    state = position(
        rbs,
        {"cell(6,1)": "X", "cell(6,2)": "X", "cell(6,3)": "X", "cell(6,7)": "O", "cell(5,1)": "O", "cell(5,2)": "O"}
    )

    action = create_agent(iterations=500, seed=1).choose(rbs, state)

    assert action == Action("drop", (("col", 4),))


def test_o_blocks_an_immediate_win_of_x(game: Game) -> None:
    rbs = game(FOURINAROW)
    # bottom row X X X . . . ., with O on (5,1) and (5,2), O to play
    state = position(
        rbs,
        {"cell(6,1)": "X", "cell(6,2)": "X", "cell(6,3)": "X", "cell(5,1)": "O", "cell(5,2)": "O", "turn": "O"}
    )

    action = create_agent(iterations=500, seed=1).choose(rbs, state)

    assert action == Action("drop", (("col", 4),))
