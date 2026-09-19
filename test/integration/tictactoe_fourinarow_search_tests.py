from collections.abc import Callable
import pytest

from openmind.agent.constant.tictactoe_constant import VARIANTS
from openmind.agent.factory.agent_factory import create_agent
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.structure.model.coordinates import Coordinates
from openmind.world.model.action import Action
from openmind.world.model.state import State

pytestmark = pytest.mark.log_level("INFO")

type Game = Callable[[str], RuleBasedSystem]

FOURINAROW = "tictactoe/" + VARIANTS["fourinarow"].name


def position(rbs: RuleBasedSystem, marks: dict[Coordinates, str], turn: str = "X") -> State:
    cell = rbs.start().model("cell")
    for where, mark in marks.items():
        cell = cell.placed(where, mark)
    return rbs.start().with_model("cell", cell).with_model("turn", turn)


def test_x_takes_an_immediate_win(game: Game) -> None:
    rbs = game(FOURINAROW)
    # bottom row X X X . . . O, with O on (5,1) and (5,2)
    state = position(
        rbs,
        {(6, 1): "X", (6, 2): "X", (6, 3): "X", (6, 7): "O", (5, 1): "O", (5, 2): "O"},
    )

    action = create_agent(iterations=500, seed=1).choose(rbs, state)

    assert action == Action("drop", (("col", 4),))


def test_o_blocks_an_immediate_win_of_x(game: Game) -> None:
    rbs = game(FOURINAROW)
    # bottom row X X X . . . ., with O on (5,1) and (5,2), O to play
    state = position(
        rbs,
        {(6, 1): "X", (6, 2): "X", (6, 3): "X", (5, 1): "O", (5, 2): "O"},
        turn="O",
    )

    action = create_agent(iterations=500, seed=1).choose(rbs, state)

    assert action == Action("drop", (("col", 4),))
