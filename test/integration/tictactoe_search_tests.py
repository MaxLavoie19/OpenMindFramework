from collections.abc import Callable

import pytest

from openmind.agent.factory.agent_factory import create_agent
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.structure.model.coordinates import Coordinates
from openmind.world.model.action import Action
from openmind.world.model.state import State

pytestmark = pytest.mark.log_level("INFO")

type Game = Callable[[str], RuleBasedSystem]


def position(rbs: RuleBasedSystem, marks: dict[Coordinates, str]) -> State:
    cell = rbs.start().model("cell")
    for where, mark in marks.items():
        cell = cell.placed(where, mark)
    return rbs.start().with_model("cell", cell)


def test_x_takes_an_immediate_win(game: Game) -> None:
    rbs = game("tictactoe")
    # X X . / O O . / . . .
    state = position(rbs, {(1, 1): "X", (1, 2): "X", (2, 1): "O", (2, 2): "O"})

    action = create_agent(iterations=500, seed=1).choose(rbs, state)

    assert action == Action("place", (("col", 3), ("row", 1)))


def test_x_blocks_an_immediate_win_of_o(game: Game) -> None:
    rbs = game("tictactoe")
    # X . . / O O . / . X .
    state = position(rbs, {(1, 1): "X", (3, 2): "X", (2, 1): "O", (2, 2): "O"})

    action = create_agent(iterations=500, seed=1).choose(rbs, state)

    assert action == Action("place", (("col", 3), ("row", 2)))
