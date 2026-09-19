from collections.abc import Callable

from openmind.agent.factory.tictactoe_factory import create_tictactoe_initial_state
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.structure.model.coordinates import Coordinates
from openmind.structure.model.map import Map
from openmind.structure.model.value import Value
from openmind.world.model.state import State


type Game = Callable[[str], RuleBasedSystem]


def legal_cells(game: Game, state: State) -> list[tuple[Value, Value]]:
    actions = game("tictactoe").actions(state)
    assert {action.name for action in actions} <= {"place"}
    return [(dict(action.parameters)["row"], dict(action.parameters)["col"]) for action in actions]


def initial_state_with(marks: dict[Coordinates, str], turn: str = "X") -> State:
    state = create_tictactoe_initial_state()
    cell = state.model("cell")
    for where, mark in marks.items():
        cell = cell.placed(where, mark)
    return state.with_model("cell", cell).with_model("turn", turn)


def test_initial_state_allows_every_cell(game: Game) -> None:
    assert legal_cells(game, create_tictactoe_initial_state()) == [
        (row, col) for row in (1, 2, 3) for col in (1, 2, 3)
    ]


def test_partly_filled_board_allows_exactly_its_empty_cells(game: Game) -> None:
    state = initial_state_with({(1, 1): "X", (2, 2): "O", (3, 1): "X"}, turn="O")

    assert legal_cells(game, state) == [(1, 2), (1, 3), (2, 1), (2, 3), (3, 2), (3, 3)]


def test_full_board_allows_nothing(game: Game) -> None:
    cells = [(row, col) for row in (1, 2, 3) for col in (1, 2, 3)]
    marks = ["X", "O", "X", "X", "O", "O", "O", "X", "X"]

    assert legal_cells(game, initial_state_with(dict(zip(cells, marks)))) == []


def test_finished_game_allows_nothing(game: Game) -> None:
    state = initial_state_with({(1, 1): "X", (1, 2): "X", (1, 3): "X", (2, 1): "O", (2, 2): "O"}, turn="O")
    state = state.with_model("payoff", Map.of({"X": 1.0, "O": 0.0}))

    assert legal_cells(game, state) == []
