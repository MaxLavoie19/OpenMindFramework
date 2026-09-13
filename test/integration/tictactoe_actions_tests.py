from openmind.agent.factory.tictactoe_factory import (
    create_tictactoe_initial_state,
    create_tictactoe_problem,
)
from openmind.csp.service.solver import Solver
from openmind.expression.service.interpreter import Interpreter
from openmind.world.builder.state_builder import StateBuilder
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.state import State
from openmind.world.model.value import Value


def legal_cells(state: State) -> list[tuple[Value, Value]]:
    actions = Solver(Interpreter(VariableNameMapper())).solve(create_tictactoe_problem(), state)
    assert {action.name for action in actions} <= {"place"}
    return [(dict(action.parameters)["row"], dict(action.parameters)["col"]) for action in actions]


def initial_state_with(changes: dict[str, Value]) -> State:
    builder = StateBuilder()
    for name, value in (dict(create_tictactoe_initial_state().variables) | changes).items():
        builder.with_variable(name, value)
    return builder.build()


def test_initial_state_allows_every_cell() -> None:
    assert legal_cells(create_tictactoe_initial_state()) == [
        (row, col) for row in (1, 2, 3) for col in (1, 2, 3)
    ]


def test_partly_filled_board_allows_exactly_its_empty_cells() -> None:
    state = initial_state_with({"cell(1,1)": "X", "cell(2,2)": "O", "cell(3,1)": "X", "turn": "O"})

    assert legal_cells(state) == [(1, 2), (1, 3), (2, 1), (2, 3), (3, 2), (3, 3)]


def test_full_board_allows_nothing() -> None:
    cells = [f"cell({row},{col})" for row in (1, 2, 3) for col in (1, 2, 3)]
    marks = ["X", "O", "X", "X", "O", "O", "O", "X", "X"]

    assert legal_cells(initial_state_with(dict(zip(cells, marks)))) == []


def test_finished_game_allows_nothing() -> None:
    state = initial_state_with(
        {
            "cell(1,1)": "X",
            "cell(1,2)": "X",
            "cell(1,3)": "X",
            "cell(2,1)": "O",
            "cell(2,2)": "O",
            "turn": "O",
            "payoff(X)": 1.0,
            "payoff(O)": 0.0,
        }
    )

    assert legal_cells(state) == []
