from openmind.agent.constant.tictactoe_constant import VARIANTS
from openmind.agent.factory.tictactoe_factory import (
    create_tictactoe_initial_state,
    create_tictactoe_problem,
    create_tictactoe_transitions,
)
from openmind.csp.factory.csp_factory import create_solver
from openmind.predictor.factory.predictor_factory import create_predictor
from openmind.world.builder.state_builder import StateBuilder
from openmind.world.model.action import Action
from openmind.world.model.state import State
from openmind.world.model.value import Value

FOURINAROW = VARIANTS["fourinarow"]
# A full board without four in a row, row 1 at the top
DRAWN_BOARD = ("OXXXOOX", "XXOOXXO", "OOOXOOO", "OXOXOXX", "XOXXXOX", "OXOOXOX")


def drop(state: State, *columns: int) -> State:
    predictor = create_predictor()
    transitions = create_tictactoe_transitions(FOURINAROW)
    for col in columns:
        ((state, probability),) = predictor.predict(transitions, state, Action("drop", (("col", col),))).outcomes
        assert probability == 1.0
    return state


def play(*columns: int) -> State:
    return drop(create_tictactoe_initial_state(FOURINAROW), *columns)


def values_of(state: State, *names: str) -> tuple[Value, ...]:
    variables = dict(state.variables)
    return tuple(variables[name] for name in names)


def legal_columns(state: State) -> list[Value]:
    return [dict(action.parameters)["col"] for action in create_solver().solve(create_tictactoe_problem(FOURINAROW), state)]


def test_a_mark_falls_to_the_lowest_empty_cell_of_its_column() -> None:
    state = play(4, 4)

    assert values_of(state, "cell(6,4)", "cell(5,4)", "cell(4,4)", "turn") == ("X", "O", None, "X")


def test_four_across_wins() -> None:
    state = play(1, 1, 2, 2, 3, 3, 4)

    assert values_of(state, "payoff(X)", "payoff(O)") == (1.0, 0.0)
    assert legal_columns(state) == []


def test_four_down_a_column_wins_for_o() -> None:
    state = play(1, 2, 1, 2, 1, 2, 3, 2)

    assert values_of(state, "payoff(X)", "payoff(O)") == (0.0, 1.0)


def test_four_on_a_rising_diagonal_wins() -> None:
    # X ends on (6,1) (5,2) (4,3) (3,4)
    state = play(1, 2, 2, 3, 4, 3, 3, 4, 5, 4, 4)

    assert values_of(state, "cell(3,4)", "payoff(X)", "payoff(O)") == ("X", 1.0, 0.0)


def test_four_on_a_falling_diagonal_wins() -> None:
    # X ends on (3,4) (4,5) (5,6) (6,7)
    state = play(7, 6, 6, 5, 4, 5, 5, 4, 3, 4, 4)

    assert values_of(state, "cell(3,4)", "payoff(X)", "payoff(O)") == ("X", 1.0, 0.0)


def test_a_full_column_is_no_longer_legal() -> None:
    state = play(1, 1, 1, 1, 1, 1)

    assert values_of(state, "payoff(X)", "payoff(O)") == (None, None)
    assert legal_columns(state) == [2, 3, 4, 5, 6, 7]


def test_filling_the_board_without_a_line_is_a_draw() -> None:
    builder = StateBuilder().with_variable("turn", DRAWN_BOARD[0][0])
    builder.with_variable("payoff(X)", None).with_variable("payoff(O)", None)
    for row, marks in enumerate(DRAWN_BOARD, start=1):
        for col, mark in enumerate(marks, start=1):
            builder.with_variable(f"cell({row},{col})", None if (row, col) == (1, 1) else mark)

    state = drop(builder.build(), 1)

    assert values_of(state, "cell(1,1)", "payoff(X)", "payoff(O)") == ("O", 0.5, 0.5)
    assert legal_columns(state) == []
