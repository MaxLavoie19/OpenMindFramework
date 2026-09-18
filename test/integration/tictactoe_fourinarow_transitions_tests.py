from collections.abc import Callable

from openmind.agent.constant.tictactoe_constant import VARIANTS
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.world.builder.state_builder import StateBuilder
from openmind.world.model.action import Action
from openmind.world.model.state import State
from openmind.world.model.value import Value

type Game = Callable[[str], RuleBasedSystem]

FOURINAROW = "tictactoe/" + VARIANTS["fourinarow"].name
# A full board without four in a row, row 1 at the top
DRAWN_BOARD = ("OXXXOOX", "XXOOXXO", "OOOXOOO", "OXOXOXX", "XOXXXOX", "OXOOXOX")


def drop(rbs: RuleBasedSystem, state: State, *columns: int) -> State:
    for col in columns:
        ((state, probability),) = rbs.outcomes(state, Action("drop", (("col", col),))).outcomes
        assert probability == 1.0
    return state


def play(game: Game, *columns: int) -> State:
    rbs = game(FOURINAROW)
    return drop(rbs, rbs.start(), *columns)


def values_of(state: State, *names: str) -> tuple[Value, ...]:
    variables = dict(state.variables)
    return tuple(variables[name] for name in names)


def legal_columns(game: Game, state: State) -> list[Value]:
    return [dict(action.parameters)["col"] for action in game(FOURINAROW).actions(state)]


def test_a_mark_falls_to_the_lowest_empty_cell_of_its_column(game: Game) -> None:
    state = play(game, 4, 4)

    assert values_of(state, "cell(6,4)", "cell(5,4)", "cell(4,4)", "turn") == ("X", "O", None, "X")


def test_four_across_wins(game: Game) -> None:
    state = play(game, 1, 1, 2, 2, 3, 3, 4)

    assert values_of(state, "payoff(X)", "payoff(O)") == (1.0, 0.0)
    assert legal_columns(game, state) == []


def test_four_down_a_column_wins_for_o(game: Game) -> None:
    state = play(game, 1, 2, 1, 2, 1, 2, 3, 2)

    assert values_of(state, "payoff(X)", "payoff(O)") == (0.0, 1.0)


def test_four_on_a_rising_diagonal_wins(game: Game) -> None:
    # X ends on (6,1) (5,2) (4,3) (3,4)
    state = play(game, 1, 2, 2, 3, 4, 3, 3, 4, 5, 4, 4)

    assert values_of(state, "cell(3,4)", "payoff(X)", "payoff(O)") == ("X", 1.0, 0.0)


def test_four_on_a_falling_diagonal_wins(game: Game) -> None:
    # X ends on (3,4) (4,5) (5,6) (6,7)
    state = play(game, 7, 6, 6, 5, 4, 5, 5, 4, 3, 4, 4)

    assert values_of(state, "cell(3,4)", "payoff(X)", "payoff(O)") == ("X", 1.0, 0.0)


def test_a_full_column_is_no_longer_legal(game: Game) -> None:
    state = play(game, 1, 1, 1, 1, 1, 1)

    assert values_of(state, "payoff(X)", "payoff(O)") == (None, None)
    assert legal_columns(game, state) == [2, 3, 4, 5, 6, 7]


def test_filling_the_board_without_a_line_is_a_draw(game: Game) -> None:
    builder = StateBuilder().with_variable("turn", DRAWN_BOARD[0][0])
    builder.with_variable("payoff(X)", None).with_variable("payoff(O)", None)
    for row, marks in enumerate(DRAWN_BOARD, start=1):
        for col, mark in enumerate(marks, start=1):
            builder.with_variable(f"cell({row},{col})", None if (row, col) == (1, 1) else mark)

    state = drop(game(FOURINAROW), builder.build(), 1)

    assert values_of(state, "cell(1,1)", "payoff(X)", "payoff(O)") == ("O", 0.5, 0.5)
    assert legal_columns(game, state) == []
