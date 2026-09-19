from collections.abc import Callable

from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.structure.model.map import Map
from openmind.world.model.action import Action
from openmind.world.model.state import State


type Game = Callable[[str], RuleBasedSystem]


def play(game: Game, *cells: tuple[int, int]) -> State:
    rbs = game("tictactoe")
    state = rbs.start()
    for row, col in cells:
        ((state, probability),) = rbs.outcomes(state, Action("place", (("col", col), ("row", row)))).outcomes
        assert probability == 1.0
    return state


def legal_action_count(game: Game, state: State) -> int:
    return len(game("tictactoe").actions(state))


def test_turns_alternate(game: Game) -> None:
    first, second = play(game, (2, 2)), play(game, (2, 2), (1, 1))

    assert (first.model("cell")[2, 2], first.value("turn")) == ("X", "O")
    assert (second.model("cell")[2, 2], second.model("cell")[1, 1], second.value("turn")) == ("X", "O", "X")


def test_winning_move_sets_payoffs_and_ends_the_game(game: Game) -> None:
    # X takes row 1 while O takes (2,1) and (2,2)
    state = play(game, (1, 1), (2, 1), (1, 2), (2, 2), (1, 3))

    assert state.model("payoff") == Map.of({"X": 1.0, "O": 0.0})
    assert legal_action_count(game, state) == 0


def test_full_board_without_a_line_is_a_draw(game: Game) -> None:
    # X O X / X O O / O X X
    state = play(game, (1, 1), (1, 2), (1, 3), (2, 2), (2, 1), (2, 3), (3, 2), (3, 1), (3, 3))

    assert state.model("payoff") == Map.of({"X": 0.5, "O": 0.5})
    assert legal_action_count(game, state) == 0


def test_line_completed_with_the_last_empty_cell_is_a_win(game: Game) -> None:
    # X X O / O X X / O O X, X completes the diagonal with (3,3)
    state = play(game, (1, 1), (1, 3), (1, 2), (2, 1), (2, 2), (3, 1), (2, 3), (3, 2), (3, 3))

    assert state.model("payoff") == Map.of({"X": 1.0, "O": 0.0})
