import pickle
from collections.abc import Callable

import pytest

from openmind.inference.service.mechanics import Mechanics
from openmind.parallel.service.memory_meter import MemoryMeter
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.structure.model.grid import Grid
from openmind.world.model.state import State

type Game = Callable[[str], RuleBasedGame]


def mechanics() -> Mechanics:
    return Mechanics(StateNamespaceMapper(), MemoryMeter())


def a_position() -> State:
    return State.of(cell=Grid.of([[None, None], [None, None]]), turn="X")


def test_the_same_position_gives_the_same_view_rather_than_a_new_one(game: Game) -> None:
    """A view is what an expression reads a position through, and generated expressions read thousands of them
    over one position. Built anew each time, the reading is the building."""
    held, played = mechanics(), game("tictactoe")
    start = played.start()

    assert held.view(played, start) is held.view(played, start)


def test_clearing_forgets_what_the_process_kept(game: Game) -> None:
    held, played = mechanics(), game("tictactoe")
    start = played.start()
    first = held.view(played, start)

    held.clear()

    assert held.memory_entries() == 0
    assert held.view(played, start) is not first


def test_a_state_s_models_are_read_the_way_a_rule_reads_them() -> None:
    """A scalar as its value, a grid as itself."""
    read = mechanics().variables(a_position())

    assert read["turn"] == "X"
    assert read["cell"][1, 1] is None


def test_every_action_a_player_could_take_here_has_its_outcomes(game: Game) -> None:
    played = game("tictactoe")
    start = played.start()

    moves = mechanics().moves(played, start, "X")

    assert len(moves) == len(played.actions(start, player="X"))
    assert all(len(outcomes) == 1 for outcomes in moves), "nothing in tictactoe is chancy"
    assert all(probability == 1.0 for outcomes in moves for _, probability in outcomes)


def test_a_player_the_game_gives_no_action_has_no_moves(game: Game) -> None:
    """Asked as if it were their turn, and where the game says they can do nothing, nothing comes back."""
    played = game("tictactoe")

    assert mechanics().moves(played, played.start(), "nobody") == ()


def test_how_many_of_a_player_s_actions_change_each_part(game: Game) -> None:
    """Every cell of an empty board is changed by exactly the one move that takes it, and whose turn it is by
    all of them."""
    played = game("tictactoe")
    start = played.start()

    changes = mechanics().changes(played, start, "X")

    assert changes[("cell", (1, 1))] == pytest.approx(1.0)
    assert changes[("turn", None)] == pytest.approx(float(len(played.actions(start, player="X"))))


def test_a_part_no_action_changes_is_left_out_rather_than_counted_at_nothing(game: Game) -> None:
    played = game("tictactoe")

    changes = mechanics().changes(played, played.start(), "X")

    assert all(count > 0 for count in changes.values())


def test_a_cell_can_be_set_to_something_the_game_would_never_have_put_there() -> None:
    """How a rule is put to the test: build the position that would break it and ask. Whether the game could
    have reached it is not the question."""
    edited = mechanics().with_value(a_position(), "cell", (2, 2), "O")

    assert edited.model("cell").at((2, 2)) == "O"
    assert edited.model("cell").at((1, 1)) is None


def test_setting_a_cell_the_position_has_not_says_which_one() -> None:
    with pytest.raises(KeyError, match=r"no cell cell\[\(9, 9\)\]"):
        mechanics().with_value(a_position(), "cell", (9, 9), "O")


def test_setting_a_model_the_position_has_not_says_so() -> None:
    with pytest.raises(KeyError, match="no cell lamp"):
        mechanics().with_value(a_position(), "lamp", (1, 1), "on")


def test_copying_a_cell_carries_every_grid_that_has_both() -> None:
    """What a game keeps across two grids moves together, which is what a move does to a piece and its colour."""
    before = State.of(
        piece=Grid.of([["rook", None], [None, None]]), colour=Grid.of([["white", None], [None, None]])
    )

    copied = mechanics().copied(before, (1, 1), (2, 2))

    assert copied.model("piece").at((2, 2)) == "rook"
    assert copied.model("colour").at((2, 2)) == "white"


def test_copying_leaves_alone_a_grid_that_has_not_got_both_cells() -> None:
    before = State.of(wide=Grid.of([["a", "b"], ["c", "d"]]), narrow=Grid.of([["e"]]))

    copied = mechanics().copied(before, (1, 1), (2, 2))

    assert copied.model("wide").at((2, 2)) == "a"
    assert copied.model("narrow").at((1, 1)) == "e"


def test_what_a_process_kept_is_left_behind_when_it_is_sent_to_another(game: Game) -> None:
    """Views and moves are one process's; a copy arriving somewhere else starts with nothing and still works."""
    held, played = mechanics(), game("tictactoe")
    held.view(played, played.start())

    sent = pickle.loads(pickle.dumps(held))

    assert held.memory_entries() == 1
    assert sent.memory_entries() == 0
    assert sent.variables(a_position())["turn"] == "X"
